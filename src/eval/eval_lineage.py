"""
eval_lineage.py - Fase 8
Valuta l'attribuzione dei DATI DI ADDESTRAMENTO: dato un modello addestrato con
pairing per lineage, verifica se i generatori MAI VISTI in training vengono
comunque ricondotti al dataset su cui sono stati addestrati.

Produce i tre blocchi di evidenza previsti per l'articolo:

  1. GEOMETRIA. Matrice delle distanze medie fra classi (diagonale = compattezza
     intra-classe) + dispersione intra-classe. E' la versione quantitativa del
     "prendo il vettore medio di CelebA, quello di StarGAN, calcolo la distanza".

  2. DECISIONE CON SOGLIA CALIBRATA. Per ogni immagine si calcola la distanza dal
     centroide di ciascuna lineage; lo score e' la differenza fra le due distanze.
     La soglia e' fissata al punto di EER sulle sole classi VISTE in training, poi
     applicata invariata ai generatori mai visti. Si riportano AUC, EER, soglia,
     accuratezza e FAR/FRR.
     Questo e' il punto che rende il risultato utilizzabile come evidenza: una
     soglia senza un tasso d'errore noto, misurato su dati mai visti, non dice
     nulla. (E' anche cio' che un vaglio di ammissibilita' tecnica richiede.)

  3. GENERALIZZAZIONE PER GENERATORE. Per ogni architettura held-out: quota di
     immagini attribuite alla lineage corretta, distanza media dai due centroidi,
     e 1-NN. Separati esplicitamente fra editing-GAN (evidenza debole: l'output
     contiene i pixel dell'input reale) e noise-GAN (evidenza forte: generano da
     rumore, l'unica origine possibile del segnale sono i dati di addestramento).

Uso:
  python -m src.eval.eval_lineage --checkpoint runs/lineage_clip_l14/best.pt \
      --manifest data/manifest_lineage.csv --out report/lineage_clip_l14
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

from src.data.siamese_dataset import SingleImageDataset, labels_of
from src.models.siamese import load_encoder

# generatori che partono da un'immagine reale: il loro output CONTIENE gia' i
# pixel del dataset sorgente, quindi l'attribuzione e' evidenza debole
EDITING_GANS = {"stargan", "attgan", "gdwct"}


@torch.no_grad()
def embed(checkpoint, manifest, split, size, device, bs, nw, patch=None,
          ppi=1, ppolicy="flat"):
    model, spec, epoch = load_encoder(checkpoint, device)
    ds = SingleImageDataset(manifest, split, spec, lineage_filter=None,
                            image_size=(patch or size), patch_size=patch,
                            patches_per_image=ppi, patch_policy=ppolicy)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw,
                        pin_memory=(device == "cuda"))
    Z = [model.forward_one(x.to(device)).cpu().numpy() for x, _ in loader]
    Z = np.concatenate(Z).astype(np.float32)
    return Z, ds.df, epoch


def pdist(a, b):
    aa = (a * a).sum(1)[:, None]
    bb = (b * b).sum(1)[None, :]
    return np.sqrt(np.maximum(aa + bb - 2.0 * a @ b.T, 0.0))


def class_distance_matrix(Z, lab, classes):
    M = np.zeros((len(classes), len(classes)))
    for i, ca in enumerate(classes):
        a = Z[lab == ca]
        for j, cb in enumerate(classes):
            b = Z[lab == cb]
            if len(a) == 0 or len(b) == 0:
                M[i, j] = np.nan
                continue
            d = pdist(a, b)
            if ca == cb:
                n = len(a)
                M[i, j] = (d.sum() - np.trace(d)) / max(n * (n - 1), 1)
            else:
                M[i, j] = d.mean()
    return M


def eer_threshold(scores, y):
    """Soglia al punto di Equal Error Rate. score alto = classe positiva (y=1)."""
    fpr, tpr, thr = roc_curve(y, scores)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float(thr[i]), float((fpr[i] + fnr[i]) / 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default="data/manifest_lineage.csv")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--n-per-class", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trained-on", nargs="*", default=None,
                    help="architetture viste in training (default: dedotte dal manifest)")
    ap.add_argument("--patch-size", type=int, default=None,
                    help="se impostato, valuta su patch native invece che immagini intere")
    ap.add_argument("--patches-per-image", type=int, default=1)
    ap.add_argument("--patch-policy", default="flat")
    ap.add_argument("--out", default="report/lineage")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # quali architetture ha visto il modello: quelle presenti nel train split
    import pandas as pd
    full = pd.read_csv(args.manifest, dtype={"source_id": str}, keep_default_na=False)
    seen = (set(args.trained_on) if args.trained_on
            else set(full[full.split == "train"].architecture.unique()))
    print(f"Architetture VISTE in training : {sorted(seen)}")

    Z, df, epoch = embed(args.checkpoint, args.manifest, "test", args.image_size,
                         device, args.batch_size, args.num_workers,
                         patch=args.patch_size, ppi=args.patches_per_image,
                         ppolicy=args.patch_policy)
    arch = labels_of(df, "architecture")
    lin = labels_of(df, "lineage")
    cls = labels_of(df, "class")
    print(f"Checkpoint epoca {epoch} | device {device} | test: {len(Z)} immagini")

    # bilancia per CLASSE, cosi' le held-out (tutte nel test) non drogano le stime
    keep = []
    for c in np.unique(cls):
        ci = np.where(cls == c)[0]
        keep.append(rng.choice(ci, args.n_per_class, replace=False)
                    if len(ci) > args.n_per_class else ci)
    keep = np.concatenate(keep)
    Z, arch, lin, cls = Z[keep], arch[keep], lin[keep], cls[keep]
    print("  per classe:", {c: int((cls == c).sum()) for c in sorted(np.unique(cls))})
    np.savez(out / "embeddings.npz", Z=Z, arch=arch, lineage=lin, cls=cls)

    lineages = sorted(np.unique(lin).tolist())
    if len(lineages) != 2:
        raise SystemExit(f"attese 2 lineage, trovate {lineages}")
    LA, LB = lineages

    # ---------- 1. geometria ----------
    classes = sorted(np.unique(cls).tolist())
    M = class_distance_matrix(Z, cls, classes)
    print("\n== 1. Distanza media fra classi (diag = compattezza intra-classe) ==")
    print("              " + "".join(f"{c[:10]:>12s}" for c in classes))
    for i, c in enumerate(classes):
        star = "*" if c.replace("real_", "") not in seen and not c.startswith("real") else " "
        print(f"{star}{c[:12]:>12s} " + "".join(f"{M[i, j]:12.3f}" for j in range(len(classes))))
    print("  (* = generatore MAI VISTO in training)")

    # ---------- 2. decisione con soglia calibrata ----------
    # centroidi costruiti SOLO sulle classi viste: un centroide che includesse le
    # held-out userebbe informazione che in un caso reale non si avrebbe
    seen_mask = np.array([(a == "real") or (a in seen) for a in arch])
    cA = Z[seen_mask & (lin == LA)].mean(0)
    cB = Z[seen_mask & (lin == LB)].mean(0)
    dA = np.linalg.norm(Z - cA, axis=1)
    dB = np.linalg.norm(Z - cB, axis=1)
    score = dA - dB                     # >0 -> piu' vicino a LB
    y = (lin == LB).astype(int)

    auc_seen = roc_auc_score(y[seen_mask], score[seen_mask])
    tau, eer = eer_threshold(score[seen_mask], y[seen_mask])
    pred = np.where(score > tau, LB, LA)

    unseen = ~seen_mask
    auc_unseen = (roc_auc_score(y[unseen], score[unseen])
                  if len(np.unique(y[unseen])) > 1 else None)
    acc_seen = float((pred[seen_mask] == lin[seen_mask]).mean())
    acc_unseen = float((pred[unseen] == lin[unseen]).mean())
    # FAR/FRR calcolati con LB come classe "positiva"
    far = float((pred[unseen & (lin == LA)] == LB).mean()) if (unseen & (lin == LA)).any() else None
    frr = float((pred[unseen & (lin == LB)] == LA).mean()) if (unseen & (lin == LB)).any() else None

    print(f"\n== 2. Decisione di lineage con soglia calibrata sulle sole classi viste ==")
    print(f"  soglia tau (EER su classi viste) : {tau:+.4f}   (EER {eer*100:.1f}%)")
    print(f"  AUC   classi viste               : {auc_seen:.4f}   acc {acc_seen*100:.1f}%")
    if auc_unseen is not None:
        print(f"  AUC   generatori MAI VISTI       : {auc_unseen:.4f}   acc {acc_unseen*100:.1f}%")
    if far is not None:
        print(f"  su mai visti -> FAR({LA}->{LB}) {far*100:.1f}%  FRR({LB}->{LA}) {frr*100:.1f}%")

    # ---------- 3. per generatore ----------
    print(f"\n== 3. Attribuzione per generatore ==")
    print(f"{'classe':>14s} {'lineage':>7s} {'visto':>6s} {'tipo':>8s} "
          f"{'d_celeba':>9s} {'d_ffhq':>8s} {'corretti':>9s}")
    per_class = {}
    for c in classes:
        m = cls == c
        a = arch[m][0]
        true_lin = lin[m][0]
        is_seen = (a == "real") or (a in seen)
        kind = ("reale" if a == "real"
                else "editing" if a in EDITING_GANS else "noise")
        rate = float((pred[m] == true_lin).mean())
        per_class[c] = dict(lineage=true_lin, seen=bool(is_seen), kind=kind,
                            mean_d_celeba=float(dA[m].mean()),
                            mean_d_ffhq=float(dB[m].mean()),
                            correct_rate=rate, n=int(m.sum()))
        print(f"{c:>14s} {true_lin:>7s} {'si' if is_seen else 'NO':>6s} {kind:>8s} "
              f"{dA[m].mean():9.3f} {dB[m].mean():8.3f} {rate*100:8.1f}%")

    # lettura sintetica separata per tipo di evidenza
    noise_unseen = [c for c, v in per_class.items()
                    if not v["seen"] and v["kind"] == "noise"]
    edit_unseen = [c for c, v in per_class.items()
                   if not v["seen"] and v["kind"] == "editing"]
    print("\n--- lettura ---")
    if edit_unseen:
        r = np.mean([per_class[c]["correct_rate"] for c in edit_unseen])
        print(f"editing-GAN mai visti {edit_unseen}: {r*100:.1f}% attribuiti correttamente")
        print("  -> evidenza DEBOLE: l'output contiene i pixel dell'immagine reale di input,")
        print("     quindi non distingue 'addestrato su' da 'ha ricevuto in input'.")
    if noise_unseen:
        r = np.mean([per_class[c]["correct_rate"] for c in noise_unseen])
        print(f"noise-GAN mai visti {noise_unseen}: {r*100:.1f}% attribuiti correttamente")
        print("  -> evidenza FORTE: generano da rumore, l'unica origine possibile del")
        print("     segnale sono i dati di addestramento.")
        if r >= 0.8:
            print("  VERDETTO: la traccia del dataset di addestramento TRASFERISCE a")
            print("            generatori mai visti.")
        elif r >= 0.6:
            print("  VERDETTO: trasferimento PARZIALE.")
        else:
            print("  VERDETTO: NESSUN trasferimento (vicino al caso).")

    save_figs(Z, cls, lin, per_class, score, tau, seen_mask, LA, LB, out)

    report = dict(checkpoint=args.checkpoint, epoch=epoch, manifest=args.manifest,
                  seen_architectures=sorted(seen), lineages=[LA, LB],
                  classes=classes, class_distance_matrix=M.tolist(),
                  threshold=dict(tau=tau, eer_seen=eer, auc_seen=float(auc_seen),
                                 auc_unseen=(None if auc_unseen is None else float(auc_unseen)),
                                 acc_seen=acc_seen, acc_unseen=acc_unseen,
                                 far_unseen=far, frr_unseen=frr),
                  per_class=per_class)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nReport -> {out/'report.json'}")


def save_figs(Z, cls, lin, per_class, score, tau, seen_mask, LA, LB, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except Exception:
        print("  (matplotlib assente: salto le figure)")
        return
    classes = sorted(per_class)
    cmap = plt.get_cmap("tab10")

    # --- PCA: cerchio pieno = visto, croce = mai visto ---
    p2 = PCA(n_components=2).fit_transform(Z)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for k, c in enumerate(classes):
        m = cls == c
        v = per_class[c]
        ax.scatter(p2[m, 0], p2[m, 1], s=10, alpha=0.55, color=cmap(k % 10),
                   marker="o" if v["seen"] else "X",
                   label=f"{c}{'' if v['seen'] else '  (mai visto)'}")
    ax.legend(markerscale=2, fontsize=8, loc="best")
    ax.set_title("Spazio di embedding per lineage (PCA 2D)\n"
                 "X = generatore mai visto in training")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.tight_layout(); fig.savefig(out / "pca_lineage.png", dpi=140)
    plt.close(fig)

    # --- distanze ai due centroidi, con la soglia ---
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(classes))
    dc = [per_class[c]["mean_d_celeba"] for c in classes]
    dfq = [per_class[c]["mean_d_ffhq"] for c in classes]
    ax.bar(xs - 0.2, dc, 0.4, label=f"distanza dal centroide {LA}")
    ax.bar(xs + 0.2, dfq, 0.4, label=f"distanza dal centroide {LB}")
    ax.set_xticks(xs)
    ax.set_xticklabels([c + ("" if per_class[c]["seen"] else "\n(mai visto)")
                        for c in classes], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("distanza media nell'embedding")
    ax.set_title("Distanza media dai centroidi di lineage")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out / "centroid_distances.png", dpi=140)
    plt.close(fig)

    # --- distribuzione dello score con la soglia calibrata ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for lab, m, sty in [(f"{LA} (visti)", seen_mask & (lin == LA), dict(alpha=.55)),
                        (f"{LB} (visti)", seen_mask & (lin == LB), dict(alpha=.55)),
                        (f"{LA} (mai visti)", (~seen_mask) & (lin == LA),
                         dict(histtype="step", lw=2)),
                        (f"{LB} (mai visti)", (~seen_mask) & (lin == LB),
                         dict(histtype="step", lw=2))]:
        if m.any():
            ax.hist(score[m], bins=60, density=True, label=lab, **sty)
    ax.axvline(tau, color="k", ls="--", lw=1.5, label=f"soglia tau={tau:+.3f}")
    ax.set_xlabel(f"score  =  d(centroide {LA}) - d(centroide {LB})   "
                  f"[<0 -> {LA}, >0 -> {LB}]")
    ax.set_ylabel("densita'")
    ax.set_title("Decisione di lineage: soglia calibrata sui visti, applicata ai mai visti")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "score_distribution.png", dpi=140)
    plt.close(fig)
    print(f"  figure -> {out}/pca_lineage.png , centroid_distances.png , score_distribution.png")


if __name__ == "__main__":
    main()
