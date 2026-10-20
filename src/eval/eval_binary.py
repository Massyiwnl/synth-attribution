"""
eval_binary.py - Fase 8, variante BINARIA (decisione 1 del 19/10)

Dal verbale: sopra l'encoder siamese GIA' ADDESTRATO si allena un solo layer
lineare 128 -> 2, con etichetta di lineage (celeba / ffhq) e non di classe.
L'encoder resta congelato. Le due uscite sono lo spazio su cui si leggono
distanze fra medie, scatter e Grad-CAM (non l'embedding a 128 dimensioni).

Protocollo:
  - encoder congelato, nessun riaddestramento
  - testa: Linear(128, 2), cross-entropy, Adam, EPOCHS fisse (nessuno stop sul test)
  - training: solo TRAIN split, solo classi viste
  - soglia tau: punto di EER sul VAL split, solo classi viste
  - metriche: TEST split, campionato per classe come in eval_lineage.py

Lo stesso script rivaluta il metodo a centroidi in due varianti, per un confronto
allo stesso livello:
  - centroidi, protocollo originale: centroidi e soglia calcolati sul TEST (come
    eval_lineage.py). Deve riprodurre il report.json del run originale: e' il test
    di regressione (--published).
  - centroidi, pulito: centroidi sul TRAIN, soglia sul VAL.

Patch, politica e patch per immagine si leggono dal checkpoint (cfg['data']):
il default e' quello con cui il modello e' stato addestrato.

Uso:
  python -m src.eval.eval_binary --checkpoint runs/lineage_grid16_resnet18/best.pt \
      --n-per-class 800 --published report/lineage_grid16_resnet18/report.json \
      --out report/binary_grid16_resnet18
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.data.siamese_dataset import SingleImageDataset, labels_of
from src.models.siamese import load_encoder
from src.eval.eval_lineage import EDITING_GANS, class_distance_matrix, eer_threshold

SPLITS = ("train", "val", "test")
# verbale r. 267-270: verde = lineage CelebA, rosso = lineage FFHQ.
# Se si sceglie un'altra coppia (accessibilita'), si cambia solo qui.
COLOR = {"celeba": "#2ca02c", "ffhq": "#d62728"}
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def read_cfg(checkpoint):
    """Parametri di training letti dal checkpoint."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    d = ckpt["cfg"].get("data", {})
    return dict(manifest=d.get("manifest"),
                patch_size=d.get("patch_size"),
                patch_policy=d.get("patch_policy") or "flat",
                ppi=d.get("patches_per_image") or 1)


@torch.no_grad()
def embed_split(model, spec, manifest, split, cfg, device, bs, nw):
    ds = SingleImageDataset(manifest, split, spec, lineage_filter=None,
                            image_size=(cfg["patch_size"] or 256),
                            patch_size=cfg["patch_size"],
                            patches_per_image=cfg["ppi"],
                            patch_policy=cfg["patch_policy"])
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw,
                        pin_memory=(device == "cuda"))
    Z = np.concatenate([model.forward_one(x.to(device)).cpu().numpy()
                        for x, _ in loader]).astype(np.float32)
    return Z, ds.df


def labels(df, seen_set):
    arch = labels_of(df, "architecture")
    return dict(arch=arch, lin=labels_of(df, "lineage"), cls=labels_of(df, "class"),
                seen=np.array([(a == "real") or (a in seen_set) for a in arch]))


def balance(cls, n, seed):
    """Stesso campionamento di eval_lineage.py: al piu' n righe per classe."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(cls):
        ci = np.where(cls == c)[0]
        keep.append(rng.choice(ci, n, replace=False) if len(ci) > n else ci)
    return np.concatenate(keep)


def centroids(Z, lin, seen, LA, LB):
    return Z[seen & (lin == LA)].mean(0), Z[seen & (lin == LB)].mean(0)


def centroid_score(Z, cA, cB):
    """Come eval_lineage.py: >0 -> piu' vicino al centroide di LB."""
    return np.linalg.norm(Z - cA, axis=1) - np.linalg.norm(Z - cB, axis=1)


def metrics(score, lin, seen, tau, LA, LB):
    """Metriche di eval_lineage.py, con soglia tau fissata da fuori. Ritorna anche le predizioni."""
    y = (lin == LB).astype(int)
    pred = np.where(score > tau, LB, LA)
    unseen = ~seen
    m = dict(tau=float(tau),
             auc_seen=float(roc_auc_score(y[seen], score[seen])),
             auc_unseen=(float(roc_auc_score(y[unseen], score[unseen]))
                         if len(np.unique(y[unseen])) > 1 else None),
             acc_seen=float((pred[seen] == lin[seen]).mean()),
             acc_unseen=float((pred[unseen] == lin[unseen]).mean()),
             far_unseen=(float((pred[unseen & (lin == LA)] == LB).mean())
                         if (unseen & (lin == LA)).any() else None),
             frr_unseen=(float((pred[unseen & (lin == LB)] == LA).mean())
                         if (unseen & (lin == LB)).any() else None))
    return m, pred


def train_head(Z, y, epochs, lr, seed, device, Zv, yv):
    """Solo il layer lineare: l'encoder e' gia' congelato e messo in eval."""
    torch.manual_seed(seed)
    head = nn.Linear(Z.shape[1], 2).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    X = torch.from_numpy(Z).to(device)
    Y = torch.from_numpy(y).long().to(device)
    Xv = torch.from_numpy(Zv).to(device)
    Yv = torch.from_numpy(yv).long().to(device)
    gen = torch.Generator().manual_seed(seed)
    log = []
    for ep in range(epochs):
        perm = torch.randperm(len(X), generator=gen).to(device)
        tot = 0.0
        for i in range(0, len(X), 256):
            idx = perm[i:i + 256]
            loss = nn.functional.cross_entropy(head(X[idx]), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        with torch.no_grad():
            lv = head(Xv)
            log.append(dict(epoch=ep + 1, train_loss=tot / len(X),
                            val_loss=float(nn.functional.cross_entropy(lv, Yv)),
                            val_acc=float((lv.argmax(1) == Yv).float().mean())))
    head.eval()
    return head, log


@torch.no_grad()
def logits(head, Z, device):
    return head(torch.from_numpy(Z).to(device)).cpu().numpy()


def save_figures(Lt, s, pred, T, per_class, tau, LA, LB, out):
    """Figure con etichette in inglese (verbale r. 1-3, 341-347)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = sorted(per_class)

    # 1. spazio di output a 2 dimensioni: sostituisce la PCA
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for k, c in enumerate(classes):
        m = T["cls"] == c
        v = per_class[c]
        col = COLOR[v["lineage"]]
        ax.scatter(Lt[m, 0], Lt[m, 1], s=12, alpha=0.6,
                   marker=MARKERS[k % len(MARKERS)],
                   facecolors=col if v["seen"] else "none", edgecolors=col,
                   label=f"{c}{'' if v['seen'] else ' (unseen)'}")
    lo, hi = np.percentile(Lt, [0.5, 99.5])
    xs = np.linspace(lo, hi, 50)
    ax.plot(xs, xs + tau, "k--", lw=1, label=f"decision boundary (tau = {tau:+.3f})")
    ax.set_xlabel(f"logit {LA} (CelebA lineage)")
    ax.set_ylabel(f"logit {LB} (FFHQ lineage)")
    ax.legend(fontsize=8, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(out / "output_space_2d.png", dpi=140)
    plt.close(fig)

    # 2. distribuzione dello score con la soglia
    fig, ax = plt.subplots(figsize=(8, 4.8))
    lo, hi = np.percentile(s, [0.5, 99.5])
    bins = np.linspace(lo, hi, 80)
    for lineage in (LA, LB):
        for is_seen in (True, False):
            m = (T["lin"] == lineage) & (T["seen"] == is_seen)
            if not m.any():
                continue
            lab = f"{lineage}, {'seen' if is_seen else 'unseen'}"
            if is_seen:
                ax.hist(s[m], bins=bins, density=True, color=COLOR[lineage],
                        alpha=0.5, label=lab)
            else:
                ax.hist(s[m], bins=bins, density=True, histtype="step",
                        color=COLOR[lineage], lw=2, label=lab)
    ax.axvline(tau, color="k", ls="--", lw=1.2, label=f"tau = {tau:+.3f}")
    ax.set_xlabel(f"score = logit {LB} - logit {LA}   (below 0: {LA}; above 0: {LB})")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "score_distribution.png", dpi=140)
    plt.close(fig)

    # 3. matrice 2x2 sui generati: lineage vera (riga) contro predetta (colonna)
    fake = T["arch"] != "real"
    panels = [("all generated images (test)", fake),
              ("unseen generators only", fake & (~T["seen"]))]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, (name, m) in zip(axes, panels):
        C = np.array([[np.sum((T["lin"][m] == t) & (pred[m] == p)) for p in (LA, LB)]
                      for t in (LA, LB)], dtype=float)
        R = C / C.sum(1, keepdims=True).clip(min=1)
        ax.imshow(R, vmin=0, vmax=1, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{R[i, j] * 100:.1f}%\nn = {int(C[i, j])}",
                        ha="center", va="center", fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"predicted {LA}", f"predicted {LB}"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels([f"true {LA}", f"true {LB}"])
        ax.set_title(name, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "confusion_fakes_2x2.png", dpi=140)
    plt.close(fig)


def fmt(name, m):
    f = lambda v: "   n/a" if v is None else f"{v:6.4f}"
    p = lambda v: "  n/a" if v is None else f"{v * 100:5.1f}%"
    return (f"{name:<30s} {f(m['auc_seen'])} {f(m['auc_unseen'])} "
            f"{p(m['acc_seen'])} {p(m['acc_unseen'])} "
            f"{p(m['far_unseen'])} {p(m['frr_unseen'])}  {m['tau']:+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default=None, help="default: quello del checkpoint")
    ap.add_argument("--n-per-class", type=int, default=400,
                    help="righe per classe nel TEST (400 immagini intere, 800 patch: "
                         "e' cio' che hanno usato i run originali)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--published", default=None,
                    help="report.json originale, per il test di regressione del metodo a centroidi")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = read_cfg(args.checkpoint)
    manifest = args.manifest or cfg["manifest"]
    model, spec, epoch = load_encoder(args.checkpoint, device)

    full = pd.read_csv(manifest, dtype={"source_id": str}, keep_default_na=False)
    seen_set = set(full[full.split == "train"].architecture.unique())
    print(f"checkpoint epoca {epoch} | manifest {manifest} | "
          f"patch {cfg['patch_size']} {cfg['patch_policy']} x{cfg['ppi']} | device {device}")
    print(f"architetture viste: {sorted(seen_set)}")

    Z, L = {}, {}
    for split in SPLITS:
        Z[split], df = embed_split(model, spec, manifest, split, cfg, device,
                                   args.batch_size, args.num_workers)
        L[split] = labels(df, seen_set)
        print(f"  {split:5s}: {len(Z[split])} righe")

    lineages = sorted(np.unique(L["test"]["lin"]).tolist())
    if len(lineages) != 2:
        raise SystemExit(f"attese 2 lineage, trovate {lineages}")
    LA, LB = lineages

    # il test viene campionato come in eval_lineage.py, per confrontare numeri uguali
    keep = balance(L["test"]["cls"], args.n_per_class, args.seed)
    Zt = Z["test"][keep]
    T = {k: v[keep] for k, v in L["test"].items()}
    yt = (T["lin"] == LB).astype(int)

    # --- metodo a centroidi, protocollo originale (centroidi e soglia sul TEST) ---
    cA, cB = centroids(Zt, T["lin"], T["seen"], LA, LB)
    s_old = centroid_score(Zt, cA, cB)
    tau_old, eer_old = eer_threshold(s_old[T["seen"]], yt[T["seen"]])
    m_old, _ = metrics(s_old, T["lin"], T["seen"], tau_old, LA, LB)
    m_old["eer_seen"] = eer_old

    # --- metodo a centroidi, pulito (centroidi sul TRAIN, soglia sul VAL) ---
    Ltr, Lv = L["train"], L["val"]
    cA_c, cB_c = centroids(Z["train"], Ltr["lin"], Ltr["seen"], LA, LB)
    yv = (Lv["lin"] == LB).astype(int)
    tau_c, eer_c = eer_threshold(centroid_score(Z["val"], cA_c, cB_c)[Lv["seen"]],
                                 yv[Lv["seen"]])
    m_c, _ = metrics(centroid_score(Zt, cA_c, cB_c), T["lin"], T["seen"], tau_c, LA, LB)
    m_c["eer_seen"] = eer_c

    # --- classificatore binario sopra l'encoder congelato ---
    ytr = (Ltr["lin"] == LB).astype(int)
    tr_m, va_m = Ltr["seen"], Lv["seen"]
    head, hlog = train_head(Z["train"][tr_m], ytr[tr_m], args.epochs, args.lr,
                            args.seed, device, Z["val"][va_m], yv[va_m])
    Lval = logits(head, Z["val"], device)
    sval = Lval[:, 1] - Lval[:, 0]
    tau_b, eer_b = eer_threshold(sval[va_m], yv[va_m])
    Lt = logits(head, Zt, device)                       # spazio di output a 2 dim
    s_b = Lt[:, 1] - Lt[:, 0]
    m_b, pred_b = metrics(s_b, T["lin"], T["seen"], tau_b, LA, LB)
    m_b["eer_seen"] = eer_b
    # riferimento: stessa testa, soglia calibrata sul test (come il protocollo originale)
    tau_bt, _ = eer_threshold(s_b[T["seen"]], yt[T["seen"]])
    m_bt, _ = metrics(s_b, T["lin"], T["seen"], tau_bt, LA, LB)

    per_class = {}
    for c in np.unique(T["cls"]):
        m = T["cls"] == c
        a = T["arch"][m][0]
        per_class[c] = dict(
            lineage=str(T["lin"][m][0]), seen=bool(T["seen"][m][0]),
            kind=("real" if a == "real" else "editing" if a in EDITING_GANS else "noise"),
            n=int(m.sum()), correct_rate=float((pred_b[m] == T["lin"][m][0]).mean()),
            mean_score=float(s_b[m].mean()))

    # geometria nello spazio di output (verbale r. 280-307, decisione 1)
    classes = sorted(per_class)
    M2 = class_distance_matrix(Lt, T["cls"], classes)

    save_figures(Lt, s_b, pred_b, T, per_class, tau_b, LA, LB, out)
    torch.save(dict(state_dict=head.state_dict(), epochs=args.epochs, lr=args.lr,
                    seed=args.seed), out / "head.pt")
    np.savez(out / "test_output_2d.npz", L=Lt, cls=T["cls"], lin=T["lin"],
             arch=T["arch"], seen=T["seen"])

    print("\n                               AUC vis AUC mai  acc vis acc mai    FAR     FRR      tau")
    print(fmt("centroidi, protocollo orig.", m_old))
    print(fmt("centroidi, pulito", m_c))
    print(fmt("BINARIO, pulito", m_b))
    print(fmt("binario, tau su test (rif.)", m_bt))

    if args.published:
        pub = json.loads(Path(args.published).read_text())["threshold"]
        print("\nregressione vs report.json originale (centroidi, protocollo orig.):")
        for k, pk in [("auc_seen", "auc_seen"), ("auc_unseen", "auc_unseen"),
                      ("acc_seen", "acc_seen"), ("acc_unseen", "acc_unseen"), ("tau", "tau")]:
            ours, theirs = m_old[k], pub[pk]
            diff = None if (ours is None or theirs is None) else ours - theirs
            print(f"  {k:<11s} ours {ours}  published {theirs}  diff {diff}")

    report = dict(
        checkpoint=args.checkpoint, encoder_epoch=epoch, manifest=manifest,
        patch=dict(size=cfg["patch_size"], policy=cfg["patch_policy"], per_image=cfg["ppi"]),
        n_per_class_test=args.n_per_class, seen_architectures=sorted(seen_set),
        lineages=[LA, LB],
        head=dict(type="Linear(128, 2)", loss="cross_entropy", optimizer="Adam",
                  epochs=args.epochs, lr=args.lr, seed=args.seed, log=hlog),
        methods=dict(centroid_original_protocol=m_old,
                     centroid_clean=m_c,
                     binary_clean=m_b,
                     binary_tau_on_test_seen=m_bt),
        per_class=per_class,
        class_distance_matrix_2d=dict(classes=classes, matrix=M2.tolist()),
    )
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nReport -> {out / 'report.json'}")


if __name__ == "__main__":
    main()
