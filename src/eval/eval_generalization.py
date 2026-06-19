"""
eval_generalization.py - Fase 3
Valuta la GENERALIZZAZIONE dell'embedding a un generatore MAI visto in training
(architettura tenuta in holdout -> presente solo nel test split).

Domanda chiave: l'embedding, allenato senza l'architettura held-out, la colloca
comunque come un cluster distinto e riconoscibile? Oppure la confonde con un
generatore noto (es. SG3 scambiato per SG2) o con i reali (fallimento di detection)?

Tutte le metriche sono calcolate su un sottoinsieme BILANCIATO del test split
(stesso numero di immagini per classe), cosi' che la classe held-out - che e'
tutta nel test - non droghi le statistiche.

Uso:
  python -m src.eval.eval_generalization \
      --checkpoint runs/ffhq_holdout_sg3/best.pt \
      --manifest data/manifest.csv \
      --lineage ffhq --holdout stylegan3 \
      --out report/ffhq_holdout_sg3
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.data.siamese_dataset import SingleImageDataset
from src.models.siamese import SiameseEncoder


@torch.no_grad()
def embed_test(checkpoint, manifest, lineage, size, device, batch_size, num_workers):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    mcfg = ckpt["cfg"]["model"]
    model = SiameseEncoder(mcfg["backbone"], mcfg["pretrained"], mcfg["embedding_dim"],
                           front_end=mcfg.get("front_end", "none")).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = SingleImageDataset(manifest, "test", size, lineage_filter=lineage)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device == "cuda"))
    Z = []
    for x, _ in loader:
        Z.append(model.forward_one(x.to(device)).cpu().numpy())
    Z = np.concatenate(Z).astype(np.float32)
    arch = ds.df["architecture"].to_numpy()
    return Z, arch, ckpt.get("epoch")


def balance(Z, arch, n_per_class, seed=0):
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(arch):
        ci = np.where(arch == c)[0]
        if len(ci) > n_per_class:
            ci = rng.choice(ci, n_per_class, replace=False)
        keep.append(ci)
    keep = np.concatenate(keep)
    return Z[keep], arch[keep]


def pdist(a, b):
    aa = (a * a).sum(1)[:, None]
    bb = (b * b).sum(1)[None, :]
    return np.sqrt(np.maximum(aa + bb - 2.0 * a @ b.T, 0.0))


def class_distance_matrix(Z, arch, classes):
    M = np.zeros((len(classes), len(classes)))
    for i, ca in enumerate(classes):
        a = Z[arch == ca]
        for j, cb in enumerate(classes):
            b = Z[arch == cb]
            d = pdist(a, b)
            if ca == cb:
                n = len(a)
                M[i, j] = (d.sum() - np.trace(d)) / max(n * (n - 1), 1)
            else:
                M[i, j] = d.mean()
    return M


def intra_spread(Z, arch, classes):
    out = {}
    for c in classes:
        zc = Z[arch == c]
        out[c] = float(np.linalg.norm(zc - zc.mean(0, keepdims=True), axis=1).mean())
    return out


def save_pca(Z, arch, classes, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except Exception:
        print("  (matplotlib/sklearn.decomposition non disponibili: salto lo scatter PCA. "
              "Per averlo: pip install matplotlib)")
        return
    p2 = PCA(n_components=2).fit_transform(Z)
    plt.figure(figsize=(7, 6))
    for c in classes:
        m = arch == c
        plt.scatter(p2[m, 0], p2[m, 1], s=8, alpha=0.6, label=c)
    plt.legend(); plt.title("Embedding test (PCA 2D)")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
    print(f"  scatter PCA -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--lineage", default=None, help="celeba | ffhq | (vuoto = tutte)")
    ap.add_argument("--holdout", required=True, help="architettura held-out, es. stylegan3")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--n-per-class", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="report/generalization")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    H = args.holdout

    Z, arch, epoch = embed_test(args.checkpoint, args.manifest, args.lineage,
                                args.image_size, device, args.batch_size, args.num_workers)
    print(f"Checkpoint epoca {epoch} | device {device} | test (lineage={args.lineage}): "
          f"{len(Z)} immagini")
    print("  per classe:", {c: int((arch == c).sum()) for c in np.unique(arch)})
    if H not in set(arch):
        print(f"ATTENZIONE: l'architettura held-out '{H}' non e' nel test split. "
              f"Hai rigenerato il manifest con holdout_architectures=[{H}]?")
        return

    Zb, ab = balance(Z, arch, args.n_per_class, args.seed)
    classes = sorted(np.unique(ab).tolist())
    np.savez(out / "embeddings.npz", Z=Zb, arch=ab)

    # --- distanze fra classi ---
    M = class_distance_matrix(Zb, ab, classes)
    print("\nDistanza media fra classi (diag = compattezza intra-classe):")
    print("           " + "".join(f"{c[:9]:>11s}" for c in classes))
    for i, c in enumerate(classes):
        print(f"  {c[:9]:>9s} " + "".join(f"{M[i, j]:11.3f}" for j in range(len(classes))))

    # --- 1-NN ---
    D = pdist(Zb, Zb)
    iu = np.triu_indices(len(Zb), k=1)
    d_pairs = D[iu]
    same_pairs = (ab[iu[0]] == ab[iu[1]]).astype(int)
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(1)
    nn_class = ab[nn]
    per_class_nn = {c: float((nn_class[ab == c] == c).mean()) for c in classes}
    print("\n1-NN same-class rate (la piu' vicina e' della stessa classe?):")
    for c in classes:
        print(f"  {c:>10s}: {per_class_nn[c]*100:5.1f}%")

    # confusione della classe held-out
    h_mask = ab == H
    h_nn = nn_class[h_mask]
    conf = {c: int((h_nn == c).sum()) for c in classes}
    print(f"\nDove finisce il 1-NN di '{H}' (held-out):")
    for c in classes:
        print(f"  -> {c:>10s}: {conf[c]} ({conf[c]/h_mask.sum()*100:4.1f}%)")

    # --- AUC di verifica ---
    overall_auc = roc_auc_score(same_pairs, -d_pairs)
    hi = (ab[iu[0]] == H) | (ab[iu[1]] == H)
    y_h = ((ab[iu[0]] == H) & (ab[iu[1]] == H)).astype(int)
    heldout_auc = (roc_auc_score(y_h[hi], -d_pairs[hi])
                   if len(np.unique(y_h[hi])) > 1 else None)
    print(f"\nAUC verifica complessiva (tutte le classi): {overall_auc:.4f}")
    if heldout_auc is not None:
        print(f"AUC verifica focalizzata su '{H}'        : {heldout_auc:.4f}")
        print(f"  (coppie {H}-{H} genuine vs {H}-altro impostore: alto = cluster separabile)")

    # --- verdetto sintetico ---
    h_rate = per_class_nn[H]
    nearest_known = max((c for c in classes if c != H), key=lambda c: -M[classes.index(H), classes.index(c)])
    print("\n--- lettura ---")
    if h_rate >= 0.85:
        print(f"GENERALIZZA: '{H}' (mai visto) forma un cluster coerente e separato "
              f"({h_rate*100:.0f}% dei 1-NN restano in classe).")
    elif h_rate >= 0.5:
        print(f"GENERALIZZAZIONE PARZIALE: '{H}' e' in parte separabile ({h_rate*100:.0f}%), "
              f"ma si confonde soprattutto con '{nearest_known}'. Possibile spazio per il front-end a residuo.")
    else:
        print(f"NON GENERALIZZA bene: '{H}' viene assorbito da '{nearest_known}' "
              f"({h_rate*100:.0f}% di 1-NN in classe). Qui il front-end a residuo ha senso.")

    save_pca(Zb, ab, classes, out / "pca_scatter.png")

    report = dict(checkpoint=args.checkpoint, epoch=epoch, lineage=args.lineage,
                  holdout=H, n_per_class=int(args.n_per_class), classes=classes,
                  class_distance_matrix=M.tolist(),
                  intra_spread=intra_spread(Zb, ab, classes),
                  nn_same_class_rate=per_class_nn,
                  heldout_nn_confusion=conf,
                  overall_verification_auc=float(overall_auc),
                  heldout_verification_auc=(None if heldout_auc is None else float(heldout_auc)))
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport -> {out/'report.json'}")


if __name__ == "__main__":
    main()
