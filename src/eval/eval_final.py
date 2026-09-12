"""
eval_final.py - Fase 3 (modello unico "tutto su tutti")

Caratterizza lo spazio di embedding del modello finale:
  1. Attribuzione CLOSED-SET top-1: ogni classe di test viene divisa in
     gallery/probe; si calcolano i centroidi dalla gallery e si classifica la
     probe al centroide piu' vicino. Restituisce accuratezza top-1, accuratezza
     per-classe e matrice di confusione. (E' il numero di attribuzione in
     regime chiuso che mancava nel report.)
  2. Proiezione PCA 2D di tutte le classi, per vedere la struttura dei cluster
     (utile poi per capire dove cadono i dataset esterni in Fase 4).

Le classi seguono la stessa logica del training: i reali sono separati per
lineage (real_celeba, real_ffhq), i fake sono per architettura.

Uso:
  python -m src.eval.eval_final --checkpoint runs/final_all/best.pt --out report/final
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score

from src.data.siamese_dataset import SingleImageDataset
from src.models.siamese import load_encoder


def class_labels(df):
    out = []
    for _, r in df.iterrows():
        out.append(f"real_{r['source_dataset']}" if r["label"] == "real"
                   else r["architecture"])
    return np.array(out)


@torch.no_grad()
def embed(checkpoint, manifest, split, size, device, bs, nw):
    model, spec, _ = load_encoder(checkpoint, device)
    ds = SingleImageDataset(manifest, split, spec, lineage_filter=None,
                            image_size=size)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw)
    Z = [model.forward_one(x.to(device)).cpu().numpy() for x, _ in loader]
    return np.concatenate(Z).astype(np.float32), class_labels(ds.df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--n-per-class", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="report/final")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Z, y = embed(args.checkpoint, args.manifest, "test", args.image_size,
                 device, args.batch_size, args.num_workers)
    classes = sorted(np.unique(y).tolist())
    rng = np.random.default_rng(args.seed)

    # bilancia e dividi ogni classe in gallery / probe
    gal_idx, prb_idx = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        idx = idx[:args.n_per_class]
        h = len(idx) // 2
        gal_idx.append(idx[:h]); prb_idx.append(idx[h:])
    gal_idx = np.concatenate(gal_idx); prb_idx = np.concatenate(prb_idx)

    # centroidi dalla gallery
    cents = np.stack([Z[gal_idx][y[gal_idx] == c].mean(0) for c in classes])
    # classifica la probe al centroide piu' vicino
    d = np.linalg.norm(Z[prb_idx][:, None, :] - cents[None, :, :], axis=2)
    pred = np.array(classes)[d.argmin(1)]
    true = y[prb_idx]

    top1 = float(accuracy_score(true, pred))
    per_class = {c: float((pred[true == c] == c).mean()) for c in classes}
    cm = confusion_matrix(true, pred, labels=classes)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    report = dict(checkpoint=args.checkpoint, classes=classes,
                  top1_accuracy=top1, per_class_accuracy=per_class,
                  n_probe=int(len(prb_idx)))
    (out / "report.json").write_text(json.dumps(report, indent=2))

    # --- matrice di confusione ---
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=40, ha="right"); ax.set_yticklabels(classes)
    ax.set_xlabel("predetto"); ax.set_ylabel("vero")
    thr = cm.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=9)
    ax.set_title(f"Attribuzione closed-set (top-1 = {top1:.3f})")
    fig.tight_layout(); fig.savefig(out / "confusion.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- PCA 2D di tutte le classi (probe) ---
    p2 = PCA(n_components=2).fit_transform(Z[prb_idx])
    fig, ax = plt.subplots(figsize=(7.5, 6))
    cmap = plt.get_cmap("tab10")
    for k, c in enumerate(classes):
        m = true == c
        ax.scatter(p2[m, 0], p2[m, 1], s=8, alpha=0.6, color=cmap(k % 10), label=c)
    ax.legend(markerscale=2, fontsize=9, loc="best")
    ax.set_title("Spazio di embedding del modello finale (PCA 2D)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.tight_layout(); fig.savefig(out / "pca_final.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"Top-1 closed-set: {top1:.4f}  (probe={len(prb_idx)})")
    print("Per classe:")
    for c in classes:
        print(f"  {c:>14s}: {per_class[c]:.3f}")
    print(f"\nFigure -> {out}/confusion.png , {out}/pca_final.png")
    print(f"Report -> {out}/report.json")


if __name__ == "__main__":
    main()
