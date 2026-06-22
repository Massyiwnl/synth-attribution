"""
make_sample_grids.py
Genera figure di esempio del dataset per la relazione, leggendo data/manifest.csv.

Produce (in report/figures/):
  1) dataset_samples.png : griglia con una riga per classe (lineage + generatore)
                           e K campioni casuali per riga.
  2) aligned_pairs.png   : alcuni volti CelebA mostrati come real / StarGAN / AttGAN
                           (illustra le coppie allineate degli editing-GAN e la
                            "scorciatoia biondo": il fake cambia solo i capelli).

Uso (nel .venv, dalla root del repo):
  python scripts/make_sample_grids.py
  python scripts/make_sample_grids.py --per-class 4 --pairs 5 --seed 1
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_manifest(path):
    return pd.read_csv(path, dtype={"source_id": str}, keep_default_na=False)


def _imshow(ax, path):
    ax.axis("off")
    try:
        ax.imshow(Image.open(path).convert("RGB"))
    except Exception:
        ax.text(0.5, 0.5, "?", ha="center", va="center", transform=ax.transAxes)


def dataset_samples(df, out, per_class=3, seed=0):
    rng = np.random.default_rng(seed)
    order = [("celeba", "real"), ("celeba", "stargan"), ("celeba", "attgan"),
             ("ffhq", "real"), ("ffhq", "stylegan2"), ("ffhq", "stylegan3")]
    order = [(l, a) for (l, a) in order
             if ((df.source_dataset == l) & (df.architecture == a)).any()]
    rows = len(order)
    if rows == 0:
        print("  (nessuna classe trovata nel manifest: salto dataset_samples)")
        return
    fig, axes = plt.subplots(rows, per_class, figsize=(per_class * 1.9, rows * 1.9))
    axes = np.atleast_2d(axes)
    for i, (lin, arch) in enumerate(order):
        sub = df[(df.source_dataset == lin) & (df.architecture == arch)]
        idx = sub.index.to_numpy()
        pick = rng.choice(idx, min(per_class, len(idx)), replace=False)
        for j in range(per_class):
            ax = axes[i, j]
            _imshow(ax, df.loc[pick[j], "path"]) if j < len(pick) else ax.axis("off")
        label = "real" if arch == "real" else arch
        axes[i, 0].text(-0.12, 0.5, f"{lin}\n{label}", transform=axes[i, 0].transAxes,
                        ha="right", va="center", fontsize=10)
    fig.suptitle("Esempi del dataset per classe (lineage / generatore)", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def aligned_pairs(df, out, n=4, seed=0):
    rng = np.random.default_rng(seed)
    cel = df[df.source_dataset == "celeba"]
    archs = ["real", "stargan", "attgan"]
    present = [a for a in archs if (cel.architecture == a).any()]
    if len(present) < 2:
        print("  (servono almeno real + un editing-GAN in CelebA: salto aligned_pairs)")
        return
    by_arch = {a: set(cel[cel.architecture == a]["source_id"]) for a in present}
    common = sorted(set.intersection(*by_arch.values()))
    if not common:
        print("  (nessun source_id condiviso tra real/stargan/attgan: salto aligned_pairs. "
              "Controlla il naming degli editing-GAN.)")
        return
    pick = rng.choice(common, min(n, len(common)), replace=False)
    titles = {"real": "real", "stargan": "StarGAN", "attgan": "AttGAN"}
    fig, axes = plt.subplots(len(pick), len(present),
                             figsize=(len(present) * 1.9, len(pick) * 1.9))
    axes = np.atleast_2d(axes)
    for i, sid in enumerate(pick):
        for j, a in enumerate(present):
            ax = axes[i, j]
            r = cel[(cel.architecture == a) & (cel.source_id == sid)]
            _imshow(ax, r.iloc[0]["path"]) if len(r) else ax.axis("off")
            if i == 0:
                ax.set_title(titles[a], fontsize=11)
    fig.suptitle("Coppie allineate degli editing-GAN (stesso volto, edit 'Blond_Hair')",
                 y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out-dir", default="report/figures")
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = load_manifest(args.manifest)
    print("Genero le figure di esempio del dataset...")
    dataset_samples(df, out / "dataset_samples.png", args.per_class, args.seed)
    aligned_pairs(df, out / "aligned_pairs.png", args.pairs, args.seed)
    print("Fatto.")


if __name__ == "__main__":
    main()
