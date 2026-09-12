"""
show_patch_positions.py - Fase 8 (figura di metodo)
Illustra DOVE guardano le due politiche di estrazione delle patch, su volti reali
delle due lineage.

  flat -> fra 16 posizioni casuali sceglie quella a varianza minima. Le patch
          cadono su zone uniformi (guancia, fronte, fondo): poco contenuto,
          molta texture. Ma la scelta DIPENDE DAL CONTENUTO, quindi in principio
          la posizione potrebbe correlare con la classe.
  grid -> posizioni FISSE ai centri dei quadranti, IDENTICHE per ogni immagine e
          per ogni classe. Nessuna differenza fra classi puo' derivare da dove si
          e' guardato. E' il controllo piu' stretto, al prezzo di patch meno
          uniformi (ci finiscono occhi e bocca).

Uso (nel .venv, dalla radice):
  python scripts/show_patch_positions.py --manifest data/manifest_lineage_pm128.csv
  python scripts/show_patch_positions.py --size 32 --out report/figures/patch_positions_32.png
"""
import argparse
import sys
from pathlib import Path

# lanciato come script diretto (come gli altri in scripts/): la radice del repo
# non e' su sys.path, quindi la aggiungiamo per poter importare src.*
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.data.siamese_dataset import PatchExtractor, load_image

INK, INK_SOFT = "#0b0b0b", "#52514e"
C_FLAT, C_GRID = "#eb6834", "#2a78d6"      # slot 2 e 1 della palette di riferimento


def positions(policy, size, k, img, seed=42):
    """Ricava le posizioni effettive confrontando i crop con l'immagine."""
    p = PatchExtractor(size, policy, seed=seed, patches_per_image=k)
    a = np.asarray(img.convert("L"), dtype=np.int16)
    out = []
    for i in range(k):
        crop = np.asarray(p(img, i, i).convert("L"), dtype=np.int16)
        # ricerca esatta della posizione del crop nell'immagine
        best, bd = (0, 0), None
        H, W = a.shape
        for y in range(0, H - size + 1, 4):
            for x in range(0, W - size + 1, 4):
                d = np.abs(a[y:y + size, x:x + size] - crop).sum()
                if bd is None or d < bd:
                    bd, best = d, (x, y)
                if bd == 0:
                    break
            if bd == 0:
                break
        out.append(best)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest_lineage_pm128.csv")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="report/figures/patch_positions.png")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, dtype={"source_id": str}, keep_default_na=False)
    df["cls"] = np.where(df.label == "real", "real_" + df.source_dataset, df.architecture)
    want = ["real_celeba", "stargan", "real_ffhq", "stylegan3"]
    want = [c for c in want if (df.cls == c).any()]
    rng = np.random.default_rng(args.seed)
    picks = [(c, df[df.cls == c].sample(1, random_state=args.seed).iloc[0].path)
             for c in want]

    S = args.size
    fig, axes = plt.subplots(2, len(picks), figsize=(2.5 * len(picks), 5.6))
    axes = np.atleast_2d(axes)
    for j, (cls, path) in enumerate(picks):
        img = load_image(path)
        for r, (pol, col) in enumerate([("flat", C_FLAT), ("grid", C_GRID)]):
            ax = axes[r, j]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            for (x, y) in positions(pol, S, args.k, img, seed=42):
                ax.add_patch(Rectangle((x, y), S, S, fill=False, ec=col, lw=2.2))
            if j == 0:
                ax.set_ylabel(f"{pol}", fontsize=11, color=INK,
                              fontweight="semibold", labelpad=8)
            if r == 0:
                ax.set_title(cls, fontsize=9.5, color=INK_SOFT)

    fig.suptitle(f"Dove guardano le due politiche  -  patch {S}x{S}, {args.k} per immagine\n"
                 "flat: varianza minima (dipende dal contenuto)   |   "
                 "grid: posizioni fisse, identiche per ogni immagine",
                 fontsize=11, color=INK, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura -> {out}")


if __name__ == "__main__":
    main()
