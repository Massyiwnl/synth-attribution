"""
resize_dir.py - utility
Ridimensiona tutte le immagini di una cartella a size x size (Lanczos) e le salva
in PNG nella cartella di destinazione. Serve a uniformare a 256 i fake che escono
ad altra risoluzione (StyleGAN 1024, GDWCT 216), coerentemente con i reali.

Uso:
  python scripts/resize_dir.py --src <cartella_1024> --dst data/raw/fake/stylegan2 --size 256
"""
import argparse
from pathlib import Path

from PIL import Image

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in VALID_EXT:
            continue
        im = Image.open(p).convert("RGB").resize((args.size, args.size), Image.LANCZOS)
        im.save(dst / f"{p.stem}.png")
        n += 1
    print(f"Ridimensionate {n} immagini -> {dst}")


if __name__ == "__main__":
    main()
