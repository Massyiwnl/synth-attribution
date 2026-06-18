"""
build_manifest.py - Fase 1
Scansiona l'alberatura dei dati grezzi e produce un manifest CSV unico.

Alberatura attesa (vedi data/README.md):

    <raw_root>/
        real/celeba/<id>.jpg
        real/ffhq/<id>.png
        fake/stargan/<source_id>__stargan.png    # editing-GAN: source_id = id CelebA
        fake/attgan/<source_id>__attgan.png
        fake/gdwct/<source_id>__gdwct.png
        fake/stylegan/<id>.png                    # noise-GAN: nessuna sorgente
        fake/stylegan2/<id>.png

Per le editing-GAN il nome del file codifica l'immagine reale sorgente
(<source_id>__<arch>.ext): e' cio' che ci permette di costruire coppie fake/real
ALLINEATE (Fasi 4-5). Quando rigeneriamo i fake li nominiamo cosi'.

Uso:
    python -m src.data.build_manifest --config configs/dataset.yaml
"""
import argparse
import csv
from collections import Counter
from pathlib import Path
import random

import yaml
from PIL import Image

REAL_DIRS = {"celeba": "celeba", "ffhq": "ffhq"}
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def image_size(path: Path):
    try:
        with Image.open(path) as im:
            return im.width, im.height
    except Exception:
        return -1, -1


def parse_source_id(path: Path, arch: str, aligned: set) -> str:
    stem = path.stem
    if arch in aligned:
        # convenzione di naming: <source_id>__<arch>
        return stem.split("__")[0] if "__" in stem else stem
    return ""  # noise-GAN: nessuna sorgente allineata


def collect(raw_root: str, cfg: dict):
    raw = Path(raw_root)
    aligned = set(cfg["aligned_architectures"])
    rows = []

    # --- immagini reali ---
    for ds, sub in REAL_DIRS.items():
        d = raw / "real" / sub
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in VALID_EXT:
                continue
            w, h = image_size(p)
            rows.append(dict(path=str(p), label="real", source_dataset=ds,
                             architecture="real", source_id=p.stem,
                             width=w, height=h))

    # --- immagini fake ---
    fake_root = raw / "fake"
    if fake_root.exists():
        for arch_dir in sorted(fake_root.iterdir()):
            if not arch_dir.is_dir():
                continue
            arch = arch_dir.name.lower()
            src_ds = next((ds for ds, archs in cfg["lineage"].items()
                           if arch in archs), "")
            for p in sorted(arch_dir.iterdir()):
                if p.suffix.lower() not in VALID_EXT:
                    continue
                w, h = image_size(p)
                rows.append(dict(path=str(p), label="fake", source_dataset=src_ds,
                                 architecture=arch,
                                 source_id=parse_source_id(p, arch, aligned),
                                 width=w, height=h))
    return rows


def assign_splits(rows, cfg):
    rnd = random.Random(cfg["split"]["seed"])
    holdout = set(cfg["split"].get("holdout_architectures") or [])
    ratios = cfg["split"]["ratios"]

    # stratificazione per (label, architecture) per split bilanciati
    groups = {}
    for r in rows:
        groups.setdefault((r["label"], r["architecture"]), []).append(r)

    for (_label, arch), items in groups.items():
        if arch in holdout:
            for r in items:
                r["split"] = "test"        # architettura mai vista -> solo test
            continue
        rnd.shuffle(items)
        n = len(items)
        n_tr = int(n * ratios["train"])
        n_va = int(n * ratios["val"])
        for i, r in enumerate(items):
            if i < n_tr:
                r["split"] = "train"
            elif i < n_tr + n_va:
                r["split"] = "val"
            else:
                r["split"] = "test"
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rows = collect(cfg["paths"]["raw_root"], cfg)
    if not rows:
        print("Nessuna immagine trovata. Controlla raw_root e l'alberatura "
              "(vedi data/README.md).")
        return
    rows = assign_splits(rows, cfg)

    out = Path(cfg["paths"]["manifest"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "label", "source_dataset", "architecture",
              "source_id", "width", "height", "split"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"Manifest scritto: {out}  ({len(rows)} righe)")
    counts = Counter((r["architecture"], r["split"]) for r in rows)
    for (arch, split), c in sorted(counts.items()):
        print(f"  {arch:10s} {split:5s} {c:6d}")


if __name__ == "__main__":
    main()
