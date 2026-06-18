"""
prepare_reals.py - Fase 1 (ricostruzione da zero)
Prepara il pool di immagini reali (CelebA, FFHQ) in formato canonico 256x256
e FISSA il sottoinsieme CelebA che useremo come input per le editing-GAN.

Perche' un sottoinsieme fissato: StarGAN/AttGAN/GDWCT verranno eseguite ESATTAMENTE
su queste immagini, cosi' ogni fake ha la sua real sorgente allineata (Fasi 4-5).

Tutte le reali vengono portate a 256x256 con la STESSA pipeline (crop quadrato
centrale + resize Lanczos), per non introdurre un confound di risoluzione tra le
lineage (CelebA nativo 178x218, FFHQ 1024): dopo questo passo entrambe sono a 256.

Download (manuale, una tantum):
  - CelebA allineato: 'img_align_celeba' (178x218 JPEG). Mirror: Kaggle
    (jessicali9530/celeba-dataset) o HuggingFace. Una cartella con i .jpg.
  - FFHQ: bastano ~3000 immagini a risoluzione >= 256. Mirror FFHQ-256/1024
    (HuggingFace/Kaggle) oppure i 1024 ufficiali (NVlabs/ffhq-dataset).
    NB: evita i thumbnail 128, andrebbero upscalati.

Uso:
  python -m src.data.prepare_reals \\
      --celeba-dir /path/img_align_celeba \\
      --ffhq-dir   /path/ffhq256 \\
      --out data/raw --n 3000 --size 256 --seed 42
"""
import argparse
import random
from pathlib import Path

from PIL import Image

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(d):
    # ricorsivo: gestisce gli archivi Kaggle con immagini in sottocartelle
    # (es. CelebA -> img_align_celeba/img_align_celeba/*.jpg)
    return sorted(p for p in Path(d).rglob("*")
                  if p.is_file() and p.suffix.lower() in VALID_EXT)


def canonical(img: Image.Image, size: int) -> Image.Image:
    """Crop quadrato centrale -> resize size x size."""
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    return img.resize((size, size), Image.LANCZOS)


def process(src_files, out_dir: Path, size: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = []
    for p in src_files:
        try:
            im = canonical(Image.open(p), size)
        except Exception as e:
            print(f"  skip {p.name}: {e}")
            continue
        im.save(out_dir / f"{p.stem}.png")   # PNG: non aggiunge nuovi artefatti JPEG
        ids.append(p.stem)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celeba-dir", required=True)
    ap.add_argument("--ffhq-dir", required=True)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--subset-list", default="configs/celeba_subset.txt")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    out = Path(args.out)

    # --- CelebA: campiona N, salva crop canonici, registra la lista di id ---
    celeba = list_images(args.celeba_dir)
    print(f"CelebA: trovate {len(celeba)} immagini in {args.celeba_dir}")
    if not celeba:
        raise SystemExit("Nessuna immagine CelebA trovata: controlla --celeba-dir.")
    rnd.shuffle(celeba)
    celeba = celeba[: args.n]
    celeba_ids = process(celeba, out / "real" / "celeba", args.size)
    sub = Path(args.subset_list)
    sub.parent.mkdir(parents=True, exist_ok=True)
    sub.write_text("\n".join(celeba_ids) + "\n")
    print(f"CelebA: {len(celeba_ids)} immagini -> {out / 'real' / 'celeba'}")
    print(f"Lista sottoinsieme -> {sub}  (input per le editing-GAN)")

    # --- FFHQ: campiona N, salva crop canonici (nessun allineamento coi fake) ---
    ffhq = list_images(args.ffhq_dir)
    print(f"FFHQ:   trovate {len(ffhq)} immagini in {args.ffhq_dir}")
    if not ffhq:
        raise SystemExit("Nessuna immagine FFHQ trovata: controlla --ffhq-dir.")
    rnd.shuffle(ffhq)
    ffhq = ffhq[: args.n]
    ffhq_ids = process(ffhq, out / "real" / "ffhq", args.size)
    print(f"FFHQ:   {len(ffhq_ids)} immagini -> {out / 'real' / 'ffhq'}")


if __name__ == "__main__":
    main()
