"""
fetch_ffhq_1024.py - Fase 6 (sorgente reali ad alta risoluzione)
Scarica SOLO ~N immagini FFHQ a 1024x1024 leggendo un dataset WebDataset in
streaming (default: gaunernst/ffhq-1024-wds). Scarica solo gli shard necessari,
non l'intero dataset (89 GB).

Requisiti (nel .venv):  pip install -U datasets pillow huggingface_hub
Se il dataset risultasse gated: huggingface-cli login  (account HF gratuito) e
accetta i termini sulla pagina del dataset.

Uso:
  python scripts/fetch_ffhq_1024.py --n 3000 --out data/ffhq1024_subset
Poi:
  python scripts/regen_ffhq_reals.py --src data/ffhq1024_subset --out data/raw/real/ffhq --no-match
"""
import argparse
import io
from pathlib import Path

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="gaunernst/ffhq-1024-wds")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--min-size", type=int, default=1024)
    ap.add_argument("--out", default="data/ffhq1024_subset")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Manca 'datasets'. Installa: pip install -U datasets pillow huggingface_hub")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"Streaming da {args.repo} (scarico solo ~{args.n} immagini)...")
    ds = load_dataset(args.repo, split=args.split, streaming=True)

    n = 0
    for ex in ds:
        img = None
        for v in ex.values():
            if isinstance(v, Image.Image):
                img = v; break
            if isinstance(v, (bytes, bytearray)):
                try:
                    img = Image.open(io.BytesIO(v)); break
                except Exception:
                    pass
        if img is None:
            continue
        img = img.convert("RGB")
        if min(img.size) < args.min_size:
            continue
        img.save(out / f"ffhq1024_{n:05d}.png")
        n += 1
        if n % 250 == 0:
            print(f"  {n}/{args.n}")
        if n >= args.n:
            break

    print(f"Scaricate {n} immagini a >= {args.min_size}px -> {out}")
    if n < args.n:
        print("ATTENZIONE: meno immagini del previsto. Controlla auth/termini del dataset "
              "(huggingface-cli login) o prova un altro --repo.")


if __name__ == "__main__":
    main()
