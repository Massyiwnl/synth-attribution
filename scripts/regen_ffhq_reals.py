"""
regen_ffhq_reals.py - Fase 6 (rimozione del confound di resize)
Rigenera i reali FFHQ partendo da immagini ad ALTA risoluzione (ideale: 1024)
e portandole a 256 con la STESSA pipeline delle fake StyleGAN (crop quadrato
centrale -> Lanczos 256). Cosi' reali e fake condividono il ricampionamento e il
confound "downscale di Kaggle vs nostro Lanczos" sparisce.

Modalita':
  --match-existing (default): rigenera ESATTAMENTE gli stessi 3000 volti gia' usati,
      cercando per nome file (stem) le versioni ad alta risoluzione in --src.
      Isola l'effetto del resize a parita' di contenuto.
  --no-match: campiona n volti freschi da --src (se i nomi non coincidono).

Uso:
  python scripts/regen_ffhq_reals.py --src <DIR_FFHQ_1024> --out data/raw/real/ffhq
  (poi rigenera il manifest e rilancia deep + handcrafted)
"""
import argparse
import random
from pathlib import Path

from PIL import Image

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(d):
    return sorted(p for p in Path(d).rglob("*")
                  if p.is_file() and p.suffix.lower() in VALID_EXT)


def canonical(img, size):
    """Crop quadrato centrale -> resize Lanczos (identica a prepare_reals)."""
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    return img.resize((size, size), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="cartella con FFHQ ad alta risoluzione (1024 ideale)")
    ap.add_argument("--out", default="data/raw/real/ffhq")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--match-existing", action="store_true", default=True)
    ap.add_argument("--no-match", dest="match_existing", action="store_false")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    src_files = list_images(args.src)
    if not src_files:
        raise SystemExit(f"Nessuna immagine in --src {args.src}")
    by_stem = {p.stem: p for p in src_files}
    print(f"Sorgente: {len(src_files)} immagini in {args.src}")

    targets = []
    if args.match_existing and out.exists():
        existing = sorted(out.glob("*.png"))
        stems = [p.stem for p in existing]
        matched = [by_stem[s] for s in stems if s in by_stem]
        missing = [s for s in stems if s not in by_stem]
        print(f"Reali attuali: {len(stems)} | matchati a {args.size}px-sorgente: {len(matched)}")
        if missing:
            print(f"ATTENZIONE: {len(missing)} stem non trovati nel sorgente (es. {missing[:3]}). "
                  f"Per quelli i file attuali NON verranno toccati. Se i nomi non coincidono, "
                  f"rilancia con --no-match per ricampionare da zero.")
        targets = matched
    if not targets:
        random.Random(args.seed).shuffle(src_files)
        targets = src_files[: args.n]
        print(f"Campiono {len(targets)} volti freschi (seed {args.seed}).")

    # controllo risoluzione (avviso se il sorgente non e' davvero ad alta risoluzione)
    if targets:
        w0, h0 = Image.open(targets[0]).size
        if min(w0, h0) <= args.size:
            print(f"ATTENZIONE: il sorgente sembra a {w0}x{h0} (<= {args.size}). "
                  f"Per rimuovere il confound serve risoluzione MAGGIORE di {args.size} "
                  f"(idealmente 1024). Procedo comunque.")

    out.mkdir(parents=True, exist_ok=True)
    ok = 0
    for p in targets:
        try:
            canonical(Image.open(p), args.size).save(out / f"{p.stem}.png")
            ok += 1
        except Exception as e:
            print(f"  skip {p.name}: {e}")
    print(f"Rigenerati {ok} reali FFHQ ({args.size}px, crop+Lanczos, dalla stessa pipeline "
          f"delle fake) -> {out}")
    print("Ora: rigenera il manifest e rilancia deep + handcrafted.")


if __name__ == "__main__":
    main()
