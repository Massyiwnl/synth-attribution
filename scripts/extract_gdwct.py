"""
extract_gdwct.py - Fase 1 (GDWCT)
Il test-mode di GDWCT salva griglie 4x4 di celle 216px, con colonne
[A_reale, B_reale, A->B, B->A]: i FAKE sono le colonne 3 e 4 (indici 2,3),
le prime due sono i reali di input.

Questo script estrae i due pannelli fake da ogni riga di ogni griglia,
li porta a 256px (Lanczos) e li salva in --dst con nomi progressivi.

Uso (dalla root):
  python scripts/extract_gdwct.py --src <cartella_output_gdwct> --dst data/raw/fake/gdwct
  # opzionale: --limit 3000 per fermarsi a N fake
"""
import argparse
from pathlib import Path
from PIL import Image

CELL = 216          # dimensione nativa della cella GDWCT
COLS, ROWS = 4, 4   # griglia
FAKE_COLS = (2, 3)  # A->B e B->A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="cartella con le griglie di output GDWCT")
    ap.add_argument("--dst", default="data/raw/fake/gdwct")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--cell", type=int, default=CELL)
    ap.add_argument("--limit", type=int, default=0, help="max fake da estrarre (0 = tutti)")
    args = ap.parse_args()

    src = Path(args.src); dst = Path(args.dst); dst.mkdir(parents=True, exist_ok=True)
    grids = sorted([p for p in src.iterdir()
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if not grids:
        raise SystemExit(f"Nessuna griglia trovata in {src}")

    n = 0
    for gp in grids:
        im = Image.open(gp).convert("RGB")
        w, h = im.size
        cell = args.cell
        rows = h // cell
        # sanity: la griglia deve avere almeno 4 colonne
        if w // cell < 4:
            print(f"[skip] {gp.name}: larghezza {w} non compatibile con celle da {cell}")
            continue
        for r in range(rows):
            for c in FAKE_COLS:
                box = (c * cell, r * cell, (c + 1) * cell, (r + 1) * cell)
                face = im.crop(box).resize((args.size, args.size), Image.LANCZOS)
                face.save(dst / f"gdwct_{n:06d}.png")
                n += 1
                if args.limit and n >= args.limit:
                    print(f"Raggiunto il limite: {n} fake -> {dst}")
                    return
    print(f"Estratti {n} fake GDWCT -> {dst}")


if __name__ == "__main__":
    main()
