"""
normalize_pipeline.py - Fase 8 (controllo del confound di lineage)

PROBLEMA. Nel dataset attuale le due lineage hanno catene di ricampionamento
diverse e perfettamente correlate con l'etichetta che vogliamo predire:

    lineage CelebA : 178x218 / 128 / 216  -> 256   (UPSAMPLING)
    lineage FFHQ   : 1024                 -> 256   (DOWNSAMPLING Lanczos)

Un solo scalare spettrale (energia media alle alte frequenze) separa gia' le due
lineage con AUC 0.93 per-immagine, senza alcun apprendimento. Quindi un modello
che "riconosce il dataset di addestramento" potrebbe in realta' riconoscere solo
la catena di resize: e' esattamente il confound gia' diagnosticato nella Sez. 9
della relazione, ma alla scala della lineage invece che dentro FFHQ.

SOLUZIONE. Far passare OGNI immagine di OGNI classe per la stessa identica catena
(size -> via -> size, Lanczos, opzionale ricompressione JPEG uniforme). Dopo
questo passo la firma di ricampionamento e' comune a tutte le classi e non puo'
piu' portare informazione sulla lineage.

LETTURA DEL RISULTATO. Si riallena lo stesso modello sul dataset normalizzato:
  - se il clustering per lineage SOPRAVVIVE -> il segnale non e' il ricampionamento:
    e' una proprieta' dei dati di addestramento. E' la prova che serve all'articolo.
  - se COLLASSA verso il caso -> il risultato originale era l'artefatto, e va
    dichiarato come tale.

Il valore di --via controlla la severita': piu' e' basso, piu' contenuto ad alta
frequenza viene distrutto (e piu' e' severo il test). Consigliato uno sweep:
128 (severo), 192 (medio), 224 (lieve).

Uso:
  python scripts/normalize_pipeline.py --src data/raw --dst data/raw_pm128 --via 128
  python scripts/normalize_pipeline.py --src data/raw --dst data/raw_pm192 --via 192
  # poi: build_manifest con raw_root=<dst> e manifest=data/manifest_pm128.csv
"""
import argparse
import io
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def process_one(job):
    src, dst, via, size, jpeg = job
    try:
        im = Image.open(src).convert("RGB")
        # catena IDENTICA per ogni classe: size -> via -> size
        im = im.resize((via, via), Image.LANCZOS).resize((size, size), Image.LANCZOS)
        if jpeg:
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=int(jpeg))
            buf.seek(0)
            im = Image.open(buf).convert("RGB")
        dst.parent.mkdir(parents=True, exist_ok=True)
        # PNG per tutti: elimina anche la differenza di formato fra le classi
        im.save(dst.with_suffix(".png"))
        return True
    except Exception as e:
        print(f"[errore] {src}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--via", type=int, default=128,
                    help="risoluzione del collo di bottiglia comune (piu' bassa = test piu' severo)")
    ap.add_argument("--size", type=int, default=256, help="risoluzione finale")
    ap.add_argument("--jpeg", type=int, default=None,
                    help="se impostato (es. 90), ricomprime TUTTE le classi alla stessa qualita'")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    src_root, dst_root = Path(args.src), Path(args.dst)
    if not src_root.exists():
        raise SystemExit(f"sorgente inesistente: {src_root}")

    jobs = []
    for p in sorted(src_root.rglob("*")):
        if p.is_file() and p.suffix.lower() in VALID_EXT:
            jobs.append((p, dst_root / p.relative_to(src_root), args.via,
                         args.size, args.jpeg))
    if not jobs:
        raise SystemExit(f"nessuna immagine trovata in {src_root}")

    jtag = f" + JPEG q={args.jpeg}" if args.jpeg else ""
    print(f"{len(jobs)} immagini | catena comune: {args.size} -> {args.via} -> "
          f"{args.size} (Lanczos){jtag}")
    print(f"{src_root}  ->  {dst_root}")

    ok = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_one, jobs, chunksize=64), 1):
            ok += bool(r)
            if i % 2000 == 0:
                print(f"  {i}/{len(jobs)}")
    print(f"Fatto: {ok}/{len(jobs)} immagini scritte in {dst_root}")
    print(f"\nProssimo passo:\n"
          f"  modifica configs/dataset.yaml -> paths.raw_root: {dst_root}\n"
          f"  (e paths.manifest: data/manifest_{dst_root.name}.csv), poi\n"
          f"  python -m src.data.build_manifest --config configs/dataset.yaml")


if __name__ == "__main__":
    main()
