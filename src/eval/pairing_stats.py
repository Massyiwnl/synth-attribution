"""
pairing_stats.py - Fase 8
Misura DI CHE COSA sono fatte le coppie che la rete impara ad avvicinare.

Serve a rendere verificabile l'affermazione centrale sul ruolo dei dati reali.
Nel lavoro precedente il task era una classificazione piatta in cui real_celeba e
StarGAN erano due classi DIVERSE: la supervisione insegnava a separarle, quindi il
modello non poteva costruire alcuna nozione di appartenenza fra un dataset e i
modelli addestrati su di esso. Con il pairing per lineage sono la STESSA classe, e
la loss le avvicina esplicitamente.

La statistica che conta e' la quota di coppie genuine (real, fake): sono quelle che
trasportano l'informazione "questa foto autentica e questa immagine generata
vengono dallo stesso posto".

Uso:
  python -m src.eval.pairing_stats --manifest data/manifest_lineage_pm128.csv
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from src.data.siamese_dataset import SiamesePairDataset
from src.models.backbones import build_backbone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest_lineage_pm128.csv")
    ap.add_argument("--policy", default="lineage")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stride", type=int, default=1,
                    help="campiona una coppia ogni N indici (1 = tutte)")
    ap.add_argument("--out", default="report/pairing_composition.json")
    args = ap.parse_args()

    _, _, spec = build_backbone("resnet18", pretrained=False, image_size=256)
    ds = SiamesePairDataset(args.manifest, "train", spec, policy=args.policy,
                            seed=args.seed)

    df = pd.read_csv(args.manifest, dtype={"source_id": str}, keep_default_na=False)
    tr = df[df.split == "train"]
    comp = dict(manifest=args.manifest, policy=args.policy,
                train_images=int(len(tr)),
                train_real=int((tr.label == "real").sum()),
                train_fake=int((tr.label == "fake").sum()),
                pairing_groups={k: len(v) for k, v in sorted(ds.groups.items())})

    # ricostruisce le coppie con lo stesso RNG deterministico del dataset
    gen, imp = Counter(), Counter()
    for i in range(0, len(ds), args.stride):
        rng = random.Random(ds.seed * 1_000_003 + i)
        r = ds.df.iloc[i]
        if rng.random() < ds.genuine_prob:
            j = ds._genuine_partner(i, r, rng)
            gen[tuple(sorted((r.label, ds.df.iloc[j].label)))] += 1
        else:
            j = ds._impostor_partner(i, r, rng)
            imp[tuple(sorted((r.label, ds.df.iloc[j].label)))] += 1

    def norm(c):
        t = sum(c.values())
        return {f"{a}+{b}": dict(n=int(v), frac=v / t) for (a, b), v in sorted(c.items())}, t

    comp["genuine"], ng = norm(gen)
    comp["impostor"], ni = norm(imp)
    comp["n_genuine"], comp["n_impostor"] = int(ng), int(ni)

    print(f"Training: {comp['train_images']} immagini "
          f"({comp['train_real']} reali, {comp['train_fake']} generate)")
    print("Gruppi di pairing:", comp["pairing_groups"])
    print(f"\n== COPPIE GENUINE (stessa lineage, la loss le AVVICINA)  n={ng} ==")
    for k, v in comp["genuine"].items():
        print(f"  {k:12s} {v['n']:6d}  {v['frac']*100:5.1f}%")
    print(f"\n== COPPIE IMPOSTORE (lineage diverse, la loss le ALLONTANA)  n={ni} ==")
    for k, v in comp["impostor"].items():
        print(f"  {k:12s} {v['n']:6d}  {v['frac']*100:5.1f}%")

    rf = comp["genuine"].get("fake+real", {}).get("frac")
    if rf:
        print(f"\n-> il {rf*100:.1f}% delle coppie genuine e' (reale, generata): e' la")
        print("   supervisione che lega un dataset autentico ai modelli addestrati su")
        print("   di esso. Nel task di classificazione piatto quella coppia era NEGATIVA.")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(comp, indent=2))
    print(f"\nReport -> {out}")


if __name__ == "__main__":
    main()
