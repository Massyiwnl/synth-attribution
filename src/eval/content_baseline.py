"""
content_baseline.py - Fase 8 (controllo del confound di CONTENUTO)

L'obiezione piu' forte al risultato di lineage e': "CelebA e FFHQ hanno
inquadratura, allineamento, illuminazione e demografia diverse; StyleGAN3 finisce
su FFHQ semplicemente perche' GENERA VOLTI CHE SEMBRANO FFHQ. Lo si vede a occhio,
non e' una traccia forense."

Questo script la mette alla prova nel modo piu' diretto possibile: usa le feature
di un VLM pre-addestrato SENZA ALCUN ADDESTRAMENTO. Le rappresentazioni di
CLIP/DINOv2 sono dominate dalla semantica: se bastano, il contenuto spiega tutto.

Protocollo (zero-shot, nessuna testa, nessun fine-tuning):
  1. si estraggono le feature grezze del backbone congelato;
  2. i centroidi delle due lineage sono calcolati SOLO SULLE IMMAGINI REALI
     (real_celeba, real_ffhq) - cioe' solo su contenuto autentico;
  3. ogni immagine generata viene assegnata al centroide piu' vicino.

Lettura:
  - AUC zero-shot ALTA  -> il contenuto semantico basta. Il risultato del modello
    addestrato non dimostra una traccia sub-percettiva: replica una somiglianza
    visibile. Va detto esplicitamente nell'articolo.
  - AUC zero-shot BASSA (~0.5) mentre il modello addestrato resta alto -> cio' che
    il modello usa NON e' nella rappresentazione semantica: e' una traccia di
    basso livello. E' il risultato che serve alla tesi.

Uso:
  python -m src.eval.content_baseline --backbone clip_vit_l14 \
      --manifest data/manifest_lineage_pm128.csv --out report/content_pm128
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.data.siamese_dataset import SingleImageDataset, labels_of
from src.models.backbones import build_backbone, freeze

EDITING_GANS = {"stargan", "attgan", "gdwct"}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="clip_vit_l14")
    ap.add_argument("--manifest", default="data/manifest_lineage.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--n-per-class", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patch-size", type=int, default=None)
    ap.add_argument("--patches-per-image", type=int, default=1)
    ap.add_argument("--patch-policy", default="flat")
    ap.add_argument("--out", default="report/content_baseline")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    net, feat_dim, spec = build_backbone(args.backbone, pretrained=True,
                                         image_size=args.image_size)
    net = freeze(net).to(device)
    print(f"Backbone {args.backbone} (feat={feat_dim}) ZERO-SHOT: nessun addestramento")

    ds = SingleImageDataset(args.manifest, args.split, spec, lineage_filter=None,
                            image_size=args.image_size, patch_size=args.patch_size,
                            patches_per_image=args.patches_per_image,
                            patch_policy=args.patch_policy)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device == "cuda"))
    Z = np.concatenate([net(x.to(device)).float().cpu().numpy()
                        for x, _ in loader]).astype(np.float32)
    # le feature grezze hanno norme molto diverse fra backbone: normalizzo, cosi'
    # la distanza e' confrontabile con quella dell'embedding addestrato
    Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)

    cls = labels_of(ds.df, "class")
    lin = labels_of(ds.df, "lineage")
    arch = labels_of(ds.df, "architecture")

    rng = np.random.default_rng(args.seed)
    keep = []
    for c in np.unique(cls):
        ci = np.where(cls == c)[0]
        keep.append(rng.choice(ci, args.n_per_class, replace=False)
                    if len(ci) > args.n_per_class else ci)
    keep = np.concatenate(keep)
    Z, cls, lin, arch = Z[keep], cls[keep], lin[keep], arch[keep]

    lineages = sorted(np.unique(lin).tolist())
    LA, LB = lineages
    real = arch == "real"
    if not real.any():
        raise SystemExit("servono immagini reali nello split per costruire i centroidi")
    cA = Z[real & (lin == LA)].mean(0)
    cB = Z[real & (lin == LB)].mean(0)
    score = np.linalg.norm(Z - cA, axis=1) - np.linalg.norm(Z - cB, axis=1)
    pred = np.where(score > 0, LB, LA)

    fake = ~real
    auc_fake = (roc_auc_score((lin[fake] == LB).astype(int), score[fake])
                if len(np.unique(lin[fake])) > 1 else None)
    print(f"\nCentroidi dai soli REALI ({int((real & (lin == LA)).sum())} {LA}, "
          f"{int((real & (lin == LB)).sum())} {LB})")
    print(f"AUC zero-shot sui soli GENERATI: "
          f"{'n/d' if auc_fake is None else f'{auc_fake:.4f}'}")

    print(f"\n{'classe':>14s} {'lineage':>7s} {'tipo':>8s} {'corretti':>9s}")
    per_class = {}
    for c in sorted(np.unique(cls).tolist()):
        m = cls == c
        a = arch[m][0]
        kind = "reale" if a == "real" else ("editing" if a in EDITING_GANS else "noise")
        rate = float((pred[m] == lin[m][0]).mean())
        per_class[c] = dict(lineage=lin[m][0], kind=kind, correct_rate=rate)
        print(f"{c:>14s} {lin[m][0]:>7s} {kind:>8s} {rate*100:8.1f}%")

    noise = [v["correct_rate"] for v in per_class.values() if v["kind"] == "noise"]
    print("\n--- lettura ---")
    if noise:
        r = float(np.mean(noise))
        print(f"noise-GAN assegnati alla lineage corretta SENZA addestramento: {r*100:.1f}%")
        if r >= 0.9:
            print("  -> il CONTENUTO SEMANTICO basta gia'. Il modello addestrato replica")
            print("     una somiglianza visibile, non dimostra una traccia sub-percettiva.")
        elif r >= 0.65:
            print("  -> il contenuto spiega una PARTE del risultato; il resto no.")
        else:
            print("  -> il contenuto NON basta: cio' che il modello addestrato sfrutta")
            print("     non e' nella rappresentazione semantica del VLM.")

    rep = dict(backbone=args.backbone, manifest=args.manifest, zero_shot=True,
               auc_fake_only=(None if auc_fake is None else float(auc_fake)),
               per_class=per_class)
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    print(f"\nReport -> {out/'report.json'}")


if __name__ == "__main__":
    main()
