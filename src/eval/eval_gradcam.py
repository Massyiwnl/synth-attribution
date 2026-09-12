"""
eval_gradcam.py - Fase 8
Grad-CAM MEDIA per classe sull'encoder di lineage, e confronto quantitativo fra
le mappe medie.

Domanda: quando il modello riconduce un'immagine al suo dataset di addestramento,
GUARDA LE STESSE REGIONI per CelebA e per i generatori addestrati su CelebA?
Se si', e' un indizio che stia usando la stessa traccia.

Cosa produce:
  - mappa CAM media per classe (le facce di CelebA/FFHQ sono allineate, quindi
    la media spaziale ha senso);
  - mappa di SCARTO dalla media globale. E' la figura che conta davvero: una CAM
    media che evidenzia "la faccia" e' identica per tutte le classi e non
    dimostra nulla. Solo lo scarto mostra cosa e' SPECIFICO di quella classe;
  - matrice di correlazione fra le mappe di scarto: la versione numerica di
    "si focalizza sulle stesse regioni?". Attesa: correlazione alta dentro la
    stessa lineage, bassa fra lineage diverse.

CAUTELA. La Grad-CAM ha risoluzione pari alla griglia del target layer (8x8 per
ResNet a 256px, 16x16 per ViT/14 a 224). Una traccia di ricampionamento vive alle
alte frequenze ed e' spazialmente diffusa: e' quindi prevedibile che le mappe
siano poco strutturate. Una CAM piatta o rumorosa NON smentisce il risultato
quantitativo, semplicemente indica che la traccia non e' localizzata. Per questo
l'evidenza principale resta class_signatures.py (spettro/colore), e la Grad-CAM
va presentata come supporto qualitativo.

Uso:
  python -m src.eval.eval_gradcam --checkpoint runs/lineage_clip_l14/best.pt \
      --manifest data/manifest_lineage.csv --out report/gradcam_clip_l14
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.siamese_dataset import SingleImageDataset, labels_of
from src.models.siamese import load_encoder
from src.viz.gradcam import SiameseGradCAM, map_similarity


@torch.no_grad()
def embed_all(model, loader, device):
    return np.concatenate([model.forward_one(x.to(device)).cpu().numpy()
                           for x, _ in loader]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default="data/manifest_lineage.csv")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--n-per-class", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--cam-size", type=int, default=64, help="risoluzione di upsampling delle CAM")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="report/gradcam")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model, spec, epoch = load_encoder(args.checkpoint, device)
    print(f"Checkpoint epoca {epoch} | backbone input {spec.input_size}px/{spec.input_policy}")

    ds = SingleImageDataset(args.manifest, "test", spec, lineage_filter=None,
                            image_size=args.image_size)
    full = pd.read_csv(args.manifest, dtype={"source_id": str}, keep_default_na=False)
    seen = set(full[full.split == "train"].architecture.unique())

    cls = labels_of(ds.df, "class")
    lin = labels_of(ds.df, "lineage")
    arch = labels_of(ds.df, "architecture")

    # centroidi di lineage dalle sole classi VISTE (come in eval_lineage)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=args.num_workers)
    Z = embed_all(model, loader, device)
    seen_mask = np.array([(a == "real") or (a in seen) for a in arch])
    lineages = sorted(np.unique(lin).tolist())
    cent = {L: Z[seen_mask & (lin == L)].mean(0) for L in lineages}
    print("Centroidi di lineage calcolati su:", {L: int((seen_mask & (lin == L)).sum())
                                                 for L in lineages})

    rng = np.random.default_rng(args.seed)
    classes = sorted(np.unique(cls).tolist())
    cam_mean, scores_by_cls = {}, {}

    with SiameseGradCAM(model, device=device) as cam_fn:
        for c in classes:
            idx = np.where(cls == c)[0]
            if len(idx) > args.n_per_class:
                idx = rng.choice(idx, args.n_per_class, replace=False)
            own = lin[idx[0]]
            other = [L for L in lineages if L != own][0]
            acc, sc, n = 0.0, [], 0
            for b in range(0, len(idx), args.batch_size):
                bi = idx[b:b + args.batch_size]
                x = torch.stack([ds[i][0] for i in bi])
                m, s = cam_fn(x, cent[own], cent[other],
                              out_size=(args.cam_size, args.cam_size))
                acc = acc + m.sum(0); n += len(bi); sc.append(s)
            cam_mean[c] = acc / n
            scores_by_cls[c] = float(np.concatenate(sc).mean())
            print(f"  {c:>14s}  n={n:4d}  score medio {scores_by_cls[c]:+.4f}")

    # scarto dalla media globale: e' qui che si vede cosa e' SPECIFICO di una classe
    glob = np.mean([cam_mean[c] for c in classes], axis=0)
    delta = {c: cam_mean[c] - glob for c in classes}

    print("\n== Correlazione fra le mappe di SCARTO (stesse regioni?) ==")
    print("             " + "".join(f"{c[:10]:>11s}" for c in classes))
    C = np.zeros((len(classes), len(classes)))
    for i, a in enumerate(classes):
        for j, b in enumerate(classes):
            C[i, j] = map_similarity(delta[a], delta[b])
        print(f"{a[:11]:>11s}  " + "".join(f"{C[i, j]:11.3f}" for j in range(len(classes))))

    # medie within/cross lineage della correlazione
    lin_of = {c: lin[cls == c][0] for c in classes}
    within, cross = [], []
    for i, a in enumerate(classes):
        for j, b in enumerate(classes):
            if i >= j:
                continue
            (within if lin_of[a] == lin_of[b] else cross).append(C[i, j])
    print(f"\ncorrelazione media  within-lineage {np.mean(within):+.3f}  |  "
          f"cross-lineage {np.mean(cross):+.3f}")
    if np.mean(within) - np.mean(cross) > 0.15:
        print("  -> le classi della stessa lineage attivano regioni simili: coerente")
        print("     con l'ipotesi di una traccia condivisa.")
    else:
        print("  -> nessuna differenza netta: la traccia non e' localizzata")
        print("     spazialmente (atteso se vive alle alte frequenze).")

    save_figs(cam_mean, delta, classes, lin_of, seen, arch, cls, out)
    report = dict(checkpoint=args.checkpoint, epoch=epoch, classes=classes,
                  mean_score=scores_by_cls,
                  delta_correlation=C.tolist(),
                  mean_within_lineage_corr=float(np.mean(within)),
                  mean_cross_lineage_corr=float(np.mean(cross)),
                  cam_mean={c: cam_mean[c].tolist() for c in classes})
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nReport -> {out/'report.json'}")


def save_figs(cam_mean, delta, classes, lin_of, seen, arch, cls, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  (matplotlib assente: salto le figure)")
        return
    n = len(classes)
    is_seen = {c: (arch[cls == c][0] == "real") or (arch[cls == c][0] in seen)
               for c in classes}

    fig, axes = plt.subplots(2, n, figsize=(2.15 * n, 5.0))
    axes = np.atleast_2d(axes)
    vmax = max(np.abs(delta[c]).max() for c in classes)
    for j, c in enumerate(classes):
        axes[0, j].imshow(cam_mean[c], cmap="jet", vmin=0, vmax=1)
        axes[0, j].set_title(f"{c}\n{lin_of[c]}{'' if is_seen[c] else ' (mai visto)'}",
                             fontsize=7)
        axes[0, j].axis("off")
        axes[1, j].imshow(delta[c], cmap="bwr", vmin=-vmax, vmax=vmax)
        axes[1, j].axis("off")
    fig.suptitle("Grad-CAM media per classe (sopra) e scarto dalla media globale (sotto)\n"
                 "lo scarto e' la figura informativa: mostra cosa e' specifico della classe",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "gradcam_mean.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura -> {out}/gradcam_mean.png")


if __name__ == "__main__":
    main()
