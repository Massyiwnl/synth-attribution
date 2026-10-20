"""
eval_robustness.py - Fase 8, robustezza al ridimensionamento (verbale r. 453-459)

Stesso modello, stessa testa binaria e stessa soglia calibrata sul VAL in condizioni
pulite (report di eval_binary.py). Le immagini di test vengono ridotte di un fattore
f (256 -> 256*f) e riportate a 256 con interpolazione bilineare, poi valutate.
Nessun riaddestramento: la domanda e' quanto segnale sopravvive alla perdita di
informazione, non quanto il modello si adatta.

Uso:
  python -m src.eval.eval_robustness --checkpoint runs/lineage_pm128_resnet18/best.pt \
      --binary-report report/binary_pm128_resnet18 --n-per-class 400 \
      --out report/robustness_pm128_resnet18
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.siamese_dataset import build_transform, load_image, read_manifest
from src.models.siamese import load_encoder
from src.eval.eval_binary import balance, labels, metrics, read_cfg, logits

FACTORS = [1.0, 0.75, 0.5, 0.25]


class ScaledDataset(Dataset):
    """Come SingleImageDataset, ma con la riduzione f -> 256 applicata prima del transform."""

    def __init__(self, df, spec, factor, size=256):
        self.df = df.reset_index(drop=True)
        self.factor = factor
        self.size = size
        self.tf = build_transform(spec, train=False, image_size=size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img = load_image(self.df.iloc[i].path)
        if self.factor < 1.0:
            small = max(8, int(round(self.size * self.factor)))
            img = img.resize((small, small), Image.BILINEAR)
            img = img.resize((self.size, self.size), Image.BILINEAR)
        return self.tf(img), i


def save_figure(results, out):
    """Figura con etichette in inglese (verbale r. 1-3, 341-347)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rs = sorted(results, key=lambda r: -r["factor"])
    x = [r["size"] for r in rs]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))

    ax = axes[0]
    ax.plot(x, [r["metrics"]["auc_unseen"] for r in rs], "o-", label="unseen generators")
    ax.plot(x, [r["metrics"]["auc_seen"] for r in rs], "s--", label="seen classes")
    ax.set_xlabel("resolution before restoring 256 px")
    ax.set_ylabel("AUC")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(x, [r["metrics"]["acc_unseen"] for r in rs], "o-", label="unseen generators")
    ax.plot(x, [r["metrics"]["acc_seen"] for r in rs], "s--", label="seen classes")
    if "stylegan3" in rs[0]["per_class_correct"]:
        ax.plot(x, [r["per_class_correct"]["stylegan3"] for r in rs], "^:",
                label="stylegan3 (unseen)")
    ax.set_xlabel("resolution before restoring 256 px")
    ax.set_ylabel("accuracy (fixed threshold)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "robustness_scaling.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--binary-report", required=True,
                    help="cartella di eval_binary.py: contiene head.pt e report.json")
    ap.add_argument("--manifest", default=None, help="default: quello del checkpoint")
    ap.add_argument("--n-per-class", type=int, default=400)
    ap.add_argument("--factors", type=float, nargs="*", default=FACTORS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = read_cfg(args.checkpoint)
    if cfg["patch_size"]:
        raise SystemExit("la prova di robustezza e' definita per immagini intere (patch assenti)")
    manifest = args.manifest or cfg["manifest"]

    bdir = Path(args.binary_report)
    brep = json.loads((bdir / "report.json").read_text())
    tau = brep["methods"]["binary_clean"]["tau"]          # soglia del val, condizioni pulite
    ck = torch.load(bdir / "head.pt", map_location=device, weights_only=False)
    head = nn.Linear(ck["state_dict"]["weight"].shape[1], 2).to(device)
    head.load_state_dict(ck["state_dict"])
    head.eval()

    model, spec, epoch = load_encoder(args.checkpoint, device)
    full = pd.read_csv(manifest, dtype={"source_id": str}, keep_default_na=False)
    seen_set = set(full[full.split == "train"].architecture.unique())

    df_test = read_manifest(manifest, "test")
    L = labels(df_test, seen_set)
    keep = balance(L["cls"], args.n_per_class, args.seed)   # stesso campione di eval_binary
    T = {k: v[keep] for k, v in L.items()}
    LA, LB = sorted(np.unique(T["lin"]).tolist())
    print(f"checkpoint epoca {epoch} | tau del val (pulito) {tau:+.4f} | "
          f"test campionato: {len(keep)} immagini")

    results = []
    for f in args.factors:
        ds = ScaledDataset(df_test.iloc[keep], spec, f)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device == "cuda"))
        with torch.no_grad():
            Z = np.concatenate([model.forward_one(x.to(device)).cpu().numpy()
                                for x, _ in loader]).astype(np.float32)
        Lg = logits(head, Z, device)
        s = Lg[:, 1] - Lg[:, 0]
        m, pred = metrics(s, T["lin"], T["seen"], tau, LA, LB)
        pc = {}
        for c in np.unique(T["cls"]):
            mm = T["cls"] == c
            pc[str(c)] = float((pred[mm] == T["lin"][mm][0]).mean())
        size = 256 if f >= 1.0 else max(8, int(round(256 * f)))
        results.append(dict(factor=f, size=size, metrics=m, per_class_correct=pc))
        print(f"  f={f:<5g} ({size:3d} px)  AUC vis {m['auc_seen']:.4f}  "
              f"AUC mai visti {m['auc_unseen']:.4f}  acc mai visti {m['acc_unseen'] * 100:.1f}%  "
              f"StyleGAN3 {pc.get('stylegan3', float('nan')) * 100:.1f}%")

    save_figure(results, out)
    report = dict(checkpoint=args.checkpoint, encoder_epoch=epoch, manifest=manifest,
                  binary_report=str(bdir), tau_clean=tau, n_per_class=args.n_per_class,
                  lineages=[LA, LB], results=results)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nReport -> {out / 'report.json'}")


if __name__ == "__main__":
    main()
