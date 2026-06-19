"""
train_siamese.py - Fase 2/3
Allena la rete siamese (ResNet18 + contrastive loss) per l'attribution.
Validazione con metriche SEPARATE per within-celeba / within-ffhq / overall, cosi'
da distinguere la traccia generativa (within-lineage) dal confound di dataset.

Il modello migliore viene scelto sulla media delle AUC within-lineage disponibili
(la metrica onesta), non sull'overall (gonfiato dal confound cross-lineage).

Uso (dalla radice, con .venv + torch GPU):
  python -m src.train.train_siamese --config configs/train.yaml
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

from src.data.siamese_dataset import SiamesePairDataset
from src.models.siamese import SiameseEncoder
from src.losses.contrastive import ContrastiveLoss


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def compute_metrics(d, l):
    scores = -d                          # piu' alto -> piu' "genuine"
    auc = roc_auc_score(l, scores)
    fpr, tpr, thr = roc_curve(l, scores)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[i] + fnr[i]) / 2.0)
    acc = float(((scores >= thr[i]).astype(int) == l).mean())
    return auc, eer, acc


def bucket(D, L, mask):
    d, l = D[mask], L[mask]
    if len(d) == 0 or len(np.unique(l)) < 2:
        return None
    auc, eer, acc = compute_metrics(d, l)
    return dict(auc=auc, eer=eer, acc=acc, n=int(len(l)),
                gen_d=float(d[l == 1].mean()), imp_d=float(d[l == 0].mean()))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    D, L, C = [], [], []
    for x1, x2, y, lc in loader:
        z1, z2 = model(x1.to(device), x2.to(device))
        D.append(torch.nn.functional.pairwise_distance(z1, z2).cpu().numpy())
        L.append(y.numpy()); C.append(lc.numpy())
    D, L, C = np.concatenate(D), np.concatenate(L), np.concatenate(C)
    return {
        "overall": bucket(D, L, np.ones(len(L), dtype=bool)),
        "within_celeba": bucket(D, L, C == 0),
        "within_ffhq": bucket(D, L, C == 1),
    }


def fmt(name, b):
    if b is None:
        return f"  [{name:13s}] n/d"
    return (f"  [{name:13s}] AUC={b['auc']:.4f} EER={b['eer']:.4f} acc={b['acc']:.4f} "
            f"d_gen={b['gen_d']:.3f} d_imp={b['imp_d']:.3f} (n={b['n']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["train"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("ATTENZIONE: nessuna GPU, training molto lento. Reinstalla torch con CUDA.")

    d = cfg["data"]
    seed = cfg["train"]["seed"]
    lf = d.get("lineage_filter") or None
    print(f"Policy: {d['pairing_policy']}  lineage_filter: {lf}")
    train_ds = SiamesePairDataset(d["manifest"], "train", d["image_size"],
                                  policy=d["pairing_policy"], genuine_prob=d["genuine_prob"],
                                  seed=seed, lineage_filter=lf)
    val_ds = SiamesePairDataset(d["manifest"], "val", d["image_size"],
                                policy=d["pairing_policy"], genuine_prob=0.5,
                                seed=seed, lineage_filter=lf)
    print(f"Coppie: train={len(train_ds)}  val={len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=d["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                            num_workers=d["num_workers"], pin_memory=True)

    model = SiameseEncoder(cfg["model"]["backbone"], cfg["model"]["pretrained"],
                           cfg["model"]["embedding_dim"],
                           front_end=cfg["model"].get("front_end", "none")).to(device)
    criterion = ContrastiveLoss(cfg["loss"]["margin"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"])
    use_amp = bool(cfg["train"].get("amp", True)) and device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    out = Path(cfg["train"]["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    best = 0.0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['train']['epochs']}")
        for x1, x2, y, _ in pbar:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=("cuda" if use_amp else "cpu"), enabled=use_amp):
                z1, z2 = model(x1, x2)
                loss = criterion(z1, z2, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        sched.step()

        m = evaluate(model, val_loader, device)
        print(fmt("overall", m["overall"]))
        print(fmt("within_celeba", m["within_celeba"]))
        print(fmt("within_ffhq", m["within_ffhq"]))
        within = [b["auc"] for b in (m["within_celeba"], m["within_ffhq"]) if b]
        score = float(np.mean(within)) if within else m["overall"]["auc"]

        torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": cfg}, out / "last.pt")
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": cfg, "val": m},
                       out / "best.pt")
            print(f"  -> nuovo best (score within-lineage={best:.4f}) salvato")

    print(f"Fine. Best score within-lineage: {best:.4f}")


if __name__ == "__main__":
    main()
