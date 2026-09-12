"""
train_siamese.py - Fasi 2/3/8
Allena la rete siamese (backbone + contrastive loss) per l'attribuzione, in due
regimi a seconda di data.pairing_policy:
  - "architecture" -> attribuzione del GENERATORE (fasi 2/3). Validazione con
    metriche SEPARATE within-celeba / within-ffhq / overall, e selezione del best
    sulle within-lineage: l'overall sarebbe gonfiata dal confound cross-lineage.
  - "lineage"      -> attribuzione dei DATI DI ADDESTRAMENTO (fase 8). Qui le
    coppie impostore sono cross-lineage per costruzione (e' il segnale voluto),
    quindi la metrica di selezione corretta e' l'overall.

Il backbone puo' essere allenabile (resnet18/50) o CONGELATO (clip_*, dinov2_*),
nel qual caso si allena solo la testa di proiezione.

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

from src.data.siamese_dataset import SiamesePairDataset, LINEAGE_POLICIES
from src.models.siamese import SiameseEncoder
from src.models.backbones import is_timm
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
    # override opzionali (per il leave-one-generator-out lanciato da script)
    ap.add_argument("--lineage-filter", default=None, help="celeba | ffhq | all")
    ap.add_argument("--only-fake", default=None, help="allena su reali + SOLO questo generatore")
    ap.add_argument("--out", default=None, help="cartella di output (override train.out_dir)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--backbone", default=None, help="override model.backbone")
    ap.add_argument("--manifest", default=None, help="override data.manifest")
    ap.add_argument("--freeze-backbone", dest="freeze", action="store_true", default=None)
    ap.add_argument("--no-freeze-backbone", dest="freeze", action="store_false")
    ap.add_argument("--forensic-aug", dest="aug", action="store_true", default=None)
    ap.add_argument("--front-end", default=None, help="none | highpass | srm")
    ap.add_argument("--patch-size", type=int, default=None)
    ap.add_argument("--patches-per-image", type=int, default=None)
    ap.add_argument("--patch-policy", default=None, help="flat | random | grid")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    # applica gli override
    if args.lineage_filter is not None:
        cfg["data"]["lineage_filter"] = None if args.lineage_filter == "all" else args.lineage_filter
    if args.out is not None:
        cfg["train"]["out_dir"] = args.out
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.manifest is not None:
        cfg["data"]["manifest"] = args.manifest
    if args.aug is not None:
        cfg["data"]["forensic_aug"] = args.aug
    if args.backbone is not None:
        cfg["model"]["backbone"] = args.backbone
        # default sensato: i VLM si usano congelati, le ResNet end-to-end.
        # Un --freeze-backbone / --no-freeze-backbone esplicito ha la precedenza.
        if args.freeze is None:
            cfg["model"]["freeze_backbone"] = is_timm(args.backbone)
    if args.freeze is not None:
        cfg["model"]["freeze_backbone"] = args.freeze
    if args.front_end is not None:
        cfg["model"]["front_end"] = args.front_end
    if args.patch_size is not None:
        cfg["data"]["patch_size"] = args.patch_size
    if args.patches_per_image is not None:
        cfg["data"]["patches_per_image"] = args.patches_per_image
    if args.patch_policy is not None:
        cfg["data"]["patch_policy"] = args.patch_policy
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    only_fake = args.only_fake

    set_seed(cfg["train"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("ATTENZIONE: nessuna GPU, training molto lento. Reinstalla torch con CUDA.")

    d = cfg["data"]
    mc = cfg["model"]
    seed = cfg["train"]["seed"]
    lf = d.get("lineage_filter") or None
    policy = d["pairing_policy"]

    # con le patch e' la patch la vera risoluzione d'ingresso del backbone:
    # cosi' una CNN la riceve a dimensione nativa e il transform non reinterpola
    patch = d.get("patch_size") or None
    eff_size = patch or d["image_size"]

    # il modello va costruito PRIMA dei dataset: e' il backbone a dettare
    # risoluzione e normalizzazione delle immagini (BackboneSpec)
    model = SiameseEncoder(mc["backbone"], mc["pretrained"], mc["embedding_dim"],
                           front_end=mc.get("front_end", "none"),
                           freeze_backbone=mc.get("freeze_backbone", False),
                           image_size=eff_size,
                           head_hidden=mc.get("head_hidden"),
                           head_dropout=mc.get("head_dropout", 0.0),
                           input_policy=mc.get("input_policy")).to(device)
    spec = model.spec
    n_all = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Backbone: {mc['backbone']} (feat={model.feat_dim}, "
          f"{'CONGELATO' if model.frozen else 'allenabile'}) | input {spec.input_size}px "
          f"policy={spec.input_policy}")
    print(f"Parametri: {n_tr/1e6:.2f}M allenabili su {n_all/1e6:.2f}M totali")

    print(f"Policy: {policy}  lineage_filter: {lf}  only_fake: {only_fake}")
    aug = bool(d.get("forensic_aug", False))
    if aug:
        print("ForensicJitter ATTIVO in training (resample+JPEG casuali su tutte le classi)")
    pk = dict(patch_size=patch, patches_per_image=d.get("patches_per_image", 1),
              patch_policy=d.get("patch_policy", "flat"))
    if patch:
        print(f"PATCH {patch}x{patch} nativa, policy={pk['patch_policy']}, "
              f"{pk['patches_per_image']} per immagine")
    train_ds = SiamesePairDataset(d["manifest"], "train", spec,
                                  policy=policy, genuine_prob=d["genuine_prob"],
                                  seed=seed, lineage_filter=lf, only_fake=only_fake,
                                  image_size=eff_size, forensic_aug=aug, **pk)
    val_ds = SiamesePairDataset(d["manifest"], "val", spec,
                                policy=policy, genuine_prob=0.5,
                                seed=seed, lineage_filter=lf, only_fake=only_fake,
                                image_size=eff_size, **pk)
    print(f"Coppie: train={len(train_ds)}  val={len(val_ds)}")
    print("Gruppi di pairing:", {k: len(v) for k, v in sorted(train_ds.groups.items())})
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=d["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                            num_workers=d["num_workers"], pin_memory=True)

    criterion = ContrastiveLoss(cfg["loss"]["margin"])
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=cfg["train"]["lr"],
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
        # Quale AUC usare per scegliere il best dipende dal task:
        #  - pairing per ARCHITETTURA: l'overall e' gonfiata dalle coppie impostore
        #    cross-lineage (banali), quindi si seleziona sulle within-lineage;
        #  - pairing per LINEAGE: le impostore SONO per definizione cross-lineage
        #    (e' proprio il segnale che vogliamo), i bucket within degenerano a sole
        #    coppie genuine -> bucket() torna None. Qui l'overall E' la metrica giusta.
        if policy in LINEAGE_POLICIES:
            score = m["overall"]["auc"]
        else:
            within = [b["auc"] for b in (m["within_celeba"], m["within_ffhq"]) if b]
            score = float(np.mean(within)) if within else m["overall"]["auc"]

        # Con backbone congelato i suoi pesi sono identici a quelli pre-addestrati:
        # salvarli renderebbe ogni checkpoint da ~1.2 GB per ViT-L/14. Si salva
        # quindi il solo stato addestrabile; load_encoder ricostruisce il backbone
        # da timm e carica il resto con strict=False.
        def _state():
            if not model.frozen:
                return model.state_dict(), False
            keep = {k: v for k, v in model.state_dict().items()
                    if not k.startswith("backbone.")}
            return keep, True

        st, partial = _state()
        torch.save({"model": st, "partial": partial, "epoch": epoch, "cfg": cfg},
                   out / "last.pt")
        if score > best:
            best = score
            torch.save({"model": st, "partial": partial, "epoch": epoch,
                        "cfg": cfg, "val": m}, out / "best.pt")
            print(f"  -> nuovo best (score within-lineage={best:.4f}) salvato")

    print(f"Fine. Best score within-lineage: {best:.4f}")


if __name__ == "__main__":
    main()
