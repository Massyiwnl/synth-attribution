"""
gen_attgan.py - Fase 1 (generazione editing-GAN: AttGAN)
Genera i fake AttGAN ESATTAMENTE sul sottoinsieme CelebA fissato da prepare_reals,
allineati: ogni output e' <source_id>__attgan.png.

Usa il modello pre-addestrato di elvisyjlin/AttGAN-PyTorch. Lo script istanzia
direttamente il Generator (legge la config da setting.txt) e legge gli attributi
originali dal CSV di CelebA. L'edit replica test.py del repo: flip dell'attributo
target + check_attribute_conflict + normalizzazione in [-1,1].

Note: AttGAN usa BatchNorm -> qui G.eval() e' corretto (a differenza di StarGAN).
Lo script NON usa il data.py del repo (che usa np.str/np.int, rimossi da numpy>=1.24)
e neutralizza l'import di torchsummary, cosi' non serve installare dipendenze extra.

Setup (sul portatile):
  git clone https://github.com/elvisyjlin/AttGAN-PyTorch    (cartella esterna)
  # Pesi: apri http://bit.ly/attgan-pretrain (rimanda a Google Drive), scarica
  #   128_shortcut1_inject1_none.zip e scompattalo in <repo>/output/, ottenendo
  #   <repo>/output/128_shortcut1_inject1_none/{setting.txt, checkpoint/weights.*.pth}

  python scripts/gen_attgan.py ^
      --repo "C:/Users/massi/Desktop/AttGAN-PyTorch" ^
      --exp-dir "C:/Users/massi/Desktop/AttGAN-PyTorch/output/128_shortcut1_inject1_none" ^
      --src-dir data/raw/real/celeba ^
      --attr-file "C:/Users/massi/Desktop/celeba/list_attr_celeba.csv" ^
      --out data/raw/fake/attgan ^
      --attr Blond_Hair --size 256
  (in PowerShell: tutto su UNA riga, senza i ^)
"""
import argparse
import json
import os
import sys
import types
from glob import glob
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm


def load_cfg(exp_dir):
    with open(os.path.join(exp_dir, "setting.txt")) as f:
        return json.load(f, object_hook=lambda d: argparse.Namespace(**d))


def find_latest_ckpt(exp_dir):
    files = glob(os.path.join(exp_dir, "checkpoint", "*.pth"))
    assert files, f"Nessun checkpoint in {os.path.join(exp_dir, 'checkpoint')}"
    return sorted(files, key=lambda x: int(x.rsplit(".", 2)[1]))[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="clone di elvisyjlin/AttGAN-PyTorch")
    ap.add_argument("--exp-dir", required=True,
                    help="cartella output/<experiment> (contiene setting.txt e checkpoint/)")
    ap.add_argument("--src-dir", default="data/raw/real/celeba")
    ap.add_argument("--attr-file", required=True, help="list_attr_celeba.csv")
    ap.add_argument("--out", default="data/raw/fake/attgan")
    ap.add_argument("--attr", default="Blond_Hair")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--test-int", type=float, default=1.0)
    args = ap.parse_args()

    # attgan.py importa torchsummary solo per le summary() dentro la classe AttGAN,
    # che noi NON usiamo (instanziamo Generator direttamente). Stub -> niente install.
    if "torchsummary" not in sys.modules:
        stub = types.ModuleType("torchsummary")
        stub.summary = lambda *a, **k: None
        sys.modules["torchsummary"] = stub

    sys.path.insert(0, args.repo)
    from attgan import Generator               # definizione del repo
    from data import check_attribute_conflict  # gestione conflitti attributi del repo

    cfg = load_cfg(args.exp_dir)
    attrs = list(cfg.attrs)
    assert args.attr in attrs, f"'{args.attr}' non e' tra gli attributi del modello: {attrs}"
    idx = attrs.index(args.attr)
    thres_int = float(getattr(cfg, "thres_int", 0.5))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G = Generator(
        cfg.enc_dim, cfg.enc_layers, cfg.enc_norm, cfg.enc_acti,
        cfg.dec_dim, cfg.dec_layers, cfg.dec_norm, cfg.dec_acti,
        len(attrs), cfg.shortcut_layers, cfg.inject_layers, cfg.img_size,
    ).to(dev)
    # checkpoint accademico fidato -> weights_only=False (torch>=2.6 ha True di default)
    state = torch.load(find_latest_ckpt(args.exp_dir), map_location=dev, weights_only=False)
    G.load_state_dict(state["G"])
    G.eval()   # AttGAN usa BatchNorm: eval() e' corretto

    # attributi originali dal CSV di CelebA (valori -1/1 -> 0/1)
    df = pd.read_csv(args.attr_file)
    df = df.set_index(df.columns[0])           # prima colonna = image_id (es. 000001.jpg)

    pre = T.Compose([
        T.Resize((cfg.img_size, cfg.img_size)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in Path(args.src_dir).iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    done = skipped = 0
    with torch.no_grad():
        for p in tqdm(files, desc="AttGAN"):
            key = p.stem + ".jpg"              # gli image_id nel CSV finiscono in .jpg
            if key not in df.index:
                skipped += 1
                continue
            vals = df.loc[key, attrs].to_numpy()
            att_a = torch.tensor((vals + 1) // 2, dtype=torch.float).view(1, -1)
            att_b = att_a.clone()
            att_b[:, idx] = 1 - att_b[:, idx]                      # flip dell'attributo target
            att_b = check_attribute_conflict(att_b, attrs[idx], attrs)
            att_b_ = (att_b * 2 - 1) * thres_int                   # -> +/- thres_int
            att_b_[:, idx] = att_b_[:, idx] * args.test_int / thres_int  # target a intensita' piena
            x = pre(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
            y = G(x, att_b_.to(dev))                               # forward enc-dec
            y = F.interpolate(y, size=args.size, mode="bilinear", align_corners=False)
            save_image(y, out / f"{p.stem}__attgan.png", normalize=True, value_range=(-1., 1.))
            done += 1
    msg = f"AttGAN: {done} fake -> {out}"
    if skipped:
        msg += f"  ({skipped} immagini senza attributi nel CSV, saltate)"
    print(msg)


if __name__ == "__main__":
    main()
