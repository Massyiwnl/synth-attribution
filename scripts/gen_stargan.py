"""
gen_stargan.py - Fase 1 (generazione editing-GAN: StarGAN)
Genera i fake StarGAN ESATTAMENTE sul sottoinsieme CelebA fissato da prepare_reals,
preservando l'allineamento: ogni output e' <source_id>__stargan.png, cosi'
build_manifest.py lo ricollega alla sua real (data/raw/real/celeba/<source_id>.png).

Usa il modello pre-addestrato del repo ufficiale yunjey/stargan (ne importa la
definizione del Generator: nessuna reimplementazione).

Setup (sul portatile, ambiente PyTorch con GPU se disponibile):
  git clone https://github.com/yunjey/stargan
  cd stargan && bash download.sh pretrained-celeba-128x128
  # i pesi finiscono in stargan/stargan_celeba_128/models/200000-G.ckpt

  python gen_stargan.py \\
      --repo /path/stargan \\
      --ckpt /path/stargan/stargan_celeba_128/models/200000-G.ckpt \\
      --src-dir data/raw/real/celeba \\
      --out data/raw/fake/stargan \\
      --native 128 --size 256 --transform blond

Da verificare contro il repo (raramente cambia): l'ordine di selected_attrs e il
c_dim del checkpoint. Default StarGAN-CelebA: c_dim=5,
attrs=[Black_Hair, Blond_Hair, Brown_Hair, Male, Young].
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image

# attributi del modello CelebA pre-addestrato (stesso ordine del repo)
ATTRS = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"]


def target_vector(c_dim: int, transform: str) -> torch.Tensor:
    # edit LOCALIZZATO (capelli) -> utile per la ground-truth di localizzazione (Fase 5)
    c = torch.zeros(1, c_dim)
    if transform == "blond":
        c[0, ATTRS.index("Blond_Hair")] = 1
    elif transform == "black":
        c[0, ATTRS.index("Black_Hair")] = 1
    elif transform == "brown":
        c[0, ATTRS.index("Brown_Hair")] = 1
    elif transform == "male":
        c[0, ATTRS.index("Male")] = 1
    else:
        raise ValueError(f"transform sconosciuto: {transform}")
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path al clone di yunjey/stargan")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--src-dir", default="data/raw/real/celeba")
    ap.add_argument("--out", default="data/raw/fake/stargan")
    ap.add_argument("--native", type=int, default=128, help="risoluzione di input del modello")
    ap.add_argument("--size", type=int, default=256, help="risoluzione di salvataggio")
    ap.add_argument("--c-dim", type=int, default=5)
    ap.add_argument("--transform", default="blond")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    from model import Generator  # definizione presa dal repo

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G = Generator(conv_dim=64, c_dim=args.c_dim, repeat_num=6).to(dev)
    G.load_state_dict(torch.load(args.ckpt, map_location=dev))
    # IMPORTANTE: NON usare G.eval(). Il Generator di StarGAN usa InstanceNorm con
    # track_running_stats=True; in eval() userebbe running stats inutilizzabili e
    # produce output corrotti (cast rosa + pattern a griglia, vedi issue #102 del repo).
    # In train mode l'InstanceNorm calcola le statistiche per-immagine -> output corretto.
    # E' deterministico (niente dropout/BatchNorm) e gira sotto torch.no_grad().
    G.train()

    # i crop canonici sono gia' quadrati -> resize alla risoluzione del modello + normalize
    pre = T.Compose([
        T.Resize((args.native, args.native)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    c = target_vector(args.c_dim, args.transform).to(dev)

    files = sorted(p for p in Path(args.src_dir).iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    with torch.no_grad():
        for p in files:
            x = pre(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
            y = G(x, c)                              # forward del repo: (immagine, label target)
            y = (y.clamp(-1, 1) + 1) / 2             # denormalizza in [0, 1]
            y = F.interpolate(y, size=args.size, mode="bilinear", align_corners=False)
            save_image(y, out / f"{p.stem}__stargan.png")
    print(f"StarGAN: {len(files)} fake -> {out}")


if __name__ == "__main__":
    main()