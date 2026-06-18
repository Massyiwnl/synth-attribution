"""
siamese_dataset.py - Fasi 1/2
Dataset PyTorch a partire dal manifest, con costruzione di coppie per la rete siamese.

Convenzione etichette (coerente con la contrastive loss della Fase 2):
    y = 1  -> coppia GENUINE   (stessa origine)   -> distanza piccola desiderata
    y = 0  -> coppia IMPOSTORE  (origine diversa)  -> distanza >= margine

Politiche di pairing:
    "same_source": genuine se condividono source_dataset (celeba/ffhq); impostore altrimenti.
    "aligned":     come sopra, ma se l'ANCORA e' un fake di editing-GAN, la partner
                   genuine preferita e' la sua REAL sorgente (stesso source_id) ->
                   coppia allineata pixel-a-pixel (base per residuo e localizzazione).
"""
import random

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(size: int, train: bool = True):
    # Augmentation FORENSIC-AWARE: niente color jitter / blur / jpeg, distruggono
    # la traccia ad alta frequenza. Il flip orizzontale non altera lo spettro.
    ops = [T.Resize((size, size))]
    if train:
        ops.append(T.RandomHorizontalFlip(0.5))
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


class SingleImageDataset(Dataset):
    """Per estrarre embedding in fase di valutazione (Fase 3)."""

    def __init__(self, manifest: str, split: str, size: int):
        df = pd.read_csv(manifest)
        self.df = df[df.split == split].reset_index(drop=True)
        self.tf = build_transform(size, train=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.tf(load_image(r.path)), i


class SiamesePairDataset(Dataset):
    def __init__(self, manifest: str, split: str, size: int,
                 policy: str = "aligned", genuine_prob: float = 0.5, seed: int = 42):
        df = pd.read_csv(manifest)
        self.df = df[df.split == split].reset_index(drop=True)
        self.tf = build_transform(size, train=(split == "train"))
        self.policy = policy
        self.genuine_prob = genuine_prob
        self.rng = random.Random(seed)

        # indici raggruppati per dataset di origine
        self.by_source = {
            s: self.df.index[self.df.source_dataset == s].tolist()
            for s in self.df.source_dataset.dropna().unique() if s != ""
        }
        # mappa "source_dataset/source_id" -> indice della REAL corrispondente
        reals = self.df[self.df.label == "real"]
        self.real_by_id = {
            f"{row.source_dataset}/{row.source_id}": idx
            for idx, row in reals.iterrows()
        }

    def __len__(self):
        return len(self.df)

    def _genuine_partner(self, i, r):
        # coppia allineata: fake editing-GAN <-> sua real sorgente
        if (self.policy == "aligned" and r.label == "fake"
                and isinstance(r.source_id, str) and r.source_id):
            j = self.real_by_id.get(f"{r.source_dataset}/{r.source_id}")
            if j is not None and j != i:
                return j
        # fallback: stessa origine
        pool = [k for k in self.by_source.get(r.source_dataset, []) if k != i]
        return self.rng.choice(pool) if pool else i

    def _impostor_partner(self, i, r):
        others = [s for s in self.by_source if s != r.source_dataset]
        if not others:
            return i
        s = self.rng.choice(others)
        return self.rng.choice(self.by_source[s])

    def __getitem__(self, i):
        r = self.df.iloc[i]
        if self.rng.random() < self.genuine_prob:
            j, y = self._genuine_partner(i, r), 1.0
        else:
            j, y = self._impostor_partner(i, r), 0.0
        a = self.tf(load_image(r.path))
        b = self.tf(load_image(self.df.iloc[j].path))
        return a, b, torch.tensor(y, dtype=torch.float32)
