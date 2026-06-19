"""
siamese_dataset.py - Fasi 1/2/3
Dataset PyTorch dal manifest, con coppie per la rete siamese.

Etichette (coerenti con la contrastive loss):
    y = 1  -> coppia GENUINE   -> distanza piccola
    y = 0  -> coppia IMPOSTORE  -> distanza >= margine

Politiche di pairing:
    "architecture": genuine = stessa classe generativa; impostore = classe diversa.
                    Classe = architecture per i fake; "real_<source>" per i reali
                    (real_celeba e real_ffhq sono classi distinte).
    "same_source":  genuine = stesso source_dataset (celeba/ffhq).
    "aligned":      come same_source, ma per i fake di editing-GAN la partner genuine
                    preferita e' la real sorgente allineata pixel-a-pixel.

lineage_filter: se 'celeba' o 'ffhq', restringe il dataset a quella lineage
                (test pulito within-lineage, senza confound di dataset).

__getitem__ restituisce anche il codice di lineage della coppia, per separare le
metriche in valutazione:  0 = within-celeba, 1 = within-ffhq, -1 = cross-lineage.

Coppie DETERMINISTICHE per indice (seed da i): val riproducibile, nessuna
correlazione fra worker del DataLoader.
"""
import random

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LINEAGE_CODE = {"celeba": 0, "ffhq": 1}


def build_transform(size: int, train: bool = True):
    # Augmentation forensic-aware: solo flip orizzontale (non altera lo spettro).
    ops = [T.Resize((size, size))]
    if train:
        ops.append(T.RandomHorizontalFlip(0.5))
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def read_manifest(manifest: str, split: str, lineage_filter=None) -> pd.DataFrame:
    df = pd.read_csv(manifest, dtype={"source_id": str}, keep_default_na=False)
    df = df[df.split == split]
    if lineage_filter:
        df = df[df.source_dataset == lineage_filter]
    return df.reset_index(drop=True)


def class_of(label, architecture, source_dataset):
    """Classe generativa: architecture per i fake, real_<source> per i reali."""
    return architecture if label == "fake" else f"real_{source_dataset}"


class SingleImageDataset(Dataset):
    """Per estrarre embedding in valutazione (Fase 3)."""

    def __init__(self, manifest, split, size, lineage_filter=None):
        self.df = read_manifest(manifest, split, lineage_filter)
        self.tf = build_transform(size, train=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.tf(load_image(r.path)), i


class SiamesePairDataset(Dataset):
    def __init__(self, manifest, split, size, policy="architecture",
                 genuine_prob=0.5, seed=42, lineage_filter=None):
        self.df = read_manifest(manifest, split, lineage_filter)
        self.tf = build_transform(size, train=(split == "train"))
        self.policy = policy
        self.genuine_prob = genuine_prob
        self.seed = seed

        self.groups = {}
        for idx, row in self.df.iterrows():
            self.groups.setdefault(self._group_key(row), []).append(idx)
        reals = self.df[self.df.label == "real"]
        self.real_by_id = {f"{r.source_dataset}/{r.source_id}": idx
                           for idx, r in reals.iterrows()}

    def _group_key(self, r):
        if self.policy == "architecture":
            return class_of(r.label, r.architecture, r.source_dataset)
        return r.source_dataset            # same_source / aligned

    def __len__(self):
        return len(self.df)

    def _genuine_partner(self, i, r, rng):
        if self.policy == "aligned" and r.label == "fake" and r.source_id:
            j = self.real_by_id.get(f"{r.source_dataset}/{r.source_id}")
            if j is not None and j != i:
                return j
        pool = [k for k in self.groups.get(self._group_key(r), []) if k != i]
        return rng.choice(pool) if pool else i

    def _impostor_partner(self, i, r, rng):
        key = self._group_key(r)
        others = [g for g in self.groups if g != key]
        if not others:
            return i
        return rng.choice(self.groups[rng.choice(others)])

    def __getitem__(self, i):
        rng = random.Random(self.seed * 1_000_003 + i)
        r = self.df.iloc[i]
        if rng.random() < self.genuine_prob:
            j, y = self._genuine_partner(i, r, rng), 1.0
        else:
            j, y = self._impostor_partner(i, r, rng), 0.0
        rp = self.df.iloc[j]
        lc = LINEAGE_CODE.get(r.source_dataset, -2) if r.source_dataset == rp.source_dataset else -1
        a = self.tf(load_image(r.path))
        b = self.tf(load_image(rp.path))
        return a, b, torch.tensor(y, dtype=torch.float32), torch.tensor(lc, dtype=torch.long)
