"""
siamese_dataset.py - Fasi 1/2/3/8
Dataset PyTorch dal manifest, con coppie per la rete siamese.

Etichette (coerenti con la contrastive loss):
    y = 1  -> coppia GENUINE   -> distanza piccola
    y = 0  -> coppia IMPOSTORE  -> distanza >= margine

Politiche di pairing:
    "architecture": genuine = stessa classe generativa; impostore = classe diversa.
                    Classe = architecture per i fake; "real_<source>" per i reali
                    (real_celeba e real_ffhq sono classi distinte).
                    -> attribuzione del GENERATORE (fasi 2/3).
    "lineage":      genuine = stesso dataset di addestramento (celeba/ffhq),
                    indipendentemente dal generatore; impostore = lineage diversa.
                    -> attribuzione dei DATI DI ADDESTRAMENTO (fase 8). E' la
                    politica che raggruppa deliberatamente real_celeba con
                    StarGAN/AttGAN/GDWCT, e real_ffhq con StyleGAN2/3.
    "same_source":  alias storico di "lineage".
    "aligned":      come lineage, ma per i fake di editing-GAN la partner genuine
                    preferita e' la real sorgente allineata pixel-a-pixel.

lineage_filter: se 'celeba' o 'ffhq', restringe il dataset a quella lineage
                (test pulito within-lineage, senza confound di dataset).
                Per il task di lineage va lasciato a None: servono entrambe.

__getitem__ restituisce anche il codice di lineage della coppia, per separare le
metriche in valutazione:  0 = within-celeba, 1 = within-ffhq, -1 = cross-lineage.

Coppie DETERMINISTICHE per indice (seed da i): val riproducibile, nessuna
correlazione fra worker del DataLoader.
"""
import io
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

from src.models.backbones import BackboneSpec, IMAGENET_MEAN, IMAGENET_STD

LINEAGE_CODE = {"celeba": 0, "ffhq": 1}
LINEAGE_POLICIES = {"lineage", "same_source", "aligned"}

_INTERP = {"bicubic": T.InterpolationMode.BICUBIC,
           "bilinear": T.InterpolationMode.BILINEAR,
           "lanczos": T.InterpolationMode.LANCZOS}


def as_spec(spec_or_size, image_size: int = 256) -> BackboneSpec:
    """Accetta una BackboneSpec oppure un int (compatibilita' con le fasi 2/3,
    dove il transform era cablato a ResNet/ImageNet)."""
    if isinstance(spec_or_size, BackboneSpec):
        return spec_or_size
    return BackboneSpec(input_size=int(spec_or_size or image_size),
                        mean=IMAGENET_MEAN, std=IMAGENET_STD,
                        interpolation="bilinear", input_policy="resize")


class ForensicJitter:
    """Augmentation MIRATA al confound di ricampionamento: forza la rete a non
    poter distinguere le classi in base alla catena di resize/compressione.
    Applicata a tutte le classi con la stessa distribuzione, quindi non puo'
    introdurre un nuovo segnale di classe: puo' solo DISTRUGGERE quello spurio.
    Va usata in training; in valutazione si usa il dataset normalizzato offline."""

    def __init__(self, p_resample=0.5, p_jpeg=0.5, sizes=(128, 160, 192, 224),
                 quality=(75, 95), seed=0):
        self.p_resample, self.p_jpeg = p_resample, p_jpeg
        self.sizes, self.quality = sizes, quality
        self.seed = seed

    def __call__(self, img: Image.Image, idx: int) -> Image.Image:
        rng = random.Random(self.seed * 7_654_321 + idx)
        w, h = img.size
        if rng.random() < self.p_resample:
            s = rng.choice(self.sizes)
            img = img.resize((s, s), Image.LANCZOS).resize((w, h), Image.LANCZOS)
        if rng.random() < self.p_jpeg:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=rng.randint(*self.quality))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img


class PatchExtractor:
    """Estrae una patch a RISOLUZIONE NATIVA (nessun resize, nessuna
    interpolazione: il ricampionamento e' proprio cio' che non vogliamo aggiungere).

    Perche' le patch. Sull'immagine intera l'attribuzione di lineage e'
    sovra-determinata: il CONTENUTO (inquadratura, allineamento, demografia,
    illuminazione di CelebA vs FFHQ) basta gia' da solo - un CLIP zero-shot, senza
    alcun addestramento, assegna StyleGAN3 a FFHQ nel 98% dei casi. Finche' la
    spiegazione percettiva funziona, non si puo' sostenere che l'attribuzione sia
    forense. Una patch 64x64 rimuove quasi tutta quella semantica ma conserva la
    statistica di basso livello (texture, rumore, tracce del generatore).

    policy:
      'flat'   -> fra `candidates` posizioni casuali sceglie quella a VARIANZA
                  MINIMA, cioe' la regione piu' uniforme (guancia, fondo, fronte).
                  E' l'accorgimento classico in forense: le zone piatte portano la
                  traccia del generatore senza portare contenuto. ATTENZIONE: la
                  scelta DIPENDE DAL CONTENUTO, quindi in linea di principio la
                  posizione selezionata potrebbe correlare con la classe.
      'random' -> posizione casuale, indipendente dal contenuto.
      'grid'   -> posizioni FISSE su griglia regolare, IDENTICHE per ogni immagine:
                  i centri stanno a (i+0.5)/m di larghezza e altezza, con
                  m = ceil(sqrt(patches_per_image)). E' il controllo piu' stretto:
                  nessuna differenza fra classi puo' derivare da DOVE si e'
                  guardato, perche' si guarda sempre negli stessi punti. Con k=4
                  i centri cadono al 25% e 75% dei due assi (guance, fronte, mento
                  in modo simmetrico).
    """

    def __init__(self, size=64, policy="flat", candidates=16, seed=0,
                 patches_per_image=1):
        self.size = size
        self.policy = policy
        self.candidates = candidates
        self.seed = seed
        # usato solo da 'grid': determina la disposizione dei centri
        self.m = max(1, int(np.ceil(np.sqrt(max(1, patches_per_image)))))

    def __call__(self, img: Image.Image, idx: int, patch_idx: int = 0) -> Image.Image:
        S = self.size
        W, H = img.size
        if W < S or H < S:                       # immagine piu' piccola della patch
            img = img.resize((max(W, S), max(H, S)), Image.LANCZOS)
            W, H = img.size

        if self.policy == "grid":
            m = self.m
            row, col = divmod(int(patch_idx) % (m * m), m)
            cx = (col + 0.5) / m * W
            cy = (row + 0.5) / m * H
            x = int(min(max(round(cx - S / 2), 0), W - S))
            y = int(min(max(round(cy - S / 2), 0), H - S))
            return img.crop((x, y, x + S, y + S))

        rng = random.Random(self.seed * 2_654_435_761 + idx)
        if self.policy == "random":
            x, y = rng.randrange(W - S + 1), rng.randrange(H - S + 1)
            return img.crop((x, y, x + S, y + S))

        g = np.asarray(img.convert("L"), dtype=np.float32)
        best, best_var = None, None
        for _ in range(self.candidates):
            x, y = rng.randrange(W - S + 1), rng.randrange(H - S + 1)
            v = float(g[y:y + S, x:x + S].var())
            if best_var is None or v < best_var:
                best, best_var = (x, y), v
        x, y = best
        return img.crop((x, y, x + S, y + S))


def expand_patches(df, k):
    """Ripete ogni riga k volte, aggiungendo patch_idx: cosi' ogni patch e' un
    campione a se' e tutto il codice di etichettatura a valle resta invariato."""
    if k is None or k <= 1:
        out = df.copy()
        out["patch_idx"] = 0
        return out
    out = df.loc[df.index.repeat(k)].reset_index(drop=True)
    out["patch_idx"] = np.tile(np.arange(k), len(df))
    return out


def build_transform(spec_or_size, train: bool = True, image_size: int = 256):
    """Transform derivato dal backbone.

    input_policy='crop' -> crop centrale (NESSUN ricampionamento: preserva le
    statistiche dei pixel, importante perche' il resize e' esattamente
    l'artefatto che non vogliamo misurare).
    input_policy='resize' -> resize classico.
    Augmentation: solo flip orizzontale, che non altera lo spettro di potenza.
    """
    spec = as_spec(spec_or_size, image_size)
    ops = []
    if spec.input_policy == "crop":
        # porta almeno a input_size solo se l'immagine fosse piu' piccola
        ops.append(T.Resize(spec.input_size,
                            interpolation=_INTERP.get(spec.interpolation,
                                                      T.InterpolationMode.BICUBIC)))
        ops.append(T.CenterCrop(spec.input_size))
    else:
        ops.append(T.Resize((spec.input_size, spec.input_size),
                            interpolation=_INTERP.get(spec.interpolation,
                                                      T.InterpolationMode.BILINEAR)))
    if train:
        ops.append(T.RandomHorizontalFlip(0.5))
    ops += [T.ToTensor(), T.Normalize(spec.mean, spec.std)]
    return T.Compose(ops)


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def read_manifest(manifest: str, split: str, lineage_filter=None, only_fake=None,
                  drop_architectures=None) -> pd.DataFrame:
    df = pd.read_csv(manifest, dtype={"source_id": str}, keep_default_na=False)
    df = df[df.split == split]
    if lineage_filter:
        df = df[df.source_dataset == lineage_filter]
    if only_fake:
        # tiene solo reali + UN generatore (per il leave-one-generator-out):
        # i reali fanno da esempio negativo, il generatore da positivo.
        df = df[(df.label == "real") | (df.architecture == only_fake)]
    if drop_architectures:
        df = df[~df.architecture.isin(set(drop_architectures))]
    return df.reset_index(drop=True)


def class_of(label, architecture, source_dataset):
    """Classe generativa: architecture per i fake, real_<source> per i reali."""
    return architecture if label == "fake" else f"real_{source_dataset}"


def labels_of(df, mode="class"):
    """Etichette per la valutazione.
      'class'        -> real_celeba / stargan / stylegan2 ...  (attribuzione generatore)
      'lineage'      -> celeba / ffhq                          (attribuzione dati training)
      'architecture' -> real / stargan / stylegan2 ...         (compat fasi 2/3)
    """
    import numpy as np
    if mode == "lineage":
        return df.source_dataset.to_numpy()
    if mode == "architecture":
        return df.architecture.to_numpy()
    return np.array([class_of(r.label, r.architecture, r.source_dataset)
                     for r in df.itertuples()])


class SingleImageDataset(Dataset):
    """Per estrarre embedding in valutazione (Fasi 3/8)."""

    def __init__(self, manifest, split, spec, lineage_filter=None, only_fake=None,
                 drop_architectures=None, image_size=256, patch_size=None,
                 patches_per_image=1, patch_policy="flat", seed=0):
        df = read_manifest(manifest, split, lineage_filter, only_fake,
                           drop_architectures)
        self.patcher = (PatchExtractor(patch_size, patch_policy, seed=seed,
                                       patches_per_image=patches_per_image)
                        if patch_size else None)
        # con le patch ogni riga diventa `patches_per_image` campioni distinti
        self.df = expand_patches(df, patches_per_image if self.patcher else 1)
        self.tf = build_transform(spec, train=False, image_size=image_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = load_image(r.path)
        if self.patcher is not None:
            img = self.patcher(img, i, int(r.patch_idx))
        return self.tf(img), i


class SiamesePairDataset(Dataset):
    def __init__(self, manifest, split, spec, policy="architecture",
                 genuine_prob=0.5, seed=42, lineage_filter=None, only_fake=None,
                 drop_architectures=None, image_size=256, forensic_aug=False,
                 patch_size=None, patches_per_image=1, patch_policy="flat"):
        df = read_manifest(manifest, split, lineage_filter, only_fake,
                           drop_architectures)
        self.patcher = (PatchExtractor(patch_size, patch_policy, seed=seed,
                                       patches_per_image=patches_per_image)
                        if patch_size else None)
        self.df = expand_patches(df, patches_per_image if self.patcher else 1)
        self.train = (split == "train")
        self.tf = build_transform(spec, train=self.train, image_size=image_size)
        self.policy = policy
        self.genuine_prob = genuine_prob
        self.seed = seed
        self.jitter = (ForensicJitter(seed=seed)
                       if (forensic_aug and self.train) else None)

        self.groups = {}
        for idx, row in self.df.iterrows():
            self.groups.setdefault(self._group_key(row), []).append(idx)
        reals = self.df[self.df.label == "real"]
        self.real_by_id = {f"{r.source_dataset}/{r.source_id}": idx
                           for idx, r in reals.iterrows()}

    def _group_key(self, r):
        if self.policy == "architecture":
            return class_of(r.label, r.architecture, r.source_dataset)
        return r.source_dataset            # lineage / same_source / aligned

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

    def _prep(self, path, idx, patch_idx=0):
        img = load_image(path)
        # ordine: il jitter simula resize/compressione sull'immagine INTERA
        # (com'e' nella realta'), la patch si ritaglia dopo
        if self.jitter is not None:
            img = self.jitter(img, idx)
        if self.patcher is not None:
            img = self.patcher(img, idx, patch_idx)
        return self.tf(img)

    def __getitem__(self, i):
        rng = random.Random(self.seed * 1_000_003 + i)
        r = self.df.iloc[i]
        if rng.random() < self.genuine_prob:
            j, y = self._genuine_partner(i, r, rng), 1.0
        else:
            j, y = self._impostor_partner(i, r, rng), 0.0
        rp = self.df.iloc[j]
        lc = LINEAGE_CODE.get(r.source_dataset, -2) if r.source_dataset == rp.source_dataset else -1
        a = self._prep(r.path, i, int(getattr(r, 'patch_idx', 0)))
        b = self._prep(rp.path, j, int(getattr(rp, 'patch_idx', 0)))
        return a, b, torch.tensor(y, dtype=torch.float32), torch.tensor(lc, dtype=torch.long)
