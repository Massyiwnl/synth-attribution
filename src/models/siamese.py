"""
siamese.py - Fasi 2/4/8
Encoder siamese: backbone (ResNet allenabile o VLM congelato) + testa di
proiezione -> embedding L2-normalizzato. La rete e' condivisa fra i due rami.

Due regimi d'uso:
  - backbone allenabile (resnet18/resnet50): la baseline delle fasi precedenti,
    si allena tutto end-to-end;
  - backbone CONGELATO (clip_*, dinov2_*): si usa solo l'embedding finale del VLM
    e si allena SOLO la testa. Piu' veloce, molto meno soggetto a overfitting sul
    dataset, e permette di confrontare rappresentazioni pre-addestrate diverse a
    parita' di protocollo.

Front-end opzionale a residuo/high-pass (Fase 4): sopprime il contenuto ed espone
la traccia generativa PRIMA dell'encoder. Output a 3 canali, cosi' il primo livello
convoluzionale pre-addestrato resta utilizzabile e il confronto RGB-vs-residuo
cambia solo il front-end.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import build_backbone, freeze


def _gaussian_kernel(k: int = 5, sigma: float = 1.0):
    ax = torch.arange(k) - (k - 1) / 2.0
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k2 = torch.outer(g, g)
    return k2 / k2.sum()


# I 3 filtri SRM classici (high-pass per residui forensi), normalizzati.
_SRM = [
    torch.tensor([[0, 0, 0, 0, 0],
                  [0, -1, 2, -1, 0],
                  [0, 2, -4, 2, 0],
                  [0, -1, 2, -1, 0],
                  [0, 0, 0, 0, 0]], dtype=torch.float32) / 4.0,
    torch.tensor([[-1, 2, -2, 2, -1],
                  [2, -6, 8, -6, 2],
                  [-2, 8, -12, 8, -2],
                  [2, -6, 8, -6, 2],
                  [-1, 2, -2, 2, -1]], dtype=torch.float32) / 12.0,
    torch.tensor([[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, -2, 1, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]], dtype=torch.float32) / 2.0,
]


class ResidualFrontEnd(nn.Module):
    """Pre-filtro a residuo. Output sempre a 3 canali.
       mode 'highpass': x - gaussian_blur(x), per canale (mantiene il colore)
       mode 'srm'     : 3 filtri SRM sulla luminanza -> 3 mappe di residuo
    Un BatchNorm finale riallinea la scala del residuo a cio' che il backbone attende.
    I kernel sono FISSI (buffer, non addestrati); si addestra solo il BatchNorm."""

    def __init__(self, mode: str = "highpass", k: int = 5, sigma: float = 1.0):
        super().__init__()
        self.mode = mode
        if mode == "highpass":
            g = _gaussian_kernel(k, sigma)
            self.register_buffer("gauss", g[None, None].repeat(3, 1, 1, 1))  # (3,1,k,k)
            self.pad = k // 2
        elif mode == "srm":
            self.register_buffer("srm", torch.stack(_SRM)[:, None])          # (3,1,5,5)
            self.pad = 2
        else:
            raise ValueError(f"front_end mode sconosciuto: {mode}")
        self.bn = nn.BatchNorm2d(3)

    def forward(self, x):
        if self.mode == "highpass":
            low = F.conv2d(x, self.gauss, padding=self.pad, groups=3)
            r = x - low
        else:  # srm sulla luminanza
            gray = x.mean(dim=1, keepdim=True)
            r = F.conv2d(gray, self.srm, padding=self.pad)
        return self.bn(r)


class ProjectionHead(nn.Module):
    """Testa addestrabile sopra il backbone. Con hidden=feat_dim e senza dropout
    riproduce ESATTAMENTE la testa delle fasi 2/3, cosi' i checkpoint vecchi
    restano caricabili."""

    def __init__(self, feat_dim, embedding_dim=128, hidden=None, dropout=0.0):
        super().__init__()
        hidden = hidden or feat_dim
        layers = [nn.Linear(feat_dim, hidden), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, embedding_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SiameseEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True,
                 embedding_dim: int = 128, front_end: str = "none",
                 freeze_backbone: bool = False, image_size: int = 256,
                 head_hidden: int = None, head_dropout: float = 0.0,
                 input_policy: str = None):
        super().__init__()
        self.front = (nn.Identity() if front_end in (None, "none")
                      else ResidualFrontEnd(front_end))
        net, feat_dim, spec = build_backbone(backbone, pretrained, image_size,
                                             input_policy)
        self.backbone = net
        self.spec = spec
        self.feat_dim = feat_dim
        self.frozen = bool(freeze_backbone)
        if self.frozen:
            freeze(self.backbone)
        # 'proj' e' un Sequential anche nella vecchia versione: i checkpoint
        # delle fasi 2/3 hanno chiavi proj.0.* / proj.2.*, identiche a queste.
        self.proj = ProjectionHead(feat_dim, embedding_dim, head_hidden,
                                   head_dropout).net

    def train(self, mode: bool = True):
        """Un backbone congelato deve restare in eval() anche dentro model.train():
        altrimenti BatchNorm/dropout continuerebbero ad aggiornarsi e il backbone
        non sarebbe davvero fisso."""
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def features(self, x):
        """Embedding grezzo del backbone (pre-testa). Usato per il caching."""
        return self.backbone(self.front(x))

    def head(self, h):
        return F.normalize(self.proj(h), p=2, dim=1)

    def forward_one(self, x):
        return self.head(self.features(x))

    def forward(self, x1, x2):
        return self.forward_one(x1), self.forward_one(x2)


def load_encoder(checkpoint, device="cpu"):
    """Ricostruisce l'encoder dal checkpoint (che porta con se' la cfg completa)
    e restituisce (model, spec, epoch). Unico punto in cui i default dei campi
    nuovi vengono applicati ai checkpoint delle fasi precedenti."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    mc = ckpt["cfg"]["model"]
    dc = ckpt["cfg"].get("data", {})
    # se il modello e' stato allenato su PATCH, la risoluzione d'ingresso e' la
    # patch, non l'immagine: usare image_size qui farebbe ricampionare la patch
    # a 256 in valutazione, cioe' darebbe al modello un input con statistiche
    # completamente diverse da quelle viste in training.
    img = dc.get("patch_size") or dc.get("image_size", 256)
    model = SiameseEncoder(mc["backbone"], mc["pretrained"], mc["embedding_dim"],
                           front_end=mc.get("front_end", "none"),
                           freeze_backbone=mc.get("freeze_backbone", False),
                           image_size=img,
                           head_hidden=mc.get("head_hidden"),
                           head_dropout=mc.get("head_dropout", 0.0),
                           input_policy=mc.get("input_policy")).to(device)
    if ckpt.get("partial"):
        # checkpoint senza backbone (congelato): i pesi del backbone arrivano da
        # timm in build_backbone, qui si caricano solo front-end e testa
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        assert not unexpected, f"chiavi inattese nel checkpoint: {unexpected}"
        assert all(k.startswith("backbone.") for k in missing),             f"mancano pesi non-backbone: {[k for k in missing if not k.startswith('backbone.')]}"
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()
    return model, model.spec, ckpt.get("epoch")
