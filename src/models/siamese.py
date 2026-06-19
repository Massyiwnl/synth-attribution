"""
siamese.py - Fase 2/4
Encoder ResNet18 (pre-addestrato ImageNet) + testa di proiezione ->
embedding L2-normalizzato. La rete e' condivisa fra i due rami (pesi unici).

Front-end opzionale a residuo/high-pass (Fase 4): sopprime il contenuto ed
espone la traccia generativa PRIMA dell'encoder, per spingere il modello a
imparare la traccia invece della semantica (e, si spera, generalizzare meglio
a generatori mai visti). Output a 3 canali -> la conv1 di ResNet (3->64) resta
quella pre-addestrata, cosi' il confronto RGB-vs-residuo cambia solo il front-end.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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
    Un BatchNorm finale riallinea la scala del residuo a cio' che la conv1 attende.
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


class SiameseEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True,
                 embedding_dim: int = 128, front_end: str = "none"):
        super().__init__()
        if backbone != "resnet18":
            raise ValueError("Per ora e' supportato solo resnet18")
        self.front = (nn.Identity() if front_end in (None, "none")
                      else ResidualFrontEnd(front_end))
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        feat_dim = net.fc.in_features        # 512
        net.fc = nn.Identity()               # rimuovo il classificatore -> features grezze
        self.backbone = net
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, embedding_dim),
        )

    def forward_one(self, x):
        h = self.backbone(self.front(x))     # (B, 512)
        z = self.proj(h)                     # (B, embedding_dim)
        return F.normalize(z, p=2, dim=1)    # embedding su ipersfera unitaria

    def forward(self, x1, x2):
        return self.forward_one(x1), self.forward_one(x2)
