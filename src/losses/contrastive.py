"""
contrastive.py - Fase 2
Contrastive loss (Hadsell, Chopra & LeCun). Convenzione:
  y = 1 -> coppia genuine (stessa origine)  -> distanza piccola
  y = 0 -> coppia impostore                 -> distanza >= margine

  L = y * d^2 + (1 - y) * max(0, margin - d)^2

con d = distanza euclidea fra embedding L2-normalizzati (quindi d in [0, 2]).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z1, z2, y):
        d = F.pairwise_distance(z1, z2, p=2)            # (B,)
        loss_genuine = y * d.pow(2)
        loss_impostor = (1.0 - y) * F.relu(self.margin - d).pow(2)
        return (loss_genuine + loss_impostor).mean()
