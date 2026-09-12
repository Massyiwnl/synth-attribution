"""
gradcam.py - Fase 8
Grad-CAM per un encoder siamese di METRIC LEARNING, su backbone CNN e su ViT/VLM.

Due complicazioni rispetto alla Grad-CAM classica, entrambe gestite qui.

1. NON C'E' UN LOGIT DI CLASSE.
   La rete non classifica: produce un embedding. Serve quindi uno scalare da
   derivare. Usiamo lo STESSO score della decisione di lineage:

       s(x) = || z(x) - c_altra || - || z(x) - c_propria ||

   cioe' "quanto questa immagine e' attratta dal centroide della sua lineage
   piuttosto che dall'altro". Il gradiente di s rispetto alle mappe di attivazione
   risponde a: quali regioni spingono l'immagine verso la SUA lineage. E' la
   domanda giusta per l'articolo, e a differenza di un logit generico ha un
   significato univoco.

2. IL TARGET LAYER E' DIVERSO FRA CNN E ViT.
   - CNN (ResNet): l'ultimo blocco convoluzionale, (B, C, H, W). Diretto.
   - ViT (CLIP / DINOv2): l'output di un blocco e' una sequenza di token
     (B, 1+N, D) (o (B, N, D) senza CLS). Va scartato il CLS, e gli N token di
     patch vanno rimessi in griglia sqrt(N) x sqrt(N). Inoltre sull'ULTIMO blocco
     le mappe ViT sono spesso degeneri: per convenzione si usa il penultimo, che
     e' quello che 'blocks[-2]' seleziona di default.

ATTENZIONE ALLA LETTURA. Con input_policy='crop' il backbone vede il crop centrale
(es. 224 su 256): la CAM copre quel crop, non l'immagine intera. Le facce di CelebA
e FFHQ sono allineate, quindi la media spaziale per classe e' interpretabile; ma
una CAM media che evidenzia semplicemente "la faccia" NON e' una prova di nulla:
va sempre confrontata con la CAM media GLOBALE (vedi eval_gradcam.py, che riporta
sia la mappa media sia il suo scarto dalla media globale).
"""
import numpy as np
import torch
import torch.nn.functional as F


class SiameseGradCAM:
    """Grad-CAM su un SiameseEncoder, con score = attrazione verso un centroide."""

    def __init__(self, model, target_layer=None, device="cuda"):
        self.model = model
        self.device = device
        self.layer = target_layer or self._auto_layer(model)
        self.acts = None
        self.grads = None
        self._h = [
            self.layer.register_forward_hook(self._save_act),
            self.layer.register_full_backward_hook(self._save_grad),
        ]

    # ---- selezione automatica del target layer ----
    @staticmethod
    def _auto_layer(model):
        bb = model.backbone
        if hasattr(bb, "layer4"):                 # ResNet
            return bb.layer4
        if hasattr(bb, "blocks"):                 # ViT timm (CLIP, DINOv2)
            # penultimo blocco: sull'ultimo le mappe ViT tendono a degenerare
            return bb.blocks[-2]
        raise ValueError("Impossibile dedurre il target layer: passalo esplicitamente")

    def _save_act(self, m, i, o):
        self.acts = o

    def _save_grad(self, m, gi, go):
        self.grads = go[0]

    def close(self):
        for h in self._h:
            h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # ---- da token ViT a griglia ----
    @staticmethod
    def _to_grid(t):
        """(B, T, D) -> (B, D, g, g) scartando i token non-patch; (B,C,H,W) invariato."""
        if t.dim() == 4:
            return t
        if t.dim() != 3:
            raise ValueError(f"forma di attivazione inattesa: {tuple(t.shape)}")
        B, T, D = t.shape
        g = int(round(T ** 0.5))
        if g * g != T:                      # ci sono token extra (CLS, registri)
            g = int(T ** 0.5)
            t = t[:, T - g * g:, :]         # tieni gli ULTIMI g*g: i patch token
        return t.transpose(1, 2).reshape(B, D, g, g)

    def __call__(self, x, c_own, c_other, out_size=None):
        """x: (B,3,H,W) gia' normalizzato. c_own/c_other: centroidi (D,).
        Ritorna (cam (B,h,w) in [0,1], score (B,))."""
        self.model.eval()
        x = x.to(self.device).requires_grad_(True)   # serve perche' il grafo venga
        c_own = torch.as_tensor(c_own, device=self.device, dtype=torch.float32)
        c_other = torch.as_tensor(c_other, device=self.device, dtype=torch.float32)

        z = self.model.forward_one(x)
        d_own = torch.norm(z - c_own[None], dim=1)
        d_other = torch.norm(z - c_other[None], dim=1)
        score = d_other - d_own                      # alto = attratto dalla sua lineage

        self.model.zero_grad(set_to_none=True)
        score.sum().backward()

        A = self._to_grid(self.acts)                 # (B, D, g, g)
        G = self._to_grid(self.grads)
        w = G.mean(dim=(2, 3), keepdim=True)         # global average pooling dei grad
        cam = F.relu((w * A).sum(1, keepdim=True))   # (B,1,g,g)
        if out_size:
            cam = F.interpolate(cam, size=out_size, mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)

        # normalizzazione per-immagine: rende le CAM mediabili fra loro
        B = cam.shape[0]
        flat = cam.reshape(B, -1)
        lo = flat.min(1).values[:, None, None]
        hi = flat.max(1).values[:, None, None]
        cam = (cam - lo) / (hi - lo + 1e-8)
        return cam.detach().cpu().numpy(), score.detach().cpu().numpy()


def map_similarity(a, b):
    """Correlazione di Pearson fra due mappe medie (in [-1,1])."""
    x, y = a.ravel() - a.mean(), b.ravel() - b.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denom) if denom > 0 else 0.0
