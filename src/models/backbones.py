"""
backbones.py - Fase 8 (attribuzione di lineage)
Registry dei backbone per l'encoder siamese, cosi' che lo STESSO protocollo possa
girare su ResNet (allenata end-to-end, come nelle fasi precedenti) e su un VLM
pre-addestrato CONGELATO (CLIP / DINOv2), di cui si usa solo l'embedding finale.

Ogni backbone dichiara la propria BackboneSpec (risoluzione attesa, normalizzazione,
interpolazione): il transform NON e' piu' cablato nel dataset, lo decide il backbone.

Nota forense sulla risoluzione (input_policy):
    i VLM vogliono 224, il nostro dataset e' a 256. Fare un resize 256->224
    introduce un ULTERIORE ricampionamento, cioe' proprio il tipo di artefatto che
    stiamo cercando di NON misurare (vedi il confound di Sez. 9 della relazione).
    Per questo il default per i backbone congelati e' 'crop': un crop centrale
    224x224 preserva esattamente le statistiche dei pixel originali, senza
    reinterpolare nulla. 'resize' resta disponibile per l'ablazione.
"""
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class BackboneSpec:
    """Come vanno preparate le immagini per questo backbone."""
    input_size: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    interpolation: str = "bicubic"
    input_policy: str = "crop"      # crop | resize  (vedi nota forense sopra)


# backbone torchvision allenabili end-to-end (la baseline delle fasi precedenti)
_TORCHVISION = {
    "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2),
}

# backbone timm: VLM/self-supervised, pensati per essere usati CONGELATI
_TIMM = {
    "clip_vit_b16": "vit_base_patch16_clip_224.openai",
    "clip_vit_l14": "vit_large_patch14_clip_224.openai",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
}

AVAILABLE = sorted(list(_TORCHVISION) + list(_TIMM))


def is_timm(name: str) -> bool:
    return name in _TIMM


def build_backbone(name: str, pretrained: bool = True, image_size: int = 256,
                   input_policy: str = None):
    """Restituisce (modulo, feat_dim, spec). Il modulo emette un vettore (B, feat_dim)."""
    if name in _TORCHVISION:
        ctor, weights = _TORCHVISION[name]
        net = ctor(weights=weights if pretrained else None)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        # le ResNet sono completamente convoluzionali + pooling adattivo:
        # accettano 256 nativo, nessun ricampionamento necessario
        spec = BackboneSpec(input_size=image_size, mean=IMAGENET_MEAN,
                            std=IMAGENET_STD, interpolation="bilinear",
                            input_policy=input_policy or "resize")
        return net, feat_dim, spec

    if name in _TIMM:
        import timm
        net = timm.create_model(_TIMM[name], pretrained=pretrained, num_classes=0)
        cfg = net.pretrained_cfg
        want = int(cfg["input_size"][-1])
        if name.startswith("dinov2"):
            # il default DINOv2 e' 518 (troppo costoso e comunque un upscale dal
            # nostro 256): lo riportiamo a 224, che e' multiplo di patch 14
            net = timm.create_model(_TIMM[name], pretrained=pretrained,
                                    num_classes=0, img_size=224)
            want = 224
        # se il backbone vuole meno pixel di quelli che abbiamo, croppiamo
        policy = input_policy or ("crop" if want <= image_size else "resize")
        spec = BackboneSpec(input_size=want, mean=tuple(cfg["mean"]),
                            std=tuple(cfg["std"]),
                            interpolation=cfg.get("interpolation", "bicubic"),
                            input_policy=policy)
        return net, net.num_features, spec

    raise ValueError(f"backbone sconosciuto: {name!r}. Disponibili: {AVAILABLE}")


def freeze(module: nn.Module) -> nn.Module:
    """Congela i pesi e mette in eval: il backbone diventa un estrattore fisso.
    eval() e' essenziale per BatchNorm/dropout: senza, le statistiche si
    aggiornerebbero comunque durante il training e il backbone non sarebbe fisso."""
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()
    return module
