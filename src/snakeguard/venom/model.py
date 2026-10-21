"""VenomNet: ImageNet-pretrained backbone with a binary venom head and an auxiliary species head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class VenomNet(nn.Module):
    """forward(x) -> (venom_logit (B,), species_logits (B, S)).
    `features` is the last conv block (the natural CAM target layer for B3)."""

    def __init__(self, n_species: int, backbone: str = "efficientnet_b0", pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        if backbone != "efficientnet_b0":
            raise ValueError(f"unsupported backbone {backbone}")
        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.efficientnet_b0(weights=weights)
        self.backbone_name = backbone
        self.features = net.features          # (B, 1280, h, w)
        self.pool = nn.AdaptiveAvgPool2d(1)
        dim = net.classifier[1].in_features
        self.dropout = nn.Dropout(dropout)
        self.venom_head = nn.Linear(dim, 1)
        self.species_head = nn.Linear(dim, n_species)
        self.n_species = n_species

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.dropout(self.embed(x))
        return self.venom_head(z)[:, 0], self.species_head(z)
