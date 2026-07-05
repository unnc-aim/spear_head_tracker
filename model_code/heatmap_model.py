"""
Heatmap model for minimum bbox extraction.

The backbone choices match tracker_model.py, while the prediction head outputs
heatmap and confidence logit maps resized to the input image size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tracker_model import ConvBNAct, build_backbone


@dataclass(frozen=True)
class HeatmapModelConfig:
    img_size: Tuple[int, int] = (640, 640)
    backbone: str = "efficientnet"
    width_mult: float = 1.0
    head_channels: int = 128

    def to_dict(self) -> Dict[str, object]:
        return {
            "img_size": self.img_size,
            "backbone": self.backbone,
            "width_mult": self.width_mult,
            "head_channels": self.head_channels,
        }


class HeatmapTrackerModel(nn.Module):
    """Backbone + heatmap segmentation head."""

    def __init__(
        self,
        img_size: Tuple[int, int] = (640, 640),
        backbone: str = "efficientnet",
        width_mult: float = 1.0,
        head_channels: int = 128,
    ) -> None:
        super().__init__()
        self.config = HeatmapModelConfig(
            img_size=img_size,
            backbone=backbone,
            width_mult=width_mult,
            head_channels=head_channels,
        )
        self.backbone = build_backbone(backbone, width_mult)
        self.head = nn.Sequential(
            ConvBNAct(self.backbone.out_channels, head_channels, kernel_size=1, padding=0),
            ConvBNAct(head_channels, head_channels),
            nn.Conv2d(head_channels, 2, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        logits = self.head(self.backbone(x))
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

    @torch.no_grad()
    def predict_heatmap(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.forward(x)[:, 0:1])

    @torch.no_grad()
    def predict_heatmap_and_confidence(self, x: Tensor) -> tuple[Tensor, Tensor]:
        probabilities = torch.sigmoid(self.forward(x))
        return probabilities[:, 0:1], probabilities[:, 1:2]


def build_heatmap_model(
    img_size: Tuple[int, int] = (640, 640),
    backbone: str = "efficientnet",
    width_mult: float = 1.0,
    head_channels: int = 128,
) -> HeatmapTrackerModel:
    return HeatmapTrackerModel(
        img_size=img_size,
        backbone=backbone,
        width_mult=width_mult,
        head_channels=head_channels,
    )
