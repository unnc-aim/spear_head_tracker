"""
Backbone bbox tracker model.

The model returns a YOLO-style inference tensor:
    [batch, num_predictions, 5]

Each predicted box is:
    [x_center, y_center, width, height, object_conf]

Coordinates are normalized to 0..1. The tracker has no classification branch.
Internally, each final backbone feature-map cell predicts one anchor-free bbox.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torchvision.models import efficientnet_b0, mobilenet_v3_small, resnet18


@dataclass(frozen=True)
class TrackerModelConfig:
    """Configuration saved together with exported model weights."""

    img_size: Tuple[int, int] = (640, 640)
    num_boxes: int = 100
    width_mult: float = 1.0
    fpn_channels: int = 128
    backbone: str = "efficientnet"

    def to_dict(self) -> Dict[str, object]:
        return {
            "img_size": self.img_size,
            "num_boxes": self.num_boxes,
            "width_mult": self.width_mult,
            "fpn_channels": self.fpn_channels,
            "backbone": self.backbone,
        }


class ConvBNAct(nn.Module):
    """Conv2d + BatchNorm2d + SiLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class EfficientNetB0Backbone(nn.Module):
    """EfficientNet-B0 features backbone returning final feature map."""

    def __init__(self) -> None:
        super().__init__()
        self.features = efficientnet_b0(weights=None).features
        self.out_channels = 1280

    def forward(self, x: Tensor) -> Tensor:
        return self.features(x)


class ResNetBackbone(nn.Module):
    """ResNet-18 backbone returning layer4 feature map."""

    def __init__(self) -> None:
        super().__init__()
        model = resnet18(weights=None)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1)
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.out_channels = 512

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


class MobileNetBackbone(nn.Module):
    """MobileNetV3-Small features backbone returning final feature map."""

    def __init__(self) -> None:
        super().__init__()
        self.features = mobilenet_v3_small(weights=None).features
        self.out_channels = 576

    def forward(self, x: Tensor) -> Tensor:
        return self.features(x)


class DarknetBackbone(nn.Module):
    """Small Darknet-style CNN backbone returning final feature map."""

    def __init__(self, width_mult: float = 1.0) -> None:
        super().__init__()

        def channels(value: int) -> int:
            return max(8, int(value * width_mult))

        c1, c2, c3, c4, c5 = (
            channels(32),
            channels(64),
            channels(128),
            channels(256),
            channels(512),
        )
        self.out_channels = c5
        self.stem = nn.Sequential(
            ConvBNAct(3, c1, stride=2),
            ConvBNAct(c1, c2, stride=2),
            ConvBNAct(c2, c2),
        )
        self.stage3 = nn.Sequential(ConvBNAct(c2, c3, stride=2), ConvBNAct(c3, c3))
        self.stage4 = nn.Sequential(ConvBNAct(c3, c4, stride=2), ConvBNAct(c4, c4))
        self.stage5 = nn.Sequential(ConvBNAct(c4, c5, stride=2), ConvBNAct(c5, c5))

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.stage5(x)


def build_backbone(backbone: str, width_mult: float) -> nn.Module:
    if backbone == "efficientnet":
        return EfficientNetB0Backbone()
    if backbone == "resnet":
        return ResNetBackbone()
    if backbone == "mobilenet":
        return MobileNetBackbone()
    if backbone == "darknet":
        return DarknetBackbone(width_mult=width_mult)
    raise ValueError(f"Unsupported backbone: {backbone}")


class AnchorFreeBBoxHead(nn.Module):
    """Small anchor-free bbox/objectness head."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            ConvBNAct(channels, channels),
            ConvBNAct(channels, channels),
            nn.Conv2d(channels, 5, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)


class YOLOBBoxTracker(nn.Module):
    """
    Backbone anchor-free bbox tracker with YOLO-style bbox output.

    It predicts anchor-free dense bbox candidates from the final backbone feature map. There is no
    global pooling, no fixed fully-connected slot head, and no classification
    branch.
    """

    def __init__(
        self,
        num_boxes: int = 100,
        img_size: Tuple[int, int] = (640, 640),
        width_mult: float = 1.0,
        fpn_channels: int = 128,
        backbone: str = "efficientnet",
    ) -> None:
        super().__init__()
        if num_boxes <= 0:
            raise ValueError("num_boxes must be positive.")

        self.config = TrackerModelConfig(
            img_size=img_size,
            num_boxes=num_boxes,
            width_mult=width_mult,
            fpn_channels=fpn_channels,
            backbone=backbone,
        )
        self.num_boxes = num_boxes
        self.output_dim = 5

        self.backbone = build_backbone(backbone, width_mult)
        self.neck = ConvBNAct(self.backbone.out_channels, fpn_channels, kernel_size=1, padding=0)
        self.head = AnchorFreeBBoxHead(fpn_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Return bbox predictions.

        Output shape:
            [batch, num_predictions, 5]
        Output values:
            x_center, y_center, width, height, object_conf
        """
        feature = self.neck(self.backbone(x))
        return self._decode_level(self.head(feature))

    @staticmethod
    def _decode_level(raw: Tensor) -> Tensor:
        batch_size, _, height, width = raw.shape
        raw = raw.permute(0, 2, 3, 1).contiguous()

        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=raw.device, dtype=raw.dtype),
            torch.arange(width, device=raw.device, dtype=raw.dtype),
            indexing="ij",
        )
        x_center = (x_grid.unsqueeze(0) + torch.sigmoid(raw[..., 0])) / width
        y_center = (y_grid.unsqueeze(0) + torch.sigmoid(raw[..., 1])) / height
        box_width = torch.sigmoid(raw[..., 2])
        box_height = torch.sigmoid(raw[..., 3])
        object_conf = torch.sigmoid(raw[..., 4])

        decoded = torch.stack((x_center, y_center, box_width, box_height, object_conf), dim=-1)
        return decoded.view(batch_size, height * width, 5)

    @torch.no_grad()
    def predict_yolo_boxes(
        self,
        x: Tensor,
        conf_threshold: float = 0.25,
    ) -> list[Tensor]:
        """
        Convert raw predictions into per-image rows.

        Each returned row is:
            [x_center, y_center, width, height, confidence]
        """
        predictions = self.forward(x)
        results: list[Tensor] = []

        for image_predictions in predictions:
            bbox = image_predictions[:, 0:4]
            scores = image_predictions[:, 4]
            keep = scores >= conf_threshold

            if keep.any():
                rows = torch.cat((bbox[keep], scores[keep].unsqueeze(-1)), dim=-1)
            else:
                rows = image_predictions.new_zeros((0, 5))
            results.append(rows)

        return results


def build_tracker_model(
    num_boxes: int = 100,
    img_size: Tuple[int, int] = (640, 640),
    width_mult: float = 1.0,
    fpn_channels: int = 128,
    backbone: str = "efficientnet",
) -> YOLOBBoxTracker:
    """Factory used by training, inference, and export scripts."""
    return YOLOBBoxTracker(
        num_boxes=num_boxes,
        img_size=img_size,
        width_mult=width_mult,
        fpn_channels=fpn_channels,
        backbone=backbone,
    )
