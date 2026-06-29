"""Export HeatmapTrackerModel to heatmap_model.pt."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from heatmap_model import build_heatmap_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export heatmap model.")
    parser.add_argument("--output", type=Path, default=Path("heatmap_model.pt"))
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--img-size", type=int, nargs=2, default=(384,384), metavar=("H", "W"))
    parser.add_argument("--backbone", choices=("resnet", "darknet", "mobilenet", "efficientnet"), default="efficientnet")
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--head-channels", type=int, default=128)
    return parser.parse_args()


def main(path=None) -> None:
    args = parse_args()
    if path is not None:
        args.weights=path
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = build_heatmap_model(
        img_size=(args.img_size[0], args.img_size[1]),
        backbone=args.backbone,
        width_mult=args.width_mult,
        head_channels=args.head_channels,
    )
    if args.weights is not None:
        checkpoint = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))

    torch.save(
        {
            "model_name": model.__class__.__name__,
            "model_state_dict": model.state_dict(),
            "config": model.config.to_dict(),
            "output_format": "[batch, 1, height, width] logits",
        },
        args.output,
    )
    print(f"Exported heatmap model to: {args.output}")


if __name__ == "__main__":
    main("runs/heatmap_train/best.pth")
