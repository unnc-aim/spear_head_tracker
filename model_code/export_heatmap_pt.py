"""Export HeatmapTrackerModel to TorchScript heatmap_model.pt."""

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
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        ensure_confidence_state_dict(state_dict, args.weights)
        model.load_state_dict(state_dict)

    model.eval()
    example_input = torch.zeros(1, 3, args.img_size[0], args.img_size[1])
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(str(args.output))
    print(f"Exported TorchScript heatmap model to: {args.output}")


def ensure_confidence_state_dict(state_dict: dict[str, torch.Tensor], checkpoint_path: Path) -> None:
    head_weight = state_dict.get("head.2.weight")
    if isinstance(head_weight, torch.Tensor) and head_weight.shape[0] == 2:
        return
    raise ValueError(
        f"Incompatible heatmap checkpoint: {checkpoint_path}. "
        "Expected a 2-channel confidence-head checkpoint. "
        "Please retrain the heatmap model before exporting TorchScript."
    )


if __name__ == "__main__":
    main("runs/heatmap_train/best.pth")
