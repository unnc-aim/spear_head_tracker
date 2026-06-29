"""
Export YOLOBBoxTracker to a .pt file.

Examples:
    python model_code/export_tracker_pt.py
    python model_code/export_tracker_pt.py --output runs/tracker_model.pt
    python model_code/export_tracker_pt.py --format torchscript --output runs/tracker_model_scripted.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tracker_model import build_tracker_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO-style bbox tracker.")
    parser.add_argument("--output", type=Path, default=Path("tracker_model.pt"))
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--num-boxes", type=int, default=100)
    parser.add_argument("--img-size", type=int, nargs=2, default=(640, 640), metavar=("H", "W"))
    parser.add_argument("--backbone", choices=("resnet", "darknet", "mobilenet", "efficientnet"), default="efficientnet")
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--fpn-channels", type=int, default=128)
    parser.add_argument(
        "--format",
        choices=("checkpoint", "torchscript"),
        default="checkpoint",
        help="checkpoint saves state_dict/config; torchscript saves a traced inference module.",
    )
    return parser.parse_args()


def load_weights_if_needed(model: torch.nn.Module, weights_path: Path | None) -> None:
    if weights_path is None:
        return

    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)


def export_checkpoint(model: torch.nn.Module, output_path: Path) -> None:
    torch.save(
        {
            "model_name": model.__class__.__name__,
            "model_state_dict": model.state_dict(),
            "config": model.config.to_dict(),
            "output_format": "[batch, num_predictions, 5]",
            "box_format": "normalized xywh + object_conf",
        },
        output_path,
    )


def export_torchscript(model: torch.nn.Module, output_path: Path, img_size: tuple[int, int]) -> None:
    model.eval()
    dummy = torch.zeros(1, 3, img_size[0], img_size[1])
    traced = torch.jit.trace(model, dummy)
    traced.save(str(output_path))


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    img_size = (args.img_size[0], args.img_size[1])
    model = build_tracker_model(
        num_boxes=args.num_boxes,
        img_size=img_size,
        backbone=args.backbone,
        width_mult=args.width_mult,
        fpn_channels=args.fpn_channels,
    )
    load_weights_if_needed(model, args.weights)

    if args.format == "checkpoint":
        export_checkpoint(model, args.output)
    else:
        export_torchscript(model, args.output, img_size)

    print(f"Exported {args.format} model to: {args.output}")


if __name__ == "__main__":
    main()
