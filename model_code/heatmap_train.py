"""
Train heatmap tracker model.

The model predicts a binary heatmap. During inference, a minimum bbox can be
computed from positive heatmap pixels.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from heatmap_dataset import (
    HeatmapDataset,
    create_heatmap_dataset,
    heatmap_collate_fn,
    read_yolo_rows,
    yolo_box_to_pixels,
)
from heatmap_model import build_heatmap_model
from tracker_dataset import labels_dir_from_images_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train heatmap tracker.")
    parser.add_argument("--data", type=Path, default=Path("dataset_6/data.yaml"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/heatmap_train"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--img-size", type=int, nargs=2, default=(640, 640), metavar=("H", "W"))
    parser.add_argument("--backbone", choices=("resnet", "darknet", "mobilenet", "efficientnet"), default="resnet")
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--head-channels", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gray-threshold", type=int, default=100)
    parser.add_argument("--gaussian-sigma", type=float, default=1.0)
    parser.add_argument("--max-fill-distance", type=float, default=24.0)
    parser.add_argument("--center-prior-sigma", type=float, default=0.3)
    parser.add_argument("--pos-weight", type=float, default=5.0)
    parser.add_argument("--conf-loss-weight", type=float, default=1.0)
    parser.add_argument("--test-split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--test-threshold", type=float, default=0.9)
    parser.add_argument("--test-label-threshold", type=float, default=None)
    parser.add_argument("--test-output", type=Path, default=None)
    parser.add_argument("--test-mask-output", type=Path, default=None)
    parser.add_argument("--test-seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def make_loader(args: argparse.Namespace, split: str, shuffle: bool) -> DataLoader:
    dataset = create_heatmap_dataset(
        data_yaml=args.data,
        split=split,
        img_size=(args.img_size[0], args.img_size[1]),
        cache_dir=args.cache_dir,
        gray_threshold=args.gray_threshold,
        gaussian_sigma=args.gaussian_sigma,
        max_fill_distance=args.max_fill_distance,
        center_prior_sigma=args.center_prior_sigma,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=heatmap_collate_fn,
    )


def heatmap_loss(
    logits: Tensor,
    targets: Tensor,
    pos_weight: float,
    conf_loss_weight: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    positive_weight = logits.new_tensor(pos_weight)
    heatmap_logits = logits[:, 0:1]
    confidence_logits = logits[:, 1:2]
    heatmap_loss_value = F.binary_cross_entropy_with_logits(
        heatmap_logits,
        targets,
        pos_weight=positive_weight,
    )
    confidence_loss_value = F.binary_cross_entropy_with_logits(
        confidence_logits,
        targets,
        pos_weight=positive_weight,
    )
    total_loss = heatmap_loss_value + conf_loss_weight * confidence_loss_value
    return total_loss, {
        "loss": float(total_loss.detach().cpu()),
        "heatmap_loss": float(heatmap_loss_value.detach().cpu()),
        "confidence_loss": float(confidence_loss_value.detach().cpu()),
    }


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    pos_weight: float,
    conf_loss_weight: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"loss": 0.0, "heatmap_loss": 0.0, "confidence_loss": 0.0}
    num_batches = 0
    progress = tqdm(dataloader, desc="train" if is_train else "val", leave=False)

    for batch in progress:
        images = batch["images"].to(device)
        heatmaps = batch["heatmaps"].to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss, metrics = heatmap_loss(logits, heatmaps, pos_weight, conf_loss_weight)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        for key in totals:
            totals[key] += metrics[key]
        num_batches += 1
        progress.set_postfix(loss=f"{totals['loss'] / num_batches:.4f}")

    return {key: value / max(num_batches, 1) for key, value in totals.items()}


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_name": model.__class__.__name__,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": model.config.to_dict(),
            "best_val_loss": best_val_loss,
            "output_format": "[batch, 2, height, width] logits: heatmap + confidence",
        },
        output_path,
    )


def ensure_confidence_checkpoint(checkpoint: dict[str, object], checkpoint_path: Path) -> None:
    output_format = str(checkpoint.get("output_format", ""))
    state_dict = checkpoint.get("model_state_dict")
    head_weight = state_dict.get("head.2.weight") if isinstance(state_dict, dict) else None
    has_two_channel_head = isinstance(head_weight, Tensor) and head_weight.shape[0] == 2
    if "heatmap + confidence" in output_format or has_two_channel_head:
        return
    raise ValueError(
        f"Incompatible heatmap checkpoint: {checkpoint_path}. "
        "The confidence-head model requires a 2-channel checkpoint. "
        "Please retrain the heatmap model and export a new TorchScript .pt."
    )


def load_resume(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    resume_path: Path | None,
    device: torch.device,
) -> tuple[int, float]:
    if resume_path is None or not resume_path.exists():
        print("Start new training")
        return 0, float("inf")
    print("Load from existing checkpoint")
    checkpoint = torch.load(resume_path, map_location=device)
    ensure_confidence_checkpoint(checkpoint, resume_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch", 0)), float(checkpoint.get("best_val_loss", float("inf")))


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ensure_confidence_checkpoint(checkpoint, checkpoint_path)
    config = checkpoint["config"]
    model = build_heatmap_model(
        img_size=tuple(config["img_size"]),
        backbone=str(config["backbone"]),
        width_mult=float(config["width_mult"]),
        head_channels=int(config["head_channels"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def test(
    data: str | Path = "dataset_6/data.yaml",
    checkpoint_path: str | Path = "runs/heatmap_train/best.pth",
    output_path: str | Path = "runs/heatmap_train/test_heatmap_prediction.jpg",
    mask_output_path: str | Path | None = None,
    split: str = "test",
    threshold: float = 0.5,
    label_threshold: float | None = None,
    img_size: tuple[int, int] = (640, 640),
    cache_dir: str | Path = "data",
    gray_threshold: int = 150,
    gaussian_sigma: float = 8.0,
    max_fill_distance: float = 24.0,
    center_prior_sigma: float = 0.45,
    device: str | torch.device | None = None,
    seed: int | None = None,
    draw_labels: bool = True,
) -> Path:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    mask_output_path = Path(mask_output_path) if mask_output_path is not None else default_mask_output_path(output_path)

    model = load_model_from_checkpoint(checkpoint_path, device)
    dataset = HeatmapDataset(
        data_yaml=data,
        split=split,
        img_size=img_size,
        cache_dir=cache_dir,
        gray_threshold=gray_threshold,
        gaussian_sigma=gaussian_sigma,
        max_fill_distance=max_fill_distance,
        center_prior_sigma=center_prior_sigma,
    )
    index = random.Random(seed).randrange(len(dataset))
    sample = dataset[index]
    image_tensor = sample["image"].unsqueeze(0).to(device)
    probabilities = torch.sigmoid(model(image_tensor))[0].detach().cpu().numpy()
    heatmap = probabilities[0]
    confidence = probabilities[1]
    combined_confidence = heatmap * confidence
    prediction_mask = heatmap > threshold
    label_heatmap = sample["heatmap"][0].numpy()
    label_mask = label_heatmap > (threshold if label_threshold is None else label_threshold)

    image_array = (sample["image"].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image_array)
    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2 in connected_component_boxes(label_mask):
        draw.rectangle((x1, y1, x2, y2), outline="yellow", width=5)

    if draw_labels:
        label_path = label_path_for_dataset_sample(dataset, index)
        for x1, y1, x2, y2 in read_label_boxes(label_path, image.width, image.height):
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)

    best_box = best_confidence_box(prediction_mask, combined_confidence)
    if best_box is not None:
        x1, y1, x2, y2, score = best_box
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=2)
        draw.text((x1, max(0, y1 - 12)), f"conf {score:.3f}", fill="lime")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    mask_output_path.parent.mkdir(parents=True, exist_ok=True)
    draw_heatmap_mask_visualization(image_array, heatmap, prediction_mask, combined_confidence).save(mask_output_path)
    print(f"Saved heatmap test visualization to: {output_path}")
    print(f"Saved heatmap mask visualization to: {mask_output_path}")
    return output_path


def default_mask_output_path(output_path: Path) -> Path:
    suffix = output_path.suffix or ".jpg"
    return output_path.with_name(f"{output_path.stem}_mask{suffix}")


def draw_heatmap_mask_visualization(
    image_array: np.ndarray,
    heatmap: np.ndarray,
    prediction_mask: np.ndarray,
    combined_confidence: np.ndarray | None = None,
) -> Image.Image:
    visual_map = heatmap if combined_confidence is None else combined_confidence
    heatmap_uint8 = (visual_map.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    heatmap_rgb = np.stack([heatmap_uint8, heatmap_uint8, heatmap_uint8], axis=-1)

    original = image_array.astype(np.float32)
    dimmed = original * 0.35 + heatmap_rgb.astype(np.float32) * 0.65
    dimmed[prediction_mask] = np.array([0.0, 255.0, 0.0], dtype=np.float32)
    return Image.fromarray(dimmed.clip(0, 255).astype(np.uint8))


def best_confidence_box(
    mask: np.ndarray,
    confidence_map: np.ndarray,
) -> tuple[int, int, int, int, float] | None:
    best_box: tuple[int, int, int, int, float] | None = None
    for x1, y1, x2, y2 in connected_component_boxes(mask):
        region_mask = mask[y1 : y2 + 1, x1 : x2 + 1]
        region_confidence = confidence_map[y1 : y2 + 1, x1 : x2 + 1]
        if not region_mask.any():
            continue
        score = float(region_confidence[region_mask].mean())
        if best_box is None or score > best_box[4]:
            best_box = (x1, y1, x2, y2, score)
    return best_box


def label_path_for_dataset_sample(dataset: HeatmapDataset, index: int) -> Path:
    image_path = dataset.image_paths[index]
    label_dir = labels_dir_from_images_dir(dataset.image_dir)
    return label_dir / f"{image_path.stem}.txt"


def read_label_boxes(label_path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    if not label_path.exists():
        return []

    boxes: list[tuple[int, int, int, int]] = []
    for _, bbox in read_yolo_rows(label_path):
        x1, y1, x2, y2 = yolo_box_to_pixels(bbox, width, height)
        x2 = max(x1, min(width - 1, x2 - 1))
        y2 = max(y1, min(height - 1, y2 - 1))
        boxes.append((x1, y1, x2, y2))
    return boxes


def connected_component_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(x, y)]
            visited[y, x] = True
            min_x = max_x = x
            min_y = max_y = y

            while stack:
                current_x, current_y = stack.pop()
                min_x = min(min_x, current_x)
                max_x = max(max_x, current_x)
                min_y = min(min_y, current_y)
                max_y = max(max_y, current_y)

                for next_x, next_y in (
                    (current_x - 1, current_y),
                    (current_x + 1, current_y),
                    (current_x, current_y - 1),
                    (current_x, current_y + 1),
                ):
                    if not (0 <= next_x < width and 0 <= next_y < height):
                        continue
                    if visited[next_y, next_x] or not mask[next_y, next_x]:
                        continue
                    visited[next_y, next_x] = True
                    stack.append((next_x, next_y))

            boxes.append((min_x, min_y, max_x, max_y))

    return boxes


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_loader = make_loader(args, "train", shuffle=True)
    val_loader = make_loader(args, "val", shuffle=False)

    model = build_heatmap_model(
        img_size=(args.img_size[0], args.img_size[1]),
        backbone=args.backbone,
        width_mult=args.width_mult,
        head_channels=args.head_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    resume_path = args.resume
    start_epoch, best_val_loss = load_resume(model, optimizer, resume_path, device)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            args.pos_weight,
            args.conf_loss_weight,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            None,
            args.pos_weight,
            args.conf_loss_weight,
        )
        val_loss = val_metrics["loss"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(args.output_dir / "best.pth", model, optimizer, epoch, best_val_loss)
        save_checkpoint(args.output_dir / "latest.pth", model, optimizer, epoch, best_val_loss)

        print(
            f"epoch {epoch}/{args.epochs} "
            f"train loss={train_metrics['loss']:.4f} "
            f"hm={train_metrics['heatmap_loss']:.4f} "
            f"conf={train_metrics['confidence_loss']:.4f} "
            f"val loss={val_metrics['loss']:.4f} "
            f"hm={val_metrics['heatmap_loss']:.4f} "
            f"conf={val_metrics['confidence_loss']:.4f}"
        )

        test_output = args.test_output or args.output_dir / "test_heatmap_prediction.jpg"
        test_mask_output = args.test_mask_output or default_mask_output_path(test_output)
        test(
            data=args.data,
            checkpoint_path=args.output_dir / "latest.pth",
            output_path=test_output,
            mask_output_path=test_mask_output,
            split=args.test_split,
            threshold=args.test_threshold,
            label_threshold=args.test_label_threshold,
            img_size=(args.img_size[0], args.img_size[1]),
            cache_dir=args.cache_dir,
            gray_threshold=args.gray_threshold,
            gaussian_sigma=args.gaussian_sigma,
            max_fill_distance=args.max_fill_distance,
            center_prior_sigma=args.center_prior_sigma,
            device=device,
            seed=args.test_seed,
        )


if __name__ == "__main__":
    # main()
    test()
