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

from heatmap_dataset import HeatmapDataset, create_heatmap_dataset, heatmap_collate_fn
from heatmap_model import build_heatmap_model
from tracker_dataset import labels_dir_from_images_dir, load_yolo_data_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train heatmap tracker.")
    parser.add_argument("--data", type=Path, default=Path("dataset_6/data.yaml"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/heatmap_train"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--img-size", type=int, nargs=2, default=(640, 640), metavar=("H", "W"))
    parser.add_argument("--backbone", choices=("resnet", "darknet", "mobilenet", "efficientnet"), default="efficientnet")
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--head-channels", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gray-threshold", type=int, default=150)
    parser.add_argument("--gaussian-sigma", type=float, default=8.0)
    parser.add_argument("--pos-weight", type=float, default=5.0)
    parser.add_argument("--test-split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--test-threshold", type=float, default=0.5)
    parser.add_argument("--test-output", type=Path, default=None)
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
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=heatmap_collate_fn,
    )


def heatmap_loss(logits: Tensor, targets: Tensor, pos_weight: float) -> Tensor:
    positive_weight = logits.new_tensor(pos_weight)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=positive_weight)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    pos_weight: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    num_batches = 0
    progress = tqdm(dataloader, desc="train" if is_train else "val", leave=False)

    for batch in progress:
        images = batch["images"].to(device)
        heatmaps = batch["heatmaps"].to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = heatmap_loss(logits, heatmaps, pos_weight)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += float(loss.detach().cpu())
        num_batches += 1
        progress.set_postfix(loss=f"{total_loss / num_batches:.4f}")

    return {"loss": total_loss / max(num_batches, 1)}


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
            "output_format": "[batch, 1, height, width] logits",
        },
        output_path,
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
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch", 0)), float(checkpoint.get("best_val_loss", float("inf")))


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
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
    split: str = "test",
    threshold: float = 0.5,
    img_size: tuple[int, int] = (640, 640),
    cache_dir: str | Path = "data",
    gray_threshold: int = 150,
    gaussian_sigma: float = 8.0,
    device: str | torch.device | None = None,
    seed: int | None = None,
    draw_labels: bool = True,
) -> Path:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    model = load_model_from_checkpoint(checkpoint_path, device)
    dataset = HeatmapDataset(
        data_yaml=data,
        split=split,
        img_size=img_size,
        cache_dir=cache_dir,
        gray_threshold=gray_threshold,
        gaussian_sigma=gaussian_sigma,
    )
    index = random.Random(seed).randrange(len(dataset))
    sample = dataset[index]
    image_tensor = sample["image"].unsqueeze(0).to(device)
    heatmap = torch.sigmoid(model(image_tensor))[0, 0].detach().cpu().numpy()
    mask = heatmap > threshold

    image_array = (sample["image"].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image_array)
    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2 in connected_component_boxes(mask):
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=3)

    if draw_labels:
        label_path = resolve_label_path(data, split, index)
        for x1, y1, x2, y2 in read_label_boxes(label_path, image.width, image.height):
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved heatmap test visualization to: {output_path}")
    return output_path


def resolve_label_path(data_yaml: str | Path, split: str, index: int) -> Path:
    data_config = load_yolo_data_config(data_yaml)
    if split == "train":
        image_dir = data_config.train_images
    elif split == "val":
        if data_config.val_images is None:
            raise ValueError("data.yaml does not define val split.")
        image_dir = data_config.val_images
    else:
        if data_config.test_images is None:
            raise ValueError("data.yaml does not define test split.")
        image_dir = data_config.test_images

    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    label_dir = labels_dir_from_images_dir(image_dir)
    return label_dir / f"{image_paths[index].stem}.txt"


def read_label_boxes(label_path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    if not label_path.exists():
        return []

    boxes: list[tuple[int, int, int, int]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        x_center, y_center, box_width, box_height = (float(value) for value in parts[1:])
        x1 = int(max(0, (x_center - box_width / 2.0) * width))
        y1 = int(max(0, (y_center - box_height / 2.0) * height))
        x2 = int(min(width - 1, (x_center + box_width / 2.0) * width))
        y2 = int(min(height - 1, (y_center + box_height / 2.0) * height))
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
        train_metrics = run_epoch(model, train_loader, device, optimizer, args.pos_weight)
        val_metrics = run_epoch(model, val_loader, device, None, args.pos_weight)
        val_loss = val_metrics["loss"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(args.output_dir / "best.pth", model, optimizer, epoch, best_val_loss)
        save_checkpoint(args.output_dir / "latest.pth", model, optimizer, epoch, best_val_loss)

        print(
            f"epoch {epoch}/{args.epochs} "
            f"train loss={train_metrics['loss']:.4f} val loss={val_metrics['loss']:.4f}"
        )

        test_output = args.test_output or args.output_dir / "test_heatmap_prediction.jpg"
        test(
            data=args.data,
            checkpoint_path=args.output_dir / "latest.pth",
            output_path=test_output,
            split=args.test_split,
            threshold=args.test_threshold,
            img_size=(args.img_size[0], args.img_size[1]),
            cache_dir=args.cache_dir,
            gray_threshold=args.gray_threshold,
            gaussian_sigma=args.gaussian_sigma,
            device=device,
            seed=args.test_seed,
        )


if __name__ == "__main__":
    # main()
    test()
