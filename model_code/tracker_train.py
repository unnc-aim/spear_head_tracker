"""
Train YOLOBBoxTracker with YOLO-format labels.

Example:
    python model_code/tracker_train.py --data dataset_6/data.yaml --epochs 50

Dataset labels must be standard YOLO txt rows:
    class_id x_center y_center width height

The class_id column is ignored by this bbox tracker.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tracker_dataset import (
    IMAGE_EXTENSIONS,
    YOLODataConfig,
    create_tracker_dataset,
    labels_dir_from_images_dir,
    load_yolo_data_config,
    tracker_collate_fn,
)
from tracker_model import build_tracker_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO-style bbox tracker.")
    parser.add_argument("--data", type=Path, required=True, help="Dataset root path or YOLO data.yaml path.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tracker_train"))
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--num-boxes", type=int, default=100)
    parser.add_argument("--img-size", type=int, nargs=2, default=(640, 640), metavar=("H", "W"))
    parser.add_argument("--backbone", choices=("resnet", "darknet", "mobilenet", "efficientnet"), default="efficientnet")
    parser.add_argument("--width-mult", type=float, default=2)
    parser.add_argument("--fpn-channels", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conf-loss-weight", type=float, default=1.0)
    parser.add_argument("--bbox-loss-weight", type=float, default=5.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--visualize-test", action="store_true", help="Draw predictions on one random image.")
    parser.add_argument("--no-visualize-test", action="store_false", dest="visualize_test")
    parser.add_argument("--visualize-split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--visualize-conf", type=float, default=0.25)
    parser.add_argument("--visualize-top-k", type=int, default=5)
    parser.add_argument("--visualize-output", type=Path, default=None)
    parser.add_argument("--visualize-seed", type=int, default=None)
    parser.set_defaults(visualize_test=True)
    return parser.parse_args()


def resolve_data_config(data_path: Path) -> YOLODataConfig:
    if data_path.is_file():
        return load_yolo_data_config(data_path)

    return YOLODataConfig(
        root=data_path,
        train_images=data_path / "images" / "train",
        val_images=data_path / "images" / "val",
        test_images=data_path / "images" / "test",
        num_classes=1,
        class_names=["bbox"],
    )


def make_dataloader(
    data_root: Path | None,
    split: str,
    img_size: tuple[int, int],
    max_boxes: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    image_dir: Path | None = None,
) -> DataLoader:
    dataset = create_tracker_dataset(
        root=data_root,
        split=split,
        img_size=img_size,
        max_boxes=max_boxes,
        image_dir=image_dir,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=tracker_collate_fn,
    )


def tracker_loss(
    predictions: Tensor,
    target_boxes: Tensor,
    valid_mask: Tensor,
    bbox_weight: float = 5.0,
    conf_weight: float = 1.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> tuple[Tensor, dict[str, float]]:
    """
    Compute bbox tracker loss with Hungarian matching per image.

    Every GT box is matched to one dense prediction by minimum GIoU bbox cost.
    Matched predictions receive bbox regression loss and positive objectness;
    all unmatched predictions receive negative objectness.
    """
    pred_bbox = predictions[..., 0:4]
    pred_conf = predictions[..., 4]

    target_bbox = target_boxes[..., 1:5]
    target_conf = torch.zeros_like(pred_conf)
    matched_pred_boxes: list[Tensor] = []
    matched_target_boxes: list[Tensor] = []

    for batch_index in range(predictions.shape[0]):
        gt_boxes = target_bbox[batch_index][valid_mask[batch_index]]
        if gt_boxes.numel() == 0:
            continue

        pred_boxes = pred_bbox[batch_index]
        pred_indices, gt_indices = hungarian_match(pred_boxes.detach(), gt_boxes.detach())
        if pred_indices.numel() == 0:
            continue

        target_conf[batch_index, pred_indices] = 1.0
        matched_pred_boxes.append(pred_boxes[pred_indices])
        matched_target_boxes.append(gt_boxes[gt_indices])

    conf_loss = focal_binary_cross_entropy(
        pred_conf,
        target_conf,
        alpha=focal_alpha,
        gamma=focal_gamma,
    )

    if matched_pred_boxes:
        bbox_loss = generalized_iou_loss(
            torch.cat(matched_pred_boxes, dim=0),
            torch.cat(matched_target_boxes, dim=0),
        )
    else:
        bbox_loss = predictions.new_tensor(0.0)

    total = bbox_weight * bbox_loss + conf_weight * conf_loss
    metrics = {
        "loss": float(total.detach().cpu()),
        "bbox_loss": float(bbox_loss.detach().cpu()),
        "conf_loss": float(conf_loss.detach().cpu()),
    }
    return total, metrics


def focal_binary_cross_entropy(
    inputs: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    bce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = inputs * targets + (1.0 - inputs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    focal_factor = (1.0 - p_t).pow(gamma)
    return (alpha_t * focal_factor * bce_loss).mean()


def hungarian_match(pred_boxes: Tensor, target_boxes: Tensor) -> tuple[Tensor, Tensor]:
    """Return matched prediction indices and target indices for one image."""
    cost = 1.0 - generalized_box_iou(pred_boxes, target_boxes)
    try:
        from scipy.optimize import linear_sum_assignment

        pred_indices, target_indices = linear_sum_assignment(cost.detach().cpu().numpy())
        return (
            torch.as_tensor(pred_indices, device=pred_boxes.device, dtype=torch.long),
            torch.as_tensor(target_indices, device=pred_boxes.device, dtype=torch.long),
        )
    except ImportError:
        return hungarian_match_dynamic_programming(cost)


def hungarian_match_dynamic_programming(cost: Tensor) -> tuple[Tensor, Tensor]:
    """
    Lightweight assignment fallback for small GT counts.

    This solves min-cost one-to-one assignment over predictions for each target
    with DP over the target bitmask. It is practical here because YOLO label
    files usually contain only a few tracked boxes per image.
    """
    num_predictions, num_targets = cost.shape
    if num_targets == 0:
        empty = torch.empty(0, device=cost.device, dtype=torch.long)
        return empty, empty
    if num_targets > 20:
        raise RuntimeError("Install scipy for Hungarian matching when an image has more than 20 boxes.")

    full_mask = (1 << num_targets) - 1
    states: dict[int, tuple[float, list[tuple[int, int]]]] = {0: (0.0, [])}
    cpu_cost = cost.detach().cpu()

    for pred_index in range(num_predictions):
        next_states = dict(states)
        for mask, (value, pairs) in states.items():
            if mask == full_mask:
                continue
            for target_index in range(num_targets):
                bit = 1 << target_index
                if mask & bit:
                    continue
                next_mask = mask | bit
                next_value = value + float(cpu_cost[pred_index, target_index])
                if next_mask not in next_states or next_value < next_states[next_mask][0]:
                    next_states[next_mask] = (next_value, [*pairs, (pred_index, target_index)])
        states = next_states

    pairs = states[full_mask][1]
    pred_indices = torch.tensor([pair[0] for pair in pairs], device=cost.device, dtype=torch.long)
    target_indices = torch.tensor([pair[1] for pair in pairs], device=cost.device, dtype=torch.long)
    return pred_indices, target_indices


def box_xywh_to_xyxy(boxes: Tensor) -> Tensor:
    x_center, y_center, width, height = boxes.unbind(dim=-1)
    half_width = width / 2.0
    half_height = height / 2.0
    return torch.stack(
        (
            x_center - half_width,
            y_center - half_height,
            x_center + half_width,
            y_center + half_height,
        ),
        dim=-1,
    )


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    boxes1 = box_xywh_to_xyxy(boxes1).clamp(0.0, 1.0)
    boxes2 = box_xywh_to_xyxy(boxes2).clamp(0.0, 1.0)

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection_wh = (right_bottom - left_top).clamp(min=0.0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]

    union = area1[:, None] + area2[None, :] - intersection
    iou = intersection / union.clamp(min=1e-7)

    enclosing_left_top = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    enclosing_right_bottom = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclosing_wh = (enclosing_right_bottom - enclosing_left_top).clamp(min=0.0)
    enclosing_area = enclosing_wh[..., 0] * enclosing_wh[..., 1]

    return iou - (enclosing_area - union) / enclosing_area.clamp(min=1e-7)


def generalized_iou_loss(pred_boxes: Tensor, target_boxes: Tensor) -> Tensor:
    pairwise_giou = generalized_box_iou(pred_boxes, target_boxes)
    matched_giou = pairwise_giou.diagonal()
    return (1.0 - matched_giou).mean()


def box_area(boxes: Tensor) -> Tensor:
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0.0)
    return wh[:, 0] * wh[:, 1]


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    totals = {"loss": 0.0, "bbox_loss": 0.0, "conf_loss": 0.0}
    num_batches = 0

    phase = "train" if is_train else "val"
    progress = tqdm(dataloader, desc=phase, leave=False)
    for batch in progress:
        images = batch["images"].to(device)
        boxes = batch["boxes"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        with torch.set_grad_enabled(is_train):
            predictions = model(images)
            loss, metrics = tracker_loss(
                predictions=predictions,
                target_boxes=boxes,
                valid_mask=valid_mask,
                bbox_weight=args.bbox_loss_weight,
                conf_weight=args.conf_loss_weight,
                focal_alpha=args.focal_alpha,
                focal_gamma=args.focal_gamma,
            )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        for key in totals:
            totals[key] += metrics[key]
        num_batches += 1
        progress.set_postfix({key: f"{totals[key] / num_batches:.4f}" for key in totals})

    return {key: value / max(num_batches, 1) for key, value in totals.items()}


def load_resume(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    resume_path: Path | None,
    device: torch.device,
) -> tuple[int, float]:
    if resume_path is None:
        return 0, float("inf")

    checkpoint = torch.load(resume_path, map_location=device)
    if not is_bbox_only_checkpoint(checkpoint):
        raise ValueError(
            f"Checkpoint is not compatible with bbox-only tracker: {resume_path}. "
            "Please train a new bbox-only checkpoint or remove the old latest.pth."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch", 0)), float(checkpoint.get("best_val_loss", float("inf")))


def resolve_resume_path(output_dir: Path, resume_path: Path | None) -> Path | None:
    if resume_path is not None:
        return resume_path

    # latest_path = output_dir / "latest.pth"
    # if not latest_path.exists():
    #     return None

    # checkpoint = torch.load(latest_path, map_location="cpu")
    # if is_bbox_only_checkpoint(checkpoint):
    #     return latest_path

    # print(f"Skipping incompatible classification checkpoint: {latest_path}")
    return None


def is_bbox_only_checkpoint(checkpoint: dict[str, object]) -> bool:
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        return False
    return "num_classes" not in config and checkpoint.get("output_format") in {
        "[batch, num_boxes, 5]",
        "[batch, num_predictions, 5]",
    }


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    class_names: list[str],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_name": model.__class__.__name__,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": model.config.to_dict(),
            "class_names": class_names,
            "best_val_loss": best_val_loss,
            "output_format": "[batch, num_predictions, 5]",
            "box_format": "normalized xywh + object_conf",
        },
        output_path,
    )


def format_metrics(metrics: dict[str, float]) -> str:
    return " ".join(f"{key}={value:.4f}" for key, value in metrics.items())


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not is_bbox_only_checkpoint(checkpoint):
        raise ValueError(
            f"Checkpoint is not compatible with bbox-only tracker: {checkpoint_path}. "
            "Expected output_format='[batch, num_predictions, 5]' and no num_classes in config."
        )
    config = checkpoint["config"]
    model = build_tracker_model(
        num_boxes=int(config["num_boxes"]),
        img_size=tuple(config["img_size"]),
        width_mult=float(config["width_mult"]),
        fpn_channels=int(config.get("fpn_channels", 128)),
        backbone=str(config.get("backbone", "efficientnet")),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def resolve_split_image_dir(data_config: YOLODataConfig, split: str) -> Path:
    if split == "train":
        return data_config.train_images
    if split == "val":
        if data_config.val_images is None:
            raise ValueError("No val split is configured.")
        return data_config.val_images
    if data_config.test_images is None:
        raise ValueError("No test split is configured.")
    return data_config.test_images


def find_image_paths(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Visualization image directory not found: {image_dir}")
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_to_tensor(image: Image.Image, img_size: tuple[int, int], device: torch.device) -> Tensor:
    height, width = img_size
    resized = resize_image_for_model(image, img_size)
    data = torch.frombuffer(bytearray(resized.tobytes()), dtype=torch.uint8)
    data = data.view(height, width, 3)
    return data.permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)


def resize_image_for_model(image: Image.Image, img_size: tuple[int, int]) -> Image.Image:
    height, width = img_size
    return image.convert("RGB").resize((width, height), Image.BILINEAR)


def draw_prediction_boxes(
    image: Image.Image,
    predictions: Tensor,
    class_names: list[str],
    conf_threshold: float,
    fallback_top_k: int = 5,
) -> Image.Image:
    drawn = image.convert("RGB").copy()
    draw = ImageDraw.Draw(drawn)
    font = ImageFont.load_default()
    width, height = drawn.size

    boxes = predictions[:, 0:4]
    scores = predictions[:, 4]
    keep = scores >= conf_threshold
    used_fallback = False
    if not keep.any() and fallback_top_k > 0 and scores.numel() > 0:
        top_k = min(fallback_top_k, scores.numel())
        top_indices = torch.topk(scores, k=top_k).indices
        keep = torch.zeros_like(scores, dtype=torch.bool)
        keep[top_indices] = True
        used_fallback = True

    palette = ["lime", "cyan", "yellow", "magenta", "orange", "white"]
    for box_index, (bbox, score) in enumerate(zip(boxes[keep], scores[keep])):
        x_center, y_center, box_width, box_height = bbox.tolist()
        x1 = int((x_center - box_width / 2.0) * width)
        y1 = int((y_center - box_height / 2.0) * height)
        x2 = int((x_center + box_width / 2.0) * width)
        y2 = int((y_center + box_height / 2.0) * height)
        x1, x2 = sorted((max(0, x1), min(width - 1, x2)))
        y1, y2 = sorted((max(0, y1), min(height - 1, y2)))

        color = palette[box_index % len(palette)]
        prefix = "top" if used_fallback else "track"
        label = f"{prefix} {float(score):.2f}"

        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        text_box = draw.textbbox((x1, y1), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_y = max(0, y1 - text_height - 4)
        draw.rectangle((x1, label_y, x1 + text_width + 6, label_y + text_height + 4), fill=color)
        draw.text((x1 + 3, label_y + 2), label, fill="black", font=font)

    if used_fallback:
        draw.text(
            (12, 12),
            f"no predictions >= {conf_threshold:.2f}; showing top {int(keep.sum())}",
            fill="yellow",
            font=font,
        )
    elif not keep.any():
        draw.text((12, 12), f"no predictions >= {conf_threshold:.2f}", fill="red", font=font)

    return drawn


def read_yolo_label_file(label_path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    if not label_path.exists():
        return []

    labels: list[tuple[int, tuple[float, float, float, float]]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            raise ValueError(
                f"Invalid YOLO label at {label_path}:{line_number}. "
                "Expected: class_id x_center y_center width height"
            )
        class_id = int(float(parts[0]))
        x_center, y_center, box_width, box_height = (float(value) for value in parts[1:])
        labels.append((class_id, (x_center, y_center, box_width, box_height)))
    return labels


def draw_label_boxes(
    image: Image.Image,
    labels: list[tuple[int, tuple[float, float, float, float]]],
    class_names: list[str],
) -> Image.Image:
    drawn = image.convert("RGB").copy()
    draw = ImageDraw.Draw(drawn)
    font = ImageFont.load_default()
    width, height = drawn.size

    for class_id, bbox in labels:
        x_center, y_center, box_width, box_height = bbox
        x1 = int((x_center - box_width / 2.0) * width)
        y1 = int((y_center - box_height / 2.0) * height)
        x2 = int((x_center + box_width / 2.0) * width)
        y2 = int((y_center + box_height / 2.0) * height)
        x1, x2 = sorted((max(0, x1), min(width - 1, x2)))
        y1, y2 = sorted((max(0, y1), min(height - 1, y2)))

        label_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
        label = f"label {label_name}"
        draw.rectangle((x1, y1, x2, y2), outline="red", width=4)
        text_box = draw.textbbox((x1, y2), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_y = min(height - text_height - 4, y2 + 2)
        draw.rectangle((x1, label_y, x1 + text_width + 6, label_y + text_height + 4), fill="red")
        draw.text((x1 + 3, label_y + 2), label, fill="white", font=font)

    return drawn


def resolve_label_path_for_image(data_config: YOLODataConfig, split: str, image_path: Path) -> Path:
    image_dir = resolve_split_image_dir(data_config, split)
    label_dir = labels_dir_from_images_dir(image_dir)
    return label_dir / f"{image_path.stem}.txt"


@torch.no_grad()
def visualize_random_prediction(
    model: nn.Module,
    data_config: YOLODataConfig,
    split: str,
    img_size: tuple[int, int],
    device: torch.device,
    output_path: Path,
    conf_threshold: float,
    seed: int | None,
    fallback_top_k: int = 5,
) -> Path:
    image_dir = resolve_split_image_dir(data_config, split)
    image_paths = find_image_paths(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found for visualization in: {image_dir}")

    rng = random.Random(seed)
    image_path = rng.choice(image_paths)
    image = Image.open(image_path).convert("RGB")

    model.eval()
    model_input = image_to_tensor(image, img_size, device)
    predictions = model(model_input)[0].detach().cpu()
    visualized = draw_prediction_boxes(
        image=image,
        predictions=predictions,
        class_names=data_config.class_names,
        conf_threshold=conf_threshold,
        fallback_top_k=fallback_top_k,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    visualized.save(output_path)
    print(f"Saved visualization for {image_path} to: {output_path}")
    return output_path


@torch.no_grad()
def test(
    data: str | Path = "dataset_6/data.yaml",
    checkpoint_path: str | Path = "runs/tracker_train/best.pth",
    output_path: str | Path = "runs/tracker_train/test_prediction.jpg",
    image_path: str | Path | None = None,
    split: str = "test",
    conf_threshold: float = 0.25,
    device: str | torch.device | None = None,
    seed: int | None = None,
    draw_labels: bool = True,
    fallback_top_k: int = 5,
) -> Path:
    """
    Load best.pth, draw predicted bboxes on one image, and save the result.

    This function is intentionally standalone so it can be imported and called
    directly from another script or a Python console.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_config = resolve_data_config(Path(data))
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
    class_names = checkpoint.get("class_names", data_config.class_names)
    img_size = tuple(checkpoint["config"]["img_size"])

    if image_path is None:
        image_dir = resolve_split_image_dir(data_config, split)
        image_paths = find_image_paths(image_dir)
        if not image_paths:
            raise FileNotFoundError(f"No images found in: {image_dir}")
        image_path = random.Random(seed).choice(image_paths)
    else:
        image_path = Path(image_path)

    image = Image.open(image_path).convert("RGB")
    visual_image = resize_image_for_model(image, img_size)
    model_input = image_to_tensor(image, img_size, device)
    predictions = model(model_input)[0].detach().cpu()
    visualized = draw_prediction_boxes(
        image=visual_image,
        predictions=predictions,
        class_names=list(class_names),
        conf_threshold=conf_threshold,
        fallback_top_k=fallback_top_k,
    )
    if draw_labels:
        label_path = resolve_label_path_for_image(data_config, split, Path(image_path))
        labels = read_yolo_label_file(label_path)
        visualized = draw_label_boxes(visualized, labels, list(class_names))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    visualized.save(output_path)
    print(f"Saved test visualization for {image_path} to: {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    img_size = (args.img_size[0], args.img_size[1])
    data_config = resolve_data_config(args.data)

    train_loader = make_dataloader(
        data_root=data_config.root,
        split="train",
        img_size=img_size,
        max_boxes=args.num_boxes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        image_dir=data_config.train_images,
    )

    val_loader = None
    if data_config.val_images is not None and data_config.val_images.exists():
        val_loader = make_dataloader(
            data_root=data_config.root,
            split="val",
            img_size=img_size,
            max_boxes=args.num_boxes,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            image_dir=data_config.val_images,
        )

    model = build_tracker_model(
        num_boxes=args.num_boxes,
        img_size=img_size,
        width_mult=args.width_mult,
        fpn_channels=args.fpn_channels,
        backbone=args.backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    resume_path = resolve_resume_path(args.output_dir, args.resume)
    if resume_path is not None:
        print(f"Resuming from checkpoint: {resume_path}")
    start_epoch=-1
    best_val_loss=100
    start_epoch, best_val_loss = load_resume(model, optimizer, resume_path, device)
    

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            optimizer=optimizer,
            args=args,
        )

        message = f"epoch {epoch}/{args.epochs} train {format_metrics(train_metrics)}"
        val_loss = train_metrics["loss"]

        if val_loader is not None:
            val_metrics = run_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                optimizer=None,
                args=args,
            )
            val_loss = val_metrics["loss"]
            message += f" val {format_metrics(val_metrics)}"

        checkpoint_for_test = args.output_dir / "latest.pth"
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                args.output_dir / "best.pth",
                model,
                optimizer,
                epoch,
                best_val_loss,
                data_config.class_names,
            )
            checkpoint_for_test = args.output_dir / "best.pth"

        latest_path = args.output_dir / "latest.pth"
        save_checkpoint(latest_path, model, optimizer, epoch, best_val_loss, data_config.class_names)

        print(message)

        if val_loader is not None:
            test_output = args.visualize_output
            if test_output is None:
                test_output = args.output_dir / f"{args.visualize_split}_prediction.jpg"
            test(
                data=args.data,
                checkpoint_path=checkpoint_for_test,
                output_path=test_output,
                split=args.visualize_split,
                conf_threshold=args.visualize_conf,
                device=device,
                seed=args.visualize_seed,
                fallback_top_k=args.visualize_top_k,
            )

    print(f"Training complete. Latest checkpoint: {args.output_dir / 'latest.pth'}")

    if args.visualize_test:
        visualize_output = args.visualize_output
        if visualize_output is None:
            visualize_output = args.output_dir / f"{args.visualize_split}_prediction.jpg"
        visualize_random_prediction(
            model=model,
            data_config=data_config,
            split=args.visualize_split,
            img_size=img_size,
            device=device,
            output_path=visualize_output,
            conf_threshold=args.visualize_conf,
            seed=args.visualize_seed,
            fallback_top_k=args.visualize_top_k,
        )


if __name__ == "__main__":
    # main()
    test()
