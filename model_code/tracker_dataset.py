"""
Dataset utilities for YOLO-format bbox labels.

Expected directory layout:
    dataset_root/
        images/
            train/
                0001.jpg
            val/
                0002.jpg
        labels/
            train/
                0001.txt
            val/
                0002.txt

Each label row must use standard YOLO format:
    class_id x_center y_center width height

All coordinates are normalized to 0..1.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class YOLODataConfig:
    """Resolved paths and class metadata from a YOLO data.yaml file."""

    root: Path
    train_images: Path
    val_images: Path | None
    test_images: Path | None
    num_classes: int
    class_names: list[str]


def load_yolo_data_config(data_yaml: str | Path) -> YOLODataConfig:
    """Load the subset of YOLO data.yaml used by this trainer."""
    data_yaml = Path(data_yaml)
    raw_config = _parse_simple_yaml(data_yaml)

    dataset_root = _resolve_path(data_yaml.parent, raw_config.get("path", "."))
    train_images = _resolve_path(dataset_root, _require_key(raw_config, "train"))
    val_images = _resolve_optional_path(dataset_root, raw_config.get("val"))
    test_images = _resolve_optional_path(dataset_root, raw_config.get("test"))

    class_names = _parse_class_names(raw_config.get("names"))
    num_classes = int(raw_config.get("nc", len(class_names) if class_names else 1))
    if not class_names:
        class_names = [str(index) for index in range(num_classes)]
    if len(class_names) != num_classes:
        raise ValueError(
            f"{data_yaml} has nc={num_classes}, but names contains {len(class_names)} entries."
        )

    return YOLODataConfig(
        root=dataset_root,
        train_images=train_images,
        val_images=val_images,
        test_images=test_images,
        num_classes=num_classes,
        class_names=class_names,
    )


def labels_dir_from_images_dir(image_dir: str | Path) -> Path:
    """Infer labels/<split> from images/<split>."""
    image_dir = Path(image_dir)
    parts = list(image_dir.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts)
    return image_dir.parent.parent / "labels" / image_dir.name


class YOLOTrackerDataset(Dataset):
    """Load images and YOLO label txt files for bbox tracker training."""

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "train",
        img_size: tuple[int, int] = (640, 640),
        max_boxes: int = 100,
        image_dir: str | Path | None = None,
        label_dir: str | Path | None = None,
        num_classes: int | None = None,
        cache_dir: str | Path = "data",
    ) -> None:
        self.root = Path(root) if root is not None else None
        self.split = split
        self.img_size = img_size
        self.max_boxes = max_boxes
        self.num_classes = num_classes
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if image_dir is None:
            if self.root is None:
                raise ValueError("Either root or image_dir must be provided.")
            image_dir = self.root / "images" / split
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir) if label_dir is not None else labels_dir_from_images_dir(self.image_dir)
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        self.image_paths = self._find_images(self.image_dir)
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in: {self.image_dir}")

        self.images_path = self._cache_path()
        if not self._cache_is_valid():
            self._generate_image_cache()
        self.images = np.load(self.images_path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        image_path = self.image_paths[index]
        label_path = self.label_dir / f"{image_path.stem}.txt"

        image = self._load_cached_image(index)
        boxes, valid_mask = self._load_labels(label_path)

        return {
            "image": image,
            "boxes": boxes,
            "valid_mask": valid_mask,
            "image_path": str(image_path),
        }

    @staticmethod
    def _find_images(image_dir: Path) -> list[Path]:
        return sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def _cache_path(self) -> Path:
        height, width = self.img_size
        image_dir_key = hashlib.sha1(str(self.image_dir.resolve()).encode("utf-8")).hexdigest()[:10]
        return self.cache_dir / f"tracker_{image_dir_key}_{self.split}_images_{height}x{width}.npy"

    def _cache_is_valid(self) -> bool:
        if not self.images_path.exists():
            return False
        try:
            cached = np.load(self.images_path, mmap_mode="r")
        except (OSError, ValueError):
            return False
        height, width = self.img_size
        return cached.shape == (len(self.image_paths), height, width, 3) and cached.dtype == np.uint8

    def _generate_image_cache(self) -> None:
        height, width = self.img_size
        images = np.zeros((len(self.image_paths), height, width, 3), dtype=np.uint8)
        for index, image_path in enumerate(self.image_paths):
            images[index] = self._read_resized_image(image_path)
        np.save(self.images_path, images)

    def _read_resized_image(self, image_path: Path) -> np.ndarray:
        height, width = self.img_size
        image = Image.open(image_path).convert("RGB")
        image = image.resize((width, height), Image.BILINEAR)
        return np.asarray(image, dtype=np.uint8)

    def _load_cached_image(self, index: int) -> Tensor:
        image = torch.from_numpy(np.array(self.images[index], copy=True)).float().div(255.0)
        return image.permute(2, 0, 1)

    def _load_labels(self, label_path: Path) -> tuple[Tensor, Tensor]:
        boxes = torch.zeros(self.max_boxes, 5, dtype=torch.float32)
        valid_mask = torch.zeros(self.max_boxes, dtype=torch.bool)

        rows = self._read_label_rows(label_path)
        for row_index, row in enumerate(rows[: self.max_boxes]):
            boxes[row_index] = torch.tensor(row, dtype=torch.float32)
            valid_mask[row_index] = True

        return boxes, valid_mask

    def _read_label_rows(self, label_path: Path) -> list[list[float]]:
        if not label_path.exists():
            return []

        rows: list[list[float]] = []
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
            coords = [float(value) for value in parts[1:]]
            if self.num_classes is not None and not 0 <= class_id < self.num_classes:
                raise ValueError(
                    f"class_id {class_id} is outside 0..{self.num_classes - 1} at "
                    f"{label_path}:{line_number}"
                )
            if any(value < 0.0 or value > 1.0 for value in coords):
                raise ValueError(
                    f"YOLO coordinates must be normalized to 0..1 at {label_path}:{line_number}"
                )
            rows.append([float(class_id), *coords])

        return rows


def tracker_collate_fn(batch: Sequence[dict[str, Tensor | str]]) -> dict[str, Tensor | list[str]]:
    """Collate dict samples produced by YOLOTrackerDataset."""
    return {
        "images": torch.stack([sample["image"] for sample in batch if isinstance(sample["image"], Tensor)]),
        "boxes": torch.stack([sample["boxes"] for sample in batch if isinstance(sample["boxes"], Tensor)]),
        "valid_mask": torch.stack(
            [sample["valid_mask"] for sample in batch if isinstance(sample["valid_mask"], Tensor)]
        ),
        "image_paths": [str(sample["image_path"]) for sample in batch],
    }


def create_tracker_dataset(
    root: str | Path | None,
    split: str,
    img_size: tuple[int, int],
    max_boxes: int,
    image_dir: str | Path | None = None,
    label_dir: str | Path | None = None,
    num_classes: int | None = None,
    cache_dir: str | Path = "data",
) -> YOLOTrackerDataset:
    """Small factory used by training scripts."""
    return YOLOTrackerDataset(
        root=root,
        split=split,
        img_size=img_size,
        max_boxes=max_boxes,
        image_dir=image_dir,
        label_dir=label_dir,
        num_classes=num_classes,
        cache_dir=cache_dir,
    )


def _parse_simple_yaml(data_yaml: Path) -> dict[str, str]:
    config: dict[str, str] = {}
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        config[key.strip()] = value.strip()
    return config


def _require_key(config: dict[str, str], key: str) -> str:
    value = config.get(key)
    if not value:
        raise ValueError(f"Missing required key in data.yaml: {key}")
    return value


def _resolve_path(base: Path, value: str) -> Path:
    value = _strip_quotes(value)
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _resolve_optional_path(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    return _resolve_path(base, value)


def _parse_class_names(value: str | None) -> list[str]:
    if not value:
        return []
    parsed = ast.literal_eval(value)
    if isinstance(parsed, dict):
        return [str(parsed[index]) for index in sorted(parsed)]
    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed]
    raise ValueError("names in data.yaml must be a list or dict.")


def _strip_quotes(value: str) -> str:
    return value.strip().strip("'").strip('"')
