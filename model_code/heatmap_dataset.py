"""
Heatmap dataset generation and loading.

For each image, the label heatmap is 1 inside YOLO bbox regions where the
resized image pixels are black-to-gray. Masked-out pixels inside each bbox get
a Gaussian value based on distance to the nearest passing pixel. Generated
train/val/test arrays are cached under data/ as npy files and reused on later
runs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from tracker_dataset import IMAGE_EXTENSIONS, YOLODataConfig, labels_dir_from_images_dir, load_yolo_data_config


class HeatmapDataset(Dataset):
    """Dataset backed by generated image/heatmap npy files."""

    def __init__(
        self,
        data_yaml: str | Path,
        split: str,
        img_size: tuple[int, int] = (640, 640),
        cache_dir: str | Path = "data",
        gray_threshold: int = 50,
        gaussian_sigma: float = 8.0,
        max_fill_distance: float = 24.0,
        center_prior_sigma: float = 0.3,
    ) -> None:
        self.data_config = load_yolo_data_config(data_yaml)
        self.split = split
        self.img_size = img_size
        self.cache_dir = Path(cache_dir)
        self.gray_threshold = gray_threshold
        self.gaussian_sigma = gaussian_sigma
        self.max_fill_distance = max_fill_distance
        self.center_prior_sigma = center_prior_sigma
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir = self._split_image_dir()
        self.image_paths = self._find_image_paths(self.image_dir)

        height, width = self.img_size
        size_tag = f"{height}x{width}"
        cache_key = cache_key_for_data_yaml(data_yaml)
        self.images_path = self.cache_dir / f"heatmap_{cache_key}_{split}_images_{size_tag}.npy"
        self.labels_path = (
            self.cache_dir
            / (
                f"heatmap_{cache_key}_{split}_labels_{size_tag}"
                f"_gray{gray_threshold:g}_gaussian_s{gaussian_sigma:g}"
                f"_d{max_fill_distance:g}_c{center_prior_sigma:g}.npy"
            )
        )
        if not self.images_path.exists() or not self.labels_path.exists():
            self._generate_cache()

        self.images = np.load(self.images_path, mmap_mode="r")
        self.labels = np.load(self.labels_path, mmap_mode="r")

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        image = torch.from_numpy(np.array(self.images[index], copy=True)).float().div(255.0)
        label = torch.from_numpy(np.array(self.labels[index], copy=True)).float()
        return {
            "image": image.permute(2, 0, 1),
            "heatmap": label.unsqueeze(0),
        }

    def _generate_cache(self) -> None:
        label_dir = labels_dir_from_images_dir(self.image_dir)
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in: {self.image_dir}")

        height, width = self.img_size
        images = np.zeros((len(self.image_paths), height, width, 3), dtype=np.uint8)
        labels = np.zeros((len(self.image_paths), height, width), dtype=np.float32)

        for index, image_path in enumerate(self.image_paths):
            image = Image.open(image_path).convert("RGB").resize((width, height), Image.BILINEAR)
            image_array = np.asarray(image, dtype=np.uint8)
            heatmap = np.zeros((height, width), dtype=np.float32)

            gray_mask = self._gray_pixel_mask(image_array)
            for _, bbox in read_yolo_rows(label_dir / f"{image_path.stem}.txt"):
                x1, y1, x2, y2 = yolo_box_to_pixels(bbox, width, height)
                if x2 <= x1 or y2 <= y1:
                    continue
                bbox_mask = gray_mask[y1:y2, x1:x2].astype(bool)
                heatmap[y1:y2, x1:x2] = np.maximum(
                    heatmap[y1:y2, x1:x2],
                    gaussian_fill_from_mask(
                        bbox_mask,
                        sigma=self.gaussian_sigma,
                        max_distance=self.max_fill_distance,
                        center_prior_sigma=self.center_prior_sigma,
                    ),
                )

            images[index] = image_array
            labels[index] = heatmap

        np.save(self.images_path, images)
        np.save(self.labels_path, labels)

    def _split_image_dir(self) -> Path:
        if self.split == "train":
            return self.data_config.train_images
        if self.split == "val":
            if self.data_config.val_images is None:
                raise ValueError("data.yaml does not define val split.")
            return self.data_config.val_images
        if self.data_config.test_images is None:
            raise ValueError("data.yaml does not define test split.")
        return self.data_config.test_images

    def _find_image_paths(self, image_dir: Path) -> list[Path]:
        return sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def _gray_pixel_mask(self, image_array: np.ndarray) -> np.ndarray:
        max_channel = image_array.max(axis=-1)
        min_channel = image_array.min(axis=-1)
        low_saturation = (max_channel.astype(np.int16) - min_channel.astype(np.int16)) <= 50
        dark_to_gray = max_channel <= self.gray_threshold
        return (low_saturation & dark_to_gray).astype(np.uint8)


def read_yolo_rows(label_path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    if not label_path.exists():
        return []

    rows: list[tuple[int, tuple[float, float, float, float]]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO row in {label_path}: {line}")
        class_id = int(float(parts[0]))
        bbox = tuple(float(value) for value in parts[1:])
        rows.append((class_id, bbox))
    return rows


def cache_key_for_data_yaml(data_yaml: str | Path) -> str:
    data_path = Path(data_yaml).resolve()
    digest = hashlib.sha1(str(data_path).encode("utf-8")).hexdigest()[:10]
    return f"{data_path.stem}_{digest}"


def yolo_box_to_pixels(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x_center, y_center, box_width, box_height = bbox
    x1 = int(max(0, (x_center - box_width / 2.0) * width))
    y1 = int(max(0, (y_center - box_height / 2.0) * height))
    x2 = int(min(width, (x_center + box_width / 2.0) * width))
    y2 = int(min(height, (y_center + box_height / 2.0) * height))
    return x1, y1, x2, y2


def heatmap_collate_fn(batch: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
    return {
        "images": torch.stack([sample["image"] for sample in batch]),
        "heatmaps": torch.stack([sample["heatmap"] for sample in batch]),
    }


def create_heatmap_dataset(
    data_yaml: str | Path,
    split: str,
    img_size: tuple[int, int],
    cache_dir: str | Path = "data",
    gray_threshold: int = 150,
    gaussian_sigma: float = 8.0,
    max_fill_distance: float = 24.0,
    center_prior_sigma: float = 0.45,
) -> HeatmapDataset:
    return HeatmapDataset(
        data_yaml=data_yaml,
        split=split,
        img_size=img_size,
        cache_dir=cache_dir,
        gray_threshold=gray_threshold,
        gaussian_sigma=gaussian_sigma,
        max_fill_distance=max_fill_distance,
        center_prior_sigma=center_prior_sigma,
    )


def gaussian_fill_from_mask(
    mask: np.ndarray,
    sigma: float,
    max_distance: float,
    center_prior_sigma: float,
) -> np.ndarray:
    if mask.size == 0:
        return mask.astype(np.float32)
    if mask.any():
        distance = distance_to_nearest_true(mask)
        values = np.exp(-0.5 * (distance / max(sigma, 1e-6)) ** 2)
        values[distance > max_distance] = 0.0
        values *= bbox_center_prior(mask.shape, center_prior_sigma)
        values[mask] = 1.0
        return values.astype(np.float32)
    return np.zeros(mask.shape, dtype=np.float32)


def bbox_center_prior(shape: tuple[int, int], sigma: float) -> np.ndarray:
    height, width = shape
    if height <= 0 or width <= 0:
        return np.zeros(shape, dtype=np.float32)

    y_coords, x_coords = np.indices((height, width), dtype=np.float32)
    x_center = (width - 1) / 2.0
    y_center = (height - 1) / 2.0
    x_scale = max(width * sigma, 1e-6)
    y_scale = max(height * sigma, 1e-6)
    distance_sq = ((x_coords - x_center) / x_scale) ** 2 + ((y_coords - y_center) / y_scale) ** 2
    return np.exp(-0.5 * distance_sq).astype(np.float32)


def distance_to_nearest_true(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import distance_transform_edt

        return distance_transform_edt(~mask)
    except ImportError:
        return distance_to_nearest_true_fallback(mask)


def distance_to_nearest_true_fallback(mask: np.ndarray) -> np.ndarray:
    true_points = np.argwhere(mask)
    if true_points.size == 0:
        return np.full(mask.shape, np.inf, dtype=np.float32)

    height, width = mask.shape
    y_coords, x_coords = np.indices((height, width))
    min_distance_sq = np.full(mask.shape, np.inf, dtype=np.float32)
    for true_y, true_x in true_points:
        distance_sq = (y_coords - true_y) ** 2 + (x_coords - true_x) ** 2
        min_distance_sq = np.minimum(min_distance_sq, distance_sq)
    return np.sqrt(min_distance_sq).astype(np.float32)
