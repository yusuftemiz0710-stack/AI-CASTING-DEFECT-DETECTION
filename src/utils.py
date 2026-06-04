"""Shared utilities for data, training, inference, and plotting."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .paths import FIGURES_DIR, LOGS_DIR, PROJECT_ROOT, TABLES_DIR, ensure_project_dirs, resolve_project_path


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MODEL_CLASS_NAMES = ["ok_front", "def_front"]


def set_seed(seed: int) -> None:
    """Set random seeds for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def setup_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    """Create a file and console logger without duplicating handlers."""
    ensure_project_dirs()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = LOGS_DIR / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write UTF-8 JSON with stable indentation."""
    output_path = resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a UTF-8 JSON file."""
    with resolve_project_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: str | Path) -> str:
    """Return SHA-256 hash for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_image_files(root: str | Path) -> list[Path]:
    """Recursively list supported image files under a directory."""
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(path for path in root_path.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def safe_relative(path: str | Path, start: str | Path = PROJECT_ROOT) -> str:
    """Return a project-relative string when possible."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path(start).resolve()))
    except ValueError:
        return str(path)


def load_manifest(path: str | Path | None = None) -> pd.DataFrame:
    """Load the dataset manifest table."""
    manifest_path = Path(path) if path else TABLES_DIR / "dataset_manifest.csv"
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    return pd.read_csv(manifest_path)


def class_to_label(config: dict[str, Any]) -> dict[str, int]:
    """Return binary labels with the defect class as positive label 1."""
    positive = config["defect_positive_class"]
    mapping: dict[str, int] = {}
    for name in config["class_names"]:
        mapping[name] = 1 if name == positive else 0
    return mapping


def label_to_class() -> dict[int, str]:
    """Return the model output label-to-class mapping."""
    return {0: "ok_front", 1: "def_front"}


def validate_image(path: str | Path) -> tuple[int, int, str]:
    """Open an image and return width, height, and PIL mode."""
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
    if width <= 0 or height <= 0:
        raise ValueError("image has invalid size")
    return width, height, mode


def pil_load_rgb(path: str | Path) -> Image.Image:
    """Load an image as RGB with EXIF orientation handled."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")


def compute_binary_metrics(y_true: Iterable[int], y_prob_defect: Iterable[float], threshold: float = 0.5) -> dict[str, float]:
    """Compute defect-oriented binary classification metrics."""
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_prob_arr = np.asarray(list(y_prob_defect), dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)
    labels = [0, 1]
    cm = confusion_matrix(y_true_arr, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    def safe_metric(func: Any, default: float = float("nan")) -> float:
        try:
            return float(func())
        except Exception:
            return default

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision_defect": float(precision_score(y_true_arr, y_pred, pos_label=1, zero_division=0)),
        "recall_defect": float(recall_score(y_true_arr, y_pred, pos_label=1, zero_division=0)),
        "f1_defect": float(f1_score(y_true_arr, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": safe_metric(lambda: roc_auc_score(y_true_arr, y_prob_arr)),
        "pr_auc": safe_metric(lambda: average_precision_score(y_true_arr, y_prob_arr)),
        "false_negative": int(fn),
        "false_positive": int(fp),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "support": int(len(y_true_arr)),
    }
    return metrics


def save_metrics_csv(path: str | Path, metrics: dict[str, Any]) -> None:
    """Save one-row metrics dictionary as CSV."""
    output_path = resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(output_path, index=False, encoding="utf-8")


def plot_learning_curve(history: pd.DataFrame, output_path: str | Path, title: str) -> None:
    """Save a two-panel learning curve figure."""
    output = resolve_project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["epoch"], history["train_accuracy"], label="Train accuracy")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="Validation accuracy")
    axes[1].plot(history["epoch"], history["val_f1_defect"], label="Validation F1 defect")
    axes[1].set_title("Accuracy / F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_image_grid(
    image_paths: list[str | Path],
    output_path: str | Path,
    title: str,
    subtitles: list[str] | None = None,
    max_images: int = 12,
) -> None:
    """Save a grid of image examples."""
    output = resolve_project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = image_paths[:max_images]
    if not selected:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "Uygun örnek bulunamadı.", ha="center", va="center")
        ax.axis("off")
        fig.suptitle(title)
        fig.savefig(output, dpi=160)
        plt.close(fig)
        return

    cols = min(4, len(selected))
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes_arr = np.atleast_1d(axes).ravel()
    for idx, (path, ax) in enumerate(zip(selected, axes_arr)):
        image = pil_load_rgb(path)
        ax.imshow(image)
        ax.axis("off")
        if subtitles:
            ax.set_title(subtitles[idx], fontsize=8)
    for ax in axes_arr[len(selected) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def copy_file(src: str | Path, dst: str | Path) -> None:
    """Copy a file while creating the destination directory."""
    src_path = resolve_project_path(src)
    dst_path = resolve_project_path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


class SmallCNN:
    """Factory namespace for the custom CNN model."""

    @staticmethod
    def build(num_classes: int = 2):
        """Build a compact CNN suitable for CPU training."""
        import torch.nn as nn

        return nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes),
        )


def build_transfer_model(model_name: str = "mobilenet_v2", pretrained: bool = True, num_classes: int = 2):
    """Build a lightweight transfer-learning model."""
    os.environ.setdefault("TORCH_HOME", str((PROJECT_ROOT / "data" / "raw" / "torch_cache").resolve()))
    import torch.nn as nn
    from torchvision import models

    if model_name != "mobilenet_v2":
        raise ValueError(f"Unsupported transfer model: {model_name}")
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.25), nn.Linear(in_features, num_classes))
    return model


def build_transforms(image_size: int, train: bool = False):
    """Build torchvision transforms for training or evaluation."""
    from torchvision import transforms

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size + 24, image_size + 24)),
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8),
                transforms.ColorJitter(brightness=0.12, contrast=0.12),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


@dataclass
class TorchDatasetConfig:
    """Small holder for manifest dataset settings."""

    manifest: pd.DataFrame
    image_size: int
    train: bool


class ManifestImageDataset:
    """PyTorch dataset backed by the manifest CSV."""

    def __init__(self, manifest: pd.DataFrame, image_size: int, train: bool = False):
        from torch.utils.data import Dataset

        class _Dataset(Dataset):
            def __init__(self, rows: pd.DataFrame, size: int, is_train: bool):
                self.rows = rows.reset_index(drop=True)
                self.transform = build_transforms(size, train=is_train)

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, index: int):
                row = self.rows.iloc[index]
                image = pil_load_rgb(row["image_path"])
                tensor = self.transform(image)
                return tensor, int(row["label"]), row["image_path"]

        self.dataset = _Dataset(manifest, image_size, train)

    def unwrap(self):
        """Return the actual torch Dataset instance."""
        return self.dataset


def save_torch_checkpoint(
    path: str | Path,
    model: Any,
    model_type: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a model checkpoint with enough metadata for inference."""
    import torch

    output = resolve_project_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_type": model_type,
        "state_dict": model.state_dict(),
        "config": {
            "image_size": int(config["image_size"]),
            "class_names": MODEL_CLASS_NAMES,
            "defect_positive_class": config["defect_positive_class"],
            "threshold": float(config.get("threshold", 0.5)),
            "transfer_model": config.get("transfer_model", "mobilenet_v2"),
        },
        "metrics": metrics,
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, output)


def torch_load_checkpoint(path: str | Path, map_location: str = "cpu"):
    """Load a saved PyTorch checkpoint and rebuild the model."""
    import torch

    checkpoint_path = resolve_project_path(path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model_type = checkpoint["model_type"]
    config = checkpoint.get("config", {})
    if model_type == "custom_cnn":
        model = SmallCNN.build(num_classes=2)
    elif model_type == "transfer_mobilenet_v2":
        model = build_transfer_model("mobilenet_v2", pretrained=False, num_classes=2)
    else:
        raise ValueError(f"Unsupported checkpoint model type: {model_type}")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
