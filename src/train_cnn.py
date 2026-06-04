"""Train a compact custom CNN for casting defect classification."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .paths import FIGURES_DIR, MODELS_DIR, TABLES_DIR, ensure_project_dirs
from .utils import (
    ManifestImageDataset,
    SmallCNN,
    compute_binary_metrics,
    load_manifest,
    plot_learning_curve,
    save_metrics_csv,
    save_torch_checkpoint,
    set_seed,
    setup_logger,
)


def get_device(config: dict) -> torch.device:
    """Choose GPU when requested and available, otherwise CPU."""
    if bool(config.get("use_gpu_if_available", True)) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loaders(manifest: pd.DataFrame, config: dict) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    train_df = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_df = manifest[manifest["split"] == "val"].reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise RuntimeError("Train/validation split is empty.")
    train_ds = ManifestImageDataset(train_df, int(config["image_size"]), train=True).unwrap()
    val_ds = ManifestImageDataset(val_df, int(config["image_size"]), train=False).unwrap()
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def class_weights(manifest: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """Return inverse-frequency class weights for labels 0 and 1."""
    counts = manifest[manifest["split"] == "train"]["label"].value_counts().to_dict()
    total = sum(counts.values())
    weights = [total / (2 * max(counts.get(label, 1), 1)) for label in [0, 1]]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict]:
    """Run one train or validation epoch."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    y_true: list[int] = []
    y_prob: list[float] = []
    with torch.set_grad_enabled(is_train):
        for inputs, labels, _paths in tqdm(loader, desc="train" if is_train else "val", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach().cpu()) * inputs.size(0)
            prob = torch.softmax(outputs.detach(), dim=1)[:, 1].cpu().numpy()
            y_prob.extend(prob.tolist())
            y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
    avg_loss = total_loss / max(len(loader.dataset), 1)
    metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    return avg_loss, metrics


def train_custom_cnn(config: dict, manifest: pd.DataFrame) -> dict:
    """Train the custom CNN and save the best checkpoint by validation F1."""
    set_seed(int(config["seed"]))
    device = get_device(config)
    train_loader, val_loader = make_loaders(manifest, config)
    model = SmallCNN.build(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(manifest, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    best_metric = -1.0
    best_metrics: dict = {}
    best_state = None
    patience = int(config.get("early_stopping_patience", 4))
    stale_epochs = 0
    history: list[dict] = []

    for epoch in range(1, int(config["epochs_cnn"]) + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_defect": val_metrics["f1_defect"],
            "val_recall_defect": val_metrics["recall_defect"],
        }
        history.append(row)
        current = float(val_metrics[config.get("model_selection_metric", "f1_defect")])
        if current > best_metric:
            best_metric = current
            best_metrics = {**val_metrics, "model": "custom_cnn", "split": "val", "epoch": epoch, "device": str(device)}
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("Custom CNN did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    save_torch_checkpoint(MODELS_DIR / "custom_cnn.pt", model, "custom_cnn", config, best_metrics)
    history_df = pd.DataFrame(history)
    history_df.to_csv(TABLES_DIR / "custom_cnn_history.csv", index=False, encoding="utf-8")
    plot_learning_curve(history_df, FIGURES_DIR / "custom_cnn_learning_curve.png", "Custom CNN öğrenme eğrileri")
    save_metrics_csv(TABLES_DIR / "custom_cnn_metrics.csv", best_metrics)
    return best_metrics


def main() -> int:
    """CLI entry point."""
    ensure_project_dirs()
    logger = setup_logger("train_cnn", "train_cnn_module.log")
    try:
        config = load_config()
        manifest = load_manifest()
        metrics = train_custom_cnn(config, manifest)
        logger.info("Custom CNN metrics: %s", metrics)
        return 0
    except Exception as exc:
        logger.exception("Custom CNN training failed: %s", exc)
        print(f"Custom CNN training failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
