"""Train a MobileNetV2 transfer-learning model."""

from __future__ import annotations

import sys

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .paths import FIGURES_DIR, MODELS_DIR, TABLES_DIR, ensure_project_dirs
from .train_cnn import class_weights, get_device, make_loaders, run_epoch
from .utils import (
    build_transfer_model,
    compute_binary_metrics,
    load_manifest,
    plot_learning_curve,
    save_metrics_csv,
    save_torch_checkpoint,
    set_seed,
    setup_logger,
)


def train_transfer(config: dict, manifest: pd.DataFrame) -> dict:
    """Train a frozen-backbone MobileNetV2 classifier head."""
    set_seed(int(config["seed"]))
    logger = setup_logger("train_transfer", "train_transfer_module.log")
    device = get_device(config)
    train_loader, val_loader = make_loaders(manifest, config)

    pretrained_used = True
    try:
        model = build_transfer_model(config.get("transfer_model", "mobilenet_v2"), pretrained=True, num_classes=2)
    except Exception as exc:
        pretrained_used = False
        logger.warning("ImageNet pretrained weights could not be loaded; falling back to random weights. Error: %s", exc)
        model = build_transfer_model(config.get("transfer_model", "mobilenet_v2"), pretrained=False, num_classes=2)

    model = model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    criterion = nn.CrossEntropyLoss(weight=class_weights(manifest, device))
    optimizer = torch.optim.Adam(trainable, lr=float(config["learning_rate"]))
    best_metric = -1.0
    best_metrics: dict = {}
    best_state = None
    patience = int(config.get("early_stopping_patience", 4))
    stale_epochs = 0
    history: list[dict] = []

    for epoch in range(1, int(config["epochs_transfer"]) + 1):
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
            best_metrics = {
                **val_metrics,
                "model": "transfer_mobilenet_v2",
                "split": "val",
                "epoch": epoch,
                "device": str(device),
                "pretrained_used": pretrained_used,
            }
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("Transfer model did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    save_torch_checkpoint(
        MODELS_DIR / "transfer_best.pt",
        model,
        "transfer_mobilenet_v2",
        config,
        best_metrics,
        extra={"pretrained_used": pretrained_used},
    )
    history_df = pd.DataFrame(history)
    history_df.to_csv(TABLES_DIR / "transfer_history.csv", index=False, encoding="utf-8")
    plot_learning_curve(history_df, FIGURES_DIR / "transfer_learning_curve.png", "Transfer learning öğrenme eğrileri")
    save_metrics_csv(TABLES_DIR / "transfer_metrics.csv", best_metrics)
    return best_metrics


def main() -> int:
    """CLI entry point."""
    ensure_project_dirs()
    logger = setup_logger("train_transfer", "train_transfer_module.log")
    try:
        config = load_config()
        manifest = load_manifest()
        metrics = train_transfer(config, manifest)
        logger.info("Transfer metrics: %s", metrics)
        return 0
    except Exception as exc:
        logger.exception("Transfer learning failed: %s", exc)
        print(f"Transfer learning failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
