"""Evaluate the selected deep model on the held-out test set."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .paths import FIGURES_DIR, MODELS_DIR, PREDICTIONS_DIR, TABLES_DIR, ensure_project_dirs
from .utils import (
    ManifestImageDataset,
    compute_binary_metrics,
    copy_file,
    label_to_class,
    load_manifest,
    save_image_grid,
    save_metrics_csv,
    setup_logger,
    torch_load_checkpoint,
    write_json,
)


def _read_metric(path: Path, model_path: Path | None) -> dict | None:
    """Read a metrics CSV and attach model path metadata."""
    if not path.exists():
        return None
    row = pd.read_csv(path).iloc[0].to_dict()
    if model_path is not None:
        row["model_path"] = str(model_path)
    return row


def select_best_model(config: dict) -> tuple[Path, pd.DataFrame]:
    """Select the best deployable PyTorch model by configured validation metric."""
    metric_name = config.get("model_selection_metric", "f1_defect")
    tolerance = float(config.get("selection_tolerance", 0.0))
    prefer_transfer = bool(config.get("prefer_transfer_if_within_tolerance", False))
    rows: list[dict] = []
    for metrics_path, model_path in [
        (TABLES_DIR / "baseline_metrics.csv", None),
        (TABLES_DIR / "custom_cnn_metrics.csv", MODELS_DIR / "custom_cnn.pt"),
        (TABLES_DIR / "transfer_metrics.csv", MODELS_DIR / "transfer_best.pt"),
    ]:
        row = _read_metric(metrics_path, model_path)
        if row:
            rows.append(row)
    if not rows:
        raise FileNotFoundError("No model metrics found. Train models before evaluation.")

    comparison = pd.DataFrame(rows)
    comparison.to_csv(TABLES_DIR / "model_comparison.csv", index=False, encoding="utf-8")
    deployable = comparison[comparison["model_path"].notna()].copy()
    if deployable.empty:
        raise FileNotFoundError("No deployable PyTorch model found for best_model.pt.")
    deployable[metric_name] = pd.to_numeric(deployable[metric_name], errors="coerce")
    sorted_models = deployable.sort_values(metric_name, ascending=False)
    top_score = float(sorted_models.iloc[0][metric_name])
    eligible = sorted_models[sorted_models[metric_name] >= top_score - tolerance].copy()
    if prefer_transfer:
        transfer_rows = eligible[eligible["model"].astype(str).str.contains("transfer", case=False, na=False)]
        best = transfer_rows.iloc[0] if not transfer_rows.empty else sorted_models.iloc[0]
    else:
        best = sorted_models.iloc[0]
    selected_path = Path(best["model_path"])
    best_path = MODELS_DIR / "best_model.pt"
    copy_file(selected_path, best_path)
    write_json(
        MODELS_DIR / "best_model_info.json",
        {
            "selected_model": str(best.get("model")),
            "source_path": str(selected_path),
            "best_model_path": str(best_path),
            "selection_metric": metric_name,
            "selection_metric_value": float(best[metric_name]),
            "selection_tolerance": tolerance,
            "prefer_transfer_if_within_tolerance": prefer_transfer,
        },
    )
    return best_path, comparison


def predict_dataframe(model: torch.nn.Module, manifest: pd.DataFrame, config: dict, device: torch.device) -> pd.DataFrame:
    """Run model inference for a manifest subset."""
    dataset = ManifestImageDataset(manifest, int(config["image_size"]), train=False).unwrap()
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config.get("num_workers", 0)))
    model.to(device)
    model.eval()
    y_true: list[int] = []
    y_prob: list[float] = []
    paths: list[str] = []
    with torch.no_grad():
        for inputs, labels, batch_paths in tqdm(loader, desc="test inference"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            prob = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            y_prob.extend(prob.tolist())
            y_true.extend(labels.numpy().astype(int).tolist())
            paths.extend(list(batch_paths))
    threshold = float(config.get("threshold", 0.5))
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    label_names = label_to_class()
    result = manifest.reset_index(drop=True).copy()
    result["true_label"] = y_true
    result["true_class"] = [label_names[int(value)] for value in y_true]
    result["defect_probability"] = y_prob
    result["ok_probability"] = 1 - np.asarray(y_prob)
    result["predicted_label"] = y_pred
    result["predicted_class"] = [label_names[int(value)] for value in y_pred]
    result["threshold"] = threshold
    result["correct"] = result["true_label"] == result["predicted_label"]
    return result


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Save confusion matrix figure with defect-oriented labels."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_xticks([0, 1], ["def_front", "ok_front"])
    ax.set_yticks([0, 1], ["def_front", "ok_front"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def plot_curves(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    """Save ROC and precision-recall curves."""
    if len(np.unique(y_true)) < 2:
        for name, title in [("roc_curve.png", "ROC curve"), ("pr_curve.png", "Precision-Recall curve")]:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "Test setinde tek sınıf olduğu için eğri çizilemedi.", ha="center", va="center")
            ax.axis("off")
            ax.set_title(title)
            fig.savefig(FIGURES_DIR / name, dpi=160)
            plt.close(fig)
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Rastgele")
    ax.set_title("ROC eğrisi")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate / defect recall")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=160)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label="Model")
    ax.set_title("Precision-Recall eğrisi")
    ax.set_xlabel("Recall defect")
    ax.set_ylabel("Precision defect")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pr_curve.png", dpi=160)
    plt.close(fig)


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """Compute and plot threshold sweep metrics."""
    rows = []
    for threshold in np.round(np.linspace(0.05, 0.95, 19), 2):
        row = compute_binary_metrics(y_true, y_prob, threshold=float(threshold))
        rows.append(row)
    sweep = pd.DataFrame(rows)
    sweep.to_csv(TABLES_DIR / "threshold_sweep.csv", index=False, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep["threshold"], sweep["precision_defect"], marker="o", label="Precision defect")
    ax.plot(sweep["threshold"], sweep["recall_defect"], marker="o", label="Recall defect")
    ax.plot(sweep["threshold"], sweep["f1_defect"], marker="o", label="F1 defect")
    ax.set_title("Threshold analizi")
    ax.set_xlabel("Defect probability threshold")
    ax.set_ylabel("Skor")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "threshold_sweep.png", dpi=160)
    plt.close(fig)
    return sweep


def save_prediction_tables(predictions: pd.DataFrame) -> None:
    """Save prediction and misclassification tables."""
    columns = [
        "image_path",
        "filename",
        "true_class",
        "predicted_class",
        "defect_probability",
        "ok_probability",
        "threshold",
        "correct",
    ]
    predictions[columns].to_csv(PREDICTIONS_DIR / "test_predictions.csv", index=False, encoding="utf-8")
    misclassified = predictions[~predictions["correct"]].copy()
    misclassified[columns].to_csv(PREDICTIONS_DIR / "misclassified_examples.csv", index=False, encoding="utf-8")
    subtitles = [
        f"G:{row.true_class} T:{row.predicted_class} p_def={row.defect_probability:.2f}"
        for row in misclassified.head(12).itertuples()
    ]
    save_image_grid(
        misclassified["image_path"].head(12).tolist(),
        FIGURES_DIR / "misclassified_grid.png",
        "Yanlış sınıflandırılan örnekler",
        subtitles=subtitles,
        max_images=12,
    )


def evaluate() -> dict:
    """Select and evaluate the best model."""
    config = load_config()
    best_path, _comparison = select_best_model(config)
    device = torch.device("cuda" if bool(config.get("use_gpu_if_available", True)) and torch.cuda.is_available() else "cpu")
    model, checkpoint = torch_load_checkpoint(best_path, map_location=str(device))
    manifest = load_manifest()
    test_df = manifest[manifest["split"] == "test"].reset_index(drop=True)
    if test_df.empty:
        raise RuntimeError("Test split is empty.")
    predictions = predict_dataframe(model, test_df, config, device)
    y_true = predictions["true_label"].astype(int).to_numpy()
    y_prob = predictions["defect_probability"].astype(float).to_numpy()
    y_pred = predictions["predicted_label"].astype(int).to_numpy()
    metrics = compute_binary_metrics(y_true, y_prob, threshold=float(config.get("threshold", 0.5)))
    metrics.update({"model": checkpoint.get("model_type", "unknown"), "split": "test", "model_path": str(best_path)})
    save_metrics_csv(TABLES_DIR / "final_test_metrics.csv", metrics)

    report = classification_report(
        y_true,
        y_pred,
        labels=[1, 0],
        target_names=["def_front", "ok_front"],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(TABLES_DIR / "classification_report.csv", encoding="utf-8")
    plot_confusion(y_true, y_pred)
    plot_curves(y_true, y_prob)
    threshold_sweep(y_true, y_prob)
    save_prediction_tables(predictions)
    return metrics


def main() -> int:
    """CLI entry point."""
    ensure_project_dirs()
    logger = setup_logger("evaluate", "evaluate_module.log")
    try:
        metrics = evaluate()
        logger.info("Final test metrics: %s", metrics)
        return 0
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
