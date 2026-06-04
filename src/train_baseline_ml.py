"""Train a classical HOG + Logistic Regression baseline model."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .config import load_config
from .paths import MODELS_DIR, TABLES_DIR, ensure_project_dirs
from .utils import compute_binary_metrics, load_manifest, save_metrics_csv, setup_logger


def extract_hog_features(image_paths: list[str], size: int = 128) -> np.ndarray:
    """Extract HOG features from image paths."""
    features: list[np.ndarray] = []
    for path in tqdm(image_paths, desc="HOG feature extraction"):
        image = Image.open(path).convert("L").resize((size, size))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        feature = hog(
            arr,
            orientations=9,
            pixels_per_cell=(12, 12),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        features.append(feature.astype(np.float32))
    return np.vstack(features)


def train_baseline(manifest: pd.DataFrame, config: dict) -> dict:
    """Train and validate the baseline model."""
    train_df = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_df = manifest[manifest["split"] == "val"].reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise RuntimeError("Train/validation split is empty.")

    x_train = extract_hog_features(train_df["image_path"].tolist(), size=128)
    y_train = train_df["label"].astype(int).to_numpy()
    x_val = extract_hog_features(val_df["image_path"].tolist(), size=128)
    y_val = val_df["label"].astype(int).to_numpy()

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=int(config["seed"]),
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    val_prob = model.predict_proba(x_val)[:, 1]
    metrics = compute_binary_metrics(y_val, val_prob, threshold=float(config.get("threshold", 0.5)))
    metrics.update({"model": "baseline_hog_logreg", "split": "val", "feature_size": 128})

    model_payload = {
        "model": model,
        "feature_type": "hog",
        "feature_size": 128,
        "class_names": ["ok_front", "def_front"],
        "metrics": metrics,
    }
    joblib.dump(model_payload, MODELS_DIR / "baseline_ml.joblib")
    save_metrics_csv(TABLES_DIR / "baseline_metrics.csv", metrics)
    return metrics


def main() -> int:
    """CLI entry point."""
    ensure_project_dirs()
    logger = setup_logger("train_baseline_ml", "train_baseline_ml_module.log")
    try:
        config = load_config()
        manifest = load_manifest()
        metrics = train_baseline(manifest, config)
        logger.info("Baseline metrics: %s", metrics)
        return 0
    except Exception as exc:
        logger.exception("Baseline training failed: %s", exc)
        print(f"Baseline training failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
