"""Configuration loading for the casting defect project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .paths import PROJECT_ROOT, ensure_project_dirs


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration and normalize project-relative paths."""
    ensure_project_dirs()
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    paths = config.get("paths", {})
    for key, value in list(paths.items()):
        paths[key] = str((PROJECT_ROOT / value).resolve()) if not Path(value).is_absolute() else str(Path(value))
    config["paths"] = paths
    return config


def get_split_ratios(config: dict[str, Any]) -> tuple[float, float, float]:
    """Return train/validation/test split ratios from config."""
    split = config["train_val_test_split"]
    train = float(split["train"])
    val = float(split["val"])
    test = float(split["test"])
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")
    return train, val, test
