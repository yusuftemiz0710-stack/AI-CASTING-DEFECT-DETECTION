"""Prepare manifest and stratified splits for casting image data."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .config import get_split_ratios, load_config
from .paths import EXTERNAL_DIR, RAW_DIR, SPLITS_DIR, TABLES_DIR, ensure_project_dirs
from .utils import (
    SUPPORTED_EXTENSIONS,
    class_to_label,
    safe_relative,
    setup_logger,
    sha256_file,
    validate_image,
)


def find_class_image_paths(class_names: list[str]) -> list[tuple[Path, str]]:
    """Find image paths belonging to class directories named in config."""
    class_dirs: list[tuple[Path, str]] = []
    for root in [RAW_DIR, EXTERNAL_DIR]:
        if not root.exists():
            continue
        for class_name in class_names:
            for class_dir in root.rglob(class_name):
                if not class_dir.is_dir():
                    continue
                if any(path.suffix.lower() in SUPPORTED_EXTENSIONS for path in class_dir.rglob("*")):
                    class_dirs.append((class_dir.resolve(), class_name))

    split_aware_dirs = [(path, class_name) for path, class_name in class_dirs if infer_original_split(path) in {"train", "val", "test"}]
    if split_aware_dirs:
        class_dirs = split_aware_dirs

    records: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for class_dir, class_name in class_dirs:
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                resolved = image_path.resolve()
                if resolved not in seen:
                    records.append((resolved, class_name))
                    seen.add(resolved)
    return records


def infer_original_split(path: Path) -> str:
    """Infer existing train/test/validation split from path parts."""
    parts = {part.lower() for part in path.parts}
    if "test" in parts:
        return "test"
    if "val" in parts or "valid" in parts or "validation" in parts:
        return "val"
    if "train" in parts:
        return "train"
    return "unknown"


def _stratified_split_hashes(
    hash_labels: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Split hash-level rows with stratification when feasible."""
    if len(hash_labels) < 4 or hash_labels["label"].nunique() < 2 or hash_labels["label"].value_counts().min() < 2:
        train_hashes, test_hashes = train_test_split(hash_labels["hash"], test_size=test_size, random_state=seed)
        return list(train_hashes), list(test_hashes)
    train_hashes, test_hashes = train_test_split(
        hash_labels["hash"],
        test_size=test_size,
        random_state=seed,
        stratify=hash_labels["label"],
    )
    return list(train_hashes), list(test_hashes)


def assign_splits(rows: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Assign train/val/test splits at hash-group level to prevent leakage."""
    seed = int(config["seed"])
    train_ratio, val_ratio, test_ratio = get_split_ratios(config)

    hash_labels = (
        rows.groupby("hash")
        .agg(label=("label", "first"), class_name=("class_name", "first"), original_split=("original_split", lambda values: sorted(set(values))))
        .reset_index()
    )
    hash_to_split: dict[str, str] = {}
    has_existing_test = hash_labels["original_split"].apply(lambda values: "test" in values).any()
    has_existing_train = hash_labels["original_split"].apply(lambda values: "train" in values).any()

    if has_existing_train and has_existing_test:
        test_hashes = hash_labels[hash_labels["original_split"].apply(lambda values: "test" in values)]["hash"].tolist()
        remaining = hash_labels[~hash_labels["hash"].isin(test_hashes)].copy()
        relative_val = val_ratio / max(train_ratio + val_ratio, 1e-8)
        train_hashes, val_hashes = _stratified_split_hashes(remaining, relative_val, seed)
    else:
        train_val_hashes, test_hashes = _stratified_split_hashes(hash_labels, test_ratio, seed)
        train_val = hash_labels[hash_labels["hash"].isin(train_val_hashes)].copy()
        relative_val = val_ratio / max(train_ratio + val_ratio, 1e-8)
        train_hashes, val_hashes = _stratified_split_hashes(train_val, relative_val, seed)

    for value in train_hashes:
        hash_to_split[value] = "train"
    for value in val_hashes:
        hash_to_split[value] = "val"
    for value in test_hashes:
        hash_to_split[value] = "test"

    rows = rows.copy()
    rows["split"] = rows["hash"].map(hash_to_split)
    missing = rows["split"].isna().sum()
    if missing:
        raise RuntimeError(f"Split assignment failed for {missing} rows")
    return rows


def build_manifest(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate images, compute hashes, and build the dataset manifest."""
    logger = setup_logger("prepare_dataset", "prepare_dataset_module.log")
    mapping = class_to_label(config)
    raw_records = find_class_image_paths(list(config["class_names"]))
    if not raw_records:
        raise FileNotFoundError("No def_front/ok_front images found under data/raw or data/external.")

    logger.info("Found %d candidate images.", len(raw_records))
    rows: list[dict] = []
    bad_rows: list[dict] = []
    for image_path, class_name in tqdm(raw_records, desc="Validating images"):
        try:
            width, height, mode = validate_image(image_path)
            digest = sha256_file(image_path)
            rows.append(
                {
                    "image_path": str(image_path),
                    "filename": image_path.name,
                    "class_name": class_name,
                    "label": mapping[class_name],
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "original_split": infer_original_split(image_path),
                    "hash": digest,
                }
            )
        except Exception as exc:
            bad_rows.append({"image_path": str(image_path), "class_name": class_name, "error": str(exc)})

    if bad_rows:
        bad_df = pd.DataFrame(bad_rows)
        bad_df.to_csv(TABLES_DIR / "bad_images.csv", index=False, encoding="utf-8")
        logger.warning("Skipped %d unreadable/invalid images. See outputs/tables/bad_images.csv", len(bad_rows))

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("All candidate images were invalid.")

    label_conflicts = manifest.groupby("hash")["label"].nunique()
    conflicting_hashes = set(label_conflicts[label_conflicts > 1].index)
    if conflicting_hashes:
        logger.warning("Dropping %d hash groups with conflicting class labels.", len(conflicting_hashes))
        manifest = manifest[~manifest["hash"].isin(conflicting_hashes)].copy()

    manifest = assign_splits(manifest, config)
    duplicate_report = (
        manifest.groupby("hash")
        .agg(
            duplicate_count=("image_path", "count"),
            class_names=("class_name", lambda values: ";".join(sorted(set(values)))),
            splits=("split", lambda values: ";".join(sorted(set(values)))),
            example_path=("image_path", "first"),
        )
        .reset_index()
        .sort_values("duplicate_count", ascending=False)
    )
    duplicate_report = duplicate_report[duplicate_report["duplicate_count"] > 1]
    return manifest, duplicate_report


def leakage_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return hash groups that appear in more than one split."""
    leakage = (
        manifest.groupby("hash")["split"]
        .agg(lambda values: ";".join(sorted(set(values))))
        .reset_index(name="splits")
    )
    return leakage[leakage["splits"].str.contains(";")].copy()


def main() -> int:
    """CLI entry point."""
    ensure_project_dirs()
    config = load_config()
    logger = setup_logger("prepare_dataset", "prepare_dataset_module.log")
    try:
        manifest, duplicate_report = build_manifest(config)
        leakage = leakage_summary(manifest)
        if not leakage.empty:
            leakage.to_csv(TABLES_DIR / "leakage_report.csv", index=False, encoding="utf-8")
            raise RuntimeError("Hash leakage detected across splits. See outputs/tables/leakage_report.csv")

        manifest_path = TABLES_DIR / "dataset_manifest.csv"
        manifest.to_csv(manifest_path, index=False, encoding="utf-8")
        duplicate_report.to_csv(TABLES_DIR / "duplicate_report.csv", index=False, encoding="utf-8")

        for split_name in ["train", "val", "test"]:
            split_df = manifest[manifest["split"] == split_name].copy()
            split_df.to_csv(SPLITS_DIR / f"{split_name}.csv", index=False, encoding="utf-8")

        counts = manifest.groupby(["split", "class_name"]).size().unstack(fill_value=0)
        logger.info("Prepared manifest: %s", manifest_path)
        logger.info("Split/class counts:\n%s", counts)
        return 0
    except Exception as exc:
        logger.exception("Dataset preparation failed: %s", exc)
        print(f"Dataset preparation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
