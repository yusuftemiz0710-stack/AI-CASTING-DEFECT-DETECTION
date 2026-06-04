import pandas as pd

from src.paths import TABLES_DIR
from src.utils import load_manifest


def test_manifest_exists_and_has_classes():
    manifest_path = TABLES_DIR / "dataset_manifest.csv"
    assert manifest_path.exists(), "dataset_manifest.csv is missing"
    manifest = load_manifest(manifest_path)
    assert {"def_front", "ok_front"}.issubset(set(manifest["class_name"]))
    counts = manifest["class_name"].value_counts().to_dict()
    assert counts.get("def_front", 0) > 0
    assert counts.get("ok_front", 0) > 0


def test_no_hash_leakage_between_splits():
    manifest = load_manifest()
    assert {"train", "val", "test"}.issubset(set(manifest["split"]))
    split_counts = manifest.groupby("hash")["split"].nunique()
    leaking = split_counts[split_counts > 1]
    assert leaking.empty, f"Hash leakage detected: {leaking.head().to_dict()}"
