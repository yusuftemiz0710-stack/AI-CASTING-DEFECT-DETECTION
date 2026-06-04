"""Download or verify the Kaggle casting product image dataset."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from .config import load_config
from .paths import EXTERNAL_DIR, RAW_DIR, ensure_project_dirs
from .utils import SUPPORTED_EXTENSIONS, setup_logger


DATASET_SLUG = "ravirajsinh45/real-life-industrial-dataset-of-casting-product"


def _class_dirs_exist(root: Path, class_names: list[str]) -> bool:
    """Return True if all class directories with images exist somewhere under root."""
    found: set[str] = set()
    for class_name in class_names:
        for candidate in root.rglob(class_name):
            if candidate.is_dir() and any(path.suffix.lower() in SUPPORTED_EXTENSIONS for path in candidate.rglob("*")):
                found.add(class_name)
                break
    return set(class_names).issubset(found)


def dataset_available(class_names: list[str]) -> bool:
    """Check whether the required dataset classes are already physically present."""
    return _class_dirs_exist(RAW_DIR, class_names) or _class_dirs_exist(EXTERNAL_DIR, class_names)


def _copy_download_if_outside_project(downloaded_path: Path, logger) -> None:
    """Copy a downloaded KaggleHub folder into data/raw if it is outside the workspace."""
    downloaded_path = downloaded_path.resolve()
    project_raw = RAW_DIR.resolve()
    try:
        downloaded_path.relative_to(project_raw)
        logger.info("KaggleHub cache is already under data/raw: %s", downloaded_path)
        return
    except ValueError:
        pass

    target = RAW_DIR / "kagglehub_download"
    if target.exists():
        logger.info("Existing copied KaggleHub data found: %s", target)
        return
    logger.info("Copying KaggleHub download into project data/raw: %s", target)
    shutil.copytree(downloaded_path, target)


def download_with_kagglehub(logger) -> bool:
    """Try downloading the public dataset with kagglehub."""
    os.environ.setdefault("KAGGLEHUB_CACHE", str(RAW_DIR / "kagglehub_cache"))
    try:
        import kagglehub
    except Exception as exc:
        logger.warning("kagglehub import failed: %s", exc)
        return False

    try:
        logger.info("Downloading Kaggle dataset with kagglehub: %s", DATASET_SLUG)
        downloaded = Path(kagglehub.dataset_download(DATASET_SLUG))
        logger.info("kagglehub returned path: %s", downloaded)
        _copy_download_if_outside_project(downloaded, logger)
        return True
    except Exception as exc:
        logger.warning("kagglehub download failed: %s", exc)
        return False


def download_with_kaggle_api(logger) -> bool:
    """Try downloading and unzipping the dataset with the Kaggle API."""
    target_zip = RAW_DIR / "casting_product_dataset.zip"
    command = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET_SLUG,
        "-p",
        str(RAW_DIR),
        "--unzip",
    ]
    try:
        logger.info("Trying Kaggle API download: %s", " ".join(command))
        completed = subprocess.run(command, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if completed.stdout:
            logger.info(completed.stdout)
        if completed.stderr:
            logger.warning(completed.stderr)
        if completed.returncode == 0:
            return True
        if target_zip.exists():
            with zipfile.ZipFile(target_zip, "r") as archive:
                archive.extractall(RAW_DIR)
            return True
        return False
    except Exception as exc:
        logger.warning("Kaggle API download failed: %s", exc)
        return False


def main() -> int:
    """CLI entry point."""
    ensure_project_dirs()
    config = load_config()
    logger = setup_logger("download_data", "download_data_module.log")
    class_names = list(config["class_names"])

    if dataset_available(class_names):
        logger.info("Dataset already exists under data/raw or data/external.")
        return 0

    if download_with_kagglehub(logger) and dataset_available(class_names):
        logger.info("Dataset downloaded and verified with kagglehub.")
        return 0

    logger.warning("kagglehub did not produce a verified dataset. Trying Kaggle API.")
    if download_with_kaggle_api(logger) and dataset_available(class_names):
        logger.info("Dataset downloaded and verified with Kaggle API.")
        return 0

    message = (
        "Dataset could not be downloaded automatically. Place the Kaggle dataset contents under "
        "data/raw or data/external, then run: python -m src.prepare_dataset. See data/README_DATA.md."
    )
    logger.error(message)
    print(message, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
