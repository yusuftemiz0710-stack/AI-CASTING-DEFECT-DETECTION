"""Project path helpers."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
LOGS_DIR = OUTPUTS_DIR / "logs"
DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

REQUIRED_DIRS = [
    CONFIG_DIR,
    RAW_DIR,
    EXTERNAL_DIR,
    PROCESSED_DIR,
    SPLITS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    PREDICTIONS_DIR,
    LOGS_DIR,
    DOCS_DIR,
    NOTEBOOKS_DIR,
]


def ensure_project_dirs() -> None:
    """Create the expected project directory tree."""
    for directory in REQUIRED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def project_path(*parts: str) -> Path:
    """Return an absolute path inside the project root."""
    return PROJECT_ROOT.joinpath(*parts)


def resolve_project_path(path_like: str | Path) -> Path:
    """Resolve a path that may be absolute or project-relative."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
