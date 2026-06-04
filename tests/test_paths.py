from pathlib import Path

from src.paths import PROJECT_ROOT, REQUIRED_DIRS, ensure_project_dirs


def test_required_directories_exist():
    ensure_project_dirs()
    assert PROJECT_ROOT.exists()
    for directory in REQUIRED_DIRS:
        assert directory.exists(), f"Missing directory: {directory}"
