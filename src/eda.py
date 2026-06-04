"""Exploratory data analysis for the casting defect dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .config import load_config
from .paths import FIGURES_DIR, REPORTS_DIR, TABLES_DIR, ensure_project_dirs
from .utils import load_manifest, pil_load_rgb, save_image_grid, setup_logger


def plot_class_distribution(manifest: pd.DataFrame) -> None:
    """Plot class counts by split."""
    counts = manifest.groupby(["split", "class_name"]).size().unstack(fill_value=0).loc[["train", "val", "test"]]
    ax = counts.plot(kind="bar", figsize=(8, 5), color=["#c24e38", "#3f7f5f"])
    ax.set_title("Sınıf dağılımı")
    ax.set_xlabel("Veri bölümü")
    ax.set_ylabel("Görüntü sayısı")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=160)
    plt.close()


def plot_image_size_distribution(manifest: pd.DataFrame) -> None:
    """Plot width and height distributions."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(manifest["width"], manifest["height"], alpha=0.35, s=14, c=manifest["label"], cmap="Set1")
    ax.set_title("Görüntü boyutu dağılımı")
    ax.set_xlabel("Genişlik (piksel)")
    ax.set_ylabel("Yükseklik (piksel)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "image_size_distribution.png", dpi=160)
    plt.close(fig)


def plot_pixel_histogram(manifest: pd.DataFrame, sample_size: int = 400) -> None:
    """Plot grayscale pixel intensity histograms by class."""
    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(42)
    for class_name, group in manifest.groupby("class_name"):
        paths = group["image_path"].tolist()
        if len(paths) > sample_size:
            paths = rng.choice(paths, size=sample_size, replace=False).tolist()
        values: list[np.ndarray] = []
        for path in tqdm(paths, desc=f"Pixel histogram {class_name}"):
            image = Image.open(path).convert("L").resize((96, 96))
            values.append(np.asarray(image, dtype=np.uint8).ravel())
        if values:
            pixels = np.concatenate(values)
            ax.hist(pixels, bins=50, density=True, histtype="step", linewidth=1.8, label=class_name)
    ax.set_title("Piksel yoğunluğu histogramı")
    ax.set_xlabel("Gri seviye (0-255)")
    ax.set_ylabel("Yoğunluk")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pixel_intensity_histogram.png", dpi=160)
    plt.close(fig)


def estimate_grayscale_ratio(manifest: pd.DataFrame, sample_size: int = 250) -> float:
    """Estimate how many sampled RGB images are effectively grayscale."""
    sample = manifest.sample(min(sample_size, len(manifest)), random_state=42)
    grayscale_count = 0
    for path in sample["image_path"]:
        arr = np.asarray(pil_load_rgb(path))
        if np.mean(np.abs(arr[:, :, 0].astype(float) - arr[:, :, 1].astype(float))) < 1.5 and np.mean(
            np.abs(arr[:, :, 1].astype(float) - arr[:, :, 2].astype(float))
        ) < 1.5:
            grayscale_count += 1
    return grayscale_count / max(len(sample), 1)


def write_summary(manifest: pd.DataFrame, config: dict) -> None:
    """Write dataset summary tables and Turkish EDA interpretation."""
    class_counts = manifest.groupby("class_name").size().rename("count").reset_index()
    split_counts = manifest.groupby(["split", "class_name"]).size().rename("count").reset_index()
    summary_rows = [
        {"metric": "total_images", "value": len(manifest)},
        {"metric": "unique_hashes", "value": manifest["hash"].nunique()},
        {"metric": "duplicate_images", "value": len(manifest) - manifest["hash"].nunique()},
        {"metric": "mean_width", "value": round(float(manifest["width"].mean()), 2)},
        {"metric": "mean_height", "value": round(float(manifest["height"].mean()), 2)},
        {"metric": "grayscale_ratio_sample", "value": round(estimate_grayscale_ratio(manifest), 3)},
        {"metric": "positive_class", "value": config["defect_positive_class"]},
    ]
    pd.DataFrame(summary_rows).to_csv(TABLES_DIR / "dataset_summary.csv", index=False, encoding="utf-8")
    class_counts.to_csv(TABLES_DIR / "class_counts.csv", index=False, encoding="utf-8")
    split_counts.to_csv(TABLES_DIR / "split_counts.csv", index=False, encoding="utf-8")

    counts = dict(zip(class_counts["class_name"], class_counts["count"]))
    max_count = max(counts.values())
    min_count = min(counts.values())
    imbalance_ratio = max_count / max(min_count, 1)
    imbalance_text = (
        f"Sınıflar arasında yaklaşık {imbalance_ratio:.2f} kat fark vardır; ağırlıklı loss ve F1/recall takibi bu nedenle önemlidir."
        if imbalance_ratio > 1.25
        else "Sınıf dağılımı belirgin şekilde dengesiz değildir; yine de defect recall kalite kontrol açısından izlenmiştir."
    )
    report = f"""# EDA Yorumu

- Toplam görüntü sayısı: {len(manifest)}
- Sınıf sayıları: {counts}
- Pozitif sınıf: `{config["defect_positive_class"]}`. Bu sınıf hatalı parça anlamına geldiği için recall ve false negative sayısı ayrıca raporlanmıştır.
- {imbalance_text}
- Örneklemde gri tonlu görüntü oranı yaklaşık `{estimate_grayscale_ratio(manifest):.2f}` olarak tahmin edilmiştir.
- Veri seti gerçek ders fotoğraflarının yerine kullanılan açık bir veri setidir. Bu nedenle modelin 236 numaralı odadaki gerçek alüminyum enjeksiyon döküm parçalarına genellemesi ayrıca doğrulanmalıdır.
"""
    (REPORTS_DIR / "eda_interpretation.md").write_text(report, encoding="utf-8")


def main() -> int:
    """CLI entry point."""
    ensure_project_dirs()
    logger = setup_logger("eda", "eda_module.log")
    try:
        config = load_config()
        manifest = load_manifest()
        plot_class_distribution(manifest)
        plot_image_size_distribution(manifest)
        plot_pixel_histogram(manifest)
        for class_name in config["class_names"]:
            class_paths = manifest[manifest["class_name"] == class_name].sample(
                min(12, (manifest["class_name"] == class_name).sum()), random_state=42
            )["image_path"].tolist()
            save_image_grid(class_paths, FIGURES_DIR / f"sample_grid_{class_name}.png", f"Örnek görüntüler: {class_name}")
        write_summary(manifest, config)
        logger.info("EDA outputs created under outputs/figures and outputs/tables.")
        return 0
    except Exception as exc:
        logger.exception("EDA failed: %s", exc)
        print(f"EDA failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
