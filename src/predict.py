"""Command-line single-image prediction for casting defect detection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .config import load_config
from .paths import MODELS_DIR, PREDICTIONS_DIR, ensure_project_dirs
from .utils import build_transforms, load_manifest, pil_load_rgb, setup_logger, torch_load_checkpoint, write_json


def predict_image(image_path: str | Path, model_path: str | Path, threshold: float | None = None) -> dict:
    """Predict one image and return a serializable result dictionary."""
    config = load_config()
    model, checkpoint = torch_load_checkpoint(model_path, map_location="cpu")
    model.eval()
    threshold = float(threshold if threshold is not None else config.get("threshold", 0.5))
    transform = build_transforms(int(config["image_size"]), train=False)
    image = pil_load_rgb(image_path)
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
    ok_prob = float(probs[0])
    defect_prob = float(probs[1])
    predicted_class = "def_front" if defect_prob >= threshold else "ok_front"
    comment = (
        "Parça hatalı olabilir, yeniden kontrol önerilir."
        if predicted_class == "def_front"
        else "Parça görsel olarak sağlam sınıfa daha yakın görünüyor."
    )
    result = {
        "image_path": str(Path(image_path)),
        "model_path": str(Path(model_path)),
        "predicted_class": predicted_class,
        "defect_probability": defect_prob,
        "ok_probability": ok_prob,
        "threshold": threshold,
        "comment": comment,
        "model_type": checkpoint.get("model_type", "unknown"),
    }
    write_json(PREDICTIONS_DIR / "single_prediction.json", result)
    return result


def sample_image_from_manifest() -> Path:
    """Return the first test image from the manifest for smoke prediction."""
    manifest = load_manifest()
    test_df = manifest[manifest["split"] == "test"]
    if test_df.empty:
        test_df = manifest
    return Path(test_df.iloc[0]["image_path"])


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Predict casting defect from one image.")
    parser.add_argument("--image", type=str, help="Path to an image file.")
    parser.add_argument("--model", type=str, default=str(MODELS_DIR / "best_model.pt"), help="Path to model checkpoint.")
    parser.add_argument("--threshold", type=float, default=None, help="Defect probability threshold.")
    parser.add_argument("--sample-from-manifest", action="store_true", help="Use one test image from the manifest.")
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    ensure_project_dirs()
    logger = setup_logger("predict", "predict_module.log")
    try:
        args = parse_args()
        if args.sample_from_manifest:
            image_path = sample_image_from_manifest()
        elif args.image:
            image_path = Path(args.image)
        else:
            raise ValueError('Provide --image "path/to/image.jpg" or --sample-from-manifest.')
        result = predict_image(image_path, args.model, args.threshold)
        logger.info("Prediction result: %s", result)
        print(f"class prediction: {result['predicted_class']}")
        print(f"defect probability: {result['defect_probability']:.4f}")
        print(f"ok probability: {result['ok_probability']:.4f}")
        print(f"threshold: {result['threshold']:.2f}")
        print(result["comment"])
        return 0
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        print(f"Prediction failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
