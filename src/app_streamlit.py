"""Streamlit demo for the casting defect prediction system."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import streamlit as st
import torch
from PIL import Image

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

if __package__ in {None, ""}:
    from src.config import load_config
    from src.explain_gradcam import GradCAM, overlay_cam
    from src.paths import MODELS_DIR, PREDICTIONS_DIR, ensure_project_dirs
    from src.utils import build_transforms, torch_load_checkpoint
else:
    from .config import load_config
    from .explain_gradcam import GradCAM, overlay_cam
    from .paths import MODELS_DIR, PREDICTIONS_DIR, ensure_project_dirs
    from .utils import build_transforms, torch_load_checkpoint


@st.cache_resource
def load_model(model_path: str):
    """Load model once for the Streamlit process."""
    model, checkpoint = torch_load_checkpoint(model_path, map_location="cpu")
    model.eval()
    return model, checkpoint


def predict_pil(model: torch.nn.Module, image: Image.Image, image_size: int) -> tuple[float, float]:
    """Predict ok and defect probabilities for a PIL image."""
    transform = build_transforms(image_size, train=False)
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze(0).cpu().numpy()
    return float(probs[0]), float(probs[1])


def main() -> None:
    """Run Streamlit app."""
    ensure_project_dirs()
    st.set_page_config(page_title="Casting Defect Prediction System", layout="centered")
    st.title("AI-Powered Casting Defect Detection System")
    st.warning("This system is a prototype/course project and does not replace professional quality control decisions.")

    config = load_config()
    model_path = MODELS_DIR / "best_model.pt"
    if not model_path.exists():
        st.error("Model file not found. Please run the training pipeline first (e.g., `python -m src.evaluate` or `.\\run_all.ps1`).")
        return

    threshold = st.slider("Defect probability threshold", min_value=0.05, max_value=0.95, value=float(config.get("threshold", 0.5)), step=0.05)
    uploaded = st.file_uploader("Upload a casting part image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"])
    if uploaded is None:
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    model, checkpoint = load_model(str(model_path))
    ok_prob, defect_prob = predict_pil(model, image, int(config["image_size"]))
    predicted_class = "def_front" if defect_prob >= threshold else "ok_front"
    st.metric("Prediction", "DEFECTIVE (def_front)" if predicted_class == "def_front" else "NORMAL (ok_front)")
    st.progress(min(max(defect_prob, 0.0), 1.0), text=f"Defect probability: {defect_prob:.3f}")
    st.write(f"OK probability: `{ok_prob:.3f}`")
    if predicted_class == "def_front":
        st.error("Warning: The casting part is likely defective. Secondary manual inspection is recommended.")
    else:
        st.success("Pass: The casting part appears visually normal.")

    gradcam_model = model
    gradcam_path = model_path
    if not hasattr(gradcam_model, "features") and (MODELS_DIR / "transfer_best.pt").exists():
        gradcam_path = MODELS_DIR / "transfer_best.pt"
        gradcam_model, _gradcam_checkpoint = load_model(str(gradcam_path))

    if hasattr(gradcam_model, "features"):
        temp_path = PREDICTIONS_DIR / "_streamlit_uploaded.png"
        image.save(temp_path)
        cam_runner = GradCAM(gradcam_model, gradcam_model.features[-1])
        try:
            transform = build_transforms(int(config["image_size"]), train=False)
            tensor = transform(image).unsqueeze(0)
            cam, _target, _prob_defect = cam_runner(tensor, target_class=None)
            overlay = overlay_cam(str(temp_path), cam, int(config["image_size"]))
            st.image(overlay, caption=f"Grad-CAM Heatmap ({gradcam_path.name})", use_container_width=True)
        finally:
            cam_runner.close()
    else:
        st.caption("The selected model structure does not support Grad-CAM visualization.")


if __name__ == "__main__":
    main()
