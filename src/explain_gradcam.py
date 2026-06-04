"""Generate Grad-CAM style explanations for the transfer model."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import load_config
from .evaluate import predict_dataframe
from .paths import FIGURES_DIR, MODELS_DIR, ensure_project_dirs
from .utils import build_transforms, load_manifest, pil_load_rgb, setup_logger, torch_load_checkpoint


class GradCAM:
    """Minimal Grad-CAM implementation for convolutional PyTorch models."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, _module, _inputs, output) -> None:
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient) -> None:
        self.gradients = gradient.detach()

    def __call__(self, tensor: torch.Tensor, target_class: int | None = None) -> tuple[np.ndarray, int, float]:
        tensor = tensor.clone().detach().requires_grad_(True)
        self.activations = None
        self.gradients = None
        self.model.zero_grad(set_to_none=True)
        output = self.model(tensor)
        probs = torch.softmax(output, dim=1)
        if target_class is None:
            target_class = int(torch.argmax(probs, dim=1).item())
        score = output[:, target_class].sum()
        score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations.detach()).sum(dim=1).squeeze(0)
        cam = torch.relu(cam).detach().cpu().numpy()
        cam = (cam - cam.min()) / max(cam.max() - cam.min(), 1e-8)
        return cam, target_class, float(probs[:, 1].detach().cpu().item())

    def close(self) -> None:
        """Remove hooks."""
        self.forward_handle.remove()


def overlay_cam(image_path: str, cam: np.ndarray, image_size: int) -> np.ndarray:
    """Overlay a heatmap on an RGB image."""
    image = pil_load_rgb(image_path).resize((image_size, image_size))
    arr = np.asarray(image)
    cam_resized = cv2.resize(cam, (image_size, image_size))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.58 * arr + 0.42 * heatmap).clip(0, 255).astype(np.uint8)
    return overlay


def save_gradcam_grid(
    rows: pd.DataFrame,
    model: torch.nn.Module,
    cam_runner: GradCAM,
    config: dict,
    output_name: str,
    title: str,
    device: torch.device,
    max_images: int = 4,
) -> None:
    """Save a Grad-CAM grid for selected rows."""
    image_size = int(config["image_size"])
    output_path = FIGURES_DIR / output_name
    selected = rows.head(max_images)
    if selected.empty:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "Uygun örnek bulunamadı.", ha="center", va="center")
        ax.axis("off")
        ax.set_title(title)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return

    transform = build_transforms(image_size, train=False)
    fig, axes = plt.subplots(1, len(selected), figsize=(4 * len(selected), 4))
    axes_arr = np.atleast_1d(axes)
    for ax, row in zip(axes_arr, selected.itertuples()):
        image = pil_load_rgb(row.image_path)
        tensor = transform(image).unsqueeze(0).to(device)
        cam, target_class, prob_defect = cam_runner(tensor, target_class=None)
        overlay = overlay_cam(row.image_path, cam, image_size)
        ax.imshow(overlay)
        ax.axis("off")
        ax.set_title(f"G:{row.true_class}\nT:{row.predicted_class} p_def={prob_defect:.2f}", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def generate_gradcam() -> None:
    """Generate Grad-CAM examples for correct defect, correct ok, and wrong predictions."""
    config = load_config()
    checkpoint_path = MODELS_DIR / "transfer_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError("Transfer checkpoint not found: outputs/models/transfer_best.pt")

    device = torch.device("cuda" if bool(config.get("use_gpu_if_available", True)) and torch.cuda.is_available() else "cpu")
    model, checkpoint = torch_load_checkpoint(checkpoint_path, map_location=str(device))
    model.to(device)
    model.eval()
    if not hasattr(model, "features"):
        raise RuntimeError("Grad-CAM target layer is defined for MobileNetV2 transfer model.")
    target_layer = model.features[-1]
    cam_runner = GradCAM(model, target_layer)
    try:
        test_df = load_manifest()
        test_df = test_df[test_df["split"] == "test"].reset_index(drop=True)
        predictions = predict_dataframe(model, test_df, config, device)
        correct_def = predictions[(predictions["true_label"] == 1) & (predictions["predicted_label"] == 1)].sort_values(
            "defect_probability", ascending=False
        )
        correct_ok = predictions[(predictions["true_label"] == 0) & (predictions["predicted_label"] == 0)].sort_values(
            "defect_probability", ascending=True
        )
        wrong = predictions[~predictions["correct"]].sort_values("defect_probability", ascending=False)
        save_gradcam_grid(
            correct_def,
            model,
            cam_runner,
            config,
            "gradcam_def_examples.png",
            "Grad-CAM: doğru sınıflandırılmış def_front örnekleri",
            device,
        )
        save_gradcam_grid(
            correct_ok,
            model,
            cam_runner,
            config,
            "gradcam_ok_examples.png",
            "Grad-CAM: doğru sınıflandırılmış ok_front örnekleri",
            device,
        )
        save_gradcam_grid(
            wrong,
            model,
            cam_runner,
            config,
            "gradcam_misclassified_examples.png",
            "Grad-CAM: yanlış sınıflandırılmış örnekler",
            device,
        )
    finally:
        cam_runner.close()


def main() -> int:
    """CLI entry point."""
    ensure_project_dirs()
    logger = setup_logger("explain_gradcam", "explain_gradcam_module.log")
    try:
        generate_gradcam()
        logger.info("Grad-CAM figures created.")
        return 0
    except Exception as exc:
        logger.exception("Grad-CAM generation failed: %s", exc)
        print(f"Grad-CAM generation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
