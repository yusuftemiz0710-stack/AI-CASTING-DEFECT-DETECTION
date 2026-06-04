import torch

from src.config import load_config
from src.utils import build_transforms, load_manifest, pil_load_rgb


def test_preprocessing_tensor_shape_and_normalization():
    config = load_config()
    manifest = load_manifest()
    image_path = manifest.iloc[0]["image_path"]
    image = pil_load_rgb(image_path)
    transform = build_transforms(int(config["image_size"]), train=False)
    tensor = transform(image)
    assert tuple(tensor.shape) == (3, int(config["image_size"]), int(config["image_size"]))
    assert torch.isfinite(tensor).all()
    assert tensor.dtype == torch.float32
