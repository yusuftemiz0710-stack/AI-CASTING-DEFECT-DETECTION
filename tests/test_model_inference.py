from pathlib import Path

from src.paths import MODELS_DIR
from src.predict import predict_image
from src.utils import load_manifest, torch_load_checkpoint


def test_best_model_loads():
    model_path = MODELS_DIR / "best_model.pt"
    assert model_path.exists(), "best_model.pt is missing"
    model, checkpoint = torch_load_checkpoint(model_path, map_location="cpu")
    assert checkpoint["model_type"] in {"custom_cnn", "transfer_mobilenet_v2"}
    assert model is not None


def test_single_image_prediction_probabilities():
    manifest = load_manifest()
    image_path = Path(manifest[manifest["split"] == "test"].iloc[0]["image_path"])
    result = predict_image(image_path, MODELS_DIR / "best_model.pt")
    assert result["predicted_class"] in {"def_front", "ok_front"}
    assert 0.0 <= result["defect_probability"] <= 1.0
    assert 0.0 <= result["ok_probability"] <= 1.0
    assert abs(result["defect_probability"] + result["ok_probability"] - 1.0) < 1e-5
