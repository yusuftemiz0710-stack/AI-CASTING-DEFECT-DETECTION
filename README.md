# AI-Powered Casting Defect Prediction System

This project is a deliverable prototype for predicting casting defects from surface images, designed as a term project for the MAK 353 Manufacturing Methods course.

## Folder Structure

- `configs/`: Experiment configuration files
- `data/`: Raw data, external data, data splits, and dataset manifests
- `src/`: Data downloading, preprocessing, training, evaluation, inference, and web demo scripts
- `tests/`: Pytest test suite for validating pipeline integrity
- `outputs/`: Trained models, tables, figures, predictions, log files, and summaries
- `docs/`: Core project reports, summaries, presentations, and bibliographies

## Installation

```powershell
cd "C:\Users\ysfis\Desktop\MAK 353\proje"
.\run_all.ps1
```

If you want to manually activate the virtual environment and install requirements:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

## Downloading the Dataset

Automatic download:

```powershell
python -m src.download_data
```

If you encounter network or API key issues with Kaggle, you can download the dataset manually from the Kaggle dataset page and unzip it under `data/raw` or `data/external`:

```powershell
kaggle datasets download -d ravirajsinh45/real-life-industrial-dataset-of-casting-product -p data/raw --unzip
python -m src.prepare_dataset
```

## Training

```powershell
python -m src.train_baseline_ml
python -m src.train_cnn
python -m src.train_transfer
```

## Evaluation

```powershell
python -m src.evaluate
python -m src.explain_gradcam
```

## Interactive Web Demo

```powershell
streamlit run src/app_streamlit.py
```

## Single Image Inference

```powershell
python -m src.predict --image "path\to\image.jpg" --model outputs\models\best_model.pt
```

## Run Tests

```powershell
pytest tests
```

Test results are logged to `outputs/logs/test_results.txt`.

## Critical Generated Outputs

- `outputs/models/best_model.pt`
- `outputs/models/baseline_ml.joblib`
- `outputs/tables/dataset_manifest.csv`
- `outputs/tables/final_test_metrics.csv`
- `outputs/figures/confusion_matrix.png`
- `outputs/figures/roc_curve.png`
- `outputs/figures/pr_curve.png`
- `outputs/figures/threshold_sweep.png`
- `outputs/figures/gradcam_def_examples.png`
- `outputs/predictions/test_predictions.csv`
- `docs/rapor.md`

## Known Limitations

This system is a prototype developed using an open casting defect dataset. It was built under time and equipment constraints instead of photographing real high-pressure die-cast parts in physical laboratories.

Since the open dataset consists of casting parts with a single geometry and controlled lighting conditions, the models may overfit to this specific geometry. To deploy this system in a real industrial casting line, the models should be retrained and validated using actual casting parts and varied camera environments.

## Final Submission Checklist

The final project checklist can be found at `outputs/reports/final_checklist.md`.
