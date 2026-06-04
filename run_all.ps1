param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

New-Item -ItemType Directory -Force outputs/logs | Out-Null

function Invoke-Step {
    param(
        [string]$Name,
        [string]$LogFile,
        [scriptblock]$Command
    )
    Write-Host ""
    Write-Host "==== $Name ===="
    $logPath = Join-Path $ProjectRoot $LogFile
    New-Item -ItemType Directory -Force (Split-Path -Parent $logPath) | Out-Null
    & $Command 2>&1 | Tee-Object -FilePath $logPath
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name. Log: $logPath"
    }
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating Python virtual environment..."
    python -m venv .venv
}

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Pip = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"

if (-not $SkipInstall) {
    Invoke-Step "Upgrade pip" "outputs/logs/install_pip_upgrade.txt" { & $Python -m pip install --upgrade pip }
    Invoke-Step "Install requirements" "outputs/logs/install_requirements.txt" { & $Pip install -r requirements.txt }
    Invoke-Step "Write requirements lock" "outputs/logs/install_lock.txt" { & $Pip freeze | Out-File -Encoding utf8 requirements-lock.txt }
}

$env:KAGGLEHUB_CACHE = Join-Path $ProjectRoot "data\raw\kagglehub_cache"
$env:TORCH_HOME = Join-Path $ProjectRoot "data\raw\torch_cache"
$env:PYTHONPATH = $ProjectRoot

Invoke-Step "Download or verify dataset" "outputs/logs/download_data.txt" { & $Python -m src.download_data }
Invoke-Step "Prepare dataset manifest and splits" "outputs/logs/prepare_dataset.txt" { & $Python -m src.prepare_dataset }
Invoke-Step "Generate EDA outputs" "outputs/logs/eda.txt" { & $Python -m src.eda }
Invoke-Step "Train baseline ML model" "outputs/logs/train_baseline_ml.txt" { & $Python -m src.train_baseline_ml }
Invoke-Step "Train custom CNN model" "outputs/logs/train_cnn.txt" { & $Python -m src.train_cnn }
Invoke-Step "Train transfer learning model" "outputs/logs/train_transfer.txt" { & $Python -m src.train_transfer }
Invoke-Step "Evaluate best model" "outputs/logs/evaluate.txt" { & $Python -m src.evaluate }
Invoke-Step "Generate Grad-CAM explanations" "outputs/logs/gradcam.txt" { & $Python -m src.explain_gradcam }
Invoke-Step "Generate sample prediction" "outputs/logs/predict_sample.txt" { & $Python -m src.predict --sample-from-manifest }
Invoke-Step "Build report assets and documents" "outputs/logs/report_assets.txt" { & $Python -m src.report_assets }
Invoke-Step "Run pytest suite" "outputs/logs/test_results.txt" { & $Python -m pytest tests -q }
Invoke-Step "Finalize report checklist" "outputs/logs/finalize_report.txt" { & $Python -m src.report_assets --finalize }

Write-Host ""
Write-Host "Project run completed. See outputs/reports/final_summary.txt"
Get-Content outputs/reports/final_summary.txt
