# Yapay Zekâ ile Döküm Hatası Tahmin Sistemi

MAK 353 İmal Usulleri dönem projesi için döküm ürünlerinin yüzey görüntüsünden hatalı/sağlam tahmini yapan teslim edilebilir prototip.

## Klasör Yapısı

- `configs/`: deney ayarları
- `data/`: ham veri, harici veri, split dosyaları ve veri notları
- `src/`: indirme, hazırlama, eğitim, değerlendirme, tahmin ve demo kodları
- `tests/`: pytest kontrolleri
- `outputs/`: modeller, tablolar, grafikler, tahminler, loglar ve rapor çıktıları
- `docs/`: ana rapor, kısa özet, sunum notu ve kaynakça

## Kurulum

```powershell
cd "C:\Users\ysfis\Desktop\MAK 353\proje"
.\run_all.ps1
```

Sanal ortam elle etkinleştirilecekse:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

## Veri İndirme

Otomatik:

```powershell
python -m src.download_data
```

Kaggle kimliği/ağ sorunu yaşanırsa veri setini Kaggle sayfasından indirip `data/raw` veya `data/external` altına çıkarın:

```powershell
kaggle datasets download -d ravirajsinh45/real-life-industrial-dataset-of-casting-product -p data/raw --unzip
python -m src.prepare_dataset
```

## Eğitim

```powershell
python -m src.train_baseline_ml
python -m src.train_cnn
python -m src.train_transfer
```

## Değerlendirme

```powershell
python -m src.evaluate
python -m src.explain_gradcam
```

## Demo

```powershell
streamlit run src/app_streamlit.py
```

## Tek Görüntü Tahmini

```powershell
python -m src.predict --image "path\to\image.jpg" --model outputs\models\best_model.pt
```

## Testler

```powershell
pytest tests
```

Test çıktısı `outputs/logs/test_results.txt` dosyasına kaydedilir.

## Üretilen Kritik Çıktılar

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

## Bilinen Sınırlılıklar

Normalde ders kapsamında gerçek alüminyum enjeksiyon döküm parçalarının fotoğraflarıyla yapılabilecek bu çalışma, zaman ve erişim kısıtı nedeniyle açık veri seti kullanılarak prototip seviyesinde gerçekleştirilmiştir.

Veri seti tek tip parça ve kontrollü ışık koşullarına sahip olabilir. Gerçek alüminyum enjeksiyon döküm parçalarında kullanılmadan önce yeni fotoğraflarla yeniden eğitim/validasyon yapılmalıdır.

## Teslim Kontrol Listesi

Final kontrol listesi: `outputs/reports/final_checklist.md`
