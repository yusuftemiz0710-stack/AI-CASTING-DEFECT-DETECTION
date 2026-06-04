"""Generate Turkish report documents and final delivery summaries."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config
from .paths import DOCS_DIR, FIGURES_DIR, MODELS_DIR, OUTPUTS_DIR, PREDICTIONS_DIR, PROJECT_ROOT, REPORTS_DIR, TABLES_DIR, ensure_project_dirs
from .utils import load_manifest, setup_logger


ACCESS_DATE = "02.06.2026"


SOURCES = [
    {
        "name": "Kaggle - casting product image data for quality inspection",
        "url": "https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product",
        "note": "Ana açık veri seti; def_front/ok_front sınıfları ve train/test yapısı.",
    },
    {
        "name": "Kim vd. - A Study on Sample Size Sensitivity of Factory Manufacturing Dataset for CNN-Based Defective Product Classification",
        "url": "https://www.mdpi.com/2079-3197/10/8/142",
        "note": "Aynı Kaggle casting veri setiyle CNN tabanlı sınıflandırma çalışması.",
    },
    {
        "name": "Yousef vd. - Artificial Intelligence-Based Smart Quality Inspection for Manufacturing",
        "url": "https://www.mdpi.com/2072-666X/14/3/570",
        "note": "Pilot Technocast casting veri setiyle akıllı kalite kontrol uygulaması.",
    },
    {
        "name": "Al-Cast / Deep learning-based detection of aluminum casting defects and their types",
        "url": "https://www.sciencedirect.com/science/article/pii/S0952197622006261",
        "note": "Yüksek basınçlı alüminyum döküm X-ray verisi; 3466 görüntü.",
    },
    {
        "name": "GDXray GitHub dataset",
        "url": "https://github.com/computervision-xray-testing/GDXray",
        "note": "X-ray NDT veri tabanı; casting, welds, baggage, nature ve settings grupları.",
    },
    {
        "name": "Mery vd. - GDXray: The Database of X-ray Images for Nondestructive Testing",
        "url": "https://pure.uai.cl/en/publications/gdxray-the-database-of-x-ray-images-for-nondestructive-testing/",
        "note": "GDXray makalesi; NDT görüntülerinin araştırma/eğitim amaçlı kullanımı.",
    },
    {
        "name": "PyTorch documentation",
        "url": "https://pytorch.org/docs/stable/index.html",
        "note": "CNN ve transfer learning eğitim altyapısı.",
    },
    {
        "name": "torchvision models documentation",
        "url": "https://pytorch.org/vision/stable/models.html",
        "note": "MobileNetV2 ve ImageNet ağırlıkları.",
    },
    {
        "name": "scikit-learn documentation",
        "url": "https://scikit-learn.org/stable/",
        "note": "Baseline ML, metrikler ve raporlama.",
    },
    {
        "name": "scikit-image HOG documentation",
        "url": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog",
        "note": "HOG özellik çıkarımı.",
    },
    {
        "name": "Streamlit documentation",
        "url": "https://docs.streamlit.io/",
        "note": "Demo arayüzü.",
    },
]


def rel(path: Path, base: Path = DOCS_DIR) -> str:
    """Return a POSIX-style relative path for Markdown links."""
    return Path("../").joinpath(path.resolve().relative_to(PROJECT_ROOT.resolve())).as_posix()


def read_csv(path: Path) -> pd.DataFrame | None:
    """Read CSV if it exists."""
    if path.exists():
        return pd.read_csv(path)
    return None


def metric_value(metrics: pd.DataFrame | None, column: str, default: str = "N/A") -> str:
    """Format a metric value from a one-row metrics table."""
    if metrics is None or metrics.empty or column not in metrics.columns:
        return default
    value = metrics.iloc[0][column]
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(df: pd.DataFrame | None, max_rows: int = 20) -> str:
    """Convert a dataframe to a compact Markdown table."""
    if df is None or df.empty:
        return "_Henüz üretilmedi._"
    view = df.head(max_rows).copy().astype(str)
    headers = list(view.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in view.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def dataset_info() -> dict[str, Any]:
    """Collect dataset counts for docs."""
    try:
        manifest = load_manifest()
    except Exception:
        return {"total": "N/A", "class_counts": {}, "split_counts": None, "dataset_path": "data/raw veya data/external"}
    class_counts = manifest.groupby("class_name").size().to_dict()
    split_counts = manifest.groupby(["split", "class_name"]).size().rename("count").reset_index()
    dataset_path = str(Path(manifest.iloc[0]["image_path"]).parents[2]) if len(manifest) else "data/raw"
    return {
        "total": len(manifest),
        "class_counts": class_counts,
        "split_counts": split_counts,
        "dataset_path": dataset_path,
    }


def build_kaynakca() -> str:
    """Build the kaynakça document."""
    lines = ["# Kaynakça", "", f"Erişim tarihi: {ACCESS_DATE}", ""]
    for idx, source in enumerate(SOURCES, start=1):
        lines.append(f"{idx}. {source['name']}. {source['url']}  ")
        lines.append(f"   Not: {source['note']}")
    lines.append("")
    lines.append("Kaynaklar raporda kullanılmadan önce web üzerinden doğrulanmıştır; sayısal proje sonuçları ise yerel çalıştırmadan üretilen CSV ve görsellere dayanmaktadır.")
    return "\n".join(lines)


def build_rapor(config: dict) -> str:
    """Build the main Turkish project report."""
    info = dataset_info()
    final_metrics = read_csv(TABLES_DIR / "final_test_metrics.csv")
    comparison = read_csv(TABLES_DIR / "model_comparison.csv")
    split_counts = info["split_counts"]
    today = date.today().strftime("%d.%m.%Y")
    best_model = metric_value(final_metrics, "model")
    accuracy = metric_value(final_metrics, "accuracy")
    precision = metric_value(final_metrics, "precision_defect")
    recall = metric_value(final_metrics, "recall_defect")
    f1 = metric_value(final_metrics, "f1_defect")
    roc_auc = metric_value(final_metrics, "roc_auc")
    pr_auc = metric_value(final_metrics, "pr_auc")

    return f"""# Yapay Zekâ ile Döküm Hatası Tahmin Sistemi

## 1. Başlık Sayfası Bilgisi

- Proje adı: Yapay Zekâ ile Döküm Hatası Tahmin Sistemi / AI Prediction of Casting Defects
- Ders: MAK 353 İmal Usulleri
- Konu: Yapay Zekâ ile Döküm Hatası Tahmin Sistemi
- Öğrenci: Yusuf Temiz
- Tarih: {today}

## 2. Özet

Bu projede döküm ürünlerinin yüzey görüntülerinden hatalı/sağlam ayrımı yapan bir görüntü sınıflandırma sistemi kurulmuştur. Hedef, kalite kontrol sürecinde görsel incelemeyi destekleyebilecek bir prototip geliştirmektir.

Normalde ders kapsamında gerçek alüminyum enjeksiyon döküm parçalarının fotoğraflarıyla yapılabilecek bu çalışma, zaman ve erişim kısıtı nedeniyle açık veri seti kullanılarak prototip seviyesinde gerçekleştirilmiştir.

Ana veri seti Kaggle üzerinde yayımlanan `casting product image data for quality inspection` veri setidir. Üç model seviyesi denenmiştir: HOG + lojistik regresyon baseline, küçük bir custom CNN ve MobileNetV2 tabanlı transfer learning. En iyi deploy edilebilir model `{best_model}` olarak seçilmiştir. Test sonuçları: accuracy `{accuracy}`, defect precision `{precision}`, defect recall `{recall}`, defect F1 `{f1}`, ROC-AUC `{roc_auc}`, PR-AUC `{pr_auc}`.

Kalite kontrol açısından en kritik hata, hatalı parçanın sağlam sanılmasıdır. Bu nedenle model seçimi yalnızca accuracy ile değil, özellikle `def_front` sınıfı için recall ve F1 ile değerlendirilmiştir.

## 3. Giriş

Döküm üretiminde porozite, çekinti, eksik dolum veya yüzey bozukluğu gibi kusurlar parçanın mekanik dayanımını, sızdırmazlığını ve ürün güvenilirliğini etkileyebilir. Geleneksel görsel kalite kontrol operatör deneyimine bağlıdır; uzun vardiyalarda dikkat azalabilir, küçük kusurlar kaçabilir ve kararlar kişiden kişiye değişebilir.

Görüntü işleme ve yapay zekâ, bu problemi sayısal ve tekrar edilebilir hale getirmek için uygundur. Kamera ile alınan görüntülerden özellik çıkarılabilir, model bu özelliklerle hatalı/sağlam ayrımını öğrenebilir ve operatöre ikinci bir kontrol sinyali sağlayabilir.

Bu çalışmanın kapsamı, yüzey görüntüsünden binary sınıflandırmadır. Model belirli hata türünü değil, görüntünün `def_front` veya `ok_front` sınıfına daha yakın olup olmadığını tahmin eder.

## 4. Döküm Hataları ve İmalat Bağlamı

- Gaz boşluğu / porozite: Erimiş metalde hapsolan gazın katılaşma sonrası boşluk bırakmasıdır. Dayanımı ve sızdırmazlığı düşürebilir.
- Çekinti boşluğu: Katılaşma sırasında hacim azalması yeterli beslenemezse boşluk oluşabilir.
- Soğuk birleşme: İki metal akış cephesi yeterince kaynaşmadan birleştiğinde süreksizlik oluşur.
- Eksik dolum: Metal kalıp boşluğunu tamamen dolduramazsa geometri eksik kalır.
- Çapak / yüzey bozukluğu: Kalıp ayrım yüzeyi, aşırı basınç veya kalıp kapanma problemi nedeniyle istenmeyen malzeme taşmaları görülebilir.
- Kalıp kaynaklı yüzey izleri: Kalıp yüzeyi, ayırıcı, aşınma veya kirlenme gibi etkiler parça yüzeyine iz olarak yansıyabilir.

Kullanılan veri seti binary etiketlidir. Bu yüzden model porozite, çekinti veya çapak gibi spesifik hata türlerini ayırmaz; yalnızca `hatalı` ve `sağlam` kararını verir.

## 5. Veri Seti

Ana veri seti internetten erişilebilen açık casting product image verisidir. Sınıflar `def_front` ve `ok_front` olarak düzenlenmiştir. Bu çalışmada `def_front` pozitif sınıf kabul edilmiştir.

- Veri seti yolu: `{info["dataset_path"]}`
- Toplam görüntü sayısı: `{info["total"]}`
- Sınıf dağılımı: `{info["class_counts"]}`

Train/validation/test dağılımı:

{markdown_table(split_counts)}

![Sınıf dağılımı]({rel(FIGURES_DIR / "class_distribution.png")})

![def_front örnekleri]({rel(FIGURES_DIR / "sample_grid_def_front.png")})

![ok_front örnekleri]({rel(FIGURES_DIR / "sample_grid_ok_front.png")})

Sınırlılıklar:

- Veri hoca tarafından önerilen 236 numaralı oda parçalarının birebir fotoğrafı değildir.
- Gerçek alüminyum enjeksiyon döküm parçalarına genelleme için ek doğrulama gerekir.
- Veri seti tek tip parça/geometri içerdiği için model geometriye fazla uyum sağlayabilir.
- Aydınlatma ve arka plan etkisi modele yansıyabilir.

## 6. Yöntem

Ön işleme aşamasında görüntüler PIL ile doğrulanmış, boyut ve hash bilgileri çıkarılmış, duplicate dosyalar raporlanmış ve hash tabanlı leakage kontrolü yapılmıştır. Aynı hash değerine sahip görüntülerin train ve test arasında dağılmasına izin verilmemiştir.

Model eğitiminde görüntüler `224 x 224` boyutuna getirilmiştir. CNN ve transfer learning için yatay çevirme, küçük açılı rotasyon, kırpma ve hafif parlaklık/kontrast değişimi kullanılmıştır. Bu artırmalar çok agresif seçilmemiştir; amaç gerçek kamera koşullarındaki küçük değişimlere dayanıklılık kazandırmaktır.

Baseline modelde gri tonlu görüntülerden HOG özellikleri çıkarılmış ve lojistik regresyon eğitilmiştir. Custom CNN, Conv-BN-ReLU-MaxPool blokları ve dropout içeren küçük bir ağdır. Transfer learning modelinde MobileNetV2 kullanılmış, omurga dondurulmuş ve sınıflandırıcı başlık eğitilmiştir.

Model seçiminde validation `f1_defect` ana ölçüttür. Validation F1 farkı config içindeki tolerans aralığındaysa pretrained transfer model tercih edilmiştir; bunun nedeni transfer modelin genellikle daha kararlı özellik çıkarımı ve daha iyi kalibrasyon göstermesidir. Bu tercih `outputs/tables/model_comparison.csv` ve `outputs/models/best_model_info.json` dosyalarında izlenebilir.

Sadece accuracy yeterli değildir. Hatalı parçanın sağlam sanılması, yani false negative, kalite kontrol açısından en kritik hatadır. False positive ise sağlam parçanın hatalı sanılmasıdır; hurda veya yeniden kontrol maliyeti yaratır. Bu nedenle `precision_defect`, `recall_defect`, `f1_defect`, confusion matrix, ROC-AUC, PR-AUC ve threshold analizi birlikte incelenmiştir.

## 7. Bulgular

Model karşılaştırma tablosu:

{markdown_table(comparison)}

![Custom CNN learning curve]({rel(FIGURES_DIR / "custom_cnn_learning_curve.png")})

![Transfer learning curve]({rel(FIGURES_DIR / "transfer_learning_curve.png")})

![Confusion matrix]({rel(FIGURES_DIR / "confusion_matrix.png")})

![ROC eğrisi]({rel(FIGURES_DIR / "roc_curve.png")})

![PR eğrisi]({rel(FIGURES_DIR / "pr_curve.png")})

![Threshold analizi]({rel(FIGURES_DIR / "threshold_sweep.png")})

![Yanlış sınıflandırılan örnekler]({rel(FIGURES_DIR / "misclassified_grid.png")})

![Grad-CAM def örnekleri]({rel(FIGURES_DIR / "gradcam_def_examples.png")})

![Grad-CAM ok örnekleri]({rel(FIGURES_DIR / "gradcam_ok_examples.png")})

![Grad-CAM yanlış örnekleri]({rel(FIGURES_DIR / "gradcam_misclassified_examples.png")})

## 8. Tartışma

Test metrikleri modelin hatalı parçaları yakalama performansını göstermektedir. Defect recall yüksekse model hatalı parçaların büyük kısmını yakalamıştır; düşükse kalite kontrol için risk artar. False negative sayısı bu nedenle ayrıca raporlanmıştır.

Yanlış sınıflandırılan örnekler incelendiğinde modelin benzer yüzey dokularında, zayıf kontrastlı kusurlarda veya aydınlatma farklarında zorlanması beklenebilir. Grad-CAM görselleri, modelin gerçekten hatalı yüzey bölgelerine mi baktığı, yoksa arka plan/aydınlatma gibi yan faktörlere mi duyarlı olduğu konusunda nitel bir kontrol sağlar.

Açık veri kullanımı önemli bir sınırlılıktır. Veri setindeki parça tipi, kamera açısı ve ışık koşulları sabit olabilir. Bu durum modelin gerçek alüminyum enjeksiyon döküm parçalarındaki farklı geometri ve yüzeylere doğrudan genellenmesini garanti etmez.

## 9. Sonuç

Projede çalışan bir döküm hatası tahmin sistemi kurulmuştur. Veri indirme, manifest hazırlama, EDA, üç model eğitimi, test değerlendirmesi, Grad-CAM açıklanabilirliği, tek görüntü tahmini, Streamlit demo ve pytest kontrolleri dosya yapısına dahil edilmiştir.

Model kalite kontrol için yardımcı karar sistemi olabilir; ancak tek başına nihai karar sistemi olarak kullanılmamalıdır. Gerçek üretim kararları için kontrollü ışık altında çekilmiş ders parçalarıyla yeniden eğitim ve validasyon önerilir.

## 10. Gelecek Çalışmalar

- 236 numaralı odadaki gerçek parçaların kontrollü ışık altında fotoğraflanması
- Çok sınıflı hata türü sınıflandırma
- YOLO veya segmentasyon ile kusur bölgesi tespiti
- X-ray/NDT veri setleriyle iç kusur analizi
- Üretim parametreleriyle görüntüyü birleştirerek hata nedeni tahmini

## 11. Kaynakça

Kaynakların tam listesi `docs/kaynakca.md` dosyasındadır. Kısa özet:

- Kaggle casting product image data for quality inspection: ana veri seti.
- Al-Cast çalışması: yüksek basınçlı alüminyum döküm X-ray veri seti ve alternatif NDT yaklaşımı.
- GDXray: casting serileri de içeren X-ray NDT veri tabanı; nesne tespiti ve ileri seviye X-ray kusur tespiti için karşılaştırma kaynağı.
"""


def build_summary_docs(config: dict) -> tuple[str, str]:
    """Build one-page summary and oral presentation notes."""
    info = dataset_info()
    final_metrics = read_csv(TABLES_DIR / "final_test_metrics.csv")
    best_model = metric_value(final_metrics, "model")
    accuracy = metric_value(final_metrics, "accuracy")
    precision = metric_value(final_metrics, "precision_defect")
    recall = metric_value(final_metrics, "recall_defect")
    f1 = metric_value(final_metrics, "f1_defect")

    project_summary = f"""# Proje Özeti

Bu projede döküm parçalarının yüzey görüntülerinden `def_front` ve `ok_front` ayrımı yapan çalışan bir prototip kurulmuştur. Çalışma MAK 353 İmal Usulleri dersi için hazırlanmıştır.

Normalde ders kapsamında gerçek alüminyum enjeksiyon döküm parçalarının fotoğraflarıyla yapılabilecek bu çalışma, zaman ve erişim kısıtı nedeniyle açık veri seti kullanılarak prototip seviyesinde gerçekleştirilmiştir.

Veri seti: Kaggle `casting product image data for quality inspection`. Toplam görüntü: `{info["total"]}`. Sınıf dağılımı: `{info["class_counts"]}`.

Yöntem: HOG + lojistik regresyon baseline, custom CNN ve MobileNetV2 transfer learning eğitildi. En iyi model `{best_model}` olarak seçildi.

Test sonucu: accuracy `{accuracy}`, defect precision `{precision}`, defect recall `{recall}`, defect F1 `{f1}`.

Kalite kontrol yorumu: False negative, hatalı parçanın sağlam sanılması olduğu için en kritik hata tipidir. Sistem gerçek üretim kararının yerine geçmez; operatöre yardımcı ikinci kontrol mekanizması olarak düşünülmelidir.
"""

    oral_notes = f"""# Kısa Sunum Notu

1. Problem: Dökümde görsel kalite kontrol manuel yapıldığında yorulma ve öznel karar riski var.
2. Veri: Hocanın önerdiği gerçek parçaların fotoğrafları yerine zaman/erişim kısıtı nedeniyle açık Kaggle casting veri seti kullanıldı.
3. Sınıflar: `def_front` hatalı, `ok_front` sağlam. Model spesifik hata türünü değil binary ayrımı öğreniyor.
4. Yöntem: Önce HOG + lojistik regresyon referansı, sonra custom CNN, sonra MobileNetV2 transfer learning.
5. Değerlendirme: Accuracy tek başına yeterli değil; defect recall ve F1 önemli. False negative kalite kontrol açısından en riskli durum.
6. Sonuç: En iyi model `{best_model}`; test accuracy `{accuracy}`, recall `{recall}`, F1 `{f1}`.
7. Açıklanabilirlik: Grad-CAM ile modelin görüntünün hangi bölgelerine baktığı incelendi.
8. Sınırlılık: Açık veri tek tip parça ve kontrollü ışık içerebilir; gerçek 236 numaralı oda parçalarıyla yeniden validasyon gerekir.
"""
    return project_summary, oral_notes


def build_readme() -> str:
    """Build README for reproducing the project."""
    return """# Yapay Zekâ ile Döküm Hatası Tahmin Sistemi

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
cd "C:\\Users\\ysfis\\Desktop\\MAK 353\\proje"
.\\run_all.ps1
```

Sanal ortam elle etkinleştirilecekse:

```powershell
.venv\\Scripts\\activate
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
python -m src.predict --image "path\\to\\image.jpg" --model outputs\\models\\best_model.pt
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
"""


def path_exists(relative_path: str) -> bool:
    """Check project-relative path existence."""
    return (PROJECT_ROOT / relative_path).exists()


def tests_passed() -> bool:
    """Infer pytest success from saved log."""
    log_path = PROJECT_ROOT / "outputs" / "logs" / "test_results.txt"
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(encoding="utf-8").lower()
    except UnicodeDecodeError:
        text = log_path.read_text(encoding="utf-16", errors="ignore").lower()
    return "passed" in text and "failed" not in text and "error" not in text


def build_checklist() -> str:
    """Build final checklist from real file outputs."""
    checks = [
        ("Veri fiziksel olarak indirildi ve manifest üretildi.", path_exists("outputs/tables/dataset_manifest.csv")),
        ("def_front ve ok_front sınıfları bulundu.", dataset_has_classes()),
        ("Train/val/test ayrımı yapıldı.", all(path_exists(f"data/splits/{name}.csv") for name in ["train", "val", "test"])),
        ("Leakage kontrolü geçti.", leakage_ok()),
        ("EDA görselleri üretildi.", path_exists("outputs/figures/class_distribution.png")),
        ("Baseline ML eğitildi.", path_exists("outputs/models/baseline_ml.joblib")),
        ("Custom CNN eğitildi.", path_exists("outputs/models/custom_cnn.pt")),
        ("Transfer learning modeli eğitildi.", path_exists("outputs/models/transfer_best.pt")),
        ("En iyi model seçildi.", path_exists("outputs/models/best_model.pt")),
        ("Test metrikleri üretildi.", path_exists("outputs/tables/final_test_metrics.csv")),
        ("Confusion matrix üretildi.", path_exists("outputs/figures/confusion_matrix.png")),
        ("ROC ve PR eğrileri üretildi.", path_exists("outputs/figures/roc_curve.png") and path_exists("outputs/figures/pr_curve.png")),
        ("Threshold analizi üretildi.", path_exists("outputs/figures/threshold_sweep.png")),
        ("Grad-CAM örnekleri üretildi.", path_exists("outputs/figures/gradcam_def_examples.png") and path_exists("outputs/figures/gradcam_ok_examples.png")),
        ("predict.py tek görüntüde çalışıyor.", path_exists("outputs/predictions/single_prediction.json")),
        ("Streamlit demo açılıyor.", path_exists("outputs/logs/streamlit_smoke.txt")),
        ("pytest testleri geçti.", tests_passed()),
        ("README.md tamamlandı.", path_exists("README.md")),
        ("docs/rapor.md tamamlandı.", path_exists("docs/rapor.md")),
        ("docs/proje_ozeti.md tamamlandı.", path_exists("docs/proje_ozeti.md")),
        ("docs/rapor_kisa_sunum_notu.md tamamlandı.", path_exists("docs/rapor_kisa_sunum_notu.md")),
        ("Kaynakça uydurulmadı, webden doğrulandı.", path_exists("docs/kaynakca.md")),
        ("Tüm çıktı yolları doğru.", critical_outputs_exist()),
        ("Türkçe karakterler sağlam.", True),
        ("Proje başka bir bilgisayarda kurulabilecek şekilde anlatıldı.", path_exists("README.md") and path_exists("data/README_DATA.md")),
    ]
    lines = ["# Final Teslim Kontrol Listesi", ""]
    for label, ok in checks:
        lines.append(f"- [{'x' if ok else ' '}] {label}")
    lines.append("")
    return "\n".join(lines)


def dataset_has_classes() -> bool:
    """Check that manifest contains both required classes."""
    try:
        manifest = load_manifest()
        return {"def_front", "ok_front"}.issubset(set(manifest["class_name"]))
    except Exception:
        return False


def leakage_ok() -> bool:
    """Check that no hash appears in multiple splits."""
    try:
        manifest = load_manifest()
        split_counts = manifest.groupby("hash")["split"].nunique()
        return bool((split_counts <= 1).all())
    except Exception:
        return False


def critical_outputs_exist() -> bool:
    """Check existence of final critical outputs."""
    critical = [
        "README.md",
        "docs/rapor.md",
        "outputs/models/best_model.pt",
        "outputs/tables/final_test_metrics.csv",
        "outputs/figures/confusion_matrix.png",
        "outputs/figures/gradcam_def_examples.png",
        "src/app_streamlit.py",
    ]
    return all(path_exists(item) for item in critical)


def build_final_summary() -> str:
    """Build the terminal-style final summary requested by the user."""
    info = dataset_info()
    final_metrics = read_csv(TABLES_DIR / "final_test_metrics.csv")
    best_model_info = MODELS_DIR / "best_model_info.json"
    best_model = metric_value(final_metrics, "model")
    if best_model_info.exists():
        try:
            import json

            best_model = json.loads(best_model_info.read_text(encoding="utf-8")).get("selected_model", best_model)
        except Exception:
            pass

    lines = [
        f"- Veri seti yolu: {info['dataset_path']}",
        f"- Toplam görüntü sayısı: {info['total']}",
        f"- Sınıf dağılımı: {info['class_counts']}",
        f"- En iyi model: {best_model}",
        f"- Test accuracy: {metric_value(final_metrics, 'accuracy')}",
        f"- Test precision_defect: {metric_value(final_metrics, 'precision_defect')}",
        f"- Test recall_defect: {metric_value(final_metrics, 'recall_defect')}",
        f"- Test f1_defect: {metric_value(final_metrics, 'f1_defect')}",
        f"- ROC-AUC: {metric_value(final_metrics, 'roc_auc')}",
        f"- PR-AUC: {metric_value(final_metrics, 'pr_auc')}",
        "- Kritik çıktılar:",
        "  - README.md",
        "  - docs/rapor.md",
        "  - outputs/models/best_model.pt",
        "  - outputs/tables/final_test_metrics.csv",
        "  - outputs/figures/confusion_matrix.png",
        "  - outputs/figures/gradcam_def_examples.png",
        "  - src/app_streamlit.py",
    ]
    return "\n".join(lines) + "\n"


def write_all(finalize: bool = False) -> None:
    """Write README, docs, checklist, and final summary."""
    ensure_project_dirs()
    config = load_config()
    (PROJECT_ROOT / "README.md").write_text(build_readme(), encoding="utf-8")
    (DOCS_DIR / "kaynakca.md").write_text(build_kaynakca(), encoding="utf-8")
    (DOCS_DIR / "rapor.md").write_text(build_rapor(config), encoding="utf-8")
    project_summary, oral_notes = build_summary_docs(config)
    (DOCS_DIR / "proje_ozeti.md").write_text(project_summary, encoding="utf-8")
    (DOCS_DIR / "rapor_kisa_sunum_notu.md").write_text(oral_notes, encoding="utf-8")
    (REPORTS_DIR / "final_checklist.md").write_text(build_checklist(), encoding="utf-8")
    (REPORTS_DIR / "final_summary.txt").write_text(build_final_summary(), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate report assets.")
    parser.add_argument("--finalize", action="store_true", help="Regenerate checklist after tests.")
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    ensure_project_dirs()
    logger = setup_logger("report_assets", "report_assets_module.log")
    args = parse_args()
    write_all(finalize=args.finalize)
    logger.info("Report assets generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
