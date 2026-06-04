# Veri Seti Notu

Bu proje için ana veri seti Kaggle üzerinde yayımlanan **casting product image data for quality inspection** veri setidir:

- Kaggle slug: `ravirajsinh45/real-life-industrial-dataset-of-casting-product`
- Beklenen sınıflar: `def_front` ve `ok_front`
- Beklenen kullanım: döküm ürünlerinin yüzey fotoğrafından hatalı/sağlam ayrımı

## Otomatik indirme

PowerShell:

```powershell
cd "C:\Users\ysfis\Desktop\MAK 353\proje"
.\run_all.ps1
```

Tek veri komutu:

```powershell
.venv\Scripts\activate
python -m src.download_data
```

Kod önce `kagglehub.dataset_download("ravirajsinh45/real-life-industrial-dataset-of-casting-product")` dener. Başarısız olursa Kaggle API komutu denenir:

```powershell
kaggle datasets download -d ravirajsinh45/real-life-industrial-dataset-of-casting-product -p data/raw --unzip
```

## Manuel indirme

Kaggle kimliği veya ağ erişimi yoksa:

1. Kaggle veri seti sayfasını açın.
2. Veri setini ZIP olarak indirin.
3. ZIP içeriğini bu klasörlerden birine çıkarın:
   - `data/raw`
   - `data/external`
4. Şu komutu çalıştırın:

```powershell
python -m src.prepare_dataset
```

Hazırlama tamamlandığında `outputs/tables/dataset_manifest.csv` oluşmalıdır.

## Ders bağlamı sınırlılığı

Normalde ders kapsamında gerçek alüminyum enjeksiyon döküm parçalarının fotoğraflarıyla yapılabilecek bu çalışma, zaman ve erişim kısıtı nedeniyle açık veri seti kullanılarak prototip seviyesinde gerçekleştirilmiştir.
