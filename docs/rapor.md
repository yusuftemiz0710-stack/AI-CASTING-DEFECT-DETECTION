# Yapay Zekâ ile Döküm Hatası Tahmin Sistemi

## 1. Başlık Sayfası Bilgisi

- Proje adı: Yapay Zekâ ile Döküm Hatası Tahmin Sistemi / AI Prediction of Casting Defects
- Ders: MAK 353 İmal Usulleri
- Konu: Yapay Zekâ ile Döküm Hatası Tahmin Sistemi
- Öğrenci: Yusuf Temiz
- Tarih: 02.06.2026

## 2. Özet

Bu projede döküm ürünlerinin yüzey görüntülerinden hatalı/sağlam ayrımı yapan bir görüntü sınıflandırma sistemi kurulmuştur. Hedef, kalite kontrol sürecinde görsel incelemeyi destekleyebilecek bir prototip geliştirmektir.

Normalde ders kapsamında gerçek alüminyum enjeksiyon döküm parçalarının fotoğraflarıyla yapılabilecek bu çalışma, zaman ve erişim kısıtı nedeniyle açık veri seti kullanılarak prototip seviyesinde gerçekleştirilmiştir.

Ana veri seti Kaggle üzerinde yayımlanan `casting product image data for quality inspection` veri setidir. Üç model seviyesi denenmiştir: HOG + lojistik regresyon baseline, küçük bir custom CNN ve MobileNetV2 tabanlı transfer learning. En iyi deploy edilebilir model `transfer_mobilenet_v2` olarak seçilmiştir. Test sonuçları: accuracy `0.9576`, defect precision `0.9545`, defect recall `0.9735`, defect F1 `0.9639`, ROC-AUC `0.9930`, PR-AUC `0.9956`.

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

- Veri seti yolu: `C:\Users\ysfis\Desktop\MAK 353\proje\data\raw\kagglehub_cache\datasets\ravirajsinh45\real-life-industrial-dataset-of-casting-product\versions\2\casting_data\casting_data`
- Toplam görüntü sayısı: `7348`
- Sınıf dağılımı: `{'def_front': 4211, 'ok_front': 3137}`

Train/validation/test dağılımı:

| split | class_name | count |
| --- | --- | --- |
| test | def_front | 453 |
| test | ok_front | 326 |
| train | def_front | 3094 |
| train | ok_front | 2315 |
| val | def_front | 664 |
| val | ok_front | 496 |

![Sınıf dağılımı](../outputs/figures/class_distribution.png)

![def_front örnekleri](../outputs/figures/sample_grid_def_front.png)

![ok_front örnekleri](../outputs/figures/sample_grid_ok_front.png)

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

| threshold | accuracy | precision_defect | recall_defect | f1_defect | roc_auc | pr_auc | false_negative | false_positive | true_positive | true_negative | support | model | split | feature_size | epoch | device | model_path | pretrained_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.9646551724137932 | 0.9799691833590138 | 0.9578313253012049 | 0.9687738004569688 | 0.9955274727944036 | 0.997085879082786 | 28 | 13 | 636 | 483 | 1160 | baseline_hog_logreg | val | 128.0 | nan | nan | nan | nan |
| 0.5 | 0.9922413793103448 | 0.9910044977511244 | 0.9954819277108434 | 0.9932381667918858 | 0.9995050767586476 | 0.999635859842933 | 3 | 6 | 661 | 490 | 1160 | custom_cnn | val | nan | 3.0 | cpu | C:\Users\ysfis\Desktop\MAK 353\proje\outputs\models\custom_cnn.pt | nan |
| 0.5 | 0.9741379310344828 | 0.9817629179331308 | 0.9728915662650602 | 0.9773071104387292 | 0.9950143315196268 | 0.9967206411949492 | 18 | 12 | 646 | 484 | 1160 | transfer_mobilenet_v2 | val | nan | 9.0 | cpu | C:\Users\ysfis\Desktop\MAK 353\proje\outputs\models\transfer_best.pt | True |

![Custom CNN learning curve](../outputs/figures/custom_cnn_learning_curve.png)

![Transfer learning curve](../outputs/figures/transfer_learning_curve.png)

![Confusion matrix](../outputs/figures/confusion_matrix.png)

![ROC eğrisi](../outputs/figures/roc_curve.png)

![PR eğrisi](../outputs/figures/pr_curve.png)

![Threshold analizi](../outputs/figures/threshold_sweep.png)

![Yanlış sınıflandırılan örnekler](../outputs/figures/misclassified_grid.png)

![Grad-CAM def örnekleri](../outputs/figures/gradcam_def_examples.png)

![Grad-CAM ok örnekleri](../outputs/figures/gradcam_ok_examples.png)

![Grad-CAM yanlış örnekleri](../outputs/figures/gradcam_misclassified_examples.png)

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
