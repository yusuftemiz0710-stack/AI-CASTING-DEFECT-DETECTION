# Kısa Sunum Notu

1. Problem: Dökümde görsel kalite kontrol manuel yapıldığında yorulma ve öznel karar riski var.
2. Veri: Hocanın önerdiği gerçek parçaların fotoğrafları yerine zaman/erişim kısıtı nedeniyle açık Kaggle casting veri seti kullanıldı.
3. Sınıflar: `def_front` hatalı, `ok_front` sağlam. Model spesifik hata türünü değil binary ayrımı öğreniyor.
4. Yöntem: Önce HOG + lojistik regresyon referansı, sonra custom CNN, sonra MobileNetV2 transfer learning.
5. Değerlendirme: Accuracy tek başına yeterli değil; defect recall ve F1 önemli. False negative kalite kontrol açısından en riskli durum.
6. Sonuç: En iyi model `transfer_mobilenet_v2`; test accuracy `0.9576`, recall `0.9735`, F1 `0.9639`.
7. Açıklanabilirlik: Grad-CAM ile modelin görüntünün hangi bölgelerine baktığı incelendi.
8. Sınırlılık: Açık veri tek tip parça ve kontrollü ışık içerebilir; gerçek 236 numaralı oda parçalarıyla yeniden validasyon gerekir.
