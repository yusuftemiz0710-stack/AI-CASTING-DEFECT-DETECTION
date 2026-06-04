# Proje Özeti

Bu projede döküm parçalarının yüzey görüntülerinden `def_front` ve `ok_front` ayrımı yapan çalışan bir prototip kurulmuştur. Çalışma MAK 353 İmal Usulleri dersi için hazırlanmıştır.

Normalde ders kapsamında gerçek alüminyum enjeksiyon döküm parçalarının fotoğraflarıyla yapılabilecek bu çalışma, zaman ve erişim kısıtı nedeniyle açık veri seti kullanılarak prototip seviyesinde gerçekleştirilmiştir.

Veri seti: Kaggle `casting product image data for quality inspection`. Toplam görüntü: `7348`. Sınıf dağılımı: `{'def_front': 4211, 'ok_front': 3137}`.

Yöntem: HOG + lojistik regresyon baseline, custom CNN ve MobileNetV2 transfer learning eğitildi. En iyi model `transfer_mobilenet_v2` olarak seçildi.

Test sonucu: accuracy `0.9576`, defect precision `0.9545`, defect recall `0.9735`, defect F1 `0.9639`.

Kalite kontrol yorumu: False negative, hatalı parçanın sağlam sanılması olduğu için en kritik hata tipidir. Sistem gerçek üretim kararının yerine geçmez; operatöre yardımcı ikinci kontrol mekanizması olarak düşünülmelidir.
