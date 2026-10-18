# Price Scanner v9 — hızlı ve temkinli canlı tarama

Bu pakette `app.py`, `price_ocr.py`, `tax_rules.py`, `requirements.txt` ve test dosyaları var.

## Güncellemeler

- **Canlı kamera:** Her kare için en çok **tek** EasyOCR çalıştırılır. v8'in bazı kareleri iki kez OCR'den geçirmesi kaldırıldı. Sorunlu karelerde ikinci filtre aynı kareyi bloklamak yerine başka bir kameradan gelen kareye uygulanır.
- **Daha küçük işlem görüntüsü:** Kameradan giden JPEG en fazla 880 piksel genişlik ve kalite 0.80; OCR hızlı modda en fazla 800 piksel kullanır. Görünür kamera önizlemesi canlı kalır.
- **Kamera sıklığı:** 1550 ms yerine 1000 ms aralıklarla görüntü gönderilir. Aradaki eski kareler kuyrukta birikmez.
- **Fiyat seçimi:** İndirim yüzdeleri, çizili eski fiyatlar ve yorum sayıları rakama eklenmez. Örnek `%50 / 9,950원 / 20,000원 (üstü çizili) / ★4.9 (99,999+)` için 9.950 seçilmesi hedeflenir.
- **Kırmızı indirim filtresi:** Kırmızı 50% yazısının kenarlarındaki açık renk halesi de maskelenir; yanındaki siyah fiyat korunur. Fiyatın kendisi kırmızıysa, farklı sonraki karede normal OCR de kullanılır.
- **Temkinli doğrulama:** Seçilen rakam iki ayrı kameradan gelen karede eşleşmedikçe hesaplamada kullanılmaz. Çok uzun birleşmiş diziler ve alttaki yorum sayıları onaylanmaz. Belirsizlikte son fiyatın yerine yanlış bir rakam yazılmaz.
- **Diagnostik:** Ekranda son OCR işlem süresi görüntülenir (net fiyat bulunamadığında). Bu, sunucunun ne kadar süre harcadığını anlamaya yardımcı olur.

## Yükleme

1. ZIP'i aç.
2. GitHub deposunun **ana klasöründeki** `app.py`, `price_ocr.py`, `tax_rules.py`, `requirements.txt` dosyalarını yenileriyle değiştir. `price_ocr.py` dosyasını atlama.
3. Streamlit Cloud'un yeni commit'i dağıtmasını bekle ve iPhone'da sayfayı yenile. `test_*.py` dosyaları isteğe bağlıdır, uygulama çalışması için gerekli değildir.

## Test

`python -m unittest discover -p 'test_*.py'`

28 otomatik kod testi geçen senaryoları içerir; bu testler gerçek EasyOCR modeli veya iPhone kamerasının hızını **ölçmez**. İlk OCR model indirmesi / Streamlit Cloud CPU yükü hâlâ ilk taramayı geciktirebilir. Telefon üzerinde mutlaka gerçek etiketle kontrol edilmeli.
