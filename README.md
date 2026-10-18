# Price Scanner v10 — Tek Fiyat / Dar Alan

Bu sürüm, canlı kamerada çok sayı bulunan tüm ürün ekranını okumak yerine tek bir fiyatı dar yeşil çerçevenin içinde okur.

## Nasıl kullanılır?

1. `Canlı kamera` seçip kamerayı aç.
2. Fiyatı **tek başına** dar yeşil kutuya getir. Örneğin sadece `28,600원~` görünsün; `%28`, çizili `39,900원`, puan ve yorumlar kutunun dışında kalsın.
3. Fiyat uzun ve kutuya sığmıyorsa `↔ Alanı genişlet` düğmesine dokun. Geniş alan modunda yüzde, çizili fiyat ve yorum sayısı ayıklama kuralları uygulanır.
4. Para sembolleri rakam olarak hesaplanmaz; nokta/virgül ayracı ve rakamlar hesaplanır.

## Neler değişti?

- Varsayılan çerçeve, önizlemenin %52 genişliğinde ve %30 yüksekliğinde; önceki sürümde yaklaşık %81 × %62 idi.
- **Gerçekte OCR'ye giden fotoğraf, ekrandaki yeşil kutunun içidir**. Kamera görüntüsünün geri kalanı sunucuya gönderilmez.
- Dar alanda hızlı tek satır tanıma için `EasyOCR.recognize` kullanılır. Metin algılama (`readtext`) yalnızca ilk iki deneme başarılı olmazsa sonraki karede devreye girer. Bir kare için iki ağır OCR çalıştırılmaz.
- Her 850 ms'de güncel kare gönderilmeye çalışılır; kuyrukta eski kare biriktirilmez.
- Aynı fiyatın farklı iki kamera karesinde okunması gerekir. Yeşil alan dar/geniş değişince eski sonuç temizlenir.
- Fotoğraf modunda da orta alana kırpılır; yanlış sayı alma riski yaratan tüm fotoğrafı yeniden tarama kaldırıldı.
- Canlı kamera kararlı olduğunda kullanılacak hedef süre **5 saniye**. Ağ, sunucu yoğunluğu, ilk model yüklemesi, telefon ve görüntü kalitesi yüzünden kesin süre garantisi verilemez.

## GitHub / Streamlit kurulumu

ZIP içindeki aşağıdaki dosyaları aynı depodaki ana klasöre koy veya eskileriyle değiştir:

- `app.py`
- `price_ocr.py`
- `price_tight.py` **(yeni, mutlaka yükle)**
- `tax_rules.py`
- `requirements.txt`

Streamlit Cloud uygulamanın başlangıç dosyası `app.py` olarak kalmalı. `test_*.py` dosyaları yalnızca test amacıyla eklenmiştir.

## Geliştirici testleri

`python -m unittest discover -p 'test_*.py'` komutuyla metin ayrıştırma, seçilen fiyat, simge/yüzde/yorum, tek OCR çağrısı ve tax-free senaryoları kontrol edilir. Bu testler gerçek kamera hızını ya da EasyOCR'nin görüntüdeki doğruluğunu garanti etmez.
