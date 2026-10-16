# Price Scanner — Ülke Seçimli Sürüm

Streamlit + WebRTC + EasyOCR ile telefon kamerasından fiyat okuyarak TL karşılığını ve **ülkeye göre tax-free tahminini** hesaplayan örnek uygulama.

## Nasıl çalışır?

1. **Ülke seç:** Güney Kore, Japonya, Fransa, Almanya, İtalya, Tayland veya Birleşik Krallık. Para birimi otomatik seçilir.
2. **Kur otomatik gelir:** Frankfurter API, `https://api.frankfurter.dev/v2/rate/KRW/TRY` gibi bir endpoint üzerinden **online** günlük referans kurunu getirir. Tarihi ve 1 birimin TL karşılığı gösterilir. Kur 15 dakika cache'lenir; elle yenileme düğmesi vardır. Canlı borsa tick verisi değil, günlük güncellenen referans kurdur. İnternet yoksa yanlış sonuç üretmemek için hesap yapılmaz.
3. **Tax free ülkeden seçilir:** Ülke kuralları `tax_rules.py` içinde kaynak URL'leriyle birlikte saklanır. **Tax free kuralları canlı API ile çekilmez**, dönemsel kontrol gerektirir (son kontrol: 2026-10-17). Genel amaçlı ve her operatör için kesin net iade döndüren ücretsiz bir servis varsayılmamıştır.
4. **Kamerayı fiyata tut:** Ortadaki yeşil çerçevede en uzun rakam dizisi aranır. Birden fazla karede tutarlı fiyat elde edince TL tutarları görüntülenir. Yanlışsa manuel tutar girilebilir.

## Önemli iade ayrımı

- **Güney Kore:** Standart KDV %10, ancak KDV dahil etiketin içindeki vergi payı %9,09. Varsayılan **%7**, mağaza ve iade operatörü kesintilerini hesaba katmak için örnek tahmini efektif net iade yüzdesidir, resmî sabit oran değildir. 15.000 KRW minimum; anında iade 1 milyon KRW altında tek fiş ve 5 milyon KRW seyahat toplamı koşuluna bağlıdır.
- **Japonya:** Standart KDV dahil fiyat için teorik geri alınabilecek pay 10/110 ≈ %9,09. Standart vergili alımlar için 5.000 JPY vergi hariç fiş eşiği (5.500 JPY vergi dahil). 1 Kasım 2026'da çıkışta gümrük teyidi sonrası iade sistemine geçilecektir. %8 indirimli vergili ürünler burada modellenmez.
- **Fransa, Almanya, İtalya ve Tayland:** Ülkenin **standart KDV'sine** göre **teorik azami iade** hesaplanır; net iade değildir. Ürün, fiş eşiği, operatör komisyonları doğrulanmalı. Tayland için 2.000 THB mağaza-gün eşiği kullanılır; gerçek iade resmî iade tablosuna göre hesaplanabilir (gelecek sürüm).
- **İngiltere, İskoçya, Galler:** Genel valizle çıkışta alışveriş KDV iadesi yok; Kuzey İrlanda farklıdır.

## Kurulum

Python 3.10+ önerilir.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

Telefon için uygulamayı HTTPS adresinde (ör. Streamlit Community Cloud) yayınlayıp tarayıcıda aç; kamera iznini ver. `streamlit-webrtc` bazı ağlarda TURN sunucusu olmadan bağlantı kuramayabilir. Bu yüzden fotoğraf çekme modu da ekli. İlk EasyOCR çalışması model indirebilir.

## Test

```bash
python -m unittest discover -v
```

## Dosyalar

- `app.py`: Streamlit ve canlı kamera / OCR
- `tax_rules.py`: ülke ayarları, vergi-iade kuralları, kaynaklar ve matematik
- `test_tax_rules.py`: veri girmeden çalışan birim testleri
- `requirements.txt`: paket listesi

## Yayınlama

Bu klasördeki `app.py`, `tax_rules.py`, `requirements.txt` ve isteğe bağlı `README.md` ile `.gitignore` dosyasını GitHub repository köküne yükleyin. Streamlit Community Cloud'da giriş dosyası `app.py` ve Python 3.12 seçin. İlk çalıştırmada EasyOCR modelleri indirilebilir. Canlı WebRTC kamerası bazı mobil ağlarda TURN ayarı gerektirebilir, o durumda Fotoğraf çek modunu kullanın.
