# Price Scanner v12 — Tam fiyat satırı

Bu sürüm, **72.000 yerine 4.000** gibi yalnızca fiyatın son hanelerini algılama sorununu azaltmayı hedefler. v10'daki tanıma yöntemi yeşil bant içindeki uzun ve çoğu boşluk olan görüntüyü tek satır olarak küçültüp OCR'ye veriyordu. Yeni sürüm görüntünün içindeki asıl yazı satırını hızlı görüntü işleme ile bulup dikey olarak yakınlaştırır; yatay olarak **hiçbir rakamı kırpmaz**.

## Kurulum ve çalıştırma

```bash
pip install -r requirements.txt
streamlit run app.py
```

İlk çalıştırmada EasyOCR kendi OCR modelini indirir (internet gerekir, bir kez).

## v12 — "72,400 yerine 4900 okuyor" hatası

**Sebep (kodda ve görüntüde doğrulandı):** `price_tight.py` içindeki `_focus_price_line`, yüksekliği kadrajın %72'sinden büyük her rakamı "gürültü" sayıp atıyordu. Fiyat yeşil bandı doldurunca (arayüzün istediği de bu) bütün rakamlar atılıyor, geriye virgül ve küçük parçalar kalıyor ve kod **rakamların yalnızca alt ~%30'unu** OCR'a gönderiyordu. Sonuç: 72,400 → "4,900" gibi anlamsız okumalar. Koyu zeminde açık yazı (beyaz rakam, kırmızı etiket) ise maske yüzünden daha da kötüydü.

**Değişiklikler**

- `price_tight.py`: satır bulma baştan yazıldı. Uzun rakamlar atılmaz; her glyph tam dahil edilir; açık/koyu yazı ve renkli zemin desteklenir; güvenilmezse bütün kırpım kullanılır; rakamlar görüntü kenarına değiyorsa OCR'a boşluk payı eklenir.
- `price_tight.py`: fiyatın başı/sonu yeşil çerçeveden taşıyorsa (kesik fiyat) okuma **yayımlanmaz**, kullanıcıya "çerçevenin dışında kalıyor" denir. Tam iki rakamın arasından kesilen fiyat bu yöntemle fark edilemez.
- `price_ocr.py`: KRW/JPY için `72,4` → 724, `72,40` → 7240, `9,950.00` → 995000 gibi bozuk gruplamalar artık reddedilir (`.00` güvenle atılır). Diğer para birimleri değişmedi.
- `app.py`: ⚙️ bölümüne **"🔍 OCR'nin gördüğü görüntüyü göster (teşhis)"** eklendi. Yeşil kutunun içini, OCR'a giden görüntüyü ve ham OCR çıktısını canlı gösterir. Yanlış okuma sürerse nedeni bu üç satırdan anlarsın.

## Test

`python -m unittest discover -p 'test_*.py'`

62 test geçer (43 eski + 15 `test_focus_v12.py` + 4 `test_parse_zero_decimal.py`). Yeni testler eski kodda başarısız olacak şekilde yazıldı. Bunlar gerçek EasyOCR modeli kullanmaz; OCR'ın **aldığı görüntünün** tam olduğunu doğrular. Gerçek iPhone/EasyOCR doğruluğunu kanıtlamaz.

## Nasıl kullanılır?

1. Canlı kamerayı aç.
2. **Fiyatın bütün rakamlarını** yeşil yatay bandın içine sığdır. İlk rakam da son rakam da kenarlardan biraz uzakta olsun.
3. `72.000`, `72,000`, `72,000원`, `₩72,000`, `72,000 won`, `72,000 WONDER`, `72,000 mut` gibi yazılarda **hesaba sadece rakam ve binlik/ondalık ayraç** alınır. Para birimi, ülke seçiminden gelir.
4. Yanlış tutarı gördüğünde alt taraftaki **Fiyat düzeltme, kur yenileme ve tax free ayrıntıları** bölümünden `OCR'nin son seçtiği rakam` satırına bakabilirsin. Böylece sorun kameranın harfleri/rakamları tanımasında mı, fiyat ayrıştırmada mı anlaşılır.
5. Üzerinde birden fazla fiyat varsa tek güncel tutarı banda hizala; gerekirse “Alanı genişlet”i kullan.

## Teknik değişiklikler

- Yeşil kutu: Önizlemenin %64 genişliği ve %22 yüksekliği (v10: %52 ve %30). Yani yatayda daha uzun fiyatlar sığarken dikeyde daha az gereksiz bilgi alınır.
- Kamera fotoğrafı sadece yeşil kutunun içinden gönderilir ve **ilk/son rakamı yatayda otomatik kesen ikinci bir kırpma yapılmaz**.
- Kamera JPEG kalitesi 0.80 → 0.90, en fazla aktarım genişliği 760 → 960 piksel. İnce font detayları korunur.
- OCR'den önce OpenCV ile en belirgin yazı satırı seçilir. Boş arka plan kırpılır; böylece EasyOCR'nin uzun görüntüyü 64px yüksekliğe sıkıştırıp rakamları bozması azaltılır.
- Canlı modda her karede yalnızca tek EasyOCR çağrısı yapılır. Farklı karelerde orijinal, kontrastlı ve gerekirse metin algılayıcılı okuma denenir. İki bağımsız karede tutarlı okuma onaylanır.
- `%28 72,000` ifadesinde iki sayı ayrı değerlendiriliyor; para sembolleri sayıya eklenmiyor.
- `app.py`, `price_ocr.py`, `price_tight.py`, `tax_rules.py`, `requirements.txt` aynı klasörde olmalı.

## Test

`python -m unittest discover -p 'test_*.py'`

Testler gerçek iPhone OCR doğruluğunu veya kesin bir 5 saniye süresini kanıtlamaz; gerçek kamera ve Streamlit CPU performansına bağlıdır.
