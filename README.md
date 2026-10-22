# Price Converter — elle fiyat girilen sürüm

Kamera, OCR, fotoğraf yükleme, otomatik etiket tanıma veya önceki tarama cache'i yoktur.

## GitHub / Streamlit Community Cloud

1. Bu arşivdeki `app.py`, `money_input.py`, `tax_rules.py`, `requirements.txt` dosyalarını GitHub deponun **ana dizinine** yükle; aynı isimdeki dosyaların üzerine yaz.
2. Eski `price_ocr.py`, `price_tight.py`, `price_second_opinion.py`, `scanner_guard.py`, `packages.txt` ve ilgili testleri artık kullanmıyoruz; depodan silmek daha temizdir.
3. Streamlit Cloud ana dosya: `app.py`. Uygulamanın yeniden yayımlanmasını beklemeden repoda doğru commit'in oluştuğunu doğrula.

## Kullanım

- Ülkeyi seç.
- Fiyatı elle gir: Kore'de **72.470** veya **72,470** => **72.470 KRW**.
- Ekranda normal TL ve tax free sonrası tahmini TL otomatik görünür. Yeni ürün için `Fiyatı temizle` düğmesini kullan veya mevcut tutarı değiştir.
- Döviz kuru Frankfurter'ın yayımladığı en son referans kurdur (15 dk önbellek). Canlı banka alış/satış kuru değildir.
- Kore için varsayılan tahmini iade oranı **%7**'dir. `Tax free oranı ve kur ayarları` bölümünden değiştirilebilir. Bu oran kesin veya resmî bir sabit oran değildir.
- Vergi iadesi için ülke/mağaza/fiş/ürün şartları olabilir; aşağıda açıklanır. Kore'de 15.000 KRW altı bir *tek ürün* tutarı fiş toplamını yansıtmayabilir.
- Kur servisi çalışmazsa kur alanından manuel değer girebilirsin. Geçersiz veya bilinmeyen kurla hesap yapılmaz.

## Yerel çalıştırma

`pip install -r requirements.txt`

`streamlit run app.py`
