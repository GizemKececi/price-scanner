"""Telefon kamerasıyla fiyat tarayan, TL ve tahmini tax-free tutarı hesaplayan Streamlit MVP."""

from __future__ import annotations

# Mac ve Linux sistem sertifikalarını kullanarak SSL doğrulamasını koru.
import truststore
truststore.inject_into_ssl()

import re
import threading
import time
from decimal import Decimal, ROUND_HALF_UP
from io import BytesIO
from queue import Empty, Full, Queue

import av
import cv2
import easyocr
import numpy as np
import requests
import streamlit as st
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from tax_rules import COUNTRIES, RULES_REVIEWED, estimate_refund

st.set_page_config(page_title="Price Scanner", page_icon="🔎", layout="centered")

ZERO_DECIMAL = {"KRW", "JPY"}
NUMBERS = re.compile(r"(?<!\d)\d[\d.,\s]*\d|(?<!\d)\d(?!\d)")


@st.cache_data(ttl=900, show_spinner=False)
def fetch_fx(currency: str) -> tuple[Decimal, str]:
    """Frankfurter günlük referans kuru: 1 birim kaynak para kaç TRY."""
    response = requests.get(
        f"https://api.frankfurter.dev/v2/rate/{currency}/TRY",
        timeout=12,
    )
    response.raise_for_status()
    data = response.json()
    return Decimal(str(data["rate"])), data["date"]


@st.cache_resource(show_spinner="OCR modeli hazırlanıyor (ilk seferde indirilebilir)…")
def load_reader():
    # İlk çalıştırmada EasyOCR model dosyalarını indirir.
    return easyocr.Reader(["en"], gpu=False)


def parse_amount(raw: str, currency: str) -> Decimal | None:
    """OCR metnini binlik ve ondalık ayraçlarını hesaba katarak sayıya çevir."""
    digits = re.sub(r"[^0-9]", "", raw)
    if not digits:
        return None
    raw = raw.strip().replace(" ", "")
    if currency in ZERO_DECIMAL:
        # Won ve yen etikette genellikle tam sayı olarak yazılır.
        return Decimal(digits)

    # Ondalık ayraç, varsa en sağdaki virgül/noktadır; sonundaki
    # 1-2 hane kuruş/cent olarak kabul edilir.
    last_comma, last_dot = raw.rfind(","), raw.rfind(".")
    sep_idx = max(last_comma, last_dot)
    if sep_idx >= 0 and len(raw) - sep_idx - 1 in (1, 2):
        decimals = len(raw) - sep_idx - 1
        return Decimal(digits[:-decimals] + "." + digits[-decimals:])
    return Decimal(digits)


def scan_crop(bgr: np.ndarray, currency: str, reader) -> tuple[Decimal | None, str]:
    """En uzun rakam dizisi; eşitlikte daha büyük görünür alan."""
    h, w = bgr.shape[:2]
    # Küçük baskıların okunmasını kolaylaştırmak için büyüt.
    if w < 1100:
        factor = min(2.0, 1100 / max(w, 1))
        bgr = cv2.resize(bgr, None, fx=factor, fy=factor)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    results = reader.readtext(
        rgb, detail=1, paragraph=False,
        allowlist="0123456789., ",
    )

    candidates = []
    for box, text, confidence in results:
        if confidence < 0.25:
            continue
        for match in NUMBERS.finditer(text):
            raw = match.group().strip()
            value = parse_amount(raw, currency)
            if value is None or value <= 0:
                continue
            digit_length = sum(ch.isdigit() for ch in raw)
            if digit_length < 2:  # Tarih, yüzde işareti, tek karakter vb. azalt.
                continue
            pts = np.array(box, dtype=np.float32)
            bounding = cv2.boundingRect(pts)
            area = bounding[2] * bounding[3]
            candidates.append((digit_length, area, confidence, value, raw))

    if not candidates:
        return None, ""
    _digits, _area, _confidence, amount, raw = max(candidates, key=lambda x: x[:3])
    return amount, raw


def center_region(img: np.ndarray):
    h, w = img.shape[:2]
    x1, x2 = int(w * 0.09), int(w * 0.91)
    y1, y2 = int(h * 0.32), int(h * 0.65)
    return x1, y1, x2, y2


def format_tr(value: Decimal, places: int = 2) -> str:
    q = Decimal("1") if places == 0 else Decimal("0.01")
    s = f"{value.quantize(q, rounding=ROUND_HALF_UP):,.{places}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


class ScanState:
    def __init__(self):
        self.lock = threading.Lock()
        self.queue: Queue = Queue(maxsize=1)
        self.price: Decimal | None = None
        self.raw = ""
        self.last_frame = 0.0
        self.worker_started = False
        self.currency = "KRW"
        self.country = "Güney Kore"
        self.status = "Kamerayı başlatıp fiyatı yeşil kutuya hizalayın."


def start_worker(state: ScanState, reader):
    if state.worker_started:
        return
    state.worker_started = True

    def run():
        previous = None
        repetitions = 0
        while True:
            try:
                crop, selected_currency = state.queue.get(timeout=2)
            except Empty:
                continue
            try:
                number, raw = scan_crop(crop, selected_currency, reader)
                if number is None:
                    previous, repetitions = None, 0
                    continue
                if number == previous:
                    repetitions += 1
                else:
                    previous, repetitions = number, 1
                # Tek bir bulanık karede yanlış fiyat yakalamamak için
                # 2 arka arkaya benzer okuma bekle.
                if repetitions >= 2:
                    with state.lock:
                        state.price = number
                        state.raw = raw
                        state.status = "Fiyat okundu."
            except Exception as exc:
                with state.lock:
                    state.status = f"OCR hatası: {type(exc).__name__}"

    threading.Thread(target=run, daemon=True, name="ocr_worker").start()


st.title("🔎 Price Scanner")
st.caption("Ülkeni seç → kamera ile fiyatı oku → güncel kur ve tax free tahminini TL olarak gör.")

country = st.selectbox(
    "🌍 Nereye seyahat ediyorsun?",
    list(COUNTRIES),
    format_func=lambda key: COUNTRIES[key].label,
    index=0,
)
rule = COUNTRIES[country]
currency = rule.currency
st.caption(f"Para birimi otomatik seçildi: **{currency}** · Standart KDV: **%{rule.vat}**")

with st.expander("ℹ️ Bu ülkenin tax free kuralı"):
    st.write(rule.summary)
    st.write(rule.eligibility_note)
    st.caption(f"Kuralların son manuel kontrolü: {RULES_REVIEWED}. Vergi kuralları canlı API ile yenilenmiyor.")
    for i, source_url in enumerate(rule.sources, 1):
        st.markdown(f"[Resmî kaynak {i}]({source_url})")

with st.expander("⚙️ Gelişmiş: iade tahminini düzenle"):
    use_custom = st.checkbox(
        "Tahmini net iadeyi kendim gireyim",
        value=False,
        disabled=rule.method == "unavailable",
        help="Gerçek fişte yazan efektif iade oranını biliyorsan buradan değiştirebilirsin.",
    )
    theoretical_pct = rule.vat / (Decimal("100") + rule.vat) * Decimal("100")
    custom_pct = st.number_input(
        "Etiket fiyatının yüzde kaçı iade edilecek?",
        min_value=0.0, max_value=float(theoretical_pct),
        value=float(rule.estimated_net_pct or theoretical_pct),
        step=0.1,
        disabled=not use_custom or rule.method == "unavailable",
    )
    override_pct = Decimal(str(custom_pct)) if use_custom else None

if st.button("🔄 Güncel kuru yeniden sorgula", use_container_width=True):
    fetch_fx.clear()
    st.rerun()

try:
    fx, fx_date = fetch_fx(currency)
except requests.RequestException:
    st.error("Online döviz kuruna ulaşılamadı. İnternet bağlantını kontrol edip tekrar dene. Manuel/eski kur kullanılmıyor.")
    st.stop()
except (KeyError, ValueError, TypeError):
    st.error("Kur servisinden beklenmeyen bir cevap geldi; dönüşüm yapılmadı.")
    st.stop()

st.metric(f"Güncel referans kur · 1 {currency}", f"{format_tr(fx, 6)} TL")
if currency in ZERO_DECIMAL:
    st.caption(f"1.000 {currency} = {format_tr(fx * 1000, 2)} TL")
st.caption(f"Frankfurter · Kur tarihi: **{fx_date}** · 15 dakikada bir yeniden sorgulanabilir. Bu piyasa referans kurudur, kartının kesin alış kuru değildir.")

manual = st.number_input(
    f"Gerekirse fiyatı elle düzelt ({currency}; 0 = kamera)",
    min_value=0.0, value=0.0, step=1000.0 if currency in ZERO_DECIMAL else 10.0,
)

if "scanner_state" not in st.session_state:
    st.session_state.scanner_state = ScanState()
state: ScanState = st.session_state.scanner_state

with state.lock:
    if state.currency != currency or state.country != country:
        state.price = None
        state.raw = ""
        state.currency = currency
        state.country = country

mode = st.radio("Tarama modu", ["Canlı kamera", "Fotoğraf çek"], horizontal=True)

if mode == "Canlı kamera":
    # İlk başlatmada OCR modelinin yüklenmesi birkaç saniye sürebilir.
    reader = load_reader()
    start_worker(state, reader)

    def video_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        x1, y1, x2, y2 = center_region(img)
        now = time.monotonic()
        if now - state.last_frame > 0.95:
            state.last_frame = now
            item = (img[y1:y2, x1:x2].copy(), currency)
            try:
                state.queue.put_nowait(item)
            except Full:
                # İşleme yetişemiyorsak yeni kareyi at; kamera donmasın.
                pass

        # Ekranın çerçeve dışındaki kısmını karart.
        dark = (img * 0.42).astype(np.uint8)
        dark[y1:y2, x1:x2] = img[y1:y2, x1:x2]
        img = dark
        color = (120, 240, 100)  # BGR açık yeşil
        cv2.rectangle(img, (x1, y1), (x2, y2), color, max(2, img.shape[1] // 220))
        cv2.putText(img, "PRICE SCAN", (x1, max(26, y1 - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        with state.lock:
            detected = state.price
        if detected is not None:
            cv2.putText(img, f"{detected:,.0f} {currency}",
                        (x1, min(img.shape[0] - 16, y2 + 35)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="price-scan-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": {"facingMode": {"ideal": "environment"}},
            "audio": False,
        },
        video_frame_callback=video_callback,
        async_processing=True,
    )
    st.caption("Kamerada START'e basın. Çerçeve dışı karartılır; sadece yeşil alan taranır. Ön kamera açılırsa cihaz seçiminden arka kamerayı seçin.")

else:
    image_file = st.camera_input("Fiyatı ortadaki kutuya getirip fotoğraf çek")
    if image_file is not None:
        img_pil = Image.open(BytesIO(image_file.getvalue())).convert("RGB")
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        x1, y1, x2, y2 = center_region(img)
        cropped = img[y1:y2, x1:x2]
        st.image(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), caption="Taranan orta bölge", use_container_width=True)
        detected, raw = scan_crop(cropped, currency, load_reader())
        if detected is not None:
            with state.lock:
                state.price, state.raw = detected, raw
            st.success(f"Algılanan fiyat: {format_tr(detected, 0 if currency in ZERO_DECIMAL else 2)} {currency}")
        else:
            st.warning("Ortadaki alanda bir fiyat okunamadı. Daha yakından ve ışıkta tekrar deneyin.")


@st.fragment(run_every="1s")
def show_results():
    with state.lock:
        scanned = state.price
        raw = state.raw
        status = state.status

    amount = Decimal(str(manual)) if manual > 0 else scanned
    if amount is None:
        st.info(status)
        return

    calc = estimate_refund(amount, rule, override_pct=override_pct)
    after_tax = amount - calc.refund
    original_tl = amount * fx
    after_tax_tl = after_tax * fx
    saved_tl = calc.refund * fx

    st.divider()
    st.caption(("Manuel tutar" if manual > 0 else "Kameradan okunan") + f": {format_tr(amount, 0 if currency in ZERO_DECIMAL else 2)} {currency}" + (f" (OCR: {raw})" if raw and manual == 0 else ""))
    st.metric("Normal fiyat (TL)", f"{format_tr(original_tl)} TL")
    if calc.status == "maximum":
        st.metric("Teorik tax free sonrası (en iyi senaryo)", f"{format_tr(after_tax_tl)} TL")
        st.caption("Bu tutar iade komisyonları düşülmeden hesaplanmıştır, fiilî tutar genellikle daha yüksek olur.")
    elif calc.status == "estimated":
        st.metric("Tahmini tax free sonrası (TL)", f"{format_tr(after_tax_tl)} TL")
        st.caption("Net iade varsayımıdır; gerçek oran işlem ve iade operatörüne bağlıdır.")
    elif calc.status == "ineligible":
        st.metric("Tax free sonrası (alt sınır sağlanmadı)", f"{format_tr(original_tl)} TL")
        st.warning("Fiyat, kayıtlı tax free alışveriş eşiğinin altında. Aynı fişte farklı ürünler varsa toplam tutar eşiği sağlayabilir.")
    else:
        st.metric("Tax free sonrası (genel iade yok)", f"{format_tr(original_tl)} TL")
    st.caption(
        f"İade ({'teorik üst sınır' if calc.status == 'maximum' else 'tahmin'}): "
        f"{format_tr(calc.refund, 0 if currency in ZERO_DECIMAL else 2)} {currency} "
        f"(%{format_tr(calc.pct_of_price, 2)}) ≈ {format_tr(saved_tl)} TL"
    )
    if country == "Güney Kore" and amount >= Decimal("1000000"):
        st.info("Bu fiş Kore'deki anında iade üst sınırının dışında; uygun alışverişlerde daha sonra iade süreci gerekebilir.")


show_results()
st.caption("⚠️ Döviz kuru online güncellenir; tax free ülke kuralları uygulamadaki statik bilgilerden gelir. Ürün ve mağaza uygunluğu, fiş toplamı, iade komisyonları ve bankanın kart kuru nedeniyle fiilî tutar değişebilir.")
