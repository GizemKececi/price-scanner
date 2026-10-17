"""Price Scanner — kompakt mobil kamera, doğrulamalı OCR ve tam gösterilen online kur."""

from __future__ import annotations

# macOS'taki yerel SSL sorunlarında sistem sertifikalarından yararlan.
import truststore
truststore.inject_into_ssl()

import html
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

# Görünür kamera alanı kısa ve yataydır; burası OCR'de kullanılan merkez bant.
# iPhone'un dikey kamera görüntüsü önizlemede object-fit: cover ile ortadan kırpılır.
OCR_X_MIN, OCR_X_MAX = 0.08, 0.92
OCR_Y_MIN, OCR_Y_MAX = 0.31, 0.69
OCR_INTERVAL_SECONDS = 0.95
ZERO_DECIMAL = {"KRW", "JPY"}
NUMBERS = re.compile(r"(?<!\d)\d[\d.,\s]*\d|(?<!\d)\d(?!\d)")

st.markdown(
    """
    <style>
      header[data-testid="stHeader"] { height: 0; min-height: 0; display: none; }
      [data-testid="stToolbar"] { display: none; }
      #MainMenu { visibility: hidden; }
      .block-container { padding-top: .7rem; padding-bottom: 1.5rem; max-width: 610px; }
      [data-testid="stVerticalBlock"] { gap: .42rem; }
      div[data-testid="stMetric"] { padding: 0; }
      .ps-topline {font-weight: 700; font-size: 1.17rem; line-height: 1.3; margin: .1rem 0 .2rem; }
      .ps-rate {font-size: .92rem; padding: .34rem .56rem; border: 1px solid rgba(128,128,128,.25); border-radius: 9px; line-height: 1.5; }
      .ps-meta {color: #888; font-size: .74rem; margin-top: .1rem;}
      .ps-tip {font-size: .76rem; color: #888; margin-bottom: .1rem;}
      .ps-detected {font-size: .94rem; font-weight: 600; margin: .15rem 0 .36rem;}
      .ps-status {font-size: .82rem; color: #777; padding: .35rem .1rem;}
      .ps-values {display:grid; grid-template-columns:1fr 1fr; gap: .55rem; }
      .ps-value {border: 1px solid rgba(128,128,128,.22); border-radius: 10px; padding: .68rem .53rem; min-width:0;}
      .ps-value-label {font-size:.76rem; line-height:1.3; color:#888; min-height: 2.0em;}
      .ps-value-number {font-size: clamp(1.22rem, 5.3vw, 1.82rem); font-weight:750; line-height:1.35; white-space:nowrap; letter-spacing:-.035em;}
      .ps-saving {font-size:.79rem; color:#888; padding: .25rem 0 0;}
      @media (max-width: 640px) {
        .block-container {padding-top: .35rem; padding-left: .85rem; padding-right: .85rem;}
        .ps-value {padding: .54rem .48rem;}
        .ps-value-label {font-size:.72rem;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_tr(value: Decimal, places: int = 2) -> str:
    quantizer = Decimal("1") if places == 0 else Decimal("0.01")
    s = f"{value.quantize(quantizer, rounding=ROUND_HALF_UP):,.{places}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def format_fx_exact(value: Decimal) -> str:
    """Asla yuvarlama: API'den gelip matematikte kullanılan Decimal'ın tüm basamakları."""
    return format(value, "f").replace(".", ",")


@st.cache_data(ttl=900, show_spinner=False)
def fetch_fx(currency: str) -> tuple[Decimal, str]:
    """Frankfurter'ın son yayımladığı günlük referans kuru, 1 birim = kaç TRY."""
    response = requests.get(
        f"https://api.frankfurter.dev/v2/rate/{currency}/TRY",
        timeout=12,
    )
    response.raise_for_status()
    data = response.json()
    return Decimal(str(data["rate"])), data["date"]


@st.cache_resource(show_spinner="OCR modeli hazırlanıyor…")
def load_reader():
    return easyocr.Reader(["en"], gpu=False)


def parse_amount(raw: str, currency: str) -> Decimal | None:
    digits = re.sub(r"[^0-9]", "", raw)
    if not digits:
        return None
    raw = raw.strip().replace(" ", "")
    if currency in ZERO_DECIMAL:
        return Decimal(digits)
    separator = max(raw.rfind(","), raw.rfind("."))
    if separator >= 0 and len(raw) - separator - 1 in (1, 2):
        decimals = len(raw) - separator - 1
        return Decimal(digits[:-decimals] + "." + digits[-decimals:])
    return Decimal(digits)


def center_region(img: np.ndarray) -> tuple[int, int, int, int]:
    height, width = img.shape[:2]
    return (
        int(width * OCR_X_MIN), int(height * OCR_Y_MIN),
        int(width * OCR_X_MAX), int(height * OCR_Y_MAX),
    )


def scan_crop(bgr: np.ndarray, currency: str, reader) -> tuple[Decimal | None, str]:
    """Güvenilir ve uzun rakam dizisini seç. OCR doğrulamasını asla atlama."""
    if bgr.size == 0:
        return None, ""
    h, w = bgr.shape[:2]
    if w < 1100:
        scale = min(2.0, 1100 / max(w, 1))
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    results = reader.readtext(
        rgb,
        detail=1,
        paragraph=False,
        allowlist="0123456789., ",
    )

    candidates = []
    for box, text, confidence in results:
        if confidence < 0.30:
            continue
        for match in NUMBERS.finditer(text):
            raw = match.group().strip()
            digit_count = sum(c.isdigit() for c in raw)
            # Won/Yen etiketinde tek hane / 46 gibi parçalı OCR okumaları fiyat sayma.
            min_digits = 3 if currency in ZERO_DECIMAL else 2
            if digit_count < min_digits:
                continue
            amount = parse_amount(raw, currency)
            if amount is None or amount <= 0:
                continue
            _x, _y, area_w, area_h = cv2.boundingRect(np.asarray(box, dtype=np.float32))
            candidates.append((digit_count, area_w * area_h, confidence, amount, raw))

    if not candidates:
        return None, ""
    *_, value, raw = max(candidates, key=lambda x: x[:3])
    return value, raw


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
        self.proposed: Decimal | None = None
        self.proposed_hits = 0
        self.proposed_at = 0.0
        self.status = "START'e bas ve fiyatı yeşil alanın ortasına getir."


def enqueue_fresh(state: ScanState, item):
    """OCR yoğunken sırada bayat kare biriktirme, en son kadrajı işle."""
    try:
        state.queue.put_nowait(item)
        return
    except Full:
        pass
    try:
        state.queue.get_nowait()
    except Empty:
        pass
    try:
        state.queue.put_nowait(item)
    except Full:
        pass


def start_worker(state: ScanState, reader):
    with state.lock:
        if state.worker_started:
            return
        state.worker_started = True

    def run():
        while True:
            try:
                crop, selected_currency = state.queue.get(timeout=2)
            except Empty:
                continue
            try:
                number, raw = scan_crop(crop, selected_currency, reader)
                now = time.monotonic()
                with state.lock:
                    if selected_currency != state.currency:
                        continue
                    if number is None:
                        if now - state.proposed_at > 5:
                            state.proposed = None
                            state.proposed_hits = 0
                        if state.price is None:
                            state.status = "Henüz net fiyat yok. Biraz yaklaştırıp sabit tut."
                        continue

                    if number != state.proposed or (now - state.proposed_at > 5):
                        state.proposed = number
                        state.proposed_hits = 1
                    else:
                        state.proposed_hits += 1
                    state.proposed_at = now

                    # İlk tanıma iki tutarlı OCR okuması gerektirir.
                    # Daha önce onaylı fiyat varken onu değiştirmek için üç okuma gerekir.
                    required_hits = 2 if state.price is None else 3
                    if number == state.price:
                        state.status = "✅ Fiyat doğrulandı"
                    elif state.proposed_hits >= required_hits:
                        state.price = number
                        state.raw = raw
                        state.status = "✅ Fiyat doğrulandı"
                    else:
                        state.status = (
                            f"{format_tr(number, 0 if selected_currency in ZERO_DECIMAL else 2)} "
                            f"{selected_currency} doğrulanıyor "
                            f"({state.proposed_hits}/{required_hits})…"
                        )
            except Exception as exc:
                with state.lock:
                    state.status = f"OCR hatası: {type(exc).__name__}"

    threading.Thread(target=run, name="price_scanner_ocr_worker", daemon=True).start()


# --- Ülke ve kur: sık kullanılan bilgiler yukarıda, ayrıntılar en altta ---
st.markdown('<div class="ps-topline">🔎 Price Scanner</div>', unsafe_allow_html=True)
country = st.selectbox(
    "Nereye seyahat ediyorsun?",
    list(COUNTRIES),
    format_func=lambda name: COUNTRIES[name].label,
    label_visibility="collapsed",
)
rule = COUNTRIES[country]
currency = rule.currency

try:
    fx, fx_date = fetch_fx(currency)
except (requests.RequestException, KeyError, ValueError, TypeError):
    st.error("Güncel online kura erişilemedi; yanlış veya eski kurla hesaplama yapılmıyor.")
    st.stop()

# Ekranda yazan fx ile TL hesaplamasında kullanılan fx AYNI Decimal nesnesidir.
st.markdown(
    '<div class="ps-rate">'
    f'💱 <strong>1 {html.escape(currency)} = {html.escape(format_fx_exact(fx))} TL</strong>'
    f'<div class="ps-meta">Frankfurter · {html.escape(fx_date)} referans kuru · 15 dk önbellek</div>'
    '</div>',
    unsafe_allow_html=True,
)

if "scanner_state_v3" not in st.session_state:
    st.session_state.scanner_state_v3 = ScanState()
state: ScanState = st.session_state.scanner_state_v3

with state.lock:
    if state.country != country or state.currency != currency:
        state.country, state.currency = country, currency
        state.price = None
        state.raw = ""
        state.proposed = None
        state.proposed_hits = 0
        state.status = "Ülke değişti. Yeni para birimi için fiyatı okut."
        # Ülke değişince önceden kalan OCR karelerini temizle.
        while not state.queue.empty():
            try:
                state.queue.get_nowait()
            except Empty:
                break

mode = st.radio("Tarama şekli", ["Canlı kamera", "Fotoğraf çek"], horizontal=True, label_visibility="collapsed")

if mode == "Canlı kamera":
    reader = load_reader()
    start_worker(state, reader)

    def video_callback(frame: av.VideoFrame) -> av.VideoFrame:
        # SENDONLY: Tarayıcı gerçek kamera önizlemesini kendi çizer, sunucu dönüşünü beklemez.
        now = time.monotonic()
        with state.lock:
            if now - state.last_frame < OCR_INTERVAL_SECONDS:
                return frame
            state.last_frame = now
            current_currency = state.currency
        img = frame.to_ndarray(format="bgr24")
        x1, y1, x2, y2 = center_region(img)
        enqueue_fresh(state, (img[y1:y2, x1:x2].copy(), current_currency))
        return frame

    webrtc_streamer(
        key="price-scanner-local-compact-v3",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": {
                "facingMode": {"ideal": "environment"},
                "width": {"ideal": 960},
                "height": {"ideal": 720},
                "frameRate": {"ideal": 24, "max": 30},
            },
            "audio": False,
        },
        video_html_attrs={
            "autoPlay": True,
            "controls": False,
            "muted": True,
            "style": {
                "width": "100%",
                "height": "225px",
                "objectFit": "cover",
                "border": "2px solid #43d790",
                "borderRadius": "10px",
                "background": "#121212",
            },
        },
        video_frame_callback=video_callback,
        async_processing=True,
    )
    st.markdown(
        '<div class="ps-tip">Yeşil çerçevenin ortasındaki fiyatı 2–3 saniye sabit tut · '
        'OCR iki okumayla doğrular, fiyat değişimini üç okumayla doğrular.</div>',
        unsafe_allow_html=True,
    )
else:
    image_file = st.camera_input("Fiyatı ortalayarak fotoğraf çek")
    if image_file is not None:
        pil = Image.open(BytesIO(image_file.getvalue())).convert("RGB")
        img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        x1, y1, x2, y2 = center_region(img)
        cropped = img[y1:y2, x1:x2]
        # Fotoğraf modunda kullanıcı manuel çektiği için ilk bölgede tek okuma yeterli;
        # okunamazsa tam kareyi de dener. Canlı kamerada iki/üç teyit korunur.
        detected, raw = scan_crop(cropped, currency, load_reader())
        if detected is None:
            detected, raw = scan_crop(img, currency, load_reader())
        if detected is not None:
            with state.lock:
                state.price, state.raw = detected, raw
                state.status = "✅ Fotoğraftan fiyat okundu"
        else:
            st.warning("Fiyat bulunamadı; daha yakından, aydınlıkta tekrar çek.")


@st.fragment(run_every="1s")
def show_results():
    with state.lock:
        scanned, raw, status = state.price, state.raw, state.status

    # OCR henüz bir fiyat bulamadıysa da manuel fiyatın kullanılabilmesi gerekir.
    manual_value = Decimal(str(st.session_state.get(f"ps_manual_{currency}", 0.0)))
    if scanned is None and manual_value <= 0:
        st.markdown(f'<div class="ps-status">{html.escape(status)}</div>', unsafe_allow_html=True)
        return

    # Fiyat elle girilmişse, OCR'den gelen veriyi silmeden bu tutarı hesapla.
    amount = manual_value if manual_value > 0 else scanned
    calc = estimate_refund(amount, rule, override_pct=override_pct_value())
    initial_tl = amount * fx
    discounted_tl = (amount - calc.refund) * fx
    saving_tl = calc.refund * fx
    amount_str = format_tr(amount, 0 if currency in ZERO_DECIMAL else 2)
    mode_label = "✏️ Elle girilen" if manual_value > 0 else "✅ Okunan"

    if calc.status == "maximum":
        tax_label = "Tax free teorik (TL)"
    elif calc.status == "estimated":
        tax_label = "Tax free sonrası (TL)"
    elif calc.status == "ineligible":
        tax_label = "Tax free eşiği altında"
    else:
        tax_label = "Vergi iadesi yok (TL)"

    st.markdown(
        f'<div class="ps-detected">{mode_label}: {html.escape(amount_str)} {html.escape(currency)}'
        f' <span style="font-weight:400;color:#888;font-size:.75rem;">· {html.escape(status) if manual_value <= 0 else "manuel tutar"}</span></div>'
        '<div class="ps-values">'
        '<div class="ps-value">'
        '<div class="ps-value-label">Normal fiyat (TL)</div>'
        f'<div class="ps-value-number">{html.escape(format_tr(initial_tl))} TL</div>'
        '</div>'
        '<div class="ps-value">'
        f'<div class="ps-value-label">{html.escape(tax_label)}</div>'
        f'<div class="ps-value-number">{html.escape(format_tr(discounted_tl))} TL</div>'
        '</div>'
        '</div>'
        '<div class="ps-saving">'
        f'Tahmini iade: {html.escape(format_tr(calc.refund, 0 if currency in ZERO_DECIMAL else 2))} '
        f'{html.escape(currency)} = {html.escape(format_tr(saving_tl))} TL'
        '</div>',
        unsafe_allow_html=True,
    )
    if calc.status == "maximum":
        st.caption("Teorik maksimumdur; iade operatörü kesinti yapabilir.")
    elif calc.status == "ineligible":
        st.caption("Tek ürün tutarı iade alt sınırında değil; fiş toplamı farklı olabilir.")
    elif calc.status == "estimated":
        st.caption("Vergi iadesi tahminidir; mağaza ve iade operatörüne göre değişebilir.")


# İade oranı opsiyonel gelişmiş ayar. Widget değeri fragment'ten okunabilsin diye
# varsayılanı Session State'te tutuyoruz; uygulama ilk kez çalışırken de hazır.
def override_pct_value() -> Decimal | None:
    enabled = st.session_state.get(f"ps_custom_enabled_{country}", False)
    if not enabled:
        return None
    return Decimal(str(st.session_state.get(f"ps_custom_pct_{country}", 0.0)))


show_results()

with st.expander("⚙️ Fiyat düzeltme, kur yenileme ve tax free ayrıntıları"):
    typed = st.number_input(
        f"Fiyatı elle gir ({currency}) — 0 = kamerayı kullan",
        min_value=0.0, value=0.0,
        step=1000.0 if currency in ZERO_DECIMAL else 10.0,
        key=f"ps_manual_{currency}",
    )
    if st.button("🔄 Kuru güncelle", use_container_width=True):
        fetch_fx.clear()
        st.rerun()
    st.caption("Para birimi otomatik belirlenir. Frankfurter günlük referans kurudur; banka/kart kuru farklı olabilir.")
    st.write(rule.summary)
    st.write(rule.eligibility_note)
    st.caption(f"Tax free kurallarının manuel kontrolü: {RULES_REVIEWED}.")
    for index, source in enumerate(rule.sources, 1):
        st.markdown(f"[Tax free resmî kaynak {index}]({source})")
    max_pct = float(rule.vat / (Decimal("100") + rule.vat) * Decimal("100"))
    st.checkbox(
        "Tahmini net iadeyi elle değiştirmek istiyorum",
        key=f"ps_custom_enabled_{country}",
        disabled=rule.method == "unavailable",
    )
    st.number_input(
        "Etiket fiyatının yüzde kaçı iade?",
        min_value=0.0,
        max_value=max_pct,
        value=float(rule.estimated_net_pct or Decimal(str(max_pct))),
        step=0.1,
        key=f"ps_custom_pct_{country}",
        disabled=not st.session_state.get(f"ps_custom_enabled_{country}", False) or rule.method == "unavailable",
    )

st.caption("Online referans kur ve tahmini tax free. Ürün, fiş, operatör kesintisi veya kart kuru nedeniyle gerçek tutar değişebilir.")
