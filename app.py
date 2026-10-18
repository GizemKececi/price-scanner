"""Price Scanner — WebRTC gerektirmeyen akıcı kamera, doğrulamalı OCR, tam online kur."""

from __future__ import annotations

# macOS'taki yerel SSL sorunlarında sistem sertifikalarından yararlan.
import truststore
truststore.inject_into_ssl()

import html
import base64
import re
import threading
import time
from decimal import Decimal, ROUND_HALF_UP
from io import BytesIO
from queue import Empty, Full, Queue

import cv2
import easyocr
import numpy as np
import requests
import streamlit as st
from PIL import Image

from tax_rules import COUNTRIES, RULES_REVIEWED, estimate_refund
from price_ocr import scan_crop

st.set_page_config(page_title="Price Scanner", page_icon="🔎", layout="centered")

# Görünür kamera alanı kısa ve yataydır; burası OCR'de kullanılan merkez bant.
# iPhone'un dikey kamera görüntüsü önizlemede object-fit: cover ile ortadan kırpılır.
OCR_X_MIN, OCR_X_MAX = 0.085, 0.915
OCR_Y_MIN, OCR_Y_MAX = 0.32, 0.68
ZERO_DECIMAL = {"KRW", "JPY"}

st.markdown(
    """
    <style>
      header[data-testid="stHeader"] { height: 0; min-height: 0; display: none; }
      [data-testid="stToolbar"] { display: none; }
      #MainMenu { visibility: hidden; }
      .block-container { padding-top: 1.15rem; padding-bottom: 1.8rem; max-width: 610px; }
      [data-testid="stVerticalBlock"] { gap: .56rem; }
      /* Breathing room between app title, country selector and exchange rate. */
      div[data-testid="stSelectbox"] { margin-top: .30rem; margin-bottom: .12rem; }
      div[data-testid="stRadio"] { margin-top: .18rem; margin-bottom: .32rem; }
      div[data-testid="stMetric"] { padding: 0; }
      .ps-topline {display:flex; align-items:center; gap:.45rem; font-weight: 760;
        font-size: 1.24rem; letter-spacing: -.02em; line-height: 1.35;
        padding: .36rem 0 .56rem; margin: .14rem 0 .2rem;}
      .ps-title-icon {font-size:1.25rem;line-height:1;flex:none;}
      .ps-rate {font-size: .92rem; padding: .7rem .77rem;
        border: 1px solid rgba(128,128,128,.25); border-radius: 12px; line-height: 1.55;
        margin-top:.12rem; margin-bottom:.16rem;}
      .ps-meta {color: #9a9a9a; font-size: .75rem; margin-top: .22rem;line-height:1.5;}
      .ps-tip {font-size: .76rem; color: #888; margin-bottom: .1rem;}
      .ps-detected {font-size: .94rem; font-weight: 600; margin: .15rem 0 .36rem;}
      .ps-status {font-size: .82rem; color: #777; padding: .35rem .1rem;}
      .ps-values {display:grid; grid-template-columns:1fr 1fr; gap: .55rem; }
      .ps-value {border: 1px solid rgba(128,128,128,.22); border-radius: 10px; padding: .68rem .53rem; min-width:0;}
      .ps-value-label {font-size:.76rem; line-height:1.3; color:#888; min-height: 2.0em;}
      .ps-value-number {font-size: clamp(1.22rem, 5.3vw, 1.82rem); font-weight:750; line-height:1.35; white-space:nowrap; letter-spacing:-.035em;}
      .ps-saving {font-size:.79rem; color:#888; padding: .25rem 0 0;}
      @media (max-width: 640px) {
        .block-container {padding-top: 1rem; padding-left: 1rem; padding-right: 1rem;}
        .ps-topline {padding-bottom:.57rem;}
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



def center_region(img: np.ndarray) -> tuple[int, int, int, int]:
    height, width = img.shape[:2]
    return (
        int(width * OCR_X_MIN), int(height * OCR_Y_MIN),
        int(width * OCR_X_MAX), int(height * OCR_Y_MAX),
    )



class ScanState:
    def __init__(self):
        self.lock = threading.Lock()
        self.queue: Queue = Queue(maxsize=1)
        self.price: Decimal | None = None
        self.raw = ""
        self.worker_started = False
        self.currency = "KRW"
        self.country = "Güney Kore"
        self.proposed: Decimal | None = None
        self.proposed_hits = 0
        self.proposed_at = 0.0
        self.last_frame_received = 0.0
        self.last_ocr_at = 0.0
        self.frames_received = 0
        self.confirmed_at = 0.0
        self.status = "Kamerayı başlatıp fiyatı yeşil çerçevenin içine getir."


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
                reading = scan_crop(crop, selected_currency, reader)
                now = time.monotonic()
                with state.lock:
                    if selected_currency != state.currency:
                        continue
                    state.last_ocr_at = now
                    number, raw = reading.amount, reading.raw
                    if number is None:
                        # Bir süre net fiyat yoksa eski ürünün fiyatını gösterme.
                        if now - state.proposed_at > 7:
                            state.proposed = None
                            state.proposed_hits = 0
                        if state.price is not None and now - state.confirmed_at > 9:
                            state.price = None
                            state.raw = ""
                        state.status = reading.note or "OCR net bir fiyat bulamadı; tekrar deniyor."
                        continue

                    if number != state.proposed or now - state.proposed_at > 9:
                        # Başka fiyata geçildiğinde eski tutarı geçici olarak kaldır:
                        # ekran yeni ürünün fiyatını eskisiymiş gibi göstermesin.
                        if state.price is not None and number != state.price:
                            state.price = None
                            state.raw = ""
                        state.proposed = number
                        state.proposed_hits = 1
                    else:
                        state.proposed_hits += 1
                    state.proposed_at = now

                    # Aynı OCR hatasını yeniden okumak tek başına kesin garanti
                    # olamaz. Para simgesi önce OCR'de korundu, şüpheli 1xxxx
                    # tutarlar kontrastlı görüntüyle çapraz kontrol edildi.
                    required_hits = 2
                    if number == state.price:
                        state.confirmed_at = now
                        state.status = "✅ Fiyat doğrulandı"
                    elif state.proposed_hits >= required_hits:
                        state.price = number
                        state.raw = raw
                        state.confirmed_at = now
                        state.status = "✅ Fiyat doğrulandı"
                    else:
                        state.status = (
                            f"{format_tr(number, 0 if selected_currency in ZERO_DECIMAL else 2)} "
                            f"{selected_currency} doğrulanıyor "
                            f"({state.proposed_hits}/{required_hits})…"
                        )
            except Exception as exc:
                with state.lock:
                    # Hataları sessizce yutma; telefon ekranından anlaşılabilsin.
                    state.last_ocr_at = time.monotonic()
                    state.status = f"OCR hatası: {type(exc).__name__}: {str(exc)[:90]}"

    threading.Thread(target=run, name="price_scanner_ocr_worker", daemon=True).start()


# --- Ülke ve kur: sık kullanılan bilgiler yukarıda, ayrıntılar en altta ---
st.markdown('<div class="ps-topline"><span class="ps-title-icon">🔎</span><span>Price Scanner</span></div>', unsafe_allow_html=True)
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

if "scanner_state_v8" not in st.session_state:
    st.session_state.scanner_state_v8 = ScanState()
state: ScanState = st.session_state.scanner_state_v8

with state.lock:
    if state.country != country or state.currency != currency:
        state.country, state.currency = country, currency
        state.price = None
        state.raw = ""
        state.proposed = None
        state.proposed_hits = 0
        state.confirmed_at = 0.0
        state.status = "Ülke değişti. Yeni para birimi için fiyatı okut."
        # Ülke değişince önceden kalan OCR karelerini temizle.
        while not state.queue.empty():
            try:
                state.queue.get_nowait()
            except Empty:
                break

mode = st.radio("Tarama şekli", ["Canlı kamera", "Fotoğraf çek"], horizontal=True, label_visibility="collapsed")

# Kamerayı WebRTC ile sunucuya bağlamak yerine tarayıcının getUserMedia
# API'siyle DOĞRUDAN telefonda açıyoruz. CANLI ÖNİZLEME ağ gecikmesinden etkilenmez.
# Sunucuya sadece yeşil çerçevenin içindeki JPEG kareler gider.
CAMERA_HTML = """
<div class="ps-camera-wrap">
  <video id="ps-local-video" muted autoplay playsinline webkit-playsinline></video>
  <div id="ps-guide"><span>FİYATI BURAYA GETİR</span></div>
  <div id="ps-camera-message">Kamerayı başlatmak için düğmeye dokun.</div>
</div>
<div class="ps-controls">
  <button type="button" id="ps-start">📷 Kamerayı aç</button>
  <button type="button" id="ps-stop" hidden>■ Durdur</button>
  <span id="ps-camera-status" role="status" aria-live="polite">Hazır</span>
</div>
"""

CAMERA_CSS = """
.ps-camera-wrap {width:100%;height:225px;position:relative;overflow:hidden;
  background:#111;border-radius:10px;border:1px solid #444;}
#ps-local-video {width:100%;height:100%;object-fit:cover;display:block;}
#ps-guide {pointer-events:none;position:absolute;left:8.5%;right:8.5%;top:18%;bottom:18%;
  border:2px solid #4be49a;border-radius:8px;box-shadow:0 0 0 800px rgba(0,0,0,.26);}
#ps-guide span {position:absolute;top:-1.35em;left:0;color:#5bf9ac;font-size:11px;
  font-weight:700;letter-spacing:.06em;text-shadow:0 1px 2px #000;}
#ps-camera-message {position:absolute;left:10%;right:10%;top:40%;
  text-align:center;color:#ddd;font-size:13px;}
.ps-controls {display:flex;align-items:center;gap:10px;padding-top:7px;}
.ps-controls button {border:0;border-radius:7px;background:#ee5959;color:white;
  padding:7px 13px;font-size:13px;font-weight:600;min-height:35px;cursor:pointer;}
.ps-controls button[hidden] {display:none;}
#ps-camera-status {font-size:12px;color:#999;}
"""

CAMERA_JS = r"""
export default function (component) {
  const root = component.parentElement;
  const video = root.querySelector('#ps-local-video');
  const msg = root.querySelector('#ps-camera-message');
  const startBtn = root.querySelector('#ps-start');
  const stopBtn = root.querySelector('#ps-stop');
  const status = root.querySelector('#ps-camera-status');
  const guide = root.querySelector('#ps-guide');
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d', {willReadFrequently:false});
  let stream = null;
  let timer = null;
  let disposed = false;

  function stopCamera() {
    if (timer !== null) { clearInterval(timer); timer = null; }
    if (stream) { stream.getTracks().forEach(track => track.stop()); stream = null; }
    video.srcObject = null;
    startBtn.hidden = false;
    stopBtn.hidden = true;
    msg.style.display = 'block';
    status.textContent = 'Durduruldu';
  }

  // Görüntüdeki yeşil alanı birebir al: object-fit:cover kırpmasını hesaba kat.
  function capture() {
    if (!stream || video.readyState < 2 || !video.videoWidth || !video.videoHeight) return;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    const cw = video.clientWidth;
    const ch = video.clientHeight;
    if (!cw || !ch) return;
    const scale = Math.max(cw / vw, ch / vh);
    const visibleW = cw / scale;
    const visibleH = ch / scale;
    const offX = (vw - visibleW) / 2;
    const offY = (vh - visibleH) / 2;
    const gx = guide.offsetLeft / scale;
    const gy = guide.offsetTop / scale;
    const gw = guide.clientWidth / scale;
    const gh = guide.clientHeight / scale;
    const sx = Math.max(0, offX + gx);
    const sy = Math.max(0, offY + gy);
    const sw = Math.max(1, Math.min(vw - sx, gw));
    const sh = Math.max(1, Math.min(vh - sy, gh));
    // Kırmızı/tırtıklı farklı fontlarda rakam çizgileri korunabilsin.
    const factor = Math.min(1, 1600 / sw);
    canvas.width = Math.max(1, Math.floor(sw * factor));
    canvas.height = Math.max(1, Math.floor(sh * factor));
    ctx.drawImage(video, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
    const jpeg = canvas.toDataURL('image/jpeg', 0.88);
    component.setTriggerValue('frame', jpeg.split(',')[1]);
    status.textContent = 'Fiyat kontrol ediliyor…';
  }

  async function startCamera() {
    if (stream) return;
    startBtn.disabled = true;
    status.textContent = 'Kamera açılıyor…';
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: {facingMode:{ideal:'environment'}, width:{ideal:1280}, height:{ideal:720}},
        audio: false
      });
      if (disposed) {s.getTracks().forEach(track=>track.stop());return;}
      stream = s;
      video.srcObject = s;
      await video.play();
      msg.style.display = 'none';
      startBtn.hidden = true;
      stopBtn.hidden = false;
      status.textContent = 'Canlı kamera açık';
      timer = setInterval(capture, 1550);
      setTimeout(capture, 500);
    } catch(err) {
      status.textContent = 'Kamera açılamadı: ' + (err?.name || 'izin veya tarayıcı');
      msg.textContent = 'Safari kamera iznini kontrol et. Gerekirse Fotoğraf çek modunu kullan.';
    } finally { startBtn.disabled = false; }
  }
  startBtn.addEventListener('click', startCamera);
  stopBtn.addEventListener('click', stopCamera);
  return () => {disposed = true;stopCamera();};
}
"""

if mode == "Canlı kamera":
    # Tamamen tarayıcı tabanlı önizleme: WebRTC, STUN veya TURN gerektirmez.
    reader = load_reader()
    start_worker(state, reader)

    def on_browser_frame():
        payload = st.session_state.get("ps_browser_cam_v8")
        # Components v2 değerleri Streamlit sürümüne göre dict veya
        # attribute erişimli sonuç nesnesi olabiliyor.
        if isinstance(payload, dict):
            raw_b64 = payload.get("frame")
        else:
            raw_b64 = getattr(payload, "frame", None) if payload is not None else None
        if not isinstance(raw_b64, str) or len(raw_b64) > 3_000_000:
            return
        try:
            decoded = base64.b64decode(raw_b64, validate=True)
            arr = cv2.imdecode(np.frombuffer(decoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            if arr is not None:
                with state.lock:
                    selected_currency = state.currency
                with state.lock:
                    state.frames_received += 1
                    state.last_frame_received = time.monotonic()
                enqueue_fresh(state, (arr, selected_currency))
        except (ValueError, TypeError):
            pass

    local_camera = st.components.v2.component(
        "price_scanner_browser_camera_v8",
        html=CAMERA_HTML, css=CAMERA_CSS, js=CAMERA_JS,
    )
    local_camera(
        key="ps_browser_cam_v8",
        on_frame_change=on_browser_frame,
        height=282,
    )
    st.markdown(
        '<div class="ps-tip">Telefon ekranında canlı önizleme · fiyatı yeşil çerçeveye getir. '
        'Yalnızca rakamlar ve ayraçlar (.,) hesaplanır. Diğer semboller fiyat değildir. Birden fazla rakamda merkezdeki belirgin fiyat aranır.</div>',
        unsafe_allow_html=True,
    )
else:
    image_file = st.camera_input("Fiyatı ortalayarak fotoğraf çek")
    if image_file is not None:
        pil = Image.open(BytesIO(image_file.getvalue())).convert("RGB")
        img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        x1, y1, x2, y2 = center_region(img)
        cropped = img[y1:y2, x1:x2]
        # Fotoğraf modunda görüntü sabittir; şüpheli OCR çapraz kontrol edilir.
        reading = scan_crop(cropped, currency, load_reader())
        if reading.amount is None:
            reading = scan_crop(img, currency, load_reader())
        if reading.amount is not None:
            with state.lock:
                state.price, state.raw = reading.amount, reading.raw
                state.confirmed_at = time.monotonic()
                state.status = "✅ Fotoğraftan fiyat okundu"
        else:
            st.warning(reading.note or "Fiyat bulunamadı; yakından tekrar çek.")


def override_pct_value() -> Decimal | None:
    enabled = st.session_state.get(f"ps_custom_enabled_{country}", False)
    if not enabled:
        return None
    return Decimal(str(st.session_state.get(f"ps_custom_pct_{country}", 0.0)))


@st.fragment(run_every="1s")
def show_results():
    with state.lock:
        scanned, raw, status = state.price, state.raw, state.status
        received, last_seen = state.frames_received, state.last_frame_received
        last_ocr = state.last_ocr_at
    if mode == "Canlı kamera" and last_seen:
        wait = time.monotonic() - last_seen
        if wait > 9:
            status = "Kameradan yeni kare gelmiyor. Durdurup kamerayı yeniden aç."
        elif not last_ocr:
            status = "Kare geldi; ilk OCR hazırlanıyor…"

    # OCR henüz bir fiyat bulamadıysa da manuel fiyatın kullanılabilmesi gerekir.
    manual_value = Decimal(str(st.session_state.get(f"ps_manual_{currency}", 0.0)))
    if scanned is None and manual_value <= 0:
        st.markdown(f'<div class="ps-status">{html.escape(status)}</div>', unsafe_allow_html=True)
        if mode == "Canlı kamera" and received:
            st.caption(f"Kameradan {received} kare alındı. Sunucu OCR durumunu yukarıda gösterir.")
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
