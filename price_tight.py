"""Live OCR for exactly one complete price line inside the green guide.

The colored guide is a user-selected ROI, *not* a document-scanning hint.
The text in that ROI must be localized and cropped vertically BEFORE passing
it to EasyOCR's whole-image recognizer. Recognizing a tall, mostly-empty ROI
compresses small numbers and may drop leading digits (e.g. 72,000 -> 4,000).

A one-frame result is provisional; the app compares independent frames with
alternating preprocessing. Currency words/symbols are allowed around a numeric
value but are never used as numeric digits.
"""
from __future__ import annotations

import re
from decimal import Decimal

import cv2
import numpy as np

from price_ocr import Reading, _candidates, _pick, _suspect, _recognition_image


def _focus_price_line(bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    """Find the main text row using lightweight morphology (no OCR detector).

    Keep the ENTIRE x-range, including the first and last digits. Only trim
    whitespace above/below the central line. A very conservative fall-back
    returns the input untouched if no reliable row can be found.
    """
    h, w = bgr.shape[:2]
    if h < 50 or w < 100:
        return bgr, False
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Locally adaptive binarization works with backlit labels, red prices,
    # neutral digits, and shaded backgrounds without assuming a font color.
    block = max(15, min(51, (min(h, w) // 4) | 1))
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, block, 13)
    # Remove image texture/points that cannot possibly be a digit stroke.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    cleaned = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if (area >= max(5, h // 35) and 3 <= cw <= w * .50
                and max(5, h * .065) <= ch <= h * .72):
            cleaned[y:y+ch, x:x+cw][labels[y:y+ch, x:x+cw] == i] = 255
    if cv2.countNonZero(cleaned) < 30:
        return bgr, False

    # Prefer the bold price row around the middle of a shallow guide. Other
    # rows are allowed in the frame but should not affect this recognition.
    center = cleaned[:, int(w * .035):int(w * .965)]
    rows = np.count_nonzero(center, axis=1).astype(np.float32)
    rows = cv2.GaussianBlur(rows.reshape(-1, 1), (1, 0), max(2, h * .025)).ravel()
    yy = np.arange(h)
    center_weight = np.exp(-.5 * ((yy - h*.5) / max(1, h*.27))**2)
    peak = int(np.argmax(rows * (.65 + .35 * center_weight)))
    strength = float(rows[peak])
    if strength < max(3, w * .017):
        return bgr, False
    cutoff = max(2.5, strength * .27)
    lo, hi = peak, peak
    while lo > 0 and rows[lo-1] >= cutoff:
        lo -= 1
    while hi < h-1 and rows[hi+1] >= cutoff:
        hi += 1
    # Expand to include digit ascenders/descenders, then add a little padding.
    pad = max(8, int(h*.055))
    top, bottom = max(0, lo-pad), min(h, hi+pad+1)
    band_height = bottom-top
    if not (max(18,h*.12) <= band_height <= h*.85):
        return bgr, False
    # Avoid cropping a faint first row when stronger ink exists near top/bottom.
    # Whole width is always preserved; the digits on the left are never clipped.
    return bgr[top:bottom, :].copy(), True


def _recognition_input(bgr: np.ndarray, attempt: int) -> tuple[np.ndarray, bool]:
    line, focused = _focus_price_line(bgr)
    # Change contrast between independent frames rather than two OCR passes.
    if attempt % 3 == 1:
        kind = 'contrast'
    else:
        kind = 'original'
    img = _recognition_image(line, kind, fast=True)
    # EasyOCR normalizes input height; preserving line proportions prevents
    # 5-digit prices being downsampled into unreadably skinny characters.
    h, w = img.shape[:2]
    if h < 60 and w:
        factor = min(2.25, 60 / max(h, 1), 1200 / w)
        if factor > 1.02:
            img = cv2.resize(img, None, fx=factor, fy=factor,
                             interpolation=cv2.INTER_CUBIC)
    return img, focused


def _recognize(img_rgb: np.ndarray, reader):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return reader.recognize(gray, decoder='greedy', detail=1,
                            paragraph=False, batch_size=1)


def _digits(raw: str) -> int:
    return len(re.sub(r'\D', '', raw))


def scan_tight_fast(bgr: np.ndarray, currency: str, reader, attempt: int = 0) -> Reading:
    if bgr is None or not bgr.size:
        return Reading(None, note='Kamera görüntüsü gelmedi')

    image, focused = _recognition_input(bgr, attempt)
    try:
        # Two quick recognizer attempts with different visual preprocessing;
        # only the third, when earlier ones failed, uses CRAFT text detection.
        if attempt % 3 == 2 or not hasattr(reader, 'recognize'):
            results = reader.readtext(image, detail=1, paragraph=False,
                                      width_ths=.20, canvas_size=920, mag_ratio=1)
        else:
            results = _recognize(image, reader)
    except (TypeError, ValueError, AttributeError):
        results = reader.readtext(image, detail=1, paragraph=False,
                                  width_ths=.20, canvas_size=920, mag_ratio=1)

    candidates = _candidates(results, currency, image.shape, image)
    picked, rivals = _pick(candidates)
    if picked is None:
        return Reading(None, note='Birden çok sayı var; fiyatı tek başına ortala' if rivals
                       else 'Tüm fiyatı kutuya sığdırıp biraz yaklaştır',
                       alternatives=rivals)

    if picked.crossed or _suspect(picked, currency, results):
        return Reading(None, note='Şüpheli veya eski fiyat; tekrar okunuyor')

    # Single tight crop: require a FULL grouped amount, not a floating tail
    # of digits falsely interpreted as an independent number. EasyOCR may
    # break a price into multiple tokens: if there is evidence of a missing
    # left fragment, don't publish the small tail as confirmed price.
    digits = _digits(picked.raw)
    if currency in ('KRW', 'JPY') and digits < 3:
        return Reading(None, note='Eksik rakam; tüm fiyatı ortala')
    threshold = .58 if picked.currency_near else (.68 if focused else .76)
    if picked.confidence < threshold:
        return Reading(None, note='Rakam yeterince net değil; biraz yaklaş')

    # If recognizer returned multiple independent numbers, reject a partial
    # suffix even if the ranker favored it. Different-number ambiguity can be
    # resolved by the user's tighter framing, not speculative concatenation.
    if len(candidates) > 1:
        other = [c for c in candidates if c.value != picked.value and not c.crossed]
        for c in other:
            ax0, ay0, ax1, ay1 = picked.bounds
            bx0, by0, bx1, by1 = c.bounds
            aligned = min(ay1, by1)-max(ay0, by0) >= .45*min(ay1-ay0,by1-by0)
            close = 0 <= ax0-bx1 < max(12, ay1-ay0)*.8
            if aligned and close and not c.currency_near:
                return Reading(None, note='Fiyatın ilk rakamları ayrı okunmuş; yeniden deneniyor')

    return Reading(picked.value, picked.raw, picked.confidence,
                   note='Tüm fiyat satırı okundu')
