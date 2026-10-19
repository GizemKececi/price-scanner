"""Live OCR for exactly one complete price line inside the green guide.

The colored guide is a user-selected ROI, *not* a document-scanning hint.
The text in that ROI must be localized and cropped vertically BEFORE passing
it to EasyOCR's whole-image recognizer. Recognizing a tall, mostly-empty ROI
compresses small numbers and may drop leading digits (e.g. 72,000 -> 4,000).

A one-frame result is provisional; the app compares independent frames with
alternating preprocessing. Currency words/symbols are allowed around a numeric
value but are never used as numeric digits.

v12: the vertical focus step used to drop glyphs taller than 72 % of the crop,
so a price that FILLED the guide band reached the OCR as a thin slice of its
bottom (72,400 -> "4,900"). It now never rejects tall glyphs, keeps every glyph
row, adds a margin, and refuses prices that run into the guide's edge.
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from decimal import Decimal

import cv2
import numpy as np

from price_ocr import Reading, _candidates, _pick, _suspect, _recognition_image


@dataclass(frozen=True)
class Focus:
    """Where the price row is inside the guide crop.

    top/bottom are row indices of the vertical band that holds the WHOLE
    price. The horizontal range is never trimmed.
    """
    top: int
    bottom: int
    located: bool            # a text row was reliably found
    clipped_left: bool = False   # a glyph of the row touches the left edge
    clipped_right: bool = False  # ... or the right edge
    ink_top: int = 0         # rows actually covered by the glyphs
    ink_bottom: int = 0


def _ink_mask(bgr: np.ndarray) -> np.ndarray:
    """Polarity-agnostic "local contrast" mask: dark-on-light, light-on-dark,
    red-on-white and white-on-red digits all become foreground.

    The reference blur is wider than a bold stroke, so large/bold digits do
    not turn hollow (a small adaptive-threshold window did that).
    """
    h, w = bgr.shape[:2]
    small = cv2.GaussianBlur(bgr, (3, 3), 0)          # JPEG / sensor noise
    sigma = max(6.0, min(h, w) / 4.0)
    # The reference ("local mean") is a very smooth image, so compute it on a
    # reduced copy: ~10x faster than a huge full-resolution Gaussian.
    f = 4 if min(h, w) >= 96 else (2 if min(h, w) >= 48 else 1)
    if f > 1:
        tiny = cv2.resize(small, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
        tiny = cv2.GaussianBlur(tiny, (0, 0), sigma / f)
        local = cv2.resize(tiny, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        local = cv2.GaussianBlur(small, (0, 0), sigma)
    diff = np.abs(small.astype(np.int16) - local.astype(np.int16)).max(axis=2)
    top = float(np.percentile(diff, 99))
    if top < 24:                                        # no text-like contrast
        return np.zeros((h, w), np.uint8)
    return np.uint8(diff >= max(14.0, top * .35)) * 255


def _edge_clip(bgr: np.ndarray, lo: int, hi: int) -> tuple[bool, bool]:
    """Does the price row run into the left/right edge of the crop?

    The background colour of the text row is its median colour (text covers
    well under half of the band). If the thin strip along a crop edge is not
    background-coloured, a glyph is touching or crossing that edge, so the
    first/last digits are outside the guide.
    """
    h, w = bgr.shape[:2]
    band = bgr[max(0, lo):min(h, hi)]
    if band.size == 0:
        return False, False
    bg = np.median(band.reshape(-1, 3), axis=0)
    diff = np.abs(band.astype(np.int16) - bg).max(axis=2)
    contrast = float(np.percentile(diff, 99))
    if contrast < 40:
        return False, False
    fg = diff >= contrast * .5
    m = max(3, int(round(w * .012)))
    return bool(fg[:, :m].mean() >= .10), bool(fg[:, -m:].mean() >= .10)


def _analyze_line(bgr: np.ndarray) -> Focus:
    """Find the vertical band of the main text row WITHOUT cutting any glyph.

    Rules that protect against the "only the bottom of the digits reached the
    OCR" failure (e.g. 72,400 -> 4,900):
      * glyphs are never rejected for being tall: a price may fill the band;
      * rows are bridged across tiny gaps inside a glyph row;
      * every glyph that is mostly inside the band is included completely;
      * if the band cannot be trusted, the WHOLE crop is used.
    """
    h, w = bgr.shape[:2]
    whole = Focus(0, h, False)
    if h < 40 or w < 100:
        return whole
    ink = _ink_mask(bgr)
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    min_area = max(6, int(h * w * .00012))
    glyphs = []
    cleaned = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        x, y, cw, ch, area = (int(v) for v in stats[i])
        # lower bounds drop speckle; width bound drops long rules/frames.
        # NO upper height bound: big digits are exactly what we must keep.
        if area >= min_area and cw >= 2 and ch >= max(4, h * .05) and cw <= w * .50:
            glyphs.append((x, y, cw, ch))
            cleaned[y:y + ch, x:x + cw][_labels[y:y + ch, x:x + cw] == i] = 255
    if len(glyphs) < 2:
        return whole

    center = cleaned[:, int(w * .035):int(w * .965)]
    rows = np.count_nonzero(center, axis=1).astype(np.float32)
    rows = cv2.GaussianBlur(rows.reshape(-1, 1), (1, 0), max(1.5, h * .02)).ravel()
    yy = np.arange(h)
    center_weight = np.exp(-.5 * ((yy - h * .5) / max(1, h * .30)) ** 2)
    peak = int(np.argmax(rows * (.65 + .35 * center_weight)))
    strength = float(rows[peak])
    if strength < max(3, w * .012):
        return whole

    cutoff = max(2.0, strength * .10)
    gap = max(3, int(h * .05))
    lo = hi = peak
    # walk outwards, bridging small dips INSIDE a glyph row
    miss = 0
    for r in range(peak - 1, -1, -1):
        if rows[r] >= cutoff:
            lo, miss = r, 0
        else:
            miss += 1
            if miss > gap:
                break
    miss = 0
    for r in range(peak + 1, h):
        if rows[r] >= cutoff:
            hi, miss = r, 0
        else:
            miss += 1
            if miss > gap:
                break
    hi += 1
    # include every glyph that is mostly inside the band, completely
    for _ in range(3):
        grew = False
        for x, y, cw, ch in glyphs:
            inside = min(hi, y + ch) - max(lo, y)
            if inside >= .30 * ch and (y < lo or y + ch > hi):
                lo, hi, grew = min(lo, y), max(hi, y + ch), True
        if not grew:
            break

    band = hi - lo
    if band < max(12, h * .08):
        return whole
    pad = max(6, int(band * .12))
    top, bottom = max(0, lo - pad), min(h, hi + pad)
    if bottom - top >= h * .88:           # nothing worth trimming
        top, bottom = 0, h

    clipped_left, clipped_right = _edge_clip(bgr, lo, hi)
    return Focus(top, bottom, True, clipped_left, clipped_right, lo, hi)


def _focus_price_line(bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    """Crop whitespace above/below the main price row; keep the ENTIRE width."""
    focus = _analyze_line(bgr)
    if not focus.located:
        return bgr, False
    return bgr[focus.top:focus.bottom, :].copy(), True


def _pad_line(line: np.ndarray, add_top: int, add_bottom: int) -> np.ndarray:
    """Add a background-coloured margin above/below ONLY where it is missing,
    so glyphs never touch the image edge (they are then read less reliably)."""
    if add_top <= 0 and add_bottom <= 0:
        return line
    bg = tuple(int(v) for v in np.median(line.reshape(-1, 3), axis=0))
    return cv2.copyMakeBorder(line, max(0, add_top), max(0, add_bottom), 0, 0,
                              cv2.BORDER_CONSTANT, value=bg)


def _prepare(bgr: np.ndarray, attempt: int) -> tuple[np.ndarray, Focus]:
    focus = _analyze_line(bgr)
    line = bgr[focus.top:focus.bottom, :].copy() if focus.located else bgr
    if focus.located:
        margin = max(4, int(round((focus.ink_bottom - focus.ink_top) * .12)))
        line = _pad_line(line,
                         margin - (focus.ink_top - focus.top),
                         margin - (focus.bottom - focus.ink_bottom))
    # Change contrast between independent frames rather than two OCR passes.
    kind = 'contrast' if attempt % 3 == 1 else 'original'
    img = _recognition_image(line, kind, fast=True)
    # EasyOCR normalizes input height; preserving line proportions prevents
    # 5-digit prices being downsampled into unreadably skinny characters.
    h, w = img.shape[:2]
    if h < 60 and w:
        factor = min(2.25, 60 / max(h, 1), 1200 / w)
        if factor > 1.02:
            img = cv2.resize(img, None, fx=factor, fy=factor,
                             interpolation=cv2.INTER_CUBIC)
    return img, focus


def _recognition_input(bgr: np.ndarray, attempt: int) -> tuple[np.ndarray, bool]:
    img, focus = _prepare(bgr, attempt)
    return img, focus.located


def _recognize(img_rgb: np.ndarray, reader):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return reader.recognize(gray, decoder='greedy', detail=1,
                            paragraph=False, batch_size=1)


def _digits(raw: str) -> int:
    return len(re.sub(r'\D', '', raw))


def scan_tight_fast(bgr: np.ndarray, currency: str, reader, attempt: int = 0) -> Reading:
    if bgr is None or not bgr.size:
        return Reading(None, note='Kamera görüntüsü gelmedi')

    image, focus = _prepare(bgr, attempt)
    focused = focus.located
    # Diagnostics only (shown in the app's debug panel): exactly what the OCR saw.
    debug = {'crop': cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 'ocr_input': image,
             'tokens': [], 'attempt': attempt, 'located': focused}

    def done(reading: Reading) -> Reading:
        return dataclasses.replace(reading, debug=debug)

    # A glyph of the price row touches the crop edge: the first/last digits
    # are outside the guide, and the OCR would happily report just the rest
    # of the number (72,400 -> 400). Never publish that; ask for re-framing.
    if focus.clipped_left or focus.clipped_right:
        where = ('başı' if focus.clipped_left and not focus.clipped_right else
                 'sonu' if focus.clipped_right and not focus.clipped_left else 'iki ucu')
        return done(Reading(None, note=f'Fiyatın {where} çerçevenin dışında kalıyor; '
                                       'biraz uzaklaş veya ↔ Alanı genişlet'))

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
    debug['tokens'] = [(str(r[1]), float(r[2])) for r in results if len(r) >= 3]

    candidates = _candidates(results, currency, image.shape, image)
    picked, rivals = _pick(candidates)
    if picked is None:
        return done(Reading(None, note='Birden çok sayı var; fiyatı tek başına ortala' if rivals
                       else 'Tüm fiyatı kutuya sığdırıp biraz yaklaştır',
                       alternatives=rivals))

    if picked.crossed or _suspect(picked, currency, results):
        return done(Reading(None, note='Şüpheli veya eski fiyat; tekrar okunuyor'))

    # Single tight crop: require a FULL grouped amount, not a floating tail
    # of digits falsely interpreted as an independent number. EasyOCR may
    # break a price into multiple tokens: if there is evidence of a missing
    # left fragment, don't publish the small tail as confirmed price.
    digits = _digits(picked.raw)
    if currency in ('KRW', 'JPY') and digits < 3:
        return done(Reading(None, note='Eksik rakam; tüm fiyatı ortala'))
    threshold = .58 if picked.currency_near else (.68 if focused else .76)
    if picked.confidence < threshold:
        return done(Reading(None, note='Rakam yeterince net değil; biraz yaklaş'))

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
                return done(Reading(None, note='Fiyatın ilk rakamları ayrı okunmuş; yeniden deneniyor'))

    return done(Reading(picked.value, picked.raw, picked.confidence,
                   note='Tüm fiyat satırı okundu'))
