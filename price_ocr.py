"""Price Scanner OCR v9: isolate a CURRENT price, not the longest number.

Symbols are contextual metadata only. Calculations use numeric text exclusively.
A discount %, struck-through old price, star rating, review count, or SKU is
never concatenated into the price. Ambiguous evidence causes a rescan.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

import cv2
import numpy as np

ZERO_DECIMAL = {"KRW", "JPY"}
NUMBER_GROUP = re.compile(
    r"(?<![0-9])(?:[0-9]{1,3}(?:[,. \u00a0'’][0-9]{3})+(?:[.,][0-9]{1,2})?|[0-9]+(?:[.,][0-9]{1,2})?)(?![0-9])"
)
CURRENCY_CONTEXT = re.compile(r"[₩￦원¥￥円$€£₺₹฿₫]", re.I)
PERCENT_SUFFIX = re.compile(r"^\s*(?:[%％]|퍼센트|percent\b|pct\b|\bOFF\b)", re.I)
REVIEW_COUNT_SUFFIX = re.compile(r"^\s*\+|^\s*\)\s*\+?|^\s*명\b", re.I)
STRIKE_TEXT = re.compile(r"(?:~~|̶|̸|̷)")
OLD_PRICE_WORDS = re.compile(r"\b(?:was|before|original|regular|rrp|list\s*price|old\s*price)\b|정가|원가|정상가|기존가", re.I)
REVIEW_WORDS = re.compile(r"review|rating|reviews|후기|리뷰|평점|구매평|구매후기", re.I)
SKU_WORDS = re.compile(r"\b(?:sku|ean|isbn|barcode|model|ref|item\s*no)\b|품번|상품번호", re.I)
STARS = re.compile(r"[★☆⭐✩✭✰]")
# Safety ceiling only for automatic extraction: a price above this needs manual
# confirmation. It prevents absurd OCR concatenation (509,995,021 KRW).
MAX_AUTO_PRICE = {"KRW": 50_000_000, "JPY": 50_000_000, "USD": 1_000_000,
                  "EUR": 1_000_000, "GBP": 1_000_000, "CNY": 5_000_000,
                  "THB": 10_000_000}


@dataclass(frozen=True)
class Reading:
    amount: Decimal | None
    raw: str = ""
    confidence: float = 0.0
    note: str = ""
    alternatives: tuple[Decimal, ...] = ()


@dataclass(frozen=True)
class Candidate:
    value: Decimal
    raw: str
    score: float
    confidence: float
    bounds: tuple[float, float, float, float]
    currency_near: bool = False
    crossed: bool = False
    contrast: float = 0.0
    context: str = ""


def parse_amount(raw: str, currency: str) -> Decimal | None:
    """A parsed price contains digits, grouping and optional decimal ONLY."""
    s = unicodedata.normalize("NFKC", raw).strip().replace("\u00a0", " ")
    if not NUMBER_GROUP.fullmatch(s):
        return None
    digits = re.sub(r"[^0-9]", "", s)
    if not digits or len(digits) > 10:
        return None
    if currency in ZERO_DECIMAL:
        return Decimal(digits)
    clean = s.replace(" ", "").replace("'", "").replace("’", "")
    last_sep = max(clean.rfind("."), clean.rfind(","))
    places = len(clean) - last_sep - 1 if last_sep >= 0 else 0
    if 1 <= places <= 2:
        try:
            return Decimal(digits[:-places] + "." + digits[-places:])
        except InvalidOperation:
            return None
    return Decimal(digits)


def _recognition_image(bgr: np.ndarray, kind: str, *, fast: bool = False) -> np.ndarray:
    height, width = bgr.shape[:2]
    # Previous versions upscaled almost every frame to 1180px, substantially
    # slowing CPU-only EasyOCR. Live mode stays close to its input resolution.
    if fast:
        scale = min(1.0, 800 / max(width, 1))
    else:
        scale = min(1.20, 960 / max(width, 1))
    if kind == "original":
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif kind == "redless":
        # Discount badges are often red beside a BLACK current price. Suppress
        # just red ink, retaining black numbers. Do NOT use on an all-red label.
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        blue, green, red = cv2.split(bgr)
        red_mask = ((hsv[:, :, 1] > 55) & (red.astype(np.int16) > green.astype(np.int16) + 20) &
                    (red.astype(np.int16) > blue.astype(np.int16) + 18) &
                    (red > 65) & ((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 170)))
        img = bgr.copy()
        # Red price badges have a pale antialiased outline. Remove the outline
        # together with red ink so '50%' does not survive as white digits.
        red_mask = cv2.dilate(red_mask.astype(np.uint8),
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))) > 0
        img[red_mask] = (245, 245, 245)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    return img


def _has_red_black_mix(bgr: np.ndarray) -> bool:
    if bgr is None or bgr.size == 0:
        return False
    small = cv2.resize(bgr, (min(300,bgr.shape[1]), min(160,bgr.shape[0])))
    b,g,r = cv2.split(small)
    is_red = (r.astype(np.int16) > g.astype(np.int16) + 24) & (r > 90) & (g < 190)
    neutral_dark = (np.maximum(np.maximum(b,g),r) < 125) & (np.abs(r.astype(np.int16)-g.astype(np.int16)) < 45)
    return int(is_red.sum()) > 30 and int(neutral_dark.sum()) > 30


def _bounds(box):
    points = np.asarray(box, dtype=np.float32)
    mn, mx = points.min(axis=0), points.max(axis=0)
    return float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])


def _contrast(image_rgb: np.ndarray, bounds) -> float:
    h, w = image_rgb.shape[:2]
    x0,y0,x1,y1 = bounds
    roi = image_rgb[max(0,int(y0)):min(h,int(y1)), max(0,int(x0)):min(w,int(x1))]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    return float(np.percentile(gray, 83) - np.percentile(gray, 14))


def _has_strikethrough(image_rgb: np.ndarray, bounds) -> bool:
    """Detect actual continuous ink crossing the middle of a printed number."""
    h, w = image_rgb.shape[:2]
    x0,y0,x1,y1 = bounds
    x0,x1 = max(0,int(x0)), min(w,int(x1))
    y0,y1 = max(0,int(y0)), min(h,int(y1))
    bw,bh = x1-x0,y1-y0
    if bw < 40 or bh < 16:
        return False
    roi = cv2.cvtColor(image_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    if np.percentile(roi, 83) - np.percentile(roi, 14) < 16:
        return False  # Blank/uniform region cannot be struck-through text.
    threshold = max(70, min(235, float(np.percentile(roi, 80))-16))
    ink = np.uint8(roi < threshold)*255
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                        cv2.getStructuringElement(cv2.MORPH_RECT, (max(25,int(bw*.57)),1)))
    lo, hi = int(bh*.28), max(int(bh*.72), int(bh*.28)+1)
    for line in horizontal[lo:hi]:
        if cv2.countNonZero(line) >= bw*.58:
            return True
    return False


def _same_line(a, b) -> bool:
    a0, a1 = a[1], a[3]
    b0, b1 = b[1], b[3]
    height = min(a1-a0, b1-b0)
    return height > 0 and max(0, min(a1,b1)-max(a0,b0)) >= height*.38


def _normalize_results(results):
    rows = []
    for result in results:
        if len(result) < 3:
            continue
        box, text, confidence = result[:3]
        text = unicodedata.normalize("NFKC", str(text)).strip()
        if not text:
            continue
        x0,y0,x1,y1 = _bounds(box)
        if x1 <= x0 or y1 <= y0:
            continue
        rows.append((text, max(0.0, min(1.0,float(confidence))), (x0,y0,x1,y1)))
    return rows


def _numeric_segments(rows):
    """Preserve separators/geometry and forbid joining discount with price."""
    segments = []
    for text, confidence, bounds in rows:
        x0,y0,x1,y1 = bounds
        for match in NUMBER_GROUP.finditer(text):
            raw = match.group()
            # An ending comma may be a broken OCR box: '9,' + '950'.
            end = match.end()
            if end < len(text) and text[end] in ',.' and (end+1 == len(text) or not text[end+1].isdigit()):
                raw += text[end]
                end += 1
            left = text[:match.start()]
            right = text[end:]
            seg_bounds = (x0+(x1-x0)*match.start()/len(text), y0,
                          x0+(x1-x0)*end/len(text), y1)
            segments.append((raw, confidence, seg_bounds, left, right, text))
    # Join only a clearly split thousand-group, requiring explicit ',' or '.'
    # in the left box. Joining bare 50 + 995 would fabricate 50,995.
    first = list(segments)
    for i,(raw,conf,bb,left,right,whole) in enumerate(first):
        if not re.fullmatch(r"[0-9]{1,3}[,.]",raw):
            continue
        if PERCENT_SUFFIX.match(right) or PERCENT_SUFFIX.match(whole):
            continue
        for j,(raw2,conf2,bb2,left2,right2,whole2) in enumerate(first):
            if i == j or not re.fullmatch(r"[0-9]{3}", raw2):
                continue
            if not _same_line(bb,bb2):
                continue
            gap = bb2[0]-bb[2]
            max_height = max(bb[3]-bb[1],bb2[3]-bb2[1])
            if 0 <= gap <= .45*max_height:
                segments.append((raw+raw2,min(conf,conf2)*.96,
                                 (bb[0],min(bb[1],bb2[1]),bb2[2],max(bb[3],bb2[3])),
                                 left,right2,whole+whole2))
    return segments


def _context_flags(raw, bounds, left, right, whole, rows):
    """Symbols tell us the ROLE of a number but never become numeric digits."""
    before = left[-24:]
    after = right[:24]
    near_currency = bool(CURRENCY_CONTEXT.search(before[-3:] + after[:3]))
    # A percent sign AFTER this particular number marks a discount ("50%").
    # Critically, a percent sign BEFORE the following price in the same OCR
    # string ("50% 9,950원") belongs to 50, NOT 9,950. Never inherit it.
    percent = bool(PERCENT_SUFFIX.match(after))
    # Some labels use "%50"; only classify as percent if the prefix is
    # adjacent AND the value is percentage-like, not an actual price.
    if re.search(r"[%％]$", before) and raw.isdigit() and int(raw) <= 100:
        percent = True
    review = bool(REVIEW_COUNT_SUFFIX.match(after) or
                  (re.search(r"\(\s*$",before) and re.match(r"\s*\+?\)?",after)))
    old = bool(STRIKE_TEXT.search(whole) or OLD_PRICE_WORDS.search(before[-15:]))
    if SKU_WORDS.search(before[-20:]):
        return True, percent, review, old, near_currency  # forbidden
    # Context can live in adjacent EasyOCR boxes.
    for other, _, other_bb in rows:
        if other == whole or not _same_line(bounds,other_bb):
            continue
        gap_left = bounds[0]-other_bb[2]
        gap_right = other_bb[0]-bounds[2]
        h = max(bounds[3]-bounds[1],other_bb[3]-other_bb[1])
        if 0 <= gap_right < 1.1*h:
            if other.startswith(('%','％')) or re.match(r"^(?:off|percent|할인)", other,re.I):
                percent = True
            if other.startswith(('+',')+','+)')):
                review = True
            if CURRENCY_CONTEXT.match(other):
                near_currency = True
        if 0 <= gap_left < 1.2*h:
            if other.strip() in ("%", "％") and raw.isdigit() and int(raw) <= 100:
                percent = True
            if CURRENCY_CONTEXT.search(other[-3:]):
                near_currency = True
            if OLD_PRICE_WORDS.search(other):
                old = True
        # Star-rating/review count on same line; near either side.
        close = (abs(gap_left) < 2*h or abs(gap_right) < 2*h)
        if close and (STARS.search(other) or REVIEW_WORDS.search(other)):
            review = True
    if REVIEW_WORDS.search(before[-15:]):
        review = True
    return False, percent, review, old, near_currency


def _candidates(results, currency: str, image_shape, image_rgb=None) -> list[Candidate]:
    height,width = image_shape[:2]
    rows = _normalize_results(results)
    segments = _numeric_segments(rows)
    raw_candidates = []
    for raw, conf, bounds, left, right, whole in segments:
        value = parse_amount(raw, currency)
        numdigits = len(re.sub(r"[^0-9]","",raw))
        if value is None or value <= 0 or conf < .20:
            continue
        if value > MAX_AUTO_PRICE.get(currency,1_000_000):
            continue
        if numdigits < (3 if currency in ZERO_DECIMAL else 2):
            continue
        forbidden,percent,review,old,currency_hint = _context_flags(raw,bounds,left,right,whole,rows)
        if forbidden or percent or review:
            continue
        x0,y0,x1,y1 = bounds
        cx,cy = (x0+x1)/2,(y0+y1)/2
        bh = y1-y0
        if bh < 6:
            continue
        # Preserve label's visual hierarchy, never prefer MORE digits or
        # mathematically bigger amounts. Rating rows near bottom are demoted.
        center_x = max(0.0,1.0-abs(cx-width*.5)/(width*.56))
        center_y = max(0.0,1.0-abs(cy-height*.49)/(height*.60))
        prominence = min(1.0,bh/max(height*.28,1))
        formatted = bool(re.fullmatch(r"\d{1,3}(?:[,. '’]\d{3})+",raw))
        contrast = _contrast(image_rgb,bounds) if image_rgb is not None else 0.0
        crossed = old or (_has_strikethrough(image_rgb,bounds) if image_rgb is not None else False)
        score = (.23*center_x+.25*center_y+.20*prominence+.14*conf+
                 .12*int(formatted)+.32*int(currency_hint)+
                 .06*min(contrast/110,1))
        if crossed:
            score -= .78
        if cy > height*.83 and not currency_hint:
            score -= .20
        if numdigits >= 8 and not currency_hint:
            score -= .18
        raw_candidates.append(Candidate(value,raw,score,conf,bounds,currency_hint,crossed,contrast,whole))

    # Old prices tend to be lighter and ABOVE the active current price;
    # compare appearance rather than assuming the larger number is old.
    adjusted = []
    for candidate in raw_candidates:
        score = candidate.score
        if not candidate.crossed:
            for other in raw_candidates:
                if candidate is other or candidate.bounds[1] >= other.bounds[1]:
                    continue
                dx = abs((candidate.bounds[0]+candidate.bounds[2])/2 -
                         (other.bounds[0]+other.bounds[2])/2)
                vertical_gap = other.bounds[1]-candidate.bounds[3]
                if dx < width*.40 and -height*.045 <= vertical_gap < height*.26:
                    if candidate.contrast > 0 and other.contrast > candidate.contrast*1.75:
                        score -= .34
                    if (candidate.bounds[3]-candidate.bounds[1]) < (other.bounds[3]-other.bounds[1])*.75:
                        score -= .16
        adjusted.append(Candidate(candidate.value,candidate.raw,score,candidate.confidence,
                                  candidate.bounds,candidate.currency_near,candidate.crossed,
                                  candidate.contrast,candidate.context))
    unique = {}
    for c in adjusted:
        if c.value not in unique or c.score > unique[c.value].score:
            unique[c.value] = c
    return sorted(unique.values(),key=lambda c:c.score,reverse=True)


def _pick(candidates: list[Candidate]) -> tuple[Candidate|None,tuple[Decimal,...]]:
    if not candidates:
        return None,()
    possible = [c for c in candidates if not c.crossed and c.score >= .25]
    if not possible:
        return None, tuple(c.value for c in candidates[:3])
    best = possible[0]
    # A box that is fully inside the chosen price is an OCR fragment, not a
    # competing shop price (e.g. broken "9," + "950" -> "9,950").
    def contained(fragment, whole):
        a,b=fragment.bounds,whole.bounds
        return (a[0]>=b[0]-2 and a[2]<=b[2]+2 and
                a[1]>=b[1]-2 and a[3]<=b[3]+2 and fragment.value!=whole.value)
    rivals = [c for c in possible[1:] if c.score >= best.score-.12 and not contained(c,best)]
    if rivals:
        return None,tuple(dict.fromkeys(c.value for c in [best]+rivals))[:3]
    return best,()


def _suspect(candidate: Candidate, currency: str, results) -> bool:
    # If the OCR dropped the percent sign and glued 50 with 9,950, another
    # independently preprocessed read is mandatory, even if confidence high.
    text = " ".join(str(x[1]) for x in results if len(x)>=2)
    raw_digits = re.sub(r"\D","",candidate.raw)
    if len(raw_digits) >= 7 and not candidate.currency_near:
        return True
    if currency in ZERO_DECIMAL and not candidate.currency_near and re.fullmatch(r"1\d{1,2}[,.]\d{3}",candidate.raw):
        return True  # ₩ is sometimes read as a leading 1.
    # An independent discount token ("28% 72,000 WON") must NOT poison the
    # separate 72,000 price. Only suspect concatenation when the OCR output
    # contains a percentage yet offers no independently parsed percent number.
    has_percent = bool(re.search(r"[%％]", text))
    percent_is_separate = bool(re.search(r"(?:^|\s)[0-9]{1,3}\s*[%％]\s+", text))
    return (has_percent and not percent_is_separate and not candidate.currency_near and
            re.match(r"^[0-9]{5,}$",raw_digits) is not None)


def scan_crop(bgr: np.ndarray, currency: str, reader) -> Reading:
    if bgr is None or not bgr.size:
        return Reading(None,note="Kameradan görüntü gelmedi")
    base = _recognition_image(bgr,"original")
    results = reader.readtext(base,detail=1,paragraph=False, width_ths=.25,
                              canvas_size=1024, mag_ratio=1.0)
    candidates = _candidates(results,currency,base.shape,base)
    primary, alternatives_a = _pick(candidates)
    red_black = _has_red_black_mix(bgr)
    # Red 50% labels can be joined to the black 9,950 by a noisy recognizer,
    # producing 509,950 (or worse). Re-read with red ink suppressed before
    # confirming long numbers in mixed red/black labels.
    primary_suspicious = bool(primary and (
        _suspect(primary,currency,results) or
        (red_black and len(re.sub(r"\D", "", primary.raw)) >= 6)
    ))
    if primary and primary.confidence>=.55 and not primary_suspicious:
        return Reading(primary.value,primary.raw,primary.confidence,"Fiyat ayrıştırıldı")

    # One extra attempt only if needed; keep camera responsive. Different
    # visual treatment removes red discount labels without deleting black price.
    kind = "redless" if red_black else "contrast"
    second = _recognition_image(bgr,kind)
    results2 = reader.readtext(second,detail=1,paragraph=False, width_ths=.25,
                               canvas_size=1024, mag_ratio=1.0)
    candidates2 = _candidates(results2,currency,second.shape,second)
    secondary, alternatives_b = _pick(candidates2)
    if primary and secondary and primary.value == secondary.value:
        return Reading(primary.value,primary.raw,min(primary.confidence,secondary.confidence),"İki taramada eşleşti")
    if secondary and secondary.confidence >= .55 and not _suspect(secondary,currency,results2):
        if primary and primary.value != secondary.value:
            if (red_black and primary_suspicious and secondary.currency_near and
                    secondary.confidence >= .80 and secondary.value < primary.value):
                # A very specific case: independent red-suppressed read isolates
                # the currency-marked dark price. Live mode still requires TWO
                # separate camera frames to agree before it publishes the price.
                return Reading(secondary.value, secondary.raw,secondary.confidence,
                               "İndirim etiketinden ayrıştırıldı")
            # Conflicting evidence is not a confirmed price.
            return Reading(None,note="İki farklı rakam algılandı; fiyatı ortala",alternatives=(primary.value,secondary.value))
        if alternatives_a and secondary.value not in alternatives_a and not secondary.currency_near:
            return Reading(None,note="Birden fazla sayı var; fiyatı ortala",alternatives=alternatives_a)
        return Reading(secondary.value,secondary.raw,secondary.confidence,"Filtrelenerek okundu")
    # Optional fallback: red ink filter may suppress a truly red price; don't
    # accept it as absent. A strong primary in that case is still usable.
    if primary and not primary_suspicious and primary.confidence >= .65 and not alternatives_b:
        return Reading(primary.value,primary.raw,primary.confidence,"İlk görüntüden okundu")
    choices = tuple(dict.fromkeys(alternatives_a+alternatives_b+
                                  ((primary.value,) if primary else ())+
                                  ((secondary.value,) if secondary else ())))[:3]
    if choices:
        return Reading(None,note="Eski fiyat, indirim veya yorum sayısı ayrıştırılamadı; biraz yaklaştır", alternatives=choices)
    return Reading(None,note="Güncel fiyat bulunamadı; fiyatı çerçevenin ortasına getir")


def scan_crop_fast(bgr: np.ndarray, currency: str, reader, attempt: int = 0) -> Reading:
    """ONE EasyOCR inference per live frame; never block to reread this frame.

    A black price beside red discount text is read after suppressing red ink.
    If uncertain, caller alternates preprocessing on a *new* camera frame.
    Currency symbols and other glyphs are context, not numeric digits.
    """
    if bgr is None or not bgr.size:
        return Reading(None, note="Kameradan görüntü gelmedi")
    mixed = _has_red_black_mix(bgr)
    if mixed:
        kind = "redless" if attempt % 3 != 2 else "original"
    else:
        kind = "contrast" if attempt % 3 == 2 else "original"
    image = _recognition_image(bgr,kind, fast=True)
    # Tight grouping prevents separate discount and price glyphs joining.
    results = reader.readtext(image, detail=1, paragraph=False, width_ths=.22,
                              canvas_size=960, mag_ratio=1.0)
    candidates = _candidates(results,currency,image.shape,image)
    selected, rivals = _pick(candidates)
    if selected is None:
        if rivals:
            return Reading(None, note="Birden çok rakam var; fiyatı ortaya getir", alternatives=rivals)
        return Reading(None, note="Fiyat seçilemedi; merkeze yaklaştır")

    # Never publish visually crossed out numbers, a suspiciously concatenated
    # discount and price, or a low-confidence guess. The next frame can try a
    # different visual treatment without doing a second slow OCR now.
    threshold = .50 if selected.currency_near else .63
    if selected.crossed or selected.confidence < threshold:
        return Reading(None, note="Okuma belirsiz; yeniden deneniyor")
    # User points the middle of the guide at the actual price. Lower rows
    # without a currency hint are commonly star ratings or review counts,
    # especially when OCR has dropped parentheses and '+' symbols.
    if not selected.currency_near and (selected.bounds[1] + selected.bounds[3])*0.5 > image.shape[0]*0.77:
        return Reading(None,note="Alt satır yorum sayısı olabilir; fiyatı ortala")
    if mixed and kind == 'original' and len(re.sub(r'\D','',selected.raw)) >= 6:
        return Reading(None, note="İndirim ile fiyat karışabilir; yeniden deneniyor")
    if _suspect(selected,currency,results):
        return Reading(None,note="Şüpheli rakam; yeniden kontrol ediliyor")
    # Guard against a missing decimal/grouping separator creating concatenated
    # six-plus digit numbers when the same frame contains several labels.
    digits = re.sub(r'\D','',selected.raw)
    if len(digits) >= 7 and not selected.currency_near:
        return Reading(None, note="Birleşik rakam olabilir; fiyatı merkeze getir")
    return Reading(selected.value, selected.raw, selected.confidence,
                   note="Fiyat bulundu; ikinci kare ile kontrol ediliyor")
