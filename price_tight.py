"""Single-price, small-guide OCR for live camera.

Default mode is NOT a general OCR document parser. The user aligns ONE price
inside the short guide. A fast EasyOCR recognition-only path avoids the text
detector's CPU cost, with independent frames required to confirm results.
Symbols provide contextual evidence; they never contribute numerical digits.
"""
from __future__ import annotations

import re
from decimal import Decimal
from dataclasses import dataclass

import cv2
import numpy as np

from price_ocr import Reading, _candidates, _pick, _recognition_image, _suspect


_PRICE_LIKE = re.compile(r"\d")


def _read_single_line(image_rgb: np.ndarray, reader):
    """Use EasyOCR recognizer without slower text detection (one short line)."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    # EasyOCR's recognize can take grayscale and a whole-image line. A plain
    # single-line guide avoids CRAFT entirely in the common case.
    return reader.recognize(gray, decoder='greedy', detail=1,
                            paragraph=False, batch_size=1)


def scan_tight_fast(bgr: np.ndarray, currency: str, reader, attempt: int = 0) -> Reading:
    if bgr is None or not bgr.size:
        return Reading(None, note='Kamera görüntüsü gelmedi')
    # Do not force an allowlist of numbers: won (₩) can be misread as a leading 1.
    # Change treatment across DIFFERENT frames when needed, not 2 full OCR per frame.
    kind = 'contrast' if attempt % 3 == 1 else 'original'
    image = _recognition_image(bgr, kind, fast=True)
    try:
        # Third attempt only after unsuccessful narrow reads: detector as rescue.
        # Never run both methods on the same frame.
        if attempt % 3 == 2:
            results = reader.readtext(image, detail=1, paragraph=False,
                                      width_ths=.12, canvas_size=840, mag_ratio=1)
        elif hasattr(reader, 'recognize'):
            results = _read_single_line(image,reader)
        else:
            # Compatibility with mock/readtext-only and future recognizers.
            results = reader.readtext(image, detail=1, paragraph=False,
                                      width_ths=.12, canvas_size=840, mag_ratio=1)
    except (TypeError, ValueError, AttributeError):
        results = reader.readtext(image, detail=1, paragraph=False,
                                  width_ths=.12, canvas_size=840, mag_ratio=1)

    # A single line can still contain '28% 28,600원': parsed as distinct segments.
    candidates = _candidates(results, currency, image.shape, image)
    picked, rivals = _pick(candidates)
    if picked is None:
        return Reading(None, note='Birden fazla rakam var; yalnızca fiyatı kutuya al' if rivals
                       else 'Rakam net değil; fiyatı kutunun ortasına yaklaştır',
                       alternatives=rivals)

    digit_count = len(re.sub(r'\D', '', picked.raw))
    if picked.crossed or _suspect(picked,currency,results):
        return Reading(None, note='Şüpheli veya çizili rakam; tek güncel fiyatı hizala')
    # Never treat tiny rating or 1-2 digit promotion as a full KRW/JPY price.
    if currency in ('KRW','JPY') and digit_count < 3:
        return Reading(None, note='Bu rakam fiyat gibi görünmüyor')
    # When no currency survived recognition, demand stronger OCR confidence.
    # Two independent frames still required by app worker.
    threshold = .61 if picked.currency_near else .75
    if picked.confidence < threshold:
        return Reading(None, note='Fiyat yeterince net değil; biraz yaklaş')
    # The crop is intentionally tight. Wide labels are scanned only when the
    # user chooses "Alanı genişlet"; never search outside the green rectangle.
    return Reading(picked.value, picked.raw, picked.confidence,
                   note='Dar alandaki fiyat bulundu')
