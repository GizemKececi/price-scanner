"""Locale-aware user-entered price parsing; no OCR or camera dependencies."""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

WHOLE_UNIT_CURRENCIES = {"KRW", "JPY"}
MAX_AMOUNT = Decimal("100000000000")


def parse_price(raw: str, currency: str) -> Decimal | None:
    """Accept 72.470 / 72,470 as KRW 72470; EU/US decimal price formats.

    Never guess arbitrary letters or exponential notation into a currency amount.
    Empty inputs return None so no old value can appear for a new label.
    """
    value = raw.strip()
    if not value:
        return None
    value = re.sub(r"(?i)\b(?:krw|jpy|eur|gbp|thb|won|yen)\b", "", value)
    value = re.sub(r"[₩¥€£฿원￥円\s\u00a0\u202f]", "", value)
    if not value or not re.fullmatch(r"\d[\d.,]*", value):
        raise ValueError("Yalnızca sayı yaz. Örneğin: 72.470")
    if value.endswith(('.', ',')) or ',,' in value or '..' in value:
        raise ValueError("Fiyatın sonundaki ayıracı kaldır.")

    if currency in WHOLE_UNIT_CURRENCIES:
        # KRW and JPY typically have no fractional units in retail labels.
        # 72.470 and 72,470 are thousands, not 72.47.
        if not re.fullmatch(r"\d{1,3}([.,]\d{3})+|\d+", value):
            raise ValueError("Bu para birimi için tam sayı gir. Örn: 72.470")
        normalized = value.replace('.', '').replace(',', '')
    else:
        # First handle standard Western/EU formats with explicit decimal digits.
        if '.' in value and ',' in value:
            decimal_mark = '.' if value.rfind('.') > value.rfind(',') else ','
            fraction = value.rsplit(decimal_mark, 1)[-1]
            if len(fraction) not in (1, 2):
                raise ValueError("Ondalık bölüm 1 veya 2 basamak olmalı.")
            integer = value.rsplit(decimal_mark, 1)[0]
            other = ',' if decimal_mark == '.' else '.'
            if not re.fullmatch(rf"\d{{1,3}}(?:\{other}\d{{3}})+|\d+", integer):
                raise ValueError("Binlik ayraçlarını kontrol et.")
            normalized = integer.replace(other, '') + '.' + fraction
        elif '.' in value or ',' in value:
            mark = '.' if '.' in value else ','
            parts = value.split(mark)
            if len(parts) > 2:
                if len(parts[0]) not in (1, 2, 3) or any(len(x) != 3 for x in parts[1:]):
                    raise ValueError("Binlik ayraçlarını kontrol et.")
                normalized = ''.join(parts)
            elif len(parts[1]) == 3 and 1 <= len(parts[0]) <= 3:
                normalized = ''.join(parts)  # 1.299 / 1,299 -> 1299
            elif len(parts[1]) in (1, 2):
                normalized = parts[0] + '.' + parts[1]
            else:
                raise ValueError("Sayı biçimini kontrol et.")
        else:
            normalized = value

    try:
        amount = Decimal(normalized)
    except InvalidOperation as exc:
        raise ValueError("Geçerli bir fiyat gir.") from exc
    if amount <= 0:
        raise ValueError("Fiyat sıfırdan büyük olmalı.")
    if amount > MAX_AMOUNT:
        raise ValueError("Bu fiyat çok büyük. Rakamı kontrol et.")
    return amount


def parse_rate(raw: str) -> Decimal | None:
    """Manual FX input (supports 0,03697 and 0.03697)."""
    raw = raw.strip().replace(' ', '')
    if not raw:
        return None
    if not re.fullmatch(r"\d+(?:[.,]\d+)?", raw):
        raise ValueError("Kuru 0,03697 gibi yaz.")
    amount = Decimal(raw.replace(',', '.'))
    if amount <= 0 or amount > Decimal('1000000'):
        raise ValueError("Geçerli bir kur gir.")
    return amount
