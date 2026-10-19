"""KRW/JPY have no decimals: truncated reads must not become other prices."""
import unittest
from decimal import Decimal

from price_ocr import parse_amount


class ZeroDecimalParsing(unittest.TestCase):
    def test_valid_forms(self):
        for raw in ("72,400", "72.400", "72 400", "72400", "5,000", "9950"):
            for cur in ("KRW", "JPY"):
                with self.subTest(raw=raw, cur=cur):
                    self.assertEqual(parse_amount(raw, cur), Decimal(raw.replace(",", "").replace(".", "").replace(" ", "")))

    def test_truncated_groupings_are_rejected(self):
        # Previously: 72,4 -> 724 ; 72,40 -> 7240 ; 4,90 -> 490
        for raw in ("72,4", "72,40", "4,90", "9,95", "1,2"):
            with self.subTest(raw=raw):
                self.assertIsNone(parse_amount(raw, "KRW"))

    def test_trailing_zero_decimals_are_ignored_not_multiplied(self):
        # Previously: 9,950.00 -> 995000 (100x too high)
        self.assertEqual(parse_amount("9,950.00", "KRW"), Decimal(9950))
        self.assertEqual(parse_amount("9,950.0", "JPY"), Decimal(9950))

    def test_non_zero_decimal_currencies_unchanged(self):
        self.assertEqual(parse_amount("99,50", "EUR"), Decimal("99.50"))
        self.assertEqual(parse_amount("1,299.90", "USD"), Decimal("1299.90"))
        self.assertEqual(parse_amount("1.299,90", "EUR"), Decimal("1299.90"))


if __name__ == "__main__":
    unittest.main()
