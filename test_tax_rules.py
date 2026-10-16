import unittest
from decimal import Decimal
from tax_rules import COUNTRIES, estimate_refund


class RefundTests(unittest.TestCase):
    def test_korea_default_example(self):
        result = estimate_refund(Decimal("129000"), COUNTRIES["Güney Kore"])
        self.assertEqual(result.refund, Decimal("9030"))
        self.assertEqual(result.status, "estimated")

    def test_korea_minimum(self):
        result = estimate_refund(Decimal("14999"), COUNTRIES["Güney Kore"])
        self.assertEqual(result.refund, 0)
        self.assertEqual(result.status, "ineligible")

    def test_japan_inclusive_vat_fraction(self):
        result = estimate_refund(Decimal("11000"), COUNTRIES["Japonya"])
        self.assertEqual(result.refund, Decimal("1000"))
        self.assertEqual(result.status, "maximum")

    def test_japan_below_minimum(self):
        self.assertEqual(estimate_refund(Decimal("5499"), COUNTRIES["Japonya"]).refund, 0)

    def test_uk_unavailable(self):
        self.assertEqual(estimate_refund(Decimal("120"), COUNTRIES["Birleşik Krallık (İngiltere, İskoçya, Galler)"]).status, "unavailable")

    def test_override_is_capped_at_actual_vat(self):
        result = estimate_refund(Decimal("110000"), COUNTRIES["Güney Kore"], Decimal("99"))
        self.assertEqual(result.refund, Decimal("10000"))

    def test_french_standard_vat_ceiling(self):
        result = estimate_refund(Decimal("120"), COUNTRIES["Fransa"])
        self.assertEqual(result.refund, Decimal("20"))


if __name__ == "__main__":
    unittest.main()
