"""Focused one-line price mode: deterministic recognizer tests, no model downloads."""
import unittest
from decimal import Decimal
import numpy as np
from price_tight import scan_tight_fast


def box(x,y,w,h):
    return [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]

class FakeRecognize:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0
    def recognize(self,image,**kwargs):
        self.calls += 1
        assert image.ndim == 2
        return self.rows

class TightTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((135,720,3), 244, dtype=np.uint8)

    def test_current_price_only(self):
        r=FakeRecognize([(box(210,35,325,68),'28,600원~',.96)])
        found=scan_tight_fast(self.image,'KRW',r)
        self.assertEqual(found.amount,Decimal(28600))
        self.assertEqual(r.calls,1)

    def test_currency_prefix_5000(self):
        r=FakeRecognize([(box(180,30,380,72),'₩5,000',.98)])
        self.assertEqual(scan_tight_fast(self.image,'KRW',r).amount,Decimal(5000))

    def test_percent_not_counted(self):
        r=FakeRecognize([(box(130,35,470,75),'28% 28,600원',.94)])
        self.assertEqual(scan_tight_fast(self.image,'KRW',r).amount,Decimal(28600))

    def test_review_not_counted(self):
        r=FakeRecognize([(box(205,40,310,70),'★ 4.8 (6,314)',.94)])
        self.assertIsNone(scan_tight_fast(self.image,'KRW',r).amount)

    def test_old_price_not_counted(self):
        r=FakeRecognize([(box(215,40,295,70),'~~39,900원~~',.95)])
        self.assertIsNone(scan_tight_fast(self.image,'KRW',r).amount)

    def test_ambiguous_two_prices_not_confirmed(self):
        r=FakeRecognize([(box(80,30,240,75),'9,950',.95), (box(355,30,240,75),'28,600',.95)])
        self.assertIsNone(scan_tight_fast(self.image,'KRW',r).amount)

    def test_low_confidence_doesnt_confirm(self):
        r=FakeRecognize([(box(210,35,325,68),'6,002',.34)])
        self.assertIsNone(scan_tight_fast(self.image,'KRW',r).amount)

    def test_symbols_do_not_become_numbers(self):
        r=FakeRecognize([(box(210,35,325,68),'28,600원',.96)])
        result=scan_tight_fast(self.image,'KRW',r)
        self.assertEqual(result.raw,'28,600')

if __name__ == '__main__':
    unittest.main()
