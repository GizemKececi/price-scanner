"""Regression tests for full-length tight price scanning."""
import unittest
from decimal import Decimal

import cv2
import numpy as np

from price_tight import _focus_price_line, _recognition_input, scan_tight_fast


def box(x, y, w, h):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def sample_price(text='72,000 WON'):
    img = np.full((225, 760, 3), 248, dtype=np.uint8)
    cv2.putText(img, text, (112, 140), cv2.FONT_HERSHEY_SIMPLEX,
                2, (23, 23, 23), 6, cv2.LINE_AA)
    return img


class OCRMock:
    def __init__(self, text, conf=.98):
        self.text = text
        self.conf = conf
        self.sizes = []

    def recognize(self, image, **kwargs):
        self.sizes.append(image.shape)
        return [(box(60, 10, 440, max(20, image.shape[0]-20)), self.text, self.conf)]


class V11Tests(unittest.TestCase):
    def test_located_price_is_short_row_but_keeps_entire_width(self):
        img = sample_price()
        line, ok = _focus_price_line(img)
        self.assertTrue(ok)
        self.assertLess(line.shape[0], img.shape[0] * .60)
        self.assertEqual(line.shape[1], img.shape[1])

    def test_won_suffix_does_not_change_price(self):
        img = sample_price()
        for suffix in ['72,000원', '72,000 won', '72,000 WONDER',
                       '72,000 mot', '₩72,000', '72.000 KRW', '72,000~']:
            with self.subTest(suffix=suffix):
                model = OCRMock(suffix)
                result = scan_tight_fast(img, 'KRW', model)
                self.assertEqual(result.amount, Decimal(72000))
                self.assertEqual(len(model.sizes), 1)

    def test_small_standalone_price_does_not_gain_digits(self):
        model = OCRMock('4,000 mut')
        self.assertEqual(scan_tight_fast(sample_price('4,000'), 'KRW', model).amount,
                         Decimal(4000))

    def test_focus_prevents_unneeded_whitespace_shrink(self):
        image, focused = _recognition_input(sample_price(), 0)
        self.assertTrue(focused)
        self.assertLess(image.shape[0], 135)
        self.assertGreater(image.shape[1], 650)

    def test_price_and_percent_are_not_joined(self):
        m = OCRMock('28% 72,000 WON')
        self.assertEqual(scan_tight_fast(sample_price(), 'KRW', m).amount,
                         Decimal(72000))

    def test_struck_through_not_used(self):
        m = OCRMock('~~72,000원~~')
        self.assertIsNone(scan_tight_fast(sample_price(), 'KRW', m).amount)

    def test_low_confidence_is_not_promoted(self):
        m = OCRMock('4,000', .33)
        self.assertIsNone(scan_tight_fast(sample_price(), 'KRW', m).amount)


if __name__ == '__main__':
    unittest.main()
