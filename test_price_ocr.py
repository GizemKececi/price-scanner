"""Offline OCR logic tests using deterministic fake recognizer (no model download)."""
import unittest
from decimal import Decimal

import numpy as np
import cv2

from price_ocr import _candidates, _has_strikethrough, parse_amount, scan_crop, scan_crop_fast, _recognition_image


def box(x, y, w, h):
    return [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]


class Reader:
    def __init__(self, *passes):
        self.passes = list(passes)
        self.calls = 0

    def readtext(self, image, detail=1, paragraph=False, **kwargs):
        i = min(self.calls, len(self.passes)-1)
        self.calls += 1
        return self.passes[i]


class PriceOCRTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((400,1100,3),dtype=np.uint8)

    def test_live_one_ocr_call_per_frame(self):
        reader = Reader([(box(380,148,320,78), '9,950원', .94)])
        result = scan_crop_fast(self.image, 'KRW', reader)
        self.assertEqual(result.amount, Decimal(9950))
        self.assertEqual(reader.calls,1)

    def test_live_discount_and_review_not_concatenated(self):
        image = np.full((400,1100,3), 220, dtype=np.uint8)
        image[150:210,110:260] = (30,40,230)
        image[155:218,360:630] = (28,28,28)
        reader = Reader([(box(110,145,170,72), '50%', .98),
                         (box(355,145,350,86), '9,950원~', .92),
                         (box(390,327,245,36), '(99,999+)', .95)])
        result = scan_crop_fast(image, 'KRW', reader)
        self.assertEqual(result.amount, Decimal(9950))
        self.assertEqual(reader.calls,1)

    def test_live_no_giant_combined_price(self):
        reader = Reader([(box(180,145,520,95), '509,995,021',.99)])
        result = scan_crop_fast(self.image, 'KRW', reader)
        self.assertIsNone(result.amount)
        self.assertEqual(reader.calls,1)

    def test_live_struck_old_price_only(self):
        reader = Reader([(box(200,150,320,80), '~~20,000원~~', .92)])
        result = scan_crop_fast(self.image,'KRW',reader)
        self.assertIsNone(result.amount)
        self.assertEqual(reader.calls,1)

    def test_live_bottom_review_digits_dont_become_price(self):
        reader = Reader([(box(350,320,280,55), '99,999', .99)])
        result = scan_crop_fast(self.image, 'KRW', reader)
        self.assertIsNone(result.amount)
        self.assertEqual(reader.calls,1)

    def test_live_suspicious_won_not_confirmed(self):
        reader = Reader([(box(310,150,420,90), '15,000', .94)])
        result = scan_crop_fast(self.image,'KRW',reader)
        self.assertIsNone(result.amount)

    def test_won_symbols_not_amount(self):
        self.assertEqual(parse_amount("5,000", "KRW"), Decimal("5000"))
        self.assertEqual(parse_amount("9.950", "KRW"), Decimal("9950"))
        self.assertIsNone(parse_amount("₩5,000", "KRW"))
        self.assertIsNone(parse_amount("$", "USD"))
        reader = Reader([(box(380,150,280,80), "₩5,000", .89)])
        reading = scan_crop(self.image, "KRW", reader)
        self.assertEqual(reading.amount, Decimal("5000"))
        self.assertEqual(reading.raw, "5,000")

    def test_price_9950_among_many_numbers(self):
        reader = Reader([
            (box(80,10,160,28), "12345678", .99),
            (box(90,280,150,28), "20% OFF", .95),
            (box(330,140,460,104), "₩9,950", .89),
            (box(840,320,120,25), "45900", .99),
        ])
        reading = scan_crop(self.image, "KRW", reader)
        self.assertEqual(reading.amount, Decimal("9950"))
        self.assertLessEqual(reader.calls, 2)

    def test_decimals(self):
        self.assertEqual(parse_amount("99,50", "EUR"), Decimal("99.50"))
        self.assertEqual(parse_amount("1,299.90", "EUR"), Decimal("1299.90"))
        self.assertEqual(parse_amount("1.299,90", "EUR"), Decimal("1299.90"))
        self.assertEqual(parse_amount("9 950", "KRW"), Decimal("9950"))
        self.assertEqual(parse_amount("9950", "KRW"), Decimal("9950"))

    def test_disagreement_does_not_guess(self):
        reader = Reader(
            [(box(340,130,400,100), "15,000", .95)],
            [(box(350,130,390,100), "5,000", .92)],
        )
        result = scan_crop(self.image, "KRW", reader)
        self.assertIsNone(result.amount)
        self.assertEqual(reader.calls, 2)

    def test_real_15000_still_possible_after_crosscheck(self):
        reader = Reader(
            [(box(340,130,400,100), "15,000", .91)],
            [(box(340,130,400,100), "15,000", .89)],
        )
        result = scan_crop(self.image, "KRW", reader)
        self.assertEqual(result.amount, Decimal("15000"))

    def test_split_number_9_and_950(self):
        reader = Reader([
            (box(380,148,72,82), "9,", .93),
            (box(458,148,290,82), "950", .94),
        ])
        reading = scan_crop(self.image, "KRW", reader)
        self.assertEqual(reading.amount, Decimal("9950"))

    def test_discount_and_current_price_in_the_same_ocr_string(self):
        # Regression for user's screenshot: if EasyOCR merges 50% with 9,950원,
        # the percent belongs to 50, NOT to the price to its right.
        reader = Reader([(box(260,138,560,100), "50% 9,950원~", .97)])
        reading = scan_crop(self.image, "KRW", reader)
        self.assertEqual(reading.amount, Decimal("9950"))
        self.assertEqual(reading.raw, "9,950")

    def test_discount_and_reviews_with_old_price(self):
        # Discount %50, faded original 20,000 won, current 9,950 won,
        # star 4.9 and review count 99,999+ may appear in the same image.
        rows = [
            (box(280,28,250,33), "20,000원", .80),
            (box(120,140,170,72), "50%", .97),
            (box(305,132,395,98), "9,950원~", .94),
            (box(120,332,85,29), "★ 4.9", .98),
            (box(280,330,220,32), "(99,999+)", .95),
        ]
        reader = Reader(rows)
        reading = scan_crop(self.image, "KRW", reader)
        self.assertEqual(reading.amount, Decimal("9950"))
        self.assertEqual(reading.raw, "9,950")

    def test_review_count_is_not_price(self):
        reader = Reader([(box(420,140,330,90), "9,950원", .90),
                         (box(350,290,300,55), "(99,999+)", .99)])
        reading = scan_crop(self.image, "KRW", reader)
        self.assertEqual(reading.amount, Decimal("9950"))

    def test_red_discount_misjoined_to_price_corrected(self):
        image = np.full((400,1100,3), 210, dtype=np.uint8)
        image[130:225,100:280] = (0,0,255)  # red discount letters
        for x in range(314,700,42):
            image[146:220,x:x+12] = (25,25,25) # dark numeral strokes
        reader = Reader(
            [(box(200,140,490,96), "509,950원", .95)],
            [(box(310,140,375,96), "9,950원", .96)],
        )
        reading = scan_crop(image,"KRW",reader)
        self.assertEqual(reading.amount, Decimal("9950"))
        self.assertEqual(reader.calls,2)

    def test_red_discount_with_unresolved_ocr_must_not_confirm(self):
        image = np.full((400,1100,3), 210, dtype=np.uint8)
        image[130:225,100:280] = (0,0,255)
        for x in range(314,700,42):
            image[146:220,x:x+12] = (25,25,25)
        reader = Reader(
            [(box(200,140,490,96), "509,950원", .95)],
            [(box(300,140,400,96), "9,850원", .67)],
        )
        reading = scan_crop(image,"KRW",reader)
        self.assertIsNone(reading.amount)

    def test_clearly_struck_through_old_price_is_rejected(self):
        # We verify the visual line detector, not just a text string containing ~~.
        image = np.full((140,440,3), 248, dtype=np.uint8)
        cv2.putText(image, "20000", (60,93),cv2.FONT_HERSHEY_SIMPLEX,1.75,(25,25,25),4)
        bounds = (55,40,280,112)
        self.assertFalse(_has_strikethrough(image,bounds))
        cv2.line(image,(55,75),(278,75),(25,25,25),3)
        self.assertTrue(_has_strikethrough(image,bounds))

    def test_only_old_struck_price_is_not_accepted(self):
        reader = Reader([(box(400,130,300,95), "~~20,000원~~",.94)])
        reading = scan_crop(self.image,"KRW",reader)
        self.assertIsNone(reading.amount)

    def test_discount_broken_across_boxes(self):
        reader = Reader([(box(180,150,125,70), "50", .96),
                         (box(308,160,40,55), "%", .96),
                         (box(370,148,350,92), "9,950원", .92)])
        reading = scan_crop(self.image, "KRW", reader)
        self.assertEqual(reading.amount, Decimal("9950"))

    def test_similarly_prominent_numbers_require_recheck(self):
        reader = Reader(
            [(box(260,145,280,82), "9,950", .92),
             (box(560,145,280,82), "8,950", .92)],
            [(box(350,145,440,82), "9,950", .95)],
        )
        reading = scan_crop(self.image, "KRW", reader)
        self.assertEqual(reading.amount, Decimal("9950"))


if __name__ == "__main__":
    unittest.main()
