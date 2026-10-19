"""v12 regression tests: the OCR must receive the WHOLE price row.

Root cause of "72,400 read as 4,900": when the digits filled the green band
(taller than 72 % of the crop) the old focus step rejected every digit as
"too tall", kept only the comma/fragments and cropped a thin slice from the
bottom of the digits. These tests render real glyph shapes (no OCR model
needed) and check the geometry that reaches the recognizer.
"""
import unittest
from decimal import Decimal

import cv2
import numpy as np

from price_tight import (_analyze_line, _focus_price_line, _prepare,
                         scan_tight_fast)


def render(text, W=461, H=99, digit_h=60, bg=(248, 248, 248), fg=(25, 25, 25),
           x=None, noise=0.0, seed=1):
    """Return (image, (ink_top, ink_bottom)) with ground truth from a clean mask."""
    (_, th0), _ = cv2.getTextSize("0", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    scale = digit_h / th0
    thick = max(2, int(round(scale * 2.2)))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x0 = (W - tw) // 2 if x is None else (W - tw + 30 if x == 'cut_right' else x)
    y0 = (H + th) // 2
    mask = np.zeros((H, W), np.uint8)
    cv2.putText(mask, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thick, cv2.LINE_AA)
    img = np.empty((H, W, 3), np.uint8)
    img[:] = bg
    m = mask > 127
    img[m] = fg
    img = cv2.GaussianBlur(img, (0, 0), .7)
    if noise:
        rng = np.random.default_rng(seed)
        img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    rows = np.where(m.any(axis=1))[0]
    return img, (int(rows.min()), int(rows.max()) + 1)


class Recorder:
    """Recognizer double: records what it was given, returns a fixed text."""
    def __init__(self, text='72,400', conf=.97):
        self.text, self.conf, self.images = text, conf, []

    def recognize(self, image, **kwargs):
        self.images.append(image)
        h, w = image.shape[:2]
        return [([[0, 0], [w, 0], [w, h], [0, h]], self.text, self.conf)]


class WholeRowReachesOCR(unittest.TestCase):
    def test_digits_filling_the_band_are_not_sliced(self):
        # The exact failing situation: big digits in a shallow guide band.
        for digit_h in (62, 68, 72, 80, 88, 94):
            with self.subTest(digit_h=digit_h):
                img, (y0, y1) = render('72,400', digit_h=digit_h)
                f = _analyze_line(img)
                self.assertLessEqual(f.top, y0 + 1)
                self.assertGreaterEqual(f.bottom, y1 - 1)

    def test_recognizer_image_is_at_least_as_tall_as_the_scaled_digits(self):
        # Width is preserved, so scale = received_w / original_w. The received
        # image must be at least as tall as the whole (scaled) glyph height.
        for digit_h in (40, 60, 74, 88):
            with self.subTest(digit_h=digit_h):
                img, (y0, y1) = render('72,400', digit_h=digit_h)
                rec = Recorder()
                scan_tight_fast(img, 'KRW', rec)
                got = rec.images[0]
                scale = got.shape[1] / img.shape[1]
                self.assertGreaterEqual(got.shape[0], (y1 - y0) * scale - 1)

    def test_whole_width_is_always_preserved(self):
        for digit_h in (30, 60, 80):
            img, _ = render('72,400', digit_h=digit_h)
            line, _ = _focus_price_line(img)
            self.assertEqual(line.shape[1], img.shape[1])

    def test_light_digits_on_dark_label(self):
        for digit_h in (30, 50, 74):
            with self.subTest(digit_h=digit_h):
                img, (y0, y1) = render('72,400', digit_h=digit_h,
                                       bg=(35, 35, 35), fg=(245, 245, 245))
                f = _analyze_line(img)
                self.assertTrue(f.located)
                self.assertLessEqual(f.top, y0 + 1)
                self.assertGreaterEqual(f.bottom, y1 - 1)

    def test_white_digits_on_red_and_red_on_white(self):
        for bg, fg in (((40, 40, 200), (250, 250, 250)), ((250, 250, 250), (20, 20, 215))):
            for digit_h in (40, 74):
                with self.subTest(bg=bg, digit_h=digit_h):
                    img, (y0, y1) = render('72,400', digit_h=digit_h, bg=bg, fg=fg)
                    f = _analyze_line(img)
                    self.assertLessEqual(f.top, y0 + 1)
                    self.assertGreaterEqual(f.bottom, y1 - 1)

    def test_noisy_frames_still_keep_every_glyph_row(self):
        for seed in range(4):
            for digit_h in (36, 66, 84):
                with self.subTest(seed=seed, digit_h=digit_h):
                    img, (y0, y1) = render('72,400', digit_h=digit_h, noise=7, seed=seed)
                    f = _analyze_line(img)
                    self.assertLessEqual(f.top, y0 + 1)
                    self.assertGreaterEqual(f.bottom, y1 - 1)

    def test_small_digits_are_trimmed_but_never_cut(self):
        img, (y0, y1) = render('72,400', digit_h=25)
        f = _analyze_line(img)
        self.assertTrue(f.located)
        self.assertLess(f.bottom - f.top, img.shape[0] * .75)   # still zooms in
        self.assertLessEqual(f.top, y0)
        self.assertGreaterEqual(f.bottom, y1)

    def test_recognizer_margin_when_digits_touch_the_edges(self):
        img, (y0, y1) = render('72,400', digit_h=94)
        image, focus = _prepare(img, 0)
        scale = image.shape[1] / img.shape[1]
        self.assertGreater(image.shape[0], (y1 - y0) * scale * 1.05)

    def test_blank_or_noise_only_frame_uses_whole_crop(self):
        rng = np.random.default_rng(3)
        noise = np.clip(rng.normal(128, 6, (99, 461, 3)), 0, 255).astype(np.uint8)
        for img in (np.full((99, 461, 3), 240, np.uint8), noise):
            f = _analyze_line(img)
            self.assertFalse(f.located)
            self.assertEqual((f.top, f.bottom), (0, 99))


class FullPriceIsRequired(unittest.TestCase):
    def test_price_cut_by_left_edge_is_not_published(self):
        # '72,400' with its first digits outside the guide: OCR could only see
        # '400'-like tails. We must ask for re-framing instead of guessing.
        img, _ = render('72,400', digit_h=60, x=-60)
        rec = Recorder('4,400')
        result = scan_tight_fast(img, 'KRW', rec)
        self.assertIsNone(result.amount)
        self.assertEqual(rec.images, [])           # no wasted OCR call
        self.assertIn('çerçevenin dışında', result.note)

    def test_price_cut_by_right_edge_is_not_published(self):
        img, _ = render('72,400', digit_h=60, x='cut_right')
        rec = Recorder('72,4')
        result = scan_tight_fast(img, 'KRW', rec)
        self.assertIsNone(result.amount)
        self.assertIn('çerçevenin dışında', result.note)

    def test_fully_framed_price_is_published(self):
        img, _ = render('72,400', digit_h=60)
        result = scan_tight_fast(img, 'KRW', Recorder('72,400'))
        self.assertEqual(result.amount, Decimal(72400))

    def test_large_fully_framed_price_is_published(self):
        img, _ = render('72,400', digit_h=84)
        result = scan_tight_fast(img, 'KRW', Recorder('72,400'))
        self.assertEqual(result.amount, Decimal(72400))


class Diagnostics(unittest.TestCase):
    def test_debug_payload_shows_crop_input_and_tokens(self):
        img, _ = render('72,400', digit_h=60)
        result = scan_tight_fast(img, 'KRW', Recorder('72,400'))
        self.assertIsNotNone(result.debug)
        self.assertEqual(result.debug['crop'].shape, img.shape)
        self.assertEqual(result.debug['ocr_input'].ndim, 3)
        self.assertEqual(result.debug['tokens'], [('72,400', .97)])

    def test_debug_does_not_affect_equality(self):
        from price_ocr import Reading
        self.assertEqual(Reading(Decimal(1), 'x', .9, 'n', (), {'a': 1}),
                         Reading(Decimal(1), 'x', .9, 'n', ()))


if __name__ == '__main__':
    unittest.main()
