from unittest import TestCase
import numpy as np
from pytools_litho_analysis.legacy import Analyzer, Image
import cv2
import os


class TestLegacyAnalyzer(TestCase):

    def setUp(self):
        filename = os.path.join(
            os.path.dirname(__file__), "test_images", "legacy_analysis_test.jpg"
        )
        self.assertTrue(os.path.exists(filename))
        self.image_file = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        self.image = Image(self.image_file)

    def test_get_scale_only(self):
        scale = self.image.get_scale()
        self.assertEqual(round(scale, 4), 0.1176)

    def test_get_scale_and_pm(self):
        scale, pm = self.image.get_scale(return_pm=True)
        self.assertEqual(round(scale, 4), 0.1176)
        self.assertEqual(round(pm, 4), 0.0001)
