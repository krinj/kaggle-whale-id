# -*- coding: utf-8 -*-

"""
<ENTER DESCRIPTION HERE>
"""

import cv2
import numpy as np
from PIL import Image
from raid.data.interface.i_single_clf_sample import ISingleClfSample
from data.sample import Sample

__author__ = "Jakrin Juangbhanich"
__email__ = "juangbhanich.k@gmail.com"


class SiameseSample(ISingleClfSample):

    _label_map = {0: "negative", 1: "positive"}

    def __init__(self, sample_1: Sample, sample_2: Sample):
        id = f"ss_{sample_1.image_id}_{sample_2.image_id}"
        label = (1 if sample_1.label == sample_2.label else 0)
        super().__init__(id, label, self._label_map)

        self.sample_1: Sample = sample_1
        self.sample_2: Sample = sample_2

    def get_display_image(self):
        canvas = np.zeros((128, 256, 3), dtype=np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_1 = cv2.resize(self.sample_1.image, (128, 128))
        image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        image_1 = clahe.apply(image_1)
        # image_1 = cv2.equalizeHist(image_1)
        image_1 = cv2.cvtColor(image_1, cv2.COLOR_GRAY2BGR)

        image_2 = cv2.resize(self.sample_2.image, (128, 128))
        image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
        image_2 = clahe.apply(image_2)
        # image_2 = cv2.equalizeHist(image_2)
        image_2 = cv2.cvtColor(image_2, cv2.COLOR_GRAY2BGR)

        canvas[:128, :128] = image_1
        canvas[:128, 128:] = image_2

        return canvas
