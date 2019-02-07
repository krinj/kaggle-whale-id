# -*- coding: utf-8 -*-

"""
A sample piece of the data. In this case, this is an image and a label.
"""

from typing import List
import cv2
from PIL import Image


class Sample:

    CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __init__(self, image_id: str, image_dir: str, label: str=None):
        self.image_id: str = image_id
        self.label: str = label
        self.predictions: List[str] = []
        self.image_path = f"{image_dir}/{self.image_id}"

    @property
    def image(self):
        """ Lazy load the image. """
        return cv2.imread(self.image_path)

    @property
    def pil_image(self):
        image = self.image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = self.CLAHE.apply(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = Image.fromarray(image)
        return image

    def __repr__(self):
        return f"[Sample: {self.image_id} | {self.label}]"
