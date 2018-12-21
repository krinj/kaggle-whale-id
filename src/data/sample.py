# -*- coding: utf-8 -*-

"""
A sample piece of the data. In this case, this is an image and a label.
"""

from typing import List
import cv2


class Sample:
    def __init__(self, image_id: str, image_dir: str, label: str=None):
        self.image_id: str = image_id
        self.label: str = label
        self.predictions: List[str] = []
        self.image_path = f"{image_dir}/{self.image_id}"

    @property
    def image(self):
        """ Lazy load the image. """
        return cv2.imread(self.image_path)

    def __repr__(self):
        return f"[Sample: {self.image_id} | {self.label}]"
