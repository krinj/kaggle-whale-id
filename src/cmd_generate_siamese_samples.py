#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
<ENTER DESCRIPTION HERE>
"""
from k_util import pather
from raid import IDataset
from raid.visual import data_renderer

from data.siamese_generator import generate_siamese_samples

__author__ = "Jakrin Juangbhanich"
__email__ = "juangbhanich.k@gmail.com"
__version__ = "0.0.0"

if __name__ == "__main__":
    print("Hello World")
    siamese_samples = generate_siamese_samples()
    dataset = IDataset(siamese_samples, {0: "negative", 1: "positive"})

    dist_path = "output/class_dist.png"
    pather.create("output/class_dist.png")
    dataset.draw_label_distribution(dist_path)
    dataset.shuffle(balanced=False)

    data_renderer.render_samples_by_label(dataset, f"output/render_by_label", size=(128, 258))
