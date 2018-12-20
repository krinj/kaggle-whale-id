# -*- coding: utf-8 -*-

"""
Loads all the samples from the specified source.
"""
import os
from typing import List

import pandas
from k_util.logger import Logger

from data.sample import Sample
from util.settings import Settings


def load_training_samples():
    """ Load the training samples. """
    train_image_path = Settings.get("TRAIN_IMAGE_PATH")
    label_path = Settings.get("LABEL_PATH")
    data_frame = pandas.read_csv(label_path)

    samples: List[Sample] = []

    for index, row in data_frame.iterrows():
        sample = Sample(image_id=row["Image"], image_dir=train_image_path, label=row["Id"])
        samples.append(sample)

    Logger.field("Samples Loaded", len(samples))
    return samples


def load_testing_samples():
    """ Load the training samples. """
    test_image_path = Settings.get("TEST_IMAGE_PATH")
    samples: List[Sample] = []

    files = os.listdir(test_image_path)
    for file_name in files:
        if ".jpg" not in file_name and ".png" not in file_name:
            continue

        sample = Sample(image_id=file_name, image_dir=test_image_path)
        samples.append(sample)

    Logger.field("Samples Loaded", len(samples))
    return samples
