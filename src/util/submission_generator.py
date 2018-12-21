# -*- coding: utf-8 -*-

"""
Function to generate a valid CSV for a set of predicted samples.
"""

import csv
from typing import List
from data.sample import Sample


def generate_submission(samples: List[Sample], output_path: str="predictions.csv"):
    """ Given a list of predicted samples, generate a CSV that we can submit to Kaggle. """
    with open(output_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Id"])

        for sample in samples:
            prediction_string = " ".join(sample.predictions[:5])
            writer.writerow([sample.image_id, prediction_string])
