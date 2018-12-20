# -*- coding: utf-8 -*-

"""
<ENTER DESCRIPTION HERE>
"""
import csv
from typing import List
from data.sample import Sample


def generate_submission(samples: List[Sample], output_path: str="predictions.csv"):
    with open(output_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Id"])

        for sample in samples:
            prediction_string = " ".join(sample.predictions)
            writer.writerow([sample.image_id, prediction_string])
