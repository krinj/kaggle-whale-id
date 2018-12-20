# -*- coding: utf-8 -*-

"""
<ENTER DESCRIPTION HERE>
"""

from typing import List
from k_util.logger import Logger
from data.sample import Sample


def evaluate(samples: List[Sample], n: int=5):

    n_total: int = len(samples)
    n_correct: int = 0

    for sample in samples:
        n_correct += evaluate_single(sample, n)

    Logger.field("Total Score", n_correct/n_total)
    return n_correct/n_total


def evaluate_single(sample: Sample, n: int=5):
    predictions = sample.predictions[:n]
    if sample.label in predictions:
        return 1
    return 0
