# -*- coding: utf-8 -*-

"""
Functions to evaluate the quality of the predictions.
"""

from typing import List
from k_util.logger import Logger
from data.sample import Sample


def evaluate(samples: List[Sample], n: int=5):
    """ Will evaluate the score for a list of samples.
    Returns the total correct for top-n."""
    n_total: int = len(samples)
    n_correct: int = 0

    for sample in samples:
        n_correct += evaluate_single(sample, n)

    Logger.field("Total Score", n_correct/n_total)
    return n_correct/n_total


def evaluate_single(sample: Sample, n: int=5):
    """ Evaluate the score for a single sample.
    Will be 1 if the ID is in the top n-predictions, otherwise 0. """
    predictions = sample.predictions[:n]
    if sample.label in predictions:
        return 1
    return 0
