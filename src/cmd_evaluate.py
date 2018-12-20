#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Loads a bunch of data from the directory.
"""
from k_util.logger import Logger

from ai.predictor import Predictor
from data.loader import load_training_samples, load_testing_samples
from eval.evaluator import evaluate

if __name__ == "__main__":
    samples = load_training_samples()

    # Predict all
    predictor = Predictor()
    for s in samples:
        s.predictions = predictor.predict(s)

    Logger.log("Samples Predicted")

    # Evaluate the samples.
    evaluate(samples)
