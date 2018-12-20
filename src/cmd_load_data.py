#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Loads a bunch of data from the directory.
"""

from data.loader import load_training_samples, load_testing_samples

if __name__ == "__main__":
    samples = load_testing_samples()

    # Predict all
