# -*- coding: utf-8 -*-

"""
<ENTER DESCRIPTION HERE>
"""
import random
from typing import Dict, List

from data.loader import load_training_samples
from data.sample import Sample
from data.siamese_sample import SiameseSample

__author__ = "Jakrin Juangbhanich"
__email__ = "juangbhanich.k@gmail.com"


def generate_siamese_samples():

    # Loop through all samples.
    samples = load_training_samples()
    k = 0

    sample_bins: Dict[str, List[Sample]] = {}

    for sample in samples:
        if sample.label not in sample_bins:
            sample_bins[sample.label] = []
        sample_bins[sample.label].append(sample)

    positive_siamese_samples = []
    for k, bin_samples in sample_bins.items():
        if k == "new_whale":
            continue

        n_samples = len(bin_samples)
        for i in range(n_samples):
            for j in range(i, n_samples):
                siamese_sample = SiameseSample(bin_samples[i], bin_samples[j])
                positive_siamese_samples.append(siamese_sample)

    print(len(positive_siamese_samples))
    # Ok, now generate negative samples. To avoid explosion, cap it at maybe 100 per whale ID.

    samples_bins_list = list(sample_bins.values())
    n_sample_bins = len(samples_bins_list)
    n_negative_target = 5  # Negative samples to mine per sample.
    negative_samples = []

    for bi in range(n_sample_bins):
        s_bin = samples_bins_list[bi]

        for i in range(len(s_bin)):
            sample_i = s_bin[i]
            samples_left = n_negative_target

            random_bins = samples_bins_list[:bi]
            if bi < n_sample_bins - 1:
                random_bins += samples_bins_list[bi+1:]

            while samples_left > 0:
                samples_left -= 1

                # Get a random sample from the remaining batch.
                bin_j = random.choice(random_bins)
                sample_j = random.choice(bin_j)
                siamese_sample = SiameseSample(sample_i, sample_j)
                negative_samples.append(siamese_sample)

    random.shuffle(negative_samples)
    negative_samples = negative_samples[:len(positive_siamese_samples)]
    siamese_samples = positive_siamese_samples + negative_samples
    print(len(siamese_samples))
    return siamese_samples
