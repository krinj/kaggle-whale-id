# -*- coding: utf-8 -*-

"""
Interface for predicting the ID of a sample.
"""

from abc import abstractmethod
from typing import List
from data.sample import Sample


class Predictor:
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, sample: Sample) -> List[str]:
        """ Returns a list of string IDs for matched whales. """
        return ["new_whale", "w_23a388d", "w_9b5109b", "w_9c506f6", "w_0369a5c"]
