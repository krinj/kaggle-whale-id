# -*- coding: utf-8 -*-

"""
<ENTER DESCRIPTION HERE>
"""
from typing import List

from data.sample import Sample


class Predictor:
    def __init__(self):
        pass

    def predict(self, sample: Sample) -> List[str]:
        # Return the benchmark submissions.
        return ["new_whale", "w_23a388d", "w_9b5109b", "w_9c506f6", "w_0369a5c"]
