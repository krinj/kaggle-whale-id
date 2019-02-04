#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
<ENTER DESCRIPTION HERE>
"""
from data.loader import load_training_samples

__author__ = "Jakrin Juangbhanich"
__email__ = "juangbhanich.k@gmail.com"
__version__ = "0.0.0"


if __name__ == "__main__":
    samples = load_training_samples()
    samples = {
        "key": 123,
        "qwdwd": 1234
    }
    MyNumber = 1
    thing = MyNumber + 2

