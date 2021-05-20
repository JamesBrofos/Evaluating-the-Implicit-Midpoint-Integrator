import unittest

import numpy as np

import hmc


class TestSummarize(unittest.TestCase):
    def test_summarize(self):
        x = np.random.normal(size=(10000, 3))
        metrics = hmc.summarize(x, ('a', 'b', 'c'), should_print=False)

