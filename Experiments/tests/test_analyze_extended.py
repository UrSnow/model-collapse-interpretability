"""Distributional analysis invariants."""

import unittest

import torch

from project1.analyze_extended import symbol_tv_from_true_distribution
from project1.data import SymbolTaskConfig


class AnalyzeExtendedTests(unittest.TestCase):
    def test_global_oracle_uses_context_probabilities_not_sample_labels(self):
        task = SymbolTaskConfig(num_symbols=4, num_labels=2, context_pairs=2)
        prompt = torch.tensor([[0, 4, 1, 5, 0],
                               [2, 4, 3, 5, 2]])
        balanced = torch.tensor([0, 3])
        concentrated = torch.tensor([0, 2])
        self.assertAlmostEqual(symbol_tv_from_true_distribution(balanced, prompt, task), 0.5)
        self.assertAlmostEqual(symbol_tv_from_true_distribution(concentrated, prompt, task), 0.5)
        repeated = prompt.repeat(2, 1)
        exact = torch.tensor([0, 1, 2, 3])
        self.assertAlmostEqual(symbol_tv_from_true_distribution(exact, repeated, task), 0.0)


if __name__ == "__main__":
    unittest.main()
