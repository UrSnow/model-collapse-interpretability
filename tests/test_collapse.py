import unittest

import torch

from project1.collapse import fixed_split, replacement_labels
from project1.data import SymbolTaskConfig


class ConstantTeacher(torch.nn.Module):
    def forward(self, tokens):
        logits = torch.full((len(tokens), 5), -1000.0)
        logits[:, 0] = 1000.0
        return logits


class CollapseTests(unittest.TestCase):
    def test_splits_disjoint_even_with_identical_seed(self):
        task = SymbolTaskConfig()
        sets = [set(map(tuple, fixed_split(task, 100, 42, name).input_ids.tolist()))
                for name in ("train", "validation", "test", "diagnostic")]
        self.assertEqual(sum(map(len, sets)), len(set.union(*sets)))

    def test_exclusion_skips_previously_seen_test_prompts(self):
        task = SymbolTaskConfig()
        first = fixed_split(task, 32, 123, "test")
        excluded = set(map(tuple, first.input_ids.tolist()))
        second = fixed_split(task, 32, 123, "test", excluded)
        self.assertFalse(excluded & set(map(tuple, second.input_ids.tolist())))

    def test_replacement_preserves_real_rows_and_wrong_labels(self):
        batch = fixed_split(SymbolTaskConfig(), 100, 42, "train")
        labels, mask = replacement_labels(ConstantTeacher(), batch, 0.4, 1.0, 123)
        self.assertEqual(mask.sum().item(), 40)
        self.assertTrue(torch.equal(labels[~mask], batch.targets[~mask]))
        self.assertTrue((labels[mask] == 0).all())
        self.assertTrue((labels[mask] != batch.targets[mask]).any())
        repeated, repeated_mask = replacement_labels(ConstantTeacher(), batch, 0.4, 1.0, 123)
        self.assertTrue(torch.equal(labels, repeated))
        self.assertTrue(torch.equal(mask, repeated_mask))

    def test_clean_and_fully_synthetic_boundaries(self):
        batch = fixed_split(SymbolTaskConfig(), 100, 42, "train")
        clean, mask = replacement_labels(ConstantTeacher(), batch, 0, 1, 123)
        self.assertTrue(torch.equal(clean, batch.targets))
        self.assertFalse(mask.any())
        generated, mask = replacement_labels(ConstantTeacher(), batch, 1, 1, 123)
        self.assertTrue(mask.all())
        self.assertTrue((generated == 0).all())
