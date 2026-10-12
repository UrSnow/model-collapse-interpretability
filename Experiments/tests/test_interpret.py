"""Mechanistic intervention invariants."""

import unittest

import torch

from project1.data import SymbolTaskConfig, SymbolTaskGenerator
from project1.interpret import (capture_head_inputs, corrupt_matching_label,
                                patched_logits, symbol_score)
from project1.extended import oracle_split
from project1.model import InductionTransformer, TransformerConfig


class InterpretationTests(unittest.TestCase):
    def test_corruption_changes_only_matching_label_to_unused_value(self):
        task = SymbolTaskConfig()
        batch = SymbolTaskGenerator(task, 11).sample(16)
        changed = corrupt_matching_label(batch.input_ids,
                                         batch.matching_label_position, task)
        row = torch.arange(16)
        self.assertTrue(bool((changed[row, batch.matching_label_position]
                              != batch.input_ids[row, batch.matching_label_position]).all()))
        used = batch.input_ids[:, 1:-1:2]
        self.assertFalse(bool((used == changed[row, batch.matching_label_position][:, None]).any()))
        restored = changed.clone()
        restored[row, batch.matching_label_position] = batch.input_ids[
            row, batch.matching_label_position]
        self.assertTrue(torch.equal(restored, batch.input_ids))

    def test_self_patching_preserves_logits_and_removes_hooks(self):
        task = SymbolTaskConfig()
        prompt = SymbolTaskGenerator(task, 12).sample(8).input_ids
        model = InductionTransformer(TransformerConfig(task.vocab_size,
                                                       task.num_labels,
                                                       task.sequence_length))
        model.eval()
        with torch.no_grad():
            expected = model(prompt)
            source = capture_head_inputs(model, prompt)
            actual = patched_logits(model, prompt, source, {0: 0, 1: 0}, task, False)
            after = model(prompt)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertTrue(torch.allclose(after, expected, atol=1e-6))

    def test_symbol_ablation_metric_is_finite(self):
        task = SymbolTaskConfig()
        split = oracle_split(task, 8, 101, "train")
        model = InductionTransformer(TransformerConfig(task.vocab_size,
                                                       task.num_labels,
                                                       task.sequence_length + 2,
                                                       output_vocab_size=task.vocab_size))
        model.eval()
        with torch.no_grad():
            normal = symbol_score(model, split, task)
            ablated = symbol_score(model, split, task, {1: tuple(range(8))})
        self.assertTrue(0 <= normal["tv"] <= 0.5)
        self.assertTrue(0 <= ablated["tv"] <= 0.5)


if __name__ == "__main__":
    unittest.main()
