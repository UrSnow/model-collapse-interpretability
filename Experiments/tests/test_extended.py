"""Extended three-token task invariants."""

import unittest

import torch

from project1.data import SymbolTaskConfig
from project1.extended import (label_for_symbol, oracle_split, replacement_targets,
                               resampled_oracle_targets, sequence_logits,
                               target_context_index, training_loss)
from project1.model import InductionTransformer, TransformerConfig


class ExtendedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.task = SymbolTaskConfig()
        cls.split = oracle_split(cls.task, 32, 101, "train")
        config = TransformerConfig(cls.task.vocab_size, cls.task.num_labels,
                                   cls.task.sequence_length + 2,
                                   output_vocab_size=cls.task.vocab_size)
        cls.model = InductionTransformer(config)

    def test_oracle_next_label_matches_context(self):
        expected = label_for_symbol(self.split["input_ids"],
                                    self.split["next_symbols"], self.task)
        self.assertTrue(torch.equal(expected, self.split["second_labels"]))

    def test_rejects_out_of_context_symbol(self):
        prompt = self.split["input_ids"][:1]
        absent = next(value for value in range(self.task.num_symbols)
                      if value not in prompt[0, 0:-1:2].tolist())
        with self.assertRaisesRegex(ValueError, "must occur"):
            target_context_index(prompt, torch.tensor([absent]), self.task.context_pairs)

    def test_first_prediction_cannot_see_later_targets(self):
        split = self.split
        first, _, _ = sequence_logits(self.model, split["input_ids"],
                                       split["first_labels"], split["next_symbols"],
                                       self.task)
        other_first, _, _ = sequence_logits(
            self.model, split["input_ids"],
            (split["first_labels"] + 1) % self.task.num_labels,
            split["input_ids"][:, 0], self.task)
        self.assertTrue(torch.allclose(first, other_first, atol=1e-6))

    def test_loss_backpropagates_and_clean_replacement_is_unchanged(self):
        split = self.split
        loss = training_loss(self.model, split["input_ids"], split["first_labels"],
                             split["next_symbols"], split["second_labels"], self.task)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(self.model.unembed.weight.grad)
        targets = replacement_targets(self.model, split, self.task, 0.0, 1.0, 7)
        self.assertEqual(int(targets["synthetic_mask"].sum()), 0)
        for name in ("first_labels", "next_symbols", "second_labels"):
            self.assertTrue(torch.equal(targets[name], split[name]))

    def test_fully_synthetic_symbols_remain_in_context(self):
        targets = replacement_targets(self.model, self.split, self.task, 1.0, 1.0, 8)
        self.assertEqual(int(targets["synthetic_mask"].sum()), 32)
        indices = target_context_index(self.split["input_ids"],
                                       targets["next_symbols"], self.task.context_pairs)
        self.assertEqual(indices.shape, (32,))
        self.assertTrue(bool(((targets["first_labels"] >= 0) &
                              (targets["first_labels"] < self.task.num_labels)).all()))

    def test_true_generator_resampling_preserves_label_consistency(self):
        targets = resampled_oracle_targets(self.split, self.task, 1.0, 9)
        self.assertEqual(int(targets["resampled_mask"].sum()), 32)
        self.assertEqual(int(targets["synthetic_mask"].sum()), 0)
        expected = label_for_symbol(self.split["input_ids"],
                                    targets["next_symbols"], self.task)
        self.assertTrue(torch.equal(expected, targets["second_labels"]))

    def test_sampled_and_expected_symbol_objectives_are_distinct(self):
        split = self.split
        clean = training_loss(self.model, split["input_ids"], split["first_labels"],
                              split["next_symbols"], split["second_labels"], self.task,
                              torch.zeros(32, dtype=torch.bool))
        sampled = training_loss(self.model, split["input_ids"], split["first_labels"],
                                split["next_symbols"], split["second_labels"], self.task,
                                torch.ones(32, dtype=torch.bool))
        self.assertGreater(abs((clean - sampled).detach().item()), 1e-5)


if __name__ == "__main__":
    unittest.main()
