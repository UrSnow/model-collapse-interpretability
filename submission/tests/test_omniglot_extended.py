import unittest

import torch

from project1.omniglot_data import OmniglotTaskConfig, OmniglotTaskGenerator
from project1.omniglot_extended import (distribution_stats, evaluate, oracle_split,
                                        replacement_targets, resampled_oracle_targets, sequence_logits,
                                        target_labels, training_loss)
from project1.omniglot_model import OmniglotExtendedTransformer, OmniglotModelConfig


class OmniglotExtendedTests(unittest.TestCase):
    def setUp(self):
        features = torch.randn(12, 5, 4)
        task = OmniglotTaskConfig(train_classes=4, test_classes=2,
                                  num_labels=3, train_label_pairs=2)
        generator = OmniglotTaskGenerator(features, task, 8)
        self.split = oracle_split(generator, 12, "train", 41)
        self.model = OmniglotExtendedTransformer(
            OmniglotModelConfig(feature_dim=4, num_labels=3, d_model=16, num_heads=4))

    def test_pointer_resolves_to_context_feature_and_label(self):
        split = self.split
        row = torch.arange(len(split["targets"]))
        torch.testing.assert_close(split["next_class_ids"],
                                   split["class_ids"][row, split["next_indices"]])
        torch.testing.assert_close(split["second_labels"],
                                   target_labels(split, split["next_indices"]))
        self.assertTrue(bool((split["next_class_ids"] < 4).all()))

    def test_causal_outputs_and_loss(self):
        split = self.split
        first, pointer, second = sequence_logits(self.model, split, split["targets"],
                                                 split["next_indices"])
        self.assertEqual(tuple(first.shape), (12, 3))
        self.assertEqual(tuple(pointer.shape), (12, 2))
        self.assertEqual(tuple(second.shape), (12, 3))
        torch.testing.assert_close(pointer, self.model.pointer_logits(
            split["examples"], split["context_labels"], split["targets"]))
        _, _, _, maps = self.model.forward_extended(split["examples"], split["context_labels"],
                                                     split["targets"], split["next_indices"],
                                                     return_attention=True)
        self.assertEqual(tuple(maps[0].shape), (12, 4, 7, 7))
        self.assertTrue(bool((maps[0].triu(1) == 0).all()))
        changed = split["next_indices"].logical_not().long()
        changed_first, changed_pointer, _ = sequence_logits(self.model, split,
                                                              split["targets"], changed)
        torch.testing.assert_close(first, changed_first)
        torch.testing.assert_close(pointer, changed_pointer)
        targets = {"first_labels": split["targets"], "next_indices": split["next_indices"],
                   "second_labels": split["second_labels"],
                   "synthetic_mask": torch.zeros(12, dtype=torch.bool)}
        loss = training_loss(self.model, split, targets)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIsNotNone(self.model.pointer_head.weight.grad)
        self.assertTrue(0 <= evaluate(self.model, split)["greedy_joint_accuracy"] <= 1)

    def test_feedback_boundaries_and_class_distribution(self):
        split = self.split
        clean = replacement_targets(self.model, split, 0, 1, 5)
        self.assertEqual(int(clean["synthetic_mask"].sum()), 0)
        torch.testing.assert_close(clean["first_labels"], split["targets"])
        torch.testing.assert_close(clean["next_indices"], split["next_indices"])
        full = replacement_targets(self.model, split, 1, 1, 5)
        self.assertEqual(int(full["synthetic_mask"].sum()), 12)
        self.assertTrue(bool(((full["next_indices"] >= 0) & (full["next_indices"] < 2)).all()))
        stats = distribution_stats(split, full, 4)
        self.assertTrue(1 <= stats["coverage"] <= 4)
        self.assertTrue(0 <= stats["total_variation_from_fixed_oracle"] <= 1)

    def test_oracle_resampling_preserves_labels_and_objectives_differ(self):
        split = self.split
        targets = resampled_oracle_targets(split, 1, 5)
        self.assertEqual(int(targets["resampled_mask"].sum()), len(split["targets"]))
        self.assertEqual(int(targets["synthetic_mask"].sum()), 0)
        torch.testing.assert_close(targets["first_labels"], split["targets"])
        torch.testing.assert_close(targets["second_labels"],
                                   target_labels(split, targets["next_indices"]))
        expected = training_loss(self.model, split, targets, "expected")
        sampled = training_loss(self.model, split, targets, "sampled")
        self.assertNotAlmostEqual(expected.item(), sampled.item(), places=6)


if __name__ == "__main__":
    unittest.main()
