import unittest

import torch

from project1.omniglot_base import attention_diagnostic, evaluate, replacement_labels
from project1.omniglot_data import OmniglotTaskConfig, OmniglotTaskGenerator
from project1.omniglot_model import OmniglotInductionTransformer, OmniglotModelConfig


class OmniglotDataTests(unittest.TestCase):
    def setUp(self):
        self.features = torch.arange(12 * 5 * 4, dtype=torch.float32).reshape(12, 5, 4)
        self.config = OmniglotTaskConfig(train_classes=4, test_classes=2,
                                         num_labels=3, train_label_pairs=2)

    def test_query_feature_and_label_match_context(self):
        batch = OmniglotTaskGenerator(self.features, self.config, 3).sample(64, "train")
        rows = torch.arange(len(batch))
        match = batch.query_context_index
        torch.testing.assert_close(batch.class_ids[rows, match], batch.class_ids[:, 2])
        torch.testing.assert_close(batch.examples[rows, match], batch.examples[:, 2])
        torch.testing.assert_close(batch.context_labels[rows, match], batch.targets)
        torch.testing.assert_close(batch.matching_label_position, 2 * match + 1)

    def test_splits_and_seeds_are_reproducible(self):
        first = OmniglotTaskGenerator(self.features, self.config, 7)
        second = OmniglotTaskGenerator(self.features, self.config, 7)
        train = first.sample_unique(16, "train")
        repeat = second.sample_unique(16, "train")
        torch.testing.assert_close(train.class_ids, repeat.class_ids)
        torch.testing.assert_close(train.context_labels, repeat.context_labels)
        validation = first.sample_unique(8, "validation")
        test = first.sample_unique(8, "test")
        diagnostic = first.sample_unique(8, "diagnostic")
        self.assertTrue(bool((train.class_ids < 4).all()))
        self.assertTrue(bool((validation.class_ids < 4).all()))
        self.assertTrue(bool((test.class_ids >= 10).all()))
        self.assertTrue(bool(((diagnostic.class_ids >= 4) & (diagnostic.class_ids < 10)).all()))
        self.assertTrue(set(map(tuple, first.train_pairs.tolist())).isdisjoint(
            set(map(tuple, first.validation_pairs.tolist()))))


class OmniglotModelTests(unittest.TestCase):
    def setUp(self):
        features = torch.randn(12, 5, 4)
        task = OmniglotTaskConfig(train_classes=4, test_classes=2,
                                  num_labels=3, train_label_pairs=2)
        self.batch = OmniglotTaskGenerator(features, task, 8).sample(12, "train")
        self.model = OmniglotInductionTransformer(
            OmniglotModelConfig(feature_dim=4, num_labels=3, d_model=16, num_heads=4))

    def test_shapes_causal_attention_and_ablations(self):
        logits, maps = self.model(self.batch.examples, self.batch.context_labels,
                                  return_attention=True)
        self.assertEqual(tuple(logits.shape), (12, 3))
        self.assertEqual(tuple(maps[0].shape), (12, 4, 5, 5))
        self.assertTrue(bool((maps[0].triu(1) == 0).all()))
        ablated = self.model(self.batch.examples, self.batch.context_labels,
                             ablate_heads={1: (0, 1)})
        self.assertEqual(tuple(ablated.shape), (12, 3))
        self.assertTrue(bool(torch.isfinite(ablated).all()))

    def test_metrics_and_replacement_boundaries(self):
        self.assertTrue(0 <= evaluate(self.model, self.batch)["accuracy"] <= 1)
        self.assertEqual(len(attention_diagnostic(self.model, self.batch)["second_layer_induction"]), 4)
        labels, mask = replacement_labels(self.model, self.batch, 0, 1, 4)
        torch.testing.assert_close(labels, self.batch.targets)
        self.assertEqual(int(mask.sum()), 0)
        labels, mask = replacement_labels(self.model, self.batch, 1, 1, 4)
        self.assertEqual(int(mask.sum()), len(self.batch))
        self.assertTrue(bool(((labels >= 0) & (labels < 3)).all()))


if __name__ == "__main__":
    unittest.main()
