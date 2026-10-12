import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from project1.omniglot_data import OmniglotTaskConfig, OmniglotTaskGenerator
from project1.omniglot_final_evaluation import (aggregate_rows, batches_from_manifest,
                                                  episode_keys, make_manifest, prior_test_keys,
                                                  validate_manifest)


class FinalEvaluationTests(unittest.TestCase):
    def test_unique_sampling_excludes_prior_prompt(self):
        features = torch.arange(12 * 5 * 4, dtype=torch.float32).reshape(12, 5, 4)
        task = OmniglotTaskConfig(train_classes=4, test_classes=2,
                                  num_labels=3, train_label_pairs=2)
        first = OmniglotTaskGenerator(features, task, 9).sample_unique(1, "test")
        excluded = episode_keys({"class_ids": first.class_ids,
                                 "context_labels": first.context_labels})
        second = OmniglotTaskGenerator(features, task, 9).sample_unique(1, "test", excluded)
        self.assertTrue(episode_keys({"class_ids": second.class_ids,
                                      "context_labels": second.context_labels}).isdisjoint(excluded))

    def test_manifest_is_reproducible_and_reconstructs_targets(self):
        features = torch.randn(1623, 5, 4)
        first = make_manifest(features, 2, set())
        second = make_manifest(features, 2, set())
        validate_manifest(first, 2, set())
        used = set()
        for seed in first:
            for key in first[seed]:
                torch.testing.assert_close(first[seed][key], second[seed][key])
            keys = episode_keys(first[seed])
            self.assertTrue(keys.isdisjoint(used))
            used.update(keys)
            batch, extended = batches_from_manifest(features, first[seed])
            rows = torch.arange(len(batch))
            torch.testing.assert_close(extended["second_labels"],
                                       batch.context_labels[rows, extended["next_indices"]])
            torch.testing.assert_close(extended["next_class_ids"],
                                       batch.class_ids[rows, extended["next_indices"]])
        with self.assertRaisesRegex(ValueError, "repeats a prior"):
            validate_manifest(first, 2, episode_keys(first[0]))

    def test_aggregate_rejects_missing_seed(self):
        with self.assertRaisesRegex(ValueError, "Incomplete final group"):
            aggregate_rows([{"family": "base", "condition": "clean",
                             "generation": 0, "accuracy": 1.0}],
                           "condition", {"base": ("accuracy",)})

    def test_prior_splits_are_found_with_custom_study_names(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            study = root / "base-study" / "seed-00" / "clean"
            study.mkdir(parents=True)
            split = {"test": {"class_ids": torch.tensor([[8, 9, 8]]),
                              "context_labels": torch.tensor([[0, 1]])}}
            torch.save(split, study / "splits.pt")
            keys, sources = prior_test_keys(root, (root / "base-study",))
            self.assertEqual(keys, {(8, 9, 8, 0, 1)})
            self.assertEqual(len(sources), 1)


if __name__ == "__main__":
    unittest.main()
