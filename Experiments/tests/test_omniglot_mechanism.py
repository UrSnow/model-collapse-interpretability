import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from project1.omniglot_data import OmniglotTaskConfig, OmniglotTaskGenerator
from project1.omniglot_mechanism import (capture_head_inputs, corrupt_query_label,
                                         inspect_checkpoint, patched_first_logits)
from project1.omniglot_model import (OmniglotExtendedTransformer,
                                     OmniglotInductionTransformer, OmniglotModelConfig)


class OmniglotMechanismTests(unittest.TestCase):
    def setUp(self):
        features = torch.randn(12, 5, 4)
        task = OmniglotTaskConfig(train_classes=4, test_classes=2,
                                  num_labels=3, train_label_pairs=2)
        self.batch = OmniglotTaskGenerator(features, task, 6).sample(12, "train")
        self.config = OmniglotModelConfig(feature_dim=4, num_labels=3,
                                          d_model=16, num_heads=4)

    def test_corruption_changes_only_matching_label(self):
        corrupted, new_targets = corrupt_query_label(self.batch, 3)
        rows = torch.arange(len(self.batch))
        matching = self.batch.query_context_index
        other = 1 - matching
        torch.testing.assert_close(corrupted[rows, other],
                                   self.batch.context_labels[rows, other])
        torch.testing.assert_close(corrupted[rows, matching], new_targets)
        self.assertTrue(bool((new_targets != self.batch.targets).all()))
        self.assertTrue(bool((new_targets != corrupted[rows, other]).all()))

    def test_self_patch_preserves_logits_and_removes_hooks(self):
        for model_class in (OmniglotInductionTransformer, OmniglotExtendedTransformer):
            model = model_class(self.config).eval()
            expected = model(self.batch.examples, self.batch.context_labels)
            source = capture_head_inputs(model, self.batch)
            patched = patched_first_logits(model, self.batch,
                                           self.batch.context_labels, source,
                                           {0: 0, 1: 1})
            torch.testing.assert_close(expected, patched)
            self.assertTrue(all(not block.output._forward_pre_hooks
                                for block in model.blocks))

    def test_inspect_frozen_base_and_extended_checkpoints(self):
        for family, model_class in (("base", OmniglotInductionTransformer),
                                    ("extended", OmniglotExtendedTransformer)):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                (path / "generation-000").mkdir()
                model = model_class(self.config)
                (path / "protocol.json").write_text(json.dumps({"model": asdict(self.config)}),
                                                    encoding="utf-8")
                split = asdict(self.batch)
                if family == "extended":
                    indices = torch.randint(2, (len(self.batch),))
                    split["next_indices"] = indices
                    split["second_labels"] = self.batch.context_labels.gather(
                        1, indices[:, None]).squeeze(1)
                torch.save({"diagnostic": split}, path / "splits.pt")
                torch.save({"model_state": model.state_dict()},
                           path / "generation-000" / "best_model.pt")
                summary, heads = inspect_checkpoint(path, family, 0, 12)
                self.assertEqual(summary["diagnostic_size"], 12)
                self.assertEqual(len(heads), 8)
                self.assertTrue(all(row["layer"] in (0, 1) for row in heads))
                self.assertIn(summary["candidate_previous_is_max_ablation"], (0.0, 1.0))
                self.assertIn(summary["candidate_induction_is_max_ablation"], (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
