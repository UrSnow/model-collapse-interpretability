import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from project1.omniglot_curves import aggregate, train_scores
from project1.omniglot_data import OmniglotTaskConfig, OmniglotTaskGenerator
from project1.omniglot_extended import oracle_split
from project1.omniglot_inspect import candidate_heads, exemplar_transfer, label_cosine, ov_copy
from project1.omniglot_model import (OmniglotExtendedTransformer,
                                     OmniglotInductionTransformer, OmniglotModelConfig)
from project1 import omniglot_sweep


class OmniglotAnalysisToolTests(unittest.TestCase):
    def setUp(self):
        self.features = torch.randn(12, 5, 4)
        task = OmniglotTaskConfig(train_classes=4, test_classes=2,
                                  num_labels=3, train_label_pairs=2)
        self.generator = OmniglotTaskGenerator(self.features, task, 6)
        self.config = OmniglotModelConfig(feature_dim=4, num_labels=3,
                                          d_model=16, num_heads=4)

    def save_chain(self, path: Path, extended: bool) -> None:
        folder = path / "generation-000"
        folder.mkdir(parents=True)
        model = (OmniglotExtendedTransformer if extended else OmniglotInductionTransformer)(self.config)
        torch.save({"model_state": model.state_dict(), "model_config": asdict(self.config)},
                   folder / "best_model.pt")
        if extended:
            train = oracle_split(self.generator, 16, "train", 3)
            torch.save({"first_labels": train["targets"], "next_indices": train["next_indices"],
                        "second_labels": train["second_labels"]}, folder / "training_targets.pt")
        else:
            train = asdict(self.generator.sample(16, "train"))
            torch.save({"targets": train["targets"]}, folder / "training_labels.pt")
        torch.save({"train": train}, path / "splits.pt")

    def test_clean_targets_score_identically_against_truth(self):
        for extended in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                self.save_chain(Path(directory), extended)
                scores = train_scores(Path(directory), 0, extended)
                self.assertEqual(scores["train_accuracy_vs_targets"],
                                 scores["train_accuracy_vs_truth"])
                self.assertTrue(0 <= scores["train_accuracy_vs_truth"] <= 1)

    def test_curve_aggregate_groups_by_condition_and_generation(self):
        rows = [{"seed": seed, "condition": "clean", "generation": 0, "score": seed}
                for seed in range(3)]
        result = aggregate(rows)
        self.assertEqual(result[0]["n_seeds"], 3)
        self.assertEqual(result[0]["score_mean"], 1)

    def test_candidate_heads_follow_attention_scores(self):
        maps = (torch.zeros(4, 5, 5), torch.zeros(4, 5, 5))
        maps[0][2, 1, 0] = maps[0][2, 3, 2] = 1
        maps[1][3, 4, 1] = 1
        self.assertEqual(candidate_heads(maps), (2, 3))

    def test_embedding_and_transfer_diagnostics_are_bounded(self):
        model = OmniglotInductionTransformer(self.config).eval()
        cosine_mean, cosine = label_cosine(model)
        self.assertEqual(cosine.shape, (3, 3))
        self.assertTrue(0 <= cosine_mean <= 1)
        scores = ov_copy(model, 0, torch.full((4,), 0.25))
        self.assertTrue(0 <= scores["ov_copy_accuracy"] <= 1)
        self.assertTrue(0 <= scores["layer2_ov_copy_accuracy"] <= 1)
        split = asdict(self.generator.sample(16, "test"))
        transfer = exemplar_transfer(model, split, self.features)
        expected = float((model(split["examples"], split["context_labels"]).argmax(-1)
                          == split["targets"]).float().mean())
        self.assertAlmostEqual(transfer["query_same_exemplar_accuracy"], expected)

    def test_sweep_budget_selection_reads_only_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sweep_config.json").write_text(json.dumps(omniglot_sweep.sweep_config()),
                                                    encoding="utf-8")
            last = max(omniglot_sweep.BUDGETS)
            for task in omniglot_sweep.TASKS:
                for rate in omniglot_sweep.LEARNING_RATES:
                    for seed in omniglot_sweep.SEEDS:
                        folder = omniglot_sweep.run_folder(root, task, rate, seed) / "generation-000"
                        folder.mkdir(parents=True)
                        history = [{"step": step, "validation": {
                                    "loss": rate * abs(step - 1500) + 1, "accuracy": 0.5,
                                    "first_label_accuracy": 0.5}}
                                   for step in range(100, last + 1, 100)]
                        (folder / "history.json").write_text(json.dumps(history), encoding="utf-8")
            rows, summary, extra = omniglot_sweep.collect(root)
            at_1000 = [row for row in rows if row["budget"] == 1000]
            self.assertTrue(all(row["best_step"] == 1000 and row["selected_at_budget_end"]
                                for row in at_1000))
            self.assertTrue(all(row["best_step"] == 1500 for row in rows if row["budget"] >= 2000))
            self.assertEqual(extra["choice"]["base"]["budget"], 2000)


if __name__ == "__main__":
    unittest.main()
