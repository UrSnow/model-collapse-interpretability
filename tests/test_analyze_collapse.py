"""Tests for matched-chain analysis safeguards."""

import unittest

import torch

from project1.analyze_collapse import CONDITIONS, check_matched_runs


def matched_runs():
    runs = {}
    for name, fraction in CONDITIONS.items():
        runs[name] = {
            "protocol": {"generations": 1, "steps": 10,
                         "synthetic_fraction": fraction, "output_dir": name},
            "summaries": [{"generation": 0, "test": {"accuracy": 1.0}}],
            "splits": {"train": {"targets": torch.tensor([0, 1])}},
        }
    return runs


class AnalyzeCollapseTests(unittest.TestCase):
    def test_accepts_matched_protocols_and_splits(self):
        check_matched_runs(matched_runs())

    def test_rejects_changed_protocol(self):
        runs = matched_runs()
        runs["mixed"]["protocol"]["steps"] = 11
        with self.assertRaisesRegex(ValueError, "protocol differs"):
            check_matched_runs(runs)

    def test_rejects_changed_split(self):
        runs = matched_runs()
        runs["synthetic"]["splits"]["train"]["targets"] = torch.tensor([1, 0])
        with self.assertRaisesRegex(ValueError, "split differs"):
            check_matched_runs(runs)


if __name__ == "__main__":
    unittest.main()
