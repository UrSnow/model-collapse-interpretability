import unittest
import json
import tempfile
from pathlib import Path

from project1.omniglot_study import (CONDITIONS, SEEDS, SETTINGS, aggregate,
                                    audit_seed, study_config)


class OmniglotStudyTests(unittest.TestCase):
    def test_locked_design_has_matched_controls(self):
        config = study_config()
        self.assertEqual(config["seeds"], list(SEEDS))
        self.assertEqual(CONDITIONS["clean"], (0.0, "model"))
        self.assertEqual(CONDITIONS["partial"], (0.9, "model"))
        self.assertEqual(CONDITIONS["feedback"], (1.0, "model"))
        self.assertEqual(CONDITIONS["oracle"], (1.0, "oracle"))

    def test_aggregate_is_seed_level_and_requires_all_seeds(self):
        rows = [{"seed": seed, "condition": condition, "generation": generation,
                 "score": seed / 10 + generation / 100}
                for condition in CONDITIONS
                for generation in range(SETTINGS["generations"])
                for seed in SEEDS]
        result = aggregate(rows, ("score",))
        self.assertEqual(len(result), len(CONDITIONS) * SETTINGS["generations"])
        self.assertEqual(result[0]["n_seeds"], len(SEEDS))
        self.assertAlmostEqual(result[0]["score_mean"], 0.2)
        with self.assertRaisesRegex(ValueError, "Missing seeds"):
            aggregate(rows[1:], ("score",))
        paired = aggregate([row for row in rows if row["condition"] != "clean"],
                           ("score",), ("partial", "feedback", "oracle"))
        self.assertEqual(len(paired), 3 * SETTINGS["generations"])

    def test_audit_rejects_mismatched_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for condition, (fraction, source) in CONDITIONS.items():
                folder = root / "seed-00" / condition
                (folder / "generation-000").mkdir(parents=True)
                (folder / "protocol.json").write_text(json.dumps({
                    "seed": 0, "synthetic_fraction": fraction,
                    "feedback_source": source, "task": "same"}), encoding="utf-8")
                (folder / "summary.json").write_text(json.dumps(
                    [{"generation": i} for i in range(3)]), encoding="utf-8")
                for name in ("splits.pt", "generation-000/training_targets.pt",
                             "generation-000/best_model.pt"):
                    (folder / name).write_bytes(b"matched")
            self.assertEqual(len(audit_seed(root, 0, 3)), len(CONDITIONS))
            (root / "seed-00" / "oracle" / "splits.pt").write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "unmatched artifact"):
                audit_seed(root, 0, 3)


if __name__ == "__main__":
    unittest.main()
