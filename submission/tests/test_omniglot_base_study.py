import unittest

from project1.omniglot_base_study import CONDITIONS, study_config
from project1.omniglot_study import SEEDS, SETTINGS, aggregate


class OmniglotBaseStudyTests(unittest.TestCase):
    def test_base_design_matches_extended_seeds_and_budget(self):
        config = study_config()
        self.assertEqual(config["seeds"], list(SEEDS))
        self.assertEqual(config["settings"], SETTINGS)
        self.assertEqual(CONDITIONS, {"clean": 0.0, "feedback": 1.0})

    def test_base_aggregation_uses_only_base_conditions(self):
        rows = [{"seed": seed, "condition": condition, "generation": generation,
                 "accuracy": 0.5 + seed / 100}
                for condition in CONDITIONS
                for generation in range(SETTINGS["generations"])
                for seed in SEEDS]
        result = aggregate(rows, ("accuracy",), tuple(CONDITIONS))
        self.assertEqual(len(result), len(CONDITIONS) * SETTINGS["generations"])
        self.assertAlmostEqual(result[0]["accuracy_mean"], 0.52)


if __name__ == "__main__":
    unittest.main()
