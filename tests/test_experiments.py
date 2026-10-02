import unittest

from project1.analyze import phase_change_step
from project1.sweep import parse_seeds


class ExperimentUtilityTests(unittest.TestCase):
    def test_phase_change_uses_first_threshold_crossing(self) -> None:
        history = [
            {"step": 500.0, "validation_accuracy": 0.5},
            {"step": 1000.0, "validation_accuracy": 0.91},
            {"step": 1500.0, "validation_accuracy": 0.89},
        ]
        self.assertEqual(phase_change_step(history), 1000)

    def test_phase_change_can_be_absent(self) -> None:
        history = [{"step": 500.0, "validation_accuracy": 0.5}]
        self.assertIsNone(phase_change_step(history))

    def test_seed_parser_rejects_duplicates(self) -> None:
        self.assertEqual(parse_seeds("0, 2,4"), [0, 2, 4])
        with self.assertRaises(ValueError):
            parse_seeds("1,1")


if __name__ == "__main__":
    unittest.main()

