import unittest

import torch

from project1.data import SymbolTaskConfig, SymbolTaskGenerator


class SymbolTaskGeneratorTests(unittest.TestCase):
    def test_query_target_matches_context_mapping(self) -> None:
        config = SymbolTaskConfig(num_symbols=12, num_labels=4, context_pairs=4)
        batch = SymbolTaskGenerator(config, seed=7).sample(batch_size=32)
        rows = torch.arange(batch.input_ids.shape[0])

        self.assertEqual(batch.input_ids.shape, (32, 9))
        self.assertTrue(
            torch.equal(
                batch.input_ids[:, -1],
                batch.input_ids[rows, 2 * batch.query_context_index],
            )
        )
        self.assertTrue(
            torch.equal(
                batch.targets,
                batch.input_ids[rows, batch.matching_label_position]
                - config.num_symbols,
            )
        )

    def test_stream_is_reproducible(self) -> None:
        config = SymbolTaskConfig()
        first = SymbolTaskGenerator(config, seed=11).sample(8)
        second = SymbolTaskGenerator(config, seed=11).sample(8)
        self.assertTrue(torch.equal(first.input_ids, second.input_ids))
        self.assertTrue(torch.equal(first.targets, second.targets))


if __name__ == "__main__":
    unittest.main()

