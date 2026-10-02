import unittest

import torch

from project1.data import SymbolTaskConfig, SymbolTaskGenerator
from project1.diagnostics import induction_scores, previous_token_scores
from project1.model import InductionTransformer, TransformerConfig


class InductionTransformerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = SymbolTaskConfig(num_symbols=12, num_labels=4, context_pairs=4)
        self.config = TransformerConfig(
            vocab_size=self.task.vocab_size,
            num_labels=self.task.num_labels,
            max_sequence_length=self.task.sequence_length,
            d_model=16,
            num_layers=2,
            num_heads=4,
            mlp_ratio=2,
        )

    def test_forward_shapes_and_causal_attention(self) -> None:
        model = InductionTransformer(self.config)
        batch = SymbolTaskGenerator(self.task, seed=3).sample(5)
        logits, attention = model(batch.input_ids, return_attention=True)

        self.assertEqual(logits.shape, (5, 4))
        self.assertEqual(len(attention), 2)
        self.assertEqual(attention[0].shape, (5, 4, 9, 9))
        upper_triangle = torch.triu(attention[0], diagonal=1)
        self.assertTrue(torch.allclose(upper_triangle, torch.zeros_like(upper_triangle)))

    def test_diagnostic_scores_are_per_head(self) -> None:
        model = InductionTransformer(self.config)
        batch = SymbolTaskGenerator(self.task, seed=5).sample(6)
        _, attention = model(batch.input_ids, return_attention=True)
        self.assertEqual(previous_token_scores(attention[0]).shape, (4,))
        self.assertEqual(
            induction_scores(attention[1], batch.matching_label_position).shape,
            (4,),
        )

    def test_previous_token_score_uses_label_destinations(self) -> None:
        attention = torch.zeros(2, 3, 5, 5)
        attention[:, :, 1, 0] = 0.25
        attention[:, :, 3, 2] = 0.75
        attention[:, :, 2, 1] = 1.0
        expected = torch.full((3,), 0.5)
        self.assertTrue(torch.allclose(previous_token_scores(attention), expected))

    def test_head_ablation_preserves_output_shape(self) -> None:
        model = InductionTransformer(self.config)
        batch = SymbolTaskGenerator(self.task, seed=9).sample(3)
        logits = model(batch.input_ids, ablate_heads={0: (1,), 1: (2,)})
        self.assertEqual(logits.shape, (3, 4))


if __name__ == "__main__":
    unittest.main()
