"""Behavioral and mechanistic measurements for the baseline model."""

from __future__ import annotations

import torch

from .data import ICLBatch
from .model import InductionTransformer


def previous_token_scores(attention: torch.Tensor) -> torch.Tensor:
    """Mean label-position attention to its preceding symbol, per head."""

    sequence_length = attention.shape[-1]
    destination = torch.arange(1, sequence_length - 1, 2, device=attention.device)
    return attention[:, :, destination, destination - 1].mean(dim=(0, 2))


def induction_scores(
    attention: torch.Tensor,
    matching_label_position: torch.Tensor,
) -> torch.Tensor:
    """Mean query attention to the label after its matching context symbol."""

    final_attention = attention[:, :, -1, :]
    positions = matching_label_position[:, None, None].expand(
        -1, final_attention.shape[1], 1
    )
    return final_attention.gather(dim=2, index=positions).squeeze(2).mean(dim=0)


@torch.no_grad()
def attention_summary(
    model: InductionTransformer,
    batch: ICLBatch,
) -> dict[str, list[float] | float | int]:
    model.eval()
    logits, attention_maps = model(batch.input_ids, return_attention=True)
    first_layer = previous_token_scores(attention_maps[0])
    second_layer = induction_scores(
        attention_maps[1], batch.matching_label_position
    )
    accuracy = (logits.argmax(dim=-1) == batch.targets).float().mean()
    return {
        "batch_accuracy": float(accuracy.item()),
        "previous_token_scores": first_layer.cpu().tolist(),
        "best_previous_token_head": int(first_layer.argmax().item()),
        "induction_scores": second_layer.cpu().tolist(),
        "best_induction_head": int(second_layer.argmax().item()),
    }


@torch.no_grad()
def head_ablation_accuracy_drops(
    model: InductionTransformer,
    batch: ICLBatch,
) -> list[dict[str, float | int]]:
    """Measure the clean accuracy drop caused by ablating each head."""

    model.eval()
    baseline_logits = model(batch.input_ids)
    baseline_accuracy = (
        (baseline_logits.argmax(dim=-1) == batch.targets).float().mean().item()
    )
    results: list[dict[str, float | int]] = []
    for layer in range(model.config.num_layers):
        for head in range(model.config.num_heads):
            logits = model(batch.input_ids, ablate_heads={layer: (head,)})
            accuracy = (
                (logits.argmax(dim=-1) == batch.targets).float().mean().item()
            )
            results.append(
                {
                    "layer": layer,
                    "head": head,
                    "accuracy": accuracy,
                    "accuracy_drop": baseline_accuracy - accuracy,
                }
            )
    return results
