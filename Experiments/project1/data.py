"""Synthetic symbol-label task that requires in-context learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SymbolTaskConfig:
    """Configuration for randomly relabelled symbol sequences."""

    num_symbols: int = 50
    num_labels: int = 5
    context_pairs: int = 2

    def __post_init__(self) -> None:
        if self.num_symbols < self.context_pairs:
            raise ValueError("num_symbols must be at least context_pairs")
        if self.num_labels < self.context_pairs:
            raise ValueError("num_labels must be at least context_pairs")
        if self.context_pairs < 2:
            raise ValueError("context_pairs must be at least 2")

    @property
    def vocab_size(self) -> int:
        return self.num_symbols + self.num_labels

    @property
    def sequence_length(self) -> int:
        return 2 * self.context_pairs + 1


@dataclass(frozen=True)
class ICLBatch:
    """A batch plus the metadata needed for mechanistic measurements."""

    input_ids: torch.Tensor
    targets: torch.Tensor
    query_context_index: torch.Tensor
    matching_label_position: torch.Tensor

    def to(self, device: torch.device | str) -> "ICLBatch":
        return ICLBatch(
            input_ids=self.input_ids.to(device),
            targets=self.targets.to(device),
            query_context_index=self.query_context_index.to(device),
            matching_label_position=self.matching_label_position.to(device),
        )


class SymbolTaskGenerator:
    """Samples arbitrary symbol-label mappings independently per sequence.

    A sequence has the form
    ``s0 l0 s1 l1 ... sk lk query_symbol``. Symbols and labels are unique
    within the context, and the query repeats one context symbol. Label token
    IDs are offset by ``num_symbols`` while targets use zero-based label IDs.
    """

    def __init__(self, config: SymbolTaskConfig, seed: int) -> None:
        self.config = config
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def sample(self, batch_size: int) -> ICLBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        cfg = self.config
        symbol_scores = torch.rand(
            batch_size, cfg.num_symbols, generator=self.generator
        )
        symbols = symbol_scores.topk(cfg.context_pairs, dim=1).indices

        label_scores = torch.rand(
            batch_size, cfg.num_labels, generator=self.generator
        )
        labels = label_scores.topk(cfg.context_pairs, dim=1).indices

        query_context_index = torch.randint(
            cfg.context_pairs,
            (batch_size,),
            generator=self.generator,
        )
        batch_indices = torch.arange(batch_size)
        query_symbols = symbols[batch_indices, query_context_index]
        targets = labels[batch_indices, query_context_index]

        input_ids = torch.empty(
            batch_size, cfg.sequence_length, dtype=torch.long
        )
        input_ids[:, 0:-1:2] = symbols
        input_ids[:, 1:-1:2] = labels + cfg.num_symbols
        input_ids[:, -1] = query_symbols

        return ICLBatch(
            input_ids=input_ids,
            targets=targets,
            query_context_index=query_context_index,
            matching_label_position=2 * query_context_index + 1,
        )
