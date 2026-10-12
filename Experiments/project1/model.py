"""Small causal transformer with inspectable attention heads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Collection

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    num_labels: int
    max_sequence_length: int
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.0
    output_vocab_size: int | None = None

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.num_layers < 2:
            raise ValueError("induction experiments require at least two layers")
        if self.output_vocab_size is not None and self.output_vocab_size < self.num_labels:
            raise ValueError("output_vocab_size must include all labels")

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        mask = torch.tril(
            torch.ones(
                config.max_sequence_length,
                config.max_sequence_length,
                dtype=torch.bool,
            )
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        ablate_heads: Collection[int] = (),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, d_model = x.shape
        qkv = self.qkv(x).view(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        scores = query @ key.transpose(-2, -1) / self.head_dim**0.5
        scores = scores.masked_fill(
            ~self.causal_mask[:sequence_length, :sequence_length],
            torch.finfo(scores.dtype).min,
        )
        attention = F.softmax(scores, dim=-1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        head_outputs = attention @ value

        if ablate_heads:
            head_mask = torch.ones(
                self.num_heads, device=x.device, dtype=x.dtype
            )
            head_mask[list(ablate_heads)] = 0
            head_outputs = head_outputs * head_mask[None, :, None, None]

        merged = head_outputs.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, d_model
        )
        return self.output(merged), attention


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        hidden_size = config.mlp_ratio * config.d_model
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        ablate_heads: Collection[int] = (),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended, attention = self.attention(self.norm1(x), ablate_heads)
        x = x + attended
        x = x + self.mlp(self.norm2(x))
        return x, attention


class InductionTransformer(nn.Module):
    """Predicts the query label and optionally returns per-head attention."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(
            config.max_sequence_length, config.d_model
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.unembed = nn.Linear(config.d_model, config.output_vocab_size or config.num_labels)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        return_attention: bool = False,
        all_positions: bool = False,
        ablate_heads: dict[int, Collection[int]] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        sequence_length = input_ids.shape[1]
        if sequence_length > self.config.max_sequence_length:
            raise ValueError("input sequence exceeds configured maximum")

        positions = torch.arange(sequence_length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        attention_maps: list[torch.Tensor] = []
        ablate_heads = ablate_heads or {}

        for layer_index, block in enumerate(self.blocks):
            x, attention = block(x, ablate_heads.get(layer_index, ()))
            if return_attention:
                attention_maps.append(attention)

        logits = self.unembed(self.final_norm(x if all_positions else x[:, -1]))
        if return_attention:
            return logits, tuple(attention_maps)
        return logits
