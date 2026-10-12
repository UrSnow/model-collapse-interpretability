"""Attention-only causal model for interleaved Omniglot features and labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class OmniglotModelConfig:
    feature_dim: int = 512
    num_labels: int = 5
    d_model: int = 64
    num_heads: int = 8
    num_layers: int = 2

    def __post_init__(self) -> None:
        if self.num_layers != 2 or self.d_model % self.num_heads:
            raise ValueError("Expected two layers and evenly divided attention heads")
        if (self.d_model // self.num_heads) % 2:
            raise ValueError("RoPE requires an even head dimension")


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    length, head_dim = x.shape[-2:]
    positions = torch.arange(length, device=x.device, dtype=x.dtype)
    frequencies = 10000 ** (-torch.arange(0, head_dim, 2, device=x.device,
                                          dtype=x.dtype) / head_dim)
    angles = positions[:, None] * frequencies[None, :]
    cosine = torch.repeat_interleave(angles.cos(), 2, dim=-1)[None, None]
    sine = torch.repeat_interleave(angles.sin(), 2, dim=-1)[None, None]
    pairs = x.reshape(*x.shape[:-1], head_dim // 2, 2)
    rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
    return x * cosine + rotated * sine


class AttentionOnlyBlock(nn.Module):
    def __init__(self, config: OmniglotModelConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads

    def forward(self, x: torch.Tensor, ablate_heads: Collection[int] = ()) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, width = x.shape
        qkv = self.qkv(self.norm(x)).reshape(batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = (part.transpose(1, 2) for part in qkv.unbind(2))
        query, key = apply_rope(query), apply_rope(key)
        scores = query @ key.transpose(-2, -1) / self.head_dim**0.5
        mask = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
        attention = F.softmax(scores.masked_fill(mask, torch.finfo(scores.dtype).min), dim=-1)
        heads = attention @ value
        if ablate_heads:
            head_mask = torch.ones(self.num_heads, device=x.device, dtype=x.dtype)
            head_mask[list(ablate_heads)] = 0
            heads = heads * head_mask[None, :, None, None]
        merged = heads.transpose(1, 2).contiguous().reshape(batch, length, width)
        return x + self.output(merged), attention


class OmniglotInductionTransformer(nn.Module):
    def __init__(self, config: OmniglotModelConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_projection = nn.Linear(config.feature_dim, config.d_model)
        self.label_embedding = nn.Embedding(config.num_labels, config.d_model)
        self.blocks = nn.ModuleList(AttentionOnlyBlock(config) for _ in range(config.num_layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.unembed = nn.Linear(config.d_model, config.num_labels)
        nn.init.normal_(self.label_embedding.weight, std=0.02)

    def forward(self, examples: torch.Tensor, context_labels: torch.Tensor, *,
                return_attention: bool = False,
                ablate_heads: dict[int, Collection[int]] | None = None
                ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if examples.ndim != 3 or examples.shape[1] != 3 or examples.shape[2] != self.config.feature_dim:
            raise ValueError("Expected two context features and one query feature")
        if context_labels.shape != (examples.shape[0], 2):
            raise ValueError("Expected two context labels per episode")
        feature_tokens = self.feature_projection(examples)
        label_tokens = self.label_embedding(context_labels)
        context = torch.stack((feature_tokens[:, :2], label_tokens), dim=2).flatten(1, 2)
        x = torch.cat((context, feature_tokens[:, 2:]), dim=1)
        maps = []
        for layer, block in enumerate(self.blocks):
            x, attention = block(x, (ablate_heads or {}).get(layer, ()))
            if return_attention:
                maps.append(attention)
        logits = self.unembed(self.final_norm(x[:, -1]))
        if return_attention:
            return logits, tuple(maps)
        return logits


class OmniglotExtendedTransformer(OmniglotInductionTransformer):
    """Predict a label, a context-feature pointer, and its corresponding label."""

    def __init__(self, config: OmniglotModelConfig) -> None:
        super().__init__(config)
        self.pointer_head = nn.Linear(config.d_model, 2)

    def forward_extended(self, examples: torch.Tensor, context_labels: torch.Tensor,
                         first_labels: torch.Tensor, next_indices: torch.Tensor, *,
                         return_attention: bool = False,
                         ablate_heads: dict[int, Collection[int]] | None = None
                         ) -> tuple[torch.Tensor, ...]:
        if examples.ndim != 3 or examples.shape[1:] != (3, self.config.feature_dim):
            raise ValueError("Expected two context features and one query feature")
        batch = examples.shape[0]
        if context_labels.shape != (batch, 2) or first_labels.shape != (batch,) or next_indices.shape != (batch,):
            raise ValueError("Invalid extended episode shape")
        if not bool(((next_indices >= 0) & (next_indices < 2)).all()):
            raise ValueError("The next feature must be one of the context features")
        projected = self.feature_projection(examples)
        labels = self.label_embedding(context_labels)
        context = torch.stack((projected[:, :2], labels), dim=2).flatten(1, 2)
        row = torch.arange(batch, device=examples.device)
        x = torch.cat((context, projected[:, 2:3],
                       self.label_embedding(first_labels)[:, None],
                       projected[row, next_indices][:, None]), dim=1)
        maps = []
        for layer, block in enumerate(self.blocks):
            x, attention = block(x, (ablate_heads or {}).get(layer, ()))
            if return_attention:
                maps.append(attention)
        hidden = self.final_norm(x)
        result = (self.unembed(hidden[:, 4]), self.pointer_head(hidden[:, 5]),
                  self.unembed(hidden[:, 6]))
        if return_attention:
            return (*result, tuple(maps))
        return result

    def pointer_logits(self, examples: torch.Tensor, context_labels: torch.Tensor,
                       first_labels: torch.Tensor) -> torch.Tensor:
        batch = examples.shape[0]
        projected = self.feature_projection(examples)
        labels = self.label_embedding(context_labels)
        context = torch.stack((projected[:, :2], labels), dim=2).flatten(1, 2)
        x = torch.cat((context, projected[:, 2:3],
                       self.label_embedding(first_labels)[:, None]), dim=1)
        for block in self.blocks:
            x, _ = block(x)
        return self.pointer_head(self.final_norm(x[:, -1]))
