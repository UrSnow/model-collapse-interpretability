"""Omniglot feature episodes matching the two-way Singh et al. ICL task."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch


FEATURE_DATASET = "resnet18/224/feat"
FEATURE_SHA256 = "a8b8fa1f25d62e077b658c590f77e24447894fa325d97651dba22c6621e62b5b"


def load_features(path: Path) -> torch.Tensor:
    import h5py

    if not path.is_file():
        raise FileNotFoundError(f"Omniglot features not found: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != FEATURE_SHA256:
        raise ValueError(f"Omniglot feature checksum mismatch: {digest}")
    with h5py.File(path, "r") as source:
        features = torch.from_numpy(source[FEATURE_DATASET][:])
    if features.shape != (1623, 5, 512) or not bool(torch.isfinite(features).all()):
        raise ValueError(f"Unexpected Omniglot feature tensor: {tuple(features.shape)}")
    return features


@dataclass(frozen=True)
class OmniglotTaskConfig:
    train_classes: int = 50
    test_classes: int = 100
    num_labels: int = 5
    train_label_pairs: int = 8
    context_pairs: int = 2
    relabel_seed: int = 20

    def __post_init__(self) -> None:
        if self.train_classes < 2 or self.test_classes < 2:
            raise ValueError("Each class pool needs at least two classes")
        if self.num_labels < 2 or self.context_pairs != 2:
            raise ValueError("The Singh baseline is two-way with two context examples")
        pair_count = self.num_labels * (self.num_labels - 1) // 2
        if not 0 < self.train_label_pairs < pair_count:
            raise ValueError("Training and validation both need relabeling pairs")


@dataclass(frozen=True)
class OmniglotBatch:
    examples: torch.Tensor
    context_labels: torch.Tensor
    targets: torch.Tensor
    class_ids: torch.Tensor
    query_context_index: torch.Tensor
    matching_label_position: torch.Tensor

    def __len__(self) -> int:
        return len(self.targets)

    def subset(self, indices: torch.Tensor | slice) -> "OmniglotBatch":
        return OmniglotBatch(*(getattr(self, name)[indices] for name in self.__dataclass_fields__))


class OmniglotTaskGenerator:
    """Sample query/distractor classes, then independently relabel each episode."""

    def __init__(self, features: torch.Tensor, config: OmniglotTaskConfig, seed: int) -> None:
        if features.ndim != 3 or features.shape[1] < 1:
            raise ValueError("features must have shape [class, exemplar, dimension]")
        if features.shape[0] < config.train_classes + config.test_classes + 2:
            raise ValueError("No disjoint class pool remains for diagnostics")
        self.features = features
        self.config = config
        self.rng = torch.Generator().manual_seed(seed)
        pairs = torch.combinations(torch.arange(config.num_labels), r=2)
        order = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(config.relabel_seed))
        self.train_pairs = pairs[order[:config.train_label_pairs]]
        self.validation_pairs = pairs[order[config.train_label_pairs:]]

    def sample(self, size: int, split: str) -> OmniglotBatch:
        if size < 1:
            raise ValueError("size must be positive")
        cfg = self.config
        total = self.features.shape[0]
        pools = {
            "train": (0, cfg.train_classes, self.train_pairs),
            "validation": (0, cfg.train_classes, self.validation_pairs),
            "diagnostic": (cfg.train_classes, total - cfg.test_classes, self.train_pairs),
            "test": (total - cfg.test_classes, total, self.train_pairs),
        }
        if split not in pools:
            raise ValueError(f"Unknown split: {split}")
        start, stop, pairs = pools[split]
        width = stop - start
        query = torch.randint(width, (size,), generator=self.rng)
        other = torch.randint(width - 1, (size,), generator=self.rng)
        other += (other >= query).long()
        query += start
        other += start
        query_first = torch.randint(2, (size,), generator=self.rng).bool()
        context = torch.stack((torch.where(query_first, query, other),
                               torch.where(query_first, other, query)), dim=1)
        choices = torch.randint(len(pairs), (size,), generator=self.rng)
        chosen = pairs[choices]
        flip = torch.randint(2, (size,), generator=self.rng).bool()
        query_label = torch.where(flip, chosen[:, 1], chosen[:, 0])
        other_label = torch.where(flip, chosen[:, 0], chosen[:, 1])
        labels = torch.stack((torch.where(query_first, query_label, other_label),
                              torch.where(query_first, other_label, query_label)), dim=1)
        class_ids = torch.cat((context, query[:, None]), dim=1)
        query_index = torch.where(query_first, 0, 1)
        return OmniglotBatch(
            examples=self.features[class_ids, 0],
            context_labels=labels,
            targets=query_label,
            class_ids=class_ids,
            query_context_index=query_index,
            matching_label_position=2 * query_index + 1,
        )

    def sample_unique(self, size: int, split: str,
                      exclude: set[tuple[int, ...]] | None = None) -> OmniglotBatch:
        rows: list[OmniglotBatch] = []
        seen: set[tuple[int, ...]] = set()
        excluded = exclude if exclude is not None else set()
        for _ in range(max(100, size // 8)):
            batch = self.sample(128, split)
            for index in range(len(batch)):
                key = (*batch.class_ids[index].tolist(), *batch.context_labels[index].tolist())
                if key in seen or key in excluded:
                    continue
                seen.add(key)
                rows.append(batch.subset(torch.tensor([index])))
                if len(rows) == size:
                    return OmniglotBatch(*(torch.cat([getattr(row, name) for row in rows])
                                           for name in OmniglotBatch.__dataclass_fields__))
        raise ValueError(f"Could not sample {size} unique {split} episodes")
