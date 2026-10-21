"""Causal diagnostics for frozen Omniglot base and extended checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import torch
from torch.nn import functional as F

from .omniglot_base import write_json
from .omniglot_data import OmniglotBatch
from .omniglot_model import (OmniglotExtendedTransformer,
                             OmniglotInductionTransformer, OmniglotModelConfig)
from .omniglot_base_study import study_config as base_study_config
from .omniglot_study import SEEDS, locked_settings


CONDITIONS = {"base": ("clean", "feedback"),
              "extended": ("clean", "partial", "feedback", "oracle")}


def first_score(model: OmniglotInductionTransformer, batch: OmniglotBatch,
                ablate_heads: dict[int, tuple[int, ...]] | None = None) -> dict[str, float]:
    logits = model(batch.examples, batch.context_labels, ablate_heads=ablate_heads)
    return {"accuracy": float((logits.argmax(-1) == batch.targets).float().mean()),
            "nll": float(F.cross_entropy(logits, batch.targets))}


def second_score(model: OmniglotExtendedTransformer, split: dict[str, torch.Tensor],
                 ablate_heads: dict[int, tuple[int, ...]] | None = None) -> dict[str, float]:
    _, pointer, second = model.forward_extended(
        split["examples"], split["context_labels"], split["targets"],
        split["next_indices"], ablate_heads=ablate_heads)
    return {"accuracy": float((second.argmax(-1) == split["second_labels"]).float().mean()),
            "nll": float(F.cross_entropy(second, split["second_labels"])),
            "pointer_tv": float((pointer.softmax(-1) - 0.5).abs().sum(-1).mean() / 2)}


def corrupt_query_label(batch: OmniglotBatch, num_labels: int
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    labels = batch.context_labels.clone()
    rows = torch.arange(len(batch))
    match = batch.query_context_index
    other = 1 - match
    replacement = (batch.targets + 1) % num_labels
    for _ in range(num_labels):
        collision = replacement == labels[rows, other]
        replacement = torch.where(collision, (replacement + 1) % num_labels, replacement)
    if not bool(((replacement != batch.targets) &
                 (replacement != labels[rows, other])).all()):
        raise ValueError("Could not choose an unused corruption label")
    labels[rows, match] = replacement
    return labels, replacement


@torch.no_grad()
def capture_head_inputs(model: OmniglotInductionTransformer,
                        batch: OmniglotBatch) -> list[torch.Tensor]:
    captured: list[torch.Tensor | None] = [None] * model.config.num_layers
    handles = []
    for layer, block in enumerate(model.blocks):
        def hook(_module, inputs, index=layer):
            captured[index] = inputs[0].detach().clone()
        handles.append(block.output.register_forward_pre_hook(hook))
    try:
        model(batch.examples, batch.context_labels)
    finally:
        for handle in handles:
            handle.remove()
    if any(item is None for item in captured):
        raise RuntimeError("Failed to capture all attention heads")
    return captured  # type: ignore[return-value]


@torch.no_grad()
def patched_first_logits(model: OmniglotInductionTransformer, batch: OmniglotBatch,
                         context_labels: torch.Tensor,
                         clean_head_inputs: list[torch.Tensor],
                         heads: dict[int, int]) -> torch.Tensor:
    head_dim = model.config.d_model // model.config.num_heads
    handles = []
    for layer, head in heads.items():
        if not 0 <= layer < model.config.num_layers or not 0 <= head < model.config.num_heads:
            raise ValueError("Invalid layer or head")
        source = clean_head_inputs[layer]
        positions = (1, 3) if layer == 0 else (4,)
        left, right = head * head_dim, (head + 1) * head_dim
        def hook(_module, inputs, src=source, pos=positions, start=left, end=right):
            value = inputs[0].clone()
            value[:, pos, start:end] = src[:, pos, start:end]
            return (value,)
        handles.append(model.blocks[layer].output.register_forward_pre_hook(hook))
    try:
        return model(batch.examples, context_labels)
    finally:
        for handle in handles:
            handle.remove()


def log_probability(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits, dim=-1).gather(1, targets[:, None]).squeeze(1)


@torch.no_grad()
def patching_setup(model: OmniglotInductionTransformer, batch: OmniglotBatch
                   ) -> tuple[dict, list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    corrupted, new_targets = corrupt_query_label(batch, model.config.num_labels)
    clean_logits = model(batch.examples, batch.context_labels)
    corrupt_logits = model(batch.examples, corrupted)
    effective = ((clean_logits.argmax(-1) == batch.targets) &
                 (corrupt_logits.argmax(-1) != batch.targets))
    clean_logp = log_probability(clean_logits, batch.targets)
    corrupt_logp = log_probability(corrupt_logits, batch.targets)
    gap = float((clean_logp[effective] - corrupt_logp[effective]).mean()) if bool(effective.any()) else 0.0
    metrics = {"total_examples": len(batch), "effective_examples": int(effective.sum()),
               "clean_accuracy": float((clean_logits.argmax(-1) == batch.targets).float().mean()),
               "corrupt_new_target_accuracy": float((corrupt_logits.argmax(-1) == new_targets).float().mean()),
               "clean_logprob_gap_on_effective": gap}
    source = capture_head_inputs(model, batch)
    return metrics, source, corrupted, corrupt_logp, effective


@torch.no_grad()
def patch_restoration(model: OmniglotInductionTransformer, batch: OmniglotBatch,
                      corrupted: torch.Tensor, source: list[torch.Tensor],
                      corrupt_logp: torch.Tensor, effective: torch.Tensor,
                      gap: float, heads: dict[int, int]) -> float | None:
    if not bool(effective.any()) or gap <= 0:
        return None
    logits = patched_first_logits(model, batch, corrupted, source, heads)
    patched = log_probability(logits, batch.targets)
    return float((patched[effective] - corrupt_logp[effective]).mean() / gap)


@torch.no_grad()
def inspect_checkpoint(path: Path, family: str, generation: int,
                       size: int) -> tuple[dict, list[dict]]:
    protocol = json.loads((path / "protocol.json").read_text(encoding="utf-8"))
    config = OmniglotModelConfig(**protocol["model"])
    checkpoint = torch.load(path / f"generation-{generation:03d}" / "best_model.pt",
                            weights_only=True)
    model = (OmniglotExtendedTransformer(config) if family == "extended"
             else OmniglotInductionTransformer(config))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    split = torch.load(path / "splits.pt", weights_only=True)["diagnostic"]
    split = {name: value[:size] for name, value in split.items()}
    batch = OmniglotBatch(*(split[name] for name in OmniglotBatch.__dataclass_fields__))
    baseline = first_score(model, batch)
    second_baseline = second_score(model, split) if family == "extended" else None
    _, maps = model(batch.examples, batch.context_labels, return_attention=True)
    previous = torch.stack((maps[0][:, :, 1, 0], maps[0][:, :, 3, 2]), dim=-1).mean((0, 2))
    rows = torch.arange(len(batch))
    induction = maps[1][rows, :, 4, batch.matching_label_position].mean(0)
    previous_head, induction_head = int(previous.argmax()), int(induction.argmax())
    corruption, source, corrupted, corrupt_logp, effective = patching_setup(model, batch)
    per_head = []
    for layer in range(config.num_layers):
        for head in range(config.num_heads):
            intervention = {layer: (head,)}
            ablated = first_score(model, batch, intervention)
            second_ablated = second_score(model, split, intervention) if family == "extended" else None
            per_head.append({
                "layer": layer, "head": head,
                "role_attention": float(previous[head] if layer == 0 else induction[head]),
                "first_accuracy_drop": baseline["accuracy"] - ablated["accuracy"],
                "first_nll_increase": ablated["nll"] - baseline["nll"],
                "second_accuracy_drop": (second_baseline["accuracy"] - second_ablated["accuracy"]
                                         if second_baseline is not None else None),
                "second_nll_increase": (second_ablated["nll"] - second_baseline["nll"]
                                       if second_baseline is not None else None),
                "pointer_tv_change": (second_ablated["pointer_tv"] - second_baseline["pointer_tv"]
                                      if second_baseline is not None else None),
                "patch_restoration": patch_restoration(
                    model, batch, corrupted, source, corrupt_logp, effective,
                    corruption["clean_logprob_gap_on_effective"], {layer: head}),
            })
    pair = {0: (previous_head,), 1: (induction_head,)}
    pair_ablated = first_score(model, batch, pair)
    keep_only = {layer: tuple(head for head in range(config.num_heads)
                              if head != selected) for layer, selected in
                 ((0, previous_head), (1, induction_head))}
    pair_only = first_score(model, batch, keep_only)
    no_heads = first_score(model, batch, {0: tuple(range(config.num_heads)),
                                         1: tuple(range(config.num_heads))})
    pair_patch = patch_restoration(model, batch, corrupted, source, corrupt_logp,
                                   effective, corruption["clean_logprob_gap_on_effective"],
                                   {0: previous_head, 1: induction_head})
    max_drop = {layer: max(row["first_accuracy_drop"] for row in per_head
                           if row["layer"] == layer) for layer in range(config.num_layers)}
    summary = {
        "diagnostic_size": len(batch), "baseline": baseline,
        "second_baseline": second_baseline, "corruption": corruption,
        "candidate_previous_head": previous_head,
        "candidate_induction_head": induction_head,
        "candidate_previous_attention": float(previous[previous_head]),
        "candidate_induction_attention": float(induction[induction_head]),
        "candidate_previous_is_max_ablation": float(any(
            row["layer"] == 0 and row["head"] == previous_head
            and abs(row["first_accuracy_drop"] - max_drop[0]) < 1e-9 for row in per_head)),
        "candidate_induction_is_max_ablation": float(any(
            row["layer"] == 1 and row["head"] == induction_head
            and abs(row["first_accuracy_drop"] - max_drop[1]) < 1e-9 for row in per_head)),
        "candidate_pair_accuracy_drop": baseline["accuracy"] - pair_ablated["accuracy"],
        "candidate_pair_nll_increase": pair_ablated["nll"] - baseline["nll"],
        "candidate_pair_only_accuracy": pair_only["accuracy"],
        "candidate_pair_only_nll": pair_only["nll"],
        "no_heads_accuracy": no_heads["accuracy"],
        "no_heads_nll": no_heads["nll"],
        "candidate_pair_patch_restoration": pair_patch,
        "max_single_head_patch_restoration": max(
            (row["patch_restoration"] for row in per_head
             if row["patch_restoration"] is not None), default=None),
        "max_single_head_first_accuracy_drop": max(row["first_accuracy_drop"] for row in per_head),
    }
    return summary, per_head


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    metrics = ("baseline_accuracy", "candidate_previous_attention",
               "candidate_induction_attention", "candidate_pair_accuracy_drop",
               "candidate_pair_nll_increase", "candidate_pair_only_accuracy",
               "candidate_previous_is_max_ablation", "candidate_induction_is_max_ablation",
               "no_heads_accuracy", "corrupt_new_target_accuracy",
               "candidate_pair_patch_restoration", "max_single_head_patch_restoration",
               "max_single_head_first_accuracy_drop", "effective_examples")
    result = []
    generations = max(row["generation"] for row in rows) + 1
    for family, conditions in CONDITIONS.items():
        for condition in conditions:
            for generation in range(generations):
                selected = [row for row in rows if row["family"] == family
                            and row["condition"] == condition and row["generation"] == generation]
                if len(selected) != len(SEEDS):
                    raise ValueError(f"Missing mechanism seeds: {family}/{condition}/G{generation}")
                summary = {"family": family, "condition": condition,
                           "generation": generation, "n_seeds": len(selected)}
                for key in metrics:
                    values = [row[key] for row in selected if row[key] is not None]
                    summary[f"{key}_mean"] = mean(values) if values else None
                    summary[f"{key}_sd"] = stdev(values) if len(values) > 1 else None
                result.append(summary)
    return result


def run(base_runs: Path, extended_runs: Path, output: Path, size: int) -> None:
    if not 1 <= size <= 256:
        raise ValueError("Diagnostic size must be between 1 and 256")
    if output.exists():
        raise FileExistsError(output)
    generations = locked_settings(extended_runs)["generations"]
    if locked_settings(base_runs, base_study_config)["generations"] != generations:
        raise ValueError("Base and extended studies have different generation counts")
    summaries, heads = [], []
    for family, root in (("base", base_runs), ("extended", extended_runs)):
        for condition in CONDITIONS[family]:
            for seed in SEEDS:
                path = root / f"seed-{seed:02d}" / condition
                for generation in range(generations):
                    summary, head_rows = inspect_checkpoint(path, family, generation, size)
                    prefix = {"family": family, "condition": condition,
                              "seed": seed, "generation": generation}
                    summaries.append({**prefix, "baseline_accuracy": summary["baseline"]["accuracy"],
                                      **{key: value for key, value in summary.items()
                                         if key not in ("baseline", "second_baseline", "corruption")},
                                      "baseline_nll": summary["baseline"]["nll"],
                                      "second_accuracy": (summary["second_baseline"]["accuracy"]
                                                          if summary["second_baseline"] else None),
                                      "second_nll": (summary["second_baseline"]["nll"]
                                                     if summary["second_baseline"] else None),
                                      "effective_examples": summary["corruption"]["effective_examples"],
                                      "corrupt_new_target_accuracy":
                                      summary["corruption"]["corrupt_new_target_accuracy"]})
                    heads.extend({**prefix, **row} for row in head_rows)
                    print(f"Inspected {family}/{condition}/seed-{seed:02d}/G{generation}", flush=True)
    aggregates = aggregate(summaries)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "checkpoints.csv", summaries)
    write_csv(output / "heads.csv", heads)
    write_csv(output / "aggregate.csv", aggregates)
    write_json(output / "protocol.json", {"base_runs": str(base_runs),
                                          "extended_runs": str(extended_runs),
                                          "diagnostic_size": size,
                                          "selection": "top role-attention head within each checkpoint",
                                          "corruption": "replace only matching context label with unused label",
                                          "patch": "clean attention-head output into corrupted prompt"})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 4.1))
    for family, conditions in CONDITIONS.items():
        for condition in conditions:
            selected = [row for row in aggregates if row["family"] == family
                        and row["condition"] == condition]
            label = f"{family} {condition}"
            axes[0].plot(range(generations), [row["candidate_pair_accuracy_drop_mean"] for row in selected],
                         marker="o", label=label)
            axes[1].plot(range(generations), [row["candidate_pair_patch_restoration_mean"] for row in selected],
                         marker="o", label=label)
    axes[0].set(title="Candidate-pair necessity", ylabel="Diagnostic accuracy drop")
    axes[1].set(title="Candidate-pair patching", ylabel="Normalized log-probability restoration")
    for axis in axes:
        axis.set_xlabel("Generation")
        axis.set_xticks(range(generations))
        axis.axhline(0, color="0.5", linewidth=0.8)
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
                  fontsize=8, bbox_to_anchor=(0.5, 0.015))
    figure.subplots_adjust(bottom=0.26, wspace=0.3)
    figure.savefig(output / "mechanism_comparison.png", dpi=180)
    figure.savefig(output / "mechanism_comparison.pdf")
    plt.close(figure)
    print(f"Completed Omniglot mechanistic study: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-runs", type=Path, required=True)
    parser.add_argument("--extended-runs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    run(args.base_runs, args.extended_runs, args.output_dir, args.size)


if __name__ == "__main__":
    main()
