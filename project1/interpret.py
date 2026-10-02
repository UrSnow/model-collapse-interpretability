"""Mechanistic diagnostics for saved base and extended checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from .data import SymbolTaskConfig
from .diagnostics import induction_scores, previous_token_scores
from .extended import context_symbol_logits, sequence_logits
from .model import InductionTransformer, TransformerConfig


COLORS = {"clean": "#30343b", "mixed": "#168a82",
          "synthetic": "#d05b39", "resampled": "#a05b99",
          "resampled_sampled": "#3d75af"}
LABELS = {"clean": "Clean", "mixed": "90% feedback", "synthetic": "100% feedback",
          "resampled": "Oracle (soft)", "resampled_sampled": "Oracle (sampled)"}


def label_logits(model: InductionTransformer, prompt: torch.Tensor,
                 task: SymbolTaskConfig, extended: bool,
                 ablate_heads: dict[int, tuple[int, ...]] | None = None) -> torch.Tensor:
    logits = model(prompt, ablate_heads=ablate_heads)
    return logits[:, task.num_symbols:] if extended else logits


def score(model: InductionTransformer, prompt: torch.Tensor, targets: torch.Tensor,
          task: SymbolTaskConfig, extended: bool,
          ablate_heads: dict[int, tuple[int, ...]] | None = None) -> dict[str, float]:
    logits = label_logits(model, prompt, task, extended, ablate_heads)
    return {"loss": float(F.cross_entropy(logits, targets).item()),
            "accuracy": float((logits.argmax(-1) == targets).float().mean().item())}


def symbol_score(model: InductionTransformer, split: dict[str, torch.Tensor],
                 task: SymbolTaskConfig,
                 ablate_heads: dict[int, tuple[int, ...]] | None = None) -> dict[str, float]:
    _, logits, _ = sequence_logits(model, split["input_ids"], split["first_labels"],
                                    split["next_symbols"], task, ablate_heads)
    probabilities = context_symbol_logits(logits, split["input_ids"], task).softmax(-1)
    tv = (probabilities - 1 / task.context_pairs).abs().sum(-1).mean() / 2
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1).mean()
    return {"tv": float(tv), "entropy": float(entropy)}


def representation_stats(model: InductionTransformer, task: SymbolTaskConfig) -> dict[str, float]:
    embedding = F.normalize(model.token_embedding.weight.detach(), dim=-1)
    result = {}
    for name, tokens in (("symbol", embedding[:task.num_symbols]),
                         ("label", embedding[task.num_symbols:])):
        similarity = tokens @ tokens.T
        off_diagonal = ~torch.eye(len(tokens), dtype=torch.bool)
        result[f"{name}_mean_abs_cosine"] = float(similarity[off_diagonal].abs().mean())
        centered = tokens - tokens.mean(dim=0)
        singular = torch.linalg.svdvals(centered)
        probabilities = singular.square() / singular.square().sum().clamp_min(1e-12)
        positive = probabilities > 0
        result[f"{name}_effective_rank"] = float(torch.exp(
            -(probabilities[positive] * probabilities[positive].log()).sum()))
    return result


def corrupt_matching_label(prompt: torch.Tensor, match_positions: torch.Tensor,
                           task: SymbolTaskConfig) -> torch.Tensor:
    corrupted = prompt.clone()
    row = torch.arange(len(prompt))
    used = prompt[:, 1:2 * task.context_pairs:2] - task.num_symbols
    original = prompt[row, match_positions] - task.num_symbols
    replacement = (original + 1) % task.num_labels
    for _ in range(task.num_labels):
        collision = (used == replacement[:, None]).any(dim=1)
        replacement = torch.where(collision, (replacement + 1) % task.num_labels,
                                  replacement)
    if bool((used == replacement[:, None]).any()):
        raise ValueError("No unused corruption label available")
    corrupted[row, match_positions] = replacement + task.num_symbols
    return corrupted


def capture_head_inputs(model: InductionTransformer, prompt: torch.Tensor) -> list[torch.Tensor]:
    captured = [None] * model.config.num_layers
    handles = []
    for layer, block in enumerate(model.blocks):
        def hook(_module, inputs, index=layer):
            captured[index] = inputs[0].detach().clone()
        handles.append(block.attention.output.register_forward_pre_hook(hook))
    try:
        model(prompt)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def patched_logits(model: InductionTransformer, prompt: torch.Tensor,
                   clean_head_inputs: list[torch.Tensor],
                   heads: dict[int, int], task: SymbolTaskConfig,
                   extended: bool) -> torch.Tensor:
    head_dim = model.config.d_model // model.config.num_heads
    handles = []
    for layer, head in heads.items():
        source = clean_head_inputs[layer]
        positions = slice(1, task.sequence_length - 1, 2) if layer == 0 else -1
        left, right = head * head_dim, (head + 1) * head_dim
        def hook(_module, inputs, src=source, pos=positions, start=left, end=right):
            value = inputs[0].clone()
            value[:, pos, start:end] = src[:, pos, start:end]
            return (value,)
        handles.append(model.blocks[layer].attention.output.register_forward_pre_hook(hook))
    try:
        return label_logits(model, prompt, task, extended)
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def patching_scores(model: InductionTransformer, prompt: torch.Tensor,
                    targets: torch.Tensor, match_positions: torch.Tensor,
                    task: SymbolTaskConfig, extended: bool,
                    previous_head: int, induction_head: int) -> dict:
    corrupted = corrupt_matching_label(prompt, match_positions, task)
    clean_logits = label_logits(model, prompt, task, extended)
    corrupt_logits = label_logits(model, corrupted, task, extended)
    effective = ((clean_logits.argmax(-1) == targets) &
                 (corrupt_logits.argmax(-1) != targets))
    count = int(effective.sum())
    result = {"effective_examples": count, "total_examples": len(prompt)}
    if count == 0:
        return result
    clean_logp = F.log_softmax(clean_logits[effective], dim=-1).gather(
        1, targets[effective, None]).squeeze(1)
    corrupt_logp = F.log_softmax(corrupt_logits[effective], dim=-1).gather(
        1, targets[effective, None]).squeeze(1)
    gap = (clean_logp - corrupt_logp).mean().item()
    source = capture_head_inputs(model, prompt)
    for name, heads in (("previous", {0: previous_head}),
                        ("induction", {1: induction_head}),
                        ("group", {0: previous_head, 1: induction_head})):
        logits = patched_logits(model, corrupted, source, heads, task, extended)
        patched_logp = F.log_softmax(logits[effective], dim=-1).gather(
            1, targets[effective, None]).squeeze(1)
        result[f"{name}_logprob_restoration"] = float(
            ((patched_logp - corrupt_logp).mean() / gap).item())
        result[f"{name}_accuracy"] = float((logits[effective].argmax(-1)
                                            == targets[effective]).float().mean().item())
    return result


@torch.no_grad()
def inspect_checkpoint(folder: Path, extended: bool, limit: int) -> dict:
    checkpoint = torch.load(folder / "best_model.pt", weights_only=True)
    model = InductionTransformer(TransformerConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    task = SymbolTaskConfig(**checkpoint["task_config"])
    split = torch.load(folder.parent / "splits.pt", weights_only=True)["diagnostic"]
    split = {key: value[:limit] for key, value in split.items()}
    prompt = split["input_ids"]
    targets = split["first_labels"] if extended else split["targets"]
    match_positions = split["matching_label_position"]
    logits, maps = model(prompt, return_attention=True)
    previous = previous_token_scores(maps[0])
    induction = induction_scores(maps[1], match_positions)
    best_previous = int(previous.argmax())
    best_induction = int(induction.argmax())
    baseline = score(model, prompt, targets, task, extended)
    ablations = []
    for layer in range(model.config.num_layers):
        for head in range(model.config.num_heads):
            value = score(model, prompt, targets, task, extended, {layer: (head,)})
            ablations.append({"layer": layer, "head": head,
                              "loss_increase": value["loss"] - baseline["loss"],
                              "accuracy_drop": baseline["accuracy"] - value["accuracy"]})
    group = score(model, prompt, targets, task, extended,
                  {0: (best_previous,), 1: (best_induction,)})
    all_heads = tuple(range(model.config.num_heads))
    selected_only = score(model, prompt, targets, task, extended,
                          {0: tuple(head for head in all_heads if head != best_previous),
                           1: tuple(head for head in all_heads if head != best_induction)})
    layer_ablations = {}
    for name, heads in (("first_layer", {0: all_heads}),
                        ("second_layer", {1: all_heads}),
                        ("both_layers", {0: all_heads, 1: all_heads})):
        value = score(model, prompt, targets, task, extended, heads)
        layer_ablations[name] = {
            "loss_increase": value["loss"] - baseline["loss"],
            "accuracy_drop": baseline["accuracy"] - value["accuracy"]}
    result = {
        "baseline": baseline, "previous_token_scores": previous.tolist(),
        "induction_scores": induction.tolist(),
        "best_previous_head": best_previous, "best_induction_head": best_induction,
        "ablations": ablations,
        "layer_ablations": layer_ablations,
        "selected_group_ablation": {
            "loss_increase": group["loss"] - baseline["loss"],
            "accuracy_drop": baseline["accuracy"] - group["accuracy"]},
        "selected_heads_only": selected_only,
        "patching": patching_scores(model, prompt, targets, match_positions,
                                    task, extended, best_previous, best_induction),
        "representations": representation_stats(model, task),
    }
    if extended:
        baseline_symbol = symbol_score(model, split, task)
        result["symbol_mechanism"] = {
            "baseline": baseline_symbol,
            "best_induction_head_ablated": symbol_score(
                model, split, task, {1: (best_induction,)}),
            "second_layer_ablated": symbol_score(model, split, task, {1: all_heads}),
            "per_head": [{"layer": layer, "head": head,
                          **symbol_score(model, split, task, {layer: (head,)})}
                         for layer in range(model.config.num_layers)
                         for head in range(model.config.num_heads)],
        }
    return result


def plot_records(records: list[dict], output_dir: Path, family: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    metrics = (("induction", "Max induction attention score"),
               ("group_ablation", "Selected-head group accuracy drop"),
               ("layer_ablation", "Whole second-layer accuracy drop"),
               ("patching", "Selected-head patch log-prob restoration"))
    for ax, (metric, title) in zip(axes.flat, metrics):
        for condition in COLORS:
            subset = [row for row in records if row["condition"] == condition]
            if not subset:
                continue
            generations = sorted({row["generation"] for row in subset})
            def value(row):
                if metric == "induction":
                    return max(row["induction_scores"])
                if metric == "group_ablation":
                    return row["selected_group_ablation"]["accuracy_drop"]
                if metric == "layer_ablation":
                    return row["layer_ablations"]["second_layer"]["accuracy_drop"]
                return row["patching"].get("group_logprob_restoration")
            groups = [[value(row) for row in subset if row["generation"] == generation
                       and value(row) is not None] for generation in generations]
            means = [statistics.mean(group) if group else float("nan") for group in groups]
            sds = [statistics.stdev(group) if len(group) > 1 else 0 for group in groups]
            ax.errorbar(generations, means, yerr=sds, marker="o", capsize=3,
                        color=COLORS[condition], linewidth=1.8,
                        label=LABELS.get(condition, condition))
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_xticks(sorted({row["generation"] for row in records}))
        ax.grid(alpha=0.2)
        if metric != "patching":
            ax.set_ylim(bottom=0)
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.suptitle(f"{family.capitalize()} mechanistic diagnostics | diagnostic prompts")
    fig.savefig(output_dir / "mechanistic_comparison.png", dpi=180)
    fig.savefig(output_dir / "mechanistic_comparison.pdf")
    plt.close(fig)
    if family == "extended":
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, (key, title) in zip(axes, (("baseline", "Original symbol TV"),
                                           ("second_layer_ablated", "Symbol TV after second-layer ablation"))):
            for condition in COLORS:
                subset = [row for row in records if row["condition"] == condition]
                generations = sorted({row["generation"] for row in subset})
                groups = [[row["symbol_mechanism"][key]["tv"] for row in subset
                           if row["generation"] == generation] for generation in generations]
                axes_value = [statistics.mean(group) for group in groups]
                axes_sd = [statistics.stdev(group) if len(group) > 1 else 0
                           for group in groups]
                ax.errorbar(generations, axes_value, yerr=axes_sd, marker="o", capsize=3,
                            color=COLORS[condition],
                            label=LABELS.get(condition, condition))
            ax.set_title(title)
            ax.set_xlabel("Generation")
            ax.set_xticks(sorted({row["generation"] for row in records}))
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.2)
        axes[0].legend(frameon=False, ncol=2)
        fig.suptitle("Conditional next-symbol distribution under ablation")
        fig.savefig(output_dir / "symbol_ablation.png", dpi=180)
        fig.savefig(output_dir / "symbol_ablation.pdf")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("base", "extended"), required=True)
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, default=128)
    args = parser.parse_args()
    runs_dir = args.runs_dir or Path("runs/extended-study" if args.family == "extended" else "runs")
    output_dir = args.output_dir or Path(f"figures/interpret-{args.family}")
    if args.family == "base":
        runs = [(0, name, runs_dir / f"step2-base-{name}")
                for name in ("clean", "mixed", "synthetic")]
    else:
        runs = [(int(seed_dir.name[5:]), condition, seed_dir / condition)
                for seed_dir in sorted(runs_dir.glob("seed-*"))
                for condition in ("clean", "mixed", "synthetic", "resampled",
                                  "resampled_sampled")]
    records = []
    for seed, condition, folder in runs:
        summaries = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        for generation in range(len(summaries)):
            result = inspect_checkpoint(folder / f"generation-{generation:03d}",
                                        args.family == "extended", args.limit)
            records.append({"seed": seed, "condition": condition,
                            "generation": generation, **result})
            print(f"Inspected {args.family} seed {seed}, {condition}, gen {generation}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "details.json").write_text(json.dumps(records, indent=2,
                                                        allow_nan=False), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ("seed", "condition", "generation", "baseline_accuracy", "baseline_loss",
                  "max_induction_score", "group_ablation_accuracy_drop",
                  "group_ablation_loss_increase", "patch_effective_examples",
                  "group_patch_restoration", "second_layer_accuracy_drop",
                  "both_layers_accuracy_drop", "selected_heads_only_accuracy",
                  "selected_heads_only_loss", "symbol_effective_rank",
                  "symbol_tv", "symbol_tv_after_second_layer_ablation")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({"seed": row["seed"], "condition": row["condition"],
                             "generation": row["generation"],
                             "baseline_accuracy": row["baseline"]["accuracy"],
                             "baseline_loss": row["baseline"]["loss"],
                             "max_induction_score": max(row["induction_scores"]),
                             "group_ablation_accuracy_drop": row["selected_group_ablation"]["accuracy_drop"],
                             "group_ablation_loss_increase": row["selected_group_ablation"]["loss_increase"],
                             "patch_effective_examples": row["patching"]["effective_examples"],
                             "group_patch_restoration": row["patching"].get("group_logprob_restoration"),
                             "second_layer_accuracy_drop": row["layer_ablations"]["second_layer"]["accuracy_drop"],
                             "both_layers_accuracy_drop": row["layer_ablations"]["both_layers"]["accuracy_drop"],
                             "selected_heads_only_accuracy": row["selected_heads_only"]["accuracy"],
                             "selected_heads_only_loss": row["selected_heads_only"]["loss"],
                             "symbol_effective_rank": row["representations"]["symbol_effective_rank"],
                             "symbol_tv": row.get("symbol_mechanism", {}).get("baseline", {}).get("tv"),
                             "symbol_tv_after_second_layer_ablation": row.get(
                                 "symbol_mechanism", {}).get("second_layer_ablated", {}).get("tv")})
    plot_records(records, output_dir, args.family)
    print(f"Wrote {len(records)} mechanistic records to {output_dir}")


if __name__ == "__main__":
    main()
