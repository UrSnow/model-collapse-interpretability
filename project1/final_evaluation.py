"""One-shot evaluation on fresh prompts excluded from every study split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import torch

from .analyze_extended import load_study, plot, symbol_tv_from_true_distribution
from .collapse import write_json
from .data import SymbolTaskConfig
from .extended import evaluate, label_for_symbol, oracle_split, sample_teacher
from .extended_study import CONDITIONS
from .model import InductionTransformer, TransformerConfig


def final_split(study_runs: dict[str, dict], seed: int, size: int) -> tuple[dict, dict]:
    reference = study_runs["clean"]
    task = SymbolTaskConfig(**reference["protocol"]["task"])
    excluded = {tuple(prompt) for split in reference["splits"].values()
                for prompt in split["input_ids"].tolist()}
    split_seed = seed + 900000
    split = oracle_split(task, size, split_seed, "test", excluded)
    prompts = split["input_ids"]
    if any(tuple(prompt) in excluded for prompt in prompts.tolist()):
        raise ValueError("Fresh final test overlaps a study split")
    metadata = {"seed": seed, "split_seed": split_seed, "size": size,
                "excluded_prompt_count": len(excluded),
                "prompt_sha256": hashlib.sha256(prompts.numpy().tobytes()).hexdigest(),
                "policy": "hash-partitioned test prompts, excluding every original split"}
    return split, metadata


def evaluate_run(folder: Path, split: dict[str, torch.Tensor],
                 task: SymbolTaskConfig, seed: int, generation: int) -> dict:
    checkpoint = torch.load(folder / "best_model.pt", weights_only=True)
    model = InductionTransformer(TransformerConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    metrics = evaluate(model, split, task)
    first, symbol, second = sample_teacher(model, split["input_ids"], task, 1.0,
                                           seed + 50000 + generation)
    return {
        "best_step": checkpoint["step"],
        **{f"test_{key}": value for key, value in metrics.items()},
        "autoreg_first_label_error_rate": float(
            (first != split["first_labels"]).float().mean()),
        "autoreg_second_label_inconsistency_rate": float((second != label_for_symbol(
            split["input_ids"], symbol, task)).float().mean()),
        "autoreg_symbol_true_tv": symbol_tv_from_true_distribution(
            symbol, split["input_ids"], task),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/extended-study"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("runs/final-evaluation"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/final-evaluation"))
    parser.add_argument("--size", type=int, default=2048)
    args = parser.parse_args()
    if args.size < 1:
        parser.error("size must be positive")
    study = load_study(args.runs_dir)
    args.artifact_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    torch.set_num_threads(1)
    for seed, runs in study.items():
        split, metadata = final_split(runs, seed, args.size)
        folder = args.artifact_dir / f"seed-{seed}"
        folder.mkdir()
        torch.save(split, folder / "split.pt")
        write_json(folder / "protocol.json", metadata)
        task = SymbolTaskConfig(**runs["clean"]["protocol"]["task"])
        generations = runs["clean"]["protocol"]["generations"]
        for condition in CONDITIONS:
            for generation in range(generations):
                checkpoint_folder = runs[condition]["folder"] / f"generation-{generation:03d}"
                metrics = evaluate_run(checkpoint_folder, split, task, seed, generation)
                rows.append({"seed": seed, "condition": condition,
                             "generation": generation, "prompt_sha256": metadata["prompt_sha256"],
                             **metrics})
                print(f"Final test: seed {seed}, {condition}, gen {generation}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_run.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = []
    metric_names = ("test_first_label_accuracy", "test_second_label_accuracy",
                    "test_symbol_tv", "test_symbol_entropy", "test_loss",
                    "test_perplexity", "autoreg_first_label_error_rate",
                    "autoreg_second_label_inconsistency_rate", "autoreg_symbol_true_tv")
    for condition in CONDITIONS:
        for generation in range(study[next(iter(study))]["clean"]["protocol"]["generations"]):
            subset = [row for row in rows if row["condition"] == condition
                      and row["generation"] == generation]
            record = {"condition": condition, "generation": generation, "n": len(subset)}
            for key in metric_names:
                values = [row[key] for row in subset]
                record[f"{key}_mean"] = statistics.mean(values)
                record[f"{key}_sd"] = statistics.stdev(values) if len(values) > 1 else None
            aggregate.append(record)
    with (args.output_dir / "aggregate.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    (args.output_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2),
                                                     encoding="utf-8")
    # The endpoint plot consumes the same metric names as exploratory analysis.
    plot([{**row, "symbol_true_tv": row["autoreg_symbol_true_tv"]} for row in rows],
         args.output_dir)
    print(f"Wrote {len(rows)} fresh-test records to {args.output_dir}")


if __name__ == "__main__":
    main()
