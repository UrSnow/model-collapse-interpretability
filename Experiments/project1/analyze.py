"""Aggregate Step 1 runs and create a report-ready scientific figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?", default=Path("runs/step1-multiseed"))
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def read_history(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def phase_change_step(history: list[dict[str, float]], threshold: float = 0.9) -> int | None:
    return next(
        (int(row["step"]) for row in history if row["validation_accuracy"] >= threshold),
        None,
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(
        path for path in args.run_dir.glob("seed-*") if (path / "summary.json").exists()
    )
    if not run_dirs:
        raise FileNotFoundError(f"no completed seed runs found below {args.run_dir}")

    histories: list[list[dict[str, float]]] = []
    records: list[dict[str, float | int | None]] = []
    summaries: list[dict[str, object]] = []
    for run_dir in run_dirs:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        history = read_history(run_dir / "history.csv")
        histories.append(history)
        summaries.append(summary)
        records.append(
            {
                "seed": int(config["experiment"]["seed"]),
                "best_step": int(summary["best_step"]),
                "phase_change_step_90pct": phase_change_step(history),
                "validation_accuracy": float(summary["validation"]["accuracy"]),
                "test_accuracy": float(summary["test"]["accuracy"]),
                "test_loss": float(summary["test"]["loss"]),
                "test_perplexity": float(summary["test"]["perplexity"]),
                "best_previous_token_score": max(
                    summary["attention"]["previous_token_scores"]
                ),
                "best_induction_score": max(summary["attention"]["induction_scores"]),
                "largest_ablation_drop": max(
                    item["accuracy_drop"] for item in summary["head_ablation"]
                ),
            }
        )

    fieldnames = list(records[0].keys())
    with (output_dir / "seed_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    numeric_fields = (
        "validation_accuracy",
        "test_accuracy",
        "test_loss",
        "test_perplexity",
        "best_previous_token_score",
        "best_induction_score",
        "largest_ablation_drop",
    )
    aggregate = {
        "num_seeds": len(records),
        "seeds": [record["seed"] for record in records],
        "phase_change_steps_90pct": [
            record["phase_change_step_90pct"] for record in records
        ],
    }
    phase_steps = np.asarray(
        [record["phase_change_step_90pct"] for record in records], dtype=float
    )
    aggregate["phase_change_step_90pct"] = {
        "mean": float(phase_steps.mean()),
        "sample_std": float(phase_steps.std(ddof=1)) if len(phase_steps) > 1 else 0.0,
        "min": int(phase_steps.min()),
        "max": int(phase_steps.max()),
    }
    canonical_seeds = [
        record["seed"]
        for record in records
        if record["best_previous_token_score"] >= 0.8
        and record["best_induction_score"] >= 0.6
    ]
    aggregate["canonical_circuit_definition"] = {
        "previous_token_score_at_least": 0.8,
        "induction_score_at_least": 0.6,
    }
    aggregate["canonical_circuit_seeds"] = canonical_seeds
    aggregate["canonical_circuit_seed_count"] = len(canonical_seeds)
    for field in numeric_fields:
        values = np.asarray([record[field] for record in records], dtype=float)
        aggregate[field] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    (output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )

    steps = np.asarray([row["step"] for row in histories[0]])
    if any([row["step"] for row in history] != steps.tolist() for history in histories):
        raise ValueError("all runs must use the same evaluation schedule")
    validation = np.asarray(
        [[row["validation_accuracy"] for row in history] for history in histories]
    )
    previous = np.asarray(
        [[row["best_previous_token_score"] for row in history] for history in histories]
    )
    induction = np.asarray(
        [[row["best_induction_score"] for row in history] for history in histories]
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(records)))
    for index, record in enumerate(records):
        axes[0, 0].plot(steps, validation[index], color=colors[index], alpha=0.75,
                        label=f"seed {record['seed']}")
    axes[0, 0].axhline(0.5, color="#555555", linestyle="--", linewidth=1,
                       label="label-set strategy")
    axes[0, 0].axhline(0.2, color="#999999", linestyle=":", linewidth=1,
                       label="random")
    axes[0, 0].set(title="Behavioral phase change", xlabel="Training step",
                   ylabel="Validation accuracy", ylim=(0, 1.03))
    axes[0, 0].legend(fontsize=8, ncol=2)

    for values, label, color in (
        (previous, "Previous-token", "#0072B2"),
        (induction, "Induction", "#D55E00"),
    ):
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
        axes[0, 1].plot(steps, mean, label=label, color=color)
        axes[0, 1].fill_between(
            steps,
            np.clip(mean - std, 0.0, 1.0),
            np.clip(mean + std, 0.0, 1.0),
            color=color,
            alpha=0.18,
        )
    axes[0, 1].set(title="Circuit formation across seeds", xlabel="Training step",
                   ylabel="Best attention score", ylim=(0, 1.03))
    axes[0, 1].legend()

    seed_labels = [str(record["seed"]) for record in records]
    test_accuracies = [record["test_accuracy"] for record in records]
    axes[1, 0].bar(seed_labels, test_accuracies, color="#009E73")
    axes[1, 0].axhline(0.5, color="#555555", linestyle="--", linewidth=1)
    axes[1, 0].set(title="Locked test performance", xlabel="Seed",
                   ylabel="Accuracy", ylim=(0, 1.03))

    representative = int(np.argmax(test_accuracies))
    previous_scores = summaries[representative]["attention"]["previous_token_scores"]
    induction_scores = summaries[representative]["attention"]["induction_scores"]
    heads = np.arange(len(previous_scores))
    width = 0.38
    axes[1, 1].bar(heads - width / 2, previous_scores, width,
                   label="Layer 1 previous-token", color="#0072B2")
    axes[1, 1].bar(heads + width / 2, induction_scores, width,
                   label="Layer 2 induction", color="#D55E00")
    axes[1, 1].set(title=f"Final head specialization (seed {records[representative]['seed']})",
                   xlabel="Head", ylabel="Attention score", ylim=(0, 1.03), xticks=heads)
    axes[1, 1].legend(fontsize=8)

    figure.suptitle("Step 1: In-context learning and induction-head formation", fontsize=14)
    figure.savefig(output_dir / "step1_summary.png", dpi=220)
    figure.savefig(output_dir / "step1_summary.pdf")
    plt.close(figure)
    print(json.dumps(aggregate, indent=2))
    print(f"Analysis written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
