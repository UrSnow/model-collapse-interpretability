"""Audit and compare matched Step 2 base-task generation chains."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


CONDITIONS = {"clean": 0.0, "mixed": 0.9, "synthetic": 1.0}
COLORS = {"clean": "#30343b", "mixed": "#168a82", "synthetic": "#d05b39"}
FIELDS = ("condition", "generation", "synthetic_fraction", "synthetic_count",
          "label_errors", "label_error_rate", "best_step", "validation_loss",
          "validation_accuracy", "validation_perplexity", "test_loss",
          "test_accuracy", "test_perplexity", "max_induction_score")


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def check_matched_runs(runs: dict[str, dict]) -> None:
    reference = runs["clean"]
    expected = {key: value for key, value in reference["protocol"].items()
                if key not in ("output_dir", "synthetic_fraction")}
    for name, run in runs.items():
        protocol = run["protocol"]
        actual = {key: value for key, value in protocol.items()
                  if key not in ("output_dir", "synthetic_fraction")}
        if actual != expected:
            raise ValueError(f"{name}: protocol differs from clean control")
        if protocol["synthetic_fraction"] != CONDITIONS[name]:
            raise ValueError(f"{name}: unexpected synthetic fraction")
        summaries = run["summaries"]
        if len(summaries) != protocol["generations"]:
            raise ValueError(f"{name}: incomplete generation chain")
        if [row["generation"] for row in summaries] != list(range(len(summaries))):
            raise ValueError(f"{name}: missing or unordered generation")
        if any("test" not in row for row in summaries):
            raise ValueError(f"{name}: final test metrics missing")
        for split, fields in reference["splits"].items():
            other = run["splits"].get(split)
            if other is None or other.keys() != fields.keys() or any(
                    not torch.equal(value, other[key]) for key, value in fields.items()):
                raise ValueError(f"{name}: {split} split differs from clean control")
        if summaries[0] != reference["summaries"][0]:
            raise ValueError(f"{name}: generation 0 differs from clean control")


def load_runs(runs_dir: Path) -> dict[str, dict]:
    runs = {}
    for name in CONDITIONS:
        folder = runs_dir / f"step2-base-{name}"
        runs[name] = {
            "folder": folder,
            "protocol": read_json(folder / "protocol.json"),
            "summaries": read_json(folder / "summary.json"),
            "splits": torch.load(folder / "splits.pt", map_location="cpu", weights_only=True),
        }
    check_matched_runs(runs)
    return runs


def collect_rows(runs: dict[str, dict]) -> list[dict]:
    rows = []
    for name, run in runs.items():
        true_labels = run["splits"]["train"]["targets"]
        for summary in run["summaries"]:
            generation = summary["generation"]
            folder = run["folder"] / f"generation-{generation:03d}"
            history = read_json(folder / "history.json")
            if not history or history[-1]["step"] != run["protocol"]["steps"]:
                raise ValueError(f"{name} generation {generation}: incomplete training history")
            selected = [row for row in history if row["step"] == summary["best_step"]]
            if len(selected) != 1 or selected[0]["validation"] != summary["validation"]:
                raise ValueError(f"{name} generation {generation}: checkpoint selection mismatch")
            labels = torch.load(folder / "training_labels.pt", map_location="cpu",
                                weights_only=True)
            targets, mask = labels["targets"], labels["synthetic_mask"]
            if targets.shape != true_labels.shape or mask.shape != true_labels.shape:
                raise ValueError(f"{name} generation {generation}: label shape mismatch")
            count = int(mask.sum().item())
            errors = int((targets != true_labels).sum().item())
            # The training summary stores the rate after float32 rounding.
            if count != summary["synthetic_count"] or abs(
                    errors / len(targets) - summary["label_error_rate"]) > 1e-7:
                raise ValueError(f"{name} generation {generation}: label audit failed")
            if ((targets != true_labels) & ~mask).any():
                raise ValueError(f"{name} generation {generation}: clean labels changed")
            expected_count = 0 if generation == 0 else round(
                run["protocol"]["synthetic_fraction"] * len(targets))
            if count != expected_count:
                raise ValueError(f"{name} generation {generation}: wrong synthetic count")
            rows.append({
                "condition": name, "generation": generation,
                "synthetic_fraction": run["protocol"]["synthetic_fraction"],
                "synthetic_count": count, "label_errors": errors,
                "label_error_rate": errors / len(targets),
                "best_step": summary["best_step"],
                "validation_loss": summary["validation"]["loss"],
                "validation_accuracy": summary["validation"]["accuracy"],
                "validation_perplexity": summary["validation"]["perplexity"],
                "test_loss": summary["test"]["loss"],
                "test_accuracy": summary["test"]["accuracy"],
                "test_perplexity": summary["test"]["perplexity"],
                "max_induction_score": max(summary["attention"]["induction_scores"]),
            })
    return rows


def plot_generations(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    metrics = (("test_accuracy", "Clean test accuracy"),
               ("test_loss", "Clean test loss"),
               ("label_errors", "Incorrect training labels"),
               ("max_induction_score", "Maximum induction attention score"))
    for ax, (metric, title) in zip(axes.flat, metrics):
        for name in CONDITIONS:
            points = [row for row in rows if row["condition"] == name]
            ax.plot([row["generation"] for row in points],
                    [row[metric] for row in points], marker="o", linewidth=2,
                    color=COLORS[name], label=name.capitalize())
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_xticks(sorted({row["generation"] for row in rows}))
        ax.grid(alpha=0.2)
        if metric == "test_accuracy":
            ax.set_ylim(0.995, 1.0002)
        elif metric == "test_loss":
            ax.set_yscale("log")
        elif metric == "label_errors":
            ax.set_ylim(-0.05, max(row[metric] for row in rows) + 0.25)
            ax.set_yticks(range(max(row[metric] for row in rows) + 1))
        else:
            ax.set_ylim(0, 1.02)
    axes[0, 0].legend(frameon=False, ncol=3, loc="lower left")
    fig.suptitle("Base-task recursive training | seed 0 | 1,024 clean test prompts")
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_training(runs: dict[str, dict], path: Path) -> None:
    generations = runs["clean"]["protocol"]["generations"]
    fig, axes = plt.subplots(1, generations, figsize=(5 * generations, 4),
                             sharey=True, constrained_layout=True)
    if generations == 1:
        axes = [axes]
    for generation, ax in enumerate(axes):
        for name, run in runs.items():
            history = read_json(run["folder"] / f"generation-{generation:03d}"
                                / "history.json")
            ax.plot([row["step"] for row in history],
                    [row["validation"]["accuracy"] for row in history],
                    color=COLORS[name], linewidth=2, label=name.capitalize())
        ax.set_title(f"Generation {generation}")
        ax.set_xlabel("Optimization step")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Clean validation accuracy")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle("Learning curves on matched clean validation prompts")
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/step3-base"))
    args = parser.parse_args()
    runs = load_runs(args.runs_dir)
    rows = collect_rows(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    plot_generations(rows, args.output_dir / "generation_comparison.png")
    plot_training(runs, args.output_dir / "learning_curves.png")
    print(f"Audited {len(runs)} matched chains; wrote {len(rows)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
