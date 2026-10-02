"""Audit and summarize matched extended-task recursive experiments."""

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

from .extended import distribution_stats, label_for_symbol, sample_teacher
from .extended_study import CONDITIONS
from .data import SymbolTaskConfig
from .model import InductionTransformer, TransformerConfig


COLORS = {"clean": "#30343b", "mixed": "#168a82",
          "synthetic": "#d05b39", "resampled": "#a05b99",
          "resampled_sampled": "#3d75af"}
LABELS = {"clean": "Clean", "mixed": "90% feedback", "synthetic": "100% feedback",
          "resampled": "Oracle (soft)", "resampled_sampled": "Oracle (sampled)"}
METRICS = ("first_label_accuracy", "second_label_accuracy", "symbol_tv",
           "symbol_entropy", "first_label_nll", "second_label_nll",
           "next_symbol_nll", "loss", "perplexity")


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def equal_tensors(left: dict, right: dict) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(value, right[key]) for key, value in left.items())


def symbol_tv_from_true_distribution(symbols: torch.Tensor, prompt: torch.Tensor,
                                     task: SymbolTaskConfig) -> float:
    observed = torch.bincount(symbols, minlength=task.num_symbols).float() / len(symbols)
    context = prompt[:, 0:2 * task.context_pairs:2].reshape(-1)
    expected = torch.bincount(context, minlength=task.num_symbols).float() / len(context)
    return float((observed - expected).abs().sum().item() / 2)


def load_study(root: Path) -> dict[int, dict[str, dict]]:
    seeds = sorted(int(path.name[5:]) for path in root.glob("seed-*") if path.is_dir())
    if not seeds:
        raise ValueError("No seed directories found")
    study = {}
    for seed in seeds:
        runs = {}
        for name in CONDITIONS:
            folder = root / f"seed-{seed}" / name
            runs[name] = {
                "folder": folder,
                "protocol": read_json(folder / "protocol.json"),
                "summaries": read_json(folder / "summary.json"),
                "splits": torch.load(folder / "splits.pt", weights_only=True),
            }
        reference = runs["clean"]
        expected = {key: value for key, value in reference["protocol"].items()
                    if key not in ("output_dir", "synthetic_fraction", "feedback_source",
                                   "oracle_resample_objective")}
        for name, run in runs.items():
            protocol = run["protocol"]
            actual = {key: value for key, value in protocol.items()
                      if key not in ("output_dir", "synthetic_fraction", "feedback_source",
                                     "oracle_resample_objective")}
            if actual != expected or (protocol["synthetic_fraction"],
                                      protocol["feedback_source"]) != CONDITIONS[name]:
                raise ValueError(f"seed {seed}, {name}: protocol mismatch")
            objective = protocol.get("oracle_resample_objective", "expected")
            if objective != ("sampled" if name == "resampled_sampled" else "expected"):
                raise ValueError(f"seed {seed}, {name}: oracle resample objective mismatch")
            summaries = run["summaries"]
            if len(summaries) != protocol["generations"] or [
                    row["generation"] for row in summaries] != list(range(len(summaries))):
                raise ValueError(f"seed {seed}, {name}: incomplete generations")
            if any("test" not in row for row in summaries):
                raise ValueError(f"seed {seed}, {name}: missing test results")
            for split, fields in reference["splits"].items():
                if not equal_tensors(fields, run["splits"][split]):
                    raise ValueError(f"seed {seed}, {name}: {split} split mismatch")
            if summaries[0] != reference["summaries"][0]:
                raise ValueError(f"seed {seed}, {name}: generation 0 differs")
        study[seed] = runs
    return study


def audit_rows(study: dict[int, dict[str, dict]]) -> list[dict]:
    rows = []
    for seed, runs in study.items():
        for name, run in runs.items():
            protocol = run["protocol"]
            task = SymbolTaskConfig(**protocol["task"])
            oracle = run["splits"]["train"]
            for summary in run["summaries"]:
                generation = summary["generation"]
                folder = run["folder"] / f"generation-{generation:03d}"
                targets = torch.load(folder / "training_targets.pt", weights_only=True)
                history = read_json(folder / "history.json")
                if not history or history[-1]["step"] != protocol["steps"]:
                    raise ValueError(f"seed {seed}, {name}, gen {generation}: history incomplete")
                selected = [entry for entry in history if entry["step"] == summary["best_step"]]
                if len(selected) != 1 or selected[0]["validation"] != summary["validation"]:
                    raise ValueError(f"seed {seed}, {name}, gen {generation}: checkpoint mismatch")
                synthetic = targets["synthetic_mask"]
                resampled = targets["resampled_mask"]
                count = round(protocol["synthetic_fraction"] * protocol["train_size"])
                if generation == 0:
                    count = 0
                if int(synthetic.sum()) != summary["synthetic_count"] or int(
                        resampled.sum()) != summary["resampled_count"]:
                    raise ValueError(f"seed {seed}, {name}, gen {generation}: mask summary mismatch")
                if bool((synthetic & resampled).any()) or int((synthetic | resampled).sum()) != count:
                    raise ValueError(f"seed {seed}, {name}, gen {generation}: replacement count mismatch")
                untouched = ~(synthetic | resampled)
                for key in ("first_labels", "next_symbols", "second_labels"):
                    if not torch.equal(targets[key][untouched], oracle[key][untouched]):
                        raise ValueError(f"seed {seed}, {name}, gen {generation}: clean row changed")
                prompt = oracle["input_ids"]
                valid_symbols = (prompt[:, 0:2 * task.context_pairs:2]
                                 == targets["next_symbols"][:, None]).any(dim=1)
                if not bool(valid_symbols.all()):
                    raise ValueError(f"seed {seed}, {name}, gen {generation}: unsupported symbol")
                first_errors = int((targets["first_labels"] != oracle["first_labels"]).sum())
                inconsistencies = int((targets["second_labels"] != label_for_symbol(
                    prompt, targets["next_symbols"], task)).sum())
                stats = distribution_stats(targets["next_symbols"], oracle["next_symbols"],
                                           task.num_symbols)
                if first_errors != summary["first_label_errors"] or inconsistencies != summary[
                        "second_label_inconsistencies"] or stats != summary["symbol_distribution"]:
                    raise ValueError(f"seed {seed}, {name}, gen {generation}: target audit failed")
                row = {"seed": seed, "condition": name, "generation": generation,
                       "synthetic_count": int(synthetic.sum()),
                       "resampled_count": int(resampled.sum()),
                       "first_label_errors": first_errors,
                       "second_label_inconsistencies": inconsistencies,
                       "symbol_coverage": stats["coverage"],
                       "symbol_empirical_entropy": stats["entropy"],
                       "symbol_dominant_frequency": stats["dominant_symbol_frequency"],
                       "symbol_empirical_tv": stats["total_variation_from_oracle"],
                       "symbol_true_tv": symbol_tv_from_true_distribution(
                           targets["next_symbols"], prompt, task),
                       "best_step": summary["best_step"],
                       "max_induction_score": max(summary["attention"]["induction_scores"])}
                for prefix in ("validation", "test"):
                    row.update({f"{prefix}_{key}": summary[prefix][key] for key in METRICS})
                checkpoint = torch.load(folder / "best_model.pt", weights_only=True)
                model = InductionTransformer(TransformerConfig(**checkpoint["model_config"]))
                model.load_state_dict(checkpoint["model_state"])
                heldout = run["splits"]["test"]
                first, symbol, second = sample_teacher(model, heldout["input_ids"], task,
                                                       protocol["temperature"],
                                                       seed + 40000 + generation)
                first_wrong = first != heldout["first_labels"]
                second_wrong = second != label_for_symbol(heldout["input_ids"], symbol, task)
                row["autoreg_first_label_error_rate"] = float(first_wrong.float().mean())
                row["autoreg_second_label_inconsistency_rate"] = float(second_wrong.float().mean())
                row["autoreg_second_error_given_first_error"] = (
                    float(second_wrong[first_wrong].float().mean()) if bool(first_wrong.any()) else None)
                generated = distribution_stats(symbol, heldout["next_symbols"], task.num_symbols)
                row["autoreg_symbol_coverage"] = generated["coverage"]
                row["autoreg_symbol_entropy"] = generated["entropy"]
                row["autoreg_symbol_true_tv"] = symbol_tv_from_true_distribution(
                    symbol, heldout["input_ids"], task)
                rows.append(row)
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    result = []
    for condition in CONDITIONS:
        generations = sorted({row["generation"] for row in rows})
        for generation in generations:
            subset = [row for row in rows if row["condition"] == condition
                      and row["generation"] == generation]
            record = {"condition": condition, "generation": generation, "n": len(subset)}
            for key in ("test_first_label_accuracy", "test_second_label_accuracy",
                        "test_symbol_tv", "test_symbol_entropy", "test_loss",
                        "symbol_empirical_tv", "symbol_true_tv", "symbol_empirical_entropy",
                        "first_label_errors", "second_label_inconsistencies",
                        "autoreg_first_label_error_rate",
                        "autoreg_second_label_inconsistency_rate",
                        "autoreg_symbol_true_tv", "max_induction_score"):
                values = [row[key] for row in subset]
                record[f"{key}_mean"] = statistics.mean(values)
                record[f"{key}_sd"] = statistics.stdev(values) if len(values) > 1 else None
            result.append(record)
    return result


def plot(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    panels = (("test_first_label_accuracy", "First-label test accuracy"),
              ("test_second_label_accuracy", "Second-label test accuracy"),
              ("test_symbol_tv", "Conditional symbol TV from uniform"),
              ("symbol_true_tv", "Generated symbol TV from oracle distribution"))
    for ax, (key, title) in zip(axes.flat, panels):
        for name in CONDITIONS:
            group = [row for row in rows if row["condition"] == name]
            generations = sorted({row["generation"] for row in group})
            means = [statistics.mean(row[key] for row in group
                                     if row["generation"] == generation)
                     for generation in generations]
            sds = [statistics.stdev(row[key] for row in group
                                    if row["generation"] == generation)
                   for generation in generations]
            ax.errorbar(generations, means, yerr=sds, marker="o", capsize=3,
                        linewidth=1.8, color=COLORS[name], label=LABELS[name])
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_xticks(sorted({row["generation"] for row in rows}))
        ax.grid(alpha=0.2)
        if "accuracy" in key:
            ax.set_ylim(0.96, 1.005)
        elif key == "test_symbol_tv":
            ax.set_ylim(bottom=0)
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.suptitle("Extended recursive training | mean and sample SD across seeds")
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)


def plot_learning(study: dict[int, dict[str, dict]], output: Path) -> None:
    generations = study[next(iter(study))]["clean"]["protocol"]["generations"]
    fig, axes = plt.subplots(2, generations, figsize=(5 * generations, 7),
                             sharex="col", constrained_layout=True)
    if generations == 1:
        axes = axes.reshape(2, 1)
    for generation in range(generations):
        for condition in CONDITIONS:
            histories = [read_json(runs[condition]["folder"] /
                                   f"generation-{generation:03d}" / "history.json")
                         for runs in study.values()]
            steps = [entry["step"] for entry in histories[0]]
            if any([entry["step"] for entry in history] != steps for history in histories):
                raise ValueError("Unmatched evaluation schedules")
            for row_index, (metric, _) in enumerate((("first_label_accuracy", "Accuracy"),
                                                      ("symbol_tv", "TV"))):
                means = [statistics.mean(history[index]["validation"][metric]
                                         for history in histories)
                         for index in range(len(steps))]
                axes[row_index, generation].plot(steps, means, color=COLORS[condition],
                                                 linewidth=1.8,
                                                 label=LABELS[condition])
        axes[0, generation].set_title(f"Generation {generation}")
        axes[1, generation].set_xlabel("Optimization step")
        for row_index in range(2):
            axes[row_index, generation].grid(alpha=0.2)
    axes[0, 0].set_ylabel("First-label validation accuracy")
    axes[1, 0].set_ylabel("Conditional symbol TV")
    axes[0, 0].set_ylim(0, 1.02)
    axes[1, 0].set_ylim(bottom=0)
    axes[1, -1].legend(frameon=False, fontsize=8)
    fig.suptitle("Extended-task validation trajectories | mean across seeds")
    fig.savefig(output / "learning_curves.png", dpi=180)
    fig.savefig(output / "learning_curves.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/extended-study"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/extended-study"))
    args = parser.parse_args()
    study = load_study(args.runs_dir)
    rows = audit_rows(study)
    summary = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, records in (("per_run.csv", rows), ("aggregate.csv", summary)):
        with (args.output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    plot(rows, args.output_dir)
    plot_learning(study, args.output_dir)
    print(f"Audited {len(study)} seeds and {len(rows)} generation records")


if __name__ == "__main__":
    main()
