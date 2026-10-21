"""Run and audit matched clean/feedback Omniglot base-task chains."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .omniglot_analyze import sha256
from .omniglot_base import write_json
from .omniglot_study import (PROTOCOLS, SEEDS, aggregate, locked_settings, prepare_root,
                             run_commands, write_csv)


CONDITIONS = {"clean": 0.0, "feedback": 1.0}
METRICS = ("accuracy", "loss", "perplexity")


def study_config(protocol: str = "tuned") -> dict:
    return {"seeds": list(SEEDS), "conditions": CONDITIONS, "settings": PROTOCOLS[protocol]}


def run_chains(root: Path, protocol: str = "tuned", workers: int = 1, threads: int = 1,
               feature_file: Path = Path("data/omniglot_resnet18_randomized_order_s0.h5")) -> None:
    prepare_root(root, study_config(protocol))
    settings = PROTOCOLS[protocol]
    jobs = []
    for seed in SEEDS:
        for condition, fraction in CONDITIONS.items():
            folder = root / f"seed-{seed:02d}" / condition
            jobs.append((f"base seed {seed}, {condition}", folder, [
                sys.executable, "-m", "project1.omniglot_base",
                "--feature-file", str(feature_file.resolve()),
                "--output-dir", str(folder), "--seed", str(seed),
                "--generations", str(settings["generations"]),
                "--steps", str(settings["steps"]),
                "--train-size", str(settings["train_size"]),
                "--eval-size", str(settings["eval_size"]),
                "--batch-size", str(settings["batch_size"]),
                "--eval-every", str(settings["eval_every"]),
                "--learning-rate", str(settings["learning_rate"]),
                "--temperature", str(settings["temperature"]),
                "--synthetic-fraction", str(fraction)]))
    run_commands(jobs, workers, threads)


def collect_rows(root: Path) -> tuple[list[dict], list[dict]]:
    generations = locked_settings(root, study_config)["generations"]
    rows, paired = [], []
    for seed in SEEDS:
        paths = {name: root / f"seed-{seed:02d}" / name for name in CONDITIONS}
        protocols = {name: json.loads((path / "protocol.json").read_text(encoding="utf-8"))
                     for name, path in paths.items()}
        ignored = {"output_dir", "synthetic_fraction"}
        if {key: value for key, value in protocols["clean"].items() if key not in ignored} != {
                key: value for key, value in protocols["feedback"].items() if key not in ignored}:
            raise ValueError(f"Base seed {seed} protocol mismatch")
        for name, fraction in CONDITIONS.items():
            if protocols[name]["seed"] != seed or protocols[name]["synthetic_fraction"] != fraction:
                raise ValueError(f"Base seed {seed} condition mismatch: {name}")
        for artifact in ("splits.pt", "generation-000/training_labels.pt",
                         "generation-000/best_model.pt"):
            if sha256(paths["clean"] / artifact) != sha256(paths["feedback"] / artifact):
                raise ValueError(f"Base seed {seed} unmatched artifact: {artifact}")
        summaries = {name: json.loads((path / "summary.json").read_text(encoding="utf-8"))
                     for name, path in paths.items()}
        for condition, values in summaries.items():
            if [item["generation"] for item in values] != list(range(generations)):
                raise ValueError(f"Base seed {seed} incomplete summary: {condition}")
            for summary in values:
                rows.append({"seed": seed, "condition": condition,
                             "generation": summary["generation"],
                             "synthetic_count": summary["synthetic_count"],
                             "label_error_rate": summary["label_error_rate"],
                             **{metric: summary["test"][metric] for metric in METRICS}})
                if condition == "feedback":
                    control = summaries["clean"][summary["generation"]]
                    paired.append({"seed": seed, "condition": condition,
                                   "generation": summary["generation"],
                                   **{f"delta_{metric}": summary["test"][metric]
                                      - control["test"][metric] for metric in METRICS}})
    return rows, paired


def analyze(root: Path, output: Path) -> None:
    settings = locked_settings(root, study_config)
    rows, paired = collect_rows(root)
    aggregated = aggregate(rows, (*METRICS, "label_error_rate"), tuple(CONDITIONS))
    paired_aggregate = aggregate(paired, tuple(f"delta_{key}" for key in METRICS),
                                 ("feedback",))
    output.mkdir(parents=True, exist_ok=False)
    for name, values in (("per_run", rows), ("paired", paired),
                         ("aggregate", aggregated), ("paired_aggregate", paired_aggregate)):
        write_csv(output / f"{name}.csv", values)
    write_json(output / "audit.json", {"study_config": json.loads(
                                           (root / "study_config.json").read_text(encoding="utf-8")),
                                       "audited_seed_count": len(SEEDS),
                                       "matched_conditions_per_seed": list(CONDITIONS),
                                       "per_run_rows": len(rows)})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(7.5, 3.3), constrained_layout=True)
    for condition in CONDITIONS:
        selected = [item for item in aggregated if item["condition"] == condition]
        xs = [item["generation"] for item in selected]
        for axis, metric in zip(axes, ("accuracy", "loss")):
            ys = [item[f"{metric}_mean"] for item in selected]
            sd = [item[f"{metric}_sd"] for item in selected]
            axis.plot(xs, ys, marker="o", label=condition)
            axis.fill_between(xs, [y - s for y, s in zip(ys, sd)],
                              [y + s for y, s in zip(ys, sd)], alpha=0.12)
    for axis, title, ylabel in zip(axes, ("Base-task accuracy", "Base-task loss"),
                                   ("Held-out accuracy", "Held-out NLL")):
        axis.set(title=title, xlabel="Generation", ylabel=ylabel)
        axis.set_xticks(range(settings["generations"]))
        axis.grid(alpha=0.2)
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False)
    figure.savefig(output / "comparison.png", dpi=180)
    figure.savefig(output / "comparison.pdf")
    plt.close(figure)
    print(f"Audited {len(SEEDS)} base seeds and wrote analysis: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-file", type=Path,
                        default=Path("data/omniglot_resnet18_randomized_order_s0.h5"))
    parser.add_argument("--protocol", choices=tuple(PROTOCOLS), default="tuned")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if not args.analyze_only:
        run_chains(args.runs_dir, args.protocol, args.workers, args.threads, args.feature_file)
    analyze(args.runs_dir, args.output_dir)


if __name__ == "__main__":
    main()
