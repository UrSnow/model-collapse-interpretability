"""Run and audit matched clean/feedback Omniglot base-task chains."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .omniglot_analyze import sha256
from .omniglot_base import write_json
from .omniglot_study import SEEDS, SETTINGS, aggregate, write_csv


CONDITIONS = {"clean": 0.0, "feedback": 1.0}
METRICS = ("accuracy", "loss", "perplexity")


def study_config() -> dict:
    return {"seeds": list(SEEDS), "conditions": CONDITIONS, "settings": SETTINGS}


def run_chains(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "study_config.json"
    config = study_config()
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Existing base study configuration differs")
    else:
        if any(root.iterdir()):
            raise ValueError("Base study directory is not empty and has no configuration")
        write_json(config_path, config)
    for seed in SEEDS:
        for condition, fraction in CONDITIONS.items():
            folder = root / f"seed-{seed:02d}" / condition
            if (folder / "summary.json").is_file():
                print(f"Already complete: base seed {seed}, {condition}", flush=True)
                continue
            if folder.exists():
                raise ValueError(f"Incomplete base run needs manual review: {folder}")
            folder.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-m", "project1.omniglot_base",
                       "--output-dir", str(folder), "--seed", str(seed),
                       "--generations", str(SETTINGS["generations"]),
                       "--steps", str(SETTINGS["steps"]),
                       "--train-size", str(SETTINGS["train_size"]),
                       "--eval-size", str(SETTINGS["eval_size"]),
                       "--batch-size", str(SETTINGS["batch_size"]),
                       "--eval-every", str(SETTINGS["eval_every"]),
                       "--learning-rate", str(SETTINGS["learning_rate"]),
                       "--temperature", str(SETTINGS["temperature"]),
                       "--synthetic-fraction", str(fraction)]
            print(f"Training base seed {seed}, {condition}", flush=True)
            with (folder.parent / f"{condition}.log").open("w", encoding="utf-8") as stream:
                subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
            print(f"Completed base seed {seed}, {condition}", flush=True)


def collect_rows(root: Path) -> tuple[list[dict], list[dict]]:
    if json.loads((root / "study_config.json").read_text(encoding="utf-8")) != study_config():
        raise ValueError("Base study configuration differs from the locked protocol")
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
            if len(values) != SETTINGS["generations"] or [item["generation"] for item in values] != [0, 1, 2]:
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
    rows, paired = collect_rows(root)
    aggregated = aggregate(rows, (*METRICS, "label_error_rate"), tuple(CONDITIONS))
    paired_aggregate = aggregate(paired, tuple(f"delta_{key}" for key in METRICS),
                                 ("feedback",))
    output.mkdir(parents=True, exist_ok=False)
    for name, values in (("per_run", rows), ("paired", paired),
                         ("aggregate", aggregated), ("paired_aggregate", paired_aggregate)):
        write_csv(output / f"{name}.csv", values)
    write_json(output / "audit.json", {"study_config": study_config(),
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
        axis.set_xticks(range(SETTINGS["generations"]))
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
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if not args.analyze_only:
        run_chains(args.runs_dir)
    analyze(args.runs_dir, args.output_dir)


if __name__ == "__main__":
    main()
