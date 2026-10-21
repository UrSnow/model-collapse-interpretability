"""Run and audit a matched multi-seed Omniglot extended-task study."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean, stdev

from .omniglot_analyze import sha256
from .omniglot_base import write_json


CONDITIONS = {
    "clean": (0.0, "model"),
    "partial": (0.9, "model"),
    "feedback": (1.0, "model"),
    "oracle": (1.0, "oracle"),
}
SEEDS = (0, 1, 2, 3, 4)
# "pilot" is the original three-generation study. "tuned" uses the learning
# rate and update budget selected on generation-0 validation loss by
# project1.omniglot_sweep, and runs ten generations.
PROTOCOLS = {
    "pilot": {
        "generations": 3,
        "steps": 1000,
        "train_size": 4096,
        "eval_size": 256,
        "batch_size": 64,
        "eval_every": 100,
        "learning_rate": 3e-4,
        "temperature": 1.0,
        "oracle_resample_objective": "sampled",
    },
    "tuned": {
        "generations": 10,
        "steps": 5000,
        "train_size": 4096,
        "eval_size": 256,
        "batch_size": 64,
        "eval_every": 100,
        "learning_rate": 1e-3,
        "temperature": 1.0,
        "oracle_resample_objective": "sampled",
    },
}
SETTINGS = PROTOCOLS["tuned"]
MATCH_EXCEPT = {"output_dir", "synthetic_fraction", "feedback_source"}
METRICS = ("first_label_accuracy", "second_label_accuracy", "greedy_joint_accuracy",
           "first_label_nll", "second_label_nll", "next_feature_nll", "pointer_tv")


def study_config(protocol: str = "tuned") -> dict:
    return {"seeds": list(SEEDS), "conditions": {name: {"synthetic_fraction": fraction,
             "feedback_source": source} for name, (fraction, source) in CONDITIONS.items()},
            "settings": PROTOCOLS[protocol]}


def locked_settings(root: Path, config_function=study_config) -> dict:
    """Return the settings of the locked protocol recorded in a study directory."""
    config = json.loads((root / "study_config.json").read_text(encoding="utf-8"))
    for protocol in PROTOCOLS:
        if config == config_function(protocol):
            return PROTOCOLS[protocol]
    raise ValueError("Study configuration differs from every locked protocol")


def prepare_root(root: Path, config: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "study_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Existing study configuration differs")
    else:
        if any(root.iterdir()):
            raise ValueError("Study directory is not empty and has no configuration")
        write_json(config_path, config)


def run_commands(jobs: list[tuple[str, Path, list[str]]], workers: int, threads: int) -> None:
    """Train independent chains in parallel; generations within a chain stay sequential."""
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}

    def train(job: tuple[str, Path, list[str]]) -> None:
        label, folder, command = job
        if (folder / "summary.json").is_file():
            print(f"Already complete: {label}", flush=True)
            return
        if folder.exists():
            raise ValueError(f"Incomplete run needs manual review: {folder}")
        folder.parent.mkdir(parents=True, exist_ok=True)
        print(f"Training {label}", flush=True)
        with folder.with_suffix(".log").open("w", encoding="utf-8") as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, env=env)
        print(f"Completed {label}", flush=True)

    with ThreadPoolExecutor(workers) as pool:
        for future in [pool.submit(train, job) for job in jobs]:
            future.result()


def run_chains(root: Path, protocol: str = "tuned", workers: int = 1, threads: int = 1,
               feature_file: Path = Path("data/omniglot_resnet18_randomized_order_s0.h5")) -> None:
    prepare_root(root, study_config(protocol))
    settings = PROTOCOLS[protocol]
    jobs = []
    for seed in SEEDS:
        for condition, (fraction, source) in CONDITIONS.items():
            folder = root / f"seed-{seed:02d}" / condition
            jobs.append((f"seed {seed}, {condition}", folder, [
                sys.executable, "-m", "project1.omniglot_extended",
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
                "--synthetic-fraction", str(fraction),
                "--feedback-source", source,
                "--oracle-resample-objective", settings["oracle_resample_objective"]]))
    run_commands(jobs, workers, threads)


def audit_seed(root: Path, seed: int, generations: int | None = None) -> dict[str, list[dict]]:
    if generations is None:
        generations = locked_settings(root)["generations"]
    paths = {name: root / f"seed-{seed:02d}" / name for name in CONDITIONS}
    protocols = {name: json.loads((path / "protocol.json").read_text(encoding="utf-8"))
                 for name, path in paths.items()}
    reference = {key: value for key, value in protocols["clean"].items()
                 if key not in MATCH_EXCEPT}
    for name, protocol in protocols.items():
        if {key: value for key, value in protocol.items() if key not in MATCH_EXCEPT} != reference:
            raise ValueError(f"Seed {seed} protocol mismatch: {name}")
        fraction, source = CONDITIONS[name]
        if (protocol["seed"] != seed or protocol["synthetic_fraction"] != fraction
                or protocol["feedback_source"] != source):
            raise ValueError(f"Seed {seed} condition mismatch: {name}")
    for artifact in ("splits.pt", "generation-000/training_targets.pt",
                     "generation-000/best_model.pt"):
        reference_hash = sha256(paths["clean"] / artifact)
        for name, path in paths.items():
            if sha256(path / artifact) != reference_hash:
                raise ValueError(f"Seed {seed} {name} unmatched artifact: {artifact}")
    summaries = {name: json.loads((path / "summary.json").read_text(encoding="utf-8"))
                 for name, path in paths.items()}
    for name, values in summaries.items():
        if [row["generation"] for row in values] != list(range(generations)):
            raise ValueError(f"Seed {seed} incomplete summary: {name}")
    return summaries


def collect_rows(root: Path) -> tuple[list[dict], list[dict]]:
    generations = locked_settings(root)["generations"]
    rows = []
    paired = []
    for seed in SEEDS:
        summaries = audit_seed(root, seed, generations)
        for condition, values in summaries.items():
            for summary in values:
                row = {"seed": seed, "condition": condition,
                       "generation": summary["generation"],
                       "synthetic_count": summary["synthetic_count"],
                       "resampled_count": summary["resampled_count"],
                       "first_label_errors": summary["first_label_errors"],
                       "second_label_inconsistencies": summary["second_label_inconsistencies"],
                       "class_coverage": summary["class_distribution"]["coverage"],
                       "class_entropy": summary["class_distribution"]["entropy"],
                       "class_tv": summary["class_distribution"]["total_variation_from_fixed_oracle"],
                       **{key: summary["test"][key] for key in METRICS}}
                rows.append(row)
                if condition != "clean":
                    control = summaries["clean"][summary["generation"]]
                    paired.append({"seed": seed, "condition": condition,
                                   "generation": summary["generation"],
                                   **{f"delta_{key}": summary["test"][key] - control["test"][key]
                                      for key in METRICS},
                                   "delta_class_entropy": row["class_entropy"]
                                   - control["class_distribution"]["entropy"],
                                   "delta_class_tv": row["class_tv"]
                                   - control["class_distribution"]["total_variation_from_fixed_oracle"]})
    return rows, paired


def aggregate(rows: list[dict], keys: tuple[str, ...],
              conditions: tuple[str, ...] = tuple(CONDITIONS)) -> list[dict]:
    result = []
    generations = max(row["generation"] for row in rows) + 1
    for condition in conditions:
        for generation in range(generations):
            selected = [row for row in rows if row["condition"] == condition
                        and row["generation"] == generation]
            if len(selected) != len(SEEDS):
                raise ValueError(f"Missing seeds for {condition} G{generation}")
            result.append({"condition": condition, "generation": generation, "n_seeds": len(selected),
                           **{f"{key}_{stat}": func([row[key] for row in selected])
                              for key in keys for stat, func in (("mean", mean), ("sd", stdev))}})
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root: Path, output: Path) -> None:
    settings = locked_settings(root)
    rows, paired = collect_rows(root)
    indexed = {(row["seed"], row["condition"], row["generation"]): row for row in rows}
    from_g0 = []
    for row in rows:
        baseline = indexed[row["seed"], row["condition"], 0]
        from_g0.append({"seed": row["seed"], "condition": row["condition"],
                        "generation": row["generation"],
                        **{f"delta_{key}": row[key] - baseline[key]
                           for key in (*METRICS, "class_entropy", "class_tv")}})
    contrasts = []
    for seed in SEEDS:
        for generation in range(settings["generations"]):
            feedback = indexed[seed, "feedback", generation]
            oracle = indexed[seed, "oracle", generation]
            contrasts.append({"seed": seed, "condition": "feedback_minus_oracle",
                              "generation": generation,
                              **{f"delta_{key}": feedback[key] - oracle[key]
                                 for key in (*METRICS, "class_entropy", "class_tv")}})
    aggregate_rows = aggregate(rows, (*METRICS, "class_entropy", "class_tv",
                                      "first_label_errors", "second_label_inconsistencies"))
    paired_rows = aggregate(paired, tuple(key for key in paired[0] if key.startswith("delta_")),
                            tuple(name for name in CONDITIONS if name != "clean"))
    from_g0_aggregate = aggregate(from_g0, tuple(key for key in from_g0[0]
                                                   if key.startswith("delta_")))
    contrast_aggregate = aggregate(contrasts, tuple(key for key in contrasts[0]
                                                    if key.startswith("delta_")),
                                   ("feedback_minus_oracle",))
    output.mkdir(parents=True, exist_ok=False)
    for name, values in (("per_run", rows), ("paired", paired),
                         ("from_g0", from_g0), ("from_g0_aggregate", from_g0_aggregate),
                         ("aggregate", aggregate_rows), ("paired_aggregate", paired_rows),
                         ("feedback_vs_oracle", contrasts),
                         ("feedback_vs_oracle_aggregate", contrast_aggregate)):
        write_csv(output / f"{name}.csv", values)
    write_json(output / "audit.json", {"study_config": json.loads(
                                           (root / "study_config.json").read_text(encoding="utf-8")),
                                       "audited_seed_count": len(SEEDS),
                                       "matched_conditions_per_seed": list(CONDITIONS),
                                       "per_run_rows": len(rows)})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    panels = (("first_label_accuracy", "First-label accuracy", "Accuracy", (0.8, 1)),
              ("second_label_accuracy", "Second-label accuracy", "Accuracy", (0.8, 1)),
              ("pointer_tv", "Pointer deviation", "TV from uniform", (0, 0.5)))
    for condition in CONDITIONS:
        selected = [row for row in aggregate_rows if row["condition"] == condition]
        xs = [row["generation"] for row in selected]
        for axis, (key, _, _, _) in zip(axes, panels):
            ys = [row[f"{key}_mean"] for row in selected]
            sd = [row[f"{key}_sd"] for row in selected]
            axis.plot(xs, ys, marker="o", label=condition)
            axis.fill_between(xs, [y - s for y, s in zip(ys, sd)],
                              [y + s for y, s in zip(ys, sd)], alpha=0.12)
    for axis, (_, title, ylabel, limits) in zip(axes, panels):
        axis.set(title=title, xlabel="Generation", ylabel=ylabel, ylim=limits)
        axis.set_xticks(range(settings["generations"]))
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(output / "comparison.png", dpi=180)
    figure.savefig(output / "comparison.pdf")
    plt.close(figure)
    print(f"Audited {len(SEEDS)} seeds and wrote study analysis: {output}", flush=True)


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
