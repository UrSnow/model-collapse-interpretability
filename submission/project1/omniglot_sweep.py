"""Generation-0 learning-rate and update-budget sweep selected on validation loss.

Each run trains one clean generation for the longest budget. The optimizer has
no learning-rate schedule and batches use a fixed generator, so the first N
updates of a long run are identical to an N-update run; the best validation
loss within each shorter budget is read from the saved history. The runners
also write test metrics after training, but this analysis never reads them.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean, stdev

from .omniglot_base import write_json
from .omniglot_study import write_csv


TASKS = {"base": "project1.omniglot_base", "extended": "project1.omniglot_extended"}
LEARNING_RATES = (1e-4, 3e-4, 1e-3, 3e-3)
BUDGETS = (1000, 2000, 3000, 5000)
SEEDS = (0, 1, 2)
SETTINGS = {"train_size": 4096, "eval_size": 256, "batch_size": 64, "eval_every": 100}


def sweep_config() -> dict:
    return {"tasks": list(TASKS), "learning_rates": list(LEARNING_RATES),
            "budgets": list(BUDGETS), "seeds": list(SEEDS), "settings": SETTINGS,
            "selection": "lowest mean validation loss over seeds; test metrics unused"}


def run_folder(root: Path, task: str, learning_rate: float, seed: int) -> Path:
    return root / task / f"lr-{learning_rate:g}" / f"seed-{seed:02d}"


def train_one(root: Path, feature_file: Path, task: str, learning_rate: float,
              seed: int, threads: int) -> None:
    folder = run_folder(root, task, learning_rate, seed)
    if (folder / "summary.json").is_file():
        print(f"Already complete: {folder}", flush=True)
        return
    if folder.exists():
        raise ValueError(f"Incomplete run needs manual review: {folder}")
    folder.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", TASKS[task], "--feature-file", str(feature_file),
               "--output-dir", str(folder), "--seed", str(seed), "--generations", "1",
               "--steps", str(max(BUDGETS)), "--learning-rate", str(learning_rate),
               "--synthetic-fraction", "0",
               *(f"--{key.replace('_', '-')}={value}" for key, value in SETTINGS.items())]
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
    with folder.with_suffix(".log").open("w", encoding="utf-8") as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, env=env)
    print(f"Completed: {folder}", flush=True)


def run_sweep(root: Path, feature_file: Path, workers: int, threads: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "sweep_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != sweep_config():
            raise ValueError("Existing sweep configuration differs")
    else:
        if any(root.iterdir()):
            raise ValueError("Sweep directory is not empty and has no configuration")
        write_json(config_path, sweep_config())
    jobs = [(task, rate, seed) for task in TASKS for rate in LEARNING_RATES for seed in SEEDS]
    with ThreadPoolExecutor(workers) as pool:
        for future in [pool.submit(train_one, root, feature_file.resolve(), *job, threads)
                       for job in jobs]:
            future.result()


def validation_loss(row: dict) -> float:
    return row["validation"]["loss"]


def validation_accuracy(row: dict) -> float:
    validation = row["validation"]
    return validation.get("accuracy", validation.get("first_label_accuracy"))


def collect(root: Path) -> tuple[list[dict], list[dict], dict]:
    if json.loads((root / "sweep_config.json").read_text(encoding="utf-8")) != sweep_config():
        raise ValueError("Sweep configuration differs from the locked grid")
    rows, curves = [], {}
    for task in TASKS:
        for rate in LEARNING_RATES:
            for seed in SEEDS:
                folder = run_folder(root, task, rate, seed)
                history = json.loads((folder / "generation-000" / "history.json")
                                     .read_text(encoding="utf-8"))
                if history[-1]["step"] != max(BUDGETS):
                    raise ValueError(f"Incomplete history: {folder}")
                curves[task, rate, seed] = history
                for budget in BUDGETS:
                    within = [row for row in history if row["step"] <= budget]
                    best = min(within, key=validation_loss)
                    rows.append({"task": task, "learning_rate": rate, "budget": budget,
                                 "seed": seed, "best_step": best["step"],
                                 "selected_at_budget_end": best["step"] == budget,
                                 "validation_loss": validation_loss(best),
                                 "validation_accuracy": validation_accuracy(best)})
    summary, choice = [], {}
    for task in TASKS:
        for rate in LEARNING_RATES:
            for budget in BUDGETS:
                selected = [row for row in rows if (row["task"], row["learning_rate"],
                                                    row["budget"]) == (task, rate, budget)]
                summary.append({"task": task, "learning_rate": rate, "budget": budget,
                                "n_seeds": len(selected),
                                "selected_at_budget_end": sum(row["selected_at_budget_end"]
                                                              for row in selected),
                                **{f"{key}_{stat}": func([row[key] for row in selected])
                                   for key in ("validation_loss", "validation_accuracy")
                                   for stat, func in (("mean", mean), ("sd", stdev))}})
        best = min((row for row in summary if row["task"] == task),
                   key=lambda row: row["validation_loss_mean"])
        choice[task] = {key: best[key] for key in ("learning_rate", "budget",
                                                    "validation_loss_mean",
                                                    "validation_accuracy_mean")}
    return rows, summary, {"choice": choice, "curves": curves}


def analyze(root: Path, output: Path) -> None:
    rows, summary, extra = collect(root)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "per_run.csv", rows)
    write_csv(output / "summary.csv", summary)
    write_json(output / "selection.json", {"sweep_config": sweep_config(),
                                           "choice": extra["choice"]})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, constrained_layout=True)
    for column, task in enumerate(TASKS):
        for rate in LEARNING_RATES:
            histories = [extra["curves"][task, rate, seed] for seed in SEEDS]
            steps = [row["step"] for row in histories[0]]
            for axis, key in zip(axes[:, column], (validation_loss, validation_accuracy)):
                values = [mean(key(history[i]) for history in histories)
                          for i in range(len(steps))]
                axis.plot(steps, values, label=f"lr {rate:g}")
        axes[0, column].set(title=f"{task.capitalize()} task, generation 0",
                            ylabel="Validation loss", yscale="log")
        axes[1, column].set(xlabel="Update", ylabel="Validation accuracy", ylim=(0, 1.01))
        for axis in axes[:, column]:
            axis.grid(alpha=0.2)
            for budget in BUDGETS[:-1]:
                axis.axvline(budget, color="0.8", lw=0.8, zorder=0)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.savefig(output / "sweep_curves.png", dpi=180)
    figure.savefig(output / "sweep_curves.pdf")
    plt.close(figure)
    print(json.dumps(extra["choice"], indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-file", type=Path,
                        default=Path("data/omniglot_resnet18_randomized_order_s0.h5"))
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if not args.analyze_only:
        run_sweep(args.runs_dir, args.feature_file, args.workers, args.threads)
    analyze(args.runs_dir, args.output_dir)


if __name__ == "__main__":
    main()
