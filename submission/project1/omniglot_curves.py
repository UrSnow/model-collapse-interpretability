"""Training dynamics and train/validation/test summaries for recursive studies.

Reads the saved per-step histories of every chain and re-scores each frozen,
validation-selected checkpoint on its fixed training episodes, both against
the targets it was trained on and against the true labels. No model is
trained or reselected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import torch

from . import omniglot_base, omniglot_extended
from .omniglot_base import write_json
from .omniglot_data import OmniglotBatch
from .omniglot_model import OmniglotExtendedTransformer, OmniglotInductionTransformer, OmniglotModelConfig
from .omniglot_study import write_csv


def chain_folders(root: Path) -> dict[tuple[int, str], Path]:
    folders = {}
    for seed_dir in sorted(root.glob("seed-*")):
        for folder in sorted(path for path in seed_dir.iterdir() if path.is_dir()):
            if not (folder / "summary.json").is_file():
                raise ValueError(f"Incomplete chain: {folder}")
            folders[int(seed_dir.name.split("-")[1]), folder.name] = folder
    if not folders:
        raise ValueError(f"No chains found under {root}")
    return folders


def load_model(folder: Path, extended: bool):
    checkpoint = torch.load(folder / "best_model.pt", weights_only=True)
    config = OmniglotModelConfig(**checkpoint["model_config"])
    model = (OmniglotExtendedTransformer if extended else OmniglotInductionTransformer)(config)
    model.load_state_dict(checkpoint["model_state"])
    return model


def train_scores(chain: Path, generation: int, extended: bool) -> dict[str, float]:
    folder = chain / f"generation-{generation:03d}"
    model = load_model(folder, extended)
    train = torch.load(chain / "splits.pt", weights_only=True)["train"]
    if not extended:
        labels = torch.load(folder / "training_labels.pt", weights_only=True)["targets"]
        batch = OmniglotBatch(**train)
        fitted = omniglot_base.evaluate(model, OmniglotBatch(**{**train, "targets": labels}))
        truth = omniglot_base.evaluate(model, batch)
        return {"train_accuracy_vs_targets": fitted["accuracy"],
                "train_loss_vs_targets": fitted["loss"],
                "train_accuracy_vs_truth": truth["accuracy"],
                "train_loss_vs_truth": truth["loss"]}
    targets = torch.load(folder / "training_targets.pt", weights_only=True)
    fitted = omniglot_extended.evaluate(model, {**train, "targets": targets["first_labels"],
                                                "next_indices": targets["next_indices"],
                                                "second_labels": targets["second_labels"]})
    truth = omniglot_extended.evaluate(model, train)
    return {"train_accuracy_vs_targets": fitted["first_label_accuracy"],
            "train_loss_vs_targets": fitted["loss"],
            "train_accuracy_vs_truth": truth["first_label_accuracy"],
            "train_loss_vs_truth": truth["loss"]}


def accuracy(metrics: dict) -> float:
    return metrics.get("accuracy", metrics.get("first_label_accuracy"))


def collect(root: Path, extended: bool) -> tuple[list[dict], dict]:
    rows, curves = [], {}
    for (seed, condition), chain in chain_folders(root).items():
        for summary in json.loads((chain / "summary.json").read_text(encoding="utf-8")):
            generation = summary["generation"]
            history = json.loads((chain / f"generation-{generation:03d}" / "history.json")
                                 .read_text(encoding="utf-8"))
            curves[condition, generation, seed] = history
            rows.append({"seed": seed, "condition": condition, "generation": generation,
                         "best_step": summary["best_step"],
                         **train_scores(chain, generation, extended),
                         "validation_accuracy": accuracy(summary["validation"]),
                         "validation_loss": summary["validation"]["loss"],
                         "test_accuracy": accuracy(summary["test"]),
                         "test_loss": summary["test"]["loss"],
                         "test_perplexity": summary["test"]["perplexity"]})
    return rows, curves


def aggregate(rows: list[dict]) -> list[dict]:
    keys = [key for key in rows[0] if key not in ("seed", "condition", "generation")]
    groups = sorted({(row["condition"], row["generation"]) for row in rows})
    result = []
    for condition, generation in groups:
        selected = [row for row in rows if (row["condition"], row["generation"])
                    == (condition, generation)]
        result.append({"condition": condition, "generation": generation,
                       "n_seeds": len(selected),
                       **{f"{key}_{stat}": func([float(row[key]) for row in selected])
                          if len(selected) > 1 or stat == "mean" else 0.0
                          for key in keys for stat, func in (("mean", mean), ("sd", stdev))}})
    return result


def plot(curves: dict, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conditions = sorted({key[0] for key in curves}, key=lambda name: name != "clean")
    generations = sorted({key[1] for key in curves})
    panels = (("Training batch loss", lambda row: row["training_batch_loss"], "log"),
              ("Validation loss", lambda row: row["validation"]["loss"], "log"),
              ("Validation perplexity", lambda row: row["validation"]["perplexity"], "log"),
              ("Validation accuracy", lambda row: accuracy(row["validation"]), "linear"))
    figure, axes = plt.subplots(len(panels), len(generations), sharex=True, sharey="row",
                                figsize=(3.2 * len(generations), 2.3 * len(panels)),
                                constrained_layout=True, squeeze=False)
    for column, generation in enumerate(generations):
        for condition in conditions:
            histories = [history for (name, gen, _), history in curves.items()
                         if name == condition and gen == generation]
            steps = [row["step"] for row in histories[0]]
            for row_index, (_, value, _) in enumerate(panels):
                means = [mean(value(history[i]) for history in histories)
                         for i in range(len(steps))]
                axes[row_index, column].plot(steps, means, label=condition, lw=1.4)
        axes[0, column].set_title(f"Generation {generation}")
        axes[-1, column].set_xlabel("Update")
    for row_index, (label, _, scale) in enumerate(panels):
        axes[row_index, 0].set(ylabel=label, yscale=scale)
        for axis in axes[row_index]:
            axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(f"{title}: seed means")
    figure.savefig(path.with_suffix(".png"), dpi=180)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-runs-dir", type=Path, required=True)
    parser.add_argument("--extended-runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for task, root, extended in (("base", args.base_runs_dir, False),
                                 ("extended", args.extended_runs_dir, True)):
        rows, curves = collect(root, extended)
        write_csv(args.output_dir / f"{task}_per_run.csv", rows)
        write_csv(args.output_dir / f"{task}_aggregate.csv", aggregate(rows))
        plot(curves, f"{task.capitalize()} task", args.output_dir / f"{task}_training_curves")
    write_json(args.output_dir / "protocol.json", {
        "base_runs_dir": str(args.base_runs_dir), "extended_runs_dir": str(args.extended_runs_dir),
        "train_scores": "frozen validation-selected checkpoints on fixed training episodes",
        "curves": "saved histories; training loss is one minibatch at each evaluation step"})
    print(f"Wrote training dynamics: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
