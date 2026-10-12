"""Audit and summarize matched Omniglot extended clean/feedback chains."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from .omniglot_extended import evaluate, sequence_logits
from .omniglot_model import OmniglotExtendedTransformer, OmniglotModelConfig


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_pair(clean: Path, feedback: Path) -> tuple[list[dict], list[dict]]:
    protocols = [json.loads((path / "protocol.json").read_text(encoding="utf-8"))
                 for path in (clean, feedback)]
    excluded = {"output_dir", "synthetic_fraction"}
    comparable = [{key: value for key, value in protocol.items() if key not in excluded}
                  for protocol in protocols]
    if comparable[0] != comparable[1]:
        raise ValueError("Clean and feedback protocols differ beyond the synthetic fraction")
    if protocols[0]["synthetic_fraction"] != 0 or protocols[1]["synthetic_fraction"] <= 0:
        raise ValueError("Expected a clean control and a positive-feedback treatment")
    for name in ("splits.pt", "generation-000/training_targets.pt",
                 "generation-000/best_model.pt"):
        if sha256(clean / name) != sha256(feedback / name):
            raise ValueError(f"Matched-run artifact differs: {name}")
    summaries = [json.loads((path / "summary.json").read_text(encoding="utf-8"))
                 for path in (clean, feedback)]
    if len(summaries[0]) != len(summaries[1]):
        raise ValueError("Generation counts differ")
    return summaries[0], summaries[1]


@torch.no_grad()
def head_ablation(path: Path, generation: int, size: int = 256) -> dict:
    protocol = json.loads((path / "protocol.json").read_text(encoding="utf-8"))
    splits = torch.load(path / "splits.pt", weights_only=True)
    source = splits["diagnostic"]
    split = {key: value[:size] for key, value in source.items()}
    config = OmniglotModelConfig(**protocol["model"])
    model = OmniglotExtendedTransformer(config)
    checkpoint = torch.load(path / f"generation-{generation:03d}" / "best_model.pt",
                            weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    baseline = evaluate(model, split)
    effects = []
    for layer in range(config.num_layers):
        for head in range(config.num_heads):
            first, pointer, second = sequence_logits(model, split, split["targets"],
                                                     split["next_indices"],
                                                     {layer: (head,)})
            first_accuracy = (first.argmax(-1) == split["targets"]).float().mean().item()
            second_accuracy = (second.argmax(-1) == split["second_labels"]).float().mean().item()
            pointer_tv = (pointer.softmax(-1) - 0.5).abs().sum(-1).mean().item() / 2
            effects.append({"layer": layer, "head": head,
                            "first_label_accuracy_drop": baseline["first_label_accuracy"] - first_accuracy,
                            "second_label_accuracy_drop": baseline["second_label_accuracy"] - second_accuracy,
                            "pointer_tv_change": pointer_tv - baseline["pointer_tv"]})
    return {"diagnostic_size": len(split), "baseline": baseline, "ablations": effects}


def run(clean: Path, feedback: Path, output: Path, oracle: Path | None = None) -> None:
    clean_rows, feedback_rows = audit_pair(clean, feedback)
    conditions = [("clean", clean, clean_rows), ("feedback", feedback, feedback_rows)]
    if oracle is not None:
        clean_protocol = json.loads((clean / "protocol.json").read_text(encoding="utf-8"))
        oracle_protocol = json.loads((oracle / "protocol.json").read_text(encoding="utf-8"))
        excluded = {"output_dir", "synthetic_fraction", "feedback_source",
                    "oracle_resample_objective", "oracle_resample_policy"}
        if {key: value for key, value in clean_protocol.items() if key not in excluded} != {
                key: value for key, value in oracle_protocol.items() if key not in excluded}:
            raise ValueError("Oracle protocol differs from clean protocol beyond control settings")
        if oracle_protocol["feedback_source"] != "oracle" or oracle_protocol["synthetic_fraction"] <= 0:
            raise ValueError("Expected a positive-fraction oracle control")
        for name in ("splits.pt", "generation-000/best_model.pt"):
            if sha256(clean / name) != sha256(oracle / name):
                raise ValueError(f"Oracle artifact differs: {name}")
        oracle_rows = json.loads((oracle / "summary.json").read_text(encoding="utf-8"))
        if len(oracle_rows) != len(clean_rows):
            raise ValueError("Oracle generation count differs")
        conditions.append(("oracle", oracle, oracle_rows))
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for condition, _, summaries in conditions:
        for summary in summaries:
            rows.append({"condition": condition, "generation": summary["generation"],
                         "synthetic_count": summary["synthetic_count"],
                         "resampled_count": summary.get("resampled_count", 0),
                         "first_label_errors": summary["first_label_errors"],
                         "second_label_inconsistencies": summary["second_label_inconsistencies"],
                         "class_coverage": summary["class_distribution"]["coverage"],
                         "class_entropy": summary["class_distribution"]["entropy"],
                         "class_tv": summary["class_distribution"]["total_variation_from_fixed_oracle"],
                         **{f"test_{key}": value for key, value in summary["test"].items()}})
    interpretation = {condition: head_ablation(path, len(summaries) - 1)
                      for condition, path, summaries in conditions}
    (output / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output / "head_ablation.json").write_text(json.dumps(interpretation, indent=2), encoding="utf-8")
    import csv
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    for condition, _, summaries in conditions:
        xs = [item["generation"] for item in summaries]
        for axis, field, source in ((axes[0], "first_label_accuracy", "test"),
                                    (axes[1], "second_label_accuracy", "test"),
                                    (axes[2], "entropy", "class_distribution")):
            axis.plot(xs, [item[source][field] for item in summaries],
                      marker="o", label=condition)
    for axis, title, ylabel in zip(axes, ("First label", "Second label", "Generated class diversity"),
                                   ("Held-out accuracy", "Held-out accuracy", "Training class entropy")):
        axis.set_title(title)
        axis.set_xlabel("Generation")
        axis.set_ylabel(ylabel)
        axis.set_xticks(range(len(clean_rows)))
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    axes[0].set_ylim(0.9, 1.0)
    axes[1].set_ylim(0.9, 1.0)
    axes[2].set_ylim(0, math.log(50) * 1.02)
    axes[2].axhline(math.log(50), color="0.4", linestyle="--", linewidth=1,
                    label="uniform 50-class maximum")
    axes[2].legend(frameon=False, fontsize=8)
    figure.savefig(output / "comparison.png", dpi=180)
    figure.savefig(output / "comparison.pdf")
    plt.close(figure)
    print(f"Audited pair and wrote Omniglot extended analysis: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.clean, args.feedback, args.output_dir, args.oracle)


if __name__ == "__main__":
    main()
