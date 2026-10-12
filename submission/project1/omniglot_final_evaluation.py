"""Evaluate frozen Omniglot study checkpoints on fresh held-out episodes."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev

import torch

from .omniglot_analyze import sha256
from .omniglot_base import evaluate as evaluate_base, write_json
from .omniglot_base_study import CONDITIONS as BASE_CONDITIONS
from .omniglot_base_study import collect_rows as audit_base
from .omniglot_data import (FEATURE_SHA256, OmniglotBatch, OmniglotTaskConfig,
                            OmniglotTaskGenerator, load_features)
from .omniglot_extended import evaluate as evaluate_extended
from .omniglot_model import (OmniglotExtendedTransformer, OmniglotInductionTransformer,
                             OmniglotModelConfig)
from .omniglot_study import CONDITIONS as EXTENDED_CONDITIONS
from .omniglot_study import SEEDS, SETTINGS, audit_seed, study_config


BASE_METRICS = ("accuracy", "loss", "perplexity")
EXTENDED_METRICS = ("first_label_accuracy", "second_label_accuracy",
                    "greedy_joint_accuracy", "first_label_nll", "next_feature_nll",
                    "second_label_nll", "pointer_tv", "pointer_entropy", "loss")
EPISODE_FIELDS = ("class_ids", "context_labels", "targets", "query_context_index",
                  "matching_label_position")


def episode_keys(split: dict[str, torch.Tensor]) -> set[tuple[int, ...]]:
    return {(*classes, *labels) for classes, labels in zip(
        split["class_ids"].tolist(), split["context_labels"].tolist())}


def prior_test_keys(runs_dir: Path, required_dirs: tuple[Path, ...] = ()) -> tuple[
        set[tuple[int, ...]], list[dict]]:
    keys: set[tuple[int, ...]] = set()
    sources = []
    seen_hashes = set()
    for path in sorted(runs_dir.rglob("splits.pt")):
        if not (any(part.startswith("omniglot-") for part in path.relative_to(runs_dir).parts)
                or any(path.resolve().is_relative_to(root.resolve()) for root in required_dirs)):
            continue
        digest = sha256(path)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        splits = torch.load(path, map_location="cpu", weights_only=True)
        test = splits.get("test")
        if not isinstance(test, dict) or not {"class_ids", "context_labels"} <= test.keys():
            raise ValueError(f"Missing Omniglot test episode IDs: {path}")
        found = episode_keys(test)
        keys.update(found)
        sources.append({"path": str(path), "sha256": digest, "test_episodes": len(found)})
    if not sources:
        raise ValueError(f"No prior Omniglot test splits found under {runs_dir}")
    return keys, sources


def make_manifest(features: torch.Tensor, size: int,
                  excluded: set[tuple[int, ...]]) -> dict[int, dict[str, torch.Tensor]]:
    if size < 1:
        raise ValueError("Final evaluation size must be positive")
    task = OmniglotTaskConfig()
    manifest = {}
    used = set(excluded)
    for seed in SEEDS:
        batch = OmniglotTaskGenerator(features, task, 700_000 + seed).sample_unique(
            size, "test", exclude=used)
        keys = episode_keys(asdict(batch))
        if len(keys) != size or not keys.isdisjoint(used):
            raise ValueError(f"Final episode collision for seed {seed}")
        used.update(keys)
        pointer = torch.randint(2, (size,), generator=torch.Generator().manual_seed(800_000 + seed))
        manifest[seed] = {name: getattr(batch, name).clone() for name in EPISODE_FIELDS}
        manifest[seed]["next_indices"] = pointer
    return manifest


def validate_manifest(manifest: dict[int, dict[str, torch.Tensor]], size: int,
                      excluded: set[tuple[int, ...]]) -> None:
    if set(manifest) != set(SEEDS):
        raise ValueError("Final manifest has the wrong study seeds")
    used = set(excluded)
    for seed in SEEDS:
        episode = manifest[seed]
        if set(episode) != {*EPISODE_FIELDS, "next_indices"}:
            raise ValueError(f"Final manifest fields differ for seed {seed}")
        if any(len(value) != size for value in episode.values()):
            raise ValueError(f"Final manifest size differs for seed {seed}")
        keys = episode_keys(episode)
        if len(keys) != size or not keys.isdisjoint(used):
            raise ValueError(f"Final manifest repeats a prior or current prompt for seed {seed}")
        used.update(keys)


def batches_from_manifest(features: torch.Tensor, episode: dict[str, torch.Tensor]
                          ) -> tuple[OmniglotBatch, dict[str, torch.Tensor]]:
    batch = OmniglotBatch(examples=features[episode["class_ids"], 0],
                          **{name: episode[name] for name in EPISODE_FIELDS})
    rows = torch.arange(len(batch))
    pointer = episode["next_indices"]
    if not bool((batch.class_ids[rows, batch.query_context_index] == batch.class_ids[:, 2]).all()):
        raise ValueError("Final manifest has invalid query/context identity")
    if not bool((batch.context_labels[rows, batch.query_context_index] == batch.targets).all()):
        raise ValueError("Final manifest has invalid targets")
    if not bool(((pointer >= 0) & (pointer < 2)).all()):
        raise ValueError("Final manifest has invalid pointers")
    extended = {**asdict(batch), "next_indices": pointer,
                "second_labels": batch.context_labels[rows, pointer],
                "next_class_ids": batch.class_ids[rows, pointer]}
    return batch, extended


def evaluate_checkpoint(path: Path, family: str, batch: OmniglotBatch,
                        extended: dict[str, torch.Tensor]) -> dict[str, float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = OmniglotModelConfig(**checkpoint["model_config"])
    model_type = OmniglotInductionTransformer if family == "base" else OmniglotExtendedTransformer
    model = model_type(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return evaluate_base(model, batch) if family == "base" else evaluate_extended(model, extended)


def paired_rows(rows: list[dict]) -> list[dict]:
    indexed = {(row["family"], row["seed"], row["condition"], row["generation"]): row
               for row in rows}
    result = []
    for family, conditions, metrics in (("base", BASE_CONDITIONS, BASE_METRICS),
                                         ("extended", EXTENDED_CONDITIONS, EXTENDED_METRICS)):
        comparisons = [(condition, "clean") for condition in conditions if condition != "clean"]
        if family == "extended":
            comparisons.append(("feedback", "oracle"))
        for seed in SEEDS:
            for generation in range(SETTINGS["generations"]):
                for treatment, control in comparisons:
                    left = indexed[family, seed, treatment, generation]
                    right = indexed[family, seed, control, generation]
                    result.append({"family": family, "seed": seed, "generation": generation,
                                   "contrast": f"{treatment}_minus_{control}",
                                   **{f"delta_{metric}": left[metric] - right[metric]
                                      for metric in metrics}})
    return result


def aggregate_rows(rows: list[dict], group_key: str,
                   metric_keys: dict[str, tuple[str, ...]]) -> list[dict]:
    result = []
    for family, metrics in metric_keys.items():
        groups = sorted({row[group_key] for row in rows if row["family"] == family})
        for group in groups:
            for generation in range(SETTINGS["generations"]):
                selected = [row for row in rows if row["family"] == family
                            and row[group_key] == group and row["generation"] == generation]
                if len(selected) != len(SEEDS):
                    raise ValueError(f"Incomplete final group: {family}, {group}, G{generation}")
                result.append({"family": family, group_key: group, "generation": generation,
                               "n_seeds": len(selected),
                               **{f"{key}_{stat}": function([row[key] for row in selected])
                                  for key in metrics for stat, function in (("mean", mean), ("sd", stdev))}})
    return result


def plot_results(output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (output / "aggregate.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    panels = (("base", "accuracy", "Base: first-label accuracy"),
              ("extended", "first_label_accuracy", "Extended: first-label accuracy"),
              ("extended", "pointer_tv", "Extended: pointer TV from uniform"))
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    for axis, (family, metric, title) in zip(axes, panels):
        conditions = BASE_CONDITIONS if family == "base" else EXTENDED_CONDITIONS
        for condition in conditions:
            selected = [row for row in rows if row["family"] == family
                        and row["condition"] == condition]
            xs = [int(row["generation"]) for row in selected]
            means = [float(row[f"{metric}_mean"]) for row in selected]
            errors = [float(row[f"{metric}_sd"]) for row in selected]
            axis.errorbar(xs, means, yerr=errors, marker="o", capsize=2, label=condition)
        axis.set(title=title, xlabel="Generation")
        axis.set_xticks(range(SETTINGS["generations"]))
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Held-out accuracy")
    axes[1].set_ylabel("Held-out accuracy")
    axes[2].set_ylabel("Total variation")
    axes[2].legend(frameon=False, fontsize=8)
    figure.savefig(output / "final_comparison.png", dpi=180)
    figure.savefig(output / "final_comparison.pdf")
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.size <= SETTINGS["eval_size"]:
        raise ValueError("Final test must be larger than the study's 256 episodes per seed")
    audit_base(args.base_runs_dir)
    if json.loads((args.extended_runs_dir / "study_config.json").read_text(encoding="utf-8")) != study_config():
        raise ValueError("Extended study configuration differs from locked protocol")
    for seed in SEEDS:
        audit_seed(args.extended_runs_dir, seed)
    features = load_features(args.feature_file)
    excluded, sources = prior_test_keys(args.prior_runs_dir,
                                       (args.base_runs_dir, args.extended_runs_dir))
    if args.manifest_file is None:
        manifest = make_manifest(features, args.size, excluded)
    else:
        manifest = torch.load(args.manifest_file, map_location="cpu", weights_only=True)
    validate_manifest(manifest, args.size, excluded)
    checkpoint_hashes = []
    for family, root, conditions in (("base", args.base_runs_dir, BASE_CONDITIONS),
                                     ("extended", args.extended_runs_dir, EXTENDED_CONDITIONS)):
        for seed in SEEDS:
            for condition in conditions:
                for generation in range(SETTINGS["generations"]):
                    path = root / f"seed-{seed:02d}" / condition / f"generation-{generation:03d}" / "best_model.pt"
                    checkpoint_hashes.append({"family": family, "seed": seed,
                                              "condition": condition, "generation": generation,
                                              "path": str(path), "sha256": sha256(path)})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(manifest, args.output_dir / "episodes.pt")
    write_json(args.output_dir / "protocol.json", {
        "status": "frozen checkpoints; no training or checkpoint selection on final episodes",
        "size_per_seed": args.size, "seeds": list(SEEDS), "feature_sha256": FEATURE_SHA256,
        "episode_manifest_sha256": sha256(args.output_dir / "episodes.pt"),
        "replay_manifest": str(args.manifest_file) if args.manifest_file else None,
        "episode_seed_policy": "700000 + study seed; pointer seed 800000 + study seed",
        "exclusion_policy": "exact ordered class IDs plus two context labels; all prior Omniglot test splits",
        "excluded_unique_prompts": len(excluded), "prior_sources": sources,
        "checkpoints": checkpoint_hashes,
    })
    rows = []
    for seed in SEEDS:
        batch, extended = batches_from_manifest(features, manifest[seed])
        for item in checkpoint_hashes:
            if item["seed"] != seed:
                continue
            metrics = evaluate_checkpoint(Path(item["path"]), item["family"], batch, extended)
            rows.append({key: item[key] for key in ("family", "seed", "condition", "generation")}
                        | metrics)
            print(f"Final {item['family']} seed {seed} {item['condition']} G{item['generation']}",
                  flush=True)
    paired = paired_rows(rows)
    aggregate = aggregate_rows(rows, "condition", {"base": BASE_METRICS,
                                                    "extended": EXTENDED_METRICS})
    paired_aggregate = aggregate_rows(paired, "contrast", {
        "base": tuple(f"delta_{key}" for key in BASE_METRICS),
        "extended": tuple(f"delta_{key}" for key in EXTENDED_METRICS)})
    # The families have different metrics, so use a common header for each CSV.
    for name, values in (("per_run", rows), ("paired", paired), ("aggregate", aggregate),
                         ("paired_aggregate", paired_aggregate)):
        fields = list(dict.fromkeys(key for row in values for key in row))
        with (args.output_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)
    write_json(args.output_dir / "audit.json", {"checkpoint_count": len(checkpoint_hashes),
                                                  "evaluated_rows": len(rows),
                                                  "paired_rows": len(paired),
                                                  "final_unique_prompts": args.size * len(SEEDS),
                                                  "prior_overlap": 0})
    plot_results(args.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-file", type=Path,
                        default=Path("data/omniglot_resnet18_randomized_order_s0.h5"))
    parser.add_argument("--base-runs-dir", type=Path,
                        default=Path("runs/omniglot-base-study-20261009"))
    parser.add_argument("--extended-runs-dir", type=Path,
                        default=Path("runs/omniglot-extended-study-20261009"))
    parser.add_argument("--prior-runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--manifest-file", type=Path,
                        help="Replay a saved episode manifest instead of sampling fresh episodes")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        plot_results(args.output_dir)
    else:
        run(args)


if __name__ == "__main__":
    main()
