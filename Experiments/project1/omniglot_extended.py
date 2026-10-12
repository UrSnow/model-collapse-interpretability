"""Recursive three-output Omniglot experiment with context-feature pointers."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from .omniglot_base import attention_diagnostic, write_json
from .omniglot_data import FEATURE_SHA256, OmniglotTaskConfig, OmniglotTaskGenerator, load_features
from .omniglot_model import OmniglotExtendedTransformer, OmniglotModelConfig
from .train import set_seed


def oracle_split(generator: OmniglotTaskGenerator, size: int, name: str, seed: int
                 ) -> dict[str, torch.Tensor]:
    batch = generator.sample_unique(size, name)
    choice = torch.randint(2, (size,), generator=torch.Generator().manual_seed(seed))
    row = torch.arange(size)
    return {**asdict(batch), "next_indices": choice,
            "second_labels": batch.context_labels[row, choice],
            "next_class_ids": batch.class_ids[row, choice]}


def target_labels(split: dict[str, torch.Tensor], indices: torch.Tensor) -> torch.Tensor:
    return split["context_labels"].gather(1, indices[:, None]).squeeze(1)


def sequence_logits(model: OmniglotExtendedTransformer, split: dict[str, torch.Tensor],
                    first: torch.Tensor, indices: torch.Tensor,
                    ablate_heads: dict[int, tuple[int, ...]] | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return model.forward_extended(split["examples"], split["context_labels"],
                                  first, indices, ablate_heads=ablate_heads)


def training_loss(model: OmniglotExtendedTransformer, split: dict[str, torch.Tensor],
                  targets: dict[str, torch.Tensor],
                  oracle_resample_objective: str = "sampled") -> torch.Tensor:
    first, pointer, second = sequence_logits(model, split, targets["first_labels"],
                                             targets["next_indices"])
    clean_pointer_loss = -F.log_softmax(pointer, dim=-1).mean(-1)
    sampled_pointer_loss = F.cross_entropy(pointer, targets["next_indices"], reduction="none")
    sampled_mask = targets["synthetic_mask"]
    if oracle_resample_objective == "sampled":
        sampled_mask = sampled_mask | targets.get("resampled_mask", torch.zeros_like(sampled_mask))
    elif oracle_resample_objective != "expected":
        raise ValueError("Unknown oracle resample objective")
    pointer_loss = torch.where(sampled_mask, sampled_pointer_loss,
                               clean_pointer_loss).mean()
    return (F.cross_entropy(first, targets["first_labels"]) + pointer_loss
            + F.cross_entropy(second, targets["second_labels"])) / 3


@torch.no_grad()
def evaluate(model: OmniglotExtendedTransformer,
             split: dict[str, torch.Tensor]) -> dict[str, float]:
    model.eval()
    totals = {name: 0.0 for name in ("first_label_nll", "next_feature_nll",
                                      "second_label_nll", "first_label_correct",
                                      "second_label_correct", "pointer_tv",
                                      "pointer_entropy", "greedy_joint_correct")}
    size = len(split["targets"])
    for start in range(0, size, 256):
        part = {key: value[start:start + 256] for key, value in split.items()}
        first, pointer, second = sequence_logits(model, part, part["targets"],
                                                 part["next_indices"])
        probabilities = pointer.softmax(-1)
        log_prob = F.log_softmax(pointer, dim=-1)
        totals["first_label_nll"] += F.cross_entropy(first, part["targets"], reduction="sum").item()
        totals["next_feature_nll"] += -log_prob.mean(-1).sum().item()
        totals["second_label_nll"] += F.cross_entropy(second, part["second_labels"], reduction="sum").item()
        totals["first_label_correct"] += (first.argmax(-1) == part["targets"]).sum().item()
        totals["second_label_correct"] += (second.argmax(-1) == part["second_labels"]).sum().item()
        totals["pointer_tv"] += (probabilities - 0.5).abs().sum(-1).sum().item() / 2
        totals["pointer_entropy"] += -(probabilities * log_prob).sum(-1).sum().item()
        first_greedy = first.argmax(-1)
        pointer_greedy = model.pointer_logits(part["examples"], part["context_labels"],
                                               first_greedy).argmax(-1)
        _, _, last = sequence_logits(model, part, first_greedy, pointer_greedy)
        expected_last = target_labels(part, pointer_greedy)
        totals["greedy_joint_correct"] += ((first_greedy == part["targets"])
                                           & (last.argmax(-1) == expected_last)).sum().item()
    result = {name: value / size for name, value in totals.items()}
    result["first_label_accuracy"] = result.pop("first_label_correct")
    result["second_label_accuracy"] = result.pop("second_label_correct")
    result["greedy_joint_accuracy"] = result.pop("greedy_joint_correct")
    result["loss"] = (result["first_label_nll"] + result["next_feature_nll"]
                      + result["second_label_nll"]) / 3
    result["perplexity"] = math.exp(result["loss"])
    return result


@torch.no_grad()
def sample_teacher(model: OmniglotExtendedTransformer,
                   split: dict[str, torch.Tensor], temperature: float,
                   seed: int) -> dict[str, torch.Tensor]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    model.eval()
    rng = torch.Generator().manual_seed(seed)
    output = {name: [] for name in ("first_labels", "next_indices", "second_labels")}
    for start in range(0, len(split["targets"]), 256):
        part = {key: value[start:start + 256] for key, value in split.items()}
        first_logits = model(part["examples"], part["context_labels"])
        first = torch.multinomial((first_logits / temperature).softmax(-1), 1,
                                  generator=rng).squeeze(1)
        pointer = model.pointer_logits(part["examples"], part["context_labels"], first)
        indices = torch.multinomial((pointer / temperature).softmax(-1), 1,
                                    generator=rng).squeeze(1)
        _, _, second_logits = sequence_logits(model, part, first, indices)
        second = torch.multinomial((second_logits / temperature).softmax(-1), 1,
                                   generator=rng).squeeze(1)
        for name, values in zip(output, (first, indices, second)):
            output[name].append(values)
    return {name: torch.cat(values) for name, values in output.items()}


def replacement_targets(model: OmniglotExtendedTransformer,
                        split: dict[str, torch.Tensor], fraction: float,
                        temperature: float, seed: int) -> dict[str, torch.Tensor]:
    if not 0 <= fraction <= 1:
        raise ValueError("Synthetic fraction must be in [0, 1]")
    size = len(split["targets"])
    rng = torch.Generator().manual_seed(seed)
    mask = torch.zeros(size, dtype=torch.bool)
    mask[torch.randperm(size, generator=rng)[:round(fraction * size)]] = True
    targets = {"first_labels": split["targets"].clone(),
               "next_indices": split["next_indices"].clone(),
               "second_labels": split["second_labels"].clone(),
               "synthetic_mask": mask,
               "resampled_mask": torch.zeros(size, dtype=torch.bool)}
    if bool(mask.any()):
        selected = {key: value[mask] for key, value in split.items()}
        sampled = sample_teacher(model, selected, temperature, seed + 1)
        for key, value in sampled.items():
            targets[key][mask] = value
    return targets


def resampled_oracle_targets(split: dict[str, torch.Tensor], fraction: float,
                             seed: int) -> dict[str, torch.Tensor]:
    if not 0 <= fraction <= 1:
        raise ValueError("Resampling fraction must be in [0, 1]")
    size = len(split["targets"])
    rng = torch.Generator().manual_seed(seed)
    mask = torch.zeros(size, dtype=torch.bool)
    mask[torch.randperm(size, generator=rng)[:round(fraction * size)]] = True
    targets = {"first_labels": split["targets"].clone(),
               "next_indices": split["next_indices"].clone(),
               "second_labels": split["second_labels"].clone(),
               "synthetic_mask": torch.zeros(size, dtype=torch.bool),
               "resampled_mask": mask}
    if bool(mask.any()):
        indices = torch.randint(2, (int(mask.sum()),), generator=rng)
        targets["next_indices"][mask] = indices
        targets["second_labels"][mask] = target_labels(
            {key: value[mask] for key, value in split.items()}, indices)
    return targets


def distribution_stats(split: dict[str, torch.Tensor],
                       targets: dict[str, torch.Tensor], pool_size: int
                       ) -> dict[str, float | int]:
    rows = torch.arange(len(split["targets"]))
    classes = split["class_ids"][rows, targets["next_indices"]]
    counts = torch.bincount(classes, minlength=pool_size).float()
    probabilities = counts / counts.sum()
    baseline = torch.bincount(split["next_class_ids"], minlength=pool_size).float()
    baseline /= baseline.sum()
    nonzero = probabilities > 0
    return {"coverage": int(nonzero.sum()),
            "entropy": float(-(probabilities[nonzero] * probabilities[nonzero].log()).sum()),
            "dominant_class_frequency": float(probabilities.max()),
            "total_variation_from_fixed_oracle": float((probabilities - baseline).abs().sum() / 2)}


def run(args: argparse.Namespace) -> None:
    for name in ("generations", "steps", "train_size", "eval_size", "batch_size", "eval_every"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not 0 <= args.synthetic_fraction <= 1 or args.learning_rate <= 0:
        raise ValueError("Invalid training configuration")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    if args.feedback_source not in ("model", "oracle"):
        raise ValueError("Unknown feedback source")
    if args.oracle_resample_objective not in ("sampled", "expected"):
        raise ValueError("Unknown oracle resample objective")
    features = load_features(args.feature_file)
    task = OmniglotTaskConfig()
    config = OmniglotModelConfig(feature_dim=features.shape[-1], num_labels=task.num_labels)
    splits = {name: oracle_split(OmniglotTaskGenerator(features, task, args.seed + offset),
                                 args.train_size if name == "train" else args.eval_size,
                                 name, args.seed + 50000 + offset)
              for offset, name in enumerate(("train", "validation", "test", "diagnostic"), 101)}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "protocol.json", {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "task": asdict(task), "model": asdict(config), "torch_version": str(torch.__version__),
        "feature_sha256": FEATURE_SHA256, "optimizer": "Adam without weight decay",
        "next_feature_policy": "pointer to either context Omniglot feature; no fixed class vocabulary",
        "clean_pointer_objective": "expected cross-entropy under a uniform choice of context feature",
        "feedback_policy": "sample all three outputs autoregressively on selected fixed training rows",
        "oracle_resample_policy": "resample valid context pointers and corresponding labels, retaining the clean first label",
        "test_policy": "evaluate test only after validation-selected checkpoints are fixed",
    })
    torch.save(splits, args.output_dir / "splits.pt")
    train = splits["train"]
    previous = None
    summaries = []
    for generation in range(args.generations):
        folder = args.output_dir / f"generation-{generation:03d}"
        folder.mkdir()
        if previous is None:
            targets = {"first_labels": train["targets"].clone(),
                       "next_indices": train["next_indices"].clone(),
                       "second_labels": train["second_labels"].clone(),
                       "synthetic_mask": torch.zeros(len(train["targets"]), dtype=torch.bool),
                       "resampled_mask": torch.zeros(len(train["targets"]), dtype=torch.bool)}
        else:
            if args.feedback_source == "model":
                targets = replacement_targets(previous, train, args.synthetic_fraction,
                                              args.temperature, args.seed + 10000 + generation)
            else:
                targets = resampled_oracle_targets(train, args.synthetic_fraction,
                                                   args.seed + 10000 + generation)
        torch.save(targets, folder / "training_targets.pt")
        init_seed = args.seed + 20000 + generation
        set_seed(init_seed)
        model = OmniglotExtendedTransformer(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        rng = torch.Generator().manual_seed(args.seed + 30000 + generation)
        best_loss = float("inf")
        history = []
        for step in range(1, args.steps + 1):
            model.train()
            indices = torch.randint(len(train["targets"]), (args.batch_size,), generator=rng)
            part = {key: value[indices] for key, value in train.items()}
            selected = {key: value[indices] for key, value in targets.items()}
            loss = training_loss(model, part, selected, args.oracle_resample_objective)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                validation = evaluate(model, splits["validation"])
                row = {"generation": generation, "step": step,
                       "training_batch_loss": loss.item(), "validation": validation}
                history.append(row)
                print(row, flush=True)
                if validation["loss"] < best_loss:
                    best_loss = validation["loss"]
                    torch.save({"model_state": model.state_dict(), "model_config": asdict(config),
                                "task_config": asdict(task), "step": step,
                                "validation": validation}, folder / "best_model.pt")
        checkpoint = torch.load(folder / "best_model.pt", weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        summary = {
            "generation": generation, "parent_generation": generation - 1 if generation else None,
            "initialization_seed": init_seed, "synthetic_count": int(targets["synthetic_mask"].sum()),
            "resampled_count": int(targets["resampled_mask"].sum()),
            "first_label_errors": int((targets["first_labels"] != train["targets"]).sum()),
            "next_feature_differences": int((targets["next_indices"] != train["next_indices"]).sum()),
            "second_label_inconsistencies": int((targets["second_labels"] !=
                                                 target_labels(train, targets["next_indices"])).sum()),
            "class_distribution": distribution_stats(train, targets, task.train_classes),
            "best_step": checkpoint["step"], "validation": checkpoint["validation"],
            "attention": attention_diagnostic(model, _as_batch(splits["diagnostic"])),
        }
        write_json(folder / "history.json", history)
        write_json(folder / "summary.json", summary)
        summaries.append(summary)
        previous = model
    for summary in summaries:
        folder = args.output_dir / f"generation-{summary['generation']:03d}"
        checkpoint = torch.load(folder / "best_model.pt", weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        summary["test"] = evaluate(model, splits["test"])
        write_json(folder / "summary.json", summary)
    write_json(args.output_dir / "summary.json", summaries)
    print(f"Completed Omniglot extended chain: {args.output_dir}", flush=True)


def _as_batch(split: dict[str, torch.Tensor]):
    from .omniglot_data import OmniglotBatch
    return OmniglotBatch(*(split[name] for name in OmniglotBatch.__dataclass_fields__))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-file", type=Path,
                        default=Path("data/omniglot_resnet18_randomized_order_s0.h5"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument("--eval-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--synthetic-fraction", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--feedback-source", choices=("model", "oracle"), default="model")
    parser.add_argument("--oracle-resample-objective", choices=("sampled", "expected"),
                        default="sampled")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
