"""Recursive base-task study on Singh et al.'s Omniglot feature episodes."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from .omniglot_data import FEATURE_SHA256, OmniglotBatch, OmniglotTaskConfig, OmniglotTaskGenerator, load_features
from .omniglot_model import OmniglotInductionTransformer, OmniglotModelConfig
from .train import set_seed


@torch.no_grad()
def evaluate(model: OmniglotInductionTransformer, batch: OmniglotBatch) -> dict[str, float]:
    model.eval()
    loss, correct = 0.0, 0
    for start in range(0, len(batch), 256):
        stop = start + 256
        logits = model(batch.examples[start:stop], batch.context_labels[start:stop])
        targets = batch.targets[start:stop]
        loss += F.cross_entropy(logits, targets, reduction="sum").item()
        correct += (logits.argmax(-1) == targets).sum().item()
    loss /= len(batch)
    return {"loss": loss, "perplexity": math.exp(loss), "accuracy": correct / len(batch)}


@torch.no_grad()
def attention_diagnostic(model: OmniglotInductionTransformer,
                         batch: OmniglotBatch) -> dict[str, list[float]]:
    model.eval()
    sample = batch.subset(slice(0, min(128, len(batch))))
    _, maps = model(sample.examples, sample.context_labels, return_attention=True)
    previous = torch.stack((maps[0][:, :, 1, 0], maps[0][:, :, 3, 2]), dim=-1).mean((0, 2))
    row = torch.arange(len(sample))
    induction = maps[1][row, :, 4, sample.matching_label_position].mean(0)
    return {"first_layer_previous_token": previous.tolist(),
            "second_layer_induction": induction.tolist()}


@torch.no_grad()
def replacement_labels(model: OmniglotInductionTransformer, batch: OmniglotBatch,
                       fraction: float, temperature: float, seed: int
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 <= fraction <= 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid fraction or temperature")
    rng = torch.Generator().manual_seed(seed)
    mask = torch.zeros(len(batch), dtype=torch.bool)
    mask[torch.randperm(len(batch), generator=rng)[:round(fraction * len(batch))]] = True
    labels = batch.targets.clone()
    model.eval()
    for indices in mask.nonzero().flatten().split(256):
        if len(indices):
            logits = model(batch.examples[indices], batch.context_labels[indices])
            probabilities = (logits / temperature).softmax(-1)
            labels[indices] = torch.multinomial(probabilities, 1, generator=rng).squeeze(1)
    return labels, mask


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    for name in ("generations", "steps", "train_size", "eval_size", "batch_size", "eval_every"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not 0 <= args.synthetic_fraction <= 1 or args.learning_rate <= 0:
        raise ValueError("Invalid training configuration")
    features = load_features(args.feature_file)
    task = OmniglotTaskConfig()
    config = OmniglotModelConfig(feature_dim=features.shape[-1], num_labels=task.num_labels)
    splits = {name: OmniglotTaskGenerator(features, task, args.seed + offset).sample_unique(
        args.train_size if name == "train" else args.eval_size, name)
        for offset, name in enumerate(("train", "validation", "test", "diagnostic"), 101)}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "protocol.json", {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "task": asdict(task), "model": asdict(config), "torch_version": str(torch.__version__),
        "data_source": "Singh et al. preprocessed Omniglot ResNet18 features",
        "feature_sha256": FEATURE_SHA256, "optimizer": "Adam without weight decay",
        "split_policy": "train and validation use disjoint relabeling pairs; test and diagnostics use disjoint classes",
        "generation_policy": "fixed episodes, sampled target replacement, fresh model and optimizer",
        "test_policy": "test only after all validation-selected checkpoints are fixed",
    })
    torch.save({name: asdict(batch) for name, batch in splits.items()}, args.output_dir / "splits.pt")
    train = splits["train"]
    previous = None
    summaries = []
    for generation in range(args.generations):
        folder = args.output_dir / f"generation-{generation:03d}"
        folder.mkdir()
        labels = train.targets.clone()
        mask = torch.zeros(len(train), dtype=torch.bool)
        if previous is not None:
            labels, mask = replacement_labels(previous, train, args.synthetic_fraction,
                                               args.temperature, args.seed + 10000 + generation)
        torch.save({"targets": labels, "synthetic_mask": mask}, folder / "training_labels.pt")
        init_seed = args.seed + 20000 + generation
        set_seed(init_seed)
        model = OmniglotInductionTransformer(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        batch_rng = torch.Generator().manual_seed(args.seed + 30000 + generation)
        best_loss = float("inf")
        history = []
        for step in range(1, args.steps + 1):
            model.train()
            indices = torch.randint(len(train), (args.batch_size,), generator=batch_rng)
            logits = model(train.examples[indices], train.context_labels[indices])
            loss = F.cross_entropy(logits, labels[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                validation = evaluate(model, splits["validation"])
                row = {"generation": generation, "step": step,
                       "training_batch_loss": loss.item(), "validation": validation}
                history.append(row)
                print(json.dumps(row), flush=True)
                if validation["loss"] < best_loss:
                    best_loss = validation["loss"]
                    torch.save({"model_state": model.state_dict(), "model_config": asdict(config),
                                "task_config": asdict(task), "step": step,
                                "validation": validation}, folder / "best_model.pt")
        checkpoint = torch.load(folder / "best_model.pt", weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        summary = {
            "generation": generation, "parent_generation": generation - 1 if generation else None,
            "initialization_seed": init_seed, "synthetic_count": int(mask.sum()),
            "label_error_rate": (labels != train.targets).float().mean().item(),
            "best_step": checkpoint["step"], "validation": checkpoint["validation"],
            "attention": attention_diagnostic(model, splits["diagnostic"]),
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
    print(f"Completed Omniglot base chain: {args.output_dir}", flush=True)


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
    run(parser.parse_args())


if __name__ == "__main__":
    main()
