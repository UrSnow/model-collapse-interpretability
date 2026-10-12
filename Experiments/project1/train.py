"""Train and evaluate the Step 1 induction-head baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import SymbolTaskConfig, SymbolTaskGenerator
from .diagnostics import attention_summary, head_ablation_accuracy_drops
from .model import InductionTransformer, TransformerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-symbols", type=int, default=50)
    parser.add_argument("--num-labels", type=int, default=5)
    parser.add_argument("--context-pairs", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


@torch.no_grad()
def evaluate(
    model: InductionTransformer,
    task: SymbolTaskConfig,
    *,
    seed: int,
    batch_size: int,
    batches: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    stream = SymbolTaskGenerator(task, seed)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for _ in range(batches):
        batch = stream.sample(batch_size).to(device)
        logits = model(batch.input_ids)
        total_loss += F.cross_entropy(logits, batch.targets, reduction="sum").item()
        total_correct += (logits.argmax(dim=-1) == batch.targets).sum().item()
        total_examples += batch_size
    return {
        "loss": total_loss / total_examples,
        "perplexity": math.exp(total_loss / total_examples),
        "accuracy": total_correct / total_examples,
    }


def save_history(path: Path, history: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.eval_every < 1 or args.eval_batches < 1:
        raise ValueError("steps, eval_every, and eval_batches must be positive")

    set_seed(args.seed)
    device = choose_device(args.device)
    task = SymbolTaskConfig(
        num_symbols=args.num_symbols,
        num_labels=args.num_labels,
        context_pairs=args.context_pairs,
    )
    model_config = TransformerConfig(
        vocab_size=task.vocab_size,
        num_labels=task.num_labels,
        max_sequence_length=task.sequence_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    )
    model = InductionTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or Path("runs") / "step1" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": vars(args) | {"output_dir": str(output_dir)},
        "task": asdict(task),
        "model": model_config.to_dict(),
        "device": str(device),
        "train_seed": args.seed + 101,
        "validation_seed": args.seed + 202,
        "test_seed": args.seed + 303,
        "random_accuracy": 1.0 / task.num_labels,
        "label_set_accuracy": 1.0 / task.context_pairs,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    train_stream = SymbolTaskGenerator(task, config["train_seed"])
    history: list[dict[str, float | int]] = []
    diagnostic_history: list[dict[str, object]] = []
    best_validation_loss = float("inf")
    checkpoint_path = output_dir / "best_model.pt"
    diagnostic_batch = SymbolTaskGenerator(task, config["test_seed"]).sample(
        max(args.batch_size, 512)
    ).to(device)

    for step in range(1, args.steps + 1):
        model.train()
        batch = train_stream.sample(args.batch_size).to(device)
        logits = model(batch.input_ids)
        loss = F.cross_entropy(logits, batch.targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            train_accuracy = (
                (logits.argmax(dim=-1) == batch.targets).float().mean().item()
            )
            validation = evaluate(
                model,
                task,
                seed=config["validation_seed"],
                batch_size=args.batch_size,
                batches=args.eval_batches,
                device=device,
            )
            diagnostic = attention_summary(model, diagnostic_batch)
            row = {
                "step": step,
                "train_loss": float(loss.item()),
                "train_perplexity": math.exp(float(loss.item())),
                "train_accuracy": train_accuracy,
                "validation_loss": validation["loss"],
                "validation_perplexity": validation["perplexity"],
                "validation_accuracy": validation["accuracy"],
                "best_previous_token_score": max(
                    diagnostic["previous_token_scores"]
                ),
                "best_induction_score": max(diagnostic["induction_scores"]),
            }
            history.append(row)
            diagnostic_history.append({"step": step} | diagnostic)
            print(json.dumps(row), flush=True)

            if validation["loss"] < best_validation_loss:
                best_validation_loss = validation["loss"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_config": model_config.to_dict(),
                        "task_config": asdict(task),
                        "step": step,
                        "validation": validation,
                    },
                    checkpoint_path,
                )

    save_history(output_dir / "history.csv", history)
    (output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostic_history, indent=2), encoding="utf-8"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test = evaluate(
        model,
        task,
        seed=config["test_seed"],
        batch_size=args.batch_size,
        batches=max(args.eval_batches, 50),
        device=device,
    )
    summary = {
        "best_step": checkpoint["step"],
        "validation": checkpoint["validation"],
        "test": test,
        "baselines": {
            "random_accuracy": config["random_accuracy"],
            "label_set_accuracy": config["label_set_accuracy"],
        },
        "attention": attention_summary(model, diagnostic_batch),
        "head_ablation": head_ablation_accuracy_drops(model, diagnostic_batch),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Artifacts written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
