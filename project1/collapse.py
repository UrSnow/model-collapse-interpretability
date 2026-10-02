"""Step 2 base-task recursive label training with paired clean controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import ICLBatch, SymbolTaskConfig, SymbolTaskGenerator
from .diagnostics import attention_summary
from .model import InductionTransformer, TransformerConfig
from .train import set_seed


def prompt_partition(tokens: list[int]) -> int:
    """Assign identical prompts to the same split regardless of sampling seed."""
    digest = hashlib.sha256(json.dumps(tokens).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % 10


def fixed_split(task: SymbolTaskConfig, size: int, seed: int, split: str,
                exclude: set[tuple[int, ...]] | None = None) -> ICLBatch:
    buckets = {"train": set(range(7)), "validation": {7}, "test": {8}, "diagnostic": {9}}
    allowed = buckets[split]
    stream = SymbolTaskGenerator(task, seed)
    rows, seen = [], set(exclude or ())
    # Bounded rejection sampling prevents impossible requests from hanging.
    for _ in range(max(100, size * 20 // 256)):
        candidates = stream.sample(256)
        for index, tokens in enumerate(candidates.input_ids.tolist()):
            key = tuple(tokens)
            if key in seen or prompt_partition(tokens) not in allowed:
                continue
            seen.add(key)
            rows.append((candidates.input_ids[index], candidates.targets[index],
                         candidates.query_context_index[index],
                         candidates.matching_label_position[index]))
            if len(rows) == size:
                return ICLBatch(*(torch.stack(list(column)) for column in zip(*rows)))
    raise ValueError(f"Could not obtain {size} unique {split} prompts; reduce split size")


@torch.no_grad()
def evaluate_fixed(model: InductionTransformer, batch: ICLBatch) -> dict[str, float]:
    model.eval()
    loss, correct = 0.0, 0
    for start in range(0, len(batch.targets), 256):
        logits = model(batch.input_ids[start:start + 256])
        targets = batch.targets[start:start + 256]
        loss += F.cross_entropy(logits, targets, reduction="sum").item()
        correct += (logits.argmax(-1) == targets).sum().item()
    loss /= len(batch.targets)
    return {"loss": loss, "perplexity": math.exp(loss),
            "accuracy": correct / len(batch.targets)}


@torch.no_grad()
def replacement_labels(model: InductionTransformer, batch: ICLBatch,
                       fraction: float, temperature: float, seed: int
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 <= fraction <= 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("fraction must be in [0,1] and temperature finite and positive")
    rng = torch.Generator().manual_seed(seed)
    mask = torch.zeros(len(batch.targets), dtype=torch.bool)
    mask[torch.randperm(len(mask), generator=rng)[:round(fraction * len(mask))]] = True
    labels = batch.targets.clone()
    model.eval()
    indices = mask.nonzero().flatten()
    for chunk in indices.split(256):
        if len(chunk):
            probabilities = (model(batch.input_ids[chunk]) / temperature).softmax(-1)
            labels[chunk] = torch.multinomial(probabilities, 1, generator=rng).squeeze(1)
    return labels, mask


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    for field in ("generations", "steps", "train_size", "eval_size", "batch_size", "eval_every"):
        if getattr(args, field) < 1:
            raise ValueError(f"{field} must be positive")
    if not 0 <= args.synthetic_fraction <= 1 or not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("Invalid synthetic fraction or temperature")
    # Refuse reuse rather than silently combining incompatible runs.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    task = SymbolTaskConfig()
    config = TransformerConfig(task.vocab_size, task.num_labels, task.sequence_length)
    splits = {name: fixed_split(task, args.train_size if name == "train" else args.eval_size,
                                args.seed + offset, name)
              for offset, name in enumerate(("train", "validation", "test", "diagnostic"), 101)}
    write_json(args.output_dir / "protocol.json", {
        **vars(args), "output_dir": str(args.output_dir), "task": asdict(task),
        "model": asdict(config), "device": "cpu", "torch_version": str(torch.__version__),
        "protocol": "fixed prompts; target replacement; fresh model and optimizer per generation",
        "test_policy": "evaluate all selected checkpoints only after the entire chain is trained",
    })
    torch.save({name: asdict(batch) for name, batch in splits.items()}, args.output_dir / "splits.pt")
    previous = None
    summaries = []
    for generation in range(args.generations):
        folder = args.output_dir / f"generation-{generation:03d}"
        folder.mkdir()
        train = splits["train"]
        labels, mask = train.targets.clone(), torch.zeros(args.train_size, dtype=torch.bool)
        sampling_seed = args.seed + 10000 + generation
        if previous is not None:
            labels, mask = replacement_labels(previous, train, args.synthetic_fraction,
                                               args.temperature, sampling_seed)
        torch.save({"targets": labels, "synthetic_mask": mask}, folder / "training_labels.pt")
        init_seed = args.seed + 20000 + generation
        set_seed(init_seed)
        model = InductionTransformer(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        batch_rng = torch.Generator().manual_seed(args.seed + 30000 + generation)
        best_loss = float("inf")
        history = []
        for step in range(1, args.steps + 1):
            model.train()
            indices = torch.randint(args.train_size, (args.batch_size,), generator=batch_rng)
            loss = F.cross_entropy(model(train.input_ids[indices]), labels[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                validation = evaluate_fixed(model, splits["validation"])
                row = {"generation": generation, "step": step,
                       "training_batch_loss": loss.item(), "validation": validation}
                history.append(row)
                print(json.dumps(row), flush=True)
                if validation["loss"] < best_loss:
                    best_loss = validation["loss"]
                    torch.save({"model_state": model.state_dict(), "model_config": asdict(config),
                                "task_config": asdict(task), "step": step, "validation": validation},
                               folder / "best_model.pt")
        checkpoint = torch.load(folder / "best_model.pt", weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        summary = {
            "generation": generation, "parent_generation": generation - 1 if generation else None,
            "initialization_seed": init_seed, "sampling_seed": sampling_seed,
            "minibatch_seed": args.seed + 30000 + generation,
            "synthetic_count": int(mask.sum()), "dataset_size": args.train_size,
            "label_error_rate": (labels != train.targets).float().mean().item(),
            "best_step": checkpoint["step"], "validation": checkpoint["validation"],
            "attention": attention_summary(model, splits["diagnostic"]),
        }
        write_json(folder / "history.json", history)
        write_json(folder / "summary.json", summary)
        summaries.append(summary)
        previous = model
    for summary in summaries:
        folder = args.output_dir / f"generation-{summary['generation']:03d}"
        checkpoint = torch.load(folder / "best_model.pt", weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        summary["test"] = evaluate_fixed(model, splits["test"])
        write_json(folder / "summary.json", summary)
    write_json(args.output_dir / "summary.json", summaries)
    print(f"Completed base chain: {args.output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=3, help="Includes clean generation 0")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument("--eval-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--synthetic-fraction", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
