"""Recursive training for label -> next symbol -> corresponding label."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from .collapse import fixed_split, write_json
from .data import SymbolTaskConfig
from .diagnostics import induction_scores, previous_token_scores
from .model import InductionTransformer, TransformerConfig
from .train import set_seed


def oracle_split(task: SymbolTaskConfig, size: int, seed: int, name: str,
                 exclude: set[tuple[int, ...]] | None = None) -> dict[str, torch.Tensor]:
    base = fixed_split(task, size, seed, name, exclude)
    rng = torch.Generator().manual_seed(seed + 50000)
    choice = torch.randint(task.context_pairs, (size,), generator=rng)
    row = torch.arange(size)
    return {
        "input_ids": base.input_ids,
        "first_labels": base.targets,
        "next_symbols": base.input_ids[row, 2 * choice],
        "second_labels": base.input_ids[row, 2 * choice + 1] - task.num_symbols,
        "query_context_index": base.query_context_index,
        "matching_label_position": base.matching_label_position,
    }


def target_context_index(prompt: torch.Tensor, symbols: torch.Tensor,
                         context_pairs: int) -> torch.Tensor:
    choices = prompt[:, 0:2 * context_pairs:2]
    matches = choices == symbols[:, None]
    if not bool(matches.any(dim=1).all()):
        raise ValueError("next symbol must occur in the context")
    return matches.long().argmax(dim=1)


def label_for_symbol(prompt: torch.Tensor, symbols: torch.Tensor,
                     task: SymbolTaskConfig) -> torch.Tensor:
    choice = target_context_index(prompt, symbols, task.context_pairs)
    return prompt.gather(1, (2 * choice + 1)[:, None]).squeeze(1) - task.num_symbols


def sequence_logits(model: InductionTransformer, prompt: torch.Tensor,
                    first_labels: torch.Tensor, next_symbols: torch.Tensor,
                    task: SymbolTaskConfig,
                    ablate_heads: dict[int, tuple[int, ...]] | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence = torch.cat((prompt, (first_labels + task.num_symbols)[:, None],
                          next_symbols[:, None]), dim=1)
    logits = model(sequence, all_positions=True, ablate_heads=ablate_heads)
    first = logits[:, task.sequence_length - 1, task.num_symbols:]
    symbol = logits[:, task.sequence_length, :task.num_symbols]
    second = logits[:, task.sequence_length + 1, task.num_symbols:]
    return first, symbol, second


def context_symbol_logits(logits: torch.Tensor, prompt: torch.Tensor,
                          task: SymbolTaskConfig) -> torch.Tensor:
    return logits.gather(1, prompt[:, 0:2 * task.context_pairs:2])


def training_loss(model: InductionTransformer, prompt: torch.Tensor,
                  first_labels: torch.Tensor, next_symbols: torch.Tensor,
                  second_labels: torch.Tensor, task: SymbolTaskConfig,
                  synthetic_mask: torch.Tensor | None = None) -> torch.Tensor:
    first, symbol, second = sequence_logits(model, prompt, first_labels,
                                             next_symbols, task)
    choices = target_context_index(prompt, next_symbols, task.context_pairs)
    symbol = context_symbol_logits(symbol, prompt, task)
    log_prob = F.log_softmax(symbol, dim=-1)
    clean_loss = -log_prob.mean(dim=-1)
    sampled_loss = -log_prob.gather(1, choices[:, None]).squeeze(1)
    if synthetic_mask is None:
        synthetic_mask = torch.zeros_like(choices, dtype=torch.bool)
    symbol_loss = torch.where(synthetic_mask, sampled_loss, clean_loss).mean()
    return (F.cross_entropy(first, first_labels) + symbol_loss
            + F.cross_entropy(second, second_labels)) / 3


@torch.no_grad()
def evaluate(model: InductionTransformer, split: dict[str, torch.Tensor],
             task: SymbolTaskConfig) -> dict[str, float]:
    model.eval()
    totals = {key: 0.0 for key in ("first_label_nll", "next_symbol_nll",
                                      "second_label_nll", "first_label_correct",
                                      "second_label_correct", "symbol_tv",
                                      "symbol_entropy")}
    size = len(split["first_labels"])
    for start in range(0, size, 256):
        stop = start + 256
        prompt = split["input_ids"][start:stop]
        first_labels = split["first_labels"][start:stop]
        next_symbols = split["next_symbols"][start:stop]
        second_labels = split["second_labels"][start:stop]
        first, symbol, second = sequence_logits(model, prompt, first_labels,
                                                 next_symbols, task)
        symbol = context_symbol_logits(symbol, prompt, task)
        log_prob = F.log_softmax(symbol, dim=-1)
        probabilities = log_prob.exp()
        totals["first_label_nll"] += F.cross_entropy(first, first_labels,
                                                        reduction="sum").item()
        totals["next_symbol_nll"] += -log_prob.mean(dim=-1).sum().item()
        totals["second_label_nll"] += F.cross_entropy(second, second_labels,
                                                         reduction="sum").item()
        totals["first_label_correct"] += (first.argmax(-1) == first_labels).sum().item()
        totals["second_label_correct"] += (second.argmax(-1) == second_labels).sum().item()
        totals["symbol_tv"] += (probabilities - 1 / task.context_pairs).abs().sum(-1).sum().item() / 2
        totals["symbol_entropy"] += -(probabilities * log_prob).sum(-1).sum().item()
    result = {key: value / size for key, value in totals.items()}
    result["first_label_accuracy"] = result.pop("first_label_correct")
    result["second_label_accuracy"] = result.pop("second_label_correct")
    result["loss"] = (result["first_label_nll"] + result["next_symbol_nll"]
                      + result["second_label_nll"]) / 3
    result["perplexity"] = math.exp(result["loss"])
    return result


@torch.no_grad()
def sample_teacher(model: InductionTransformer, prompt: torch.Tensor,
                   task: SymbolTaskConfig, temperature: float,
                   seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    rng = torch.Generator().manual_seed(seed)
    first_rows, symbol_rows, second_rows = [], [], []
    for chunk in prompt.split(256):
        first_logits = model(chunk)[:, task.num_symbols:] / temperature
        first = torch.multinomial(first_logits.softmax(-1), 1, generator=rng).squeeze(1)
        prefix = torch.cat((chunk, (first + task.num_symbols)[:, None]), dim=1)
        symbol_logits = model(prefix)[:, :task.num_symbols] / temperature
        choices = torch.multinomial(
            context_symbol_logits(symbol_logits, chunk, task).softmax(-1),
            1, generator=rng).squeeze(1)
        symbol = chunk.gather(1, (2 * choices)[:, None]).squeeze(1)
        second_prefix = torch.cat((prefix, symbol[:, None]), dim=1)
        second_logits = model(second_prefix)[:, task.num_symbols:] / temperature
        second = torch.multinomial(second_logits.softmax(-1), 1, generator=rng).squeeze(1)
        first_rows.append(first)
        symbol_rows.append(symbol)
        second_rows.append(second)
    return torch.cat(first_rows), torch.cat(symbol_rows), torch.cat(second_rows)


def replacement_targets(model: InductionTransformer, split: dict[str, torch.Tensor],
                        task: SymbolTaskConfig, fraction: float, temperature: float,
                        seed: int) -> dict[str, torch.Tensor]:
    size = len(split["first_labels"])
    rng = torch.Generator().manual_seed(seed)
    mask = torch.zeros(size, dtype=torch.bool)
    mask[torch.randperm(size, generator=rng)[:round(fraction * size)]] = True
    result = {key: split[key].clone() for key in
              ("first_labels", "next_symbols", "second_labels")}
    result["synthetic_mask"] = mask
    result["resampled_mask"] = torch.zeros(size, dtype=torch.bool)
    if bool(mask.any()):
        sampled = sample_teacher(model, split["input_ids"][mask], task,
                                 temperature, seed + 1)
        for key, values in zip(("first_labels", "next_symbols", "second_labels"),
                               sampled):
            result[key][mask] = values
    return result


def resampled_oracle_targets(split: dict[str, torch.Tensor], task: SymbolTaskConfig,
                             fraction: float, seed: int) -> dict[str, torch.Tensor]:
    size = len(split["first_labels"])
    rng = torch.Generator().manual_seed(seed)
    mask = torch.zeros(size, dtype=torch.bool)
    mask[torch.randperm(size, generator=rng)[:round(fraction * size)]] = True
    targets = {key: split[key].clone() for key in
               ("first_labels", "next_symbols", "second_labels")}
    targets["synthetic_mask"] = torch.zeros(size, dtype=torch.bool)
    targets["resampled_mask"] = mask
    if bool(mask.any()):
        choice = torch.randint(task.context_pairs, (int(mask.sum()),), generator=rng)
        prompt = split["input_ids"][mask]
        symbols = prompt.gather(1, (2 * choice)[:, None]).squeeze(1)
        targets["next_symbols"][mask] = symbols
        targets["second_labels"][mask] = label_for_symbol(prompt, symbols, task)
    return targets


def distribution_stats(symbols: torch.Tensor, reference: torch.Tensor,
                       num_symbols: int) -> dict[str, float | int]:
    frequencies = torch.bincount(symbols, minlength=num_symbols).float()
    probabilities = frequencies / frequencies.sum()
    ref = torch.bincount(reference, minlength=num_symbols).float()
    ref /= ref.sum()
    nonzero = probabilities > 0
    return {
        "coverage": int(nonzero.sum().item()),
        "entropy": float(-(probabilities[nonzero] * probabilities[nonzero].log()).sum().item()),
        "dominant_symbol_frequency": float(probabilities.max().item()),
        "total_variation_from_oracle": float((probabilities - ref).abs().sum().item() / 2),
    }


@torch.no_grad()
def attention_diagnostic(model: InductionTransformer, split: dict[str, torch.Tensor],
                         task: SymbolTaskConfig) -> dict:
    model.eval()
    prompt = split["input_ids"][:256]
    logits, maps = model(prompt, return_attention=True)
    previous = previous_token_scores(maps[0])
    induction = induction_scores(maps[1], split["matching_label_position"][:256])
    return {
        "first_label_accuracy": float((logits[:, task.num_symbols:].argmax(-1)
                                        == split["first_labels"][:256]).float().mean().item()),
        "previous_token_scores": previous.tolist(),
        "induction_scores": induction.tolist(),
    }


def run(args: argparse.Namespace) -> None:
    for field in ("generations", "steps", "train_size", "eval_size", "batch_size", "eval_every"):
        if getattr(args, field) < 1:
            raise ValueError(f"{field} must be positive")
    if not 0 <= args.synthetic_fraction <= 1 or not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("Invalid synthetic fraction or temperature")
    if args.feedback_source not in ("model", "oracle"):
        raise ValueError("Unknown feedback source")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    task = SymbolTaskConfig()
    config = TransformerConfig(task.vocab_size, task.num_labels,
                               task.sequence_length + 2, output_vocab_size=task.vocab_size)
    splits = {name: oracle_split(task, args.train_size if name == "train" else args.eval_size,
                                 args.seed + offset, name)
              for offset, name in enumerate(("train", "validation", "test", "diagnostic"), 101)}
    write_json(args.output_dir / "protocol.json", {
        **vars(args), "output_dir": str(args.output_dir), "task": asdict(task),
        "model": asdict(config), "torch_version": str(torch.__version__),
        "symbol_policy": "uniform among context symbols; constrained sampling",
        "clean_symbol_objective": "expected cross-entropy under the uniform conditional",
        "oracle_resample_objective": args.oracle_resample_objective,
        "feedback_policy": "replace complete three-token target on selected rows",
        "test_policy": "evaluate selected checkpoints after all generations train",
    })
    torch.save(splits, args.output_dir / "splits.pt")
    previous = None
    summaries = []
    train = splits["train"]
    for generation in range(args.generations):
        folder = args.output_dir / f"generation-{generation:03d}"
        folder.mkdir()
        sampling_seed = args.seed + 10000 + generation
        targets = {key: train[key].clone() for key in
                   ("first_labels", "next_symbols", "second_labels")}
        targets["synthetic_mask"] = torch.zeros(args.train_size, dtype=torch.bool)
        targets["resampled_mask"] = torch.zeros(args.train_size, dtype=torch.bool)
        if previous is not None:
            if args.feedback_source == "model":
                targets = replacement_targets(previous, train, task, args.synthetic_fraction,
                                              args.temperature, sampling_seed)
            else:
                targets = resampled_oracle_targets(train, task, args.synthetic_fraction,
                                                   sampling_seed)
        torch.save(targets, folder / "training_targets.pt")
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
            sampled_mask = targets["synthetic_mask"][indices]
            if args.oracle_resample_objective == "sampled":
                sampled_mask = sampled_mask | targets["resampled_mask"][indices]
            loss = training_loss(model, train["input_ids"][indices],
                                 targets["first_labels"][indices],
                                 targets["next_symbols"][indices],
                                 targets["second_labels"][indices], task,
                                 sampled_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                validation = evaluate(model, splits["validation"], task)
                row = {"step": step, "train_batch_loss": loss.item(),
                       "validation": validation}
                history.append(row)
                print({"generation": generation, **row}, flush=True)
                if validation["loss"] < best_loss:
                    best_loss = validation["loss"]
                    torch.save({"model_state": model.state_dict(),
                                "model_config": asdict(config), "task_config": asdict(task),
                                "step": step, "validation": validation},
                               folder / "best_model.pt")
        checkpoint = torch.load(folder / "best_model.pt", weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        first_errors = int((targets["first_labels"] != train["first_labels"]).sum())
        symbol_errors = int((targets["next_symbols"] != train["next_symbols"]).sum())
        second_consistency_errors = int((targets["second_labels"] !=
                                         label_for_symbol(train["input_ids"],
                                                          targets["next_symbols"], task)).sum())
        summary = {
            "generation": generation, "parent_generation": generation - 1 if generation else None,
            "initialization_seed": init_seed, "sampling_seed": sampling_seed,
            "synthetic_count": int(targets["synthetic_mask"].sum()),
            "resampled_count": int(targets["resampled_mask"].sum()),
            "first_label_errors": first_errors, "next_symbol_differences": symbol_errors,
            "second_label_inconsistencies": second_consistency_errors,
            "symbol_distribution": distribution_stats(targets["next_symbols"],
                                                       train["next_symbols"], task.num_symbols),
            "best_step": checkpoint["step"], "validation": checkpoint["validation"],
            "attention": attention_diagnostic(model, splits["diagnostic"], task),
        }
        write_json(folder / "history.json", history)
        write_json(folder / "summary.json", summary)
        summaries.append(summary)
        previous = model
    for summary in summaries:
        folder = args.output_dir / f"generation-{summary['generation']:03d}"
        checkpoint = torch.load(folder / "best_model.pt", weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        summary["test"] = evaluate(model, splits["test"], task)
        write_json(folder / "summary.json", summary)
    write_json(args.output_dir / "summary.json", summaries)
    print(f"Completed extended chain: {args.output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument("--eval-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--synthetic-fraction", type=float, default=1.0)
    parser.add_argument("--feedback-source", choices=("model", "oracle"), default="model")
    parser.add_argument("--oracle-resample-objective", choices=("expected", "sampled"),
                        default="sampled")
    parser.add_argument("--temperature", type=float, default=1.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
