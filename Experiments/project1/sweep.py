"""Run the Step 1 baseline over several initialization and data seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/step1-multiseed"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    return seeds


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        run_dir = args.output_dir / f"seed-{seed}"
        summary = run_dir / "summary.json"
        if summary.exists() and not args.force:
            print(f"Skipping completed seed {seed}: {summary}", flush=True)
            continue

        command = [
            sys.executable,
            "-m",
            "project1.train",
            "--steps",
            str(args.steps),
            "--batch-size",
            str(args.batch_size),
            "--eval-every",
            str(args.eval_every),
            "--eval-batches",
            str(args.eval_batches),
            "--device",
            args.device,
            "--seed",
            str(seed),
            "--output-dir",
            str(run_dir),
        ]
        print(f"Running seed {seed}", flush=True)
        subprocess.run(command, check=True)

    print(f"Sweep complete: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

