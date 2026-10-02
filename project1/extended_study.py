"""Run matched extended-task chains across conditions and seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

from .extended import run


CONDITIONS = {
    "clean": (0.0, "model"),
    "mixed": (0.9, "model"),
    "synthetic": (1.0, "model"),
    "resampled": (1.0, "oracle"),
    "resampled_sampled": (1.0, "oracle"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/extended-study"))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    names = args.conditions.split(",")
    if len(set(seeds)) != len(seeds) or len(set(names)) != len(names):
        parser.error("Duplicate seeds or conditions")
    if any(name not in CONDITIONS for name in names):
        parser.error(f"Conditions must be drawn from {','.join(CONDITIONS)}")
    for seed in seeds:
        for name in names:
            fraction, source = CONDITIONS[name]
            output = args.output_dir / f"seed-{seed}" / name
            print(f"Starting seed {seed}, condition {name}: {output}", flush=True)
            run(argparse.Namespace(output_dir=output, seed=seed,
                                   generations=args.generations, steps=args.steps,
                                   train_size=args.train_size, eval_size=args.eval_size,
                                   batch_size=args.batch_size, eval_every=args.eval_every,
                                   temperature=args.temperature,
                                   synthetic_fraction=fraction, feedback_source=source,
                                   oracle_resample_objective=("sampled" if name == "resampled_sampled"
                                                              else "expected")))


if __name__ == "__main__":
    main()
