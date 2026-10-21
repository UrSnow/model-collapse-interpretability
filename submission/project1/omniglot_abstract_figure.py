"""Three-panel abstract figure from the final-evaluation and mechanism CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# Validated categorical slots (light print). Aqua and yellow are below 3:1
# contrast, so every series also has a distinct marker and a direct label.
STYLE = {
    "clean": ("#2a78d6", "o", "clean"),
    "feedback": ("#eb6834", "s", "100% feedback"),
    "oracle": ("#1baf7a", "^", "oracle"),
    "partial": ("#eda100", "D", "90% feedback"),
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def series(rows: list[dict], family: str, condition: str, metric: str,
           group: str = "condition") -> tuple[list[int], list[float], list[float]]:
    selected = sorted((row for row in rows if row["family"] == family
                       and row[group] == condition), key=lambda row: int(row["generation"]))
    return ([int(row["generation"]) for row in selected],
            [float(row[f"{metric}_mean"]) for row in selected],
            [float(row[f"{metric}_sd"]) for row in selected])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-dir", type=Path, required=True)
    parser.add_argument("--mechanism-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=float, default=7.0, help="Figure width in inches")
    parser.add_argument("--height", type=float, default=2.1, help="Figure height in inches")
    args = parser.parse_args()
    final = read(args.final_dir / "aggregate.csv")
    mechanism = read(args.mechanism_dir / "aggregate.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                         "font.family": "sans-serif", "pdf.fonttype": 42})
    figure, axes = plt.subplots(1, 3, figsize=(args.width, args.height),
                                constrained_layout=True)
    panels = (
        (axes[0], final, "first_label_accuracy", "(a) Held-out accuracy",
         "Accuracy", (0.895, 1.0)),
        (axes[1], mechanism, "candidate_induction_attention",
         "(b) Induction attention", "Attention weight", (0.74, 1.0)),
        (axes[2], mechanism, "candidate_pair_patch_restoration",
         "(c) Head-pair patching", "Normalized restoration", (0.0, 0.9)),
    )
    for axis, rows, metric, title, ylabel, limits in panels:
        if metric == "first_label_accuracy":
            for condition, dash in (("clean", (0, (1, 1.5))), ("feedback", (0, (4, 2)))):
                xs, ys, _ = series(rows, "base", condition, "accuracy")
                axis.plot(xs, ys, color=MUTED, lw=1.0, ls=dash, zorder=2,
                          label=f"base, {'clean' if condition == 'clean' else '100% fb'}")
        else:
            xs, ys, _ = series(rows, "base", "feedback", metric)
            axis.plot(xs, ys, color=MUTED, lw=1.0, ls=(0, (4, 2)), zorder=2,
                      label="base, 100% fb")
        for condition, (color, marker, label) in STYLE.items():
            xs, ys, sd = series(rows, "extended", condition, metric)
            axis.fill_between(xs, [y - s for y, s in zip(ys, sd)],
                              [y + s for y, s in zip(ys, sd)], color=color, alpha=0.13,
                              lw=0, zorder=1)
            axis.plot(xs, ys, color=color, lw=1.4, marker=marker, ms=3.2,
                      mec="white", mew=0.5, zorder=3, label=label)
        axis.set(title=title, xlabel="Generation", ylabel=ylabel, ylim=limits,
                 xlim=(-0.3, 9.3), xticks=range(0, 10, 3))
        axis.grid(color=GRID, lw=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(name) for name in ("clean", "oracle", "90% feedback",
                                             "100% feedback", "base, clean", "base, 100% fb")]
    figure.legend([handles[i] for i in order], [labels[i] for i in order], loc="outside lower center",
                  ncol=6, frameon=False, fontsize=6.3, handlelength=2.2, columnspacing=1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".pdf"))
    figure.savefig(args.output.with_suffix(".png"), dpi=300)
    plt.close(figure)
    print(f"Wrote {args.output.with_suffix('.pdf')}", flush=True)


if __name__ == "__main__":
    main()
