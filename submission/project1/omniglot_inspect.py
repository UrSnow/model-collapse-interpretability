"""Attention maps, embedding geometry, and exemplar transfer for frozen checkpoints.

For each validation-selected checkpoint this records, on held-out diagnostic
episodes:

* the mean attention pattern of every head, and the candidate previous-token
  and induction heads chosen by attention score;
* a direct-path OV copy score for the candidate induction head: label
  embedding -> layer-2 value/output -> unembedding, with LayerNorm applied to
  the isolated vector. It asks whether the head maps a context label to the
  same output label; it ignores other residual-stream contributions. The same
  score is also computed for all layer-2 heads weighted by their query
  attention to the matching label;
* mean absolute off-diagonal cosine similarity of the five label embeddings;
* query-exemplar transfer: first-label accuracy when the query feature is a
  different exemplar of the matching class instead of the identical exemplar
  shown in context. All training episodes use exemplar 0, so this measures
  matching beyond exact feature identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import torch
from torch.nn import functional as F

from .omniglot_base import write_json
from .omniglot_curves import chain_folders, load_model
from .omniglot_data import load_features
from .omniglot_study import write_csv


@torch.no_grad()
def attention_maps(model, split: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Mean per-head maps over episodes whose matching context pair is first."""
    keep = split["query_context_index"] == 0
    _, maps = model(split["examples"][keep], split["context_labels"][keep],
                    return_attention=True)
    return tuple(layer.mean(0) for layer in maps)


def candidate_heads(maps: tuple[torch.Tensor, ...]) -> tuple[int, int]:
    previous = (maps[0][:, 1, 0] + maps[0][:, 3, 2]) / 2
    induction = maps[1][:, 4, 1]
    return int(previous.argmax()), int(induction.argmax())


def layer_norm(x: torch.Tensor, norm: torch.nn.LayerNorm) -> torch.Tensor:
    return F.layer_norm(x, norm.normalized_shape, norm.weight, norm.bias, norm.eps)


@torch.no_grad()
def head_ov(model, head: int) -> torch.Tensor:
    """Residual-stream vector each label embedding writes through one layer-2 head."""
    block = model.blocks[1]
    width, size = model.config.d_model, block.head_dim
    value = block.qkv.weight[2 * width + head * size: 2 * width + (head + 1) * size]
    output = block.output.weight[:, head * size:(head + 1) * size]
    return layer_norm(model.label_embedding.weight, block.norm) @ value.T @ output.T


def copy_scores(model, moved: torch.Tensor, prefix: str) -> dict[str, float]:
    logits = model.unembed(layer_norm(moved, model.final_norm))
    labels = len(logits)
    diagonal = logits.diag()
    off = (logits.sum(1) - diagonal) / (labels - 1)
    return {f"{prefix}_accuracy": float((logits.argmax(1) == torch.arange(labels)).float().mean()),
            f"{prefix}_margin": float((diagonal - off).mean())}


@torch.no_grad()
def ov_copy(model, head: int, induction_attention: torch.Tensor) -> dict[str, float]:
    """Copy scores for the candidate head and the attention-weighted layer-2 sum."""
    weighted = sum(weight * head_ov(model, index)
                   for index, weight in enumerate(induction_attention.tolist()))
    return {**copy_scores(model, head_ov(model, head), "ov_copy"),
            **copy_scores(model, weighted, "layer2_ov_copy")}


def label_cosine(model) -> tuple[float, torch.Tensor]:
    embedding = F.normalize(model.label_embedding.weight.detach(), dim=-1)
    cosine = embedding @ embedding.T
    mask = ~torch.eye(len(cosine), dtype=torch.bool)
    return float(cosine[mask].abs().mean()), cosine


@torch.no_grad()
def exemplar_transfer(model, split: dict[str, torch.Tensor], features: torch.Tensor
                      ) -> dict[str, float]:
    query_class = split["class_ids"][:, 2]
    scores = {}
    for exemplar in range(features.shape[1]):
        examples = split["examples"].clone()
        examples[:, 2] = features[query_class, exemplar]
        logits = model(examples, split["context_labels"])
        scores[exemplar] = float((logits.argmax(-1) == split["targets"]).float().mean())
    return {"query_same_exemplar_accuracy": scores[0],
            "query_other_exemplar_accuracy": mean(scores[k] for k in scores if k)}


def inspect(root: Path, extended: bool, features: torch.Tensor
            ) -> tuple[list[dict], dict]:
    rows, artefacts = [], {}
    for (seed, condition), chain in chain_folders(root).items():
        diagnostic = torch.load(chain / "splits.pt", weights_only=True)["diagnostic"]
        if not torch.equal(diagnostic["examples"], features[diagnostic["class_ids"], 0]):
            raise ValueError(f"Diagnostic split does not match the feature file: {chain}")
        for folder in sorted(chain.glob("generation-*")):
            generation = int(folder.name.split("-")[1])
            model = load_model(folder, extended).eval()
            maps = attention_maps(model, diagnostic)
            previous, induction = candidate_heads(maps)
            cosine_mean, cosine = label_cosine(model)
            rows.append({"seed": seed, "condition": condition, "generation": generation,
                         "previous_head": previous, "induction_head": induction,
                         "previous_attention": float((maps[0][previous, 1, 0]
                                                      + maps[0][previous, 3, 2]) / 2),
                         "induction_attention": float(maps[1][induction, 4, 1]),
                         **ov_copy(model, induction, maps[1][:, 4, 1]),
                         "label_abs_cosine": cosine_mean,
                         **exemplar_transfer(model, diagnostic, features)})
            artefacts[seed, condition, generation] = {"maps": maps, "cosine": cosine,
                                                      "heads": (previous, induction)}
    return rows, artefacts


def aggregate(rows: list[dict]) -> list[dict]:
    keys = [key for key in rows[0] if key not in ("seed", "condition", "generation",
                                                   "previous_head", "induction_head")]
    result = []
    for condition, generation in sorted({(row["condition"], row["generation"]) for row in rows}):
        selected = [row for row in rows if (row["condition"], row["generation"])
                    == (condition, generation)]
        result.append({"condition": condition, "generation": generation,
                       "n_seeds": len(selected),
                       **{f"{key}_{stat}": func([row[key] for row in selected])
                          for key in keys for stat, func in (("mean", mean), ("sd", stdev))}})
    return result


def plot_attention(artefacts: dict, seed: int, panels: list[tuple[str, int]],
                   title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tokens = ["x1", "y1", "x2", "y2", "xq"]
    figure, axes = plt.subplots(3, len(panels), figsize=(2.3 * len(panels), 6.6),
                                constrained_layout=True, squeeze=False)
    for column, (condition, generation) in enumerate(panels):
        item = artefacts[seed, condition, generation]
        previous, induction = item["heads"]
        for row, (matrix, label, limits) in enumerate((
                (item["maps"][0][previous], f"L1 head {previous}", (0, 1)),
                (item["maps"][1][induction], f"L2 head {induction}", (0, 1)),
                (item["cosine"], "label cosine", (-1, 1)))):
            axis = axes[row, column]
            image = axis.imshow(matrix.numpy(), vmin=limits[0], vmax=limits[1],
                                cmap="viridis" if row < 2 else "RdBu_r")
            ticks = tokens if row < 2 else [str(i) for i in range(len(matrix))]
            axis.set_xticks(range(len(ticks)), ticks, fontsize=7)
            axis.set_yticks(range(len(ticks)), ticks, fontsize=7)
            axis.set_title(f"{condition} G{generation}\n{label}" if row == 0 else label,
                           fontsize=8)
        if column == len(panels) - 1:
            figure.colorbar(image, ax=axes[2, column], shrink=0.8)
    figure.suptitle(f"{title}, seed {seed}: rows query, columns key", fontsize=9)
    figure.savefig(path.with_suffix(".png"), dpi=180)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-file", type=Path,
                        default=Path("data/omniglot_resnet18_randomized_order_s0.h5"))
    parser.add_argument("--base-runs-dir", type=Path, required=True)
    parser.add_argument("--extended-runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot-seed", type=int, default=0)
    args = parser.parse_args()
    features = load_features(args.feature_file)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for task, root, extended in (("base", args.base_runs_dir, False),
                                 ("extended", args.extended_runs_dir, True)):
        rows, artefacts = inspect(root, extended, features)
        write_csv(args.output_dir / f"{task}_per_run.csv", rows)
        write_csv(args.output_dir / f"{task}_aggregate.csv", aggregate(rows))
        last = max(row["generation"] for row in rows)
        conditions = sorted({row["condition"] for row in rows}, key=lambda name: name != "clean")
        plot_attention(artefacts, args.plot_seed,
                       [("clean", 0)] + [(name, last) for name in conditions],
                       f"{task.capitalize()} task", args.output_dir / f"{task}_attention")
    write_json(args.output_dir / "protocol.json", {
        "split": "diagnostic episodes saved with each chain",
        "attention_maps": "mean over episodes whose matching context pair is first",
        "ov_copy": "direct path, LayerNorm applied to the isolated label vector",
        "exemplar_transfer": "query feature replaced by another exemplar of the same class"})
    print(f"Wrote inspection results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
