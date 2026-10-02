# Step 4: Mechanistic Interpretation

## Scope and method

Mechanistic diagnostics were run on the separate diagnostic prompts for the
three base pilot chains and all 45 extended-study checkpoints. The extended
results below summarize generation 2 across three seeds; full per-checkpoint
measurements are in [`summary.csv`](figures/interpret-extended/summary.csv)
and [`details.json`](figures/interpret-extended/details.json). The baseline
task is discrete-symbol in-context learning, not Omniglot feature induction.

For each checkpoint, the analyzer measures per-head previous-token and
induction-pattern attention, then ablates every attention head separately,
the highest-scoring previous-token and induction heads together, and entire
attention layers. It also retains *only* that selected head in each layer while
ablating all other attention heads. These interventions report first-label
accuracy and loss. The retained-head test is sufficiency relative to the other
attention heads, with embeddings, MLPs, and residual paths still present. It also
corrupts the matching context label to an unused label and patches selected
clean head outputs into the corrupted pass. Restoration is the recovered
fraction of the clean-minus-corrupt *correct-label log-probability* gap on
examples for which corruption flips a formerly correct prediction. The
extended analysis additionally measures conditional next-symbol TV after
head and layer ablations, plus symbol-embedding effective rank. Head choice
is made independently per checkpoint, not by treating head indices as aligned
across independently trained models.

## Observations

At generation 2, the mean maximum induction-attention score is 0.475 for
clean, 0.479 for 90% feedback, 0.444 for 100% feedback, 0.458 for soft-oracle
resampling, and 0.432 for sampled-oracle resampling. The sampled-oracle
control already shows a score decline, so this attention metric does not
isolate a feedback-specific circuit change. The corresponding mean symbol
embedding effective ranks are 17.3, 22.6, 22.7, 19.0, and 21.5. These are
descriptive geometry measurements, not evidence that embedding rank causes
the behavioral change.

Mean first-label accuracy drops when the *selected pair* of heads is ablated
are 0.029 (clean), 0.036 (90% feedback), 0.055 (100% feedback), 0.026
(soft-oracle), and 0.049 (sampled-oracle). In contrast, ablating the entire
second attention layer drops accuracy by 0.820, 0.708, 0.656, 0.841, and
0.776 respectively. Thus the second attention layer is functionally important,
but the highest-attention head pair alone is not sufficient to account for its
effect. Retaining only those two heads gives mean first-label accuracies of
0.464, 0.388, 0.406, 0.458, and 0.448 respectively, far below intact-model
accuracy. The selected pair is therefore not sufficient as an attention-head
subset. Redundant heads, distributed computation, and residual pathways are
plausible; these experiments do not separate them fully.

On corruption-sensitive examples, patching the selected pair restores a
mean 0.275, 0.322, 0.316, 0.335, and 0.423 of the correct-label log-probability
gap respectively. This is causal evidence that selected head outputs carry
some task-relevant information, but incomplete restoration does not prove
that the remaining information travels through a specific alternate circuit.
The [mechanistic comparison](figures/interpret-extended/mechanistic_comparison.png)
shows the seed variation.

For the extended next-symbol prediction, 100% feedback has mean conditional
TV 0.139 on diagnostic prompts. Ablating the whole second attention layer
leaves TV at 0.131. The analogous sampled-oracle values are 0.130 and 0.133;
clean values are 0.0026 and 0.0045. This intervention does not remove the
symbol skew, so the skew cannot be attributed to the second-layer induction
heads alone. The [symbol ablation figure](figures/interpret-extended/symbol_ablation.png)
shows the comparison. The output head, embeddings, first layer, and training
target imbalance remain candidate contributors; this analysis has not
localized the cause among them.

The base pilot tells a complementary story. Its selected head-pair ablation
causes no accuracy loss on the 128 diagnostic prompts in any generation,
despite high baseline accuracy. Whole-second-layer ablation causes a much
larger loss (see [`base summary`](figures/interpret-base/summary.csv)). The
generation-1 attention-score decrease appears in the clean control as well
as feedback chains, so it is not evidence of collapse in that pilot.

## Limits and conclusion

These tests establish layer-level necessity for first-label performance and
partial causal contribution of selected heads under a specific corruption.
They directly reject sufficiency of the two selected attention heads in this
ablation setup. They do not establish a minimal sufficient circuit, prove that
embedding-rank changes cause errors, or identify a unique mechanism of
conditional symbol skew. The diagnostic subset has 128 prompts per checkpoint,
and the extended study has only three seeds and two feedback transitions.
The behavioral matched control in [STEP3_REPORT.md](STEP3_REPORT.md) is
essential: one-hot oracle resampling reproduces much of the distributional
drift without recursive model outputs.

## Reproduce

From `Project1` with the verified WSL environment:

```powershell
wsl.exe .venv-wsl/bin/python -m project1.interpret --family base
wsl.exe .venv-wsl/bin/python -m project1.interpret --family extended
wsl.exe .venv-wsl/bin/python -m unittest discover -s tests -v
```

The interpretation commands reload saved checkpoints and diagnostic splits;
they do not retrain models. Run artifacts under `runs/` are required and are
not tracked by Git.
