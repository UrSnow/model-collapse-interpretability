# Step 3: Performance and Distribution Analysis

## Status and scope

Base-task, single-seed pilot and extended-task, three-seed analysis completed
1 October 2026. The extended study includes a separate fresh final test. These
are results for a discrete-token adaptation, not an exact Omniglot replication.

## Pilot protocol and audit

The three matched chains are `runs/step2-base-clean`,
`runs/step2-base-mixed`, and `runs/step2-base-synthetic`. Each has seed 0,
three generations including generation 0, 8,192 fixed training prompts,
1,024 prompts in each evaluation split, 10,000 updates per generation,
batch size 128, and temperature 1. The clean condition replaces no labels;
the mixed condition selects 90% of rows for sampled labels; the synthetic
condition selects all rows. Generation 0 is clean in every chain. Each later
model has fresh initialization and optimizer state. Validation loss chooses a
checkpoint, and the run evaluates test after the chain is trained.

The analysis script checks protocol equality apart from output path and
synthetic fraction, exact tensor equality of train/validation/test/diagnostic
splits, complete histories and generation counts, generation-0 summary
equality, saved label and mask counts, and preservation of clean labels. All
checks passed. Checkpoint test metrics are taken from the training pipeline's
saved summaries, not independently recomputed by this analyzer.

## Results

Test accuracy uses the same 1,024 clean prompts in every row. `Errors` is the
number of saved training targets that differ from the oracle among 8,192 rows.
NLL is mean test cross-entropy. Perplexity is `exp(NLL)`; the full values are in
[`comparison.csv`](figures/step3-base/comparison.csv).

| Condition | Generation | Selected synthetic rows | Errors | Test accuracy | Test NLL | Max induction attention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | 0 | 0 | 0 | 1,024/1,024 | 0.001358 | 0.989 |
| Mixed | 0 | 0 | 0 | 1,024/1,024 | 0.001358 | 0.989 |
| Synthetic | 0 | 0 | 0 | 1,024/1,024 | 0.001358 | 0.989 |
| Clean | 1 | 0 | 0 | 1,024/1,024 | 0.00000894 | 0.496 |
| Mixed | 1 | 7,373 | 0 | 1,024/1,024 | 0.00000894 | 0.496 |
| Synthetic | 1 | 8,192 | 0 | 1,024/1,024 | 0.00000894 | 0.496 |
| Clean | 2 | 0 | 0 | 1,023/1,024 | 0.001253 | 0.531 |
| Mixed | 2 | 7,373 | 0 | 1,023/1,024 | 0.001253 | 0.531 |
| Synthetic | 2 | 8,192 | 1 | 1,023/1,024 | 0.003089 | 0.535 |

The mixed condition's generation-1 and generation-2 saved labels are identical
to the clean targets, so its training data and learning histories are identical
to the clean control. In the fully synthetic condition, generation 1 also has
no incorrect labels. Only generation 2 introduces one incorrect training label
(0.0122% of rows); its test NLL is higher than the clean control's, but test
accuracy is the same. This is a tiny perturbation in one chain, not evidence of
progressive collapse. Figure [generation comparison](figures/step3-base/generation_comparison.png)
shows these outcomes; [learning curves](figures/step3-base/learning_curves.png)
show validation accuracy over optimization steps.

Maximum induction-pattern attention falls from 0.989 at generation 0 to 0.496
at generation 1 in **all** conditions. A clean control with the same drop rules
out synthetic feedback as its cause in this pilot. This attention-only score
does not establish a change in the causal circuit; that requires Step 4 tests.

## Extended-task study

The extended task predicts the query label, a next symbol uniformly drawn from
the two context symbols, then that symbol's context-specific label. Five
conditions share generation 0, three seeds (0, 1, 2), fixed prompt sets, 4,096
training prompts, 512 validation prompts, and 3,000 updates per generation:
clean, 90% and 100% model-feedback replacement, oracle resampling with the
expected-distribution symbol objective, and oracle resampling with sampled
one-hot symbol targets. The latter is the closer control for feedback rows.
There are three generations including generation 0. All 45 checkpoint records
and target tensors passed the protocol and data audit in
[`aggregate.csv`](figures/extended-study/aggregate.csv).

After the study and its controls were fixed, the frozen checkpoints were
evaluated once on 2,048 new test prompts per seed. These prompts are in the
hash-partitioned test set and explicitly exclude every original train,
validation, test, and diagnostic prompt. The new prompt tensors and SHA-256
hashes are saved under `runs/final-evaluation`; the metrics are in
[`per_run.csv`](figures/final-evaluation/per_run.csv). The table shows
generation-2 means over three independent seeds. Conditional symbol TV is
total variation from the true 50/50 distribution, averaged per prompt. Mean
NLL is the average of first-label, next-symbol, and second-label losses; its
irreducible clean component includes the next-symbol entropy.

| Condition | First-label accuracy | Second-label accuracy | Symbol TV | Mean NLL | Autoregressive second-label inconsistency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 0.9982 | 0.9977 | 0.0028 | 0.2355 | 0.0028 |
| 90% feedback | 0.9894 | 0.9896 | 0.1417 | 0.2813 | 0.0335 |
| 100% feedback | 0.9836 | 0.9844 | 0.1382 | 0.2907 | 0.0474 |
| Oracle, soft symbol objective | 0.9984 | 0.9985 | 0.0028 | 0.2351 | 0.0028 |
| Oracle, sampled symbol objective | 0.9904 | 0.9901 | 0.1263 | 0.2695 | 0.0212 |

The clean and soft-oracle controls retain near-uniform conditional symbol
predictions. The sampled-oracle control develops most of the skew seen in the
feedback conditions despite having no generated-label errors, showing that
one-hot finite-sample training is a major alternative explanation. Relative
to this matched control, 100% feedback has higher mean NLL by 0.0212 and
lower mean second-label accuracy by 0.0057. Its paired per-seed NLL differences
are +0.0226, +0.0250, and +0.0161; the second-label accuracy differences are
-0.0068, -0.0127, and +0.0024. Three seeds do not justify a strong population
claim, and accuracy is not consistently worse in every seed.

The 100% feedback condition's mean autoregressive second-label inconsistency
is 0.0474 versus 0.0212 for sampled-oracle. Its global generated-symbol TV
from the oracle distribution is 0.0586 versus 0.0526 for sampled-oracle;
all 50 symbols remain represented in training targets. This is conditional
prediction drift and error propagation over a short chain, not loss of global
symbol support. The [final comparison figure](figures/final-evaluation/comparison.png)
shows means and sample standard deviations. Longer chains and more seeds would
be needed to estimate whether the gap grows or stabilizes.

## Interpretation and limitations

The generation-0 teacher is nearly perfect on the *fixed training prompts*.
Sampling from it usually reproduces the oracle label, so selecting 90% or 100%
synthetic rows does not imply that 90% or 100% of targets are wrong. The base
pipeline generates labels only; it holds symbol prompts fixed. It therefore
cannot measure symbol diversity, tail coverage, entropy, or input-distribution
collapse. Near-ceiling accuracy also leaves little room to detect small changes.

The base pilot is one seed, three generations, and one task difficulty. Its
test results were inspected before the extended protocol was finalized, so
they are exploratory. The extended study has between-seed variability, but
only three seeds and two feedback transitions. The original extended test
also informed the addition of the sampled-oracle control; the separate fresh
test above is the final behavioral evaluation. Checkpoints were selected on
validation, not either test set. Symbols are abstract discrete tokens rather
than Omniglot features. Results should not be generalized to image-feature
induction or large language models without new experiments.

## Reproduce

From `Project1`, after the three Step 2 chains complete:

```powershell
wsl.exe .venv-wsl/bin/python -m project1.analyze_collapse
wsl.exe .venv-wsl/bin/python -m project1.analyze_extended
wsl.exe .venv-wsl/bin/python -m project1.final_evaluation --artifact-dir runs/final-evaluation-reproduction --output-dir figures/final-evaluation-reproduction
```

These commands write audited tables and PNG/PDF figures under `figures/`.
Use new output directories because raw runs are immutable and ignored by Git.
The final-evaluation command applies the same frozen checkpoints to the same
deterministically generated fresh prompts; it does not retrain models.
Mechanistic comparisons and their limits are documented in STEP4_REPORT.md.
