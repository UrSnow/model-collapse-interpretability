# Step 2: Recursive Synthetic-Data Training

## Status

Started 17 September 2026. The base task's three matched seed-0 chains
(clean, 90% synthetic, and fully synthetic) finished by 1 October 2026.
The extended three-token implementation and matched three-seed chains are
also complete. Behavioral results are in [STEP3_REPORT.md](STEP3_REPORT.md).

## Research question

How does repeated training on predictions from preceding models affect clean
in-context learning performance and induction-pattern attention?

## Base protocol, version 1

- Task: the existing discrete-symbol baseline, with 50 symbols, five labels,
  and two context pairs. This is an adaptation, not an exact replication of
  Singh's Omniglot feature task.
- Generation 0 uses clean labels. Each subsequent generation replaces a fixed
  fraction of labels with categorical samples from its immediate predecessor.
- Temperature defaults to 1.0. The clean prompt set stays fixed. Incorrect
  predictions remain in training data; oracle labels are used only for the
  clean fraction and for measuring label error.
- Dataset size stays fixed. Data are replaced rather than accumulated.
- Each generation starts with fresh model weights and a fresh AdamW optimizer.
- Defaults: 8,192 training prompts, 1,024 prompts in each evaluation split,
  10,000 updates, batch size 128, learning rate 0.0003, weight decay 0.0001.
  These are initial pilot settings, not validated convergence guarantees.
- Four unique prompt sets are assigned by SHA-256 partition: training,
  validation, test, and diagnostics. Identical prompts cannot occur across
  splits, even if independently sampled. Symbols are shared across splits;
  this tests new prompts rather than unseen symbol identities.
- Clean validation loss selects the teacher checkpoint. All test evaluation
  occurs after the chain finishes. Diagnostics use a separate split.
- Conditions with the same seed share prompts, initialization seeds and
  minibatch orders. Only their synthetic fraction differs.
- CPU execution is used for this first implementation.

This base experiment measures recursive label corruption with externally
maintained input diversity. It cannot establish collapse of the symbol input
distribution, since the model does not generate symbols here.

## Extended protocol

The extended task presents the same two symbol-label context pairs and query
symbol, then predicts three tokens in order: query label, next symbol, and the
next symbol's context-specific label. The true next-symbol distribution is
uniform over the two context symbols, which guarantees that its label is
recoverable. Model-generated symbols are constrained to those two choices.
The transformer scores all three positions in one causally masked pass;
predecessor outputs are sampled autoregressively and replace the complete
three-token target on selected rows. Prompts and dataset size remain fixed,
while each new generation gets fresh weights and optimizer state.

Clean rows use expected next-symbol cross-entropy under the *known* uniform
conditional, not one fixed sampled symbol as if it were uniquely correct.
Model-feedback rows use sampled one-hot targets. The study includes 0%, 90%,
and 100% model-generated conditions, plus two oracle-resampling controls:
one with the clean expected objective and one with a one-hot sampled objective
matched to model feedback. This distinction matters because the objective
itself changes how finite-sample symbol noise is learned.

The frozen extended pilot uses seeds 0, 1, and 2; 4,096 training prompts;
512 validation, test, and diagnostic prompts; three generations; 3,000 updates
per generation; batch size 128; evaluation every 250 steps; and temperature 1.
Validation selects checkpoints. Test metrics are evaluated after each complete
chain and analyzed after all conditions finish. The pilot's symbols are
discrete tokens, not Omniglot image features, so it is an adaptation rather
than an exact reproduction of Singh et al.'s input representation.

The first extended calibration (`runs/extended-calibration-seed0`) treated one
random oracle symbol draw per fixed prompt as a unique clean target. Its
validation conditional symbol TV rose from about 0.05 to 0.32 by 3,000 steps,
demonstrating overfitting to random targets. The corrected soft-oracle run
(`runs/extended-calibration-soft-symbol-seed0`) retained near-uniform symbol
predictions (TV about 0.0025 at 3,000 steps) while both label accuracies reached
100% on validation. This correction was selected on validation behavior before
the matched study's test results were analyzed.

## Reproduction

From Project1 using the verified WSL environment:

```powershell

# Clean-data control
wsl.exe .venv-wsl/bin/python -m project1.collapse --output-dir runs/step2-base-clean --synthetic-fraction 0

# Fully synthetic labels after generation 0
wsl.exe .venv-wsl/bin/python -m project1.collapse --output-dir runs/step2-base-synthetic --synthetic-fraction 1

# 90% synthetic labels, 10% clean labels
wsl.exe .venv-wsl/bin/python -m project1.collapse --output-dir runs/step2-base-mixed --synthetic-fraction 0.9
```

The generation count includes generation 0. Output directories must be new to
prevent overwriting previous runs. Each chain saves the protocol, all split
prompts, generation labels and synthetic masks, parent and random-seed metadata,
validation histories, selected checkpoints, attention scores, and final tests.
Restart/resume is not implemented. Each immutable run uses a new output
directory, and all planned chains completed; interruption requires starting a
new chain. This is a convenience limitation, not a missing experiment.

For the extended study, from a fresh output directory:

```powershell
wsl.exe .venv-wsl/bin/python -m project1.extended_study --output-dir runs/extended-study-reproduction
wsl.exe .venv-wsl/bin/python -m project1.analyze_extended --runs-dir runs/extended-study-reproduction --output-dir figures/extended-study-reproduction
wsl.exe .venv-wsl/bin/python -m project1.interpret --family extended --runs-dir runs/extended-study-reproduction --output-dir figures/interpret-extended-reproduction
```

The study command runs all five conditions and three seeds. Existing output
directories are refused rather than overwritten. The completed five-condition
study is under `runs/extended-study`.

## Verification and results

Verified on 1 October 2026: all 28 regression tests pass, including split
disjointness, deterministic label replacement, exact mixing counts, preservation
of incorrect synthetic labels, and clean/fully synthetic boundary conditions.

Two matched smoke chains completed (generations 0 and 1, five updates each,
128 training prompts, 64 prompts per evaluation split, batch size 16).
Artifacts are in `runs/step2-smoke-clean` and `runs/step2-smoke-synthetic`.
Their generation-0 summaries match exactly. Generation 1 of the clean control
has zero synthetic rows and zero label errors; the synthetic condition has 128
synthetic rows and 101 incorrect labels (78.91%). This high error rate reflects
an intentionally undertrained smoke-test teacher, not established model collapse.
Both smoke chains completed checkpoint selection, reload, diagnostics and
delayed test evaluation.

The full base-task pilot used three generations, 10,000 updates per generation,
and seed 0 in each condition. The clean and mixed conditions have identical
saved training labels and metrics: test accuracy is 1,024/1,024 for generations
0 and 1 and 1,023/1,024 for generation 2. The fully synthetic condition has
the same accuracy; its generation-2 training labels contain one incorrect row
out of 8,192 and its generation-2 test NLL is 0.003089 versus 0.001253 for
the clean control. This does not establish collapse. The pilot test split has
been inspected, so final protocol choices require a fresh isolated holdout.
See STEP3_REPORT.md for the full comparison and reproducible figures.

## Remaining work

- [x] Base recursive label generation, fixed-size mixing, and clean control.
- [x] Explicit split isolation and separate diagnostic data.
- [x] Persist generation ancestry, labels, seeds, metrics, and checkpoints.
- [x] Verify tests and paired short smoke chains.
- [x] Calibrate extended-task generation 0 on validation only.
- [x] Define and implement query label -> next symbol -> corresponding label.
- [x] Verify shifted targets, symbol support, and autoregressive feedback.
- [x] Add soft and matched sampled-oracle resampling controls.
- [x] Freeze the extended protocol and run three seeds and three generations.
- [x] Analyze the base pilot and extended study with a fresh final holdout.
- [ ] Add interruption-safe resume only if future longer studies require it.

## Interpretation safeguards

Step 1's online diagnostic logging used the test seed; its results remain
exploratory. Step 2 constructs new isolated splits and does not reuse those
checkpoints. Finite training sets also change the optimization problem, so
Step 1's accuracy does not guarantee convergence here. Low attention scores
alone do not identify an alternative mechanism, and single-head ablations do
not establish an entire causal circuit. STEP4_REPORT.md documents stronger
interventions and their remaining limits.
