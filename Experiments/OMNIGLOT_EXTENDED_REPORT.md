# Feature-Based Extended Task: Alignment Pilot

This records the initial seed-0 pilot. See `OMNIGLOT_MULTI_SEED_REPORT.md`
for the later five-seed controlled study; do not substitute pilot estimates
for its audited aggregate results. See `OMNIGLOT_MECHANISM_REPORT.md` for the
subsequent causal diagnostics.

## Status

The assignment asks for the Singh et al. base task, recursive training, an
extended prediction of the next symbol and its value, performance analysis,
and mechanistic interpretation. `project1.omniglot_base` and
`project1.omniglot_extended` now use the authors' verified Omniglot ResNet18
features. The earlier discrete-token work remains a preliminary adaptation
and must not be presented as feature-based results.

For the additional symbol, the extended model chooses one of the two context
Omniglot classes by pointer, appends its feature token, and predicts its
context-assigned label. This is a constrained next-symbol task, not arbitrary
image generation. It avoids learning an output classifier over class IDs,
which would fail by construction on disjoint test classes. A clean pointer
has a uniform conditional distribution; the clean loss averages over both
choices. Subsequent fully synthetic generations sample the first label,
pointer, and second label autoregressively from the predecessor.

## Pilot protocol

- Verified author feature file: 1,623 classes, five exemplars, 512 features.
- Same two-context, five-label, 50-train-class and 100-test-class task setup
  as the aligned base path; held-out validation relabeling pairs.
- Three generations, seed 0, 4,096 fixed training episodes, 256 validation,
  test, and diagnostic episodes, batch size 64, 1,000 Adam updates per
  generation, learning rate `3e-4`, no weight decay.
- Validation loss selects checkpoints. Test evaluation occurs after every
  generation's checkpoint has been selected. The clean and 100% synthetic
  runs have byte-identical splits, generation-0 targets, and generation-0
  checkpoints.
- Each recursive generation starts with fresh weights and optimizer. The
  feature prompts stay fixed; only selected training triples are replaced.

The saved run records are `runs/omniglot-extended-{clean,feedback,oracle}-seed0/`.
They are ignored by Git. The audited table, figure, and per-head diagnostic
ablations are in `figures/omniglot-extended-controlled-seed0/`. The oracle
control redraws valid context-feature pointers and their correct labels, but
never uses model-generated targets. It uses the sampled pointer objective on
redrawn rows, matching the feedback condition's pointer loss. Its splits and
generation-0 checkpoint match the clean and feedback runs byte-for-byte.

## Pilot observations

| Generation | Condition | First-label test accuracy | Second-label test accuracy | Test pointer TV from uniform | Training class entropy |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | both | 255/256 | 251/256 | 0.0119 | 3.9051 |
| 1 | clean | 251/256 | 249/256 | 0.0133 | 3.9051 |
| 1 | feedback | 244/256 | 246/256 | 0.0692 | 3.9056 |
| 1 | oracle resampling | 248/256 | 248/256 | 0.0910 | 3.9054 |
| 2 | clean | 250/256 | 250/256 | 0.0120 | 3.9051 |
| 2 | feedback | 252/256 | 248/256 | 0.1041 | 3.9041 |
| 2 | oracle resampling | 247/256 | 248/256 | 0.0550 | 3.9034 |

The feedback chain had eight wrong first labels and 26 inconsistent second
labels among 4,096 sampled triples at G1; at G2 these counts were 40 and 79.
All 50 training classes remained represented. Maximum possible entropy over
50 classes is `ln(50) = 3.9120`, so the observed diversity movement is small.
The rising feedback pointer deviation on held-out test prompts is more
visible than the class-entropy change. However, oracle resampling also changes
the class distribution and pointer behavior, and one seed with 256 test
episodes does not establish an effect specific to model-generated data. The
generation-2 feedback first-label accuracy is slightly *higher* than clean,
while second-label accuracy is slightly lower. We should not summarize this
as monotonic loss of in-context learning.

The final selected checkpoints' first-label diagnostic head ablations include
a clean layer-1 head (zero-indexed layer 0, head 6) with an 11.3 percentage
point accuracy drop and a feedback layer-1 head (layer 0, head 0) with a 16.4
point drop. These are interventions on a 256-example diagnostic split, not
proof of a unique or minimal induction circuit. The saved attention scores
and full per-head effects permit further causal checks.

## Follow-up status

1. Completed: matched clean, partial-feedback, full-feedback, and oracle
   controls across five seeds; see `OMNIGLOT_MULTI_SEED_REPORT.md`.
2. Completed: class-distribution comparisons and first-label circuit
   interventions, with second-label and pointer ablation diagnostics; see
   `OMNIGLOT_MECHANISM_REPORT.md`. These do not prove a unique circuit.
3. Pending: use a larger fresh held-out test set for final estimates after
   all analysis choices are fixed. Do not tune on the pilot test metrics.
4. Pending: write the two-page extended abstract, append the required NeurIPS and
   Faculty AI ethics statements, and audit submission files and citations.

The model uses the paper's input features and principal attention-only model
structure, but is a PyTorch reimplementation with a finite episode dataset,
different optimizer schedule, and a constrained pointer formulation of the
assignment's additional symbol. It is not a numerical reproduction of the
authors' million-step JAX experiments.
