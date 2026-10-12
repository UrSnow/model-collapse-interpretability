# Omniglot Multi-Seed Controlled Study

## Scope and audit

This is the aligned feature-based follow-up to the single-seed pilots in
`OMNIGLOT_EXTENDED_REPORT.md`. Five seeds (0-4) were run for three generations
with 4,096 fixed training episodes, 256 episodes per evaluation split, batch
size 64, 1,000 Adam updates per generation, and learning rate `3e-4` without
weight decay. The extended task has clean, 90% model-feedback, 100% model-
feedback, and 100% oracle-resampling conditions. The base task has clean and
100% model-feedback conditions. All use the same feature checksum and task
configuration. The extended oracle condition uses sampled pointer loss on
redrawn valid targets, matching the feedback condition's pointer objective.

The analyzers rejected unmatched protocols or generation-0 artifacts. For
each seed, conditions have byte-identical splits, generation-0 targets, and
generation-0 checkpoints. The base and extended studies also share the same
underlying train, validation, test, and diagnostic episode tensors for all
five seeds. Each chain starts later generations from fresh weights and selects
its checkpoint by validation loss. Test evaluation follows selection; these
are study test sets, **not** a final fresh locked evaluation.

Reproducible run records are under `runs/omniglot-{base,extended}-study-20261009/`
and are ignored by Git. Audited CSV tables and figures are under
`figures/omniglot-base-study-20261009/` and
`figures/omniglot-extended-study-20261009-final/`.

## Generation-2 outcomes

Values below are mean +/- sample standard deviation across five seeds.
Accuracy uses 256 held-out-class episodes per seed; pointer TV is deviation
from the intended uniform choice of the two context features.

| Task / condition | First-label accuracy | Second-label accuracy | Pointer TV |
| --- | ---: | ---: | ---: |
| Base clean | 0.9891 +/- 0.0043 | n/a | n/a |
| Base full feedback | 0.9797 +/- 0.0043 | n/a | n/a |
| Extended clean | 0.9773 +/- 0.0173 | 0.9703 +/- 0.0137 | 0.0136 +/- 0.0021 |
| Extended 90% feedback | 0.9727 +/- 0.0124 | 0.9547 +/- 0.0174 | 0.0844 +/- 0.0110 |
| Extended full feedback | 0.9680 +/- 0.0204 | 0.9578 +/- 0.0105 | 0.1059 +/- 0.0197 |
| Extended oracle resampling | 0.9750 +/- 0.0116 | 0.9656 +/- 0.0080 | 0.0841 +/- 0.0206 |

Paired within-seed G2 changes versus clean were -0.0094 +/- 0.0059 for
base accuracy; for extended full feedback they were -0.0094 +/- 0.0282 for
first-label accuracy, -0.0125 +/- 0.0216 for second-label accuracy, and
+0.0923 +/- 0.0204 for pointer TV. The extended 90% feedback pointer-TV
change was +0.0708 +/- 0.0112, while oracle resampling's was
+0.0705 +/- 0.0208. Directly pairing full feedback against oracle gives
+0.0218 +/- 0.0259 additional pointer TV. The latter contrast is variable
across only five seeds and does not establish a model-specific effect.

The extended full-feedback chain had mean 40.4 incorrect first labels and
72.6 inconsistent second labels among 4,096 training triples at G2; the
oracle condition had none by construction. Class coverage stayed at all 50
training classes in every condition. Mean class entropy remained near the
50-class maximum `ln(50) = 3.9120`; at G2 it was 3.9061 for extended clean,
3.9051 for full feedback, and 3.9055 for oracle resampling. Thus the
stronger observed shift is in *conditional pointer probabilities*, not loss
of class support or a substantial contraction of class diversity.

## Interpretation limits and next work

The base and extended models both learn the Omniglot task and show small
feedback-associated changes at G2, but not a demonstrated monotonic or
inevitable collapse. The oracle control shows that much of the pointer shift
can occur without model-generated errors. The five seeds vary initialization
and sampled episodes but keep the held-out class pool and relabel-pair split
fixed. The 256-example test split per seed is relatively small, and this
study was designed after an exploratory seed-0 pilot. These outcomes should
guide mechanism tests, not serve as the final untouched evaluation.

Targeted causal interventions on the separate diagnostic split were completed
across the frozen base and extended checkpoints; see
`OMNIGLOT_MECHANISM_REPORT.md`. The later locked, larger held-out evaluation
is recorded in `OMNIGLOT_FINAL_EVALUATION_REPORT.md`. The existing token-study
figures are historical and must not be mixed with these Omniglot-feature
estimates.
