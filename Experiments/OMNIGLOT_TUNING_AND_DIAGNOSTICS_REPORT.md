# Omniglot Tuning and Additional Diagnostics

20 October 2026. This report records the generation-0 hyperparameter sweep,
the decision to re-run both studies with tuned settings and ten
generations, and three diagnostics added for the project brief (training
dynamics, attention maps and embeddings, query-exemplar transfer). The
diagnostics below were run on the three-generation pilot studies
(`runs/omniglot-{base,extended}-study-20261009`) and must be re-run on the
tuned study before any number enters the abstract. That re-run is reported in
`OMNIGLOT_TUNED_STUDY_REPORT.md`.

## Why the pilot needed tuning

The pilot used one setting (Adam, learning rate `3e-4`, 1,000 updates,
batch 64) that was checked only to learn the task. Its validation-selected
checkpoints were usually the last one evaluated: 21 of 30 base checkpoints
and 33 of 60 extended checkpoints were selected at update 1,000, so
validation loss was still falling. Undertrained teachers generate more
erroneous targets, which can inflate the measured feedback effect.

## Sweep protocol

`project1.omniglot_sweep` trains one clean generation for each learning
rate in {1e-4, 3e-4, 1e-3, 3e-3}, three seeds (0-2), both tasks, 4,096
training episodes, and 5,000 updates. There is no learning-rate schedule
and minibatches come from a fixed generator, so the first N updates of a
5,000-update run equal an N-update run; the best validation loss within
budgets of 1,000, 2,000, 3,000, and 5,000 updates is read from the saved
history. Selection uses mean validation loss only. Test metrics written by
the runners were not read. Results: `figures/omniglot-sweep-20261020/`.

## Sweep results

Mean validation loss over three seeds (sample SD in brackets):

| Setting | Base | Extended |
| --- | ---: | ---: |
| lr 3e-4, 1,000 updates (pilot) | 0.0232 (0.0079) | 0.2674 (0.0154) |
| lr 1e-3, 3,000 updates | 0.0122 (0.0091) | 0.2401 (0.0073) |
| lr 1e-3, 5,000 updates | 0.0121 (0.0091) | **0.2372** (0.0050) |
| lr 1e-4, 5,000 updates | **0.0112** (0.0016) | 0.2895 (0.0805) |

Learning rate `3e-3` was unstable for both tasks. The extended loss has an
irreducible floor of ln(2)/3 = 0.231, because the clean next-feature target
is uniform over two context features and the loss averages three outputs.
Relative to that floor, the pilot setting leaves about 0.036 reducible loss
and the selected setting about 0.006. Extended validation first-label
accuracy rises from 98.6% to 99.5%. The base task is near ceiling for all
stable settings; its best setting differs from lr `1e-3`, 5,000 updates by
less than one seed SD.

## Decision

Both studies are re-run under protocol `tuned` in
`project1.omniglot_study.PROTOCOLS`: lr `1e-3`, 5,000 updates, ten
generations, the same five seeds, conditions, data sizes, and evaluation
schedule. One shared setting keeps the base and extended tasks comparable.
The pilot remains a documented, separately labelled study. Runs:
`runs/omniglot-{base,extended}-tuned-20261020/`. The refactored analysis
reproduces every pilot CSV byte-for-byte and the pilot final-evaluation
replay (`per_run.csv` SHA-256 `fcb0e92a...5fcf5`).

## Pilot diagnostics (to be repeated on the tuned study)

**Training dynamics** (`figures/omniglot-training-dynamics-20261020/`).
Generation-0 validation accuracy shows a plateau near 50% before a sharp
rise, as expected when the two-head retrieval circuit forms. Frozen
checkpoints fit their fixed training episodes almost perfectly (training
accuracy at least 98.9% against their own targets). At extended G2 under
full feedback, training first-label accuracy is 98.95% against the
corrupted targets but 99.38% against the true labels: models fit the true
retrieval rule better than the noisy targets they were trained on.

**Attention maps and embeddings** (`figures/omniglot-inspection-20261020/`).
In every condition, the selected layer-1 head moves each context feature
onto its label position and the selected layer-2 head attends from the
query to the matching label. These patterns persist at G2 under feedback.
A direct-path OV copy score (label embedding through the layer-2 value and
output weights to the unembedding) was near the five-label chance level for
the selected head and weak, with large seed variance, even when summed over
attention-weighted layer-2 heads. We treat it as inconclusive: label
information is likely read from a composed residual representation, which
activation patching tests more faithfully. Label-embedding mean absolute
cosine stays around 0.16-0.20 in all conditions.

**Query-exemplar transfer.** All splits present exemplar 0 of every class,
so the query feature exactly repeats a context feature. Replacing the query
with another exemplar of the matching class drops first-label accuracy from
about 97% to about 76% in every condition and generation; feedback does not
change it. The models therefore rely substantially on exact feature
identity and only partly on class-level similarity. This is a limitation of
our training distribution relative to Singh et al.'s additional
unseen-exemplar evaluations, not a feedback effect.
