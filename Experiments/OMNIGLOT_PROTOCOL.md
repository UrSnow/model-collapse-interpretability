# Omniglot Task Alignment

The assignment asks us to start from the in-context learning task of Singh et
al. (2024). The earlier `project1.data` and `project1.model` experiments use
learned discrete symbol embeddings and are **not** results on that input task.
They remain intact as preliminary, separately labelled experiments.

## New base path

`project1.omniglot_data` loads the authors' preprocessed ResNet18 Omniglot
features from `resnet18/224/feat` (1,623 classes, five exemplars per class,
512 features). The downloader checks SHA-256
`a8b8fa1f25d62e077b658c590f77e24447894fa325d97651dba22c6621e62b5b`.
The binary file is ignored by Git and can be obtained with
`python -m project1.omniglot_download`.

The baseline samples two different classes: one query class and one distractor.
It presents one exemplar and a randomly assigned label from each class, then
the query exemplar. The query label is determined by the matching context
example. We use 50 training classes, five labels, eight training relabeling
pairs, two held-out validation relabeling pairs, and 100 held-out test classes,
following the principal configuration in the authors' `ih_paper_runs.sh`.
Training uses exemplar 0, as in the reference's one-exemplar training split.
The relabeling split uses seed 20 but PyTorch's permutation, so the exact pair
assignment is not guaranteed to equal the reference JAX permutation.

`project1.omniglot_model` projects continuous features to the residual stream,
interleaves them with context-label embeddings, and uses two causal attention
blocks, eight heads, width 64, RoPE, and no MLP. The base runner
`project1.omniglot_base` fixes episode splits for paired recursive chains,
samples predecessor labels only on selected training rows, reinitializes each
generation, selects checkpoints using validation loss, and evaluates test
only after the chain is fixed. It saves splits, labels, masks, seeds, model
weights, training history, and summary metrics.

## Not an exact reproduction of the paper

This is a PyTorch implementation of the paper's **input task and principal
model structure**, not a claim of numerical reproduction. The paper's example
run uses a JAX/Equinox implementation, streamed batches, one million updates,
batch size 32, and learning rate `1e-5`. Our finite fixed training set and
default 10,000 updates support the recursive-training experiment and must be
calibrated on validation data. The pilot below learned the base task using
Adam without weight decay, 1,000 updates, and learning rate `3e-4`; the
default 10,000-update setting has not been separately validated. The paper's
additional evaluations on unseen
exemplars and relabeling configurations are not yet implemented.

The new `project1.omniglot_extended` pipeline uses the same feature-based
episodes. After the query label it predicts a pointer to one of the two
context Omniglot features, then the label for that chosen feature. This
context pointer represents the next symbol without a fixed class-ID output
vocabulary, which would not transfer to held-out classes. Clean rows use the
known uniform pointer conditional; synthetic rows train on an autoregressively
sampled predecessor triple. The pointer is restricted to context classes, so
this is an explicit operationalization of the brief's extra symbol output,
not an unconstrained generator of new Omniglot images or features.

The earlier token-study reports and figures remain historical, not evidence
for the feature-based runs. The seed-0 pilot is in
`OMNIGLOT_EXTENDED_REPORT.md`; the five-seed controlled study and causal
interventions are in `OMNIGLOT_MULTI_SEED_REPORT.md` and
`OMNIGLOT_MECHANISM_REPORT.md`. The later locked evaluation is documented in
`OMNIGLOT_FINAL_EVALUATION_REPORT.md`. These results do not establish
inevitable collapse.

## Exploratory verification, 9 October 2026

The new unit tests pass, and a two-generation, ten-update smoke run completed
on the verified feature file. A separate generation-0 calibration used 4,096
fixed training episodes, 256 examples in each evaluation split, batch size 64,
1,000 updates, and learning rate `3e-4`. Validation accuracy reached 255/256;
held-out-class test accuracy was 252/256. This establishes that the new model
can learn the feature-based base task under one calibrated setting, not that it
numerically reproduces the reference paper.

One matched three-generation seed-0 pilot used the same settings with either
zero or 100% synthetic-label replacement. Splits, generation-0 labels, and
generation-0 checkpoints are byte-identical across conditions. Generation-2
clean test accuracy is 253/256 (NLL 0.0394), while feedback is 249/256 (NLL
0.0812). The feedback chain has 4 incorrect generated training labels at G1
and 12 at G2 out of 4,096 selected rows each. These small, single-seed test
counts are exploratory and must not be reported as a statistically established
collapse effect. The immutable run records are under
`runs/omniglot-base-{clean,feedback}-adam-seed0/` and are ignored by Git.

Reproduce this pilot with new output directories:

```powershell
wsl.exe .venv-wsl/bin/python -m project1.omniglot_base --output-dir runs/omniglot-clean-reproduction --generations 3 --steps 1000 --train-size 4096 --eval-size 256 --batch-size 64 --eval-every 100 --synthetic-fraction 0
wsl.exe .venv-wsl/bin/python -m project1.omniglot_base --output-dir runs/omniglot-feedback-reproduction --generations 3 --steps 1000 --train-size 4096 --eval-size 256 --batch-size 64 --eval-every 100 --synthetic-fraction 1
```

Primary references:

- Project brief (local, not committed):
  `references/_NLP_2026__Mechanistic_Interp_Collapse_Project.pdf`
- Singh et al. source and feature file: https://github.com/aadityasingh/icl-dynamics
- Reference run configuration: https://github.com/aadityasingh/icl-dynamics/blob/main/ih_paper_runs.sh
