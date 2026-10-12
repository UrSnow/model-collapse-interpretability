PROJECT 1: INTERPRETABILITY OF MODEL COLLAPSE
================================================

Research implementation
-----------------------

See STEP2_REPORT.md for the separate recursive-training protocol, commands,
verification record, and remaining work. See STEP3_REPORT.md for the matched
behavioral results and STEP4_REPORT.md for mechanistic tests. The base pipeline is available via
python -m project1.collapse --help. The earlier extended token pipeline is
available via python -m project1.extended --help. The aligned feature paths
are project1.omniglot_base and project1.omniglot_extended. Existing Step 1
artifacts are retained.

Current scope
-------------

This directory retains the original discrete-token Step 1 baseline and its
Step 2 base-task and extended-task recursive training pipelines. These runs
are preliminary adaptations, not results on the exact Singh et al. input task.
The extended task predicts the query label, one next symbol, and that symbol's
label. The next symbol is uniformly sampled from the two context symbols;
model-generated symbols are constrained to that support so their labels are
well-defined. Clean symbol rows train against the full known conditional
distribution, while synthetic rows train on sampled predecessor outputs.
The study includes both a soft-oracle resampling control and a one-hot
sampled-oracle control that matches the synthetic symbol-loss objective.
This is a discrete-token adaptation of Singh et al.'s Omniglot-feature task,
not an exact reproduction of their input representation.

Aligned Omniglot feature experiments
------------------------------------

The new project1.omniglot_base path uses the preprocessed Omniglot ResNet18
features distributed by Singh et al., with two context exemplar-label pairs,
five randomly relabelled labels, disjoint train/validation relabeling pairs,
held-out test classes, and a two-layer causal attention-only RoPE model.
Download the checksum-verified feature file, then run a small smoke chain:

    python -m project1.omniglot_download
    python -m project1.omniglot_base --output-dir runs/omniglot-smoke --generations 2 --steps 10 --train-size 128 --eval-size 32 --batch-size 16 --eval-every 5

For the WSL virtual environment, prefix each command with
"wsl.exe ../.venv-wsl/bin/python" instead of "python". This is a feature-based
baseline implementation, not an exact reproduction of every paper detail.
See OMNIGLOT_PROTOCOL.md for agreements and remaining differences; the
completed controlled and final studies are below.

The separate project1.omniglot_extended path predicts the query label,
selects one of the two context Omniglot features as the next symbol, and
predicts that selected feature's label. The pointer avoids a class-ID output
vocabulary that would not generalize to held-out test classes. For a matched
clean/feedback pilot, use distinct, nonexistent output directories:

    python -m project1.omniglot_extended --output-dir runs/omniglot-extended-clean --generations 3 --steps 1000 --train-size 4096 --eval-size 256 --batch-size 64 --eval-every 100 --synthetic-fraction 0
    python -m project1.omniglot_extended --output-dir runs/omniglot-extended-feedback --generations 3 --steps 1000 --train-size 4096 --eval-size 256 --batch-size 64 --eval-every 100 --synthetic-fraction 1
    python -m project1.omniglot_extended --output-dir runs/omniglot-extended-oracle --generations 3 --steps 1000 --train-size 4096 --eval-size 256 --batch-size 64 --eval-every 100 --synthetic-fraction 1 --feedback-source oracle --oracle-resample-objective sampled
    python -m project1.omniglot_analyze --clean runs/omniglot-extended-clean --feedback runs/omniglot-extended-feedback --oracle runs/omniglot-extended-oracle --output-dir figures/omniglot-extended-pilot

The analysis command audits matched splits and generation-0 checkpoints,
exports comparison tables and a figure, and measures per-head ablation effects
on a separate diagnostic split. These are single-seed pilot commands, not a multi-seed
final study. See OMNIGLOT_EXTENDED_REPORT.md for status and limitations.

For the locked five-seed study (seeds 0-4, three generations, 1,000 updates,
4,096 training episodes, 256 evaluation episodes), use new output directories:

    python -m project1.omniglot_study --runs-dir runs/omniglot-extended-study-reproduction --output-dir figures/omniglot-extended-study-reproduction
    python -m project1.omniglot_base_study --runs-dir runs/omniglot-base-study-reproduction --output-dir figures/omniglot-base-study-reproduction

The extended study covers clean, 90% feedback, 100% feedback, and sampled
oracle-resampling conditions. The base study covers clean and 100% feedback.
Both commands audit matching splits and generation-0 artifacts within each
seed before writing per-run, paired, and aggregate tables and figures. The
runner skips completed immutable chains when restarted, but stops for manual
review if it finds an incomplete run directory.
The audited five-seed study results and their limits are in
OMNIGLOT_MULTI_SEED_REPORT.md. The separate fresh final test is below.

For causal diagnostics on the frozen checkpoints, using only the separate
diagnostic episodes:

    python -m project1.omniglot_mechanism --base-runs runs/omniglot-base-study-reproduction --extended-runs runs/omniglot-extended-study-reproduction --output-dir figures/omniglot-mechanism-reproduction --size 256

This writes per-head, per-checkpoint, and aggregate tables plus a comparison
figure. See OMNIGLOT_MECHANISM_REPORT.md for interpretation and limits.

For the locked fresh held-out evaluation of all 90 frozen checkpoints, with
1,024 episodes per seed and exact exclusion of prior Omniglot test prompts:

    python -m project1.omniglot_final_evaluation --base-runs-dir runs/omniglot-base-study-20261009 --extended-runs-dir runs/omniglot-extended-study-20261009 --prior-runs-dir runs --output-dir figures/omniglot-final-evaluation-reproduction --size 1024

To reproduce the exact reported episode set, add
--manifest-file figures/omniglot-final-evaluation-20261010/episodes.pt.
Use a new output directory. The manifest, checkpoint hashes, paired tables,
and figure are documented in OMNIGLOT_FINAL_EVALUATION_REPORT.md.

Legacy token-task definition
----------------------------

Each prompt contains randomly paired symbol and label tokens:

    s0 l0 s1 l1 ... sk lk query_symbol

The query repeats one context symbol, and the target is its context-specific
label. A new one-to-one mapping is sampled for every sequence, so a fixed
symbol-to-label rule cannot solve the task. With two transformer layers, the
expected circuit is a previous-token head followed by an induction head that
performs match-and-copy.

The default setup uses 50 symbols, five labels, and two context pairs, matching
the central task dimensions in the reference induction-head experiment. Larger
contexts are useful later as robustness tests but learn more slowly.

The original Step 1 uses different deterministic stream seeds but does not
guarantee disjoint prompts; its results are exploratory. The later recursive
studies use hash-partitioned, disjoint fixed prompt sets. Validation selects
checkpoints. A fresh test set excluding all earlier prompts is used for the
extended study's final evaluation.

Setup
-----

From this directory, create and activate a compatible Python virtual
environment, then install:

    python -m pip install -r requirements.txt

On Windows systems where Application Control blocks unsigned PyTorch DLLs,
run the same commands inside WSL using a Linux virtual environment.
The verified WSL runtime is Python 3.13.12, PyTorch 2.14.0+cpu,
NumPy 2.5.2, and Matplotlib 3.11.1. All experiments in this directory
are configured for CPU execution.

Tests
-----

    python -m unittest discover -s tests -v

Smoke run
---------

    python -m project1.train --steps 20 --eval-every 10 --eval-batches 2 --batch-size 16

Baseline experiment
-------------------

    python -m project1.train --steps 10000 --eval-every 250 --batch-size 128 --seed 0

Multi-seed replication and analysis
-----------------------------------

Run the five-seed Step 1 experiment and then aggregate it:

    python -m project1.sweep --seeds 0,1,2,3,4
    python -m project1.analyze runs/step1-multiseed

The analysis command writes per-seed and aggregate statistics plus PNG and PDF
versions of the Step 1 summary figure. A phase change is operationally defined
as the first evaluation at or above 90% validation accuracy.

Runs are written below runs/step1/ by default. Each run contains:

    config.json       Exact task, model, seed, and training configuration
    history.csv       Training and validation metrics over optimization
    diagnostics.json  Per-head circuit scores throughout training
    best_model.pt     Checkpoint selected by validation loss
    summary.json      Locked test result and mechanistic diagnostics

Verified reference run
----------------------

The seed-0 baseline was verified locally for 10,000 steps. It remained near the
non-inductive 50% strategy through step 5,000, exceeded 95% validation accuracy
at step 5,500, and selected step 9,000 as the best checkpoint:

    Validation accuracy: 0.9957
    Locked test accuracy: 0.9956
    Best previous-token score: 0.9934 (layer 1, head 2)
    Best induction score:      0.9843 (layer 2, head 5)

This is one seed and establishes implementation correctness, not statistical
reliability. The five-seed replication below followed it.

Completed Step 1 result
-----------------------

The five-seed replication is complete. Mean locked-test accuracy is 0.9980
(sample standard deviation 0.0027), and all runs cross 90% validation accuracy.
Phase-change timing varies from step 2,500 to 7,500. Four of five seeds meet the
declared canonical-circuit thresholds; one seed has lower canonical attention
scores, which does not by itself identify an alternative strategy.
See STEP1_REPORT.md and runs/step1-multiseed/analysis for the full result.

Mechanistic diagnostics
-----------------------

The summary records per-head previous-token scores in layer 1, per-head
induction scores in layer 2, and the test-batch accuracy change caused by
ablating each head. These are starting diagnostics, not sufficient evidence of
a circuit on their own. Step 4 adds patching and grouped/layer ablations but
does not claim a minimal sufficient circuit.

Reproducibility notes
---------------------

Random seeds for model initialization and all data streams are saved. PyTorch
deterministic algorithms are enabled. Generated runs and checkpoints are not
tracked by Git.

# Clean control
wsl.exe ../.venv-wsl/bin/python -m project1.collapse --output-dir runs/step2-base-clean --generations 3 --steps 10000 --synthetic-fraction 0

# Fully synthetic labels after generation 0
wsl.exe ../.venv-wsl/bin/python -m project1.collapse --output-dir runs/step2-base-synthetic --generations 3 --steps 10000 --synthetic-fraction 1

# 10% clean-label retention
wsl.exe ../.venv-wsl/bin/python -m project1.collapse --output-dir runs/step2-base-mixed --generations 3 --steps 10000 --synthetic-fraction 0.9

# After all three chains complete, audit and regenerate Step 3 tables/figures
wsl.exe ../.venv-wsl/bin/python -m project1.analyze_collapse

Legacy token extended-task study
--------------------------------

The matched study command runs seeds 0, 1, and 2, each with a clean control,
90% model-generated triples, 100% model-generated triples, soft-oracle
resampling, and one-hot sampled-oracle resampling. Generation 0 is always
clean. It uses 4,096 fixed training
prompts, 512 prompts per evaluation split, three generations, and 3,000 updates
per generation. Output directories must not already exist:

    wsl.exe ../.venv-wsl/bin/python -m project1.extended_study --output-dir runs/extended-study-reproduction

Audit saved protocols, split tensors, target masks, labels, and checkpoints;
aggregate test metrics and regenerate figures:

    wsl.exe ../.venv-wsl/bin/python -m project1.analyze_extended --runs-dir runs/extended-study-reproduction --output-dir figures/extended-study-reproduction

Mechanistic checkpoint diagnostics use the separate diagnostic split rather
than the test split. They measure per-head and grouped ablations, clean/corrupt
activation patching, attention scores, and embedding geometry:

    wsl.exe ../.venv-wsl/bin/python -m project1.interpret --family base
    wsl.exe ../.venv-wsl/bin/python -m project1.interpret --family extended --runs-dir runs/extended-study-reproduction --output-dir figures/interpret-extended-reproduction

The final evaluation uses a fresh, disjoint 2,048-prompt test set per seed and
the already selected frozen checkpoints. Use new output paths for reproduction:

    wsl.exe ../.venv-wsl/bin/python -m project1.final_evaluation --runs-dir runs/extended-study-reproduction --artifact-dir runs/final-evaluation-reproduction --output-dir figures/final-evaluation-reproduction

For a short end-to-end extended smoke test, use a new output directory:

    wsl.exe ../.venv-wsl/bin/python -m project1.extended --output-dir runs/extended-smoke --generations 2 --steps 20 --train-size 128 --eval-size 64 --batch-size 16 --eval-every 10

Runs and checkpoints are ignored by Git. Selected tables and figures under
figures/ are tracked; they do not contain the checkpoints needed for a fresh
mechanistic analysis. The study deliberately holds prompts fixed across
generations and evaluates on isolated clean validation/test prompts. On the
fresh final test, generation-2 mean first-label accuracy is 0.9982 (clean),
0.9836 (100% feedback), and 0.9904 (matched sampled-oracle). The sampled-oracle
control develops most of the symbol skew seen under feedback, so the evidence
does not support a feedback-only explanation. See STEP3_REPORT.md for details.
