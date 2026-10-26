PROJECT 1: MECHANISTIC INTERPRETABILITY OF MODEL COLLAPSE
========================================================

This is our submission code package. We study how recursive training on
generated targets changes in-context learning on the Omniglot-feature task.
Our base model predicts a query label. Our extended model predicts the query
label, a pointer to one of the context features, and the selected feature's
label. The pointer is our constrained adaptation of a next-image output; it
does not synthesize an arbitrary image.

Contents
--------

project1/       Aligned data, models, study runners, diagnostics, and evaluator
tests/          Focused tests for the submitted implementation
requirements.txt
RUN_STEPS.md    Commands and output descriptions

The extended abstract (with the NeurIPS checklist and AI declaration) is
submitted as a separate PDF. We do not include exploratory notebooks, legacy
token experiments, raw run directories, virtual environments, or downloaded
feature data in this folder.

Quick start
-----------

Use Python 3.13 with the packages in requirements.txt. Its pinned PyTorch CPU
wheel uses the official PyTorch index and is supported on our tested WSL
Linux environment. Run commands from this
directory so relative data and output paths resolve correctly. We tested with
Python 3.13.12, PyTorch 2.14.0+cpu, NumPy 2.5.2, h5py 3.16.0, and
Matplotlib 3.11.1 in WSL. Our local virtual environment is at ../.venv-wsl;
it is not part of the submitted package.
We also verified a fresh WSL Python 3.13 environment with PyTorch 2.14.0+cpu,
NumPy 2.5.4, h5py 3.16.0, and Matplotlib 3.11.2: all tests then present and
both two-generation, ten-step real-feature smoke chains passed. These short
runs check execution, not scientific convergence. The current suite has 25
tests.

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v
    python -m project1.omniglot_download

The downloader verifies the published Omniglot feature file against a fixed
SHA-256 checksum. See RUN_STEPS.md for a small smoke run, the
validation-loss hyperparameter sweep, both controlled studies, training
dynamics, attention and embedding inspection, the mechanistic analysis, and
the fresh final evaluation. The tuned study trains 30 ten-generation chains,
so it is substantially more costly than the tests or smoke run.

Our PyTorch code implements the task and configuration
described by Singh et al. (2024). We download their published preprocessed
Omniglot features from https://github.com/aadityasingh/icl-dynamics and do not
bundle their source code or data. The exploratory notebooks in ../Experiments/
are our group members' work; we adapted selected methods, not their numerical
findings, to the aligned feature task. Our extended next-symbol prediction
chooses between the two features already present in the context. It does not
generate a new image; the abstract states this constraint.

Our saved research results and detailed working reports are in ../Experiments/;
they are not required to run this package from a fresh copy. Every number in
the abstract was checked against those saved artifacts.
Commands in RUN_STEPS.md create local data, runs, and figures directories.
We exclude those generated directories when assembling the final upload.
