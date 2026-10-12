# Run steps

We run these commands from `submission/` after installing
`requirements.txt`. The commands below use `python`; on our local WSL setup,
we use `wsl.exe ../.venv-wsl/bin/python` instead.

1. Verify the package and obtain the data:

   ```text
   python -m unittest discover -s tests -v
   python -m project1.omniglot_download
   ```

   The downloader writes `data/omniglot_resnet18_randomized_order_s0.h5`.
   `project1.omniglot_data` verifies its SHA-256 before training or
   evaluation. Downloading requires network access; tests do not.

2. Run small base and extended smoke checks in new output directories:

   ```text
   python -m project1.omniglot_base --output-dir runs/base-smoke --generations 2 --steps 10 --train-size 128 --eval-size 32 --batch-size 16 --eval-every 5
   python -m project1.omniglot_extended --output-dir runs/extended-smoke --generations 2 --steps 10 --train-size 128 --eval-size 32 --batch-size 16 --eval-every 5
   ```

3. Reproduce our controlled five-seed studies:

   ```text
   python -m project1.omniglot_base_study --runs-dir runs/base-study --output-dir figures/base-study
   python -m project1.omniglot_study --runs-dir runs/extended-study --output-dir figures/extended-study
   ```

   Each command trains immutable chains, audits matched within-seed
   protocols and generation-0 artifacts, and writes per-run, paired, and
   aggregate CSVs plus figures. The base conditions are clean and full
   feedback. The extended conditions are clean, 90% feedback, full feedback,
   and matched oracle resampling. Use different nonexistent output paths if
   repeating a complete analysis.

4. Analyse the frozen models on separate diagnostic episodes:

   ```text
   python -m project1.omniglot_mechanism --base-runs runs/base-study --extended-runs runs/extended-study --output-dir figures/mechanism --size 256
   ```

5. Evaluate all validation-selected checkpoints on a fresh, larger,
   held-out test set:

   ```text
   python -m project1.omniglot_final_evaluation --base-runs-dir runs/base-study --extended-runs-dir runs/extended-study --prior-runs-dir runs --output-dir figures/final-evaluation --size 1024
   ```

   The evaluator excludes exact prompts from existing Omniglot test splits,
   saves a compact episode manifest and all checkpoint hashes, and writes
   paired and aggregate results. It does not train or select checkpoints on
   the new episodes. On our local workspace, we replayed the exact research
   manifest with:

   ```text
   python -m project1.omniglot_final_evaluation --feature-file ../Experiments/data/omniglot_resnet18_randomized_order_s0.h5 --base-runs-dir ../Experiments/runs/omniglot-base-study-20261009 --extended-runs-dir ../Experiments/runs/omniglot-extended-study-20261009 --prior-runs-dir ../Experiments/runs --manifest-file ../Experiments/figures/omniglot-final-evaluation-20261010/episodes.pt --output-dir figures/final-replay --size 1024
   ```

   That archive is not part of the submitted code package. The replay gave
   byte-identical per-run metrics after reorganization.

The commands create local `data/`, `runs/`, and `figures/` directories. They
are ignored by Git but must also be omitted when packaging this folder for
upload. The study commands use fixed seeds and settings defined in
`project1/omniglot_study.py`; exact run protocols are written beside their
outputs.

Our verified feature file is 16.6 MB. The full local research archive of run
artifacts occupies about 1.28 GB across exploratory and final experiments;
allow additional space for a fresh reproduction. We tested on CPU in WSL with
Python 3.13. A full 30-chain study is a substantial training job, so run the
smoke check first and budget compute time before starting it. We have not
recorded a portable full-study runtime benchmark.
