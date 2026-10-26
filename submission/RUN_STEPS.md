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

3. Select the learning rate and update budget on generation-0 validation
   loss:

   ```text
   python -m project1.omniglot_sweep --runs-dir runs/sweep --output-dir figures/sweep --workers 4 --threads 2
   ```

   This trains one clean generation per learning rate (1e-4 to 3e-3), seed
   (0-2), and task for 5,000 updates, then reads the best validation loss
   within 1,000-5,000 updates from each history. It writes `summary.csv`,
   `selection.json`, and validation curves. Test metrics are not used.

4. Reproduce our controlled five-seed studies with the selected setting:

   ```text
   python -m project1.omniglot_base_study --protocol tuned --workers 2 --runs-dir runs/base-study --output-dir figures/base-study
   python -m project1.omniglot_study --protocol tuned --workers 5 --runs-dir runs/extended-study --output-dir figures/extended-study
   ```

   Protocol `tuned` uses learning rate 1e-3, 5,000 updates, and ten
   generations. Protocol `pilot` reproduces our earlier three-generation,
   1,000-update study. `--workers` trains independent chains in parallel;
   generations within a chain remain sequential. Each command audits matched
   within-seed protocols and generation-0 artifacts, and writes per-run,
   paired, and aggregate CSVs plus figures. The base conditions are clean
   and full feedback. The extended conditions are clean, 90% feedback, full
   feedback, and matched oracle resampling. Use different nonexistent output
   paths if repeating a complete analysis.

5. Plot training dynamics and score frozen checkpoints on their training
   episodes, then inspect attention maps, embeddings, and query-exemplar
   transfer:

   ```text
   python -m project1.omniglot_curves --base-runs-dir runs/base-study --extended-runs-dir runs/extended-study --output-dir figures/training-dynamics
   python -m project1.omniglot_inspect --base-runs-dir runs/base-study --extended-runs-dir runs/extended-study --output-dir figures/inspection
   ```

6. Analyse the frozen models with ablation and activation patching on
   separate diagnostic episodes:

   ```text
   python -m project1.omniglot_mechanism --base-runs runs/base-study --extended-runs runs/extended-study --output-dir figures/mechanism --size 256
   ```

7. Evaluate all validation-selected checkpoints on a fresh, larger,
   held-out test set:

   ```text
   python -m project1.omniglot_final_evaluation --base-runs-dir runs/base-study --extended-runs-dir runs/extended-study --prior-runs-dir runs --output-dir figures/final-evaluation --size 1024
   ```

   The evaluator excludes exact prompts from existing Omniglot test splits
   and from any manifest passed with `--exclude-manifest`, saves a compact
   episode manifest and all checkpoint hashes, and writes paired and
   aggregate results. It does not train or select checkpoints on the new
   episodes. On our local workspace, we replayed the exact pilot manifest
   with:

   ```text
   python -m project1.omniglot_final_evaluation --feature-file ../Experiments/data/omniglot_resnet18_randomized_order_s0.h5 --base-runs-dir ../Experiments/runs/omniglot-base-study-20261009 --extended-runs-dir ../Experiments/runs/omniglot-extended-study-20261009 --prior-runs-dir ../Experiments/runs --manifest-file ../Experiments/figures/omniglot-final-evaluation-20261010/episodes.pt --output-dir figures/final-replay --size 1024
   ```

   That archive is not part of the submitted code package. The replay gave
   byte-identical per-run metrics after reorganization and after adding
   named protocols.

8. Draw the abstract figure from the final-evaluation and mechanism outputs:

   ```text
   python -m project1.omniglot_abstract_figure --final-dir figures/final-evaluation --mechanism-dir figures/mechanism --output figures/abstract/main_results
   ```

The commands create local `data/`, `runs/`, and `figures/` directories. They
are ignored by Git but must also be omitted when packaging this folder for
upload. If the feature file is elsewhere, pass `--feature-file PATH` to the
sweep, study, inspection, and final-evaluation commands. The study commands
use fixed seeds and the named protocols defined in
`project1/omniglot_study.py`; exact run protocols are written beside their
outputs.

Our verified feature file is 16.6 MB. The full local research archive of run
artifacts occupies about 1.28 GB across exploratory and final experiments;
allow additional space for a fresh reproduction. We tested on CPU in WSL with
Python 3.13 on an 8-core machine. One 5,000-update generation took about
150-200 seconds per single-threaded job with several jobs in parallel. The
sweep (24 runs, four workers) took about 20 minutes. The tuned studies train
30 ten-generation chains (300 generations); with seven parallel workers
(five extended, two base) they took 2 hours 10 minutes. Run the smoke check
first.
