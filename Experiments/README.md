# Research archive (not for submission)

This folder preserves our exploratory and validation work. It includes the
two group-member notebooks in `notebooks/`, the original full `project1/`
package and `tests/`, run artifacts in `runs/`, data in `data/`, figures in
`figures/`, working reports, reference material, and the progress presentation.

Our early discrete-token experiments are recorded in `STEP1_REPORT.md` through
`STEP4_REPORT.md`. The aligned Omniglot work is documented in
`OMNIGLOT_PROTOCOL.md`, `OMNIGLOT_EXTENDED_REPORT.md`,
`OMNIGLOT_MULTI_SEED_REPORT.md`, `OMNIGLOT_MECHANISM_REPORT.md`, and
`OMNIGLOT_FINAL_EVALUATION_REPORT.md`. We do not present token-task outcomes
as Omniglot outcomes.

Run archived commands from this directory. From the `Project1/` root, enter
it with `cd Experiments`. The existing `README.txt` is the
original research log and uses paths relative to this directory. The
presentation script also uses its own directory as its path root:

```powershell
wsl.exe ../.venv-wsl/bin/python -m unittest discover -s tests
powershell -ExecutionPolicy Bypass -File build_progress_presentation.ps1
```

The archived `runs/` and feature file are local research artifacts, not files
to upload with the submission. The old PDF in `past_drafts/` predates the
revised equal-contribution statement and must not be submitted.
The member notebooks are included in GitHub as an archive of exploratory
work, not as aligned Omniglot results or part of the final submission package.
The historical presentation stays local.
