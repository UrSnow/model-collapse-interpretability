# Locked Omniglot Final Evaluation

> **Superseded (20 Oct 2026):** the primary results are now the tuned ten-generation study in
> `OMNIGLOT_TUNED_STUDY_REPORT.md`. Numbers below describe the three-generation pilot only.

10 October 2026. This report records the final held-out evaluation of the
validation-selected, frozen checkpoints from the aligned five-seed base and
extended studies. No model was retrained or chosen using these results.

## Protocol and audit

- Five seeds, three generations, two base conditions and four extended
  conditions: 90 checkpoints evaluated once on fresh episodes.
- Each seed has 1,024 held-out-class episodes, four times the 256-episode
  study test size. The same episode manifest is used for all conditions and
  both task families within a seed. The extended task additionally uses a
  fixed, independent uniform pointer to one of the two context features.
- Exact prompt identity comprises the ordered three class IDs and two
  context labels. The new 5,120 prompts are unique across seeds and exclude
  1,306 distinct prompts in earlier Omniglot test splits found locally.
  Prior split hashes and source paths are saved in `protocol.json`.
- The source feature file is checksum-verified. The compact `episodes.pt`
  manifest, its SHA-256, and all 90 frozen checkpoint SHA-256 values are saved
  alongside the results. The manifest reconstructs examples from the
  authors' feature file; no test-set fitting occurs.
- `audit.json` confirms 90 checkpoint rows, 75 paired contrast rows, 5,120
  final prompts, and zero prior-test prompt overlap. Sample standard
  deviations below describe variation over five seeds, not a confidence
  interval for all possible data or model settings.

## Final results

Generation-2 means on the fresh held-out set:

| Task | Condition | First-label accuracy | Second-label accuracy | Pointer TV from uniform |
| --- | --- | ---: | ---: | ---: |
| Base | Clean | 98.46% | n/a | n/a |
| Base | Feedback | 97.75% | n/a | n/a |
| Extended | Clean | 97.87% | 97.29% | 0.0134 |
| Extended | 90% feedback | 96.54% | 95.10% | 0.0843 |
| Extended | 100% feedback | 95.88% | 95.61% | 0.1018 |
| Extended | Oracle resampling | 96.68% | 96.45% | 0.0834 |

At G2, the mean paired feedback-minus-clean first-label difference is -0.70
percentage points for the base task (seed SD 0.51 points) and -1.99 points
for the extended task (seed SD 1.52 points). Both differences are negative in
all five seeds. The extended second-label difference is -1.68 points and the
pointer TV difference is +0.0884. The extended oracle-minus-clean first-label
difference is -1.19 points; feedback-minus-oracle is -0.80 points (seed SD
1.88 points), with mixed signs across seeds. The oracle control also has
substantial pointer drift, so the pointer shift is not uniquely attributable
to model-generated errors.

The final figure (`figures/omniglot-final-evaluation-20261010/final_comparison.png`)
plots means and seed SDs. Full per-run, paired, and aggregate CSVs are in the
same directory. Generation-0 checkpoints match within each seed; the G0
condition curves coincide.

## Interpretation and limits

The final test supports modest recursive performance degradation under this
finite training protocol, stronger on the extended task. It does **not** show
inevitable or catastrophic collapse in three generations: all G2 mean label
accuracies remain above 95%. The clean and oracle controls, limited generation
count, five seeds, fixed training episodes, and constrained pointer output
limit causal and generality claims. Mechanistic head evidence remains in
`OMNIGLOT_MECHANISM_REPORT.md`; these final behavioral results do not by
themselves identify a unique failing circuit. The context-feature pointer is
a documented adaptation of the next-image output, not literal image synthesis.

## Reproduction

After reproducing the two studies and downloading the verified feature file,
run from `Experiments/` with a fresh output directory:

```text
python -m project1.omniglot_final_evaluation \
  --base-runs-dir runs/omniglot-base-study-20261009 \
  --extended-runs-dir runs/omniglot-extended-study-20261009 \
  --prior-runs-dir runs \
  --output-dir figures/omniglot-final-evaluation-reproduction --size 1024
```

`--prior-runs-dir` scans available Omniglot split files and saves the exact
exclusion set provenance. To replay the *same* final set without all pilot
runs, add `--manifest-file figures/omniglot-final-evaluation-20261010/episodes.pt`.
Its episode IDs and pointer choices fully specify the test. Without that
flag, a different collection of prior splits can produce a different fresh
set; treat its numbers as a new evaluation.

Verification: all 47 project unit tests passed. An actual replay from
`episodes.pt` produced a byte-identical `per_run.csv` (SHA-256
`fcb0e92a45f120ecc616145701d7dbd1e5678e4742330914fe385ec9af55fcf5`).
