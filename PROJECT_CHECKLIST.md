# Project 1: Complete Project Checklist

Last updated: 12 October 2026 (deadlines corrected).

This is the master checklist from implementation through submission. Steps 1-5
follow Section 1.4 of the project brief; Step 4A tracks group-member method
integration; Steps 6-8 cover packaging, final review, and submission. Work may
overlap, particularly experiments, interpretation, and writing. Checked items
record work already completed; unchecked items remain outstanding. Update this
file as milestones are verified.

## Current alignment snapshot

- [x] Separate the research archive in `Experiments/` from the runnable
  `submission/` package. The submission copy passes 20 focused tests,
  two real-feature smoke chains, and an exact final-evaluation replay.

- [x] Base and extended implementations now use Singh et al.'s Omniglot
  features, with the extended next symbol represented by a context-feature
  pointer. The pointer is a documented constraint, not arbitrary image output.
- [x] Matched seed-0 clean, feedback, and oracle-resampling pilots completed;
  splits and generation-0 checkpoints were audited as identical.
- [x] Aligned pilot metrics, figures, and diagnostic head ablations are saved
  in [OMNIGLOT_EXTENDED_REPORT.md](Experiments/OMNIGLOT_EXTENDED_REPORT.md) and
  `Experiments/figures/omniglot-extended-controlled-seed0/`.
- [x] Run and audit five-seed aligned base and extended controlled studies;
  see [OMNIGLOT_MULTI_SEED_REPORT.md](Experiments/OMNIGLOT_MULTI_SEED_REPORT.md).
- [x] Complete stronger mechanistic tests and a fresh final evaluation;
  see [OMNIGLOT_MECHANISM_REPORT.md](Experiments/OMNIGLOT_MECHANISM_REPORT.md)
  and [OMNIGLOT_FINAL_EVALUATION_REPORT.md](Experiments/OMNIGLOT_FINAL_EVALUATION_REPORT.md).
  The evidence does not establish inevitable collapse.
- [ ] Complete the official-template two-page abstract, ethics statements,
  final package review, and submission.

## Development and results gate

- [x] Keep the Singh et al. feature-based base task and the explicitly
  constrained context-feature continuation in the submitted implementation.
- [x] Keep our PyTorch implementation separate from the paper authors' code;
  download and checksum-verify their published features without committing
  the data file.
- [x] Keep course PDFs, the template, stale contribution drafts, and the
  unapproved member statement out of the GitHub payload.
- [x] Stage and inspect the file list: no course references, raw runs, feature
  data, or unapproved contribution statement are staged; whitespace check passes.
- [x] Keep both original member notebooks and the historical presentation
  local, per the group's decision; exclude them from the staged GitHub payload.
- [x] Review the final staged payload before committing or pushing; no member
  notebooks, presentation, course PDFs, raw data, or statement draft are staged.
- [x] Recheck the reported generation-2 table against the saved aggregate CSV;
  the frozen evaluation audit records 90 checkpoints and zero prompt overlap.
- [x] Visually inspect the final comparison figure and check representative
  base/extended run protocols against the frozen evaluation protocol.
- [ ] Review every numerical claim chosen for the abstract against the saved
  result row and figure before freezing the final PDF.
- [x] Rerun the 20 focused submission tests after the repository cleanup.
- [x] Install the CPU-only requirements in a fresh WSL Python 3.13 environment,
  run 20 tests, and complete real-feature base and extended smoke chains.
- [x] Record the measured feature-file and research-run storage footprint in
  `submission/RUN_STEPS.md`.
- [ ] Benchmark a complete study if needed for a useful runtime estimate; the
  short smoke runs do not establish full-study duration.
- [x] Complete the implementation and saved-results verification gate. The
  remaining claim audit belongs to the abstract, not a new training run.

## Dates and assessment priorities

1. **By 14 October:** obtain all members' approval of the preliminary
   contribution statement and submit it through the required course channel.
   This deadline runs in parallel with the technical review.
2. **Next:** finish code, reproducibility, result, and GitHub payload review.
   Keep the constrained next-symbol representation explicit and do not claim
   an exact numerical reproduction of Singh et al.
3. **Then:** write and verify the extended abstract using the checked results
   and official template, and complete the group review of the exact package.

- Preliminary contribution statement deadline: **14 October 2026**.
- Final project deadline: **30 October 2026**. These group-confirmed dates
  supersede the older date printed in the PDF brief; confirm the submission
  time and upload route from the current course notice.
- Main extended abstract: **two pages**, including figures, tables, and equations.
  References and the contribution statement may use additional pages.
- Rubric: method 30%, results 30%, discussion 20%, structure/introduction 10%,
  and background/related work 10%. Scientific reasoning and appropriate evidence
  matter more than achieving high accuracy or forcing collapse to appear.

## Step 1: Implement and validate the induction-head baseline

Status: the original discrete-token study is historical. A separate
Omniglot-feature baseline is implemented and final-tested across five matched
seeds. See [OMNIGLOT_PROTOCOL.md](Experiments/OMNIGLOT_PROTOCOL.md).

- [x] Implement the randomly relabelled symbol task.
- [x] Implement a two-layer causal transformer with inspectable attention.
- [x] Add training, checkpoint selection, metrics, and diagnostic utilities.
- [x] Run five initialization/data seeds and aggregate results.
- [x] Measure previous-token attention, induction attention, and head ablations.
- [x] Produce baseline plots and [STEP1_REPORT.md](Experiments/STEP1_REPORT.md).
- [x] Document why the discrete-symbol adaptation is not the exact requested
  input task; keep its results separate from Omniglot-feature evidence.
- [x] Load and checksum-verify the authors' Omniglot features.
- [x] Implement feature-label episodes with disjoint relabeling pairs and
  held-out test classes, plus a two-layer attention-only RoPE model.
- [x] Calibrate one feature-based generation-0 run using validation selection;
  verify held-out-class learning and a separate diagnostic split.
- [ ] Reconcile Step 1's report with the later audit: diagnostic logging used
  the test seed, and independent sampling seeds did not guarantee disjoint
  prompts. Label these as limitations of the historical token runs.
- [x] Replicate the Omniglot generation-0 baseline across five seeds and
  record between-seed variability; do not confuse this with the final test.

Completion evidence: reproducible baseline runs with documented task choices,
verified evaluation isolation, and conclusions proportionate to the circuit evidence.

## Step 2: Implement recursive training for base and extended tasks

Status: base and extended feature-based recursive implementations and matched
five-seed studies complete. The older [STEP2_REPORT.md](Experiments/STEP2_REPORT.md)
concerns the token adaptation; see
[OMNIGLOT_MULTI_SEED_REPORT.md](Experiments/OMNIGLOT_MULTI_SEED_REPORT.md) for aligned evidence.

- [x] Implement base-task recursive synthetic-label generation.
- [x] Support clean, fully synthetic, and mixed training conditions.
- [x] Start each generation with fresh weights and a fresh optimizer.
- [x] Keep dataset size fixed and create disjoint prompt splits.
- [x] Save labels, synthetic masks, seeds, checkpoints, and generation ancestry.
- [x] Verify unit tests and paired two-generation smoke chains.
- [x] Calibrate finite-dataset generation-0 convergence using validation data
  for the Omniglot base and extended models.
- [x] Lock the study's replacement policy, fractions, sampling temperature,
  generation count, dataset size, and training budget in saved protocols.
- [x] Implement the feature-based extended output: query label -> pointer to
  a context Omniglot feature -> that feature's context label.
- [x] Define the uniform context-feature choice and recover its assigned label.
- [x] Replace complete generated triples on selected fixed training rows.
- [x] Test causal masking, pointer support, target consistency, and
  autoregressive sampling.
- [x] Verify clean-data learning for the extended task before recursion.
- [x] Add oracle pointer resampling with the matched sampled-loss control.
- [ ] Add restart/resume only if longer runs need interruption recovery;
  current runs intentionally use immutable output directories.
- [x] Freeze the protocol and audit comparable conditions across five seeds.
- [x] Record the aligned single-seed protocol, results, limitations, and
  commands separately from the historical token report.

Completion evidence: both required tasks run through reproducible generation
chains with controls. Smoke tests alone do not count as collapse experiments.

## Step 3: Analyse model performance and distribution changes

Status: aligned five-seed cross-generation tables, controlled figures, and
the fresh final evaluation are complete.

- [x] Record baseline loss, accuracy, and perplexity across training.
- [x] Compare three generations across matched clean, full-feedback, and
  oracle-resampling conditions for seed 0.
- [x] Aggregate aligned metrics across five seeds and controlled conditions.
- [x] Evaluate frozen checkpoints on a fresh larger held-out clean
  distribution after experimental choices and checkpoint selection were fixed.
- [x] Separate first-label accuracy, next-feature distribution quality, and
  subsequent-label consistency for the extended task.
- [x] Use distributional metrics for a stochastic next feature; do not assume
  one sampled continuation is the only correct prediction.
- [x] Measure class coverage, entropy, dominant-class frequency, and
  distribution drift; no rare-class claim is supported by this uniform pilot.
- [x] Count generated first-label errors, second-label inconsistencies, and
  free-running greedy joint accuracy.
- [x] Report paired seed-level differences, means, sample standard deviations,
  and the 256-episode-per-seed evaluation size.
- [x] Compare clean, full-feedback, and oracle-resampling pilot conditions.
- [x] Compare 90% feedback / 10% clean retention across five seeds.
- [x] Compare feedback against clean and oracle-resampling controls;
  retain undertraining and sampling effects as limitations.
- [x] Preserve and report the pilot's absence of clear monotonic collapse.
- [x] Produce audited aligned multi-seed tables, figures, and a separate report.

Completion evidence: each behavioral claim has an appropriate metric, control,
and uncertainty statement supported by saved experiments.

## Step 4: Interpret the learned mechanisms

Status: aligned attention, single/grouped ablation, sufficiency, and
clean/corrupt patching diagnostics are complete across five seeds. Optional
representation and heatmap exploration remains conditional on the abstract.

- [x] Measure baseline previous-token and induction-pattern attention.
- [x] Implement baseline single-head output ablation.
- [x] Run per-head first- and second-label accuracy ablations on the aligned
  extended models using a separate diagnostic split.
- [x] Compare candidate attention roles across generations and conditions;
  within-training attention histories were not captured.
- [ ] Investigate feature representations only if a specific mechanism
  question warrants it; token PCA does not directly transfer to Omniglot.
- [x] Measure head-ablation effects on loss as well as accuracy.
- [x] Use grouped ablations to assess redundancy among selected heads.
- [x] Apply clean/corrupted activation patching to test candidate mechanisms.
- [x] Test candidate circuit sufficiency; the selected pair is insufficient.
- [x] Compare functional roles across models without assuming head indices align.
- [x] Relate circuit diagnostics to behavior without claiming a
  feedback-specific failure unsupported by controls.
- [x] Check attention against causal ablation importance; do not infer a
  unique circuit from attention scores alone.
- [x] Document observations and limits in
  [OMNIGLOT_MECHANISM_REPORT.md](Experiments/OMNIGLOT_MECHANISM_REPORT.md).

Completion evidence: a defensible interpretation of how recursive training
affects ICL, with causal tests for stronger circuit claims. Patching and grouped
ablations are planned methods, not separately mandated algorithms in the brief.

## Step 4A: GROUP MEMBERS Adopted contribution

Status: the topic-relevant methods from both notebooks have been adapted into
the aligned Omniglot project. Optional visualizations remain conditional on
the abstract. These checkboxes track method integration, not exclusive credit.

**Member 1:** local notebook `Experiments/notebooks/NLP_Project_One.ipynb`

- [x] Adopt the recursive-generation experiment structure: train a fresh model
  on predecessor-generated targets and compare later generations with G0.
  Implemented in the aligned base and extended runners.
- [x] Track generation-level loss, accuracy, and perplexity in saved histories
  and summaries; the aligned comparison figure already plots accuracy.
- [ ] Add aligned training/validation loss and perplexity trajectory plots
  across generations and conditions.
- [x] Adapt autoregressive generation of the extended outputs. The aligned
  implementation samples a query label, a context-feature pointer, and its
  corresponding predicted label, rather than generating arbitrary token strings.
- [x] Save previous-token and induction-style attention scores per generation.
- [x] Compare candidate head scores across generations and conditions using
  the frozen checkpoints; add heatmaps only if they clarify the final claim.

**Member 2:** local notebook
`Experiments/notebooks/NLP_Project_Final_Parts_1_to_4.ipynb`

- [x] Adopt teacher-forced three-output training and separate first-label,
  next-feature, and second-label evaluation. Use pointer-distribution metrics
  instead of exact-match accuracy for a stochastic next feature.
- [x] Adapt fixed train/validation/test episodes, validation-selected
  checkpoints, and held-out evaluation in the aligned runners.
- [x] Track prediction entropy and distributional diversity together with
  accuracy and loss. The aligned pilot records pointer entropy, class entropy,
  coverage, and drift.
- [x] Adapt per-head attention diagnostics and causal head ablations. The
  aligned analysis uses a separate diagnostic split and saves per-head effects.
- [x] Add explicit change-from-G0 and paired treatment-versus-clean tables
  with between-seed variability in the aligned study analysis.
- [ ] Adapt the embedding-geometry/PCA inspection to Omniglot feature and
  label embeddings, with a clear question and quantitative companion metric.
- [x] Combine attention scores, ablation effects on accuracy **and loss**, and
  across-generation functional comparisons in one aligned circuit analysis.
- [x] Run a controlled 90% partial-feedback condition alongside clean,
  full-feedback, and oracle runs across five seeds.

- [x] Confirm with both members that this checklist accurately describes their
  methods selected for adaptation.
- [x] Obtain each member's approval of their contribution description in the
  revised equal-contribution preliminary statement in `submission/`.
- [ ] Proofread the complete statement with the group, verify names and student
  numbers, and confirm the final PDF before the 14 October submission.

Do not import the notebooks' discrete-token architectures or their numerical
results as if they were Singh et al. Omniglot experiments. The adopted pieces
are methods and analysis ideas adapted to the feature-based task. Neither
notebook's observations alone establish inevitable model collapse.

Completion evidence: the adopted methods are tested on aligned Omniglot runs,
with member-approved attribution in the contribution statement.

## Step 5: Write the extended abstract

Status: pending the development and results gate. Step reports are working
research notes, not the submission.

- [ ] Obtain the official extended-abstract template from Moodle.
- [ ] Choose one clear research question and the strongest supported findings.
- [ ] Write concise introduction and relevant background with verified citations.
- [ ] Describe architecture, data, splits, generation policy, controls, tuning,
  seeds, and evaluation sufficiently for the results to be understood.
- [ ] Include readable figures/tables that directly support the research question.
- [ ] Discuss mechanisms, limitations, alternative explanations, and informative
  negative results; avoid claiming that collapse must occur in a short chain.
- [ ] Keep the main content within two pages, including all figures and equations.
- [ ] Add references and the final contribution statement on additional pages.
- [ ] Complete the Moodle NeurIPS 2024 ethics questionnaire and place it after
  references and before supplementary material. Its absence prevents marking.
- [ ] Include the required Faculty of Science AI Ethics Statement.
- [ ] Check every numerical claim against saved outputs and every citation
  against its source; review writing and formatting carefully.
- [ ] Render and inspect the final PDF for page count, readability, and clipping.

Completion evidence: a concise, fact-checked academic abstract PDF with the
required declarations, following the official template.

## Step 6: Package the reproducible implementation

Status: the aligned code and focused tests are isolated in `submission/`, with
its own README and run steps. Final abstract and package review remain.

- [x] Provide `submission/requirements.txt` and `submission/README.txt`.
- [x] Document the feature download, aligned base/extended pilot commands,
  and controlled figure regeneration in `submission/RUN_STEPS.md`.
- [x] Run 20 submission tests, 47 archive tests, real-feature base/extended
  smoke chains, and a byte-identical archived final-evaluation replay.
- [x] Record tested dependency versions, Python version, and supported runtime
  in `submission/README.txt`.
- [x] Document setup, smoke tests, baseline training, recursive experiments,
  evaluation, and figure regeneration with exact commands.
- [ ] Document compute needs, expected runtimes, and storage requirements.
- [ ] Include required configurations, seeds, data-generation instructions, and
  any small artifacts needed to reproduce the reported findings.
- [ ] Attribute reused code and data and check applicable usage conditions.
- [x] Verify installation, all 20 tests, and both real-feature smoke commands
  in a fresh WSL Python 3.13 environment.
- [x] Run the current regression suites and representative end-to-end checks.
- [x] Separate a provisional submission package from virtual environments,
  caches, raw runs, temporary files, and unrelated experiments.

Completion evidence: another person can follow `submission/README.txt` to run the code and
reproduce the reported workflow from the submitted materials.

## Step 7: Complete the final academic and submission review

Status: pending. Complete administrative tasks by their own deadlines even
while earlier technical steps remain in progress.

- [ ] Finalize the group of 1-3 people.
- [x] Create a preliminary contribution statement LaTeX draft.
- [x] Confirm all members are happy with their contribution descriptions.
- [ ] Proofread and approve the complete preliminary statement PDF.
- [ ] Submit the preliminary contribution statement by **14 October 2026**.
- [ ] Keep a contribution record and finalize the statement for the report.
- [ ] Check the final work against all mandatory brief requirements and rubric items.
- [ ] Verify both base and extended model implementations are included.
- [ ] Ensure reported metrics, captions, run configurations, and conclusions agree.
- [ ] Confirm ethics declarations, code attribution, and AI-use disclosure are complete.
- [ ] Confirm Moodle's packaging, naming, file-size, deadline, and upload requirements.
- [ ] Open the exact packaged PDF and verify the exact packaged code and README.
- [ ] Obtain final review from all group members and preserve a local final copy.

Completion evidence: the complete package has been reviewed and is ready to upload.

## Step 8: Submission

Status: pending. Group-confirmed deadline: **30 October 2026**.
Confirm the time and upload route from the current course notice.

- [ ] Upload the full code implementation through the required course channel.
- [ ] Include requirements.txt and README.txt.
- [ ] Upload the extended abstract PDF with required statements and appendices.
- [ ] Confirm all required files are present and the submission is finalized.
- [ ] Retain the submission receipt or confirmation and timestamp.
- [ ] Verify the submitted files can be opened or downloaded correctly.
- [ ] Record submission completion here only after successful confirmation.

Completion evidence: confirmed submission of the reviewed package before the deadline.
