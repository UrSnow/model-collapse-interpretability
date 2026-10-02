# Project 1: Complete Project Checklist

Last updated: 1 October 2026.

This is the master checklist from implementation through submission. Steps 1-5
follow Section 1.4 of the project brief; Steps 6-8 cover packaging, final review,
and submission. Work may overlap, particularly experiments, interpretation, and
writing. Checked items record work already completed; unchecked items remain
outstanding. Update this file as milestones are verified.

## Dates and assessment priorities

- Final submission: **27 October 2026 at 17:00**, as stated in the brief.
- Preliminary contribution statement: two weeks before final submission,
  implying **13 October 2026**. Confirm its exact deadline and submission route
  on Moodle. 
- Main extended abstract: **two pages**, including figures, tables, and equations.
  References and the contribution statement may use additional pages.
- Rubric: method 30%, results 30%, discussion 20%, structure/introduction 10%,
  and background/related work 10%. Scientific reasoning and appropriate evidence
  matter more than achieving high accuracy or forcing collapse to appear.

## Step 1: Implement and validate the induction-head baseline

Status: implementation and exploratory experiments complete. Later-study
validation safeguards are applied; course approval of the symbolic adaptation
remains unconfirmed.

- [x] Implement the randomly relabelled symbol task.
- [x] Implement a two-layer causal transformer with inspectable attention.
- [x] Add training, checkpoint selection, metrics, and diagnostic utilities.
- [x] Run five initialization/data seeds and aggregate results.
- [x] Measure previous-token attention, induction attention, and head ablations.
- [x] Produce baseline plots and [STEP1_REPORT.md](STEP1_REPORT.md).
- [x] Document the discrete-symbol adaptation relative to the requested Singh task.
- [ ] Confirm that this adaptation satisfies the course requirements; the brief's
  Omniglot feature task is not replicated exactly.
- [x] Reconcile Step 1's report with the later audit: diagnostic logging used
  the test seed, and independent sampling seeds did not guarantee disjoint prompts.
- [x] Establish the final generation-0 baseline on the isolated Step 2 splits,
  with diagnostics separate from test data and validation-only model selection.

Completion evidence: reproducible baseline runs with documented task choices,
verified evaluation isolation, and conclusions proportionate to the circuit evidence.

## Step 2: Implement recursive training for base and extended tasks

Status: base-task pilot and extended-task matched study complete. Detailed protocol and evidence are in
[STEP2_REPORT.md](STEP2_REPORT.md).

- [x] Implement base-task recursive synthetic-label generation.
- [x] Support clean, fully synthetic, and mixed training conditions.
- [x] Start each generation with fresh weights and a fresh optimizer.
- [x] Keep dataset size fixed and create disjoint prompt splits.
- [x] Save labels, synthetic masks, seeds, checkpoints, and generation ancestry.
- [x] Verify unit tests and paired two-generation smoke chains.
- [x] Calibrate extended-task generation-0 convergence using validation data.
- [x] Finalize replacement/accumulation policy, synthetic fractions, sampling
  temperature, generation count, dataset size, and training budgets.
- [x] Implement the extended output: query label -> next symbol -> its label.
- [x] Define the next-symbol distribution and ensure the associated label is
  recoverable under the chosen task rules.
- [x] Define how generated continuations enter the next generation's data.
- [x] Test target shifting, causal masking, token support, and autoregressive sampling.
- [x] Verify clean-data learning for the extended task before recursive experiments.
- [x] Add soft and sampled true-generator resampling controls.
- [ ] Add restart/resume if a longer future study needs interruption recovery;
  the completed fixed-length runs use immutable output directories.
- [x] Freeze the protocol and run comparable conditions across multiple seeds.
- [x] Update STEP2_REPORT.md with the matched base-task pilot results,
  limitations, and commands.

Completion evidence: both required tasks run through reproducible generation
chains with controls. Smoke tests alone do not count as collapse experiments.

## Step 3: Analyse model performance and distribution changes

Status: base pilot and extended multi-seed analysis complete, including a fresh
final test. See [STEP3_REPORT.md](STEP3_REPORT.md).

- [x] Record baseline loss, accuracy, and perplexity across training.
- [x] Aggregate those metrics across generations, seeds, and conditions.
- [x] Evaluate against the fixed clean distribution; retain test data for final
  reporting after experimental choices and checkpoint selection are fixed.
- [x] Separate first-label accuracy, next-symbol distribution quality, and
  subsequent-label consistency for the extended task.
- [x] Use distributional metrics for a stochastic next symbol; do not assume
  one sampled continuation is the only correct prediction.
- [x] Measure coverage, entropy, dominant-symbol frequency, and distribution
  drift. Global infrequent-symbol performance is not a strong diagnostic in
  this balanced conditional task.
- [x] Quantify generated-label errors and autoregressive error propagation.
- [x] Report variability across independent chains with clear sample sizes.
- [x] Compare clean training, clean-data retention, and resampling controls.
- [x] Distinguish recursive degradation from undertraining and sampling artifacts.
- [x] Preserve the base pilot's absence-of-collapse result and its one
  generated-label error; test autoregressive propagation in the extended study.
- [x] Produce reproducible base and extended figures, tables, and a separate
  STEP3_REPORT.md.

Completion evidence: each behavioral claim has an appropriate metric, control,
and uncertainty statement supported by saved experiments.

## Step 4: Interpret the learned mechanisms

Status: attention metrics, ablations, patching, and representation diagnostics
complete. See [STEP4_REPORT.md](STEP4_REPORT.md) for causal limits.

- [x] Measure baseline previous-token and induction-pattern attention.
- [x] Implement baseline single-head output ablation.
- [x] Compare attention patterns across selected checkpoints and generations;
  within-generation attention histories were not saved in the extended study.
- [x] Investigate symbol representations with embedding effective rank.
- [x] Measure head-ablation effects on loss as well as accuracy.
- [x] Use grouped ablations to assess redundancy among heads.
- [x] Apply clean/corrupted activation patching to test candidate mechanisms.
- [x] Test selected-head sufficiency relative to other attention heads;
  the candidate pair is not sufficient.
- [x] Compare functional roles across models without assuming head indices align.
- [x] Relate circuit changes to distributional and behavioral changes.
- [x] Investigate low-scoring successful runs without declaring an alternative
  mechanism solely from attention scores.
- [x] Document observations, causal evidence, and unresolved questions separately
  in STEP4_REPORT.md.

Completion evidence: a defensible interpretation of how recursive training
affects ICL, with causal tests for stronger circuit claims. Patching and grouped
ablations are planned methods, not separately mandated algorithms in the brief.

## Step 5: Write the extended abstract

Status: not started. Step reports are working research notes, not the submission.

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

Status: documented and tested implementation; final submission packaging and
fresh-environment verification remain.

- [x] Provide an initial requirements.txt and README.txt.
- [x] Record tested dependency versions, Python version, and supported runtime.
- [x] Document setup, smoke tests, baseline training, recursive experiments,
  evaluation, and figure regeneration with exact commands.
- [ ] Document compute needs, expected runtimes, and storage requirements.
- [x] Include required configurations, seeds, data-generation instructions, and
  any small artifacts needed to reproduce the reported findings.
- [ ] Attribute reused code and data and check applicable usage conditions.
- [ ] Verify installation and the documented workflow in a fresh environment.
- [x] Run the regression suite and representative end-to-end checks (28 tests;
  45-checkpoint audit, interpretation, and fresh evaluation).
- [ ] Assemble a submission package without virtual environments, caches,
  temporary files, or unrelated experiments.

Completion evidence: another person can follow README.txt to run the code and
reproduce the reported workflow from the submitted materials.

## Step 7: Complete the final academic and submission review

Status: pending. Complete administrative tasks by their own deadlines even
while earlier technical steps remain in progress.

- [ ] Finalize the group of 1-3 people.
- [ ] Submit the preliminary contribution statement by its confirmed deadline.
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

Status: pending. Deadline in the brief: **27 October 2026 at 17:00**.

- [ ] Upload the full code implementation through the required course channel.
- [ ] Include requirements.txt and README.txt.
- [ ] Upload the extended abstract PDF with required statements and appendices.
- [ ] Confirm all required files are present and the submission is finalized.
- [ ] Retain the submission receipt or confirmation and timestamp.
- [ ] Verify the submitted files can be opened or downloaded correctly.
- [ ] Record submission completion here only after successful confirmation.

Completion evidence: confirmed submission of the reviewed package before the deadline.
