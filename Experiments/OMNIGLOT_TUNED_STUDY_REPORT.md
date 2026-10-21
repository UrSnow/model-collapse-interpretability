# Omniglot Tuned Ten-Generation Study

20 October 2026. This is the primary study for the extended abstract. It
replaces the three-generation pilot (`OMNIGLOT_MULTI_SEED_REPORT.md`,
`OMNIGLOT_MECHANISM_REPORT.md`, `OMNIGLOT_FINAL_EVALUATION_REPORT.md`), which
remains a documented, separately labelled pilot. The settings were selected
on generation-0 validation loss; see `OMNIGLOT_TUNING_AND_DIAGNOSTICS_REPORT.md`.

## Design

Protocol `tuned` (`project1.omniglot_study.PROTOCOLS`): Adam, learning rate
`1e-3`, up to 5,000 updates per generation with validation-loss checkpoint
selection every 100 updates, batch 64, 4,096 fixed training episodes, 256
episodes in each evaluation split, ten generations (G0-G9), five seeds.
Base conditions: clean and 100% model feedback. Extended conditions: clean,
90% feedback, 100% feedback, and 100% oracle resampling of valid pointer
targets. Within each seed, splits, G0 targets, and G0 checkpoints are
byte-identical across conditions (audited). Runs:
`runs/omniglot-{base,extended}-tuned-20261020/`; study figures and CSVs:
`figures/omniglot-{base,extended}-tuned-20261020/`. Training took 2 h 10 min
with seven parallel CPU workers.

## Fresh held-out final evaluation

All 300 frozen checkpoints were evaluated once on 1,024 new held-out-class
episodes per seed (5,120 prompts), excluding 6,426 prompts from every
earlier Omniglot test split and from the pilot's final manifest
(`--exclude-manifest`); `audit.json` reports zero overlap. Results:
`figures/omniglot-final-evaluation-tuned-20261020/`. Means and sample SDs
over five seeds:

| Task / condition | G0 | G5 | G9 |
| --- | ---: | ---: | ---: |
| Base clean | 98.71 | 99.04 | 98.30 |
| Base feedback | 98.71 | 98.75 | 98.54 |
| Extended clean | 99.10 | 98.89 | 99.24 |
| Extended oracle | 99.10 | 98.22 | 98.09 |
| Extended 90% feedback | 99.10 | 95.55 | 94.45 |
| Extended 100% feedback | 99.10 | 95.04 | 91.70 |

(First-label accuracy, %.) Paired G9 contrasts, percentage points:

| Contrast | Mean +/- SD | Per-seed |
| --- | ---: | --- |
| Base feedback - clean | +0.23 +/- 0.80 | mixed signs |
| Extended feedback - clean | -7.54 +/- 0.64 | all five negative (-6.6 to -8.3) |
| Extended feedback - oracle | -6.39 +/- 0.55 | all five negative (-5.8 to -7.2) |
| Extended oracle - clean | -1.15 +/- 0.53 | all five negative |
| Extended 90% feedback - clean | -4.79 +/- 2.15 | all five negative |

Extended second-label accuracy at G9 is 99.06% clean, 97.29% oracle, and
90.33% under full feedback. Pointer TV from uniform at G9 is 0.0015 clean,
0.1031 oracle, 0.1063 for 90% feedback, and 0.1526 for full feedback.

## Error accumulation and distribution

In the study runs, the fraction of incorrect base training labels under
feedback grows from 0.04% at G1 to 0.33% at G9. In the extended task under
full feedback, incorrect first labels among 4,096 training triples grow
super-linearly: 0.4, 19.8, 60.4, 101.2, 148.4, 204.6, 245.4, 322.4, 347.4
(G1-G9 means); second-label inconsistencies reach 446.0 at G9. The oracle
condition has none by construction. Class coverage stays at all 50
training classes and next-feature class entropy stays at 3.904-3.907 nats
(maximum ln 50 = 3.912) in every condition and generation. Degradation is
therefore in label binding, not in loss of symbol support.

## Training dynamics and checkpoint selection

Under sampled targets (feedback, partial, oracle), validation loss reaches
its minimum early: mean selected update at G9 is 720 (feedback), 660
(partial), and 940 (oracle), versus 3,860 for clean. Continued training
memorizes sampled pointers: validation pointer NLL rises from about 0.73-0.77
nats at the selected checkpoint to 2.1-3.3 nats at update 5,000. Early
selection does not create the feedback gap. At G9 the best validation
first-label accuracy reached at any evaluated update is 94.8% under
feedback versus 99.7% clean and 99.8% oracle; at update 5,000 feedback falls
to 86.3%. At G9 the selected feedback checkpoints fit only 94.5% of their own
training targets and 94.4% of the true training labels, whereas clean
checkpoints fit 100%. Validation uses clean labels, so checkpoint selection
gives every generation limited access to real data; the measured
degradation is conservative in this respect. The 5,000-update budget was
selected on clean G0, where the uniform pointer target cannot be memorized.

## Mechanism

Analyses use each chain's separate 256-episode diagnostic split
(`figures/omniglot-mechanism-tuned-20261020/`). Candidate heads are selected
per checkpoint by attention score, then ablated, retained alone, and
activation-patched from clean to label-corrupted prompts.

| Extended, G9 | Clean | Oracle | 90% feedback | 100% feedback |
| --- | ---: | ---: | ---: | ---: |
| Induction-head attention to matching label | 0.945 | 0.932 | 0.853 | 0.833 |
| Candidate-pair patch restoration | 0.664 | 0.633 | 0.464 | 0.314 |
| Best single-head patch restoration | 0.682 | 0.631 | 0.456 | 0.379 |
| Previous-token attention | 0.966 | 0.970 | 0.884 | 0.908 |

Per seed at G9, full feedback lowers induction attention relative to both
oracle and clean in all five seeds (-0.07 to -0.14) and lowers candidate-pair
patch restoration relative to both in all five seeds, with large variation
in size (-0.02 to -0.72 versus oracle). Previous-token attention is lower in
four of five seeds. The base task shows no consistent change (feedback minus
clean induction attention negative in one of five seeds). Across extended
G1-G9 checkpoints, induction attention correlates with diagnostic accuracy
(r = 0.56, n = 180); the checkpoints are linked within chains, so this is
descriptive. As in the pilot, retaining only the candidate pair gives
near-chance accuracy (0.21-0.28) and pair ablation removes only 2-12 points;
retrieval is distributed across heads. The evidence supports a weakened
distributed retrieval circuit under model feedback, not failure of one head.

## Inspection

`figures/omniglot-inspection-tuned-20261020/`. Query-exemplar transfer
remains limited: replacing the query with another exemplar of the matching
class gives 73-79% accuracy versus 93-99% for the identical exemplar at G9,
so models rely substantially on exact feature identity. Under full
feedback at G9, same-exemplar diagnostic accuracy is 93.4% and the mean
absolute label-embedding cosine is 0.083, versus 0.137 clean and 0.126
oracle. The direct-path OV copy score remains inconclusive (large seed SD)
and is not used as evidence.

## Interpretation and limits

Over ten generations, recursive model feedback leaves single-label
in-context retrieval essentially unchanged but produces compounding,
model-error-specific degradation when the model also generates the next
symbol and its label. The oracle control separates this from finite-sample
target noise, which costs about one point. The degradation coincides with
weaker induction attention and less restorable retrieval signal. Limits:
five seeds; fixed training episodes; the next symbol is restricted to the two
context features; queries repeat the exact context exemplar; validation-loss
selection uses clean labels; mechanism candidates are selected and tested on
the same diagnostic split; and ten generations do not establish an
asymptotic collapse rate.
