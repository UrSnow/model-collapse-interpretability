# Step 1: In-Context Learning Through Induction Heads

## Status

Step 1 is complete. A two-layer causal transformer learns a randomly relabelled
symbol task, is evaluated on deterministic and separate train, validation, and
test streams, and exposes its attention heads for mechanistic measurement and
causal ablation.

## Research question

Can a small transformer learn a prompt-specific symbol-to-label mapping, and
does successful learning coincide with the formation of the canonical
previous-token and induction-head attention patterns?

## Task

Every sequence has two context pairs followed by a query symbol:

```text
s0 l0 s1 l1 query_symbol
```

The two symbols and two labels are unique within a prompt. Their mapping is
resampled independently for every example. The query repeats one context
symbol, and the target is its associated label. With five labels, uninformed
guessing gives 20% accuracy. A model that identifies the two labels in the
prompt but cannot match the query to the correct pair achieves 50% in
expectation.

## Model and training

| Item | Value |
|---|---:|
| Symbols | 50 |
| Labels | 5 |
| Context pairs | 2 |
| Transformer layers | 2 |
| Attention heads per layer | 8 |
| Residual dimension | 64 |
| MLP expansion | 4 |
| Batch size | 128 |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Training steps | 10,000 |
| Independent seeds | 5 |

Validation loss selects one checkpoint per seed. Hyperparameters and the
evaluation schedule are fixed across seeds. The locked test stream is evaluated
after checkpoint selection and is not used for optimization.

## Measurements

Behavior is measured using cross-entropy, perplexity, and accuracy. Circuit
formation is monitored using:

- Previous-token score: layer-1 attention from each label position to its
  immediately preceding symbol.
- Induction score: layer-2 attention from the query to the context label that
  follows the matching symbol.
- Head ablation: change in accuracy after zeroing one attention head's output.

The first validation checkpoint at or above 90% accuracy is reported as the
behavioral phase-change step. This is an operational threshold, not a claim
that the transition is mathematically discontinuous.

## Results

All five runs learned the task. Mean locked-test accuracy was 99.80% with a
sample standard deviation of 0.27 percentage points. Individual test accuracy
ranged from 99.45% to 100% and mean test perplexity was 1.0057.

| Seed | Phase step | Test accuracy | Best previous-token | Best induction |
|---:|---:|---:|---:|---:|
| 0 | 5,500 | 99.56% | 0.993 | 0.984 |
| 1 | 7,500 | 100.00% | 0.983 | 0.988 |
| 2 | 4,000 | 100.00% | 0.876 | 0.688 |
| 3 | 2,500 | 100.00% | 0.969 | 0.975 |
| 4 | 6,500 | 99.45% | 0.450 | 0.285 |

The mean phase-change step was 5,200, with a range from 2,500 to 7,500. Runs
typically remained near the 50% label-set strategy before rapidly improving.
In seeds 0, 1, and 3, behavioral improvement coincided with a sharp rise in both
canonical circuit scores. Seed 2 learned with moderately distributed attention.

Seed 4 reached 99.45% test accuracy without high canonical attention scores.
Its most important single head caused a 36.3 percentage-point accuracy loss
when ablated, suggesting a concentrated contribution. This does not identify
an alternative circuit: low attention scores and one-head ablation are
insufficient to establish one. Task accuracy alone does not prove that a
specific induction circuit was learned.

## Conclusion

The architecture reliably learns in-context symbol matching and usually forms
the expected previous-token and induction patterns. Circuit identity and
formation time vary materially across initialization seeds. This variability is
important for the model-collapse study: later generations must be compared to
their own generation-0 circuit baselines, rather than to one fixed head index or
one seed's attention pattern.

## Limitations

- Symbols are learned discrete embeddings rather than Omniglot image features.
- Attention-pattern scores are descriptive and do not by themselves prove a
  complete circuit.
- Single-head ablations can understate causal importance when heads are
  redundant.
- Only the base one-label task is addressed; recursive synthetic training and
  the extended output task belong to Step 2.
- Step 1 used independent stream seeds but did not enforce prompt disjointness.
  Online diagnostics also used the test seed. Its results are exploratory,
  not the final isolated evaluation. Steps 2-4 use disjoint hash-partitioned
  prompts and separate diagnostics; the extended study additionally uses a
  fresh final test set excluding all earlier prompts.

## Reproduction

```text
python -m unittest discover -s tests -v
python -m project1.sweep --seeds 0,1,2,3,4
python -m project1.analyze runs/step1-multiseed
```

The aggregated data and figure are in `runs/step1-multiseed/analysis`.
