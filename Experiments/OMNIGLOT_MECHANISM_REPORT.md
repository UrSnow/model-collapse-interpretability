# Omniglot Mechanistic Diagnostics

## Question and method

Do recursive synthetic labels weaken the in-context retrieval mechanism, or
does the mechanism remain functional while output behavior changes? We tested
all 90 validation-selected checkpoints from the five-seed base and extended
studies (six condition/task groups, three generations, five seeds). Every
intervention uses the separate 256-episode diagnostic split, not the study
test split. Each model's candidate previous-token head is selected by mean
layer-1 attention from context-label positions to their preceding features;
its candidate induction head is selected by mean layer-2 query attention to
the matching context label. Head indices are selected within each model and
are never assumed to align across seeds or generations.

For each head we measured first-label accuracy and NLL changes when its output
was zeroed. The extended model also records second-label accuracy/NLL and
pointer-TV changes. We then removed both selected heads together, kept only
those heads while removing all others, and compared with all attention heads
removed. For activation patching, we replaced only the matching context
label with an unused label, captured clean per-head pre-projection attention
outputs, and patched them into the corrupted pass. Restoration is normalized
by the clean-to-corrupt log-probability gap for the original label on examples
where the corruption actually changed the model's prediction.

The runnable analysis is `project1.omniglot_mechanism`. Tables for every
checkpoint and every head, the aggregate table, protocol, and figure are in
`figures/omniglot-mechanism-20261010-reviewed/`.

## Generation-2 findings

Means across five seeds; accuracy drops below are absolute percentage points
on diagnostic examples, not held-out test effects.

| Model / condition | Top previous attention | Top induction attention | Pair ablation drop | Pair-only accuracy | Pair patch restoration |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base clean | 0.968 | 0.848 | 13.2 pp | 0.212 | 0.571 |
| Base feedback | 0.966 | 0.852 | 15.9 pp | 0.207 | 0.556 |
| Extended clean | 0.958 | 0.869 | 13.5 pp | 0.241 | 0.591 |
| Extended 90% feedback | 0.909 | 0.846 | 13.2 pp | 0.242 | 0.499 |
| Extended full feedback | 0.938 | 0.846 | 13.4 pp | 0.268 | 0.566 |
| Extended oracle resampling | 0.928 | 0.861 | 16.3 pp | 0.236 | 0.655 |

The corruption changed the original prediction on an average of 244-252 of
256 diagnostic episodes per group. Corrupted-prompt accuracy on the newly
correct label remained 0.948-0.973 at G2, so the intervention usually tests
a genuine context-dependent label switch. Pair ablation also increased NLL
(G2 group means 0.31-0.43 nats). Thus selected heads causally contribute to
retrieval, not merely display a high attention pattern. But pair-only
accuracy is near the five-label chance level of 0.20 and close to the
all-heads-ablated reference (0.19-0.21). This candidate pair is **not** a
minimal sufficient circuit; other heads or distributed pathways are needed.

Attention and causal importance are related but not interchangeable. At G2,
the highest previous-token-attention head tied for the largest first-label
ablation effect in only two of five extended clean models and two of five
extended full-feedback models. The highest induction-attention head did so
in five of five models in both groups. These are diagnostic selection results,
not a head-index identity across independently trained networks.

The candidate attention scores remain high under feedback. Pair ablation
effects and patch restoration remain substantial; neither shows a consistent
feedback-specific disappearance. Extended pair patch restoration falls from
0.775 at G0 to 0.591 clean and 0.566 full feedback at G2, but it also falls
under 90% feedback and changes under oracle resampling. This comparison does
not establish that model-generated errors selectively destroy the induction
mechanism. Together with the small behavioral differences in
`OMNIGLOT_MULTI_SEED_REPORT.md`, the evidence supports a functioning but
distributed retrieval mechanism and a possible output-distribution shift,
not proven mechanistic collapse.

## Limits and next decision

Candidates are selected and evaluated on the same diagnostic set, so the
reported candidate-pair effect is descriptive and may be selection-biased.
Patching is localized to context-label positions in layer 1 and the query
position in layer 2; it does not recover an entire computational graph.
Normalized restoration is not a proportion of task accuracy. The analysis
does not show a unique causal pathway, nor does it explain every output shift.
The frozen checkpoints were subsequently evaluated on a new, larger held-out
test set without further tuning; see `OMNIGLOT_FINAL_EVALUATION_REPORT.md`.
Keep attention heatmaps or embedding geometry as optional supporting
diagnostics unless they answer a specific remaining mechanism question.
