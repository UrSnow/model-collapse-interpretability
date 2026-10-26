# Mechanistic Interpretability of Model Collapse

We study how recursive training on model-generated targets changes in-context
learning on the Omniglot-feature task. The base model predicts a query label.
Our extended model also predicts a pointer to a context feature and that
feature's label. This pointer is a constrained next-symbol representation; it
does not generate a new image.

## Submission code

[`submission/`](submission/README.txt) contains the aligned implementation,
tests, pinned requirements, and a package README. Follow
[`RUN_STEPS.md`](submission/RUN_STEPS.md) to run the tests, download the
checksum-verified public feature data, reproduce the studies, and generate
figures. The extended abstract and declarations are submitted separately.

## Research archive

`Experiments/` preserves exploratory code, selected figures, and the two
group-member notebooks. Those notebooks investigate discrete-token versions
of the task; their numerical results are not presented as Omniglot results.
Our working Markdown reports and project checklist remain local and are not
part of the GitHub or submission package. Raw runs, downloaded features,
course materials, and the historical presentation also stay local.
