# Evaluator qualification workflow

`evaluator_qualification_workflow.v1` is the fail-closed composition boundary
for a public, evaluator-bounded sim policy-ranking run. It does not launch a
provider. Paid allocations remain available only through
`blueprint_pipeline.paid_resource_allocator` after a current budget/admission
artifact authorizes the exact release.

The workflow independently revalidates the existing versioned contracts for:

- at least seven distinct policy/checkpoint adapters with exact action
  dimensions, units, rates, timestamps, bounds, and normalization;
- at least four independently captured, evaluation-admitted sites sharing one
  frozen split, including an entire held-out site (three sites remain
  integration smoke);
- at least 20 identical site/task/condition/seed trials for every policy;
- one exact runtime receipt and evaluator row for every registered cell;
- model-derived media validity, full ordered episode assembly, authoritative
  manifest completion, criterion results, and decision-grade ranking;
- provider allocation, authenticated buyer delivery, exact-attempt and global
  provider teardown, and billing-export reconciliation.

## Release and evidence binding

The request binds the source commit, source archive, release manifest,
container image, model-set manifest, and frozen data split. The model-set
manifest is the canonical digest of the exact policy artifacts, checkpoints,
and evaluator model used by the matrix. Site split manifests must match the
release split digest, provider allocations must match the release container,
and delivery must reference the canonical ranking-result digest.

Each runtime normalization request is re-executed through
`normalize_evaluator_runtime_evidence()`. The normalized action, observation,
policy output, evaluator request/checkpoint/output, next query, action-control,
criterion, provider execution, and authoritative manifest digests must match
the registered policy cell. A declared completed result is never trusted as a
substitute for these derivations.

## Independent lifecycle states

The output keeps these states separate:

1. request acceptance
2. site admission
3. policy registry
4. provider allocation
5. model execution
6. episode artifact assembly
7. media validity
8. evaluator validity
9. criterion result
10. rank result
11. delivery
12. teardown
13. billing reconciliation

It reports a scientific sim-ranking verdict separately from the public-launch
verdict. Thus a valid ranking can remain visible when delivery, teardown, or
billing is blocked, while the public-launch verdict stays blocked. Allocation
inventory never implies billing reconciliation, and model/media success never
overrides a blocked authoritative episode manifest.

## Model and benchmark boundary

Generic evaluators, OSCAR/RoboArena-inspired adapters, SC3-inspired scorers,
Cosmos, the model-neutral `roboworld_progress_v1` rubric/view-authority profile,
and future world models use replaceable versioned profiles. The RoboWorld-inspired
profile can grade progress and model-error stage for any admitted WAM; it does
not require or imply a Step Forcing backend. Compute
providers are execution locations, not evaluator identities. SC3-Eval reports
overall closed-loop Pearson correlation 0.929 across seven policies. This is
not a Blueprint measurement. Without independently accepted frozen external
outcome rows, the ranking output remains `correlation_not_measured` with null
Pearson, Spearman, and MMRV values. No result proves physical-robot performance.

## Command

```bash
PYTHONPATH=/absolute/path/to/exact/worktree/src \
python -m blueprint_pipeline.evaluator_qualification_workflow \
  --request /absolute/path/to/evaluator_qualification_workflow_request.json \
  --output /absolute/path/to/evaluator_qualification_workflow.json
```

For a valid JSON request, the command atomically writes the derived result. It
exits `0` only for a complete public-launch qualification and `2` for a blocked
result. Input must be a regular file within the bounded request size;
credential-like fields and non-canonical numeric values block request
acceptance and are never copied into output.
