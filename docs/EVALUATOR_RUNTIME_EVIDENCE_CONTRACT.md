# Evaluator runtime evidence contract

Blueprint ranks policies through evaluator evidence, not through a permanent
world-model identity. OSCAR, Cosmos 3, and future model adapters converge on
`evaluator_runtime_receipt.v1`; compute providers remain execution locations,
not evaluators.

## Runtime receipt

A receipt binds one exact model invocation to:

- runtime, adapter, backend, model family, and model version identities;
- exact runtime-output, model-artifact, adapter-code, runtime-manifest,
  license-manifest, and provider-execution digests;
- a structured provider execution record bound to the same runtime output,
  model, adapter, and runtime manifest;
- a positive fresh model-run count equal to the number of unique model-output
  digests; and
- explicit false values for fixture/proxy, fallback, and stale-output use.

OSCAR and Cosmos WAM adapter outputs can be converted with
`build_wam_runtime_receipt()`. Their completed outputs now include the exact
generated-video SHA-256, actual output count, and the separate configured
sampler/inference-step count. Sampler iterations are never reported as policy
queries or completed model runs. An arbitrary
future backend can emit the neutral receipt directly without adding a new
ranking architecture branch.

## Evaluator row normalization

`normalize_evaluator_runtime_evidence()` admits a row only when the selected
model output, evaluator-runtime output, checkpoint/model artifact, backend
identity and manifest, provider execution, step count, and infrastructure state
match the receipt. The customer policy-runtime digest remains separate and is
never replaced with or equal to the evaluator-runtime digest.
The row must also pass its selected generic, OSCAR/RoboArena, or SC3 evidence
profile.

Runtime completion never supplies or overrides task success, criterion status,
authoritative episode completion, action-control results, abstention, or
correlation. A completed generated episode with a blocked authoritative
manifest remains blocked. Visual-motion smoke and generated-video review remain
support evidence only.

`python -m blueprint_pipeline.evaluator_runtime_evidence` exposes both receipt
construction and row normalization to provider-neutral pipeline jobs. Blocked
evidence is still written for diagnosis and returns exit code 2; it is never
silently converted into an admitted row.

Provider execution records use a strict field allowlist. Unexpected fields
block admission and are omitted from the receipt; evaluator rows containing
credential-like fields are blocked and omitted from normalization output.
