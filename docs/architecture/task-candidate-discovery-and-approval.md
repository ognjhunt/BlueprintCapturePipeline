# Task Candidate Discovery and Intent Approval

Status: implemented contract boundary, version 1 (2026-07-29)

## Decision

`task_candidate_discovery.v1` separates five scene-analysis categories before it
proposes work: directly observed site facts, inferred objects/affordances,
unsupported or occluded regions, hazards, and privacy-sensitive areas. Each row
is deterministically serialized, carries its own digest, and points to supporting
frames or 3D regions when available.

A task candidate is a hypothesis, not customer intent and not task-success
evidence. Every candidate includes the proposed measurable condition, required
reset, robot capabilities, grounded objects and regions, coverage, assumptions,
missing evidence, prohibited claims, estimated evaluation cost, and optional
customer-supplied value context. Confidence never changes its immutable
`approval_required` status.

## Approval boundary

`task_candidate_decision.v1` records one append-only customer/operator action:

- approve the exact candidate;
- edit and approve a measurable task;
- reject the candidate; or
- request more capture.

The decision binds the discovery digest and candidate digest. Reject and
request-more-capture actions cannot emit an approved task. Edits must provide
explicit measurable thresholds and units; the service does not invent them.
An exact customer-supplied task uses a separate digest-bound receipt and has the
same threshold/reset requirements.

Only `approved_task_definition.v1` can compile into the existing
`decision_evidence_request.v1`. The compiler verifies that the immutable testbed
references the exact approved-task digest and keeps method selection
provider-neutral. A model/provider proposer is added to the prohibited evaluator
identities, and an attempted proposer/self-grader pairing fails closed.

## Legacy compatibility

The capture-to-package orchestrator still emits its legacy task-hypothesis
report for compatible callers. AI-inferred hypotheses now always return
`needs_confirmation`, regardless of grounding or confidence, so they cannot
populate effective task metadata without passing through the approval boundary.
Customer-authored structured intake retains its existing behavior.

## Proof boundaries and remaining work

The executable contract and checked-in schema prove deterministic artifact
construction, stale-digest rejection, approval authorization, and compilation
to the existing router request. They do not prove scene-model accuracy,
customer approval in Blueprint-WebApp, a compiled testbed, an evaluation result,
or physical task success. WebApp approval/edit/reject/recapture UX and durable
service persistence remain separate launch gates.
