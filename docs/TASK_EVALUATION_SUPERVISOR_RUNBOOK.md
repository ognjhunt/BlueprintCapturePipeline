# Task Evaluation Supervisor operator runbook

## Product path

A completed capture build always enters the Task Evaluation Supervisor stage.
The capture is enough to start reasoning, but it is not enough to invent a task,
robot, success condition, rights grant, or customer decision.

The production agent harness is OpenAI Agents SDK. There is no alternate agent
harness or deterministic agent fallback. The deterministic proof kernel is a
separate authority boundary beneath the SDK agents.

The capture lifecycle run ID includes the capture digest and a digest of the
exact model, inference ceiling, live-inference request, operator-gate state,
harness, and autonomy mode. Repeating that profile reuses the same terminal run.
Adding inference authority creates a new linked run; it does not rewrite the
earlier no-authority blocker or its ledger.

```text
capture build
  -> six specialist agent capabilities
  -> registered proposals and observations
  -> deterministic contracts and proof kernel
  -> decision, partial decision, abstention, or typed blocker
```

## What starts each capability

1. The claim interpreter runs when a customer question or capture-only ingress
   arrives. It proposes claims and clarification questions.
2. The capture/testbed supervisor runs after interpretation and whenever a new
   capture or recapture receipt is attached. It proposes only the smallest
   useful recapture.
3. The evaluation router runs when validated claims and a maintained testbed are
   available. It compiles deterministic leaf runs and chooses only qualified
   evidence methods.
4. The recovery capability runs when a structured runtime or evidence failure is
   present. It may propose recovery in shadow/advise modes and may execute only
   through a pre-authorized controller.
5. The scenario proposer runs before hidden evaluation. Its proposals remain
   unfrozen until an independent operator receipt freezes scenario IDs,
   evaluator digest, success-predicate digest, and hidden-label manifest digest.
6. The post-run diagnostician runs only after a deterministic decision artifact
   exists. It explains that artifact and cannot change it.

## Autonomy modes

- `disabled`: write boundary evidence and invoke no agent.
- `shadow`: invoke agents and record proposals; execute nothing.
- `advise`: require an operator decision before any proposal can advance.
- `execute_non_spend`: permit registered, reversible, non-spending actions.
- `execute_preauthorized`: permit only receipt-bound recovery actions through an
  injected controller and registered provider adapter.
- `candidate_policy`: fail closed in the supervisor runner. Agentic robot stacks
  enter through the separate frozen PolicyAdapter evaluation suite.

Unknown modes and missing validators, receipts, adapters, budgets, or audit
records fail closed.

## Starting from a capture build

The normal end-to-end command invokes the supervisor automatically:

```bash
python -m blueprint_pipeline.run_e2e \
  --capture-root /path/to/staged-capture \
  --provider openai
```

The standalone control-plane command is:

```bash
blueprint-route-task-evaluation supervise \
  --capture-build /path/to/completed-capture \
  --mode shadow \
  --output-dir /path/to/supervisor-output
```

Live SDK inference additionally requires the explicit CLI admission, a positive
inference budget, and `BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS=true`. Missing
live authority produces a typed blocker, not a fabricated local answer.

Capture metadata and customer text are untrusted data. Standalone manifest
filenames are normalized to `submitted_manifest.json` before the projection is
shown to an agent; task text remains visible as data because it is needed for
interpretation. Protected output validation, the tool registry, and the
authority envelope—not prompt obedience—must contain any injected instruction.

## Non-spend execution

`execute_non_spend` can inspect registered artifacts, deterministically compile
an Evidence Plan and leaf Evaluation Run specs, materialize clarification and
authorization requests, write targeted recapture proposals, and materialize
pre-evaluation scenario proposals. It cannot start capture, run a provider,
spend money, expose hidden labels, or mutate proof state.

Clarification responses and authorization grants are separate receipts created
at trusted customer/operator boundaries. Agent output never satisfies its own
request. Every accepted response is revalidated by deterministic contracts.

### Clarification return

Return a customer response through the same supervisor control plane:

```bash
blueprint-route-task-evaluation supervise \
  --clarification-request /path/to/clarification-request.json \
  --clarification-receipt /path/to/customer-response-receipt.json \
  --mode shadow \
  --output-dir /path/to/clarification-return
```

The receipt must bind the exact Blueprint request and carries only bounded JSON.
It is marked untrusted, has `proof_effect=none`, and does not establish the
claimed responder's identity. The interpreter may use it to propose a clearer
task contract, but Blueprint must still receive and validate a complete Decision
Evidence Request before compiling or running claims.

### Authorization return

Return an operator decision through the same supervisor control plane while
preserving the original Blueprint request:

```bash
blueprint-route-task-evaluation supervise \
  --authorization-request /path/to/authorization-request.json \
  --authorization-receipt /path/to/operator-authorization-receipt.json \
  --mode shadow \
  --output-dir /path/to/authorization-return
```

The request and receipt are strict, digest-bound kernel inputs. Ingesting an
approved receipt makes it visible to the recovery specialist and replay, but
does not grant authority to the agent and does not construct a recovery
controller. Actual `execute_preauthorized` recovery requires a separately
injected controller whose receipt digest matches the recorded receipt exactly.
That controller still enforces expiry, TTL, provider/action allowlists,
immutable inputs, spend, retries, watchdog, and teardown. A denied, stale,
over-scoped, agent-issued, or mismatched receipt cannot execute an action.

### Targeted recapture return

When the customer returns a targeted recapture, preserve the original request
and create a customer-bound receipt for the newly completed capture projection.
Start the follow-up run with all three inputs:

```bash
blueprint-route-task-evaluation supervise \
  --capture-build /path/to/completed-targeted-recapture \
  --targeted-recapture-request /path/to/targeted-recapture-request.json \
  --targeted-recapture-receipt /path/to/customer-submission-receipt.json \
  --mode shadow \
  --output-dir /path/to/recapture-reinspection
```

The request and receipt are required together. The receipt must bind the prior
request digest, its source digest, and a different strict capture-build digest.
It records who submitted the capture and when, but has `proof_effect=none`, does
not infer rights, and leaves `original_blocker_resolution` as
`undetermined_pending_reinspection`. Recompile or validate the maintained
Site-Task Testbed before treating the requested gap as resolved.

The rebuilt testbed must put the returned `capture_build_digest` in
`validation_envelope.capture_build_digest`. Each requested gap must either match
an `evidence_inventory[].evidence_id` or appear in that entry's
`addresses_recapture_requirements`; the same entry's
`source_capture_artifact_digest` must match a SHA-256 artifact in the returned
capture projection. If the request was made against a prior testbed, the rebuilt
testbed must also bind that digest through
`predecessor_testbed_digest` or `supersedes`. Blueprint derives and replay-checks
the reinspection result; do not accept a caller- or agent-supplied result.

## Pre-authorized recovery and provider choice

The recovery controller requires all of the following before invoking an
adapter:

- an operator-issued receipt whose digest binds provider IDs, action IDs,
  immutable input digests, cost, TTL, retries, issue time, and expiry;
- an exact immutable commit SHA and exact input-digest set;
- a registered adapter for the named provider;
- an opaque grant from Blueprint's shared paid-resource admission chokepoint;
- controller-clock validity, remaining TTL, retry capacity, and spend capacity;
- a non-scientific, retryable failure class;
- a watchdog and mandatory teardown;
- explicit `provider_zero=true` closure.

Adapter exceptions are ambiguous and are not retried automatically. Failed
evidence is retained. A teardown status without provider-zero proof is failure.

Vast is the preferred first authorized canary because Blueprint has stronger
evidence for its maintained qualification-session path. RunPod is a fallback,
not the supervisor default. The older generic allocator defaults to RunPod only
because its `strict-policy-smoke` implementation is RunPod-specific. Any paid
mutation must still enter through:

```bash
python -m blueprint_pipeline.paid_resource_allocator gpu-canary ...
```

Do not invoke provider adapters as launchers. Do not run a paid canary without a
separate explicit spend authorization.

## Agentic candidate evaluation

Pigey-like planners, VLA/TAMP compositions, and verify/recover stacks are frozen
as `blueprint_agentic_candidate_policy@1` candidates. Before hidden evaluation,
freeze the code, model/provider/version, prompts, tools, memory/skills snapshot,
budgets, retries, scenario pack, evaluator, and success predicates.

The reviewed Pigey commit currently has no repository-level software license.
Do not execute it for Blueprint merely because the repository is public. The
runtime requires an independent `pigey_license_attestation.v1` bound to the exact
commit and authorizing the intended commercial code execution. Until the rights
holder grants permission or publishes acceptable terms, the live Pigey lane is
blocked. See `docs/third-party/pigey-license-review.md`.

The neutral suite contains one direct policy, one decomposed planner+policy, and
one verify/recover supervisor. Every candidate receives the same scenario IDs,
evaluator, predicates, and claim ceiling. Candidates receive no hidden labels,
evaluator authority, proof authority, or self-grading path.

The explicitly gated neutral execution harness gives each registered candidate
runtime only its frozen public Evaluation Run spec. It verifies the emitted
trace artifact, then passes that trace and the separately digest-bound hidden
manifest to the independent evaluator. Candidate runtime outputs have a strict
allowlist and cannot contain their own verdict or score. The resulting harness
artifact is still simulation/evaluation evidence only; it is not physical or
deployment proof.

### Pigey/OpenAI cost reconciliation

Pigey's `trial.json` token totals are diagnostic only. A paid Pigey runtime must
use a dedicated, pre-provisioned OpenAI project/API-key scope and bind the exact
project ID, API-key ID, and an operator-controlled exclusive-scope attestation
digest into both the frozen runtime configuration and its independent cost
authority. `OPENAI_PROJECT` must match that frozen project.

Before candidate execution, `OpenAIProjectCandidateCostAuthority` queries the
official organization Costs endpoint with both identifiers and requires a zero
baseline for the attribution window. It reads an admin key from a permission-
restricted file and never records the key. OpenAI cost data may lag, so the
first settlement normally remains `reconciliation_required`. After the frozen
reporting delay, use the read-only reconciliation command against the preserved
candidate execution directory and the same bound authority:

```bash
blueprint-route-task-evaluation reconcile-candidate-costs \
  --execution-dir /path/to/candidate-execution \
  --openai-project-id proj_example \
  --openai-api-key-id key_example \
  --openai-admin-key-file /permission-restricted/openai-admin-key \
  --scope-attestation /path/to/openai-cost-scope-attestation.json
```

The
reconciliation is content-addressed, does not rerun the candidate or evaluator,
and remains partial if provider evidence is missing, malformed, out of scope, or
over the reservation.

The organization Costs result is provider-reported spend evidence. Do not call
it invoice reconciliation, and do not launch Pigey directly merely because the
read-only cost adapter exists. Paid execution still requires an operator receipt
and must call `paid_resource_allocator.admit_pigey_candidate_runtime` in the same
process that will execute the neutral suite. The allocator derives the
worst-case Pigey runtime from its frozen scenario count and per-scenario timeout,
binds the exact suite, authorization, source SHA, cost scope, license, spend,
retry, watchdog, and teardown fields, persists
`openai_api_candidate_allocation_admission.v1`, and injects an opaque grant into
the runtime. `execute=False` is a readiness check only: it returns no grant.
Copying or editing the JSON artifact cannot authorize execution. An OpenAI
candidate presented with the older generic unbound grant is refused.

## Replay and customer report

Preserve the complete supervisor output directory. Replay verifies the
append-only event hash chain, accepted contracts, generated-artifact digests,
tool observations, reconstructed tool registry, and deterministic Decision
Envelope. A tool observation is accepted only when its schema and digest match
and it is bound to the recorded run, specialist capability, authority envelope,
tool version, mutability, runtime identity, output digest, cost, and retry
limits. Unknown or injected fields are refused before ledger persistence.
Replay does not ask a current model to regenerate prose.

Replay also requires `proof_boundary.json` to equal the canonical boundary
compiled into the current Blueprint release and requires the durable run and
terminal report to bind that same digest. Editing a boundary Boolean and
recomputing its digest cannot enable proof mutation, hidden-label access,
deployment approval, or physical-success claims.

Terminal action accounting is reconstructed from the validated observation
inventory. Read, reversible non-spend, and preauthorized counts, the
`actions_executed` Boolean, actual cost, duration, authorized ceiling, and
receipt digest must exactly match the terminal report. A rewritten but
self-consistently hashed terminal total is refused.

Terminal inference accounting is likewise reconstructed from the accepted
manager and specialist invocation manifests plus the authority and reservation
artifacts. Reported inference cost, live and manager invocation counts,
remaining unreserved budget, and cost finality must match that reconstruction.
Replay also validates the terminal supervisor state against the ledger length
and final event, terminal-report digest, completed capabilities, mode, spent
cost, and remaining budget. Rehashing either artifact after changing those
values does not make the change authoritative.

Capability, specialist-invocation, manager-decision, manager-invocation, and
manager-refusal summaries in the terminal report are exact indexes of their
canonical artifacts, not editable labels. Replay reconstructs the manager's
terminal reason, blocker set, terminal status, and terminal-event payload from
those artifacts and the validated tool observations. It also replay-verifies
the fail-closed `disabled`, missing-preauthorization-controller, and
`candidate_policy` control-plane outcomes. A rehashed report cannot relabel a
blocked capability or turn a blocker into a completed run.

Generated scenario proposal sets and preauthorized recovery results are
contract-validated again during replay, not accepted from their digest alone.
Scenario sets must remain pre-result, unfrozen, non-authoritative, and free of
hidden labels. Recovery results must preserve exact run, receipt, provider,
action, immutable-input, cost, retry, watchdog, teardown, provider-zero, and
no-proof-effect semantics. A live recovery controller is refused before
execution unless the matching operator request and receipt are also recorded as
kernel inputs.

The same rule covers generated evidence plans, compiled leaf-run specs,
clarification requests, authorization requests, and targeted recapture
requests. Replay binds them back to the recorded run, source request, testbed,
authority inputs, and canonical plan inventory. Tool observations may reference
only artifact types declared by that exact registered tool version; unknown
agent-defined artifact types fail closed.

Manager decisions pass one deterministic eligibility validator during live
execution, interrupted-run resume, and replay. At every step Blueprint
recomputes the observed result set, eligible next specialists, and eligible
terminal reasons from the recorded context and completed capability artifacts.
The manager may select from that menu, but cannot expand or rewrite it by
returning a self-consistently hashed decision.

Manager invocation and refusal manifests are also exact contracts. Blueprint
binds manager identity and version, canonical instruction digest, authority,
tool registry, observed inputs, cumulative inference accounting, parent ledger
event, bounded error type, and no-proof-effect fields during execution, resume,
and replay. Provider-returned metadata cannot substitute a different manager or
erase a refused manager turn.

The customer report has its own exact-schema validator at creation and replay.
It rejects unknown fields, non-finite or negative spend, inconsistent
decision/partial/abstention flags, agent-authoritative output, proof mutation,
or removal of the mandatory physical-success and deployment claim boundaries,
even if all report digests are recomputed.
Replay also rebuilds the complete report from the recorded run question, kernel
inputs, capability results, invocation manifests, tool observations, and
generated-artifact references. A schema-valid rewritten report and matching
rewritten terminal digest still fail this equality check.

Blueprint durably records a trusted observation inside the run from the
registered tool binding before control returns to the SDK adapter. If the
adapter later omits, changes, or loses that observation—or the process is
interrupted—treat the specialist result as compromised, but do not erase the
action: resume revalidates the staged observation, and it remains counted and
replayable.

The deterministic customer report must identify the original question,
validated interpretation, claims, evidence, attempted/failed/skipped methods,
agent recommendations, deterministic validations, spend/runtime, outcome,
uncertainty, evidence ceiling, next useful experiment, and prohibited claims.

No supervisor artifact proves physical success, deployment readiness, safety
certification, or policy-ranking support by itself.

## Independent supervisor evaluation

The committed 12-case corpus is a public synthetic protocol fixture. Do not call
it secret held-out evidence or use its `+0.009187` fixture delta to promote an
autonomy mode. A production comparison requires a separately held
`task_evaluation_supervisor_eval_corpus.v2` with a canonical digest and an
operator-controlled freeze timestamp.

Validate the corpus without printing its hidden case properties:

```bash
blueprint-route-task-evaluation validate-supervisor-corpus \
  --corpus /sealed/evaluation_corpus.v2.json \
  --output /evaluation/corpus_validation.json
```

Before any held-out run, freeze the exact manager identity, all six specialist
identities, SDK version, instructions, tool registry, model, provider, and
inference ceiling. The spec must not contain hidden labels or extra fields:

```bash
blueprint-route-task-evaluation freeze-supervisor-evaluation \
  --corpus /sealed/evaluation_corpus.v2.json \
  --spec /evaluation/agent_configuration_spec.json \
  --output /evaluation/frozen_agent_configuration.json
```

After every case has a terminal recorded run, score the complete matrix. Supply
one `--run case_id=/path/to/run` argument for every sealed held-out case:

```bash
blueprint-route-task-evaluation evaluate-recorded-supervisor \
  --corpus /sealed/evaluation_corpus.v2.json \
  --configuration /evaluation/frozen_agent_configuration.json \
  --run case-a=/runs/case-a \
  --run case-b=/runs/case-b \
  --output-dir /evaluation/result
```

The evaluator first replays every ledger and artifact, verifies that the corpus
predates the configuration and the configuration predates every run, and checks
the recorded manager/specialist identities against the freeze. Every case must
reference a distinct run directory and replay-verified run identity. The
evaluator records matching before/after tree digests for each source run, invokes
no model, and writes outside the source run directories. Before creating its
output directory, it stream-scans every recorded run for that case's hidden
canaries and every other hidden canary in the sealed corpus. Missing cases,
reused runs, post-hoc configuration changes,
hidden-label leakage, replay failure, self-grading, critical proof violations,
or insufficient baseline improvement fail closed.
