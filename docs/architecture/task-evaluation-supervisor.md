# Task Evaluation Supervisor

Status: Phases 0 through 4 have implementation-level contracts and focused
tests. Every canonical `run_e2e` capture build enters the supervisor, including
interruption-safe continuation. Live agent inference, provider recovery, paid
compute, candidate execution, and physical actions remain fail closed unless a
separately bounded authority envelope is supplied.

## Product boundary

The Task Evaluation Supervisor is a major product control plane around the
existing Decision/Evidence proof kernel. OpenAI Agents SDK is the required
agent harness for the durable supervisor and all six specialist agents:

```text
customer question
  -> OpenAI Agents SDK supervisor interpretation, planning, observation, and replanning
  -> deterministic validation at every authority/evidence transition
  -> deterministic Decision/Evidence aggregation
  -> Decision Envelope: decision, partial decision, or abstention
```

Agents SDK controls agent turns, typed outputs, future tool calls, tracing, and
future specialist coordination. Blueprint controls the run state, allowed-tool
surface, authority, artifact ledger, and proof kernel. Agents control search,
sequencing, explanation, and recovery proposals. The
deterministic proof kernel controls schema acceptance, hashes, rights, budgets,
provider admission, frozen splits, hidden labels, success predicates, evaluator
thresholds, claim ceilings, proof transitions, and the final decision.

Supervisor artifacts are never accepted evidence. A completed provider call or
agent explanation cannot set proof booleans or upgrade a claim.

## Current-state consolidation map

| Existing component | Actual current role | Supervisor relationship |
| --- | --- | --- |
| `decision_evidence_contracts.py` | Versioned testbed, request, method, qualification, plan, result, and decision contracts | Remains the proof-kernel contract authority |
| `decision_evidence_router.py` | Deterministically chooses the cheapest qualified sufficient evidence and compiles leaf Evaluation Runs | Called by the evaluation-method-router capability; the agent does not replace its selection logic |
| `decision_evidence_execution.py` | Explicit adapter registry, normalized evidence, and deterministic Decision Envelope aggregation | Remains execution and verdict authority |
| `evaluation_run_contract.py` | Stable provider-neutral leaf-run contract | Remains the compilation target |
| `agent_runtime/orchestrator.py` | Capture-readiness review with deterministic builders and optional provider overrides | Candidate input/adapter source for the capture-testbed capability; not the durable run supervisor |
| `reconstruction_frame_dataset.py` and `arkitscenes_raw_proxy.py` | Frozen candidate/evaluator splits plus a reduced-authority public ARKit data compiler | Typed non-spend compilation support; cannot set proof or promote the proxy to Raw Contract 3.2/iPhone evidence |
| `native_360_normalization.py` | Source-preserving `.insv` probe, calibration, rig, and synchronization validation | Conditionally injected typed tool; the agent receives only capture/route digests and cannot alter calibration |
| `equirectangular_virtual_rig.py` | Fixed perspective projection with explicit shared-optical-center groups | Conditionally injected typed tool; derived views remain support pixels and cannot become independent captured observations |
| `agent_operator_runtime.py` | Gated one-shot OpenAI/Codex operator wrappers | Shared live-Agents-SDK admission gate only; not the supervisor harness |
| `site_eval_director.py` | Deterministic site/simulation planning plus optional advisory operators | Future registered non-spend tool surface after contract qualification |
| `stance_configuration_agent.py` | Deterministic bounded recovery/search | Candidate deterministic recovery tool, despite its historical “agent” name |
| `adaptive_task_stance_configurator.py` | Bounded candidate proposer with deterministic acceptance | Candidate non-spend proposal tool |
| `policy_autoresearch.py` and `autoresearch/` | Development-time candidate mutation under frozen evaluation | Kept outside hidden evaluation and proof authority |
| `robot_eval_job_orchestrator.py` | Robot-evaluation job lifecycle, provider gates, and optional operator choice | Future pre-authorized runtime tool surface; never bypassed |
| `run_e2e.py` | Required post-capture supervisor stage plus legacy optional agent review | Every completed capture build enters the new supervisor lifecycle; the old review remains a compatibility path |

The migration preserves the existing Decision/Evidence commands and contracts.
The supervisor has an explicit `supervise` operation and a separate artifact
family, and `run_e2e` now enters that same lifecycle after every completed
capture pipeline stage. There is no flag that omits the supervisor from the
normal capture-build path.

## Implemented interfaces

The package `blueprint_pipeline.task_evaluation_supervisor` provides:

- OpenAI Agents SDK as the required harness and core project dependency;
- one typed OpenAI Agents SDK manager that chooses the next eligible
  specialist, observes the exact digest-bound result, and replans after every
  completed specialist turn. Each result carries replay-checked structured
  tool observations, so the manager observes actual registered-tool outcomes
  rather than relying on specialist prose;
- one SDK `Agent` definition and strict structured-output contract for each of
  the six specialist capabilities;
- a typed injectable SDK Runner seam so hermetic tests never make live calls;
- bounded capture-build ingress that reads only known manifests and exposes an
  allowlisted projection, never arbitrary files or raw media;
- all six autonomy modes as validated enum values;
- an authority envelope that always denies proof, rights, budget, hidden-label,
  and physical-action authority;
- typed tool descriptors and a capability-gated registry;
- typed action proposals, capability results, invocation manifests, events,
  run state, and terminal reports;
- an independent deterministic supervisor evaluator with hidden expected
  properties, claim/clarification/recapture/routing/failure/abstention/audit
  metrics, critical-boundary checks, and an explicit no-self-grading flag;
- a hash-chained append-only JSONL event ledger that rejects partial records,
  sequence changes, run changes, and chain changes;
- idempotent completed-run reuse and interruption-safe continuation that invokes
  only capabilities not already committed to the ledger;
- a persistent inference reservation audit that records worst-case cost before
  each live SDK provider call, records completion separately, restores the
  cumulative ceiling after restart, and refuses to repeat a call whose prior
  billing/result state is ambiguous;
- normalized deterministic kernel-input snapshots plus a replay verifier that
  revalidates input, result, invocation, and event digests without a model call;
- one durable supervisor state machine per run;
- six OpenAI Agents SDK shadow capabilities:
  - claim and task interpreter;
  - capture and testbed supervisor;
  - evaluation method router;
  - runtime failure recovery;
  - scenario and adversarial-test proposer;
  - post-run diagnostician.
- deterministic customer decision reports bound to the validated request,
  testbed, evidence plan, normalized evidence, Decision Envelope, agent
  proposals, action artifacts, spend, runtime, uncertainty, claim ceilings,
  next experiments, and prohibited claims. Each claim names its result digests,
  method identity and family, method-profile digest, authority/proof tiers,
  self-qualification flag, validity, status, and exact claim ceiling. Spending
  distinguishes reported cost from the reserved inference ceiling and reports
  agent latency separately from registered-action duration;
- clarification and authorization request/receipt contracts in which an agent
  may request input or authority but cannot answer or approve its own request;
- pre-evaluation scenario proposal artifacts plus a separate operator-only
  freeze contract that records evaluator, success-predicate, and hidden-label
  manifest digests without exposing hidden labels;
- a 12-case public synthetic protocol fixture with four development and eight
  held-out-shaped cases, case-specific inputs, leakage canaries, 20 recorded
  human-guided baseline metrics spanning reasoning quality, spend, recovery,
  authority, leakage, scenarios, audit, and replay, plus aggregate comparison
  before autonomy promotion. Because this fixture is committed with the tests,
  it proves the scoring and non-self-grading boundary but is not secret held-out
  product evidence. The trigger-aware hermetic fixture scores `0.987500` versus
  the recorded `0.978313` baseline (`+0.009187`), with zero critical boundary
  violations; this remains below the frozen `+0.05` promotion threshold;
- a standalone recorded-run evaluation lane for a separately held, digest-bound
  `task_evaluation_supervisor_eval_corpus.v2`. It requires the corpus to be
  frozen before the agent configuration, the manager and all six specialist
  identities to be frozen before every run, an exact complete held-out case
  matrix, and deterministic replay of every run. It writes grading artifacts
  outside source run directories, invokes no model, emits no hidden case
  properties, and refuses autonomy promotion unless the sealed comparison beats
  the recorded baseline by the corpus threshold with zero critical violations;
- a pre-authorized recovery controller with provider/action allowlists,
  immutable commit and input bindings, cumulative spend, expiry/TTL, retry
  ceilings, non-retryable scientific failures, watchdogs, mandatory teardown,
  and preserved failure evidence;
- a frozen agentic candidate PolicyAdapter plus a neutral three-stack suite for
  direct policy, decomposed planner+policy, and verify/recover supervisor
  candidates under the same frozen scenarios, evaluator, and predicates.

Historical deterministic implementations remain only as frozen baselines and
fixture oracles for independent agent evaluation. They are not a production
fallback and are not selected by the supervisor product path.

## Capture-build-first start

A completed capture build is sufficient to start the supervisor. It is not
sufficient to fabricate a customer decision. The deterministic ingress reads
only registered manifest paths, records their hashes and schema names, and
builds an allowlisted metadata projection. It does not read raw media.

The Capture and Testbed Supervisor now proposes a deterministic
`task_evaluation_capture_reconstruction_route.v1` whenever that bounded build
contains an explicit capture authority profile. The route is profile-specific:
ARKit/LiDAR uses the strict metric-scaffold lane; equirectangular 360 capture
requires spherical-to-perspective normalization before SfM/3DGS; native 360
capture additionally requires original-container normalization; and monocular
video requires SfM plus a separate metric-scale qualification. Conflicting or
missing profiles fail closed. A LiDAR hint alone does not select the ARKit lane.
An iPhone ARKit declaration is reported as
`capture_profile_validation_status=required_raw_contract_gate` until the trusted
V3.2 validator inspects the retained PTS/retention evidence and actual ARKit
sensor files. The route can propose that deterministic gate, but it carries no
profile-validation digest and has `proof_effect=none`; a plain iPhone MP4 is
therefore never represented as an ARKit or LiDAR-validated capture.
For Web-uploaded 360 captures, intake probes the verified retained bytes before
quarantine cleanup and writes a source-bound `capture_profile_validation.v1`
artifact. The live service passes the allowlisted capture root into ingress so
the envelope and profile decision are observed together. A native dual-stream
decision exposes normalization as the next gate but leaves calibration pending;
it does not treat topology as calibrated camera evidence.

In `execute_non_spend`, the specialist may call the registered read-only
`plan_capture_reconstruction_route` tool using the exact capture-build digest.
The deterministic tool, rather than agent prose, returns the profile, ordered
method kinds, implementation status, registered adapters, missing stages, and
proof boundary. The result is still a planning observation: it cannot authorize
an adapter, establish reconstruction evidence, or promote a 3DGS appearance
layer into metric, semantic, collision, physics, physical-success, or deployment
truth.

When validated Phase 4 request artifacts and trusted runtimes are injected, the
same specialist can call `run_pose_estimation` and
`train_gaussian_reconstruction`. The model-facing arguments contain only the
exact request digest. Deterministic bindings reject mismatched lineage,
incompatible feature/matcher pairs, hidden-held-out access, split mutation,
self-grading, malformed results, and untyped failures. Pose and appearance
results remain candidates and cannot modify proof state.

Phase 5 registers `compile_metric_geometry`, `compile_collision_candidate`,
`qualify_collision_candidate`, `package_nurec_openusd`, and
`verify_isaac_asset` using the same digest-only execution pattern. Source and
request artifacts plus executor callables remain trusted runtime state. The
model cannot supply paths, thresholds, calibration, package contents, Isaac
checks, or provider handles. Each emitted artifact is revalidated and bound to
its predecessor before it enters the ledger; the enclosing tool observation
retains `proof_effect=none`. `package_nurec_openusd` additionally validates the
complete versioned packaging request before invoking the trusted packager; a
digest-valid but scientifically unqualified collider request is refused before
filesystem execution.

Phase 6 registers `import_external_reconstruction` for the local external-export
lane. Its only model-visible argument is the exact
`external_reconstruction_import_request.v1` digest. The trusted importer checks
source-capture binding, provenance and rights, path confinement, exact hashes,
bounded sizes, and USDZ archive safety, then emits separate rights and import
receipts. It performs no remote upload and cannot promote provider output to
raw evidence, metric geometry, collision truth, Isaac compatibility, task
success, physical success, or deployment readiness.

The registry also contains `invoke_authorized_reconstruction_provider` as an
external-side-effect descriptor restricted to `execute_preauthorized`. It is
deliberately absent from capability bindings while no qualified adapter exists.
Future binding requires the provider-neutral admission/request contracts, an
exact non-agent authorization receipt, a canonical paid-resource route, and
execution plus deletion receipts. Provider success remains derived support and
cannot self-grade or mutate deterministic proof.

When no Decision/Evidence Request or maintained Site-Task Testbed exists, the
manager first triggers claim interpretation, then capture/testbed inspection,
and stops with a typed clarification or blocker. Scenario, routing, recovery,
and post-run diagnosis are not invoked merely to satisfy a fixed call count.
They become eligible only when their required validated inputs exist. All six
specialists remain registered for every run. The proof kernel accepts no claim
until the missing task, robot, operating condition, success, rights, and
evidence contracts are deterministically valid.

The canonical lifecycle is:

```text
capture materialization -> capture pipeline -> required Task Evaluation
Supervisor -> optional legacy review/support stages -> deterministic evidence
and decision stages
```

The authenticated live capture-upload path has the same mandatory transition:
after server-side admission and QA it passes the immutable
`capture_intake_envelope.json` to the bounded ingress and returns the resulting
supervisor lifecycle record with the upload receipt. This starts the
clarification/blocker loop from a capture alone; it does not wait for
reconstruction or testbed compilation. Live inference remains disabled by
default. A deployment may preconfigure a strict model and per-run SDK budget,
but that profile is bounded to USD 100, digest-bound into the durable run ID,
and still requires the shared live-operator gate. Invalid service configuration
fails before upload processing and never grants recovery, physical, proof, or
deployment authority.

There is no `--skip-task-evaluation-supervisor` control.

The absence of live inference authority never causes a fallback to a different
harness. The required supervisor records a typed blocked result until an
inference budget and the shared provider gate are present. The completed capture
remains usable; Blueprint simply has not yet performed the agent reasoning or
made a Task Evaluation decision.

Capture lifecycle v3 binds each durable run ID to the capture digest and the
exact agent execution profile: model, live-inference request, inference ceiling,
operator-gate state, harness, and autonomy mode. Repeating an identical profile
is idempotent. Supplying new inference authority creates a new provenance-linked
run instead of trying to mutate or resume the earlier immutable blocked run.

## Shadow artifacts

An explicit shadow run writes:

```text
task_evaluation_supervisor_run.json
authority_envelope.json
tool_registry_manifest.json
proof_boundary.json
supervisor_events.jsonl
supervisor_state.json
terminal_supervisor_report.json
kernel_inputs_manifest.json
kernel_inputs/*.json
manager/decisions/step-*.json
manager/invocations/step-*.json
manager/refusals/step-*.json
capabilities/<capability>.json
invocations/<capability>.json
```

Every capability output is marked `authoritative=false`,
`proof_booleans_mutable=false`, and `proof_effect=none`. Every proposal is
classified against the tool registry and authority envelope. Shadow proposals
are recorded but never executed.

### Versioned artifact inventory

The original product names map to the following repository contracts. Blueprint
uses an append-only manager decision sequence instead of one mutable
`supervisor_plan` document so every replan has an immutable parent event and can
be replayed independently.

| Product artifact | Repository contract or artifact | Authority |
| --- | --- | --- |
| Task Evaluation Supervisor Run | `task_evaluation_supervisor_run.v1` | Durable control-plane identity; no proof effect |
| Supervisor state | `task_evaluation_supervisor_state.v1` | Deterministically reconstructed lifecycle state |
| Proposed claim graph | `proposed_claim_graph.v1` inside the claim-interpreter capability result | Agent proposal; accepted claims still require the Decision/Evidence contract |
| Supervisor plan and replans | `task_evaluation_supervisor_manager_decision.v1` sequence plus `task_evaluation_supervisor_action_proposal.v1` | Agent sequencing within the deterministic eligibility menu |
| Tool invocation and structured observation | `task_evaluation_supervisor_invocation.v1`, manager invocation manifests, and `task_evaluation_supervisor_tool_observation.v1` | Invocation is audit evidence; only registered observations may report an action result |
| Supervisor event | `task_evaluation_supervisor_event.v1` in the hash-chained JSONL ledger | Append-only audit record |
| Clarification request and receipt | `task_evaluation_clarification_request.v1` and `task_evaluation_clarification_receipt.v1` | Non-proof customer input; an agent cannot answer its own request |
| Targeted recapture request, receipt, and reinspection | `targeted_recapture_request.v1`, `task_evaluation_targeted_recapture_receipt.v1`, and `task_evaluation_recapture_reinspection.v1` | Request and receipt are non-proof; the kernel derives reinspection status |
| Capture reconstruction route | `task_evaluation_capture_reconstruction_route.v1` plus the registered `plan_capture_reconstruction_route` observation | Profile-specific planning only; no execution authorization or reconstruction proof |
| Reconstruction execution readiness | `task_evaluation_reconstruction_execution_readiness.v1` emitted by capture-supervisor lifecycle v4 | Source-bound join of route, registry, runtime bindings, and recorded control-plane authority/execution; support only and never a proof upgrade |
| Reconstruction readiness pointer | `task_evaluation_reconstruction_readiness_pointer.v1` updated after authenticated plan/authorize/execute mutations | Replaceable index over immutable content-addressed readiness history; predecessor-linked, non-authoritative, and replay-idempotent |
| Evidence-method selection | `evidence_method_selection.v1` capability artifact and the existing versioned Evidence Plan | The deterministic router qualifies and selects methods |
| Leaf-run compilation | Existing `evaluation_run_spec.v1` artifacts materialized under `generated/compiled_leaf_runs/` | Deterministic local compilation; no provider execution |
| Typed failure diagnosis and recovery | `typed_failure_diagnosis.v1`, `task_evaluation_recovery_action.v1`, and `task_evaluation_recovery_result.v1` | Diagnosis is advisory; a bounded controller owns execution and result validation |
| Authorization request and receipt | `task_evaluation_authorization_request.v1` and `task_evaluation_authorization_receipt.v1` | Operator-issued context; never an SDK credential by itself |
| Scenario proposal set and frozen manifest | `task_evaluation_scenario_proposal_set.v1` and `task_evaluation_frozen_scenario_manifest.v1` | Agent proposes before evaluation; only the operator-side freeze becomes protocol input |
| Post-run diagnosis | `post_run_diagnosis.v1` capability artifact | Explanatory only; cannot change the verdict |
| Decision proposal and customer report | Terminal `task_evaluation_supervisor_report.v1` plus `task_evaluation_customer_report.v1` | The kernel supplies the decision fields; agent text remains non-authoritative |
| Agent invocation manifest | Specialist and manager invocation manifests with SDK/model, prompt, registry, input, authority, budget, cost, latency, parent-event, and refusal bindings | Audit and replay input; no proof effect |
| Terminal supervisor report | `task_evaluation_supervisor_report.v1` | Exact replay-checked index of the terminal run; does not independently establish a claim |

Every executed tool call returns a strict, versioned observation artifact. The
same deterministic validator runs at tool creation, supervisor ingestion, and
replay, binding the observation to the exact run, specialist capability,
authority envelope, registered tool version and mutability, runtime identity,
output digest, registered output schema, cost ceiling, and retry ceiling. A
self-consistently hashed typed result is still refused when required fields,
types, constants, or the tool-specific closed-field set do not match. Unknown
fields or any other mismatched binding are refused before the result can be
written to the event ledger; raw injected tool text is not preserved as
evidence.

The same unknown-field refusal applies to every proof-adjacent supervisor
artifact: authority envelopes, tool descriptors, action proposals, capability
results, events, invocation manifests, durable runs, states, and terminal
reports. Recomputing an artifact digest does not permit an agent-selected
authority, proof, budget, hidden-label, or deployment field to hitchhike through
an otherwise valid contract. Budget-state, inference-spend, action-spend,
manager-reference, and tool-observation-reference envelopes also use exact
nested fields; the durable run is bound to Blueprint's registered supervisor
harness and manager identity.

Authority is re-derived during replay rather than trusted by digest alone. The
authority mode must match the run and terminal report, its immutable inputs must
match the run, and its allowed tools must equal the recorded registry (or be
empty in `disabled`). Every mode except `execute_preauthorized` has zero action
spend and no recovery receipt or expiry; external processing exists exactly when
agent inference is authorized.

Observation custody belongs to Blueprint's registered binding, not to the SDK
adapter. The binding durably records each validated result inside the run before
returning it to the agent harness. The adapter-reported observation set must
match that trusted set exactly. If the adapter omits or alters a result—or fails
or is interrupted after a tool action—the specialist output is blocked while
the trusted observation and any action that already occurred remain visible in
the audit, report, resume, and replay.

`replay_supervisor_run(...)` verifies the hash chain and all recorded contracts.
Replay also reconstructs the recorded tool registry from its validated
descriptors and rejects a manifest that grants undeclared shell, filesystem,
network, provider, or proof authority. When a deterministic Decision Envelope
is present, replay reproduces its digest, outcome, and claim ceiling. Agent prose
is intentionally not regenerated, so a changed future model cannot change the
replayed proof result.

Run the current vertical slice with:

```bash
blueprint-route-task-evaluation supervise \
  --capture-build /path/to/completed-capture \
  --mode shadow \
  --output-dir out/supervisor
```

`--request`, `--testbed`, evidence profiles, qualifications, plans, results, and
decisions can be added as the run accumulates them. The harness is always
OpenAI Agents SDK; there is no alternate production agent harness selector.
The standard `run_e2e` capture path invokes this lifecycle automatically and
records its status in the stage ledger and run summary. That required lifecycle
uses `execute_non_spend`: once live SDK inference has its separate budget and
operator gate, it can materialize registered local clarification and recapture
requests without waiting for a second product path. No provider, paid, physical,
or proof-mutating action becomes available. Capture-supervisor lifecycle v3
binds the run-id namespace to both the capture digest and exact execution
profile; v1 and v2 ledgers remain immutable and are never resumed under broader
authority.

A returned customer clarification also re-enters the durable run instead of
remaining an out-of-band conversation:

```bash
blueprint-route-task-evaluation supervise \
  --clarification-request /path/to/clarification-request.json \
  --clarification-receipt /path/to/customer-response-receipt.json \
  --mode shadow \
  --output-dir out/clarification-return
```

The request and receipt are exact-schema and digest-bound kernel inputs. Response
JSON is size-, depth-, and type-bounded, marked untrusted, and shown to the claim
interpreter. The claimed responder identity is not verified by the supervisor.
The receipt therefore cannot answer its own request, create claims, or become
proof: evaluation still requires a separately valid Decision Evidence Request.

An operator authorization response follows the same durable, replayable ingress
path:

```bash
blueprint-route-task-evaluation supervise \
  --authorization-request /path/to/authorization-request.json \
  --authorization-receipt /path/to/operator-authorization-receipt.json \
  --mode shadow \
  --output-dir out/authorization-return
```

Blueprint validates the exact request identity, digest, immutable inputs,
provider/action subsets, cost, TTL, retry ceiling, issuer type, and proof effect
before recording either artifact. The approved receipt is observable context,
not an executable SDK credential: the CLI injects no recovery controller and
reports `authorization_granted_to_agent=false`. Programmatic
`execute_preauthorized` execution requires a separately constructed controller
bound to the identical receipt digest. The controller then independently
rejects expiration and other runtime authority failures. This permits stale
receipts to remain in the immutable audit without making them usable.

A targeted follow-up capture re-enters the same control plane with both the
original Blueprint request and a customer-bound receipt:

```bash
blueprint-route-task-evaluation supervise \
  --capture-build /path/to/completed-targeted-recapture \
  --targeted-recapture-request /path/to/targeted-recapture-request.json \
  --targeted-recapture-receipt /path/to/customer-submission-receipt.json \
  --mode shadow \
  --output-dir out/recapture-reinspection
```

The deterministic ingress accepts only known capture manifests and a bounded
approved projection, normalizes a standalone source filename so filenames never
become agent instructions, marks the entire capture projection as untrusted SDK
input, and binds its digest to the exact prior request. The receipt records
submission, not success: it cannot infer rights, make the new
capture authoritative, or resolve the original blocker. The capture specialist
must reinspect it and a maintained testbed must be deterministically rebuilt or
validated before the blocker can change.

When that rebuilt testbed is supplied, the proof kernel derives a separate
`task_evaluation_recapture_reinspection.v1` artifact; callers and agents cannot
supply it. Resolution requires the testbed validation envelope to bind the exact
returned `capture_build_digest`, every requested gap to be covered by a named
evidence inventory entry whose source digest exists in that returned capture
projection, and a testbed-origin request to preserve predecessor lineage. A
wrong capture binding, unlinked or missing evidence, or lineage mismatch remains
a typed blocker. Even the resolved result has `proof_effect=none`: it closes the
specific capture sufficiency request, not rights, task success, physical
validation, safety, or deployment readiness.

Live inference is fail closed and requires `--allow-live-agent-sdk`, a positive
`--agent-inference-budget-usd`, and
`BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS=true`. Before every SDK call the
harness reserves a conservative worst-case input/output cost against the one
run budget and writes that reservation before entering the provider. A separate
completion artifact closes it. Failed or interrupted calls keep their
reservation; after restart, Blueprint restores the cumulative ceiling and
refuses an identical call when the earlier billing/result state is unknown.
Tests inject the SDK Runner boundary and make no API call. No live call, paid
compute, or provider action was authorized or run while implementing this
slice.

Optional `--plan`, `--result`, and `--decision` inputs let the failure-recovery
and post-run capabilities inspect already validated artifacts. They do not
change those artifacts.

## Enabled and disabled behavior

| Mode | Current behavior |
| --- | --- |
| `disabled` | Explicit fail-closed administrative/test mode; writes lifecycle/proof-boundary evidence and invokes no capability. It is not an alternate product harness or a normal capture-build path. |
| `shadow` | Runs the SDK manager and only specialists whose deterministic prerequisites are present; records proposals and executes no proposed action |
| `advise` | Runs the SDK manager and eligible specialists, validates their registered action proposals, and marks every valid proposal as requiring operator approval. It exposes no callable SDK tools, executes no action, spends no action budget, and grants no authority. An approved proposal must enter a separately authorized execution mode through a validated receipt. |
| `execute_non_spend` | Runs SDK agents with capability-scoped, registered read-only inspection tools. Tool observations are digest-bound, zero-cost, replay-validated, and have `proof_effect=none`; broader non-spend actions remain gated. |
| `execute_preauthorized` | Operational only when an operator-issued, digest-bound receipt and a scoped recovery controller are injected; otherwise blocked. The current receipt is not a cryptographic signature. The controller enforces provider/action allowlists, SHA/input bindings, spend, TTL, retries, watchdog, and teardown. |
| `candidate_policy` | The mode remains fail closed in the generic supervisor runner; candidate execution is compiled through the separate frozen neutral PolicyAdapter suite so candidate code never inherits supervisor/evaluator authority. |

Unknown modes fail before a run is created.

## Implemented phase gates and remaining evidence

Phase 2 includes capability-scoped read-only tools for claim-contract, testbed,
evidence-plan, normalized-result, and Decision Envelope inspection. It also
includes the first reversible non-spend action: the evaluation-method agent can
ask Blueprint to deterministically route the bound request and testbed, persist
the resulting Evidence Plan, and materialize validated leaf Evaluation Run
specs under that supervisor run. Generated paths are fixed by Blueprint, every
artifact is digest-bound and replay-validated, provider execution is not
started, and proof state is unchanged. The capture/testbed agent can also write
a targeted recapture proposal bound to the current capture or testbed. It may
not initiate capture, request a full-site recapture through this action, infer
rights, mutate raw capture, or create proof. Phase 2 also includes deterministic
targeted-recapture receipts that reject an unchanged capture, bind the exact
request and new capture projection, and preserve an undetermined blocker until
testbed reinspection. The kernel-derived reinspection artifact then checks exact
capture binding, named evidence coverage, source-artifact lineage, and
predecessor lineage before closing that specific gap. Capture projections are
schema-validated during live
ingress and replay; a malformed but self-consistently hashed envelope is
refused when it violates the bounded projection contract. Phase 2 also includes
deterministic customer report generation, clarification and authorization receipts,
scenario-proposal materialization with an operator-only freezing boundary, and
tests proving that identical accepted evidence yields the same kernel decision
even when agent prose changes. The scenario proposal and frozen-manifest
validators reject unknown fields,
embedded hidden labels, post-result generation flags, non-canonical scenario
ordering, or expanded operator receipts even when an attacker recomputes every
digest. The frozen public artifact contains only the hidden-label manifest
digest, never the labels themselves. The public synthetic corpus comparison now
runs each held-out-shaped input separately and correctly refuses autonomy promotion
for the fixture runner. Separate `blueprint-route-task-evaluation` operations
validate an external sealed corpus, freeze the exact manager/specialist/tool
configuration, and score replay-verified recorded runs without mutating them.
The committed fixture proves the evaluation and promotion boundary, not that a
live production model has beaten the human-guided baseline.

Phase 3 routes recovery only through an injected pre-authorized controller. The
controller is provider-neutral and no live provider was called while building
it. A concrete `VastWAMRecoveryAdapter` now wraps the existing authorized Vast
WAM runner; it is not a second launcher. It binds the exact bundle, commit, and
input digests, forwards the receipt-bounded spend and runtime ceilings, requires
the independent watchdog, disables retention, and accepts provider-zero only
from versioned teardown evidence plus terminal watchdog confirmation. It refuses
to launch when less than Vast's one-minute minimum authority window remains.
The controller uses its own clock, binds the provider and action allowlists
inside the operator-issued receipt, rejects ambiguous adapter exceptions from
automatic retry, and accepts teardown only when `provider_zero=true` is
explicitly proven. Before calling an adapter it also requires the opaque
in-process grant issued by Blueprint's shared paid-resource admission
chokepoint; a provider-supplied Boolean cannot replace that grant. Operationally,
Vast is the preferred first canary because
its Blueprint qualification-session lane has stronger prior evidence; RunPod
remains a replaceable fallback that must independently pass the same admission
and provider-zero gates. Live provider recovery must still enter through the
canonical paid allocator seam before production enablement.

The canonical allocator's older generic `strict-policy-smoke` parser still
defaults to RunPod because that specific launcher is RunPod-only. That legacy
default is not inherited by the Task Evaluation Supervisor. The newer policy-
ranking path already defaults to Vast, and a supervisor recovery action can use
only a provider named in its trusted operator receipt and installed adapter.

Phase 4 admits Pigey-like or other agentic robot stacks only as frozen candidate
`PolicyAdapter` implementations. Candidate code, model/provider, prompt, tools,
memory/skills, budgets, retries, scenario pack, evaluator, and success predicates
are digest-bound before hidden evaluation. The runtime's execution-relevant
configuration has its own digest, so changing a model, external checkout,
scenario binding, tool interface, step/timeout envelope, endpoint, or cost rate
without refreezing the candidate is rejected before any output directory or
provider action is created. Candidate agents cannot see hidden labels or grade
themselves. The implementation compiles neutral Evaluation Run
specs and provides an explicitly gated neutral execution harness: candidate
runtimes receive only public frozen specs, while an independent evaluator alone
receives the digest-bound hidden manifest. Candidate traces and evaluator
outputs are independently digest-checked, and candidate-supplied scores or
unregistered result fields are refused. A paid candidate runtime must also have
a separate Blueprint-injected cost authority matched to its provider and paid
resource class. That authority writes a digest-bound maximum reservation before
candidate execution and independently settles actual cost afterward;
candidate-reported token usage cannot satisfy the gate. Missing, mismatched,
malformed, oversized, or non-final settlements fail closed. If a paid runtime
loses its result after execution begins, the authority must reconcile the
reservation; otherwise the suite stops, preserves a typed failure, marks
reported cost non-final, and refuses to run later candidates. A concrete
external-checkout Pigey adapter binds an exact clean upstream commit and
entrypoint digest, invokes the
public CLI without a shell, passes only allowlisted environment variables,
normalizes `trial.json` into Blueprint's candidate trace, and explicitly drops
Pigey's own success value. Pigey source is not vendored, and its current adapter
marks trial-reported usage as non-authoritative. Paid Pigey execution now binds
the runtime to an exact OpenAI project ID, API-key ID, and exclusive-scope
attestation digest. A concrete `OpenAIProjectCandidateCostAuthority` takes a
zero-cost provider baseline through OpenAI's read-only organization Costs
endpoint before execution, filtering and grouping by both project and API-key
identity. Immediate cost settlement remains `reconciliation_required` until the
configured provider-reporting window closes. The delayed reconciliation path
writes content-addressed settlement evidence and does not rerun the candidate or
evaluator. Candidate token totals never satisfy this gate. This is provider-
reported cost evidence, not invoice settlement. The exact reviewed Pigey commit
currently publishes no repository-level license, so the runtime also requires a
digest-bound, independent license-or-permission attestation and remains blocked
without rights-holder permission. The canonical allocator now has an
`openai_api_candidate` admission entry and a Pigey-specific wrapper. It binds the
exact frozen suite, operator receipt, Blueprint source commit, runtime digest,
license attestation, independent cost authority, spend ceiling, retry count,
worst-case runtime, watchdog, and teardown behavior before issuing an opaque
in-process grant. A dry run writes the same prospective record but returns no
grant, and a serialized admission record is never executable authority. OpenAI
candidate grants must carry this allocation binding; the prior generic unbound
paid grant is rejected. A dedicated pre-provisioned OpenAI cost scope, valid
rights-holder permission, a live operator receipt, and explicit paid-execution
authorization are still required before a live Pigey run. No live Pigey/provider
run or physical validation is claimed.

No phase may infer physical validation, deployment approval, safety
certification, or policy-ranking support from simulation or generated media.
