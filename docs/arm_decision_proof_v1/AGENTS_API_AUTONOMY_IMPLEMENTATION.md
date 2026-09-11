# Agents API autonomy implementation

This is the implementation and verification ledger for the revised September 10
Agents API assessment and its specialist design review. The objective is the
complete authorized software workflow, not a runtime adapter alone.

## Program binding

- Backlog: ADP-009D, day-28 public-scene rehearsal; reusable dependencies for
  ADP-021/040 preparation, ADP-060/061 execution/sealing, and ADP-080 delivery.
- Observed gap: the launch specialist has no tools, multimodal SDK invocations
  cannot inspect additional evidence, CAD compilation failures are not returned
  to the author, and episode interpretation cannot investigate sampled gaps.
- Existing infrastructure: retain the supervisor tool registry, authority
  envelopes, stage replay, completed-prefix adoption, queues, qualified workers,
  scoring, official billing reconciliation, teardown and delivery readback.
- Reversible change: selectable runtime adapters and bounded specialist loops
  using those existing capabilities. There is one execution owner per operation.
- Completion evidence: merged implementation, applicable tests, deployed exact
  identities, admitted configuration, real bounded comparative runs, and an
  end-to-end receipt showing autonomous progression and consumer readback.

## Requirements and acceptance evidence

Each row remains incomplete until its named evidence exists and has been read.
Hermetic rehearsal is necessary but does not establish live service behavior.

| ID | Requirement | Required evidence | State |
| --- | --- | --- | --- |
| A01 | Provider-neutral task, tool, result and runtime identity | Contract tests; legacy receipt compatibility | pending |
| A02 | Real Agents API session adapter | Official-schema fixtures and admitted live request/readback | live protocol verified; operational job proof pending |
| A03 | Durable ownership, operation deduplication and uncertain-outcome reconciliation | Crash, duplicate event and concurrent owner tests | pending |
| A04 | Continue, inspect, cancel and cleanup across process restarts | Fake and live session lifecycle receipts | pending |
| A05 | Enforce disclosure and accepted retention/trace policy | Negative admission tests and exact account configuration | pending |
| A06 | Preserve strict SDK budgets; honestly admit managed-budget uncertainty | Reservation tests, project guard verification, official cost closeout | pending |
| A07 | Existing supervisor capabilities callable through the runtime bridge | Real registry-to-handler-to-worker-to-receipt rehearsal | pending |
| A08 | Persistent supervision and permitted capability revisitation | New-evidence revision tests without duplicate completed work | pending |
| A09 | Autonomous failure investigation and preauthorized recovery | Retained-job replay, adoption, refusal and recovery receipts | pending |
| A10 | CAD author/compile/inspect/repair loop | Actual CAD kernel fixtures, bounded repair and independent validation | pending |
| A11 | Episode evidence investigation | Synchronized interval/crop retrieval with digest and rights validation | pending |
| A12 | SAM and appearance evidence inspection with independent final acceptance | Mandatory coverage, localized defect and disclosure tests | pending |
| A13 | Preserve placement/native feedback, parameter and variation contracts | Existing focused regressions and runtime parity | pending |
| A14 | WebApp ADP operator integration and runtime status | Authenticated route tests, mocked browser coverage, live readback | pending |
| A15 | Hands-off event progression and terminal delivery | Restart/outage drill and signed consumer/media readback | pending |
| A16 | Existing Paperclip execution record retained | Selected ADP worker integration and duplicate-owner tests | pending |
| A17 | Frozen comparative corpus and promotion decision | Independent quality/cost/intervention/latency measurements | pending |
| A18 | Complete release and rollback | Protected-main checks, exact deployment, resource closeout, rollback drill | pending |

## Scientific and operational boundaries

One final verdict may follow several tool-assisted steps. Keep all required
review views, freeze candidate identity before final acceptance, and do not
rerun an unchanged rejection merely to obtain an acceptance. Candidate authors
cannot alter their validators or see withheld physical outcomes. Runtime success
is separate from scientific acceptance, billing closeout, resource release,
notification and recipient delivery.

## September 11 integration checkpoint

Foundation PR #1834 and worker/replay PR #1835 are merged on protected main.
An admitted synthetic Agents API session on source `a521bb18d4c16e9949fd885fad069980d281eaec`
returned the required structured output, retained result digest
`sha256:5879928c7d8774278f455bd42c4ff8b1fa10701c0f4ec7b046e0c76371e2ce73`,
and completed deletion with readback. Two scoped SDK calls also completed with
validated output. Their combined model-price estimate was USD 0.03002; this is
not official billing. The owner set the dedicated inference project limit to
USD 30, and the dashboard's hard-enforcement switch was observed enabled.
Billing delay and the absence of an exact managed-task cost cap remain explicit.

The next integration publishes server-admitted task references through the
existing signed WebApp interface. A durable outbox survives lost responses;
WebApp retains task state and polls the same Pipeline owner. Browser actions
select admitted tasks and cannot supply a prompt, path, tool or authority.
Cancellation before enqueue is committed atomically with task registration.
The managed project observation is checked for current scope and expiry.

The producer uses `PIPELINE_SYNC_TOKEN` and the origin of
`PIPELINE_SYNC_WEBAPP_URL`; an optional `BLUEPRINT_AGENT_WEBAPP_ADMISSION_URL`
must name `/api/internal/pipeline/agent-execution/admissions`. WebApp uses
`BLUEPRINT_AGENT_PIPELINE_BASE_URL` with the `/api/live-pipeline` prefix and its
existing `CAPTURE_UPLOAD_INTAKE_FORWARD_TOKEN`. These are controller settings,
never request fields or model tools. No raw prompt or source bytes enter this
admission publication.

Actual retained-job execution, production deployment, browser readback,
specialist integration and the frozen comparison corpus are still incomplete.

The current public beta lacks a documented exact per-run inference cost cap and
does not support ZDR or configurable tracing. Managed execution must bind an
explicitly admitted retention and budget policy; strict per-call work can use the
SDK through the same task contracts. Neither a prompt nor a timeout is an exact
spend cap. Missing usage is unknown, not zero.

## Concurrent work

Pipeline PR #1833 adds asset-authoring and policy-delivery work. Integrate with
its final released interfaces when it lands; do not overwrite its active
checkout or duplicate the CAD kernel/runtime implementation.

## Implementation checkpoint: runtime foundation

The isolated implementation now includes an environment-free Agents API
adapter, a project-scoped SDK fallback, a durable SQLite operation journal,
same-session continuation across immutable task revisions, signed webhook
wakeups, and a restartable task scheduler. Cancellation and operation start
share an atomic transition; cancellation and result acceptance are also checked
at the commit boundary. An uncertain provider request retains its operation or
reservation and cannot silently switch runtimes or dispatch again.

The capability bridge calls the existing supervisor registry and observation
validator. Image tools return admitted exact source bytes or pixel-verified
crops while preserving color interpretation. Episode tools can retrieve
recorded camera intervals and traces without altering the sealed score.

Focused validation covers the actual installed SDK through a fake HTTP
transport, the managed API through persisted wire-shaped sessions, real registry
inspection handlers, and sealed episode fixtures. The SDK incumbent regression
selection also passes. These checks protect project and disclosure binding,
pre-request reservations, interruption/restart, cancellation races, uncertain
outcomes, stale context/turn rejection, crop fidelity, score preservation, and
signature verification. They do not establish live account admission, budget
parity, deployment, a comparative pilot, or autonomous result delivery.

The remaining integration still includes the production authority/configuration
store, service and queue entrypoints, retained-stage recovery, CAD execution,
independent visual-review acceptance, the Website operator and delivery path,
the selected Paperclip worker, the frozen comparative corpus, deployment and
rollback. No requirement is marked complete on the basis of this foundation.

Validation at this checkpoint:

- `PYTHONPATH=src python -m blueprint_pipeline.impacted_test_selection --base
  origin/main`: 316 selected tests passed in 94.59 seconds. This covers the
  changed runtime and supervisor consumers plus the existing security,
  success-claim, release and paid-admission sentinels.
- `PYTHONPATH=src python -m pytest tests/test_task_evaluation_supervisor.py`:
  71 passed; preserves the incumbent supervisor's execution/replay behavior.
- Changed-file Ruff and whitespace checks pass. Independent review found no
  remaining correctness/security blocker in the foundation after its recorded
  cancellation, continuation, scope and cleanup fixes.

An earlier selector diagnostic timed out at 120 seconds after selecting 725
tests; it is not a passing receipt. The selector matched common basenames inside
unrelated longer filenames and treated every package initializer as a match.
The regression-tested correction retains module/path and literal file-loader
matches, mandatory sentinels and the existing time/breadth limits. A deadline
fixture now advances its clock after the request begins, so SDK import time
cannot substitute for the intended in-flight cancellation test.

## Production integration checkpoint

The intake now registers authenticated agent enqueue/status/cancel/cleanup
routes. Callers select a server-admitted task id; they cannot supply prompts,
paths, tools, credentials or authority. Task owners are durably bound to the
admitted task revision. The production factory rechecks the current config,
release, model, project, budget, evidence and tool scope before execution.
Secrets require private file permissions. Managed API execution defaults off
until its separate retention/project policy is admitted; the SDK path retains
its reservation gate.

Release-owned systemd units run the durable agent worker and a separate offline
retained-stage replay worker. The latter checks its actual service identity and
isolation settings, has no network/GPU or provider-secret access, and invokes
the existing saved-child replay command with an explicit clean environment.
Queued replay requests bind the existing operation and saved job bytes. An
interrupted child is preserved rather than silently rerun. SDK callers can wait
for that same operation without another model request. A trusted preparation
entrypoint builds a sanitized diagnostic task from a retained terminal child.

The focused production/runtime/replay/deployment selection passed 125 tests in
13.76 seconds. This includes the real SDK and registry loop through a fake HTTP
provider, HTTP HMAC/client ownership, cancellation and revocation, deferred
operation deduplication, offline-worker handoff, saved-input preservation and
the existing stage-replay/deployment contracts. These are hermetic checks;
deployment, actual isolated-host execution and live inference remain pending.
