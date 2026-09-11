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

Each core requirement remains incomplete until its named evidence exists and has been read.
Optional integration rows describe their activation gates. Hermetic rehearsal
is necessary but does not establish live service behavior.

| ID | Requirement | Required evidence | State |
| --- | --- | --- | --- |
| A01 | Provider-neutral task, tool, result and runtime identity | Contract tests; legacy receipt compatibility | implemented; contract and compatibility tests pass |
| A02 | Real Agents API session adapter | Official-schema fixtures and admitted live request/readback | implemented; live retained-job investigation completed with signed Website readback |
| A03 | Durable ownership, operation deduplication and uncertain-outcome reconciliation | Crash, duplicate event and concurrent owner tests | implemented; crash and duplicate-owner rehearsals pass |
| A04 | Continue, inspect, cancel and cleanup across process restarts | Fake and live session lifecycle receipts | implemented; live same-session continuation across processes and cleanup verified |
| A05 | Enforce disclosure and accepted retention/trace policy | Negative admission tests and exact account configuration | implemented; operational project policy observed, specialist live admissions pending |
| A06 | Preserve strict SDK budgets; honestly admit managed-budget uncertainty | Reservation tests, project guard verification, official cost closeout | implemented; scoped SDK and project guard verified, official comparison cost closeout pending |
| A07 | Existing supervisor capabilities callable through the runtime bridge | Real registry-to-handler-to-worker-to-receipt rehearsal | implemented; production registry and deferred replay rehearsals pass |
| A08 | Persistent supervision and permitted capability revisitation | New-evidence revision tests without duplicate completed work | accepted-run producer and single-owner failure routing implemented; earlier automatic SDK discovery, completion and cleanup observed |
| A09 | Autonomous failure investigation and preauthorized recovery | Retained-job replay, adoption, refusal and recovery receipts | implemented; real saved CPU stage now replays to completion; live preauthorized recovery/adoption proof pending |
| A10 | CAD author/compile/inspect/repair loop | Actual CAD kernel fixtures, bounded repair and independent validation | existing CAD repair loop adopted from main PR #1833; comparative pilot pending |
| A11 | Episode evidence investigation | Synchronized interval/crop retrieval with digest and rights validation | implemented for SDK and API; real-SDK image tool and receipt rehearsal passes; live admitted specialist pilot pending |
| A12 | SAM and appearance evidence inspection with independent final acceptance | Mandatory coverage, localized defect and disclosure tests | implemented for SDK and API; required full views and independent final acceptance preserved; live admitted pilot pending |
| A13 | Preserve placement/native feedback, parameter and variation contracts | Existing focused regressions and runtime parity | incumbent paths retained; regression coverage preserved |
| A14 | WebApp ADP operator integration and runtime status | Authenticated route tests, mocked browser coverage, live readback | deployed through Website PR #571; authenticated live SDK/API task, cleanup and source readback verified |
| A15 | Hands-off event progression and terminal delivery | Restart/outage drill and signed consumer/media readback | automatic diagnosis and signed consumer delivery observed; full-run outage/terminal-media proof pending |
| A16 | Optional Paperclip integration retained | Before enabling: selected-worker linkage and duplicate-owner evidence | adapter and bounded handoff retained; activation disabled and deferred; core Pipeline does not depend on it |
| A17 | Frozen comparative corpus and promotion decision | Independent quality/cost/intervention/latency measurements | 60 paired runs complete; SDK retained as incumbent; official cost and intervention measurement pending |
| A18 | Complete release and rollback | Protected-main checks, exact deployment, resource closeout, rollback drill | production promotion and exact deployment observed for prior integration; latest release and live rollback proof pending |

## Current implementation checkpoint

Website PR #571 is deployed at `47700a2464724faecc87bc69d7a4c1da6c004b08`
on both web and worker; deployment `34585076235` verified exact identity and
both health endpoints. Pipeline PR #1849 is merged at
`c2b90c7612773954b1f091193adff238a70691a7`; its production promotion is running.
The retained `sam31-0d64ff5c54c39d461b338ba6bf9a0a9818bd688abf7e80de12f1bf825538d021`
source-selection job now completes in the isolated offline replay against the
candidate code. Its diagnostic phase receipt cannot be adopted as production
execution evidence. No model call, provider mutation or GPU execution occurred.

The episode and visual producers now admit either runtime explicitly. Existing
managed tasks keep their original runtime. SDK investigations use the same image,
trace, interval and crop tools, `store=False`, disabled tracing, and an inference
reservation before execution. Their bounded default is three model turns,
100,000 input tokens and 4,000 output tokens; managed defaults remain unchanged.
These different bounds must be reported in comparative pilots. Missing managed
admission never silently switches an already-created task to another runtime.

The installed SDK receives typed image content, not a textual base64 surrogate.
A separate cumulative context ceiling counts text conservatively by UTF-8 bytes
and admitted PNG images by their unresized 32-pixel patches plus framing headroom.
The qualified model list and 1.2 image multiplier follow the September 11
[official image-input specification](https://developers.openai.com/api/docs/guides/images-vision).
Unsupported models, remote image URLs and over-budget tool context stop before
the next request. Wire-size and task-operation limits remain separately enforced.
The original one-pass SDK reviewers and summary path remain available.

Live full-run recovery/adoption, an outage/rollback rehearsal, admitted specialist
comparisons, official per-task cost and operator-intervention measurement remain open.
Paperclip issue/worker readback is a separate optional activation gate. The older
checkpoints below describe their recorded releases; their original pending
statements are historical.

## September 11 deployed integration and comparison

Pipeline `fddae5bc7a731e9541a7bbfdf02d89b5afbc71ca` is deployed with
verified production provenance from run `34565131545` (19,210 tests). The
durable agent worker is active. The exact retained-child investigation
`adp-assessment-retained-replay-20260911` completed through the real managed
API and isolated replay worker. Its replay was refused for disk capacity before
the saved stage handler ran; this is preserved as a failed rehearsal. The
investigation did not confirm or repair the original source-reference failure.
Its signed admission reached WebApp after correcting the configured HTTPS
origin from the redirecting `www` alias to `tryblueprint.io`.

Website `8a71a7fd759be33c843e34786f9db904da289421` is live on both web and
worker services; deployment `34565774036` verified identity and both health
endpoints. The selected Paperclip bridge now supports instance configuration
and secret references. A live selected Paperclip worker is still unverified.

The frozen corpus completed 30 SDK and 30 API runs on the same Pipeline release,
model, effort, cases and tools. SDK produced 29 valid outputs and 16/18 correct
held-out decisions, at a 4.20-second median. API produced 27 valid outputs and
15/18 correct held-out decisions, at a 32.70-second median. All 60 sessions were
cleaned. Summary digest:
`sha256:1045313eb4e3b3a19952348515cc0f5ad4311e6d56bc32e89583c8e2963494d8`.
These results retain the SDK as incumbent and keep managed execution in its
explicit pilot scope. They do not establish intervention savings or final cost.
`automatic_failure_runtime` therefore defaults to `openai_agents_sdk` separately
from `managed_api_enabled`. Enabling managed specialist eligibility does not
select it for routine failure investigations. An explicit API selection still
requires its independent project/disclosure admission. Existing tasks keep
their original runtime; a failed or uncertain call never switches providers.
Earlier HTTP schema rejections and a trial with missing SDK process opt-in are
retained separately and are not passing paired comparisons.

The next closeout adds automatic cleanup to newly prepared one-shot tasks,
retains supervision sessions until revocation/expiry, and keeps Website polling
until cleanup is observed. A replay CLI admission refusal now writes a typed
report so the investigator can identify a disk-capacity refusal without access
to raw logs. Neither change bypasses the original disk or execution gates.

## Accepted-run supervision producer

The accepted scene-intent controller can now create its own persistent watch
when `automatic_run_supervision` is enabled in the private agent configuration.
Registration occurs after authenticated intake validation and before advancing
the existing controller. Only sanitized status, phase, blocker and receipt
references enter the reasoning input. In-progress heartbeat changes do not
start model calls; intervention and terminal states can create a new revision.

One watch owns the run. The separate failed-child subscription defers to that
owner, which can select a unique failed child of the current digest-bound
preparation link. It then uses the existing saved-replay task builder and exact
configured recovery binding. Multiple matching children or recovery bindings
are refused. This does not renew an expired intent or create paid authority.

The watch reserves at most three revisions, each capped at the smaller of USD 1
and the server task limit. A durable lifetime ledger keeps that cap across
restarts and code releases. Compatible managed revisions continue their owned
session; a changed tool scope settles and deletes the old session before a new
one can start. SDK revisions clean their prior execution record before advancing.
A terminal controller observation closes the final reasoning session while
preserving the controller's independent execution and delivery receipts.

The flag defaults off and does not change the incumbent controller when absent.
Disabling it revokes automatic supervision; removing a recovery binding revokes
that action even after restart. Production-host and full-run proof still require
an accepted current intent and the exact deployed producer release.

## Paperclip dependency boundary

Paperclip is optional coordination for engineering issues and worker/reviewer
assignment. Pipeline already owns reasoning-task execution, persistent
supervision, saved replay, preauthorized controller recovery and Website
result delivery. Those core paths have no Paperclip dependency.

The original assessment called for retaining the existing integration. That
does not require activating an unavailable Paperclip instance before the core
rollout can finish. Keep its implemented adapters disabled unless that
engineering-worker workflow is selected. Live issue/worker readback remains a
gate for enabling that integration, not a core rollout gate. Fully automatic
source-code repair dispatch still requires an active coding-worker system;
diagnosis and an engineering handoff alone do not establish a completed repair.

## Automatic recovery configuration

An operator may install an explicitly approved supervision budget amendment in
`automatic_supervision_allowances`. Each entry binds one exact intent ID and
digest, an authorization reference, expiry, total lifetime revision limit and
total lifetime inference reservation limit. Both the amount and expiry must fit
the accepted intent. This is a reallocation within the owner's aggregate budget,
not a new aggregate spending grant or a billing reconciliation.

The amendment creates a distinct immutable supervision plan and retains the old
plan, owner record, tasks and reservation events. Every earlier reservation still
counts against the amended total across releases. For example, changing a
three-revision/$3 allowance to six revisions/$6 after three $1 reservations permits
only three additional $1 reservations. Removing or expiring the amendment revokes
its pending inference. The default remains three revisions and at most $3; neither
a restart nor a deployment changes those defaults or releases a hold.

`automatic_failure_investigation` continues to default to diagnosis/replay only.
The private production configuration can now include up to ten
`automatic_recovery_bindings`, using the existing `ControllerRecoveryBinding`
contract. A subscription receives a binding only when both its exact parent
request digest and intent ID match. Ambiguous matches are refused. The binding
must identify `failed_boundary` as its required replay and retain the exact
controller configuration, intent, preparation link and their hashes.

The failure producer then prepares the existing recovery-enabled task without a
manual enqueue. Its model must obtain a successful same-task replay before it
can ask the existing controller to progress. That controller retains current
owner consent, expiration, retries, spend, provider-zero and completed-prefix
adoption checks. Changing the selected binding creates a distinct task identity
and consumes the existing subscription cap. Removing a binding revokes pending
automatic recovery, including after service restart. In-flight operations remain
owned by the original controller and must be reconciled.

This configuration grants no new owner authority and is empty in the live
assessment pilot. No paid retry is enabled by deployment. The selected live
recovery/adoption proof remains open until an already-authorized owner intent
is available; the supplied expired historical intent is not renewed implicitly.

## SDK cancellation closeout

A cancelled or expired SDK invocation can now close its local execution after
all issued tool operations have been reconciled. The task retains a cancellation
receipt with the inference-reservation snapshot. Unknown charges remain held;
local cancellation does not claim that the provider finished the response or
released any independent resource. A later revision can proceed within the
watch's already-reserved lifetime budget, without replaying the old request.
Known cost-bound violations remain in reconciliation and require cost review.

## Scientific and operational boundaries

### Bounded engineering handoff (ADP-009D, day-28)

An unresolved technical replay can now become a bounded task for the existing
Paperclip engineering lane. This closes the gap between returning diagnostic
advice and creating work with a saved reproduction and acceptance requirements.
The private `engineering_policy_file` selects exact allowed source paths,
mandatory test paths, run-id prefixes, handoff count, patch/file limits, an
existing-worker budget reference and a finite worker timeout. It defaults off.
Capacity/authority admission refusals do not authorize code changes.
Only the canonical replay's `refused` stage-handler result is eligible;
`job_refused` remains an owner/input admission result. Offline replay reads
recognized historical administrative contracts through the existing retained
validator, while every new or explicitly paid execution keeps current admission.
The isolated command pins the invoking checkout's interpreter path after all
systemd environment files, so it cannot silently replay an installed older tree.

The producer retains one handoff per diagnosis/replay identity, reserves before
publication, and sends only machine evidence references and the admitted policy.
Model prose does not become engineering instructions. Website admission binds
the existing verified task result; dispatch rechecks current Pipeline authority,
the exact repository workspace, remaining worker budget and timeout. It creates
one Paperclip issue using supported `billingCode`, isolated-worktree and review
policy fields. A separate agent owns review. An uncertain creation is looked up
by exact identity and never blindly posted again.

The trusted controller can bind ordinary existing issues through
`python -m blueprint_pipeline.agent_execution.engineering bind-issue`. Paperclip
selects the task through that server-owned association, not unsupported issue
metadata or model-controlled task parameters. The candidate verifier reads Git
objects to check ancestry, bounded changed paths/patch size, regular file types,
and unchanged required baseline tests. It grants neither test passage nor
production promotion. The original release process remains required.

Completion artifacts for this dependency are the signed handoff receipt, exact
Paperclip issue/readback, candidate scope receipt, independent review and
release evidence. Hermetic dispatch tests or an assigned issue alone do not
prove a completed repair. Deploy the companion Website receiver before enabling
the Pipeline policy; a live Paperclip host and accepted worker policy are still
required for live dispatch.

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

The canonical private configuration is `/etc/blueprint/agent-execution.json`.
Both intake and the worker discover it; an explicit
`BLUEPRINT_AGENT_EXECUTION_CONFIG` override remains supported. Set
`webapp_admission_url` to the exact HTTPS admission endpoint and
`webapp_sync_token_file` to the existing private Pipeline sync token under
`/etc/blueprint/provider-secrets/`. The worker reads that file only for signed
publication, without inheriting the intake environment. The offline replay
worker refuses execution if this token or any configured model/webhook secret
is accessible inside its sandbox.

Actual retained-job execution, production deployment, browser readback,
specialist integration and the frozen comparison corpus are still incomplete.

Website PR #565 is merged and deployed as
`34f4b57fe195efe38b7f542936638c04d59ad12b`. CI and Render deploy workflow
`34552970789` passed: both web and worker records are live at that commit,
`/version.json` matches, and `/health` and `/health/ready` returned 200.
The admitted-task browser fixture passes; a live Pipeline task/result readback
is still required before declaring end-to-end delivery.

Pipeline production-promotion run `34547561826` exposed 11 failures. The repair
batch registers the new service modules and storage roots, separates receipt
and mesh-configuration validation from execution imports, preserves omitted
installed-source environment semantics, exposes five retained closeout
materializers and the teacher view-selection argument, and rehashes the expired
historical quality ledger. Ledger reevaluation preserves its 30-day window,
disabled closure authority, null release bindings, and open/partial statuses.
No GPU stage was launched to discover or validate these fixes.

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

## September 11 specialist and supervision implementation

Pipeline PR #1838 merged as `b26d73a0a133b9692eed7b395ca56f7fd52cec53`.
Website PR #566 merged as `99b6c4eb2b408b9f2296dbfc27ec82be9058d5f6`.
The latter adds typed episode/visual projections and the selected Paperclip
worker bridge; its hosted CI and plugin bundle build passed. The separate
plugin typecheck reports compatibility errors in unchanged legacy SDK calls;
these are not a passing plugin typecheck or live Paperclip proof.

New reasoning tasks can investigate synchronized episode intervals, traces,
and exact image crops; visual inspection preserves every required full view
and never substitutes for independent final acceptance. Supervisor revisions
retain a single owner and can continue the same settled API session. A typed
recovery request requires a successful same-task replay and delegates to the
existing scene controller without increasing the original owner limits.

The automatic failure subscription is optional server configuration. The
existing preparation producer registers its exact parent identity; a new failed
child is admitted once per input/source revision, with a bounded lifetime
inference reservation. Default rollout remains off until the operational pilot
is accepted. No automatic paid retry is enabled by this subscription.

The frozen diagnostic corpus contains 24 retained operational outcomes and six
explicitly labelled controlled variants, grouped into tuning and held-out
partitions. Corpus digest:
`sha256:675bcefa96730c197217aa73170732b5c8c4e6c6115ba7220db5e92dbc717c8e`.
The model receives only one case's evidence, never the expected answer.
Operator-time savings and production qualification must be measured rather
than inferred from a schema-valid response.
