# G1 selected-policy paid dispatch: reviewed TDD plan

## Scope and authority

Owner-requested G1 development extension of ADP-050, Day 28: use the same
retained site/task with G1 manipulation and movement candidates, expose a
team-owned selection flow, and support endpoint, container, and noncontainer
delivery through the same observation/action contract. ADP-060/080 receipt,
media, and review seams are exercised with a `development_only` ceiling.

The owner authorized bounded GPU spend and requires merge and push. Scene
841757 and model approvals already recorded on issue #2245 remain the source
of execution authority. This plan grants no public redistribution, physical
robot motion, new scene rights, or access to another team's artifacts.

Completion is the whole selection-to-private-review path. Registration,
synthetic conformance, one failed or completed episode, green tests, and a
merged PR each prove only their own boundary.

## Observed gaps

- Pipeline PR #2351 accepts a signed selected profile, checks a separate
  operator approval, seals an execution packet, runs a scored worker, and
  verifies retained media. It has no canonical paid controller for that packet.
- WebApp PR #737's UI and signed request pass hosted checks. The launch must
  remain unavailable until Pipeline dispatch and private result delivery exist.
- The live Vast Isaac image has no Docker, Podman, bubblewrap, or Docker socket;
  user namespaces are unavailable. Its local container/artifact adapters cannot
  execute there. Adding a Docker command to that worker would preserve the gap.
- The existing built-in four-policy controller owns allocation, spend gates,
  watchdogs, and provider teardown. Reuse these seams; do not launch directly
  from HTTP intake, the worker, or a new standalone provider module.
- PR #2351's 72 unreachable-materializer count reproduces on exact base
  `a661c47606f67906d594ad6c3ce951b47a47966e`, with no G1 functions in that
  population. Three import-isolation failures also reproduced on base.
  Remaining CI failures require exact baseline classification before merge.

## Required end state and evidence

| Requirement | Authoritative completion evidence |
| --- | --- |
| Choose G1 and manipulation/movement candidates on the same scene/task | Rendered authenticated UI, owner-scoped registry, exact selected IDs/profile and source packet digests in accepted intent |
| Run the selection without manual Python glue | Installed queue/CLI entry point reaches canonical allocator, single-attempt start record, current authority checks, live allocation and terminal result |
| Embodiment-specific placement and wire contract | Registered G1 preset plus exact scene packet; independent setup and action/observation validators; no hidden evaluator state reaches the policy |
| Authenticated endpoint | Origin/secret review, private credential staging, synthetic handshake, real scored G1 episode and secret-free receipts |
| Digest-pinned OCI image | Provider-native separate policy container identity, synthetic handshake, real scored episode, independent policy lease teardown and billing |
| Noncontainer archive | Governed acquisition, exact archive hash and entrypoint, isolated policy runtime, synthetic handshake, real scored episode and process/provider teardown |
| Reusable configuration | Declarative embodiment/policy profile and delivery bindings; adding a candidate does not require changing HTTP/UI orchestration or simulator task scoring |
| Complete private output | Query counts, deterministic score, lossless policy-input frames, head/overview review video, digest verification, owner-only review page and working URL |
| Resource closeout | Both leases, if present, terminal; posted provider charge receipts, fresh billing reconciliation/provider inventory; no orphan instance |
| Ship the work | Focused gates, exact PR head checks, merged/pushed Pipeline and WebApp changes, deployed identities and exercised deployed flow |

The built-in Diffusion/pi0.5 manipulation/movement campaign must retain all four
results, including task failures. A complete scored episode need not succeed
at the task. A missing episode or its media is not equivalent to a task failure.

## Implementation sequence

1. Finish the live built-in four-policy campaign, inspect native failure logs
   if needed, verify media and resource closeout, and ingest private review.
2. Finish signed selected-profile preparation and provider bundle. Recheck
   registry, owner, approval, expiry, scene digest, profile digest, runtime
   source, and implementation identity before preparation and each paid
   boundary. Bind all staged credentials/artifacts to the exact profile.
3. Add a canonical allocator probe for selected G1 policy execution. HTTPS
   uses the approved endpoint. OCI/artifact modes use a separate controlled
   policy runtime and a private stream client to the Isaac worker; no nested
   Docker or privileged host workaround. Provider allocation starts only from
   the canonical allocator. The combined policy/simulator budget is bounded.
4. Add the single-attempt queue dispatcher and settlement path. A durable start
   record precedes allocation; service interruption or observation timeout
   never causes a second launch. Independent watchdogs survive the dispatcher.
5. Bind result verification and private ingest to the accepted owner/intent.
   Register video/frames only after byte verification, retain task failure
   truth, and return the owner-only review URL to the WebApp workflow.
6. Verify the three delivery modes on the G1 site/task path, then enable the UI
   through coordinated Pipeline/WebApp merges and exact deploy receipts.

Before choosing a policy-container launch shape, inspect its provider-native
entrypoint, stdin/stream transport, nonroot execution, resource and network
isolation, and image identity through read-only request/capability checks.
Tests must pin the actual supported shape. Unknown provider capability blocks
that mode before spend; it must not be replaced by a synthetic-only success.

## TDD cases: red first, then smallest implementation

### Authority and bundle

- A valid signed intent plus matching current operator approval yields a
  credential-free, exact-byte provider bundle; every mismatch/expiry/revocation
  fails before allocation or site-observation disclosure.
- Missing or altered runtime bytes, tokenizer assets where required, source
  packet, OCI digest, archive digest, entrypoint, or HTTPS origin are rejected.
- Credential values, signed URLs, endpoint responses, and raw policy stderr
  never enter public/loggable receipts. Raw stderr stays private quarantine.

### Canonical paid path and paired leases

- Hermetic lifecycle test drives real preparation, allocator dispatch, worker,
  verification, settlement, and private ingest using fake provider transport.
  It asserts normalized G1 observations/actions and deterministic scoring.
- Endpoint mode allocates only Isaac. OCI/artifact mode allocates an isolated
  policy lease plus Isaac; total admitted rate/spend is bounded together.
- Failure before the second allocation, policy boot failure, simulator boot
  failure, malformed action, policy timeout, worker exit without receipt, and
  settlement failure each preserve typed failure and tear down every lease.
- Authority changes during slow preparation/credit/billing refresh block the
  next allocation. Busy slots are respected; available guarded slots are used.
- Replay/idempotence does not allocate twice. Parent exit cannot kill the
  independent watchdog or be interpreted as provider-zero proof.

### Scoring and delivery

- Reject zero queries, absent/altered score, missing lossless frames, incorrect
  frame counts, missing/altered videos, foreign owner/intent, or nonterminal
  provider state. Preserve scored task failure as failure.
- Owner can view their result; another owner and unauthenticated user cannot.
  Browser test exercises choose → request → progress → review with real route
  contracts. A mocked render alone does not prove deployed authorization.

### Verification gates

Run focused changed-contract tests, provider import closure, lifecycle
rehearsal, launch-bypass/spend/watchdog/teardown sentinels, and changed-file Ruff.
Use full-suite results only to classify a concrete dependency boundary or for
explicit promotion; reproduce unrelated failures on exact base instead of
changing retired lanes. Live tests use reviewed assets, current admission,
independent watchdogs, bounded spend/TTL, and retained native logs.

## Review record

Codex design review, 2026-09-27: reviewed against the selected worker/session,
canonical built-in allocator, queue dispatcher, delivery profile, paid-output
verifier, live provider capability evidence, and PR #737 workflow contract.
The plan preserves the full user goal and explicitly closes the provider and
delivery gaps. Accepted for test-first implementation. This is an agent design
review, not a new human rights approval or proof of live execution.

Main integration on PR #2351: merged tokenizer fix `fe6061cc...` into the draft;
27 focused worker/output/assembly/episode/tokenizer tests passed. The draft is
still not an operational selected-policy launch and is not ready to merge.

## Implementation checkpoint: supervised selected worker

2026-09-27: the owner scope remains the **same existing Franka run
configurator**, extended through embodiment → compatible policy/configuration
→ actual episode → private results. A separate setup form or a selector-only
integration is not an acceptable replacement. PR #737's G1 component is
currently nested in `PolicyCanarySetup`; preserve that integration.

Implemented a callable/CLI selected-worker supervisor. It runs one actual
child, retains a private log and digest-bound exit receipt, enforces a maximum
45-minute episode deadline, and terminates the child process group on timeout.
A zero exit after native Isaac close can recover only the exact bound preclose
and completed supervised episode. The paid-output verifier additionally
requires the retained zero-exit receipt and every score/frame/video byte.
Timeouts, failed exits, foreign preclose, missing exit evidence and missing
media are not completed attempts. Provider teardown and billing stay unproven
in these worker receipts.

Test-first verification: new tests initially failed on the missing supervisor;
14 selected-worker/output tests passed after implementation, including real
subprocess exits and timeout, and the new runner's normal/native-close paths
against retained shared-scene lifecycle evidence. Changed-file Ruff and CLI
import/help pass. This is hermetic worker proof, not live team-policy inference.

Final focused selected-worker/worker/output/episode/packet run: 23 passed;
changed-file Ruff and whitespace checks pass. The shared canary import-closure
run exposed an omission from PR #2404: `native_g1_policy_server_supervisor`
imports `native_g1_pi_tokenizer_assets`, which was absent from the shared runtime
module list. Added it; all six closure tests now pass, including sealed-bundle
isolated imports. All 19 policy-canary lifecycle rehearsal cases passed in the
combined run. The one-line closure repair is also submitted independently as
PR #2406 so unfinished selected-policy wiring does not hold that repair back.

Still required: exact selected provider bundle, current authority recheck after
admission, canonical selected-profile allocator/queue/settlement, separate
policy runtimes for OCI/artifact delivery, live delivery-mode qualification,
private owner URL, coordinated merge/deploy and deployed workflow proof.

## Implementation checkpoint: selected provider transport

Sealed the exact selected execution packet, derived G1 scene lineage, pinned
publisher checkout, reviewed external Isaac runtime and offline SONIC assets.
The transport remains `sealed_not_admitted`: it neither allocates compute nor
contains a policy credential. Git configuration is reduced to the public
publisher origin; host credential helpers and extraheaders are not transported.
The host loader reopens current registry/approval and checks every shipped byte,
all launch fields against the selected packet, archive traversal/type rules and
the external runtime layer's exact bytes. A self-consistently rehashed manifest
cannot change the owner-approved intent, objective or image.

Added the provider entrypoint and selected-worker adapter. Runtime provisioning
runs in a separate interpreter before worker imports; media bootstrap failures
retain a typed terminal receipt. HTTPS requires a protected credential file.
OCI/archive delivery currently refuses a missing separate runtime and never
tries nested Docker. This refusal is an unfinished capability, not completion
of the container/noncontainer requirement; canonical admission must enforce it
before spending until the paired runtime exists.

Test-first checks exposed unbound manifest fields; the loader now binds them.
Twenty-seven focused bundle/runtime/worker/supervisor cases passed; the final
isolated sealed-bundle CLI import and shell syntax case passed separately.
The isolated test uses installed numerical dependencies analogous to Isaac but
asserts every imported Blueprint/RFC8785 module comes from the sealed bundle.
Six shared provider closure cases and nineteen lifecycle rehearsal cases passed.
Changed-file Ruff passes. These are hermetic transport/worker proofs, not an
admitted selected-policy launch or live inference proof. Canonical allocator,
single-attempt queue, separate runtimes and owner result delivery remain next.

## Admission checkpoint: protected endpoint credential resolution

Observed blocker: the selected HTTPS worker accepts a protected credential
file, but no controller resolver binds that file to the current team, exact
profile and separately reviewed endpoint origin. Accepting a caller-supplied
filesystem path would leave that authority boundary open.

Small reversible implementation: an operator-owned private registry maps a
secret reference to a fixed file beneath its own `credentials/` directory.
Resolve only after reopening the existing intent/registry/operator approval;
require an active unexpired entry matching owner, profile digest, secret
reference and approved origin. Registry, directory and file permissions and
ownership must be protected. Reject symlinks, traversal and multiply linked
secret files; bound and validate token bytes without emitting them. Return a
private file binding with a safe metadata projection. Reopen this binding at
the mutation boundary; changed/revoked credentials must refuse allocation.
This uses the existing Vast private-file transport, adds no secret service and
does not alter the owner-approved provider stack.

Red-first cases: exact current binding resolves; foreign owner/profile/origin,
revocation, expiry, duplicate references, malformed registry or token, unsafe
permissions, symlinked ancestors/files, traversal and hardlinks refuse. Real
approval revocation after initial resolution also refuses. Receipts/repr/error
text contain neither credential values nor credential file paths. The resolver
does not allocate or contact the endpoint. Existing selected bundle/worker
tests continue to prove their separate contracts.

Agent design review, 2026-09-28: checked selected HTTPS client, operator approval
validator, current authority loader, provider worker and Vast secret transport.
Accepted for test-first implementation under ADP-050 Day 28; this is a required
admission dependency, not a substitute for the remaining paid dispatch, paired
runtime and live qualification requirements.

Use this resolver in a selected dispatch-input preflight that reopens the
sealed bundle and live authority, bounds launch budgets by the signed request,
and holds the private credential binding apart from loggable metadata. Its
mutation-boundary recheck must reject changed bundle bytes, valid but changed
approval, secret replacement, revocation and authority expiry. OCI/archive
must refuse a missing paired runtime here, before admission or staging, rather
than first discovering it in the paid worker. Its status explicitly means
inputs verified, not spend admitted; allocator/queue wiring remains required.

Implementation verification: 63 cases in the combined selected credential,
dispatch-input, bundle, provider-runtime, worker and supervisor run passed.
One added valid-approval replacement case initially used the wrong canonical
digest helper in its fixture. Corrected it to the approval contract's helper;
the exact failed case then passed in isolation (64 unique cases verified).
All six sealed provider import-closure and nineteen lifecycle rehearsal cases
passed together. Changed-file Ruff passed. No new dependency, secret service,
policy inference or GPU allocation was introduced by these checks.

## Provider transport checkpoint: selected policy kind

Observed blocker: the selected bundle entrypoint exists, but the shared Vast
transport does not recognize its kind or result filename. Its generic output
cap would omit large lossless observation files. Close this seam before adding
the canonical allocator probe. Register the selected kind in the existing
Isaac image/entrypoint/secret/external-layer/output/recovery contracts. Validate
the sealed manifest, execution packet and all artifact bytes without converting
input sealing into paid authorization. Retain the entire selected worker output
and require its exact terminal result. No command-execute restart fallback for
this kind: a missing log marker does not prove the owned worker never started.

Red-first cases cover real built-bundle acceptance, altered artifact and
resealed launch-field refusal, exact readiness/entrypoint selection, generated
bootstrap shell syntax, required result and full-frame retention, private token
transport without receipt leakage, and selected native terminal completion
without any physical/billing/public-rights upgrade. Typed worker failure and
zero queries never become successful inference. All existing selected worker,
provider closure and lifecycle checks remain required before a paid attempt.

Agent review: examined actual Vast bundle preflight, image bootstrap, native
terminal contract and output recovery. Accepted under ADP-050 Day 28; canonical
paid grant/one-attempt queue and all three live delivery modes remain required.

Canonical allocator increment: accept only a previously sealed selected bundle
and current owner/operator/secret registry arguments. Reuse deployed-release,
spend-lock, credit, launch-gate, paid-grant, watchdog and adapter closeout seams.
At provider create, reopen identity and selected dispatch inputs; consume an
exclusive single-attempt record binding exact intent/profile/approval/bytes and
budgets. Repeated invocation must not allocate twice. Post-run, rehash the same
bundle and independently verify the selected worker's score, frames and videos.
Do not label that result billed, delivered or publicly publishable.

Hermetic tests drive the real controller boundary through fake canonical
transport, checking revoked authority after slow preparation, duplicate
consumption, unsupported modes before adapter invocation, control identity
change, typed failures and independent watchdog/secret transport arguments.
Review: existing paid controller supplies these boundaries; no direct provider
API, new service or separate web intake is permitted. The installed selected
queue and delivery settlement remain the next integration increment.

Collection admission review: the current transport retains both ZIP and extracted
lossless output. Its declared transfer forecast is 12 GB of provider inputs plus
10 GB of output; reserve 32 GB for inputs and both output copies through the
existing disk ledger, with its separate floor and live reservations. This is a
conservative declared forecast, not a proven upper bound on episode output.
Dry transport must refuse insufficient observed headroom; execute must acquire
and renew the reservation before staging or grant use, and hold it through
verification/closeout. Test shortage before transport, reservation release after
failure, final renewal refusal before create, and CLI dispatch. Streaming and
generation bounds remain required to remove this collection limitation.

Verification, 2026-09-28: the combined selected allocator/transport/worker/
bundle/media/preflight/import/lifecycle run completed with 71 passes and one
artifact-digest refusal while source edits overlapped test bundle sealing.
The same preflight case passed in isolation once writes stopped. A concurrent
source change is consistent with that refusal; the exact mismatched member was
not retained long enough to confirm it. The two added reservation-loss cases
and the independently retained score/frame/video corruption case passed in a
targeted three-case run. After the final provider receipt change, the endpoint
worker test passed and all six sealed import-closure cases passed again. All
nineteen lifecycle cases passed in the combined run. Changed-file Ruff and
diff whitespace checks passed. These are hermetic contract checks, not GPU
qualification. A started but unverified worker now reports query state unknown,
rather than inventing zero queries. No new paid attempt or data deletion.

## Next integration checkpoint: selected intake to durable dispatch

The existing campaign preparer/dispatcher consumes a different intent schema
and always runs the four built-ins; it cannot dispatch the team's chosen profile.
Reuse its immutable records, lock, canonical allocator subprocess and fresh
spend-guard seams. A selected-policy preparer must re-open current authority,
derive manipulation/movement scene paths only from the owner-scoped registry,
seal the existing execution packet and selected bundle, and take SONIC assets
from an operator-owned configuration. Cached preparation must reverify every
bundle byte and exact approval instead of silently rebuilding under a new
release. The queue supplies approvals by exact intent ID from an operator root;
team input never supplies host paths or credential bytes.

The dispatcher must replay the complete current dry preflight, refresh billing
and spend admission after slow preparation, re-open selected authority, and
write an exclusive execution-start record before invoking the canonical paid
allocator. An interrupted start requires reconciliation of its exact attempt;
it does not authorize a replacement allocation. Terminal dispatch remains
pending settlement and private ingest. Tests cover both objective choices,
owner/registry/approval changes, stale preparation, absent approval, revoked
authority after guard refresh, interrupted execution and repeat invocation.
Agent review accepted this as the next required bridge to the existing shared
configurator; neither inspection-only preparation nor endpoint-only execution
will satisfy the full delivery requirement.

Selected preparation increment verified: the no-spend CLI now reopens authority
before and after its lock, chooses only the registered manipulation/movement
directory, seals the existing execution packet/bundle, and rechecks authority
after building. Cached bytes, exact implementation and approval are immutable;
missing or symlinked operator directories refuse before building. Five focused
cases passed together, including real fixture bundle sealing and revalidation,
navigation packet routing, cached byte tampering, release/approval drift and
revocation after lock wait. The initial fixture lacked registered publisher
source bytes and was corrected without weakening the source validator. The
lock-recheck test first demonstrated the missing recheck, then passed with it
encoded. Ruff and whitespace checks passed. Queue execution, host preparation
reservation, paid admission and settlement remain separate unfinished steps.
