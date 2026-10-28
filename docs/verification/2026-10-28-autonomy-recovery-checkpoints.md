# Autonomy recovery implementation checkpoints

These changes serve beta intake-to-evidence durability and prepared-site screening.
They preserve existing grants, capture truth, diagnostic claim ceilings and site
budget/finalist approval boundaries. They have not been deployed or accepted in
production by this lane.

## AUT-01

Implemented and tested locally: final ledger/live-output promotion uses one
current lease generation and lock, immutable attempt output records repair a
failed live-pointer write, heartbeat loss/I/O failure/expiry/death is observable,
and threaded stage work carries its originating lease. Long-stage JSON
publication, output transaction initialization/promotion, signed preparation
effects, canonical allocator entry and opaque paid-grant use are fenced.
The existing exact warm-session closeout may bypass only the intake lease guard;
its issuer/resource/allocation binding checks remain mandatory. An in-flight
provider outcome stays subject to its original identity and teardown contract.

## AUT-08

Implemented and tested locally: the existing attempt queue retains a named
allocator invocation and fsyncs its authority/bundle/provider resource identity.
A process-held flock spans allocator execution. A provider-create intention is
fsynced before entering the provider adapter. Under that same lock, recovery may
fence an invocation lacking a create intention; every delayed old start refuses
the fence. An exact retained fence can transfer the same screening reservation
to the next unspent generation without resetting the site budget. Legacy intents
and outcomes with a create intention remain reconciliation-only. Their exact
provider identity and bounded-backoff next action remain visible.

The robot-free probe uses the same invocation/fence. Its funded counters are
retained; a funding hold with no create proof still requires funding settlement
or adoption before another grant. The exact preparation-spend request is fsynced
before the RPC, as is its returned funding admission. A missing funding response
retains its attempt slot and never becomes an unfunded ingest. Historical
`create:false` readback validates the exact capture, command and allocation;
both a reserved readback and a not-reserved snapshot remain reconciliation-only.
A not-reserved snapshot cannot fence a delayed original RPC. Reads use bounded
60–600-second backoff and expose the exact accounting next action. Known explicit
WebApp transaction refusals retain exact no-reservation evidence; old invocation
records without the durable funding-request protocol remain held if they ran.
This includes the original bare allocator's launch intent, process receipt or
result, without a named invocation and even if its PID receipt was lost. An old
cached unfunded ingest does not override that ambiguity. Compatibility tests
execute the frozen `541066318` reservation function through the bare default
runner and inject a committed WebApp transaction with a lost response.
No provider-zero snapshot is used to establish
a no-create proof. Recovery of genuine lost provider responses, output transfer
and teardown still needs provider-specific evidence and acceptance observation;
these local tests do not establish those outcomes.

The default robot-free preflight preserves its dry argv even beside a retained
paid manifest. Paid wrapper invocation requires byte-for-byte argument identity.
Regression tests exercise the real staging/preflight/default-wrapper path and
the real allocator with an injected WebApp commit/response-loss boundary.

## AUT-10

Implemented and tested locally: an independent allocator-process monitor reuses
the signed current-source reader, persists exact-attempt stop intent, refuses
create after a changed/revoked source, and applies the existing 60-second polling
and 600-second monitor-unavailable grace. Signaling requires the invocation's
retained PID/start ticks and manifest identity, and lets canonical finally and
watchdogs perform teardown. Observed deadlines can only shorten a session and
are forwarded to the existing provider hard-deadline cap. Partial revoked results
remain historical and cannot become a successful current screening result.
Control observations use a dedicated local lock, reread absorbing stop intent
after remote reads, and retain the minimum deadline. Durable observation
generations keep stale successes from clearing newer service outages. The same
lock protects bounded local create-intent admission/fsync; remote calls and
teardown do not hold it. Deterministic two-observer races cover revoked consent,
shortened deadlines, newer unavailability and the final intent admission gap.

The new signed existing-run control read binds campaign, source, candidate,
configuration, attempt, allocator invocation, preparation, authority and resource.
New launches require its actual intake expiry; they cannot use a missing expiry
as indefinite authority. The wrapper reapplies any smaller observed bound before
allocator execution. Pure strict validators contain no credential/transport
imports. A live exact-resource monitor registry supplies verified observations
to an independently owned controller-to-provider callback; no global sync
credential is delivered in that callback. Delivery failure stops its exact
monitor while canonical finally/teardown remains available.

**Still incomplete at this checkpoint:** qualification owns the pinned-SSH
fixed-file delivery and GPU cell/load/query guards. Their integration and real
own-resource teardown/expiry/cancel acceptance must pass before full AUT-10 is
claimed. Older retained invocations without this contract do not acquire a new
GPU control channel. This document makes no claim that no GPU cell can begin
during an unverified provider stop/teardown propagation interval.

## Other lane1 work and production access

AUT-02 implements an optional retained-DLQ pass in the existing listener:
durable private exact payload/provenance before ack, observe-only by default,
explicit safe pre-lease allowlist, the original unattempted-delivery claim,
bounded backoff, and immutable source/attempt-bound dispositions. Existing
completed/ended/waiting/unknown effects never receive fresh retry authority.
Access denials remain named authorized-action waits. Safe CLI/status data
includes due/oldest age and retention/orphan alerts; raw payloads stay private.
Recovery counts never become terminal-exhaustion or renewed budget evidence.
Offline fault tests and existing handoff regression checks pass. Real permitted
non-customer source→DLQ→recovery acknowledgement proof remains pending; AUT-02
must not be marked acceptance verified from code/IAM evidence.

AUT-09 implements immutable site-local generations in the existing screening
ledger and source workers. It reopens current website consent and the canonical
scene intent, compares pinned scientific/runtime/configuration identities,
rebuilds admitted quotes, transfers unspent reservations in one ledger replace,
adopts completed summaries and leaves uncertain attempts with their original
operations. Historical plan bytes and site commitments survive generations;
no catalog revision, grant, credential or site budget is renewed. Originally
pinned hosted credential availability is supported; missing legacy runtime
identity remains an explicit configuration requirement. Five-minute backoff
retains authority and price failures without hot-looping.

Offline regressions cover a 25-hour capacity outage (including a canonical
pre-create slot-race refusal), one refreshed generation/one allocation, duplicate
timers, completed/unknown histories across two generations, lost ledger commits,
fresh website acknowledgment, real canonical expiry/revocation refusal, price
increases, changed runtime/source bytes, and one privately delivered hosted
credential affecting only its pinned candidate. These fixtures do not establish
production acceptance or new runtime qualification.

Default AUT-08 funding readback remains disabled until the release owner deploys
and verifies WebApp's read-only observation contract and explicitly enables
`BLUEPRINT_PREPARATION_SPEND_OBSERVATION_ENABLED=true`. An older server could
ignore `create:false`; the Pipeline must not send that compatibility probe by
default. Unknown reservations remain held while it is disabled. No observation
permits dispatch, even after the server reports `not_reserved`.

Root owns verification of the subsequently authorized existing Mac
production route and coordinates all production changes; no production operation
has been assigned to this lane. This lane made no production calls, paid/model/GPU calls, deployment, hold release,
credential change or deletion.
