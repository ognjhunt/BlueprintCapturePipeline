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
or adoption before another grant. No provider-zero snapshot is used to establish
a no-create proof. Recovery of genuine lost provider responses, output transfer
and teardown still needs provider-specific evidence and acceptance observation;
these local tests do not establish those outcomes.

## AUT-10

Implemented and tested locally: an independent allocator-process monitor reuses
the signed current-source reader, persists exact-attempt stop intent, refuses
create after a changed/revoked source, and applies the existing 60-second polling
and 600-second monitor-unavailable grace. Signaling requires the invocation's
retained PID/start ticks and manifest identity, and lets canonical finally and
watchdogs perform teardown. Observed deadlines can only shorten a session and
are forwarded to the existing provider hard-deadline cap. Partial revoked results
remain historical and cannot become a successful current screening result.

**Still incomplete:** the current beta task-context transport does not supply the
team route's run-specific cancellation record or a scoped GPU per-cell control
lease. No global sync credential is forwarded to a provider to fill that gap.
Actual authority-expiry fields must be supplied by the applicable authenticated
authority contract; absence of such a field is not proof of indefinite authority.
Full AUT-10 acceptance remains open for those integrations and genuine teardown
observation. This document makes no claim that no GPU cell can begin during the
provider's stop/teardown propagation interval.

## Other lane1 work and production access

AUT-02 retained DLQ recovery and AUT-09 site-local unspent plan refresh are not
implemented by these checkpoints. They must not be marked complete from these
changes. Production access remains blocked by the earlier CONNECT 403 to
paperclip.tryblueprint.io:443 until an authorized network correction is verified.
This lane made no production calls, paid/model/GPU calls, deployment, hold release,
credential change or deletion.
