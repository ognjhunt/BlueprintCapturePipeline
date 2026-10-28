# Preparation delivery and overdue recovery

This implements the beta preparation-to-current-website-state corrections in
AUT-03 and the immediate status correction in AUT-04. It supersedes the
silence-as-terminal interpretation in
`website-preparation-terminal-status-2026-10-18.md`.

## Delivery selection

The canonical per-capture `website_preparation_status_delivery.json` remains
the durable delivery record. Retaining a pending revision also writes a private
hint under the handoff root's `.website_preparation_pending/` directory.
Hints carry no source, grant, or acknowledgement authority. Every callback
reopens the canonical record and current source, and validates the exact status
digest after the callback before accepting its receipt.

Due callbacks have their own dispatch budget. A separate bounded round-robin
ledger sweep repairs missing hints and the crash boundary between a ledger
commit and retained delivery. A failed callback backs off from 60 seconds to
30 minutes. New revisions become due immediately; an old receipt cannot
acknowledge them. A corrupt cursor restarts only historical repair. Repair
after a complete lost index is incremental and is not covered by the two-tick
healthy-index latency target.

The worker loads canonical pending delivery before refreshing current source,
so source-lookup failure receives the same bounded backoff as transport failure.
Eligible work is ordered by its actual eligibility time (the later of pending
creation and retry deadline). An immediate-zero deadline does not outrank every
already-due retry. This preserves aging without skipping retry backoff.

If the canonical delivery record is missing or cannot be read, its bytes remain untouched and
the selected existing hint receives a durable 60-second-to-30-minute scheduling
backoff. This is an unavailable outcome, never an acknowledgement or terminal
failure. The historical repair sweep preserves that deferral for the same
revision; a new revision resets it. A compare-and-swap against the selected hint
prevents a late failure from delaying a newer revision. Restart does not reset
the backoff, so one corrupt record cannot monopolize a one-item callback budget.

The existing listener writes `.website_preparation_delivery_metrics.json` with
due backlog/age, measured successful delivery lag, actual attempts and failures,
repair work and explicit overload or hint-scan truncation. The hint scan is
capped at 20,000 pending entries. Crossing that cap is overload requiring the
existing operator workflow; it is not proof of healthy delivery latency. The
large-history test verifies one newly retained callback among 10,000 settled
ledgers reaches transport on the first healthy tick with a callback budget of
one. The fairness rehearsal exercises 100 due callbacks, failures, and a new
revision racing an old acknowledgement.

## Status truth

Two hours without another claim is an overdue observation, not proof that
Pub/Sub or retained recovery has no authorized retry. The private
`website_preparation_overdue.v1` record binds the current attempt, source digest,
lease token and status. It records a bounded cause, observed age and recovery
next action without raw private error details. Its attempt-history entry
survives a later valid claim.

Existing public v1/v2 states remain compatible. Capacity stays
`capacity_delayed`; retryable work remains retryable; an expired, overdue worker
is reported as `failed_retryable` in v2 rather than continuing to look active.
`terminal_authority_ended` retains its existing authority-ending interpretation.
Historical silence-only terminal annotations remain retained but no longer
produce `failed_terminal`.

Conclusive retry exhaustion remains unproven until AUT-02 provides a validated,
source-and-attempt-bound durable recovery disposition. A plausible reason,
broker delivery count, arbitrary digest or global listener heartbeat does not
substitute for that disposition. The joined test bridge's `overdue` command
exercises silence; its old `exhaust` alias now has that same truthful meaning.
WebApp can continue testing explicit valid terminal fixtures independently.

## Verification and rollout boundary

Focused signed-status tests cover v1/v2 negotiation, current-source refusal,
stale acknowledgement races, late retries, outage/recovery, and preservation of
private error boundaries. Large-history and concurrent-delivery rehearsals use
only owned synthetic local records and fake transport.

The existing production release owner must observe shadow index consistency,
bounded catch-up and a natural callback outage/recovery receipt before claiming
production latency acceptance. These local tests do not establish deployment,
current customer rendering, or uninterrupted production completion.
