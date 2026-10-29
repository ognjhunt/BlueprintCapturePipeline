# Continue original website preparation after retained Atlas recovery

ADP-040 / Day 28 dependency and free-beta journey blocker: a retained original
website upload must continue from its stopped native preparation after its own
Atlas reconstruction finishes. The completion artifact is an incremented original
native attempt history, adopted completed stages and an actual scored result on
the same private job. This change is development-only infrastructure; it does not
qualify geometry, physics, physical performance or a commercial proposal.

## Observed gap

The original c334 upload stopped at `clean_plate` target identification. Its
native ledger retains five attempts and no `reconstruction_resume` record. The
existing automatic continuation correctly arms only explicit Atlas-processing
waits after explicit prior waits. Turning an earlier model failure into such a
wait would rewrite its history and hide unknown outcomes.

The native ledger's original payload digest is `bedca616...`; its producer
delivery key is `sha256:492acb63...`. A retained broker-failure record matches that
payload and records delivery attempt five as `retry_exhausted_pending_pubsub_dlq`.
It is historical delivery evidence, not a current broker inspection. No cached
retry-deferral leases remained to wake on deployment. The generation-pinned
original producer envelope selected for Atlas recovery has different serialized
bytes. Its current original source and producer identity must be revalidated;
neither payload digest may be substituted for the other.

## Small reversible change

Extend the existing selected-handoff operator surface with
`--resume-preparation --atlas-status-digest <reviewed digest>`. Its existing
read-only Atlas status now includes the entire native ledger's digest plus its
state, revision and attempt count. The same snapshot retains current source,
owner/rights, task context, original grant, expiry, reservations, settled usage
and historical inference-cost unknowns.

The explicit resume requires the original source and its own ready Atlas job,
then invokes the existing native processor. Under its ordinary claim lock the
processor requires the exact reviewed failed ledger and no active or uncertain
lease. It increments the existing attempt count, preserves failure history,
adopts completed stages and performs normal current sponsorship/spend admission.
Source conflicts, withdrawals, retirement, changed snapshots and ended authority
retain their refusal. There is no broker republish, counter reset, new grant,
replacement job, fabricated wait record, timer or customer approval step.

This uses the existing reconstruction-continuation claim interface with a typed
operator review of the ledger and exact selected envelope. The native handler
and its signature are unchanged; no additional handler argument is needed.

Because Atlas is already finished, its unused request quota and price-review
date confer no authority on native work and cannot prevent that work. Native
spending remains separately admitted by the existing handler and allocator.
Global spending stops still apply.

The root runner uses the same service user, runtime-environment guard and write
namespace as existing native dispatch. Read-only status stays read-only. The
installed operator door needs its ordinary source-verified upgrade before this
new mode can be used. Existing automatic Atlas waits are unchanged.

## Verification and remaining observation

Focused hermetic coverage exercises the real native processor and lease claim,
retained attempt history, completed-stage reuse, one-shot reviewed recovery,
changed source/history/owner refusal, active/uncertain lease refusal, global
spending holds, independent Atlas/native limits, and operator client/root-runner
request and sandbox contracts. It does not make a live provider call.

The final eight focused recovery/operator suites passed 474 cases in 27.16
seconds. The paid-resource allocator bypass-verifier suite passed 27 cases in
60.96 seconds. The existing impact selector maps every changed code surface and
does not require a repository-wide suite for this final change.
Changed-file Ruff, whitespace and shell syntax checks passed. An initial run
caught malformed-mode validation using an unhashable input; it was repaired and
replayed. The final run reused the primary Python environment's installed
Pub/Sub SDK without installing or copying dependencies.

Before real execution, publish and deploy the exact change, upgrade the door,
observe the original Atlas job ready, and review a fresh current snapshot. A
green test, deployed identity, reconstruction preview or native resume receipt
alone does not prove a scored result or customer completion.
