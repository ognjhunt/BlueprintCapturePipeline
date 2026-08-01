# Protocol Amendment 37: two-GPU watchdog terminal reconciliation

Status: frozen after WAM6 owned-instance teardown and before control-plane reconciliation

Date: 2026-08-01

## Preserved state

WAM6's canonical allocator invocation created exactly one owned Vast instance,
`46519017`. The provider runtime completed its bundle, downloaded the output
archive, and destroyed that exact instance. Two repeated exact-id inspections
and the exact WAM name-prefix inventory can therefore establish absence of the
owned resource. Scientific output validation remains a separate subsequent
gate.

One unrelated NVIDIA Warehouse Vast instance remained live under the user's
prospectively amended two-GPU campaign ceiling. The independent watchdog's
owner-cancel path retained the historical global-zero rule, so it remained
armed after the owned WAM6 resource was absent. The cumulative campaign
reservation consequently remained open. No additional WAM request, evaluator
call, provider create, or physical evidence resulted from this control-plane
state.

## Generic correction

The watchdog now preserves the historical global-zero requirement by default.
For Vast only, when the already registered
`BLUEPRINT_VAST_MAX_GLOBAL_LIVE_INSTANCES` value is exactly `2`, owner-cancel
settlement may admit at most one residual unrelated live instance. Values that
are absent, malformed, zero, or greater than two fail closed to the historical
one-GPU/global-zero behavior.

Early terminal reconciliation still requires all of the following twice:

- the exact owned name-prefix inventory is API-confirmed at zero;
- the recorded owned instance ID is API-confirmed absent;
- the global inventory is API-confirmed at or below the authorized residual
  allowance; and
- no provider mutation is performed by the reconciliation path.

The result records the maximum global count, residual unrelated allowance, and
both global inventories. It may close only the matching owned pending teardown,
release only the transferred WAM paid lane, and settle only the matching
cumulative reservation. It cannot terminate, rename, or otherwise mutate an
unrelated resource.

The watchdog arm step also gains retained-session support: an identical armed
receipt may be resumed without rewriting its original bytes only when its prior
PID is no longer live. A live prior owner refuses the resume.

Twenty-five focused watchdog tests and 126 focused paid-lane, spend-guard, and
campaign-budget tests pass. The new tests prove one unrelated Vast resource is
admitted only under the exact two-GPU ceiling and that an identical dead-owner
watchdog resumes without rewriting its armed receipt.

## Recovery boundary

After this correction is committed and pushed, the obsolete WAM6 watchdog
process may be terminated locally because its exact owned instance is already
provider-absent and its owner-cancel request is present. The same watchdog
identity, prefix, provider, output directory, and original deadline must then
resume from the immutable corrected source with the two-GPU environment
binding. Success requires a terminal watchdog receipt, matching pending
teardown closure, paid-lane release, cumulative-budget settlement, and no
mutation of the unrelated Warehouse resource.

This amendment changes no policy input, WAM input, generated frame, reliability
threshold, causal threshold, judge gate, or scientific claim.
