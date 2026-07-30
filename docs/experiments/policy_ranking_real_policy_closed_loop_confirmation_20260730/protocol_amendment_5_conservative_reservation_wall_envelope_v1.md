# Protocol Amendment 5: Conservative Reservation Wall Envelope

Frozen prospectively: 2026-07-30T11:43:02-0500

## Observed operational condition

The first Vast policy-identity launch attempt reserved 14,400 GPU seconds and
USD 3 before provider mutation. Vast rejected the selected stale offer with HTTP
410 and returned no instance identifier. Repeated exact-prefix and global
inventory queries have shown zero live resources and attributable provider spend
of USD 0, but the older watchdog remains the authoritative lane owner until its
hard deadline.

Its stored reservation epoch is 1785426539.944277 and its hard deadline is
1785440939.945568, 14,400.001291 seconds later. The older watchdog rounds
elapsed time upward, so it is expected to preserve a one-second reservation
breach instead of automatically settling. That fail-closed behavior must not be
bypassed while the watchdog is live.

After the watchdog exits, reconciliation is admitted only if all of these are
true:

- exact-prefix and global Vast inventory are both zero;
- no exact Vast instance identifier was ever recorded;
- the pending teardown is terminal;
- the paid-resource lease is released;
- the old reservation remains open solely because of the deadline-rounding
  edge.

If those conditions hold, the old reservation is settled conservatively at its
full frozen maximum of 14,400 GPU seconds and USD 3. This is deliberately higher
than actual attributable provider use, which remains zero.
The settlement must use the generic fail-closed command:

    python -m blueprint_pipeline.no_allocation_budget_reconciliation \\
      reconcile-no-allocation-watchdog ...

The command requires a provider-zero snapshot taken after the terminal
watchdog evidence, an unrecorded Vast instance ID, terminal pending teardown,
released paid lane, an exact one-second rounding breach, and the matching open
reservation. It writes a new no-overwrite reconciliation receipt.

## Prospective accounting envelope

The repository's internal GPU wall-time envelope increases from 36,000 to
72,000 seconds. The change is an accounting envelope only. It does not change:

- the USD 20 internal GPU campaign cap;
- the user's USD 50 GPU ceiling;
- the user's USD 100 total campaign ceiling;
- the USD 3 policy-canary stage cap;
- the 14,400-second policy-canary TTL;
- one-GPU concurrency;
- watchdog, teardown, provider-zero, or immutable-source requirements;
- any causal, reliability, ranking, abstention, or scientific threshold.

The larger wall envelope is needed because conservative accounting produces
22,845 prior GPU seconds before a fresh 14,400-second finite retry, for 37,245
committed seconds. The fresh retry still requires a new ledger, new immutable
run identity, fresh admission, and a clean pushed source SHA.

## Claim boundary

This amendment is frozen before any real-policy output is observed. It is an
operational budget-accounting repair, not policy execution, WAM execution,
causal qualification, ranking evidence, physical evidence, or a scientific
threshold change.
