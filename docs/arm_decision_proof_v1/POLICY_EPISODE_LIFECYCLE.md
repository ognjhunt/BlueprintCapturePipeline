# ADP-009D learned-policy episode lifecycle

This contract applies to canonical learned-policy episodes in the public-scene
Franka rehearsal. It is development-only execution evidence. It does not make
an evaluation, ranking, winner, task-success, physical-truth, or superiority
claim.

## Start boundary

`episode_started` is a scientific boundary, not a worker-process timestamp. The
worker may set it only after an outcome-blind rehearsal has retained and
digest-bound all of the following:

- canonical environment reset, joint limits, joint state, and task state;
- exact policy observation construction;
- a live, identity-bound policy-server handshake without inference;
- a conservative evidence-storage reservation;
- lossless exact-composite and calibrated external/wrist/overview media
  write/readback;
- per-camera H.264 encode/decode readback;
- one no-op environment step with joint/task readback; and
- a second canonical reset that restores the measured arm state.

The readiness rehearsal has its own episode identifier and media manifest. It
never queries the candidate and cannot be presented as scientific policy input.
Any failure in these checks is `blocked_before_episode_start` with the required
typed pre-observation media gap.

## Accepted terminal classes

After `episode_started`, a canonical result is accepted only when its receipt,
readiness receipt, lifecycle section, exact policy inputs, multicamera manifest,
review videos, candidate outputs, commands, observed states, and terminal result
all pass independent digest validation. The terminal class must be exactly one
of:

1. `planned_duration_complete`: every planned policy query, executable action
   row, and settle step completed.
2. `policy_safety_terminal`: a returned candidate action was refused by a
   predeclared shape, finite-value, joint, gripper, or action-bound validator
   before the unsafe row was applied.
3. `scientific_terminal`: a predeclared checkpoint/task scientific boundary was
   observed and retained.

A policy-safety/scientific terminal is not task success. It is a legitimate
early result only when all observations up to the boundary plus a genuine
terminal simulator observation are retained with complete media. It is never
ranking-eligible merely because it is lifecycle-valid.

Transport timeouts, server death, renderer faults, disk faults, environment
exceptions, process termination, and evidence-finalization faults are not
policy-safety or scientific results. If one occurs after the start boundary,
the worker emits `post_start_infrastructure_invariant_violation`; the run is
blocked and cannot be accepted as a completed episode. The readiness rehearsal
is designed to move every predictable instance of those failures before start,
but the contract does not make the physically impossible claim that hardware or
a process can never fail after a successful check.

## Receipt enforcement

Canonical production workers require `adp009d_policy_episode.v4`. The portable
episode evidence index independently revalidates its `policy_episode_lifecycle.v1`
and `policy_episode_prestart_readiness.v1` bindings. Older v3 receipts remain
readable historical evidence but do not satisfy this lifecycle guarantee.
