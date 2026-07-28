# Cosmos3 successor follow-up — 2026-07-28

Status: `allocation_1_infrastructure_failure_provider_zero_retry_2_authorized`

This is a new experiment namespace. The closed
`policy_ranking_successor_experiment_20260727` namespace and its exact
`inconclusive` verdict are immutable historical evidence.

The primary arm keeps Blueprint's OSCAR/SC3-derived action, counterfactual,
temporal-consistency, calibration, aggregation, uncertainty, and abstention
harness while replacing only the primary world-action model with the pinned
general `nvidia/Cosmos3-Nano` forward-dynamics backend. `oscar_wam` remains a
compatibility and historical-baseline backend. Cosmos and OSCAR are not
chained serially in the primary arm.

The untouched primary input is a truthful DROID three-camera observation and
the frozen raw 16x10 action trajectory. The OSCAR skeleton renderer is retained
as intended-motion evidence and as a separately labeled baseline/ablation. A
future frozen Cosmos-plus-skeleton arm may be run only after the untouched
baseline closes; it cannot receive Cosmos-only credit and must be scored with
the skeleton masked. No fine-tuning is admitted before the untouched baseline.

If the powered causal gate later admits evaluator work, the independent judge
uses an attributed clean implementation of RoboWorld's disclosed task-progress
rubric: fixed external views decide progress and success, the wrist view flags
world-model artifacts only, and adjacent frames must confirm stable success.
SC3-derived controls, consistency checks, uncertainty, early termination, and
abstention remain a separate reliability layer. RoboWorld's reported
`r=0.989` and SC3-Eval's reported `r=0.929` come from different experiments and
are historical context, not transferable Blueprint evidence.

The current ten-rollout, one-session design is a screening/falsification arm,
not confirmatory evidence. Frames, cameras, controls, and seeds are not treated
as independent sessions. A one-session result is `inconclusive` for general
Cosmos WAM qualification even if all screening gates pass or fail.

The user's goal-scoped cost authorization now permits a powered continuation.
After the frozen one-session structural/scientific screen, a confirmatory arm
requires at least 17 independently selected DROID sessions and a separately
frozen input manifest. Each allocation remains bounded by an explicit spend
limit and hard TTL; only the earlier cumulative `$6` ceiling was superseded.

No evaluator call, benchmark-label unseal, captured-site generation, or paid
provider mutation is admitted until the causal WAM gate and its declared power
requirements pass.

## Allocation history

Allocation 1 consumed USD 0.178545 over 632.915481 seconds, never loaded the
Cosmos server, and produced zero scientific rollouts. The provider was still
pulling the sealed image when the generic 600-second no-log heartbeat expired,
despite this lane admitting an 18-minute cold pull. Teardown, independent
watchdog closure, and a fresh provider API inventory of zero live instances
were verified. This is an infrastructure incident with no scientific
interpretation.

`compute_authorization_allocation_2.json` is the active single-use retry
authorization. It remains under the same per-allocation USD 6 hard cap,
one-resource limit, 180-minute TTL, watchdog, teardown, and provider-zero
requirements. `runtime_retry_signature.json` binds the incident record, retry
authorization, and timeout repair without altering the frozen scientific
inputs or evaluator methodology.
