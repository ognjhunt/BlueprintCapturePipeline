# Protocol Amendment 34: Policy query 6 two-GPU lane reconciliation

Status: frozen after retry-1's zero-mutation lane block and before any retry-2
provider mutation

Date: 2026-08-01

## Preserved retry-1 block

The retry-1 input, source, object transport, fresh Vast preflight, and canonical
dry run passed. At live admission, one unrelated writer-owned NVIDIA Warehouse
GPU was registered on Vast. Provider admission correctly accepted one existing
GPU under the prospectively frozen maximum of two, but the paid-provider lane
reconciliation still rejected every open provider teardown record as
`paid_provider_lane_already_owned`.

No policy-query provider allocation was created, the retry-1 compute
authorization was not consumed, its campaign reservation settled at zero, and
its watchdog exited without a provider mutation. Preserved evidence includes:

- adapter SHA-256
  `f46a35b037a4a26677024f3e82ef7c2d926fe66fdfcf535f739d888594806065`;
- settled zero-cost ledger SHA-256
  `052cbd5ee2f5b70ffaad26ffd6ed3a0d969ffa55ab33efc49c8ed73ad892b587`;
  and
- launch-refresh preflight SHA-256
  `212a0975e875eddbe76e2095003b487e075e0f21fb1bcbd1e6f20263b6247d2e`.

This was a control-plane contract mismatch, not a policy, WAM, checkpoint,
observation, action, or provider-runtime failure. Cumulative spend remains
37,536 GPU seconds and USD `10.983857`.

## Generic reconciliation fix

Runtime SHA `8b2191c1a5a375ad62ffe98b7461c3ae04eb7dae` carries the
already-frozen GPU concurrency ceiling into atomic paid-lane initial
reconciliation. Existing resources are deduplicated across provider inventory
and pending-teardown records. Initial acquisition may proceed only when the
observed existing count is at or below the registered allowance. A live owner
of the same lane still blocks a second mutation, and stale-lease reclamation
still requires zero resources and zero open pending teardowns.

The focused provider, admission, authorization, campaign, and lease regression
set passes 184 tests. The official source archive SHA-256 is
`63f3572fd71a7e23fc6f03f22844e0ea40c6212bfa98b2bb9b41e6f31a6c6ffd`;
its audit file SHA-256 is
`defebec5445a848b471ce82f01c6dd267b5335aa6464354f143633360e4520d5`.

## Retry-2 binding

The retry-1 authorization is prospectively superseded without consumption.
Retry 2 retains the exact WAM5 images, registered commanded state,
`pi05_droid` checkpoint, task text, query index, one-allocation limit, USD 3
allocation cap, two-GPU campaign maximum, and Vast machine `27268` exclusion.
Only the generic runtime-source binding changes.

The retry-2 input archive SHA-256 is
`415a38b11376af9ee3de6e1f634063635509636b6738fe5d51f44af21c3d40fc`;
its receipt file SHA-256 is
`e7c5de1c9428a527f10283062c684c3edb306364fb721055499d10ddc5782b20`,
and its independent extraction receipt SHA-256 is
`058240857921721991c3a7c6ae4c3843828438a7bcb7e1c2dc64940a020a37b6`.

All object keys, provider names, and live evidence paths must be new. Signed
transport, output absence, provider preflight, budget reservation, watchdog,
canonical dry run, teardown, settlement, and provider-zero controls repeat.
No evaluator or VLM call is authorized. Success would close only interaction
six; all later scientific gates remain unproven.
