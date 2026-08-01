# Protocol Amendment 31: WAM5 adapter two-GPU retry

Status: frozen after zero-mutation allocation-9-v2 block and before retry

Date: 2026-08-01

## Preserved allocation-9-v2 result

The allocation-9-v2 canonical dry run passed from immutable pushed SHA
`f7b170b432de2a260f0cea2f2de3feb441995a83`. A fresh global inventory
reported one live Vast GPU under the two-GPU campaign ceiling. The WAM
capacity preflight independently admitted that state with
`existing_live_resource_count = 1` and
`maximum_existing_live_resources = 1`.

The subsequent live command opened and then settled a real cumulative campaign
reservation, but a deeper generic Vast adapter guard retained a legacy
zero-existing-instance rule. It blocked before offer search or provider
creation with `active_vast_instances_detected_before_new_launch`.

The failure had:

- zero provider creates and zero owned instance IDs;
- zero authorization consumption;
- zero charged GPU-seconds and USD `0.000000`;
- a settled cumulative reservation with no open reservation;
- an independently armed watchdog cancelled with no allocation; and
- no WAM output, physical evidence, outcome, policy, or evaluator access.

Preserved evidence includes:

- adapter output SHA-256
  `3d1544ac1a75943ec15e4b1dc799f1e4cfacbc7592c4d1ee66da5e0fa4240ae3`;
- adapter prelaunch inventory guard SHA-256
  `ee6c0fa0d591b99dac45996f18cd13317efafbc241b2648dd0c0030fd949425b`;
- pre-mutation reservation receipt SHA-256
  `2ec46b89f41aa3490b3b7c9be5ca4732eddc56ed4f702281906e17a9f1c2ec02`;
- settlement receipt SHA-256
  `7734b82c091d8ae8a42d9029a72e84ce555996181a1b3c870827be7e05f3a74f`;
- settled production ledger SHA-256
  `25cc296f5cc45180dd379f60606949ebb8c3a10a00ea9694f771d5d37e34b048`;
  and
- watchdog handoff SHA-256
  `97621e94d9822a02a97f345fc94a722521bccfe6d938cce5093a2c7c97c9624b`.

This is a provider-admission orchestration failure, not a WAM4 or WAM5
scientific result.

## Generic adapter correction

Immutable pushed runtime SHA
`288e2bf988cdb32c8bc30011018eac78aa3d8110` passes the frozen
`BLUEPRINT_VAST_MAX_GLOBAL_LIVE_INSTANCES` value into the provider adapter's
prelaunch inventory guard. The guard now computes
`maximum_existing_live_instances = maximum_global_live_instances - 1`,
records the exact observed and allowed counts, and:

- preserves zero-existing behavior when no positive concurrency ceiling is
  supplied;
- admits one existing GPU when the frozen ceiling is two; and
- blocks at two existing GPUs before a third launch.

The same execute-time repeated inventory probe continues to abort if total live
Vast instances exceed the frozen ceiling after allocation.

One hundred eighteen focused tests pass: five exact adapter guard/caller tests
and 113 WAM runner, successor admission, paid allocator, and cumulative-budget
tests. The caller regression proves one existing instance reaches offer search
without creating a provider resource; the direct guard tests prove two existing
instances block.

## Superseding allocation-9-v2 authorization

The unconsumed
`compute_authorization_ctrl_world_allocation_9_v2.json` and
`ctrl_world_current_reference_wam_5_gpu_profile_freeze_v2.json` are
superseded before provider mutation. The v3 records retain the exact WAM5
request, seed, histories, provider bundle, checkpoint, one-allocation limit,
two-GPU global ceiling, cost limits, hard TTL, cumulative reservation,
watchdog, teardown, provider-zero, and evidence-access prohibitions.

## Decision boundary

The next finite gate is a fresh v3 transport key, inventory below two, fresh
read-only preflight, and canonical dry run from a new exact pushed SHA. Only
then may the scientifically identical WAM5 request retry once. Gemini 3.6 Flash
and GPT-5.6 Luna remain forbidden until the complete 12-interaction episode and
causal-control matrix both pass.
