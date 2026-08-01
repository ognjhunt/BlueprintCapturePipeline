# Protocol Amendment 30: WAM5 two-GPU preflight

Status: frozen before WAM5 provider execution

Date: 2026-08-01

## Preserved pre-execution state

Amendment 29 froze allocation 9 before WAM5. No allocation-9 dry run or live
execution occurred from that freeze, its authorization was not consumed, no
provider resource was created, and no spend was incurred. Its scientific input
and provider bundle remain unchanged.

During the next read-only gate inspection, the WAM Vast capacity preflight was
found to retain a legacy global-zero requirement even though the committed
profile and launch-time guard already carried the user's prospective two-GPU
campaign ceiling. That legacy condition would unnecessarily wait when exactly
one unrelated GPU was live. This is an admission-contract defect, not a WAM4 or
WAM5 scientific failure.

## Generic correction

Immutable pushed runtime SHA
`0dc5671ee4dd275bacd468da080be2fbb7591d2d` makes the Vast WAM preflight
derive its allowed pre-existing resource count from the frozen profile:

`maximum_existing_live_resources = maximum_global_live_instances - 1`.

For the registered ceiling of two, zero or one existing Vast GPU admits one new
WAM allocation, while two existing Vast GPUs block a third. The preflight
records both the observed count and the calculated maximum. The independent
admission layer revalidates those values instead of trusting a Boolean flag.
The execute-time environment guard continues to require the exact signed value
`BLUEPRINT_VAST_MAX_GLOBAL_LIVE_INSTANCES=2`.

Ninety-five focused successor admission, paid-resource allocator, and
production campaign-budget tests pass. They include explicit zero-, one-, and
two-existing-GPU preflight cases plus independent admission revalidation. The
change is generic and does not alter a WAM action, observation, seed, history,
checkpoint, generation threshold, reliability threshold, or evaluator rule.

## Superseding allocation-9 authorization

The unconsumed
`compute_authorization_ctrl_world_allocation_9.json` and
`ctrl_world_current_reference_wam_5_gpu_profile_freeze_v1.json` are
superseded before provider mutation. The v2 authorization and profile retain:

- the exact WAM5 request SHA-256
  `5747e6dc6975c405b9c92ef2f275dbb5dab6072253327dbfbfef88653217725d`;
- the exact provider-bundle SHA-256
  `e7942b014e9acd11930236b8d7a98200ec13892b1723269c7b24197ac6c918f0`;
- one request, one allocation, and one GPU for WAM5;
- the global two-GPU ceiling;
- the USD 3 target, USD 5 allocation cap, USD 2.05/hour offer ceiling, and
  4,800-second hard TTL;
- a real cumulative campaign reservation before mutation;
- independent watchdog, teardown, and provider-zero requirements; and
- the ban on physical robot, future physical evidence, outcome, and evaluator
  access.

## Decision boundary

The next finite gate is a fresh v2 transport key, global inventory below two,
read-only Vast capacity preflight, and canonical mutation-free paid allocator
dry run from a new exact pushed SHA. Only then may one WAM5 allocation execute.
Gemini 3.6 Flash and GPT-5.6 Luna remain forbidden until the complete
12-interaction episode and causal-control matrix both pass.
