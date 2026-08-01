# Protocol Amendment 28: policy-query-5 two-GPU admission retry

Status: frozen before the policy-query-5 retry provider execution

Date: 2026-08-01

## Preserved admission block

The first live invocation bound to Amendment 27 stopped before any provider
mutation because the launch-time Vast refresh observed one unrelated live GPU
and the OpenPI preflight still required global Vast inventory zero. The frozen
preflight blockers were `openpi_gpu_preflight_not_verified` and
`openpi_gpu_provider_inventory_not_zero`. No cumulative reservation, watchdog,
provider allocation, policy request, query-5 spend, or authorization consumption
occurred.

The independent all-provider snapshot then identified one owned Vast instance,
`46507241`, named `blueprint-native-warehouse`, at USD `0.4852` per hour. The
global fleet gate passed at one live instance under the prospectively authorized
two-GPU ceiling. This experiment did not modify or close that resource.

## Generic concurrency correction

Immutable pushed runtime SHA `ede38013d6cb2a5453ed39ba39c607a7f497a639`
binds current-reference execution to the authorization's
`maximum_concurrent_gpus`. Both Vast and RunPod launch-time preflights preserve
zero existing resources as the default, but an authorization with a two-GPU
ceiling may admit at most one pre-existing provider resource. Two pre-existing
resources block before mutation, because the requested allocation would bring
the total to three. The authorization validator accepts only the registered
historical values one or two.

The focused OpenPI admission, runtime, paid-allocator, and production-budget
gate passes with 81 tests; lint, formatting, and diff integrity pass. The branch
is clean, pushed, and has `0 0` upstream divergence.

The runtime source archive SHA-256 is
`8da87e0a36c582eeb869d2e4dbeadae61353c5bab9e630aaf28f38e318f6e280`;
its audit receipt SHA-256 is
`58d837e3f7f873e1ce6c7aec965eaf3622e929b7f764deaead8a729107b1916f`.
The replacement query-5 input archive SHA-256 is
`7ab0f3c26edc8cc9661b87fbd817013f48353c291efd2ab965a9f3f55f983e15`;
its input-receipt file SHA-256 is
`d38fd9a67f3b8cf3152a08194766f4dec799457320943b2c1fd82dfbae6ed40f`;
and its independent safe-extraction receipt SHA-256 is
`3c9169cad4bac9458ef5ab5d172dd682d0d19c29ec7529f86cc24a5a106cadba`.

`compute_authorization_openpi_policy_query_5_v2.json` and its input remain
preserved but are superseded before provider mutation. The scientific request,
WAM4 observation, commanded state, policy identity, query index, physical-data
exclusions, stage cap, cumulative budget, watchdog, TTL, and teardown rules are
unchanged.

## Retry gate

The retry requires a new immutable object key, fresh all-provider inventory at
no more than one existing GPU, fresh provider preflight, canonical allocator dry
run, a real cumulative reservation, atomic consumption of the v3 authorization,
watchdog-before-allocation, and teardown/provider-zero evidence for the owned
query allocation. WAM5 remains unauthorized and judges remain forbidden.
