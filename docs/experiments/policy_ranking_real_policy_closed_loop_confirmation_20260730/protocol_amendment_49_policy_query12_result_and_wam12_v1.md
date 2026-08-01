# Protocol Amendment 49: Policy query 12 result and WAM12

Status: frozen after policy query 12 and before WAM12 provider execution

Date: 2026-08-01

## Policy-query-12 result

The twelfth policy request executed from clean pushed policy runtime SHA
`bfb6fd6cc7e5fbce5d3ec4843558695f6aca20a0`. The unchanged frozen
`pi05_droid` checkpoint consumed only WAM11's three generated final camera
views and the registered commanded state. It returned a new complete native
15x8 action. The native action file SHA-256 is
`6bf6fc255c442bce8cc300f68f14021af1d236bcd9ea71f3f5528f3124c25e67`;
the deterministic action-content SHA-256 is
`a64f6e55053ac63f910e6c21ca07adcbb12815f46f2c8bdd246739845e58aa09`.
The deterministic request SHA-256 is
`1897ff2ccf16ef332b9356f623bdac4d92197d9e88dda6e6e85f2204d703956f`.

The exact provider output archive SHA-256 is
`2f129221aa552fb238a44aba893c74bd3d43957a5f6eba23f4a4580777b910b0`.
Its exact three-member output allowlist, native shape, finite float64 values,
policy identity, request binding, and receipt digests passed. The policy
identity remained
`ef2133d7cde82ef08bd9d0cabc7091cab9c4d80779e544c19831c23ff9f15fb8`.

Policy query 12's scientific output completed, but its first control-plane
settlement failed closed because a concurrently owned Warehouse GPU remained
live. The OpenPI-owned Vast instance `46534226` was absent. Runtime fix SHA
`3bec4b6ac` passed 53 focused lease and watchdog tests and restored lane-scoped
reconciliation while preserving legacy and supervised-child fail-closed
ownership. The recovery performed no provider mutation, closed only query 12's
teardown record, released only its transferred lease, and settled 251 GPU
seconds and USD `0.052292`. The reconciliation receipt SHA-256 is
`d69450ee70938d68913748fcd8fb63dc35581485419b25cfa0fa936f6b89d4fe`.

Cumulative GPU usage is 42,360 seconds and USD `12.441698`; the reservation is
settled with zero open reservations. Together with unchanged evaluator/API
spend of USD `8.418512`, cumulative GPU plus evaluator/API spend is USD
`20.860210`. Two fresh authenticated global inventories returned zero during
reconciliation. This is not provider billing-export reconciliation.

## Frozen WAM12 request

The transition freeze identity is
`a0e1b6cc69859a03329fc5454b48b451c70cd2307a4eecae9d99c7c0bb8ae9c6`.
It binds 36 frames per camera view, a 36-state history, query 12's exact native
action, and no future physical RGB, future physical state, or physical outcome.
The WAM12 request SHA-256 is
`0dc4bbe246a1f30f5f332651ad2feb0ad8fcb27d8d66f1dd6e12ca20863e3229`;
the registered conditioning-array SHA-256 is
`724fed34ac80bc2bea2209860ec66cd08221d82c5919e0bfe0d5dc5545625aeb`.
The transition-freeze receipt file SHA-256 is
`aaa4828579ae88483d4cb687969907dc25c5391b7098854a1970050e921dbd43`.

The first local v1 transition-build attempt stopped before request staging
because the frozen OpenPI client source was not on the local import path. It
performed no provider call and its partial directory is preserved. The v2
builder explicitly bound the same frozen OpenPI client source and produced the
successful request and bundle without changing scientific inputs.

The immutable provider bundle SHA-256 is
`f5e890e4c754eddf8b26865cb6c95f71a0000502d585026be2de9913ca795fb2`;
its receipt SHA-256 is
`683f5b0b5ec0c99f82230b2da0ff1eaf2b37835950ed337c84e0de83e8b71608`.
Allocation 16 permits exactly one generated-only, three-view Ctrl-World
current-reference request with a USD 5 allocation cap, 4,800-second TTL, and
maximum global live inventory of two GPUs. Only one GPU may be allocated by
this request.

## Decision boundary

WAM12 may execute only through Blueprint's canonical paid-resource allocator
from a clean pushed experiment SHA after fresh credential, provider inventory,
object transport, cumulative budget, watchdog, preflight, and dry-run gates.
Success requires exact input/output validation, three registered camera views,
immediate reliability, provider teardown, provider-zero proof, and campaign
settlement. It would complete the registered 12-interaction label-free episode;
it would not establish causal WAM validity, task success, ranking fidelity, or
physical confirmation. Gemini and GPT-5.6 Luna remain forbidden until WAM12
passes and the complete registered causal-control matrix passes.
