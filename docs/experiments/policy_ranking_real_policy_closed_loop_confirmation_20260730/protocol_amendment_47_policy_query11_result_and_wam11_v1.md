# Protocol Amendment 47: Policy query 11 result and WAM11

Status: frozen after policy query 11 and before WAM11 provider execution

Date: 2026-08-01

## Policy-query-11 result

The eleventh policy request executed from clean pushed policy runtime SHA
`bfb6fd6cc7e5fbce5d3ec4843558695f6aca20a0`. The unchanged frozen
`pi05_droid` checkpoint consumed only WAM10's three generated final camera
views and the registered commanded state. It returned a new complete native
15x8 action. The native action file SHA-256 is
`46aa6baf13076fa1b7f409658e8c5f5b0cddf7d5f7f8e7a5eb9f751b5943a30a`;
the deterministic action-content SHA-256 is
`384b3015162b2a66033640c147c7199b93b0e19c17f79d5a103ae33ed569c9ea`.
The deterministic request SHA-256 is
`a46c57f95f46c883d09245ab90f7ec666fd8f03473baf92bb4846443668ae9f6`.

The exact provider output archive SHA-256 is
`986b45c0ccb26f9712446e7c5f596dd2f7fe4d4bdc51d04d3482d09954f53ea0`.
Its exact three-member output allowlist, native shape, finite float64 values,
policy identity, request binding, and receipt digests passed. The policy
identity remained
`ef2133d7cde82ef08bd9d0cabc7091cab9c4d80779e544c19831c23ff9f15fb8`.

Policy query 11 charged 302 conservative GPU seconds and USD `0.062917`.
Cumulative GPU usage is 41,862 seconds and USD `12.270683`; the reservation is
settled with zero open reservations. Together with unchanged evaluator/API
spend of USD `8.418512`, cumulative GPU plus evaluator/API spend is USD
`20.689195`. The owned Vast instance `46532874` was destroyed and fresh
authenticated global inventory returned zero. This is not provider
billing-export reconciliation.

## Frozen WAM11 request

The transition freeze identity is
`55614d3230a173905949e6f2946c97316a17a4ad323426b8aed579e1124b819e`.
It binds 35 frames per camera view, a 35-state history, query 11's exact native
action, and no future physical RGB, future physical state, or physical outcome.
The WAM11 request SHA-256 is
`f5595feae52078aa428e782c01f6379875ed6b461c1c3ebd08a05b4ae3a69516`;
the registered conditioning-array SHA-256 is
`0156e2bbdd35c4a55caa31250909518011cc422eadca96faa52b954a4d0ec4d9`.
The transition-freeze receipt file SHA-256 is
`812ac112ff0988f8b2894a81b5a28917c0a98a29ab433882ad8ae83c9767df02`.

The immutable provider bundle SHA-256 is
`4340416e72643f70ee9da61525e51d4b2c6add4f0ac60743393ddb9af1560c1f`;
its receipt SHA-256 is
`28448a4f857a009caea415ab61bb55ea8727189e2e4deb6ef6c00c6afaad8eec`.
Allocation 15 permits exactly one generated-only, three-view Ctrl-World
current-reference request with a USD 5 allocation cap, 4,800-second TTL, and
maximum global live inventory of two GPUs. Only one GPU may be allocated by
this request.

## Decision boundary

WAM11 may execute only through Blueprint's canonical paid-resource allocator
from a clean pushed experiment SHA after fresh credential, provider inventory,
object transport, cumulative budget, watchdog, preflight, and dry-run gates.
Success requires exact input/output validation, three registered camera views,
immediate reliability, provider teardown, provider-zero proof, and campaign
settlement. It would complete interaction eleven only; it would not establish
causal WAM validity, complete-episode coherence, task success, ranking fidelity,
or physical confirmation. Gemini and GPT-5.6 Luna remain forbidden until the
complete 12-interaction episode and causal-control matrix pass.
