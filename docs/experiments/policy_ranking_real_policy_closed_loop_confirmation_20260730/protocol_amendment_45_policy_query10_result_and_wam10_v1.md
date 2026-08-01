# Protocol Amendment 45: Policy query 10 result and WAM10

Status: frozen after policy query 10 and before WAM10 provider execution

Date: 2026-08-01

## Policy-query-10 result

The tenth policy request executed from clean pushed policy runtime SHA
`bfb6fd6cc7e5fbce5d3ec4843558695f6aca20a0`. The unchanged frozen
`pi05_droid` checkpoint consumed only WAM9's three generated final camera views
and the registered commanded state. It returned a new complete native 15x8
action. The native action file SHA-256 is
`beaf0a94b83eb4332acce6c57f99c955e70824c79d47718291a6bdf075d00564`;
the deterministic action-content SHA-256 is
`dd43ccc433018d0e28c4ba1fb9150ea09c80b7b6c95ec5ae515aa8b2c6ee83ef`.
The deterministic request SHA-256 is
`63bf30c9df6e488b7b7f375fe55daf5c396674348f321c439bb8f7be16361425`.

The exact provider output archive SHA-256 is
`ca530782ba051e4db98b9b781ca8fbfd7f7e9dfe39ceff211110c5ea47434083`.
Its exact three-member output allowlist, native shape, finite float64 values,
policy identity, request binding, and receipt digests passed. The policy
identity remained
`ef2133d7cde82ef08bd9d0cabc7091cab9c4d80779e544c19831c23ff9f15fb8`.

Policy query 10 charged 292 conservative GPU seconds and USD `0.060833`.
Cumulative GPU usage is 41,169 seconds and USD `12.098989`; the reservation is
settled with zero open reservations. Together with unchanged evaluator/API
spend of USD `8.418512`, cumulative GPU plus evaluator/API spend is USD
`20.517501`. The owned Vast instance `46530810` was destroyed and fresh
authenticated global inventory returned zero. This is not provider
billing-export reconciliation.

## Frozen WAM10 request

The transition freeze identity is
`a126784b1dcc7f3e5189ec7687716d0be47831ae856b8a145dc4019a9c01733d`.
It binds 34 frames per camera view, a 34-state history, query 10's exact native
action, and no future physical RGB, future physical state, or physical outcome.
The WAM10 request SHA-256 is
`c5bb8e1b1f290e37a85af65ac5bbd739d1c5ef2d8c05ec74488f1b7f251e6ea4`;
the registered conditioning-array SHA-256 is
`b52e838b2ad4f2cf55cbb18ffef19ccf82cda11d30fd59ff0e2b308aea722451`.
The transition-freeze receipt file SHA-256 is
`cd7c5f018ddcbf3eac3ecce78e9f5bf3fe04575caedefa57158070fa6d2c9bf8`.

The immutable provider bundle SHA-256 is
`437324e8f23f498c381123c0fa68fd87956a24778625a19c441925fdb1eb0f7d`;
its receipt SHA-256 is
`cff64f7f1245d5865c6fd63de3a96602c373a0131a18eac95b930f39948b824b`.
Allocation 14 permits exactly one generated-only, three-view Ctrl-World
current-reference request with a USD 5 allocation cap, 4,800-second TTL, and
maximum global live inventory of two GPUs. Only one GPU may be allocated by
this request.

## Decision boundary

WAM10 may execute only through Blueprint's canonical paid-resource allocator
from a clean pushed experiment SHA after fresh credential, provider inventory,
object transport, cumulative budget, watchdog, preflight, and dry-run gates.
Success requires exact input/output validation, three registered camera views,
immediate reliability, provider teardown, provider-zero proof, and campaign
settlement. It would complete interaction ten only; it would not establish
causal WAM validity, complete-episode coherence, task success, ranking fidelity,
or physical confirmation. Gemini and GPT-5.6 Luna remain forbidden until the
complete 12-interaction episode and causal-control matrix pass.
