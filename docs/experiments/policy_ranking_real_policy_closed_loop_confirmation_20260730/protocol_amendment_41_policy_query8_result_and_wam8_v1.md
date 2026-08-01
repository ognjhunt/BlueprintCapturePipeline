# Protocol Amendment 41: Policy query 8 result and WAM8

Status: frozen after policy query 8 and before WAM8 provider execution

Date: 2026-08-01

## Policy-query-8 result

The eighth policy request executed from clean pushed policy runtime SHA
`bfb6fd6cc7e5fbce5d3ec4843558695f6aca20a0`. The unchanged frozen
`pi05_droid` checkpoint consumed only WAM7's three generated final camera views
and the registered commanded state. It returned a new complete native 15x8
action. The native action file SHA-256 is
`842ec763c80ec4824cb488f617613566f4f5af131403be25e918e73665bc7650`;
the deterministic action-content SHA-256 is
`b625e24bff4a66045112101192a7b4f2ac5b40e34759dd75ba291996b418491b`.

The exact output archive SHA-256 is
`21a0bf69eeaa7ce0db86b5d92d6621587e5109e06da9b017ae456d29e80fcf89`.
The output validation completed with all three expected members. The policy
identity remained `ef2133d7cde82ef08bd9d0cabc7091cab9c4d80779e544c19831c23ff9f15fb8`.
The provider allocation charged 753 conservative GPU seconds and USD
`0.156875`; the owned Vast instance was destroyed and global Vast inventory
returned to zero. The terminal watchdog SHA-256 is
`aadc58445b06cc0daa0a88914cefbfaaa9862ff36dc60d8acd45051d4a37eb66`.

The first dry-run attempt was rejected before provider mutation because its
release-record schema was obsolete. The first live attempt was rejected before
provider mutation because the absolute output path was outside the watchdog's
name scope. Both failures are preserved. The successful versioned retry used
the same immutable input and canonical allocator.

Cumulative adapter-estimated GPU usage is 39,878 seconds and USD `11.656334`.
Together with unchanged evaluator/API spend of USD `8.418512`, cumulative GPU
plus evaluator/API spend is USD `20.074846`. This is not provider
billing-export reconciliation.

## Frozen WAM8 request

The generated-only transition builder preserved query 8's complete native
action before deriving the registered WAM conditioning and next commanded
state. The transition manifest identity is
`74c330f5da4624436d159f060d76a65d200a23da868b32b0de150af5ffda6db9`.
It binds 32 frames per camera view, a 32-state history, query 8's exact native
action, and no future physical RGB, future physical state, or physical outcome.
The WAM8 request SHA-256 is
`db965f7fa874854faeddea907e1ab0e93ae0dadec6026647f70a179886fdd24a`.

The immutable provider bundle SHA-256 is
`21380cf569a4f807edba9b0b9528be50ed76c5ea47eb4a59c9f81de3ee7adfd7`;
its receipt SHA-256 is
`7a7ebd0a6c837d7dc682408071014df49d38e05e8e433484367fa0afd22a9f35`.
Allocation 12 permits exactly one generated-only, three-view Ctrl-World
current-reference request with a USD 5 allocation cap, 4,800-second TTL, and
maximum global live inventory of two GPUs. Only one GPU may be allocated by
this request.

## Decision boundary

WAM8 may execute only through Blueprint's canonical paid-resource allocator
from a clean pushed experiment SHA after fresh credential, provider inventory,
object transport, cumulative budget, watchdog, preflight, and dry-run gates.
Success requires exact input/output validation, three registered camera views,
immediate reliability, provider teardown, provider-zero proof, and campaign
settlement. It would complete interaction eight only; it would not establish
causal WAM validity, complete-episode coherence, task success, ranking fidelity,
or physical confirmation. Gemini and GPT-5.6 Luna remain forbidden until the
complete 12-interaction episode and causal-control matrix pass.
