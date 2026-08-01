# Protocol Amendment 39: Policy query 7 result and WAM7

Status: frozen after policy query 7 and before WAM7 provider execution

Date: 2026-08-01

## Policy-query-7 result

The seventh policy request executed from the clean pushed policy runtime SHA
`bfb6fd6cc7e5fbce5d3ec4843558695f6aca20a0`. The unchanged frozen
`pi05_droid` checkpoint consumed only WAM6's three generated final camera views
and the registered commanded state. It returned a new complete native 15x8
action. The native action file SHA-256 is
`7a470fe06f24639ca9326882b0a1debfb74e2f4f0ee49fecb99d8b58bc9c05aa`;
the deterministic action-content SHA-256 is
`0c04804e0a23b9d47cc79d024c06dad9d3689f2ce5089959c5c3a8255ce32000`.

The exact output archive SHA-256 is
`6baff6d2190ab4c5556c1c53aca917ddac5f867035eb089e344b09dabde1a59b`.
The output validation completed with all three expected members. The policy
identity remained `ef2133d7cde82ef08bd9d0cabc7091cab9c4d80779e544c19831c23ff9f15fb8`.
The provider allocation charged 344 conservative GPU seconds and USD
`0.071667`; its owned Vast instance was destroyed and global Vast inventory
returned to zero. The terminal watchdog SHA-256 is
`6386b33fce179c904cd074a32f6bbbdd3d028faf61f1e3311a5c8a8fd5b2f360`.

Cumulative adapter-estimated GPU usage is 38,805 seconds and USD `11.345466`.
Together with unchanged evaluator/API spend of USD `8.418512`, cumulative GPU
plus evaluator/API spend is USD `19.763978`. This is not provider
billing-export reconciliation.

## Frozen WAM7 request

The generated-only transition builder preserved query 7's complete native
action before deriving the registered WAM conditioning and next commanded
state. Its first local attempt failed before producing a request because the
frozen OpenPI client package was not on the local module path; that empty
attempt is preserved as a local build failure. The successful second attempt
used the same frozen OpenPI client source and made no provider call.

The successful transition manifest identity is
`ff37440d5a6296e98975f13bfb5aac50a9fc60832726e307d0b6ac0dbe701f70`.
It binds 31 frames per camera view, a 31-state history, query 7's exact native
action, and no future physical RGB, future physical state, or physical outcome.
The WAM7 request SHA-256 is
`3d7158ff63ed6f1480e4acef462e6ce07ee32f5d42eee6b72ec9c2313d30de8c`.

The immutable provider bundle SHA-256 is
`8bde7a09328db4d72bcdd613874ba2146e8a66121a6f823f675efa4e8328385a`;
its receipt SHA-256 is
`7d72b429c16e2e46bf09e525d450cbc4ffcb6cc8b878a26daccbc8cd9b496ae1`.
Allocation 11 permits exactly one generated-only, three-view Ctrl-World
current-reference request with a USD 5 allocation cap, 4,800-second TTL, and
maximum global live inventory of two GPUs. Only one GPU may be allocated by
this request.

## Decision boundary

WAM7 may execute only through Blueprint's canonical paid-resource allocator
from a clean pushed experiment SHA after fresh credential, provider inventory,
object transport, cumulative budget, watchdog, preflight, and dry-run gates.
Success requires exact input/output validation, three registered camera views,
immediate reliability, provider teardown, provider-zero proof, and campaign
settlement. It would complete interaction seven only; it would not establish
causal WAM validity, complete-episode coherence, task success, ranking fidelity,
or physical confirmation. Gemini and GPT-5.6 Luna remain forbidden until the
complete 12-interaction episode and causal-control matrix pass.
