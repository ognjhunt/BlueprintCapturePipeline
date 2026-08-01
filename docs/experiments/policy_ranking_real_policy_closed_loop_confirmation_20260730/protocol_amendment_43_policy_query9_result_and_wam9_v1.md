# Protocol Amendment 43: Policy query 9 result and WAM9

Status: frozen after policy query 9 and before WAM9 provider execution

Date: 2026-08-01

## Policy-query-9 result

The ninth policy request executed from clean pushed policy runtime SHA
`bfb6fd6cc7e5fbce5d3ec4843558695f6aca20a0`. The unchanged frozen
`pi05_droid` checkpoint consumed only WAM8's three generated final camera views
and the registered commanded state. It returned a new complete native 15x8
action. The native action file SHA-256 is
`a74e349d05150c05bb4cadc4b35e5c5732873cf96a62a780ba43e7bad417fa71`;
the deterministic action-content SHA-256 is
`51da4c642df70ae449a564f6a01c6e2287511999bd12540876b3395a2bb2aca8`.
The deterministic request SHA-256 is
`6cb9d05af99b49efe383a8c84d43da07be18b0f8666bf1502eadbd58c3f0545d`.

The exact provider output archive SHA-256 is
`23ee195090bf22faaff32f56db0a35fa5f7b219b88db3fd6ac228c5d627c1f03`.
Its exact three-member output allowlist, native shape, finite float64 values,
policy identity, request binding, and receipt digests passed. The policy
identity remained
`ef2133d7cde82ef08bd9d0cabc7091cab9c4d80779e544c19831c23ff9f15fb8`.

The scientific output completed before a concurrent-lane bookkeeping race:
the owned instance was already absent, two fresh global Vast reads were zero,
and the attempt-local pending teardown was closed, but an unrelated provider
lane's still-open pending record caused the query lane release to retain. The
generic reconciliation now scopes pending teardown records to the exact paid
lane while retaining provider-inventory proof. Four focused regression cases
passed and the fix was pushed at SHA
`2b7133aaa314d54eba885e01a6bd3582f5e37d80`. The preserved reconciliation
receipt SHA-256 is
`fb1bb299afac41b31fa3d44ab486c8e7e426000b504498009ec16ecacd0ba144`.

Policy query 9 charged 363 conservative GPU seconds and USD `0.075625`.
Cumulative GPU usage is 40,631 seconds and USD `11.919617`; the reservation is
settled with zero open reservations. Together with unchanged evaluator/API
spend of USD `8.418512`, cumulative GPU plus evaluator/API spend is USD
`20.338129`. This is not provider billing-export reconciliation.

## Frozen WAM9 request

The first local transition build failed before staging the request because the
frozen OpenPI client source was not on `PYTHONPATH`. That zero-provider,
zero-cost failure is preserved. The versioned successor used the identical
query-9 action, history, state, seed, adapters, and thresholds with the frozen
OpenPI client path restored.

The successful transition freeze identity is
`639fd964b218249b6d85ffde0ccc6988b4ab2981e8a8fbb1ac4a3f8bf8419a3d`.
It binds 33 frames per camera view, a 33-state history, query 9's exact native
action, and no future physical RGB, future physical state, or physical outcome.
The WAM9 request SHA-256 is
`87d7993f03e87bf66582eb69183e3f170abe28f888414679e3f35fd3f9097cd5`;
the registered conditioning-array SHA-256 is
`6a405760dc60829b5f48487bc0a6f4c9391967d2125b71e7f1a60b94cd6573ab`.

The immutable provider bundle SHA-256 is
`da92b0c9723af9c9ab8e51ac18b221264abb422b5353a9c41ad3b7c548a5e2d2`;
its receipt SHA-256 is
`2718f56df0dfb08a53ca427331d0cd6dc8d02cb3bf1799d6f240a6e6e34b11b6`.
Allocation 13 permits exactly one generated-only, three-view Ctrl-World
current-reference request with a USD 5 allocation cap, 4,800-second TTL, and
maximum global live inventory of two GPUs. Only one GPU may be allocated by
this request.

## Decision boundary

WAM9 may execute only through Blueprint's canonical paid-resource allocator
from a clean pushed experiment SHA after fresh credential, provider inventory,
object transport, cumulative budget, watchdog, preflight, and dry-run gates.
Success requires exact input/output validation, three registered camera views,
immediate reliability, provider teardown, provider-zero proof, and campaign
settlement. It would complete interaction nine only; it would not establish
causal WAM validity, complete-episode coherence, task success, ranking fidelity,
or physical confirmation. Gemini and GPT-5.6 Luna remain forbidden until the
complete 12-interaction episode and causal-control matrix pass.
