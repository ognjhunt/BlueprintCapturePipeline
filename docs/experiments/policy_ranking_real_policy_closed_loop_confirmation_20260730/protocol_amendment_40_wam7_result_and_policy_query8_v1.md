# Protocol Amendment 40: WAM7 result and policy query 8

Status: frozen after WAM7 and before policy query 8 provider execution

Date: 2026-08-01

## Allocation-11 result

Allocation 11 executed the prospectively frozen WAM7 request from immutable
pushed experiment SHA `5dacbc6fc05f3ced356a784750b6b6c11bec7c6a`. WAM7
returned five 320x192 generated-only frames for each of the three registered
DROID camera views. The exact 22-member provider archive SHA-256 is
`80815eafb4c00a68cd32c184004c9fe7dd4363d25d3fbeda39a2c7806c627d31`;
the validated runtime-result identity is
`11b47369b4827365ae010e5a1888724b29f334cfabea1197bff1cc9ce02740d8`.
The exact member allowlist, declared frame and MP4 hashes, frame geometry,
request identity, frozen model identity, generated-only boundary, and absence
of future physical observations and outcomes all passed.

The unchanged immediate reliability gate passed independently for all three
views with no flags or abstention. Mean camera-compensated motion was
`0.169698` for exterior view 2, `0.218212` for exterior view 1, and `1.777426`
for the wrist view. This remains a necessary single-window screen only. It is
not causal action qualification, complete-episode coherence, task success, or
policy ranking.

Preserved WAM7 evidence includes:

- provider archive SHA-256
  `80815eafb4c00a68cd32c184004c9fe7dd4363d25d3fbeda39a2c7806c627d31`;
- extraction receipt SHA-256
  `fbf37b502cfbd5f0ab39aab3634a29b038a7a8332b16975e86d5fac219428d11`;
- immediate reliability report SHA-256
  `9da6589e84469b498c4d456f8045de130cc268bf9abca277891dbf98fd5104ba`;
  and
- terminal watchdog SHA-256
  `d56033ba00388d7a86520fe3ae97715226e082763b50e129e3322e93ab6a56a3`.

## Provider and accounting closure

The owned WAM7 Vast instance `46523291` was destroyed and the independent
watchdog proved it absent. One unrelated NVIDIA Warehouse instance remained;
Blueprint did not mutate or tear down that resource. The post-close read-only
inventory passed the prospectively frozen maximum of two global GPUs and has
file SHA-256
`5d97e739c768ecf0a267d8861e094027ff6d03380e5c758ffaa2e2a6dbf92717`.

WAM7 charged 320 conservative GPU seconds and USD `0.153993`. Its campaign
reservation settled with zero open reservations. The settled ledger SHA-256 is
`06f379e713bccdfe8ee3800cb5587cdd4f35927db8d0cef4e11aadbf7b399050`.
Cumulative adapter-estimated GPU usage is 39,125 seconds and USD `11.499459`.
Together with unchanged evaluator/API spend of USD `8.418512`, cumulative GPU
plus evaluator/API spend is USD `19.917971`. This is not provider billing-export
reconciliation.

## Frozen policy-query-8 input

Policy query 8 uses the unchanged frozen `pi05_droid` checkpoint and policy
runtime SHA `bfb6fd6cc7e5fbce5d3ec4843558695f6aca20a0`. Its camera
inputs are WAM7's exact final generated frames in the released Ctrl-World view
order. Its state is the commanded-prefix kinematic state derived from policy
query 7's complete native 15x8 action. No future recorded RGB, future recorded
state, physical outcome, label, or policy substitution is present.

Bindings are:

- WAM7 request SHA-256
  `3d7158ff63ed6f1480e4acef462e6ce07ee32f5d42eee6b72ec9c2313d30de8c`;
- generated-observation manifest identity
  `e2125649ff3fc23a73f9f546af354b9e5691c35e3567e8ef89d81acdcd1b7404`;
- generated-observation manifest file SHA-256
  `543d5f4fbb436558d66a5039dd62889700a4e25304648669b72ae6cb578db5cc`;
- policy-query-8 input archive SHA-256
  `1ded66965d36578d628e42e2946c30b9815cdc6e7c5df672d75487ebb2e790f3`;
- input-receipt file SHA-256
  `1b0bbdcb153181dd8ba0368c3e4ba5f22e30072064420b68a7f951cae405569f`;
  and
- safe extraction receipt SHA-256
  `6eb0eba2e8039cbb8ffd4525c8b81ffb5a075ab27e4a0ca8a02b87e5ad63b8d7`.

The first local query-8 packaging invocation completed the immutable archive
and safe extraction, then failed while reading a nonexistent top-level field
from the extraction return value. The archive and extraction were preserved.
A recovery path revalidated the archive hash, exact extracted member set,
sizes, hashes, and embedded manifest before writing the missing extraction
receipt. No provider was called and no prior evidence was overwritten.

## Decision boundary

Policy query 8 may execute only through the canonical paid-resource allocator
from the clean pushed policy runtime SHA, after fresh credential, source,
signed-object-transport, provider inventory, cumulative-budget, watchdog,
preflight, and dry-run gates. The allocation remains one GPU even though the
campaign maximum is two. A successful same-identity query 8 does not complete
interaction eight until its action conditions a validated WAM8 result. Gemini
and GPT-5.6 Luna remain forbidden until the complete 12-interaction episode and
registered causal-control matrix pass.
