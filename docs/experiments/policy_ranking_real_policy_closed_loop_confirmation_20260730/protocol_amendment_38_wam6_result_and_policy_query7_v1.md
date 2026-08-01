# Protocol Amendment 38: WAM6 result and policy query 7

Status: frozen after WAM6 and before policy query 7 provider execution

Date: 2026-08-01

## Allocation-10 result

Allocation 10 executed the prospectively frozen WAM6 request from immutable
pushed experiment SHA `73485613bc15dade81428d178187424e9670dbc3`. WAM6
returned five 320x192 generated-only frames for each of the three registered
DROID camera views. The exact 22-member provider archive SHA-256 is
`d8344a03388f36919f6ab084d0dfd44bccaeff077dac75d405fffc0dede51235`;
the validated runtime-result identity is
`b20137a549d88889a370db4d280a2dbcdcfd37c9b2d8d045adc9ae44283951a5`.
The exact allowlist, declared frame and MP4 hashes, 320x192 frame geometry,
request identity, frozen model identity, generated-only boundary, and absence
of future physical observations and outcomes all passed.

The unchanged immediate reliability gate passed independently for all three
views with no flags or abstention. Mean camera-compensated motion was
`0.140209` for exterior view 2, `0.230937` for exterior view 1, and `1.739114`
for the wrist view. This remains a necessary single-window screen only. It is
not causal action qualification, complete-episode coherence, task success, or
policy ranking.

Preserved WAM6 evidence includes:

- provider archive SHA-256
  `d8344a03388f36919f6ab084d0dfd44bccaeff077dac75d405fffc0dede51235`;
- extraction receipt SHA-256
  `56c628d5298de703be5e2c1111acf6cb0beafadf760cba448225619635756c3b`;
- immediate reliability report SHA-256
  `88e33a1a20787720e79a9593d69538f0e9f646339336641e03f6a07f32927445`;
  and
- terminal watchdog SHA-256
  `89ae5f158c999243bd900d28470a0bb3dfd49451f77d3e02b5f0b62575b13d7d`.

## Two-GPU terminal and accounting reconciliation

The owned WAM6 Vast instance `46519017` was destroyed. The independent
watchdog then proved the exact instance absent twice and the WAM6 name scope at
zero while one unrelated NVIDIA Warehouse instance remained. Under the
prospectively frozen maximum of two concurrent GPUs, that residual unrelated
instance no longer delays WAM closure. Blueprint neither mutated nor tore down
the Warehouse allocation and does not claim global provider zero.

The outer allocator exited before the late watchdog terminal receipt could
settle its open campaign reservation. Generic recovery SHA `17154c5bd` now
preserves the original fail-closed receipt, requires exact owned-instance
absence, reuses the canonical allocator settlement function, and is
idempotent. Seventy-six focused allocator and watchdog tests pass, together
with targeted lint and formatting checks.

The reconciliation charged the observed WAM6 ceiling of 355 GPU seconds and
the adapter estimate of USD `0.171192`; it made zero provider mutations. Its
receipt SHA-256 is
`5750bf21d7f691e70dcc6b2fb0a9b7bbba96f4c888f9ff011c3d0b6c5afc5999`.
The settled ledger SHA-256 is
`cee524e4d5b2b1005109becde120f4165c49bb58aa4785725e74ff1c902a5a8f`.
Cumulative adapter-estimated GPU usage is 38,461 seconds and USD `11.273799`.
Together with unchanged evaluator/API spend of USD `8.418512`, cumulative GPU
plus evaluator/API spend is USD `19.692311`. This is not provider billing-export
reconciliation.

## Frozen policy-query-7 input

Policy query 7 uses the unchanged frozen `pi05_droid` checkpoint and policy
runtime SHA `bfb6fd6cc7e5fbce5d3ec4843558695f6aca20a0`. Its camera
inputs are WAM6's exact final generated frames in the released Ctrl-World view
order. Its state is the commanded-prefix kinematic state derived from policy
query 6's complete native 15x8 action. No future recorded RGB, future recorded
state, physical outcome, label, or policy substitution is present.

Bindings are:

- WAM6 request SHA-256
  `84f5981189fea07b958254f032e6adbe91a6980eab90e725d807c4ad488d26c0`;
- generated-observation manifest identity
  `29d63302c1876633269ccbe3033cb6325558f84b04544f33adf45ebdbfcd51e2`;
- generated-observation manifest file SHA-256
  `e3cccde34935e8684417490c9f561beee51c4aa2b199b786a9060ddc372ef4e5`;
- policy-query-7 input archive SHA-256
  `5e638f73bcce71f40dac3db19a76c29b4793bbed177695d1386353d9c091d533`;
- input-receipt file SHA-256
  `89f8835a0fc647334e343a1eb4066049063e866b99d9826a352a8bfe64bf9f8b`;
  and
- safe extraction receipt SHA-256
  `1cb60a937a2bc175e5a7ff7551e10369fc2269e5579884ec321a8fbbd84b0c88`.

## Decision boundary

Policy query 7 may execute only through the canonical paid-resource allocator
from the clean pushed policy runtime SHA, after fresh credential, source,
signed-object-transport, provider inventory, cumulative-budget, watchdog,
preflight, and dry-run gates. The allocation remains one GPU even though the
campaign maximum is two. A successful same-identity query 7 does not complete
interaction seven until its action conditions a validated WAM7 result. Gemini
and GPT-5.6 Luna remain forbidden until the complete 12-interaction episode and
registered causal-control matrix pass.
