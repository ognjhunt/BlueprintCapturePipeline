# Protocol Amendment 42: WAM8 result and policy query 9

Status: frozen after WAM8 and before policy query 9 provider execution

Date: 2026-08-01

## Allocation-12 result

Allocation 12 executed the prospectively frozen WAM8 request from immutable
pushed experiment SHA `34e017aa55bb84db10592348b6da86b76d93c16d`.
The first live invocation omitted the already-authorized Vast API, launch, and
two-GPU-limit environment gates. It failed closed before authorization
consumption, provider mutation, or cost; its reservation settled at zero. The
versioned retry made those gates explicit and used the identical immutable
request.

WAM8 returned five 320x192 generated-only frames for each of the three
registered DROID camera views. The exact 22-member provider archive SHA-256 is
`89cbb3f0c57aecb2d474aabc4ba55aee3ea941a87ee45da291fa56b1fc70db02`;
the validated runtime-result identity is
`af6b4ebcae8bc8e17342fa13decb6676932333c48db4c74f1ff403a89454b6c7`.
The exact member allowlist, declared hashes, geometry, request identity, model
identity, generated-only boundary, and absence of future physical observations
and outcomes all passed.

The unchanged immediate reliability gate passed independently for all three
views with no flags or abstention. Mean camera-compensated motion was
`0.145788` for exterior view 2, `0.165579` for exterior view 1, and `2.487491`
for the wrist view. Timing correlations were `0.820332`, `0.989327`, and
`0.846612`. This is a necessary single-window screen only, not causal action
qualification, complete-episode coherence, task success, or policy ranking.

Preserved WAM8 evidence includes:

- provider archive SHA-256
  `89cbb3f0c57aecb2d474aabc4ba55aee3ea941a87ee45da291fa56b1fc70db02`;
- extraction receipt SHA-256
  `c86c661070267e43a55c9052a671607ab6786b5eec772872934191fafe30df09`;
- immediate reliability report SHA-256
  `13bd2400421fa1681b3cf2b6cceb79dc5cd259e493dacedc7f4cfdcd16738c59`;
  and
- terminal watchdog SHA-256
  `f5e28bf1128caeccb153562a82130b6cc9966040bb16b004e331597c55600184`.

## Provider and accounting closure

The owned WAM8 Vast instance `46526656` was destroyed. The independent
watchdog proved it absent and fresh authenticated global Vast inventory was
zero. WAM8 charged 390 conservative GPU seconds and USD `0.187658`; its
campaign reservation settled with zero open reservations. The settled ledger
SHA-256 is
`7b9480aa37ee6e373099ebfb6b4fe152c17391bbad0c36c1f8e3b166717e483d`.

Cumulative adapter-estimated GPU usage is 40,268 seconds and USD `11.843992`.
Together with unchanged evaluator/API spend of USD `8.418512`, cumulative GPU
plus evaluator/API spend is USD `20.262504`. This is not provider
billing-export reconciliation.

## Frozen policy-query-9 input

Policy query 9 uses the unchanged frozen `pi05_droid` checkpoint and policy
runtime SHA `bfb6fd6cc7e5fbce5d3ec4843558695f6aca20a0`. Its camera
inputs are WAM8's exact final generated frames in the released Ctrl-World view
order. Its state is the commanded-prefix kinematic state derived from policy
query 8's complete native 15x8 action. No future recorded RGB, future recorded
state, physical outcome, label, or policy substitution is present.

Bindings are:

- WAM8 request SHA-256
  `db965f7fa874854faeddea907e1ab0e93ae0dadec6026647f70a179886fdd24a`;
- generated-observation manifest identity
  `909d0379bcbfcc448392830aaa851632a9f83a5ecb6276ec50ff95f642f99bcd`;
- generated-observation manifest file SHA-256
  `11f7ea39d63ab83cb1c75dbb2f27740e250f450251ab18249538c9d99f32005e`;
- policy-query-9 input archive SHA-256
  `0dc68d43a17c367fd08f08f2af2b95e95f242dfc51e706ce04fb316a5b88419e`;
- input-receipt file SHA-256
  `27df5e38d4b7c4a48abb1637120a053aa16ea3dcd65b45df47e63115c6ec4083`;
  and
- safe extraction receipt SHA-256
  `ef313993370fcdeb066061706afbc90c3feedf77057e689ce5aa825dcaab4969`.

## Decision boundary

Policy query 9 may execute only through the canonical paid-resource allocator
from the clean pushed policy runtime SHA after fresh credential, source,
signed-object-transport, provider-inventory, cumulative-budget, watchdog,
preflight, and dry-run gates. A successful same-identity query 9 does not
complete interaction nine until its action conditions a validated WAM9 result.
Gemini and GPT-5.6 Luna remain forbidden until the complete 12-interaction
episode and registered causal-control matrix pass.
