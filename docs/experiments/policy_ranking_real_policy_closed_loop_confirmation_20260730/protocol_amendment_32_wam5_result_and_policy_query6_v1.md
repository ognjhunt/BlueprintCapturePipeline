# Protocol Amendment 32: WAM5 result and policy query 6

Status: frozen after WAM5 and before policy query 6 provider execution

Date: 2026-08-01

## Allocation-9 result

Allocation 9 executed the prospectively frozen WAM5 request from immutable
pushed experiment SHA `06cd6bbc1a2d153cea2fc7862af4091d0e5fc1d6`. The
two-GPU campaign ceiling admitted this one-GPU allocation while one unrelated
writer-owned GPU was live. Blueprint did not wait for, mutate, or tear down the
unrelated allocation.

WAM5 returned five 320x192 generated-only frames for each of the three
registered DROID camera views. The complete provider archive SHA-256 is
`663e0c05506c2012361048e42c285d4f81146dd814b18fbb89cebe1a4fdbec17`;
the validated runtime-result identity is
`2913862f977280672532c9944957c731699c2de936a16b46835fb98d2790cc0a`.
The exact archive allowlist and every declared generated-frame hash passed.

The frozen immediate reliability gate passed independently for every view,
with no flags or abstention. Mean motion was `0.069963` for exterior view 2,
`0.132912` for exterior view 1, and `2.434882` for the wrist view. This is a
necessary single-window screen only. It does not establish causal action
following, complete-episode coherence, task success, or policy ranking.

The provider session observed `319.130456` seconds at estimated cost USD
`0.153918`; cumulative accounting charges the conservative ceiling of 320 GPU
seconds. The owned Vast instance `46511487` was destroyed. Its independent
watchdog subsequently reached `provider_terminal`, confirmed the exact instance
absent, and the final global inventory was zero.

Preserved evidence includes:

- `provider_output_extract_receipt.json` SHA-256
  `186ed58903138fa39e071c2f236d7e417525ee0305ff9c11bc11f28ccaf4397c`;
- `immediate_reliability_report.json` SHA-256
  `ed8d352a4fa860a5ed687b12c67c5ab75da5263f198077ee6e3434f632733fd3`;
- terminal watchdog SHA-256
  `3b88eef807b54d3ca86a4e27e8d96ca6a2c3329a08d4f7cea7245522b4dfa8e1`;
  and
- provider adapter result SHA-256
  `665e65e681e16fe96ddba7aec2a26fbc813d773895a262debefe82bf6f3651c8`.

## Campaign-ledger settlement correction

The real pre-mutation production reservation existed, but the outer authorized
WAM runner omitted the nested adapter's mutation, runtime, and cost fields. The
allocator therefore settled the original reservation at zero. The original
ledger and faulty settlement remain unchanged. Generic runtime fix
`c0acdc4d` propagates those fields, and 16 focused authorized-runner regression
tests pass on that fix.

The append-only correction receipt SHA-256 is
`746f68e753db5cfe35a60a326f84fbba6edf07fd4880b84dca5d26564456aea4`;
the corrected ledger SHA-256 is
`81bfddfcd8ed7f4e992d2b05ba2e5f888f8630c8ffd274ad09b242f8d617f26f`.
Corrected cumulative GPU usage is 37,043 seconds and USD `10.881149`.
Together with unchanged evaluator/API spend of USD `8.418512`, cumulative GPU
plus evaluator/API spend is USD `19.299661`. This is adapter-estimated
accounting, not provider billing-export reconciliation.

## Frozen policy-query-6 input

Policy query 6 uses the unchanged frozen `pi05_droid` checkpoint and policy
runtime SHA `ede38013d6cb2a5453ed39ba39c607a7f497a639`. Its only
camera inputs are WAM5's final generated frames in the registered view order.
Its state is the commanded-prefix kinematic state derived from policy query 5's
complete native 15x8 action. No future recorded RGB, future recorded state,
physical outcome, label, or policy substitution is present.

Bindings are:

- WAM5 request SHA-256
  `5747e6dc6975c405b9c92ef2f275dbb5dab6072253327dbfbfef88653217725d`;
- generated-observation manifest identity
  `3ff963a9f0a6605cb73c7d02be79217ed496822cb0723258eb3079f432b6892f`;
- generated-observation manifest file SHA-256
  `72a7d67e259268ba7f255b6d7bc000ba2d506c152893cb4a2e48ed6e80399dfe`;
- policy-query-6 input archive SHA-256
  `a55e389e952629a8b435710b42c656f4c5bc100b234594ffd1531a9c1e05f008`;
- bundle-receipt file SHA-256
  `9b522488a8a439eaa786736e596b7a96a9a05a94cfbe851238fdc24360df75c2`;
  and
- independent extraction receipt SHA-256
  `2bb1a810af2c52235ad15bb2c3a8312cdff95f5fd0514a97f7fbf7142e5f0a3b`.

## Decision boundary

A same-identity policy query 6 will complete interaction six of the registered
12-interaction horizon. It may execute only from the clean pushed policy runtime
SHA through the canonical allocator, after fresh credential-presence,
provider-inventory, signed-object-transport, output-absence, preflight, budget,
watchdog, and dry-run gates. The allocation remains one GPU even though campaign
concurrency is capped at two. Gemini and GPT-5.6 Luna remain forbidden until the
complete episode and causal-control matrix pass.
