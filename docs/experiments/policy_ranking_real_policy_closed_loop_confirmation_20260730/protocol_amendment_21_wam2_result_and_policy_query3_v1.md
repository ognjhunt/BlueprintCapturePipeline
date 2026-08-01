# Protocol Amendment 21: WAM2 result and policy query 3

Status: frozen after WAM2 and before policy query 3 provider execution

Date: 2026-08-01

## Allocation-6 result

Allocation 6 executed the prospectively frozen WAM2 request from repository SHA
`20d32ed6e1bd3fd8acf60deaa0c463349af7ca3e` on Vast instance `46492226`.
It returned one generated-only Ctrl-World result for all three registered camera
views. Each view contains five generated 320x192 frames. The complete provider
archive SHA-256 is
`f33e5348c3420e87c3e89d685e235c2a6399787376164ea75390b48b55c9b163`;
the validated runtime-result identity is
`9eb184ffe724937a6a092e23a22b444ecbdbbbe3cd8479902f5a8930a6ff4928`.

The frozen immediate reliability gate passed independently for all three views,
with no flags and no abstention. This is necessary single-window reliability
evidence only. It is not the registered causal-control matrix, complete-episode
coherence, ranking fidelity, physical success, or confirmation.

The paid session observed `390.494895` seconds. The campaign ledger charges the
ceiling-safe 391 seconds and USD `0.109013`. Committed campaign GPU use is now
33,822 seconds and USD `9.926650`; the evaluator/API ledger remains USD
`8.418512`. The adapter destroyed the owned instance, its independent watchdog
reached provider-terminal state, and the post-run all-provider inventory proved
zero.

Preserved evidence includes:

- `adapter_output.json` SHA-256
  `d93c64397324b98d46918913fa3ed19866a440f9ae7ee5e956a6a49e030ecc44`;
- `job/vast_provider_adapter_result.json` SHA-256
  `d9b975c2cb633b387c07cf6940193d1eb7a80a22e934ecd1295cdbdd7254b444`;
- `job/vast_teardown_manifest.json` SHA-256
  `247af576babb2b066ab3a001127fca2aa9e82a24c248ca44ced578fec5ffab81`;
- the independent watchdog final record SHA-256
  `e14641068a6e6abd73e086f995f7b435b2add87282a51866ded684c50a852cad`;
- `production_campaign_budget_settlement.json` SHA-256
  `bdd5e9534b582b76da59ff68296d5d2b63505d97884ed8859ef8a750c6ab75a6`;
- `provider_output_extract_receipt.json` SHA-256
  `ad12b37f84f8e8af10db6ba63cecf68909b892dec983a192874c429236ad707d`;
- `immediate_reliability_report.json` SHA-256
  `db8e1deafd518c18ee26e79778616d438023ee866ed78b346faf8ec074e840bd`;
  and
- `provider_zero_after_wam2_20260801_v1.json` SHA-256
  `6645a9be468493f31baa52b29cc956be461c6a5c6f31ed0fa7ad1dd3bbf1cd5e`.

## Frozen policy-query-3 input

The next request uses the same frozen `pi05_droid` policy checkpoint and exact
runtime source SHA `2e46cb6b9f209ec0646744d51fbd3b6af4a54619`. Its only camera inputs are
WAM2's final generated frames in the registered view order. Its robot state is
the commanded-prefix kinematic state derived from query 2's complete native
15x8 output; no physical future pixels, future recorded state, or outcome label
is present.

The feedback-chain bindings are:

- query-2 native action SHA-256
  `569325b72e3e54bb4607450be903ccefe3774c6ea389d02aaa10a35174010793`;
- WAM2 request SHA-256
  `584d5a5aef05ac5e750342c6ba27b834f444d4d6328e9afa7eb1a55ac7c1702e`;
- transition-evidence SHA-256
  `e86516aab143f0d0ebb08548e5fb1cec4c4f93ee56ff24da16bacf031b1b784f`;
- generated-observation manifest SHA-256
  `3c57ea9ca7b3531d7d31edc8b3c6d852cc52a3c2bea5088e9f228ff87417a3db`;
- policy-query-3 input archive SHA-256
  `b513ea3b68d7c1e80354f6ea44260e578ab83f3d0a98b1609be0b3a9221e6926`;
  and
- input-receipt file SHA-256
  `dbfd3d04886f26150a1f8c27f6d65412eba11804cabed391c171cc6199d1803a`.

The object-store output key is fresh, its pre-execution absence was confirmed,
and the signed PUT/GET sentinel round trip passed. The first provider preflight
after staging observed an unrelated, independently owned Vast GPU and therefore
failed closed. Blueprint will not touch that resource. Policy query 3 may launch
only after a new mutation-free preflight proves global provider zero and all
existing one-GPU, budget, watchdog, TTL, teardown, and provider-zero controls
pass.

## Decision boundary

A complete, same-identity policy query 3 will complete the registered finite
three-interaction engineering gate. It will not complete the frozen 12-
interaction episode or the causal-control matrix. Gemini and GPT-5.6 Luna remain
forbidden until both of those later gates pass.
