# Protocol Amendment 23: WAM3 result and policy query 4

Status: frozen after WAM3 and before policy query 4 provider execution

Date: 2026-08-01

## Allocation-7 result

Allocation 7 executed the prospectively frozen WAM3 request from repository SHA
`afab1151952a15ffb4f3a1611c4a1657a468d4f5` on Vast instance `46497216`.
It returned one generated-only Ctrl-World result for all three registered camera
views. Each view contains five generated 320x192 frames. The complete provider
archive SHA-256 is
`2840a6457db99f2f8fc2bbe664eef57e3777244b15bbbf01449087909e70162a`;
the validated runtime-result identity is
`f8395f52562bfa187aeeae15ab3389935f42ec949b96ef432b95ebc881d67f48`.

The frozen immediate reliability gate passed independently for all three views,
with no flags and no abstention. This is necessary single-window reliability
evidence only. It is not the registered causal-control matrix, complete-episode
coherence, ranking fidelity, physical success, or confirmation.

The paid session observed `317.490563` seconds. The campaign ledger charges the
ceiling-safe 318 seconds and USD `0.153127`. Committed campaign GPU use is now
34,793 seconds and USD `10.215819`; the evaluator/API ledger remains USD
`8.418512`. The adapter destroyed the owned instance, its independent watchdog
reached provider-terminal state, and the post-run all-provider inventory proved
zero.

Preserved evidence includes:

- `adapter_output.json` SHA-256
  `885d9eb1fecfd61cb99dc1491c46aa0e8a853a002098beb12c2885f761ee52ab`;
- `vast_provider_adapter_result.json` SHA-256
  `dcba927c39e12d61c38e8c20b59f90f8e53eef50d302421ff6331779eab2bac0`;
- `vast_teardown_manifest.json` SHA-256
  `a1033e91688b9ae58766dcc528f5d1d6982485b650df88153ee838ea0a003b1f`;
- the independent watchdog final record SHA-256
  `b2c94025a9ef1a6669afd923f537472e221db90f2df6456f88e0b2c9516879c8`;
- `production_campaign_budget_settlement.json` SHA-256
  `6819212ff7cdf468d79e596dd4e94baba2208fb36147c898cac301adf96bf3dc`;
- `provider_output_extract_receipt.json` SHA-256
  `c627c18f596cb4a9cb9d1bb3f9a2e75379291b728913e2f78e8a6cc0a8dad25b`;
- `immediate_reliability_report.json` SHA-256
  `e22593c166c35b4efdfac3bc9da1ef84be361d3211a9a80c1ce1a7912fb8f578`;
  and
- `provider_zero_after_wam3_20260801_v1.json` SHA-256
  `dbcf1dc21db2046b32ee6434c3d632940bd815c985d7de11bff0dba5bedba3b5`.

## Frozen policy-query-4 input

The next request uses the same frozen `pi05_droid` policy checkpoint and exact
runtime source SHA `2e46cb6b9f209ec0646744d51fbd3b6af4a54619`. Its only camera inputs are
WAM3's final generated frames in the registered view order. Its robot state is
the commanded-prefix kinematic state derived from query 3's complete native
15x8 output; no physical future pixels, future recorded state, or outcome label
is present.

The feedback-chain bindings are:

- query-3 native action SHA-256
  `e6cd6ec4a18bb11d9a4c61866fa1b300a1fb66f55b8addadbf521c162038cbb6`;
- WAM3 request SHA-256
  `d98f2353f1c376412a34a4003d9c123e73e7e2c9f0ec403727c8d60c85de6491`;
- transition-evidence manifest SHA-256
  `541426903ec102fce45aba227a2f95950e77fd026741311ea2778e0b6c27262e`;
- generated-observation manifest identity
  `fb7db3b97efeed709dab02df77621e4ede34fb7e6656fac6528fd2b659b823ca`;
- generated-observation manifest file SHA-256
  `949db58f9dcc7b8850f0ffbf85bfeddbf2a31a2a25c5f5f8f578701b35628c40`;
- policy-query-4 input archive SHA-256
  `d18f33e7453fe80e906cb7b3ae134e3767885334f2a9f42e0ef6b028248fdb98`;
- input-receipt file SHA-256
  `24dd54a9c4b6e1a44a7a63c6c6d2550bfda9efc71d6d7926e248ab8427ae6f0d`;
  and
- independent input-extraction receipt SHA-256
  `775ea876e556ba0b60c924d24b47d6277e026bfb7845359bebfddba2245f9518`.

Policy query 4 may launch only after immutable publication, fresh provider-zero,
fresh output-key absence, signed transport validation, and mutation-free
preflight. It retains the existing one-GPU, budget, watchdog, TTL, teardown,
and provider-zero requirements.

## Decision boundary

A complete, same-identity policy query 4 will complete interaction four of the
registered 12-interaction horizon. It will not complete the episode or the
causal-control matrix. Gemini and GPT-5.6 Luna remain forbidden until both of
those later gates pass.
