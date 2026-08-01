# Protocol Amendment 22: policy query 3 result and WAM3

Status: frozen after policy query 3 and before WAM3 provider execution

Date: 2026-08-01

## Policy-query-3 result

The same frozen `pi05_droid` checkpoint completed policy query 3 from only
WAM2's three generated final camera frames and the registered commanded-prefix
state. The provider archive SHA-256 is
`28e3708b7cc920826848f399016c2abec7c99604c836853577c0395d339007a9`.
The complete native 15x8 action SHA-256 is
`e6cd6ec4a18bb11d9a4c61866fa1b300a1fb66f55b8addadbf521c162038cbb6`.
The policy identity SHA-256 remained
`ef2133d7cde82ef08bd9d0cabc7091cab9c4d80779e544c19831c23ff9f15fb8`.

The paid session was charged 653 GPU-seconds and USD `0.136042`. The campaign
ledger now commits 34,475 GPU-seconds and USD `10.062692`, with no open
reservation. The independent watchdog destroyed the owned Vast instance, and
the all-provider inventory proved zero after closure and again immediately
before this WAM3 freeze.

Preserved evidence includes:

- `openpi_policy_ranking_output_validation.json` SHA-256
  `52cd092b9795f450b3d4182a92893233ec734cf95e4fe9708f9b9ff05bfe3273`;
- `openpi_policy_ranking_monitor.json` SHA-256
  `d3d849aeed00a863bde54591c48c48535dff9eda18d269f5061a80896a7ece89`;
- `production_campaign_budget_ledger.json` SHA-256
  `fa342c046ca7395028426e5bb4e84fac4d8c3448300566cec7b51025ae2d5bb0`;
- `provider_zero_after_policy_query_3_20260801_v1.json` SHA-256
  `1623e54cd538cf65ce9814ff531738a9a9a0b0f269cd51d1fbc2b0fbb8aeb86e`;
  and
- `provider_zero_before_wam3_20260801_v1.json` SHA-256
  `53e4a170c3c063eb6a77d79a13de894f21f18275a5ab2ce44038f4b647100f6b`.

The first local cross-query validation artifact is preserved at SHA-256
`2a7be44e9a6709610ae5f78c1ac1a9a73b0f1e90fd088280d7683ea1e74d2687`.
It is invalid because it read identity and request hashes from the receipt's
top level rather than their registered nested locations. The corrected,
explicitly superseding validation is
`policy_requery_validation_v2.json` at SHA-256
`640363dea71abc91b160d9e7f600d5121111eba82d1bcec58c13771d15f9c54d`;
it passed with one non-null policy identity across queries 0 through 3 and four
distinct deterministic request hashes.

This completes the registered finite three-interaction engineering gate. It
does not complete the 12-interaction episode, causal qualification, ranking,
blind confirmation, captured-site transfer, or economics.

## Frozen WAM3 input

WAM3 is the fourth interaction of the unchanged 12-interaction complete
horizon. It conditions on policy query 3's complete native action through the
released Ctrl-World joint-velocity adapter. Its three view histories and
commanded Cartesian state history each contain 27 rows: the frozen 24-frame
initial history followed by the final WAM0, WAM1, and WAM2 feedback states.
No future physical RGB, future recorded state, outcome label, or policy
identity enters the WAM request.

The immutable bindings are:

- policy-query-3 native action SHA-256
  `e6cd6ec4a18bb11d9a4c61866fa1b300a1fb66f55b8addadbf521c162038cbb6`;
- released-adapter conditioning SHA-256
  `018295fe5f5da99826cc26d0b5e7c0ea034c748bdb51ed8cc3f73d8e782333bf`;
- WAM3 request SHA-256
  `d98f2353f1c376412a34a4003d9c123e73e7e2c9f0ec403727c8d60c85de6491`;
- request-manifest file SHA-256
  `c8c8755deb46d6392099e602a7afa407858ac1bf0f3bdd0f4437ce0b39cedd69`;
- transition-evidence manifest SHA-256
  `541426903ec102fce45aba227a2f95950e77fd026741311ea2778e0b6c27262e`;
- transition-evidence file SHA-256
  `05c46698026224e15603982886ed71cc77f27cd3f4a6869ba8fb5b5a6fb4329e`;
- transition-freeze receipt file SHA-256
  `500b64db4a084e0952710c7cefb4cd0d8ff8dd04027465aa90c6169d36c3e5ca`;
- provider-bundle SHA-256
  `4b36325ed206d743435a97da6fc2ea7cefa773250161f1ffb9766f1473fbe47d`;
  and
- provider-bundle receipt file SHA-256
  `2e08fd8a2d12ca56ffb59e026bc93f955122971fe3fcb465804f75361cb83a1b`.

Twenty-two focused adapter, request, and generated-observation tests pass.
WAM3 may execute only from a new immutable pushed experiment SHA through the
canonical paid-resource allocator after a fresh fail-closed provider preflight.
It retains the single-GPU limit, spend watchdog, hard TTL, independent
teardown, retained-session support, and provider-zero requirement.

## Decision boundary

If WAM3 completes and its immediate reliability gate passes, Blueprint may
construct only the registered generated observation and same-policy query 4.
Gemini 3.6 Flash and GPT-5.6 Luna remain forbidden until the complete
12-interaction episode and causal-control matrix both pass.
