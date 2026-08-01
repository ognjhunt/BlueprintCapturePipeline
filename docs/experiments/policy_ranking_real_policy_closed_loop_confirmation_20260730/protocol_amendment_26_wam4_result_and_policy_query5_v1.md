# Protocol Amendment 26: WAM4 result and policy query 5

Status: frozen after WAM4 and before policy query 5 provider execution

Date: 2026-08-01

## Allocation-8 result

Allocation 8 executed the prospectively frozen WAM4 request from immutable
pushed experiment SHA `45f06d9c9802f93fc9a9a559d47094af9f0ef7b2` on Vast instance
`46503633`. It returned one generated-only Ctrl-World result containing five
320x192 frames for each of the three registered camera views. The complete
provider archive SHA-256 is
`7c47095b34b3da449b84cedf5c59d9f6836ab0231b7c8d90e042ed1b9c52e8d9`;
the validated runtime-result identity is
`6b2d55338ab0843a88d8fab2f95c0d0364592861b7f104ee92c60f7e9985c00c`.

The frozen immediate reliability gate passed independently for all three views,
with no flags and no abstention. The mean motion scores were `0.184376` for
exterior view 2, `0.385538` for exterior view 1, and `1.729554` for the wrist
view. This is necessary single-window reliability evidence only. It is not the
registered causal-control matrix, complete-episode coherence, ranking fidelity,
physical success, or confirmation.

The provider session observed `462.379326` seconds at USD `0.205787`; cumulative
accounting charges the conservative ceiling of 463 GPU-seconds. The adapter
destroyed the owned instance, the independent watchdog reached provider-terminal
state, and the post-run all-provider inventory proved zero.

Preserved evidence includes:

- `adapter_output.json` SHA-256
  `65f66d2e33867c72293f18cd2cca9c5450c3bd0f793efece2f39d451d3cea68d`;
- `vast_provider_adapter_result.json` SHA-256
  `91eb3311ddc72dd9297c7a8248d3349cb9ecbaeb22434fb384e308eb848732af`;
- `vast_teardown_manifest.json` SHA-256
  `4191066e94eaf571d4a17021925d16f08b0e54c73b34028c23d143265b2dbad1`;
- `provider_output_extract_receipt.json` SHA-256
  `26c738892db5041d50e7d2ea11ae1353ff6592432791f66cbe075477bf607b94`;
- `immediate_reliability_report.json` SHA-256
  `5f0a3d9fa04ae1ccad6f1d07e0089f4240dc25a6adccc92b9febaf11bbfffb5d`;
  and
- `provider_zero_after_wam4_20260801_v1.json` SHA-256
  `054eeac248f5203aeab288700335c199495e075253911cd72f9e6172008a9113`.

## Cumulative-ledger disclosure and correction

The canonical allocator received a production-campaign-ledger argument for
WAM4, but this successor runtime did not materialize or reserve that cumulative
ledger before provider mutation. The paid stage still had its distinct
pre-mutation USD 5 session hard cap, USD 3 target, 80-minute live limit, USD
2.05/hour offer ceiling, independent watchdog, global one-GPU guard, and
required teardown. Those stage-local controls passed and no continuing spend
remains. This is an orchestration-evidence defect; it is not a WAM output
failure.

The defect is preserved rather than rewritten. The late reconciliation receipt
SHA-256 is
`0799f70d648bf9e3d9c851103c899682463e4b38b7011e8b88f2e3deddacb9e5`;
the reconciled ledger SHA-256 is
`6848067d8bf0561e242c5e28856cdada1221cd607cbddd3668b5cfce89c93ef3`.
They explicitly do not claim that a production reservation preceded WAM4. The
reconciled totals are 36,561 GPU-seconds and USD `10.693481`. Together with the
unchanged evaluator/API ledger of USD `8.418512`, cumulative GPU plus evaluator
spend is USD `19.111993`.

Every later paid stage must prove a real open production-campaign reservation
before provider mutation and settle it after teardown. The stage-local cap,
watchdog, TTL, concurrency, and provider-zero contracts remain independently
required.

## Frozen policy-query-5 input

The next request uses the same frozen `pi05_droid` checkpoint and corrected
generic runtime SHA `529bc39639021448c0743f36f0cde75a8f5d6098`. Its only
camera inputs are WAM4's final generated frames in the registered view order.
Its robot state is the commanded-prefix kinematic state derived from query 4's
complete native 15x8 action. No physical future pixels, future recorded state,
physical outcome, or policy substitution is present.

The feedback-chain bindings are:

- query-4 native action file SHA-256
  `8b11137e6e9979f5852e7f72e7b06db0f6bea3f45134e1dc577a17109020a63e`;
- WAM4 request SHA-256
  `67a6ed7951d0f80e2ad13344111f827a26975809d5dab9b86dea34799666df68`;
- transition-evidence manifest SHA-256
  `b453462ce8ef2864d4db8995a4b4cc38c8e09f794b72b3e63c205d086c122ed1`;
- generated-observation manifest identity
  `888121a1bbd47f19da2b5dd75017c141ac27c4a51302b6eb444aad8ad686f060`;
- generated-observation manifest file SHA-256
  `dc5ccf701defe1d8d1ca06a6078d3c1e34382c7f8aa81e4489490e0fbefd4d8a`;
- policy-query-5 input archive SHA-256
  `6a778506bdb97e16ca4b1bfbd8b060a0da647d0db695f9f3168f95e616870ee7`;
- input-receipt file SHA-256
  `0e0e56813738cfdefdec02de276805a6397181363a658aa471d213cf591727f0`;
  and
- independent input-extraction receipt SHA-256
  `e906cacbe9a4d6d5e62c44bcef9c4898d7a5f0e4269f8735e2dd69201745a993`.

Six focused observation, portable-bundle, and policy-input regression tests
pass. Policy query 5 may launch only after immutable publication, a fresh real
production-campaign reservation, fresh provider-zero, fresh output-key absence,
signed transport validation, mutation-free preflight, and canonical allocator
dry run. All existing one-GPU, spend, watchdog, TTL, teardown, and
provider-zero requirements remain in force.

## Decision boundary

A complete, same-identity policy query 5 will complete interaction five of the
registered 12-interaction horizon. WAM5 has not been authorized or executed by
this amendment. Gemini 3.6 Flash and GPT-5.6 Luna remain forbidden until both
the complete episode and causal-control matrix pass.
