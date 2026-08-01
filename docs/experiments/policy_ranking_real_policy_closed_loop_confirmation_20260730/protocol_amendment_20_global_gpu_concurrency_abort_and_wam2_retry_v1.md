# Protocol Amendment 20: global GPU concurrency abort and WAM2 retry

Status: prospectively frozen before the allocation-6 WAM2 retry

Date: 2026-08-01

## Preserved allocation-5 concurrency abort

Allocation 5 launched the immutable WAM2 input on Vast instance `46490043`
from repository SHA `0f2a2af0f23156a854e414af5172ffeaba5fea03` after a fresh
zero-inventory preflight. While that instance was running, a separate writer
started unrelated Vast instance `46490528`. The global provider inventory then
contained two live GPUs, violating this experiment's frozen one-GPU concurrency
ceiling. Blueprint did not touch the unrelated writer or resource. The owner
interrupted its own allocation-5 controller; the controller's `finally` path
destroyed `46490043`, and the independent watchdog remained armed until the
unrelated resource ended and global provider zero was independently verified.

Allocation 5 produced no Ctrl-World result. Its provider output is a valid but
empty 22-byte ZIP placeholder with SHA-256
`8739c76e681f900923b900c9df0ef75cf421d39cabb54650c4b9ad19b6a76d85`.
This is an orchestration-safety abort, not a WAM generation, reliability,
causal-qualification, or scientific result. It earns no interaction credit.
Allocation 5 is consumed and may never be reused.

The preserved evidence under external live-attempt directory
`ctrl_world_current_reference_wam_2_live_0f2a2af0_20260801_v3` includes:

- `adapter_output.json` SHA-256
  `4d22c05aefa6aec711b754151f23b3819aa18b894d4bd0da4ae220609c058afd`;
- `job/vast_provider_adapter_result.json` SHA-256
  `9ed3d358cbd6f171542af427fa27e63ff6adb32f1237677ba29e815cd661a63c`;
- `job/vast_teardown_manifest.json` SHA-256
  `d62a835aea2ec44bee7c76ff4886ff351949445da47e51c9374c013d7f598da3`;
- `job/independent_vast_watchdog/groot_oscar_runpod_canary_watchdog.json`
  SHA-256
  `b2e93b83af58745013916455c1faa19ff319032238bc3845be6f96e223b0b189`;
- `provider_session_budget_ledger.json` SHA-256
  `a38660b4f0da3a80d7a537d0f33cd676544939c4468fcafb9e4b23e0a269607b`;
- `production_campaign_budget_settlement.json` SHA-256
  `7b580d0933a28873b2baca20c0ec231c5c8f7c803769452d24ba11dcd63230ac`;
- `provider_zero_after_concurrency_abort_v1.json` SHA-256
  `d628f984b3df01d6ec7951a83b630d02992d11b5a0fbe59d2e6678129678d86d`;
- `provider_output_placeholder_receipt_v1.json` SHA-256
  `5c1cbc15ea4d6115a2622434a6e1fb3ae77a47b653db217b544c21ed303b2f87`.

The provider session observed `459.696954` seconds and estimated USD
`0.128167`. The campaign ledger conservatively charged 460 GPU seconds and USD
`0.128167`, bringing committed campaign GPU use to 33,431 seconds and USD
`9.817637`. The evaluator/API ledger remains USD `8.418512`; combined committed
campaign spend before retry is USD `18.236149`.

## Frozen generic concurrency correction

The Vast adapter now supports an opt-in
`BLUEPRINT_VAST_MAX_GLOBAL_LIVE_INSTANCES` limit. During every container-log
poll it re-fetches the global Vast instance inventory and counts only
non-terminal instances. If the active count exceeds the frozen limit, the poll
ends with `external_abort_probe_triggered`, records
`vast_global_live_instance_limit_exceeded:<observed>><limit>`, and enters the
existing fail-closed teardown path. Provider-query errors are recorded but do
not fabricate an abort, success, or provider-zero claim.

Every later execution in this experiment must set the limit to exactly `1`.
The existing zero-inventory preflight and no-active-writer checks remain
mandatory immediately before launch. This runtime check closes the observed
mid-run race; it does not authorize touching an unrelated writer or resource.
Focused regressions must prove active-versus-terminal instance counting,
automatic external-abort exit, terminal-instance exit, no-progress handling,
and missing-container behavior.

## Scientifically identical allocation-6 retry

The retry remains exactly bound to:

- `pi05_droid` policy query 2 and native action SHA-256
  `569325b72e3e54bb4607450be903ccefe3774c6ea389d02aaa10a35174010793`;
- WAM request SHA-256
  `584d5a5aef05ac5e750342c6ba27b834f444d4d6328e9afa7eb1a55ac7c1702e`;
- provider bundle SHA-256
  `cd413ca58d214a5212848a01f0b1cc9596082412a9040b6782d8df2a68bab9f9`;
- Ctrl-World checkpoint revision
  `8cf814693f411962dc866a2ddb5b785afd17a93a`;
- seed 0, the same three registered views, and the same 26-frame feedback
  history; and
- all existing reliability, collapse, causal, and abstention thresholds.

Allocation 6 is single-use, one-GPU, target USD 3, maximum USD 5, and hard-TTL
4,800 seconds. It requires a fresh immutable pushed source SHA, a fresh output
key, exact global concurrency limit 1, fail-closed admission, an armed
independent watchdog, teardown on every exit, cost settlement, and provider-zero
proof. No threshold is lowered and no observed output influenced the retry.
Judges remain forbidden.
