# Protocol Amendment 19: Vast terminal-instance polling and WAM2 retry

Status: prospectively frozen before the WAM2 provider retry

Date: 2026-08-01

## Preserved provider-startup failure

The single-use allocation 4 attempt launched Vast instance `46487397` for the
immutable WAM2 bundle at repository SHA
`1b0379cb6f80b562c2da7d4cb92567d812454a2c`. The provider reported the
instance `exited` before any container log or Ctrl-World result became
available. The output object remained the 22-byte empty ZIP placeholder. The
controller continued polling logs because it did not re-check instance state
after the initial transition to `running`; it was interrupted only after
separate read-only provider evidence proved the instance terminal and no
continuing spend. Its `finally` path completed teardown, and the independent
watchdog subsequently recorded two provider-absence confirmations.

This is a provider-startup and controller-observability failure, not a WAM
generation, reliability, causal-qualification, or scientific result. It earns
no interaction credit. Allocation 4 is consumed and may never be reused.

The preserved evidence under external live-attempt directory
`ctrl_world_current_reference_wam_2_live_1b0379cb_20260801_v2` includes:

- `adapter_output.json` SHA-256
  `ed1dd1a717da12931996387b877746942262a2af2cc2577fbdb4ad79d9f476a4`;
- `production_campaign_budget_settlement.json` SHA-256
  `820f547d920b4a98cb93865ab43cf953c917680211c739e7f4da7c7fe7705c69`;
- `provider_zero_after_failed_startup_v1.json` SHA-256
  `30149ebe0bc47cdfbc02474e746f1c142e7f74d78526ddddeecb6a2833f88a8a`;
- `job/vast_provider_adapter_result.json` SHA-256
  `4e57a7cd7f772063d88b8d718760e5265bb89d7e0c621380789ff64bd22cc648`;
- `job/vast_teardown_manifest.json` SHA-256
  `afd40f6b542cd0c807e5fccc1f92b73911e40ae315dce30b0989daf7a8a5a206`;
- `job/independent_vast_watchdog/groot_oscar_runpod_canary_watchdog.json`
  SHA-256
  `e78a6c255241a243c8df63e1cb9aac70c0e3e20c0ba2f0779873a87837f1ce4c`;
- `provider_output_placeholder_22b.zip` SHA-256
  `8739c76e681f900923b900c9df0ef75cf421d39cabb54650c4b9ad19b6a76d85`.

The provider session observed `1347.914445` seconds and estimated USD
`0.650105`. The campaign ledger conservatively charged 1,348 GPU seconds and
USD `0.650105`, bringing committed campaign GPU use to 32,971 seconds and USD
`9.689470`. The prior evaluator/API ledger remains USD `8.418512`; the combined
committed campaign spend before retry is therefore USD `18.107982`.

## Frozen generic controller correction

Every Vast log-poll iteration now performs a read-only instance-status probe
until the registered success marker appears. A provider status in the existing
terminal set causes an immediate `terminal_instance_status_observed` exit and
records the exact status in the attempt and adapter evidence. Probe errors are
recorded but do not fabricate success. This correction is generic provider
lifecycle handling and does not change a policy, action, WAM, input frame,
seed, threshold, or generated output.

Focused regression tests must prove terminal-state exit, no-progress timeout,
missing-container handling, transient missing-container recovery, dud-log
flicker handling, and container-missing retry behavior before publication and
paid retry.

## Scientifically identical allocation-5 retry

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

Allocation 5 is single-use, one-GPU, target USD 3, maximum USD 5, and hard-TTL
4,800 seconds. It requires a fresh immutable pushed source SHA, a fresh output
key, fail-closed admission, an armed independent watchdog, teardown on every
exit, cost settlement, and provider-zero proof. No threshold is lowered and no
observed output influenced the retry. Judges remain forbidden.
