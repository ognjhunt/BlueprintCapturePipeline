# Protocol Amendment 33: Policy query 6 Vast startup retry

Status: frozen after the failed query-6 provider attempt and before any retry
provider mutation

Date: 2026-08-01

## Preserved failed attempt

The first policy-query-6 allocation used the prospectively frozen query-6 input
and immutable runtime SHA `ede38013d6cb2a5453ed39ba39c607a7f497a639`.
Vast created instance `46513397` on machine `27268` with an L40 GPU, but the
instance remained in provider `loading` state and exposed no ready runtime
endpoint before the frozen 480-second startup timeout. No policy output archive
was created or observed. This is an infrastructure failure before policy
inference, not a failure of `pi05_droid`, WAM5, or the closed-loop scientific
contract.

The owned instance was destroyed. The pending-teardown record closed, the
watchdog reached `provider_terminal`, and both the exact-prefix and fresh global
Vast inventories were observed at zero. The production ledger settled 493 GPU
seconds and USD `0.102708`. Cumulative conservative GPU accounting is therefore
37,536 seconds and USD `10.983857`; unchanged evaluator/API spend is USD
`8.418512`, for combined GPU plus evaluator/API spend of USD `19.402369`.

Preserved failed-attempt evidence includes:

- adapter SHA-256
  `85569fba70b76d2a16d3a85243234419238fd59d7b4c063543333b8a48c0219b`;
- monitor SHA-256
  `07896cab43487aaa92a7b01331fc0f34772fbe5432447adba8a527f9df68bb64`;
- terminal watchdog SHA-256
  `985526869c0c19aa895ce79ada086d678fe285df98d1af59bda8ed53e25ab606`;
- settled production ledger SHA-256
  `a58e1748602ba4f33df630004ccb0ec0c9779b94444c179d87c4128dba9b6222`;
  and
- admission SHA-256
  `0d33af10c14c2c30a53a3c33169698596cd0d15cdd6048301fb7f5fb41704d64`.

The consumed authorization remains consumed. The failed evidence namespace is
immutable and is not reused for the retry.

## Generic retry control

The model-neutral Vast provider previously had no way to carry a known failed
machine exclusion from current-reference authorization through the fresh
capacity preflight and actual offer selection. Runtime SHA
`f5b3e99ae278a1b5fc0962d2fa07e094a5519351` adds that generic binding and
validates the exclusion list fail closed. Both the mutation-free capacity
snapshot and provider launch now apply the same exclusion. A focused 157-test
provider, admission, authorization, and campaign regression set passes.

The current official source archive is frozen at SHA-256
`1a800e9b804d3bae4c4cd6bfd4ac8a7e0553482bdd7d43e41d091aab0d5baf97`;
its audit file is SHA-256
`cba8d79664da3508750d9198e673fb9671387d588f626bfa72ddea95b783ad85`.

## Retry input and boundary

The WAM5 camera images, registered commanded state, policy identity,
checkpoint, task text, and query index are unchanged. Only the immutable
runtime source binding changes to carry the provider exclusion. The new input
archive SHA-256 is
`29e7bbac2aef2d2fb5657ed80065cbf374dfa005b848a4e3ab7c61045ba28bb4`;
its receipt file SHA-256 is
`5437029ae60961147b86d11e3463db4d6da3ff04894a16c883d57b516c5fbfac`,
and its independent extraction receipt SHA-256 is
`b5f585f999772aaf301aabc0c41fcaff07bcab6df6bb01b9b952e86ef7792a14`.

One retry allocation is authorized, still using one GPU under the two-GPU
campaign ceiling. Vast machine `27268` is excluded prospectively. The retry
must use a new provider name, object keys, live evidence directory, and atomic
single-use authorization. It remains subject to fresh credential presence,
global inventory, signed transport, output-absence, provider preflight,
campaign reservation, watchdog, dry-run, TTL, teardown, settlement, and
provider-zero gates.

No evaluator or VLM call is authorized. A successful retry would establish
only the sixth same-policy interaction. It would not establish a complete
episode, WAM causal validity, policy ranking, physical agreement, captured-site
transfer, or economic superiority.
