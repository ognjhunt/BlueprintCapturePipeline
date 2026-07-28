# Policy-ranking successor experiment — 2026-07-27

Status: `inconclusive_exact_main_campaign_closed_provider_zero`

This is a new experiment. Experiment 1 and Experiment 2 remain immutable historical
experiments; none of their prediction matrices, held-out partitions, outcomes, thresholds,
or verdicts are inputs to this arm.

The successor question is whether a replaceable general WAM can produce meaningfully
different future observations for different candidate-policy action trajectories, and whether
an independent frozen evaluator can recover useful policy ordering while abstaining on
unreliable or out-of-distribution cases.

The first embodiment is DROID-compatible Franka Panda. DROID is a calibration anchor, not a
permanent product boundary. The stable architecture is:

`initial observation + task instruction + normalized candidate actions -> predicted futures`

The WAM under test is the released general `nvidia/Cosmos3-Nano` checkpoint in forward-
dynamics mode. `nvidia/Cosmos3-Nano-Policy-DROID` is not the neutral WAM. The proposed
independent evaluator is `gpt-5-mini-2025-08-07`, but no evaluator call is admitted unless
Cosmos first passes causal qualification.

## Completed without paid resources

- Fresh isolated worktree from fetched `origin/main` at
  `115e9caec0755aa2fa0c47b84c92ead6a85ea074`.
- Exact Cosmos, Cosmos Framework, model-checkpoint, vLLM image, amd64 image, DROID action,
  dependency-lock, and license pins.
- Confirmation that the official path is action forward dynamics rather than generic preview
  generation.
- Fail-closed 16x10 DROID action semantics, 15 Hz timing, raw-action boundary, OpenCV
  conversion, multiview layout, deterministic control construction, policy-blind request IDs,
  artifact isolation, compute admission, and evaluator causal-validity contracts.
- Read-only Vast price/availability discovery and provider-zero check.
- A deterministic immutable provider bundle containing the exact public DROID observation,
  five frozen action streams, ten request identities, and a standalone pinned Cosmos3 runtime.
- Twenty-two focused synthetic/admission tests for action semantics, bundle integrity, the
  Blackwell hardware envelope, shared paid-resource admission, and allocator-only dispatch.

## Current state

The user explicitly changed the compute ceiling to USD 6.00 while retaining the single
RTX PRO 6000 Blackwell arm. Two execute requests first failed closed before a provider API
mutation: missing explicit Vast environment gates, then a missing shared crash-fallback bundle
contract. The repaired v3 request allocated one Blackwell instance through the shared allocator.
Vast exposed the contract as running while its cold image container was still absent; the
synchronous WAM path allowed only two missing-container polls, then destroyed the instance after
97.49 seconds. Estimated compute spend is USD 0.027521, provider-zero is independently verified,
and API/VLM spend remains zero. No scientific rollout or WAM output was produced.

The v3 evidence also exposed that its authorization required a detached watchdog armed before
allocation while only the adapter's in-process teardown was active. Teardown succeeded, but that
requirement is recorded as not met. The v4 repair reuses the bounded cold-pull policy already used
by asynchronous WAM runs and blocks every successor create call unless a detached name- and
instance-bound hard-TTL watchdog has first written armed evidence. V4 reached that repaired path,
but the downstream session guard rejected the new three-hour resource reservation because the
same three-hour value was also used as the aggregate limit and the ledger already contained 97.49
seconds from v3. The failure occurred before offer search or create, added zero spend, and provider
zero was reverified. V5 moves the same ledger check ahead of authorization consumption, reserves a
full per-resource TTL on top of attributable prior runtime, and explicitly cancels the detached
watchdog when the adapter proves no create was attempted. This is a zero-spend replacement under
the unchanged USD 6 ceiling, not additional spending authority.

## Proof boundary

This namespace currently proves implementation, provider allocation/teardown, and a closed
infrastructure failure only. It does not prove successful model runtime, generated media, WAM
causal validity, evaluator validity, simulator
outcomes, ranking fidelity, captured-site portability, warehouse portability, economics versus
physical evaluation, or physical performance.

## Exact-main campaign closeout

The final exact-main GPU attempt ran source commit
`fd352e59bd1342b13ab3b83f45a52c6af23ff28b` after PR #209 merged and its
required CI checks passed. Blackwell CUDA 13.0, compute capability `(12, 0)`,
BF16 Torch execution, the pinned vLLM/vLLM-Omni imports, and the pinned
`nvidia/Cosmos3-Nano` server all passed. Model load took 335.13 seconds.

The first direct forward-dynamics request produced a nonempty, decodable
17-frame H.264 clip. It did not pass the frozen canary: a `640x540` request
returned `640x528`, and independent re-analysis classified the clip as static.
The pinned vLLM-Omni source floors the 540-pixel action-video height by its
16-pixel spatial VAE factor. The result remains a failure; no threshold or
expected dimension was changed after inspection.

The same-process Blueprint wrapper, ten-arm causal matrix, GPT-5 mini evaluator,
disjoint benchmark ranking, and captured-site transfer were correctly not
entered. No benchmark outcomes were unsealed. The campaign therefore closes as
`inconclusive`, not as evidence of useful policy ranking and not as physical,
simulator, or captured-site success.

Three paid provider allocations used 791.277 seconds and an estimated USD
0.226920. Evaluator spend was zero. The final authenticated provider inventory
reported zero live instances and zero continuing hourly burn. See
`final_verdict.json` and `exact_main_attempt_v11/artifact_manifest.json`.
