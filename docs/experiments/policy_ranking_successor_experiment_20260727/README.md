# Policy-ranking successor experiment — 2026-07-27

Status: `paid_infrastructure_failure_closed_v4_repair_pending`

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
instance-bound hard-TTL watchdog has first written armed evidence. One paid infrastructure retry
is authorized under the unchanged USD 6 ceiling, with no additional spending authority.

## Proof boundary

This namespace currently proves implementation, provider allocation/teardown, and a closed
infrastructure failure only. It does not prove successful model runtime, generated media, WAM
causal validity, evaluator validity, simulator
outcomes, ranking fidelity, captured-site portability, warehouse portability, economics versus
physical evaluation, or physical performance.
