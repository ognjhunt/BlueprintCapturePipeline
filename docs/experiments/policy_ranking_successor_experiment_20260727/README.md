# Policy-ranking successor experiment — 2026-07-27

Status: `compute_authorized_prelaunch_admission_and_protected_main_pending`

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
RTX PRO 6000 Blackwell arm. The first execute request failed closed before a provider API
mutation because two explicit Vast environment gates were absent; no resource was allocated
and compute/API/VLM spend remained zero. One infrastructure retry is authorized after the
gate-order repair merges through protected main. No model weights have been downloaded, no WAM
output has been generated, and no API/VLM request has been sent.

## Proof boundary

This namespace currently proves implementation and zero-cost admission preparation only. It
does not prove runtime, generated media, WAM causal validity, evaluator validity, simulator
outcomes, ranking fidelity, captured-site portability, warehouse portability, economics versus
physical evaluation, or physical performance.
