# SPEC-06: SC3 forward/inverse consistency scorer + test-time uncertainty gating

- Status: Proposed
- Priority: **P1 — major** (required before selling model-backed eval; P0 if SPEC-05's real backend ships)
- Area: `src/blueprint_pipeline/sc3_eval_protocol.py`, `wam_derived_observation_harness.py`, new scorer module
- Paper: SC3-Eval (arXiv 2606.18610) §core method

## Problem

SC3-Eval's central quality mechanism is threefold consistency: (a) joint forward+inverse
dynamics training, (b) cross-view inpainting consistency, and (c) a **test-time
uncertainty signal** — the inverse-dynamics mode recovers actions from generated frames,
compares them to commanded actions, and terminates the rollout when per-chunk consistency
error exceeds a threshold. This is what keeps generated rollouts anchored to physically
plausible outcomes and prevents drift from silently corrupting evaluations.

Our implementation is a pass-through contract, not a scorer:

1. `sc3_eval_protocol.py:1-7` is "intentionally declarative … without launching a model,
   computing correlations…" — it builds readiness JSON only.
2. `wam_derived_observation_harness.py:1441-1489` reads externally supplied booleans
   (`forward_consistent`, `inverse_consistent`, `visual_evidence_used`) and requires that
   an *external* scorer ran; no action recovery or comparison is computed anywhere.
   `docs/WAM_EPISODE_CONSISTENCY_SCORER.md:70-98` confirms the scorer is a
   bring-your-own external command.
3. The existing early-termination heuristic (`wam_derived_observation_harness.py:1627-1677`)
   is a perception-confidence blend plus keyword markers — unrelated to
   commanded-vs-recovered action error — and must not be presented as the SC3 signal.
4. Horizon decoupling (predict 24 / execute 16) exists only as recorded paper facts
   (`sc3_eval_protocol.py:42-45`), not as runtime behavior.

## Why this matters for launch

Without a consistency scorer, generated-rollout evaluations have no drift detection: a
rollout that diverged from physics still gets scored, and its `predicted_success` is
noise. SC3's ablations show the inverse-dynamics term matters most exactly
out-of-distribution (PSNR degradation 1.05 OOD vs 0.47 ID) — i.e., on the novel buyer
sites that are our core use case. It is also one of the five strategy-doc preconditions
for making Cosmos 3 the preferred candidate.

## Proposed fix

1. **Ship a first-party consistency scorer module** (`sc3_consistency_scorer.py`) behind
   the same command/adapter boundary as the WAM backends:
   - input: generated rollout frames + commanded action chunks (normalized per SPEC-04)
   - runs the backend's inverse-dynamics mode (adapter capability flag; the Cosmos3
     adapter from SPEC-05 exposes it) to recover actions per chunk
   - computes per-chunk consistency error (per-dimension normalized L2, configurable)
     and emits `consistency_error[]`, `threshold`, `terminated_early`, `terminated_at_chunk`
2. **Rollout gating:** when the scorer is available, the eval harness terminates/flags
   rollouts whose per-chunk error exceeds the threshold; scored results carry
   `consistency_gated: true/false` and the error trace in the artifact.
3. **Horizon decoupling as runtime config:** predict-N/execute-M becomes an enforced
   adapter parameter (defaults 24/16 per paper) rather than a recorded fact.
4. Keep the external-scorer contract as an alternative provider path, but readiness
   (`forward_inverse_consistency_proven`) should distinguish `first_party_scorer_ran`
   from `external_scorer_attested`.
5. Rename/annotate the existing perception-confidence heuristic so it cannot be confused
   with SC3 consistency (`perception_confidence_early_stop`, docstring cross-reference).

## Acceptance criteria

- [ ] Given a fixture rollout with injected action-inconsistent frames, the scorer terminates at the correct chunk and the artifact records the error trace.
- [ ] Eval artifacts distinguish consistency-gated vs ungated rollouts; ungated learned-model rollouts are flagged in claim boundaries.
- [ ] Horizon decoupling is enforced and recorded per run.
- [ ] `sc3_eval_protocol` readiness reflects first-party scorer availability.
