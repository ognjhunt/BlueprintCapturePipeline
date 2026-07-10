# SPEC-04: Action normalization & validation for SC3-style evaluation

> [!WARNING]
> **SUPERSEDED FOR CURRENT LAUNCH STATUS.** This file is historical evidence, not a current completion or launch decision.
> Use the [current 107-gap ledger](/docs/public_launch_sc3_quality_gap_ledger_2026-07-09.json) and the [July 9 source audit](/docs/PUBLIC_LAUNCH_SC3_QUALITY_GAP_AUDIT_2026-07-09.md). Do not infer current status from “proposed,” “implemented,” or “fixed” wording below.

- Status: Proposed
- Priority: **P0 — launch blocker** (for Task Evaluation Run correctness)
- Area: `src/blueprint_pipeline/evaluation_prep_stage.py`, `sc3_eval_protocol.py`, `synthesis/cosmos_training_export.py`, `episode_spec.py`
- Paper: SC3-Eval (arXiv 2606.18610) §training data

## Problem

SC3-Eval requires 7-dimensional delta end-effector actions **normalized per-dimension
across the training corpus**, with per-chunk temporal alignment. We declare this contract
but never enforce or compute it:

1. `evaluation_prep_stage.py:311-394` defines `action_space` with `dim: 7` labels for
   three robot profiles — but these are static catalog descriptors. No per-dimension
   normalization statistics are ever computed or persisted.
2. `sc3_eval_protocol.py:307-329` marks `action_chunks` "reviewable" purely on the
   presence of a selected modality; it never checks that actions are 7-D delta-EE,
   normalized, or unit-consistent. The paper facts are recorded
   (`sc3_eval_protocol.py:34-44`) but not enforced on any real array.
3. Missing data is silently substituted instead of rejected:
   - missing/misshaped `T_world_camera` → `np.eye(4)` exported as a training target
     (`synthesis/cosmos_training_export.py:224-228`)
   - empty pointcloud back-filled with `(0,0,0)` (`geometry_stage.py:625`)
   - spawn/goal poses default to floor-frame zeros (`episode_spec.py:737-744`, `:90`)

   None of these substitutions carry a blocking flag into the exported artifact.

## Why this blocks beta

Un-normalized or unit-inconsistent actions are the classic silent killer for
action-conditioned world models: the model trains/evaluates on garbage scales and the
resulting success predictions and rank correlations are meaningless — while looking
plausible. Identity-pose and zero-fill substitution is worse: it fabricates training
targets. Task Evaluation Runs sold on these artifacts would be wrong in ways buyers can't
detect.

## Proposed fix

1. **Action normalization stage**: when episodes with real action streams enter
   evaluation prep or training export, compute per-dimension mean/std (or
   min/max, configurable) across the corpus, persist statistics alongside the package
   (`action_norm_stats.json`), and store both raw and normalized streams. Never normalize
   in place over the raw stream (capture truth stays authoritative).
2. **Validation gates** at eval/export intake:
   - dimensionality check against the robot profile's declared `action_space`
   - unit/scale sanity bounds (e.g. |Δtranslation| per step below a physical max;
     rotation deltas in expected representation; gripper width within limits)
   - per-chunk timestamp alignment between actions and frames (ties into SPEC-08)
3. **No silent substitution**: a record missing pose/geometry/action data is *skipped*
   and logged to a rejection manifest, or the whole export is blocked — never
   identity/zero-filled. Remove the `np.eye(4)` and zero-fill fallbacks; if a permissive
   dev mode is needed, gate it behind an explicit env and stamp the artifact.
4. Update `sc3_eval_protocol` readiness to require the normalization statistics artifact
   and passing validation gates before `action_chunks` can be marked reviewable.

## Acceptance criteria

- [ ] Exported eval/training packages containing action streams include per-dimension normalization statistics with corpus provenance.
- [ ] A fixture with out-of-range action deltas or wrong dimensionality is rejected with a typed reason.
- [ ] No code path exports `np.eye(4)`/zero-filled poses or `(0,0,0)` pointcloud placeholders; regression tests assert the substitution paths are gone.
- [ ] `sc3_eval_protocol` reports `action_chunks` unready when normalization stats or validation results are missing.
