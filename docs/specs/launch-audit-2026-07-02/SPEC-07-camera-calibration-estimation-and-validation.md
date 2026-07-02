# SPEC-07: Camera calibration — estimate when missing, validate always, fail closed

- Status: Proposed
- Priority: **P1 — major**
- Area: `src/blueprint_pipeline/geometry_stage.py`, `geometry_sources.py`, `synthesis/cosmos_training_export.py`, `native_runtime_backend.py`
- Paper: OSCAR (arXiv 2606.04463) — calibration via MoGe-v2 (intrinsics) + CtRNet-X (extrinsics); "skeleton-RGB misalignment directly degrades model fidelity"

## Problem

OSCAR identifies camera calibration as the binding constraint on data quality — bad
intrinsics/extrinsics directly degrade action-conditioned generation, and data
availability was "bottlenecked by camera calibration annotation requirements." Our
handling is: guess, default, or fabricate; never estimate or validate.

1. **Guessed intrinsics in training export:** `synthesis/cosmos_training_export.py:48-59`
   defaults missing width/height to 640/480 and focal to `max(width,1)`/`max(height,1)`,
   principal point to image center — and feeds these into Plücker-ray conditioning maps
   (`:229-236`). A second hard-coded default lives at `native_runtime_backend.py:1786`
   (`fx=fy=960, cx=480, cy=270`).
2. **No plausibility validation:** `geometry_stage.py:1352-1357` checks only truthiness of
   `fx/fy/width/height`; `geometry_sources.py:57-83` parses ARKit intrinsics with no
   range/consistency checks. Nothing verifies focal-vs-FOV sanity, principal point within
   bounds, or extrinsic orthonormality.
3. **No estimation path:** MoGe-v2 / CtRNet-X (or any equivalent) do not exist in the
   repo, so captures with missing calibration can only be faked (see also SPEC-01's
   fabricated `fx=max(w,h)`).

## Why this matters for launch

Every downstream product consumes calibration: geometry, retrieval, Plücker conditioning,
skeleton overlay alignment (OSCAR adapter), eval rendering. Guessed focals produce
systematically distorted conditioning that a buyer's fine-tune inherits invisibly. This
is a top-3 data-quality lever per the reference paper.

## Proposed fix

1. **Fail closed on missing calibration in export paths:** remove the 640×480/focal
   defaults from `cosmos_training_export` and the 960/480/270 constant from
   `native_runtime_backend`; a record without validated intrinsics is skipped and logged
   (rejection manifest, per SPEC-02/04 pattern).
2. **Add a calibration validation gate** (shared helper, used by geometry stage +
   exporters):
   - fx, fy within [0.3, 3.0] × max(width, height) (configurable)
   - principal point within image bounds (with tolerance)
   - fx/fy aspect consistency vs pixel aspect
   - pose rotations orthonormal within tolerance; translations within site bounds
   - record a `calibration_validation` block (pass/fail per check) in `geometry_summary.json`
3. **Add an estimation lane for calibration-less captures** (video-to-world, glasses):
   integrate a monocular-geometry intrinsics estimator (MoGe-v2 per the paper) behind a
   swappable provider interface, marked `estimated: true` with the estimator name/version
   in provenance — never silently substituting for device-reported values when those exist.
4. ARKit-reported intrinsics remain authoritative when present and valid (truth
   hierarchy); estimation is only for missing calibration, validation applies to both.

## Acceptance criteria

- [ ] No code path exports or conditions on default/guessed intrinsics; regression tests assert removal of the three constant-default sites.
- [ ] Implausible fixture intrinsics (fx=8×width, cx outside image) fail the gate with typed reasons.
- [ ] Calibration-less fixture capture routes through the estimator and its outputs are marked `estimated: true` with provider provenance.
- [ ] `geometry_summary.json` carries the per-check validation block for every processed capture.
