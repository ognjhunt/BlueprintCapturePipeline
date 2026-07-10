# SPEC-02: OSCAR-grade clip curation filters

> [!WARNING]
> **SUPERSEDED FOR CURRENT LAUNCH STATUS.** This file is historical evidence, not a current completion or launch decision.
> Use the [current 107-gap ledger](/docs/public_launch_sc3_quality_gap_ledger_2026-07-09.json) and the [July 9 source audit](/docs/PUBLIC_LAUNCH_SC3_QUALITY_GAP_AUDIT_2026-07-09.md). Do not infer current status from “proposed,” “implemented,” or “fixed” wording below.

- Status: Proposed
- Priority: **P0 — launch blocker** (for Post-Training Data Package quality)
- Area: `src/blueprint_pipeline/retrieval_index_stage.py`, `geometry_stage.py`, `geometry_sources.py`, new curation stage
- Paper: OSCAR (arXiv 2606.04463) §data pipeline

## Problem

OSCAR's result quality depends on aggressive curation: 2.16M source episodes were reduced
to 180.7K clips using four mechanism-specific filters (≥70 frames per clip, static-camera
constraint, non-trivial manipulator actions, skeleton-visibility threshold). Our pipeline
sells Post-Training Data Packages built on the same class of model, but ships almost none
of these filters:

1. The only frame gates are per-frame ARKit heuristics — `_MIN_TRAVEL_M=0.07`,
   `_MAX_GAP_SEC=0.5`, `_MIN_SHARPNESS=40.0` (`retrieval_index_stage.py:56-60`, `:952-980`).
   There is **no minimum clip length**, no static-camera/camera-stability constraint on
   clips, no non-trivial-action check, and no visibility threshold anywhere in the package.
2. The sharpness/blur gate only works on the ARKit lane. The geometry/video lanes stamp
   constants — `"sharpness_score": 100.0` for every frame (`geometry_stage.py:1190`,
   `geometry_sources.py:208`) and `"blur_score": 0.0` (`geometry_stage.py:679`,
   `video_to_world_service_runtime.py:181`) — so the blur filter is a no-op for
   glasses/android/video-to-world captures.
3. There is no exposure filter at all (over/under-exposed frames pass).

Result: arbitrarily short, shaky, blurry, or content-free clips can flow into curated
packages and eval artifacts unfiltered.

## Why this blocks beta

Data quality *is* the product. A buyer receiving 40-frame, motion-blurred, or
static-content clips in a "curated" Post-Training Data Package will churn, and any world
model fine-tuned on our packages will underperform for reasons directly attributable to
missing curation. OSCAR demonstrates the filters are load-bearing (91.6% of source data
rejected).

## Proposed fix

Add a `clip_curation_stage` that runs after materialization and before retrieval/package
export, with per-clip gates:

1. **Minimum clip length**: configurable floor (default 70 frames, per OSCAR), measured
   post-alignment.
2. **Camera-stability gate**: per-clip pose-jitter / optical-flow variance threshold.
   For robot-POV eval clips, enforce the static-camera constraint OSCAR uses for
   world-model conditioning data; for walkthrough capture, use a motion-smoothness bound
   instead of rejecting motion outright.
3. **Non-trivial content/action gate**: for manipulation episodes, require action delta
   above a floor (ties into SPEC-04); for site walkthroughs, require scene-coverage
   novelty (pose travel + view-direction diversity, extending the existing pan dedup).
4. **Real blur measurement everywhere**: compute Laplacian-variance sharpness per
   extracted frame in the geometry/video lanes instead of stamping `100.0`/`0.0`; delete
   the constant defaults so a missing measurement fails the gate rather than passing it.
5. **Exposure gate**: luminance-histogram check (reject clipped/crushed frames beyond a
   percentage threshold).
6. Emit a **rejection manifest** per bundle (counts + reasons per gate) so curation is
   auditable and rejection rates can be tracked against OSCAR's reference behavior.

All thresholds land in a config file (not code constants) so ops can tune per capture
modality.

## Acceptance criteria

- [ ] Clips under the frame floor never reach retrieval/package/eval export.
- [ ] Geometry/video-lane frames carry measured sharpness; a unit test asserts no code path emits the constant `100.0`/`0.0` scores.
- [ ] Over/under-exposed synthetic fixtures are rejected by the exposure gate in tests.
- [ ] Every curated package manifest includes the rejection manifest (gate → count).
- [ ] Thresholds are config-driven with documented defaults referencing OSCAR values.
