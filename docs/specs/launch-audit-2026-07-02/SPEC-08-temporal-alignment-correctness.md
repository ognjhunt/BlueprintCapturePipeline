# SPEC-08: Temporal alignment correctness — single time base, canonical frame ids

> [!WARNING]
> **SUPERSEDED FOR CURRENT LAUNCH STATUS.** This file is historical evidence, not a current completion or launch decision.
> Use the [current 107-gap ledger](/docs/public_launch_sc3_quality_gap_ledger_2026-07-09.json) and the [July 9 source audit](/docs/PUBLIC_LAUNCH_SC3_QUALITY_GAP_AUDIT_2026-07-09.md). Do not infer current status from “proposed,” “implemented,” or “fixed” wording below.

- Status: Proposed
- Priority: **P1 — major**
- Area: `src/blueprint_pipeline/materialization.py`, `geometry_stage.py`, `geometry_sources.py`, `frame_alignment_stage.py`
- Papers: both OSCAR and SC3-Eval assume tight frame/pose/action temporal alignment; SC3 additionally requires per-chunk alignment.

## Problem

Frame↔pose↔depth joins tolerate mixed time bases and inconsistent frame-id encodings,
which can silently mis-pair data:

1. **Mixed time bases:** `materialization.py:154-162` (`_time_value`) picks the first
   present of `t_device_sec`, `tCaptureSec`, `timestamp` — three fields that can carry
   different clocks/epochs/units. If frames carry `timestamp` and poses carry
   `t_device_sec`, nearest-neighbor matching (`_nearest_pose_time`, `:182-200`) still
   returns a "match" across incompatible clocks. Nothing asserts the two sides share a
   time base or unit.
2. **Misleading match statistics:** a frame matched by `frame_id` with no `frame_time`
   increments `matched` but contributes no delta sample (`:230-248`), so
   `pose_match_rate` can look high while the p95-delta distribution is empty or
   unrepresentative. The p95 gate protects the iPhone lane only.
3. **Frame-id normalization is inconsistent:** `_normalized_frame_id`
   (`materialization.py:143-151`) returns truthy numeric ids as unpadded strings (`"5"`),
   but the empty-value branch generates `str(max(0, index) + 1).zfill(6)` — a silent +1
   offset and 6-digit pad. Geometry/site records use 6-digit padded ids
   (`geometry_stage.py:44`, `geometry_sources.py:31-40`). Joining `"5"` against
   `"000005"`/`"000006"` misses, pushing records onto the fragile time-based fallback or
   dropping them.
4. **No dropped-frame accounting:** there is no explicit ledger of frames without poses /
   poses without frames per bundle.

## Why this matters for launch

A one-frame pose offset is exactly the "skeleton-RGB misalignment" failure mode OSCAR
calls out as directly degrading model fidelity — and it's undetectable downstream. For
SC3-style eval, per-chunk action/frame misalignment corrupts the consistency signal
itself. These bugs poison data quietly at scale.

## Proposed fix

1. **Declare and enforce a canonical time base per bundle:** intake normalizes all
   sources to one field (`t_device_sec`, float seconds, monotonic per session) and
   records the source field + conversion per stream in the bundle manifest. Materialization
   refuses to join streams whose declared time bases differ (typed blocker, not fallback).
2. **One canonical frame-id normalizer** shared by frames, poses, depth, geometry, and
   site index (module-level helper in `common.py`): always 6-digit zero-padded, no
   implicit +1, with property tests round-tripping every producer's id format.
3. **Honest match statistics:** `matched` counts only pairs with measurable deltas;
   id-only matches are reported separately (`matched_by_id_only`); p95 gate applies to
   all lanes, not just iPhone.
4. **Dropped-frame ledger:** per-bundle counts of unmatched frames/poses/depth with
   reasons, included in the QA notes of downstream packages.

## Acceptance criteria

- [ ] A fixture bundle with `timestamp`(ms) frames + `t_device_sec`(s) poses is rejected with a time-base blocker instead of silently matching.
- [ ] Property test: frame ids from all producers normalize to identical canonical ids; the +1 offset path is gone.
- [ ] `pose_match_rate` and p95 stats computed on measurable pairs only, across all capture lanes.
- [ ] Package QA notes include the dropped-frame ledger.
