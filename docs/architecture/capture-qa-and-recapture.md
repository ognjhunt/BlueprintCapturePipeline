# Capture QA and Targeted Recapture

Status: implementation in progress, version 1 (2026-07-29)

## Decision

`capture_qa_report.v1` is the deterministic QA result above an immutable
`capture_intake_envelope.v1`. The executable boundary is
`blueprint-qa-capture`. It re-verifies every declared byte before assessment,
independently decodes media metadata and video timestamps with `ffprobe`, and
binds every supplied quality-observation packet to the exact intake ID and
source-file SHA-256.

The QA result is not a reconstruction, qualification record, or Task
Evaluation Run decision. It may accept the capture for its named authority
profile, remain `analysis_required` in the `validating` state, request bounded
recapture, or reject the intake. It cannot upgrade the intake claim ceiling.

## Evidence sources

The report distinguishes:

- decoded-media observations from the local `ffprobe` invocation;
- streams declared by the immutable Capture Intake Envelope;
- `capture_quality_observations.v1` measurements emitted by capture sidecars,
  a versioned local analyzer, or an identified operator review.

An observations packet from another intake or another source-file digest fails
closed. Missing measurements remain `not_measured`; they are never converted
to passes. Missing blur, exposure, overlap, compression, or rolling-shutter
measurements keep the capture in `validating` rather than accepting it. The
report returns the cheapest local measurement or operator review
when the capture is otherwise acceptable but a later claim could need that
evidence. Operator-attested packets may record bounded boolean review findings,
but cannot supply quantified blur, exposure, overlap, compression, or
rolling-shutter fractions.

## Deterministic checks

The current executable contract checks:

- decodability, duration, resolution, codec, frame rate, and rotation metadata;
- strict decoded-PTS monotonicity and bounded gaps;
- sharp-frame, well-exposed-frame, visual-overlap, compression-quality, and
  rolling-shutter-symptom fractions when a bound quality-observation packet is
  supplied;
- privacy-sensitive content, dynamic people, moving task objects,
  task-critical occlusion, and robot-placement-area coverage when measured;
- pose, intrinsics, depth, and scale-anchor evidence without assuming that a
  video contains those streams.

Failures generate stable codes and exact instructions, including slower
overlapping passes, privacy-safe recapture, an orbit around occluded task
objects, full robot-placement-area coverage, or a calibration-board capture.
The response never uses a generic `bad scan` outcome.

## Proof boundary

A 360 or monocular capture may pass media QA while retaining a non-metric claim
ceiling. Missing scale, poses, intrinsics, depth, collision geometry, or task
coverage remains missing evidence. The report always keeps physical success,
deployment readiness, safety certification, and general policy-ranking
validity prohibited. The frozen policy-ranking verdict is
`thesis_not_supported`.

## Remaining work

The Pipeline still needs a checked-in local quality analyzer that emits the
bound observations packet from real frames, native INSV normalization,
task-aware spatial coverage measurement, privacy-review workflow and redacted
derivative integration, persistence through the hosted state machine, and the
rights-cleared real-capture proof. Therefore this contract and its hermetic
tests do not yet prove the capture-admission launch gate.
