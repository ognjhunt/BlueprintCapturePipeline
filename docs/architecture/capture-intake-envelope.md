# Capture Intake Envelope and Authority Admission

Status: implementation in progress, version 1 (2026-07-29)

## Decision

One capture means one customer upload action. It may contain several synchronized
files. `capture_intake_envelope.v1` describes that logical input without
replacing Capture Raw Contract V3/V3.1 or treating a video/reconstruction as
sensor evidence it does not contain.

The supported authority profiles are:

- `iphone_arkit_lidar`
- `iphone_arkit_non_lidar`
- `camera_360_equirectangular`
- `camera_360_native`
- `monocular_video`
- `precomputed_external_reconstruction`

The executable boundary is `blueprint-intake-capture`. It verifies regular
files beneath the declared upload root, rejects traversal, symlinks, unsupported
types, size mismatches, and digest mismatches, then stores immutable
content-addressed objects. An idempotency key cannot be rebound to different
bytes or metadata.

## Admission

`capture_intake_admission.v1` recomputes rights, consent, privacy, retention,
revocation, provider, upload, and malware/content gates. Missing profile evidence
returns profile-specific recapture instructions. It never returns a generic
"bad scan" result.

The admission claim ceiling is deliberately conservative:

- a complete iPhone ARKit/LiDAR intake may establish calibrated poses and
  initial metric geometry authority;
- non-LiDAR iPhone metric authority additionally requires a verified scale anchor;
- 360 and monocular video support observation review and task discovery, but do
  not inherently establish metric scale, camera poses, depth, or collision truth;
- external reconstructions remain derived and source-digest-bound;
- no intake profile establishes collision/contact physics, policy-ranking
  validity, physical success, deployment readiness, or safety certification;
- comparative policy ranking remains `thesis_not_supported`.

## Existing pipeline integration

When `raw/capture_intake_envelope.json` is present, capture materialization
re-verifies the referenced bytes and recomputes admission. A supplied admission
must match exactly. Only `accepted` intake reaches descriptor/QA materialization.
The resulting descriptor carries the intake digest, profile, reduced-authority
reasons, claim ceiling, and byte-verification count. Existing Raw Contract
verification remains independently required.

This is the upload/admission seam, not a second router. Approved tasks still
flow into the existing Decision/Evidence Router, whose authorization and
qualification boundaries remain unchanged.

## Remaining work

The current implementation now hands admitted bytes to the separately versioned
`capture_qa_report.v1` boundary for decoded media/PTS checks and provenance-bound
quality observations. It does not yet provide Pipeline handoff from the WebApp
resumable upload session, 360 native-container normalization, task-candidate
approval, reconstruction planning, testbed
compilation, or the hosted WebApp state machine. Those remain launch gates and
must not be inferred from intake admission.
