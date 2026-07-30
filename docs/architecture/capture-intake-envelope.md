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

## Completed-capture lifecycle

Each completed upload receipt records its immutable receive time. The signed
service exposes capture-scoped lifecycle apply and inspect operations for
consent revocation, an operator deletion request, or exact retention expiry.
The lifecycle binds the session, intake, capture digest, and envelope digest;
checks the declared revocation policy, retention deadline, and legal hold; then
writes a fail-closed marker before removing data.

Local deletion removes the exact intake payload, unshared raw object, and bound
task-discovery, reconstruction, testbed, and Task Evaluation Run work products.
A content-addressed object remains while another active intake references it.
The final non-sensitive tombstone retains hashes needed to explain prior
decisions without retaining the removed payload. Both the marker and tombstone
block re-upload reuse and all future reconstruction.

External provider deletion is a separate obligation. Local deletion records
which provider/result receipts require deletion, and a later operator record
can bind provider deletion receipt metadata. That metadata is not described as
independent provider verification. WebApp revocation and signed-download
disablement remain explicit required-but-unexecuted actions until their signed
cross-repository acknowledgements land.

This is the upload/admission seam, not a second router. Approved tasks still
flow into the existing Decision/Evidence Router, whose authorization and
qualification boundaries remain unchanged.

## Remaining work

The current implementation now hands admitted bytes to the separately versioned
`capture_qa_report.v1` boundary for decoded media/PTS checks and provenance-bound
quality observations. The separately versioned task-candidate contract now
requires digest-bound customer/operator approval before inferred intent can
reach the router. 360 native-container normalization, deployed cross-repository
lifecycle sync, and the real-capture vertical slice remain launch gates and must
not be inferred from intake admission or a local deletion tombstone.
