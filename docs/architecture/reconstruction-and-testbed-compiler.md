# Reconstruction Capability and Testbed Compiler

Status: implemented local contract boundary, deterministic frame/split kernel,
ARKitScenes raw proxy compiler, and native dual-fisheye normalization, version 1
(2026-07-30)

## Decision

Reconstruction is a replaceable capability graph beneath a Task Evaluation Run.
It is not a product and it does not always produce a 3D Gaussian splat. The
planner derives the representations required by the requested claim types and
selects the lowest-total-cost set of authorized, applicable method profiles.
Provider availability is never treated as qualification.

`reconstruction_method_profile.v1` names the exact method, version,
implementation digest, provider identity, execution mode, outputs, capture and
claim-ceiling prerequisites, authorization, qualification status, expected
cost, rights/provider constraints, and failure modes. Supported method kinds
include pose/SfM, metric scaffolds, depth fusion, photogrammetry, 3DGS, semantic
graphs, segmentation, structural priors, collision proxies, articulated assets,
USD composition, generated visual completion, and owner-attested correction.

`reconstruction_plan.v1` binds the exact intake and capture digest. It records
required representations, selected methods, rejected candidates, missing
representations, total expected cost, and the next cheapest experiment. A
generated visual completion cannot satisfy metric, collision, physics, or
articulation outputs.

## Local reconstruction control plane

The signed service exposes plan, authorize, execute, and inspect operations
under `/api/live-pipeline/reconstructions`. Planning resolves the immutable
Pipeline-owned upload receipt, revalidates the intake envelope, QA digest,
object-manifest digest, raw-object size, and raw-object SHA-256, then records the
exact adapter reference beside each selected method-profile digest. The client
cannot provide a filesystem path, command, credential, or executor.

Authorization is a separate immutable record bound to the plan and source
context digests. The registry is empty unless the operator names an exact
planned hermetic adapter. The initial executable adapters are:

- `local://decoded-observation-index-v1`, which indexes decoded PTS and extracts
  deterministic retained-video observations without claiming calibration,
  scale, geometry, collision, physics, or physical success;
- `local://arkit-metric-scaffold-v1`, which accepts only a complete Capture Raw
  Contract V3.2 LiDAR bundle whose decoded PTS, encoder retention, AR poses,
  intrinsics, coordinate semantics, and depth/confidence pairs agree;
- `local://native-360-normalization-v1`, which preserves original `.insv`
  bytes, consumes only digest-bound probe receipts and declared stream
  bindings, validates per-lens calibration and fixed rig extrinsics, and
  refuses calibrated-rig claims when lens timing, dimensions, provenance,
  rights, consent, privacy, or retention authority are invalid.

A single MP4 uploaded with an iPhone profile does not activate the bundle
adapter. Missing authority produces a partial plan, abstention, or next
experiment. Execution is local, zero-paid-compute, deterministic, retry-safe,
and stores a normalized immutable result. Adapter failure is evidence
insufficiency; it never becomes a pass.

Before this control plane is invoked, the Task Evaluation Capture and Testbed
Supervisor may emit `task_evaluation_capture_reconstruction_route.v1`. That
artifact separates iPhone ARKit/LiDAR, non-LiDAR ARKit, equirectangular 360,
native 360, monocular video, and external-reconstruction import sequences. It
also names stages that are required but not yet registered, including
spherical projection, SfM, 3DGS training, and independent scale validation.
Native 360 normalization is registered conditionally: it appears as an SDK
tool only when the trusted capture runtime injects a digest-bound normalizer.
This remains a planning aid: the control plane still owns method-profile
eligibility and separate execution authorization.

## Deterministic frame dataset and hidden split

`reconstruction_frame_dataset.py` implements the first execution kernel shared
by reconstruction methods. The decoded-observation adapter uses actual decoded
PTS rather than file order or nominal frame rate, selects bounded observations
with a versioned timeline-spacing rule, and records DTS, duration, key-frame,
rotation, pixel/color, and locally measurable image-quality metadata when those
values are available. Duplicate or reordered PTS, undecodable media, symlinked
inputs, unsafe paths, digest changes, and oversized retained video fail closed.
The original retained video remains complete and authoritative.

The kernel emits five versioned, digest-bound artifacts governed by
`docs/schemas/reconstruction_frame_dataset.v1.schema.json`:

- `reconstruction_dataset_manifest.v1`;
- `retained_frame_selection_manifest.v1`;
- `frozen_reconstruction_split_manifest.v1`;
- `candidate_reconstruction_dataset_manifest.v1`;
- `hidden_heldout_evaluator_manifest.v1`.

Training and validation pixels are materialized under a candidate-only root.
Hidden held-out pixels are materialized under a separate evaluator-only root,
and neither the candidate manifest nor the registered candidate tool result
contains those paths. Split assignments are digest-ranked, immutable, and
bound to the exact source video, actual frame timing and digests, stream
metadata, runtime, source commit, authority, and parent artifact. Identical
inputs replay the accepted artifacts; altered inputs select a different
content-addressed dataset. Fewer than three selected frames yields an explicit
insufficient-split blocker rather than a fabricated held-out result.

The Task Evaluation Supervisor registers
`compile_frozen_frame_dataset` as a typed, zero-cost, scoped mutation tool only
when a deterministic compiler callable is injected by the service. The tool
cannot change the capture profile, split, proof state, or claim ceiling. The
normal capture-build ingress intentionally projects known JSON fields without
raw media paths, so the production lifecycle does not inject this compiler yet;
it remains fail-closed until deterministic capture admission exposes an
immutable retained-media binding. Agent prose cannot substitute for the tool
result.

This slice proves decoded observation availability, frame-selection
determinism, split immutability, hidden-view isolation, and replay integrity on
hermetic fixtures. It does not prove camera calibration, ARKit alignment,
metric scale, reconstruction quality, geometry, collision, physics, Isaac
compatibility, simulator task success, physical success, or deployment.

## Strict ARKit reconstruction export

The strict V3.2 ARKit/LiDAR adapter now feeds the frozen frame kernel into
`arkit_reconstruction_dataset.py`. That compiler binds candidate-only RGB
observations to the exact encoded-frame index, decoded PTS, capture timestamp,
raw `T_world_camera`, session intrinsics, coordinate frame, and any matching
depth/confidence references. FFmpeg extraction explicitly disables display
autorotation, records the extracted pixel dimensions, and refuses the export
when those encoded pixels do not match the declared intrinsics.

The compiler emits `camera_calibration_manifest.v1`,
`camera_observation_manifest.v1`, `pose_refinement_request.v1`, and
`arkit_reconstruction_dataset_export.v1`. Their shared Draft 2020-12 contract is
`docs/schemas/arkit_reconstruction_dataset.v1.schema.json`. The candidate
observation manifest contains training and validation paths only. It cannot
change raw ARKit poses or the frozen split, cannot read held-out pixels, and
cannot enable undeclared distortion or rolling-shutter models.

The pose-refinement request remains deterministically blocked because a
repository-approved drift threshold and executable anchored bundle-adjustment
tool are not registered yet. Accordingly, this export is a calibrated
reconstruction request, not a refined trajectory or COLMAP/gsplat dataset. The
ARKit scaffold records sensor-declared meter units, but its claim ceiling no
longer says metric scale or a metric reference layer is independently proven:
confidence filtering, RGB-depth alignment, metric-scale validation, and
geometric held-out evaluation have not run.

## ARKitScenes raw proxy execution

`arkitscenes_raw_proxy.py` is an explicitly reduced-authority public-dataset
adapter. It parses the retained MOV timed-metadata stream, joins decoded frames
through original video PTS to the recorded capture timestamp and intrinsics,
binds the official timestamped trajectory, filters depth to positive samples
with confidence value 2, and feeds the shared frozen-split kernel. Candidate
RGB, cameras, and depth live in candidate-only artifacts; held-out RGB,
cameras, and depth live in evaluator-only artifacts.

The accepted local scene `40958756` run is bound to source digest
`sha256:bc493651dcc0950146e49bab91c9303a4d5f49c319c3e0b1048de1344d568e04`,
commit `ddbff2998e00fdd728cf36e3c9a1c022b378b8b0`, frozen split digest
`sha256:be386b4cd681f520fa6689b669b4efbb5b8534f991b2df815f37dfa989eed020`,
and terminal digest
`sha256:4c1d69c959ce1df03be4196b4dc2cf6c762c73fd4f474bc2c95d6cf94f64b0f6`.
It decoded 1,013 frames from 1,014 timed metadata samples, found 163 exact
pose/RGB/depth/confidence/intrinsics timestamp joins, and froze 40 observations
as 32 candidate plus 8 hidden evaluator frames. Exact-SHA replay and all six
authoritative source hashes were revalidated locally.

This is iPad public-dataset proxy evidence only. ARKitScenes lacks Blueprint's
encoder-attempt and retained-frame ledger and does not provide the required
tracking/relocalization state. It therefore proves neither Raw Contract 3.2 nor
the iPhone route, independent metric scale, appearance quality, collision,
Isaac compatibility, physical success, or deployment readiness.

## Native dual-fisheye normalization

`native_360_normalization.py` implements the source-preserving native lane. Its
probe receipt binds exact source bytes, runtime, streams, dimensions, time
bases, and monotonic PTS. The normalizer requires exact `.insv` filename, size,
and digest declarations; deterministic segment order; explicit front/rear
stream bindings; calibrated per-lens intrinsics, distortion, masks and source
provenance; a rigid nonzero-baseline rig transform; coordinate semantics; and
explicit local rights, consent, privacy, retention, no-upload, and no-paid-
compute authority.

It emits `camera_360_rig_declaration.v1`,
`dual_fisheye_stream_binding.v1`, and
`native_360_capture_normalization.v1`. Unsynchronized lenses, dimension/count
mismatch, calibration/stream mismatch, unknown calibration, inconsistent probe
runtimes, missing sensor declarations, unsafe paths, symlinks, digest drift,
oversized media, or unbound probe receipts fail closed. Even a valid result has
the ceiling `calibrated_camera_rig`: it establishes no trajectory, metric
scale, reconstruction, geometry, collision, or Isaac result. The Task
Evaluation Supervisor exposes it only through a typed, zero-cost, digest-bound
registered tool backed by an injected trusted runtime; the agent receives no
filesystem path or generic execution handle.

## Remaining reconstruction qualification gaps

The executable local kernel should be extended, not replaced. Remaining work
includes bounded ARKit pose refinement; calibrated shared-center virtual rigs
and real native `.insv` execution; frozen pose-method comparison; a pinned
headless CUDA/ONNX COLMAP plus
gsplat/3DGUT worker; independent appearance and geometry evaluation;
metric-anchor and collider qualification; reproducible NuRec/OpenUSD packaging;
headless Isaac load/render/contact checks; provider-governed external imports;
and recorded qualification or rejection of enhancement methods. No
representative real iPhone or 360 capture has completed that full path in this
implementation.

## Normalized results and layers

`reconstruction_result.v1` binds exact capture, method-profile,
implementation, runtime, and result digests. It records camera/coordinate-frame
solutions, assets, coverage, observed and generated regions, uncertainty,
invalid regions, held-out metrics, cost, provider receipt, rights/retention,
deletion evidence, and an explicit claim ceiling. Every generated region needs
a mask. A result that contains generated regions and physics-like outputs must
explicitly exclude those regions from physics use.

The compiler keeps four independent testbed layers:

- appearance: splats, images, textures, and visual meshes;
- metric/reference: calibrated frames, scale, depth, and structural planes;
- semantic: object identities, regions, relations, and uncertainty;
- physics: independently checked collision geometry, bodies, joints, and
  physics properties.

Appearance never becomes collision truth. Generated content never becomes
observed truth.

## SimReady and robot placement

`simready_asset_decision.v1` decides necessity per object and requested claim.
Physics-dependent claims require an asset; perception/visibility-only claims do
not. An asset is selectable only when it binds the source capture, is not
generated-only, and an identity independent from its provider verifies scale,
transform, support, orientation, penetration, reprojection, and physics
properties. A realistic-looking provider output cannot self-qualify.

`robot_placement_result.v1` binds the exact robot, embodiment, footprint,
sensors, controller, end effector, task object, target, approved task, capture,
method qualification, and evidence digests. It filters support, footprint,
access, collision, reset, human-clearance, and coverage failures before scoring
reach, manipulability, visibility, approach, cable/controller constraints,
stability, and calibration uncertainty. If no covered candidate remains it
abstains and requests targeted capture or measurement.

## Immutable testbed versions

`blueprint-compile-site-task-testbed` consumes an accepted, digest-verified QA
report; the approved task; the reconstruction plan/results; SimReady and
placement decisions; exact Card/evaluator/reset artifact references; and
supported condition ranges. It emits the existing
`maintained_site_task_testbed.v1`, including raw source identity, layered
reconstruction, evidence inventory, robot binding, task objects/targets, reset,
governance, validation envelope, unsupported conditions, invalidation triggers,
and provenance.

Artifacts are stored by testbed ID, version, and digest using create-once
semantics. An inter-process lock and immutable version binding prevent two
digests from occupying one logical testbed version. A correction or
reconstruction must name a new version and bind the predecessor digest. The
compiler rejects unaccepted QA, stale capture/result bindings, results from
methods absent from the exact plan, credential-bearing artifact URIs, and
same-version successors.

The signed `/api/live-pipeline/testbeds/compile` version 2 service loads the
authoritative approved task, capture intake, QA, reconstruction plan, and
execution results from Pipeline state rather than trusting caller-supplied
scientific inputs. It rejects the former client-owned reconstruction fields and
requires the exact reconstruction plan ID and execution-result digest. The same
boundary now covers SimReady decisions, robot-placement results, evaluator/reset
artifacts, condition-range claims, and predecessor manifests. The caller may
provide only an owner-attested robot configuration. Pipeline derives a
per-object/per-claim SimReady decision, emits a placement abstention until a
qualified method produces exact candidates, creates immutable evaluator/reset
support artifacts, and limits the initial condition envelope to the accepted
capture observation. Those support artifacts are written beside the immutable
testbed manifest and cards.

The entire v2 submission is validated by the closed Pydantic contract in
`site_task_testbed_compilation_contract.py`. Its generated Draft 2020-12 schema
is checked in at
`docs/schemas/site_task_testbed_compilation_submission.v2.schema.json` and is
mirrored byte-for-byte by WebApp. Unknown contract-owned fields, malformed
identifiers/digests, caller-selected scientific scope, inconsistent robot
bindings, duplicate claim IDs, live-robot authorization, paid-compute
authorization, and WebApp provider selection all fail before compilation.

After compilation it can publish the full digest-bound testbed to WebApp through
`PIPELINE_TESTBED_WEBAPP_URL`, authenticated with `PIPELINE_SYNC_TOKEN`. A 2xx
response is insufficient: Pipeline accepts only a receipt matching the exact
session, intake, task digest, testbed ID/version/digest, artifact reference, and
proof boundary. Set `PIPELINE_TESTBED_WEBAPP_SYNC_REQUIRED=true` in a deployment
that requires customer-visible state before reporting service success.

## Proof boundary

Two local evidence adapters are available behind an explicit allowlist:
`local://analytic-reachability-v1` computes only from an explicit metric robot
base position, target position, reach envelope, and calibration uncertainty;
`local://captured-visibility-v1` reads only explicit target-region coverage and
retained supporting-frame IDs. Either adapter abstains when those inputs are
missing. Neither is registered by default, launches a provider or robot, or
upgrades its result to physical success, deployment readiness, or safety.

The signed run facade exposes plan, authorize, execute, and inspect operations
under `/api/live-pipeline/task-evaluation-runs`. Planning persists the exact
request, testbed, method profiles, qualification records, and deterministic
Evidence Plan before entering `authorization_required`. Authorization is a
separate immutable record bound to the plan digest and names the exact local
adapter references. Execute will not proceed without that record, cannot enable
paid/provider/physical execution, normalizes every attempted result, and always
aggregates the attempt into a Decision Envelope whose terminal state is
`decided`, `partially_decided`, or `abstained`. Exact retries return the prior
immutable envelope.

Hermetic compilation proves deterministic composition and contract integrity.
It does not prove reconstruction fidelity, collision correctness, task success,
deployment readiness, safety certification, or physical performance. The
comparative policy-ranking verdict remains `thesis_not_supported`.
