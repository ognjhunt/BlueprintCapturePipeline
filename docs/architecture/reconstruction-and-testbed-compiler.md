# Reconstruction Capability and Testbed Compiler

Status: implemented local contract boundary, deterministic frame/split kernel,
ARKitScenes raw proxy compiler, native dual-fisheye normalization, deterministic
OpenUSD packaging, and strict Isaac reconstruction qualification runner, version
1 (2026-07-30)

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

The compiler accepts a digest-bound
`arkit_reconstruction_dataset_export_request.v1` and emits
`camera_calibration_manifest.v1`,
`camera_observation_manifest.v1`, `pose_refinement_request.v1`, and
`arkit_reconstruction_dataset_export.v1`. Their shared Draft 2020-12 contract is
`docs/schemas/arkit_reconstruction_dataset.v1.schema.json`. The candidate
observation manifest contains training and validation paths only. It cannot
change raw ARKit poses or the frozen split, cannot read held-out pixels, and
cannot enable undeclared distortion or rolling-shutter models.

Both `compile_arkit_metric_scaffold` and
`export_arkit_reconstruction_dataset` are registered SDK tools. Their real
implementations stay in trusted runtime state; the agent sees only the capture
route digest or export-request digest. The tool observations cannot change
proof state, calibration, raw poses, or split membership.

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

## Stitched equirectangular shared-center rig

`equirectangular_virtual_rig.py` accepts only retained, digest-bound 2:1
panorama observations plus a versioned stitch declaration. The declaration
records whether pixels were customer-provided, official-SDK-produced, or
externally produced; names the producer and stitch receipt; and binds the
preserved original 360 source. Unknown stitch provenance or a missing original
source fails closed.

The compiler uses the fixed
`blueprint_erp_shared_center_12x100deg_v1` profile: four yaw angles at each of
three pitch angles, identical 512-square pinhole intrinsics, and deterministic
bilinear sampling. All 12 derived views from a panorama share one explicit
optical-center group and zero relative translation. Candidate and held-out
panoramas must be compiled under separate access scopes. The agent cannot
change the view definitions or request evaluator pixels through the candidate
tool.

The output is `equirectangular_virtual_camera_rig.v1` plus a material
`equirectangular_virtual_rig_compilation.v1` receipt. Its ceiling is only
`equirectangular_virtual_camera_rig`: perspective projections are derived
support pixels, not independent physical observations. The compiler does not
establish the source camera trajectory, native lens calibration, stitch
quality, metric scale, appearance reconstruction, geometry, collision, or
Isaac compatibility.

## Remaining reconstruction qualification gaps

The executable local kernel should be extended, not replaced. Remaining work
includes bounded ARKit pose refinement; real native `.insv` and stitched-360
execution; frozen rig-constrained pose-method comparison; a built and
smoke-tested reconstruction worker; independent appearance and geometry evaluation;
metric-anchor and collider qualification; reproducible NuRec/OpenUSD packaging;
headless Isaac load/render/contact checks; provider-governed external imports;
and recorded qualification or rejection of enhancement methods. No
representative real iPhone or 360 capture has completed that full path in this
implementation.

## Headless pose and appearance worker contracts

The Phase 4 contract kernel pins a candidate `linux/amd64` stack to a digest-bound
CUDA 12.4.1 Ubuntu base, COLMAP 4.1.1 with CUDA and ONNX enabled, ONNX Runtime
1.24.4, gsplat 1.5.3, and NVIDIA 3DGRUT 1.1.0. It also records candidate compiler,
FFmpeg, Python/PyTorch, OpenCV, Trimesh, OpenUSD, QA, driver, model-asset, license,
and redistribution constraints. The ALIKED and LightGlue ONNX assets use the
digests published with COLMAP 3.13.0. This is a pinned candidate manifest, not a
built or qualified image; build and headless smoke receipts remain mandatory.

`pose_estimation_request.v1` freezes SIFT or ALIKED extraction with compatible
brute-force or LightGlue matching, camera and rig/calibration bindings, split,
seed, resources, timeout, and spend ceiling. `reconstruction_training_request.v1`
does the same for gsplat/3DGUT MCMC training, initialization geometry, iteration
budget, output, and evaluation contracts. Results preserve registered and
rejected frames and typed failures. They contain no hidden held-out labels and
cannot self-grade. Checkpoints can resume only against the exact request and
random-state digest.

The Agents SDK exposes `run_pose_estimation` and
`train_gaussian_reconstruction` using request digests only. Trusted injected
runtimes receive the validated request and a supervisor-owned output root; the
agent receives no shell, filesystem, network, database, or provider handle.
Even successful tool output is only a calibrated-trajectory or appearance-asset
candidate. The independent evaluators and later geometry/Isaac gates own any
qualification.

`reconstruction_worker_build_packet.v1` is provider-neutral and names only the
canonical `paid_resource_allocator cpu-build` seam. It cannot select a provider
or launch a build. It fails closed without a clean immutable commit, exact source
tree, recipe and dependency-lock digests, license-review receipt, budget, TTL,
retry cap, and authority. Allocation and image-build success are explicitly not
scientific success.

## Metric geometry, collision, and Isaac qualification

Phase 5 extends the existing `geometry_stage`, ParticleField/NuRec exporters,
and Isaac renderer with deterministic qualification contracts; it does not
replace those executors. `metric_geometry_manifest.v1` records confidence
filtering, observed and unsupported regions, scale state, and a separate metric
asset. It rejects generated fill and rejects treating the appearance asset as
geometry truth.

`mesh_collider_candidate_manifest.v1` is explicitly unvalidated and preserves
component and hole statistics. `collider_qualification_report.v1` computes its
decision from frozen thresholds for scale, gravity, floor/wall residuals,
coverage, visual disagreement, obstacle thickness, clearance, and robot-footprint
navigability. A passing report is limited to bounded navigation simulation;
grasping, articulation, contact-force, and deployment claims remain unsupported.

`nurec_openusd_packaging_request.v1` binds exact appearance, metric-geometry,
collider-candidate, and independently accepted collider-qualification digests.
The repository-owned OpenUSD packager accepts only safe relative asset paths
from a trusted artifact root, re-hashes both sources, composes one meter/Z-up
visual and physics frame, creates a self-contained USDZ dependency closure, and
normalizes ZIP order, timestamps, storage, and 64-byte member alignment for
byte-stable replay. This version requires each bound source asset to be
self-contained; unlisted external layers/assets and compressed, encrypted,
symlinked, corrupt, traversal, or oversized USDZ members fail closed. It then
reopens the exact package and verifies ParticleField,
collision API, and dependency presence before emitting
`nurec_openusd_packaging_result.v1`. A successful package remains a compatibility
candidate, not collision or simulator proof. `isaac_asset_verification_result.v1`
requires the exact package, expected
prims, valid units/transforms, no missing assets, loaded ParticleField, active
collision geometry, a contact surface, a non-falling test body, and nonblank
fixed-camera renders without NaNs or obvious scale mismatch. Even a passing
result proves only Isaac load/render/physics-presence compatibility—not simulator
task success, physical success, or deployment readiness.

`isaac_reconstruction_verification.py` is the normalization boundary for the
headless runtime result. It intentionally rejects visual-only
`isaac_splat_nurec_render_result.v1`. The existing ParticleField runner now has
an explicit qualification mode that emits
`isaac_splat_nurec_render_result.v2`, binds and re-hashes the exact package, and
reports meters/Z-up, transforms, unresolved dependencies, ParticleField and
active collision prim counts, a stepped live-PhysX test-body probe against an
existing static package collider, conservative obvious-scale bounds, and
digest-bound nonblank fixed-camera renders. It does not create a helper floor.
Legacy callers remain on v1 and cannot satisfy
the physics-presence gate. The v2 code path is hermetically contract-tested but
remains real-Isaac unverified until executed on the pinned GPU worker.

## Strict external reconstruction import

Phase 6 preserves the legacy Scaniverse staging command for operator
compatibility while adding a separate deterministic supervisor lane. The
`external_reconstruction_import_request.v1` contract binds local exports to the
immutable source capture and to an exact, digest-verified Scaniverse provenance
and rights declaration. The registered `import_external_reconstruction` tool
accepts only that request digest and invokes an injected repository-owned local
importer; the model receives no filesystem, network, provider, or authority
handle.

The importer validates path confinement, symlink exclusion, file count and size
limits, exact hashes, supported formats, and safe USDZ archive structure before
copying to a content-addressed directory. It emits separate rights and import
receipts and re-hashes every replay. Provider exports remain derived support:
they establish no raw-capture, scale, metric geometry, collision, Isaac, task,
physical, or deployment claim. The lane performs no remote request. A future
remote adapter must pass provider admission and explicit confidential-upload,
terms, spend, retention/deletion, and source-binding authority before receiving
bytes.

The provider-neutral remote contract family is now present:
`reconstruction_provider_admission.v1`,
`reconstruction_provider_execution_request.v1`,
`reconstruction_provider_execution_receipt.v1`, and
`reconstruction_provider_deletion_receipt.v1`. Admission is derived from trusted
legal/capability gates and records zero provider mutations. Execution requests
bind a non-agent operator receipt, exact provider/actions, confidential-upload
authority, positive budget, TTL, retries, immutable inputs, source commit,
calibration/frame declarations, and the frozen split. Runtime receipts preserve
cost, duration, failures, downloaded hashes, and complete lineage while forcing
provider qualification, metric, collision, Isaac, physical, and deployment
claims false. The tool descriptor exists, but the supervisor exposes no remote
call until a separately qualified adapter is injected through the paid boundary.

## Generated reconstruction enhancement audit

Phase 7 keeps ArtiFixer, Difix3D+, and DiffusionHarmonizer replaceable behind the
enhancer registry. `reconstruction_enhancement_method_audit.v1` freezes the
reviewed source commits, distinct source/model licenses, modes, runtime facts,
blockers, and legal next actions. All three candidates are currently rejected:
ArtiFixer pending model-license and pinned-runtime qualification, Difix3D+ for
the commercial default because its published license is non-commercial, and
Harmonizer pending its checkpoint/base-model/worker qualification.

The ArtiFixer wrapper now refuses even an installed runtime unless it receives
exact source, container, checkpoint, base-model, license, baseline, and frozen
split digests plus a real held-out manifest. The 1.3B and 14B checkpoints require
the corresponding explicit Wan2.1 model identity, and syntactically valid pins
cannot override the rejected method audit. Difix3D+ and Harmonizer are
registered as deterministic rejected candidates rather than executable
subprocesses. Every method requires an existing baseline, preserves that
baseline, excludes hidden held-out observations, and remains bounded to
generated visual support with no metric, collision, physical, or deployment
effect.

Independent baseline and enhancement grading now uses the registered
`evaluate_heldout_appearance` tool and the
`heldout_appearance_evaluation_request.v1` /
`visual_heldout_evaluation_report.v1` contract family. The agent receives only
the request digest. Real held-out paths remain in evaluator-owned trusted
state, the candidate and evaluator identities must differ, and the split and
thresholds must have been frozen before evaluation. The initial hermetic
evaluator reports PSNR, mean absolute error, and a deterministic global-SSIM
equivalent; it explicitly labels the latter and does not claim canonical
windowed SSIM or LPIPS. A passing report raises only the appearance ceiling.

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
