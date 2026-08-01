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
The subsequent `validate_camera_rig` tool independently rechecks the frozen rig
declaration, dual-fisheye binding, lens synchronization, and explicit segment
timeline. It can establish fixed calibrated rig compatibility only; it cannot
establish a trajectory or metric scale.
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
The v2 compiler isolates the split protocol digest from compiler SHA, runtime,
and implementation provenance, so unchanged selected observations and split
rules preserve exact train/validation/hidden-held-out membership across a code
or worker rebuild. Dataset provenance and the overall dataset digest still
change when the producing implementation changes.

Metric scale now has a separate `metric_scale_anchor_declaration.v1` and
`metric_scale_validation_result.v2` gate. Only a positive, independently
verified physical reference may serve as an anchor; learned or monocular depth
is explicitly rejected as the sole source. A
`reconstruction_anchor_measurement.v1` binds two frozen reconstruction-space
endpoints, the exact reconstruction and split, and an independent evaluator;
the validator computes their distance instead of trusting a candidate-supplied
scalar. It compares that measurement against the precommitted relative-error
threshold and cannot let the agent change the anchor, measurement, or threshold.

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

Later trusted bindings do not reopen or rewrite the terminal capture-ingress
ledger. `run_capture_reconstruction_supervisor_continuation` creates one linear,
content-addressed child supervisor run beneath the parent capture lifecycle for
an immutable execution-readiness snapshot. Its identity binds the capture,
source commit, typed request/source digests, requested registered tool IDs, and
the implementation digest of every injected runtime. A changed compiler,
evaluator, request, or readiness state therefore creates a distinct continuation
instead of replaying stale output. The parent report and prior readiness
snapshots remain immutable.

The continuation uses the same OpenAI Agents SDK supervisor harness and exposes
only digest-scoped registered tools whose runtime bindings actually exist. It
rejects undeclared context fields, missing runtimes, unregistered tools, and
route stages that lack the recorded control-plane authority needed for bounded
execution. The agent receives neither callable handles nor unrestricted shell,
filesystem, network, database, or provider access. The continuation receipt is
support evidence only: it cannot grant rights or budget, mutate calibration or
splits, change proof state, or establish physical success. The live service must
still construct these trusted bindings from admitted capture artifacts; an HTTP
caller cannot supply a callable runtime.

This slice proves decoded observation availability, frame-selection
determinism, split immutability, hidden-view isolation, and replay integrity on
hermetic fixtures. It does not prove camera calibration, ARKit alignment,
metric scale, reconstruction quality, geometry, collision, physics, Isaac
compatibility, simulator task success, physical success, or deployment.

The committed `tests/fixtures/reconstruction_vertical_v1/fixture_spec.json`
now drives three integrated, zero-spend vertical replays. The iPhone fixture
compiles a frozen candidate/evaluator split and an ARKit-bound reconstruction
dataset, then abstains because no resolved, smoke-tested worker image is
available. The native 360 fixture preserves synthetic `.insv`-family bytes and
a digest-bound recorded probe receipt, validates the dual-fisheye
timing/calibration and fixed rig, then abstains before pose and
metric scale. The stitched 360 fixture projects twelve deterministic
shared-center views, then abstains before pose and scale. Each replay emits a
deterministic terminal report preserving the blocker. These are hermetic
contract fixtures—not representative real capture, trainer, provider, or
Isaac execution evidence.

A current real stitched-video proxy execution is recorded in
`docs/evidence/ricoh360_bridge_equirectangular_39e3baa9.json`. It downloads the
author-published Ricoh360 original-video archive, validates its ZIP structure,
hashes and probes the 3840x1920 Ricoh Theta V `bridge.MP4`, validates 193
strictly increasing variable-spaced decoded PTS values, freezes 16 observations
into 11 training, two validation, and three hidden held-out panoramas, and
compiles physically separate candidate and evaluator shared-center rigs. This
is real stitched-360 input evidence, but it is a public outdoor dataset proxy:
there is no native dual-fisheye source, physical camera trajectory, independent
metric scale, qualified stitch result, appearance training, collision layer, or
Isaac result. The repository license is MIT, while a separate dataset license
was not stated; only the explicitly authorized local research evaluation was
performed, with commercial, redistribution, provider-upload, and paid-compute
authority remaining absent.

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
threshold must be supplied by a later frozen
`pose_refinement_execution_request.v1`; the export itself cannot choose one.
`run_pose_refinement` is now a registered worker boundary for ARKit-anchored
bundle adjustment or pose-graph refinement. It preserves the raw pose manifest,
forbids hidden-view access, and rejects a nominally successful result when its
maximum translation or rotation drift exceeds the precommitted limits. No
qualified refiner runtime has been executed on a real bundle yet. Accordingly,
the export alone is a calibrated reconstruction request, not a refined
trajectory or COLMAP/gsplat dataset. The
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

The first accepted local scene `40958756` run is bound to source digest
`sha256:bc493651dcc0950146e49bab91c9303a4d5f49c319c3e0b1048de1344d568e04`,
commit `ddbff2998e00fdd728cf36e3c9a1c022b378b8b0`, frozen split digest
`sha256:be386b4cd681f520fa6689b669b4efbb5b8534f991b2df815f37dfa989eed020`,
and terminal digest
`sha256:4c1d69c959ce1df03be4196b4dc2cf6c762c73fd4f474bc2c95d6cf94f64b0f6`.
It decoded 1,013 frames from 1,014 timed metadata samples, found 163 exact
pose/RGB/depth/confidence/intrinsics timestamp joins, and froze 40 observations
as 32 candidate plus 8 hidden evaluator frames. Exact-SHA replay and all six
authoritative source hashes were revalidated locally.

A current-tree replay at clean commit
`b2d7297fc3b28d2bb0a7b02ff3901137d70f51d3` independently downloaded the
same six Apple-hosted source artifacts, reproduced the same aggregate source
digest, decoded 1,013 frames, joined the same 163 exact multimodal timestamps,
and froze 28 training, 4 validation, and 8 hidden held-out observations. Its
configuration-bound split digest is
`sha256:8d12da972bc99eabddd6061476121bcfff9d266435ae088ed32247f83a0f058c`
and terminal digest is
`sha256:0a37ab5a1a7b52c8917c58c840d6a8110e6e981463ffd89c1466c44c5991b22e`.
All proxy and shared dataset schemas passed, the terminal self-digest passed,
and an identical rerun preserved the first timestamp and terminal bytes. The
exact compact receipt is committed at
`docs/evidence/arkitscenes_raw_proxy_40958756_b2d7297f.json`; public dataset
bytes remain outside Git. The downloaded Apple license text had SHA-256
`1b6a8700127de50c9d56f8f33eb202a64f6f212fd4b133435f7c8b6bccd3db59`.
The subsequent deterministic terminal report at commit
`27faf76309b94103cbe778aece6b549cb973d615` abstains at the first unavailable
appearance-worker gate and records terminal digest
`sha256:241b32ca74ea4b2723378052bf3c5e73dbad578a0f98ea5fe980a5ff588ac898`.
It preserves decoded-observation and dataset-proxy calibrated-trajectory
ceilings only; every metric-qualification, collision, Isaac, simulation,
physical, and deployment ceiling remains false. The exact report is
`docs/evidence/arkitscenes_reconstruction_terminal_report_40958756_27faf763.json`.

This is iPad public-dataset proxy evidence only. ARKitScenes lacks Blueprint's
encoder-attempt and retained-frame ledger and does not provide the required
tracking/relocalization state. It therefore proves neither Raw Contract 3.2 nor
the iPhone route, independent metric scale, appearance quality, collision,
Isaac compatibility, physical success, or deployment readiness.

## Native dual-fisheye normalization

`native_360_normalization.py` implements the source-preserving native lane. Its
bounded local ffprobe executor hashes the exact source and runtime before and
after execution, invokes no shell, limits time and output bytes, strictly parses
container/stream facts and decoded-frame PTS/DTS, and emits the existing typed
probe receipt. That receipt binds exact source bytes, runtime, streams,
dimensions, time bases, and monotonic decoded PTS. It explicitly does not infer
lens identity, calibration, IMU, gyro, trajectory, or metric scale. The
composed local entrypoint probes every declared segment before invoking the
normalizer, checks local rights and consent before any decode, and persists each
exact receipt under the content-addressed normalization root for replay. The
normalizer requires exact `.insv` filename, size, and digest declarations;
deterministic segment order; explicit front/rear
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

The local executor has been exercised against a generated two-stream MOV
container carrying an `.insv` filename: ffprobe 7.1.1 observed both 64x64 video
streams and their three decoded timestamps. This proves executable container
probing and receipt compilation only. It is not representative Insta360 media
and proves no native stream topology, embedded metadata, calibration, or 360
route qualification.

`native_360_frame_dataset.py` binds already-decoded lens images back to the
validated source path, source digest, declared stream index, decoded index, and
per-lens PTS in the dual-fisheye binding. The shared frozen dataset kernel now
supports optional physical-camera and synchronized-observation-group identities:
front and rear pixels from one physical instant always receive the same frozen
train, validation, or hidden-held-out assignment. It rejects missing/rebound
lenses, duplicate camera membership, invalid images, digest/dimension mismatch,
and hidden-counterpart leakage. Camera calibration is bound into the dataset but
the compiler's proof effect remains decoded-observation availability only; it
does not establish trajectory or metric scale. The hermetic native fixture now
reaches this grouped split gate and abstains at the later pose-worker and metric-
anchor gates. Multi-segment native decoding is supported only when every segment
has a declared, non-overlapping capture-timeline start. Each segment retains its
own immutable source reference and local PTS; the compiler derives the global
observation time from that declared start plus relative front-lens PTS. Missing,
invalid, or overlapping segment timing fails closed rather than inferring order
from filenames or segment position.

Fixed rig extrinsics are accepted only with an explicit transform direction
(`rear_camera_from_front_rig` or its declared inverse) and translation units in
meters. The legacy matrix label alone is intentionally insufficient: without
those declarations the normalizer abstains, so a downstream COLMAP adapter
cannot silently guess `cam_from_rig` semantics or manufacture calibration.
Each per-lens valid-pixel mask is now a retained capture artifact, not a digest-
only assertion: its capture-relative path is checked for traversal and symlink
escape, the bytes are rehashed against the calibration declaration, and the
accepted mask is copied immutably into the normalization artifact. Missing or
tampered masks invalidate that lens calibration and lower the claim ceiling.

The same module now provides a bounded ffmpeg lens decoder. It uses only the
validated source path, source digest, declared physical-lens stream indices, and
declared frame-pair ordinals; invokes no shell; applies per-frame timeout and
output limits; disables autorotation; preserves fisheye distortion; validates
the exact PNG dimensions and digests; and emits the replayable
`native_360_lens_decode_manifest.v1`. Decoded pixels remain accessible only to
the trusted dataset compiler until frozen candidate/hidden materialization.
The grouped dataset compiler requires this exact manifest and binds its digest
as a parent; runtime/source/frame substitutions are refused rather than silently
accepted as equivalent decoded observations.

A two-segment hermetic replay exercises the complete local front end: both
source files are independently hash-bound, 20 paired lens observations decode,
local PTS resets remain attached to their segment, the declared segment starts
produce one strictly increasing capture timeline, and the shared compiler freezes
front/rear pairs atomically without hidden-view leakage. This is contract and
local-executor evidence only, not representative Insta360 qualification.

`native_360_pose_request.py` closes the deterministic handoff from that frozen
dataset to the existing registered pose worker. It reproduces the camera-rig
validation result, binds the dataset, split, all retained source digests, rig
calibration, stream binding, worker image and source SHA, build/smoke receipts,
prequalified feature assets, fisheye camera model, seed, resource request,
timeout, retry cap, and spend cap into `pose_estimation_request.v1`. Hidden
pixels, an invalid capture timeline, unpinned learned models, or remote execution
without provider authority retained in both the execution envelope and capture
artifact fail closed. Compilation does not execute COLMAP and proves neither a
trajectory nor metric scale; the resulting request keeps scale at
`anchor_required`.

`native_360_colmap_plan.py` compiles that admitted request into an inert,
replayable COLMAP 4.0.4 execution plan. Candidate observations are rematerialized
under `front/` and `rear/` with identical filenames for each synchronized rig
frame; a trusted supervisor may rebase their manifest-relative locations under
one safe artifact root, while hidden-held-out locations remain structurally
forbidden. Calibration masks must reproduce the accepted normalization receipt,
and a separate safe normalization artifact root can be applied without changing
their recorded relative paths. The plan binds
per-lens calibrated valid-pixel masks, exact fisheye intrinsics/distortion, and
the declared transform direction before producing COLMAP `cam_from_rig`
quaternion/translation values. It emits argv arrays for headless feature
extraction, `rig_configurator`, deterministic sequential matching, and mapping
with sensor extrinsics and intrinsics held fixed. SIFT/ALIKED and compatible
brute-force/LightGlue pairings use one frozen protocol. Unknown camera axes,
missing masks, malformed transforms, mismatched distortion models, incomplete
front/rear groups, or any hidden path fail closed. The plan grants no shell or
network access, does not execute COLMAP, and has an `execution_plan_only` claim
ceiling; a registered trusted runtime and a typed pose result remain required.

`native_360_colmap_runner.py` is that bounded trusted runtime. It independently
rehashes every candidate image and calibrated mask, rejects symlinks and unsafe
paths, materializes the fixed rig workspace, and invokes only the five admitted
COLMAP subcommands as argv arrays with `shell=False`, bounded time, and bounded
logs. The final model-converter step produces text camera/image/point records;
the runner parses only the registered candidate image names to build the typed
`pose_estimation_result.v1` registered/rejected inventory. Unknown returned
images, missing or malformed model files, timeouts, startup failures, and
nonzero commands become typed failures with retained logs. Both successes and
failures are immutable and replay without repeating an unchanged attempt. The
service wrapper fits the existing registered `run_pose_estimation` callable and
accepts only the exact request digest. Its supervisor runtime identity also binds
the plan digest, opaque input-root identity, execution bounds, and runner kind;
the agent still receives no plan, filesystem, shell, or runtime handle. Hermetic
tests use a fake process runner,
so this establishes adapter behavior but not a real COLMAP trajectory.

The native reconstruction route now orders normalization before frozen dataset
compilation. A trusted runtime-only compiler service composes decode and grouped
split under the registered `compile_frozen_frame_dataset` tool; the model sees
only capture-build and route digests. The dataset publishes the decode manifest
as a supporting artifact reference, which the supervisor independently resolves,
rehashes, and records in the tool observation ledger.
Timeout, malformed or absent output, changed source/runtime bytes, symlinked
targets, wrong dimensions, and immutable conflicts fail closed.

An installed-runtime smoke traversed a generated two-stream 64x64 MOV carrying
an `.insv` filename through ffprobe, normalization, ffmpeg lens decoding, and
the grouped frozen-split compiler: two streams with three frames each produced
three synchronized pairs and six decoded lens observations. Its ceiling is
decoded-observation availability only. The container, calibration, and rig were
synthetic, so this is not native Insta360, real 360, metric-scale, pose,
appearance, collision, or Isaac proof.

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
includes bounded ARKit pose refinement; real native `.insv` execution and a
representative indoor stitched-360 execution; frozen rig-constrained
pose-method comparison; a built and
smoke-tested reconstruction worker; independent appearance and geometry evaluation;
metric-anchor and collider qualification; reproducible NuRec/OpenUSD packaging;
headless Isaac load/render/contact checks; provider-governed external imports;
and recorded qualification or rejection of enhancement methods. No
representative real iPhone or 360 capture has completed that full path in this
implementation.

## Headless pose and appearance worker contracts

The Phase 4 contract kernel pins a candidate `linux/amd64` stack to a digest-bound
CUDA 12.4.1 Ubuntu base, COLMAP 4.0.4 at official tag commit
`9c23f6942fe69962e06030905e77067c8673382f` with CUDA and ONNX enabled, ONNX Runtime
1.24.4, gsplat 1.5.3, and NVIDIA 3DGRUT 1.1.0. It also records candidate compiler,
FFmpeg, Python/PyTorch, OpenCV, Trimesh, OpenUSD, QA, driver, model-asset, license,
and redistribution constraints. Python 3.11.9 and FFmpeg 6.1.1 source archives
are SHA-256 verified. The Linux/amd64 Python environment is resolved into a
hash-enforced 107-package runtime lock plus a separate hash-enforced build-tool
bootstrap; only the recorded pure-Python `asciitree` and
`antlr4-python3-runtime` source distributions may build, with build isolation
disabled. Runtime dependencies declared by the pinned gsplat and 3DGRUT sources
are included. The baseline's unconditionally imported fused-SSIM CUDA extension
is built from pinned upstream commit
`1272e21a282342e89537159e4bad508b19b34157`; the embedded healthcheck verifies
that source revision and the fused-SSIM, NCore, SlangTorch, and Hydra imports.
The ALIKED and LightGlue ONNX assets use the
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

`reconstruction_worker_build_packet.v2` is provider-neutral and names only the
canonical `paid_resource_allocator cpu-build` seam. It cannot select a provider
or launch a build. It fails closed without a clean immutable commit, exact source
tree, recipe and dependency-lock digests, digest-bound license inventory and v2
human review receipt, budget, TTL, retry cap, and non-agent authority. The inventory binds
all 107 hash-locked Python dependencies, source components, model assets, the
worker stack, and the exact license-policy digest. It grants no authority. A v2
review receipt must acknowledge every recorded inventory blocker and every
component identity, is limited to a private internal build, and cannot be issued
from agent prose or a legacy receipt. The
`reconstruction_worker_paid_execution_envelope.v1` record then binds the exact
source SHA, worker stack, inventory, review receipt, dollar cap, two-hour-or-less
TTL, retry cap, and authority identity. The canonical CPU-build admission rejects
any drift between that archived envelope and its separate spend input before a
provider call. Allocation and image-build success are
explicitly not scientific success. The v1 schemas remain available only for
replay of recorded historical admissions.

The current clean-SHA admission at
`ff9deb59bd2ac96a3ffc72d2ea70abdeb6fb9912` binds its historical Git tree,
Dockerfile, dependency lock, and candidate stack manifest. It is deterministically
`blocked` only by `worker_license_review_receipt_missing`, the missing numeric
budget, TTL, retry cap, and paid-authority identity. The recorded stack and
admission artifacts are
`docs/evidence/reconstruction_worker_stack_manifest_ff9deb59.json` and
`docs/evidence/reconstruction_worker_build_admission_ff9deb59.json`. No build or
provider launch occurred. That historical stack named unpublished COLMAP 4.1.1
and is now a superseded support artifact, not an executable current pin; it is
retained so the rejected admission remains auditable. The current executable
contract uses official COLMAP 4.0.4 and requires a new build packet and receipts.
The recorded ARKitScenes candidate dataset is also
regression-tested against `reconstruction_training_request.v1`: without a
resolved digest-pinned worker image, request compilation fails closed before
any trainer can receive candidate pixels.

`reconstruction_worker_remote_build_packet.v2` materializes the corresponding
exact-source build context for that seam. It includes only tracked package
sources and the required Docker recipe, lock inputs, lock generator, and resolved
lock files, records every member digest,
rejects symlinks and unsafe archive members, emits a byte-deterministic archive,
and binds the executable build script to the source commit, recipe, dependency
lock, worker-stack manifest, license-inventory digest, v2 license-review receipt,
paid-execution-envelope digest, and context manifest. All three governance
artifacts are included as archive members and re-hashed by the remote script
before the first registry mutation. The license receipt can authorize only a private internal build; it cannot grant
redistribution or commercial distribution rights, and an agent cannot issue it.
The script builds only `linux/amd64`, requests
BuildKit provenance and SBOM attestations, resolves the pushed registry digest,
and emits `reconstruction_worker_build_receipt.v2`. The shared CPU builder
independently validates both the archive and returned receipt and still requires
its separate numeric spend/TTL envelope, watchdog, teardown, and provider-zero
proof. A build receipt remains below the GPU runtime-smoke and scientific gates.

`reconstruction_worker_build_receipt_normalization.py` closes the evidence seam
between that shared CPU builder and the stable worker contract consumed by pose
and training request compilation. It accepts only a ready, canonical, untampered
v2 packet; a digest- and registry-repository-bound remote build receipt; the
matching completed outer builder result; and a teardown record whose independent
lookup confirms provider absence. Source SHA, worker stack, build context,
license inventory, human review receipt, paid-authority envelope, image digest,
duration, and spend must all agree and remain inside the frozen cap and TTL. The
normalized `reconstruction_worker_build_receipt.v1` retains digests for every
parent record and explicitly records that the GPU healthcheck has not run. It
therefore proves only exact image construction plus CPU-builder teardown, never
GPU compatibility, reconstruction quality, Isaac compatibility, physical
success, or deployment readiness.

After the Vast-first runtime smoke, `reconstruction_worker_smoke_receipt.py`
normalizes the digest-bound healthcheck, execution, teardown, provider-zero, and
paid-authority records into the canonical
`reconstruction_worker_smoke_test_receipt.v1` consumed by the training-request
compiler. It refuses stale image/SHA bindings, partial healthcheck ledgers,
ambiguous mutations, nonzero provider inventory, failed teardown, authority
drift, and cost or duration beyond the frozen envelope. The normalized receipt
proves worker-image compatibility only and has no reconstruction-quality effect.

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

Candidate training output does not enter packaging directly.
`appearance_asset_manifest.v1` verifies the successful training receipt and
standard-3DGS PLY digest, preserves the trained spherical-harmonic bands, and
authors a meter/Z-up `ParticleField3DGaussianSplat` USD. The manifest binds the
source capture, frozen split, calibration, training request/result, worker
image, source commit, authority, and exact input/output digests. It records that
the layer is reconstructed rather than captured evidence and cannot prove
metric geometry, collision geometry, held-out quality, or Isaac rendering.

`nurec_openusd_packaging_request.v1` binds exact appearance, metric-geometry,
collider-candidate, and independently accepted collider-qualification digests.
The frozen request carries the complete validated appearance manifest and
cross-checks its capture, split, calibration, coordinate frame, prim path,
artifact reference, and asset digest, so citing a valid manifest beside a
different USD fails closed.
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

The current local OpenUSD execution receipt is
`docs/evidence/openusd_local_packaging_fixture_5d9675f6.json`. On clean commit
`5d9675f6`, OpenUSD 24.8 authored and reopened a byte-stable, self-contained,
meter/Z-up USDZ with one ParticleField appearance prim and one collision-API
prim. The exact replay matched, and final dependency inspection found no
missing or unresolved assets. This is packaging-mechanics evidence over a
synthetic fixture. Its metric and collider measurements are declared fixture
values, so it does not qualify real metric geometry, collision, Isaac load or
render behavior, physics contact, task success, physical success, or deployment.

`isaac_reconstruction_verification.py` is the normalization boundary for the
headless runtime result. It intentionally rejects visual-only
`isaac_splat_nurec_render_result.v1`. The existing ParticleField runner now has
an explicit qualification mode that emits
`isaac_splat_nurec_render_result.v3`, binds and re-hashes the exact package, and
reports meters/Z-up, transforms, unresolved dependencies, ParticleField and
active collision prim counts, a stepped live-PhysX test-body probe against an
existing static package collider, conservative obvious-scale bounds, and
digest-bound nonblank fixed-camera renders. It does not create a helper floor.
The typed `isaac_asset_verification_request.v1` freezes the exact package,
camera set, runner implementation, pinned runtime image, expected prim paths,
and physics-probe configuration before execution. The independent normalizer
then re-hashes the retrieved USDZ and every PNG, decodes the PNGs itself, and
rejects runtime-reported dimensions or pixel statistics that do not match the
retrieved bytes. `isaac_verification_worker_bundle.v1` packages those exact
inputs deterministically without allocating or authorizing paid compute; any
GPU execution must still enter through
`python -m blueprint_pipeline.paid_resource_allocator gpu-canary`. Legacy
callers remain on v1 and cannot satisfy the physics-presence gate. The v3 code
path and worker bundle are hermetically contract-tested but remain real-Isaac
unverified until executed on the pinned GPU worker.

For reconstruction GPU admission, the canonical `gpu-canary` seam can refresh
the supplied preflight bundle with mutation-free Vast marketplace and billable
inventory calls by using `--reconstruction-refresh-preflight`. The refresh
reuses the frozen watchdog and conflicting-owner declarations, requires an
explicit disk floor and hourly-rate ceiling, records both scoped and global
inventory, and performs zero provider mutations. It does not reserve an offer
or bypass the independent budget, TTL, retry, clean/pushed-SHA, image, transport,
watchdog, or paid-lane gates.

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

`run_generated_repair_candidate` is a registered digest-only gate. Under the
current audits it emits `generated_repair_candidate_result.v1` with
`blocked_not_qualified`, no generated artifacts, zero cost, and the exact
method-audit blockers. It cannot start ArtiFixer, Difix3D+, or Harmonizer by
inventing a qualified status or altered audit digest. A future executable
version requires a new reviewed contract after the relevant model/license,
worker, checkpoint, and real held-out gates pass.

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

`diagnose_reconstruction_failure` is registered as a digest-only, non-spend
recovery tool. Its `reconstruction_failure_diagnosis.v1` output uses the shared
worker failure taxonomy, preserves every attempt, and fingerprints the failure
code plus immutable input and configuration. A first transient failure may
propose one bounded retry. A second identical failure is rewritten to
`repeated_identical_blocker`, forbids an unchanged retry, and limits the legal
next actions to preserving evidence and abstaining. Diagnosis never executes
recovery or grants authority.

The registered `generate_reconstruction_report` tool compiles a
`reconstruction_terminal_report.v1` from a frozen request digest. It records
the original capture and request, rights, selected and rejected observations,
all attempted/failed/skipped/recovered methods, separate appearance/metric/
collision/Isaac artifacts, independent metrics, cost, provider and teardown
state, agent proposals, deterministic validations, blockers, and what could
change the result. Its eleven boolean evidence ceilings remain distinct from
the overall usability decision. The report is a replayable explanation: it
cannot mutate proof, hide a supplied failed attempt, grant authority, or turn
simulation into physical or deployment success.

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

## Pinned worker and paid canary boundary

`deploy/docker/reconstruction_worker/Dockerfile` is the Linux/amd64 headless
candidate recipe for pinned FFmpeg, CUDA/ONNX COLMAP, gsplat, 3DGRUT, OpenUSD,
and deterministic QA dependencies. The build and runtime healthcheck validates
source revisions, model bytes, imports, display absence, and (at runtime) the
NVIDIA device. A checked-in recipe is not a resolved image, build receipt, GPU
smoke receipt, or scientific qualification.

The `reconstruction-worker-smoke` probe is reachable only through
`python -m blueprint_pipeline.paid_resource_allocator gpu-canary`. Its request
is provider-neutral and binds the exact source commit, resolved image,
reconstruction dataset, frozen split, calibration, configuration, budget, TTL,
retry cap, and authority. The first qualification provider is Vast. Admission
requires a fresh supported offer, provider-zero inventory, no conflicting
owner, and an independent watchdog. The focused-tested execute adapter now
requires a fresh opaque `gpu_render` grant from the canonical allocator, private
0600 HTTPS PUT/GET URL files, and the exact digest-pinned worker image. It runs
only the image healthcheck, validates the GPU/image/source-SHA result, retrieves
the result before teardown, independently verifies scoped and global
provider-zero state, reconciles its paid-lane lease, and emits digest-bound
execution, teardown, and provider-zero receipts. Accepted receipts can be
replayed offline; tampering fails closed. This is an implemented execution
contract, not a built-image or live-provider result: no resolved reconstruction
worker image has yet been built or run, and admission, allocation, or a passing
worker smoke can never establish reconstruction quality.

The current live gate therefore remains external and explicit: provide a clean
immutable source commit, a resolved `linux/amd64` image digest built from the
pinned recipe, immutable dataset/split/calibration digests, numeric dollar
budget, numeric TTL and retry cap, an authority identifier, and the two private
signed URL files. Invoke it only through `python -m
blueprint_pipeline.paid_resource_allocator gpu-canary --probe-kind
reconstruction-worker-smoke`; direct adapter launch has no admission capability.
