# Measurement adapter execution runbook

This runbook covers the repo-owned, fail-closed execution port for measurement
benchmark adapters. The port currently admits local development cases only.
Qualification-split execution must occur on an independently controlled runner
that can prove hidden-label isolation, immutable runtime identity, and a clean
rerun; the local command refuses that split.

## Contracts

`blueprint_pipeline.measurement_adapter_execution` defines four checked
artifacts in `docs/schemas/measurement_adapter_execution.v1.schema.json`:

1. `measurement_adapter_execution_request.v1`
2. `measurement_adapter_worker_result.v1`
3. `measurement_adapter_execution_receipt.v1`
4. `measurement_adapter_execution_bundle.v1`

The request binds the adapter descriptor, benchmark specification, public case
manifest, implementation digest, exact target engine, backend, precision,
seed, solver settings, and timeout. The worker never receives qualification
labels or physical measurements. The runner uses an argv-only command without
a shell, a reduced environment, a temporary working directory, and an explicit
execution gate. Receipts retain log digests and byte counts, not log content.

## Plan and execute

Build the request through
`build_measurement_adapter_execution_request`. Then plan without mutation:

```bash
python -m blueprint_pipeline.measurement_adapter_execution \
  --request /path/to/execution_request.json \
  --output /path/to/plan_bundle.json \
  --command-arg /path/to/python \
  --command-arg=-m \
  --command-arg blueprint_pipeline.measurement_mujoco_adapter
```

The result is `planned_not_executed` and contains no prediction. To run the
same local development request, add `--execute`:

```bash
python -m blueprint_pipeline.measurement_adapter_execution \
  --execute \
  --request /path/to/execution_request.json \
  --output /path/to/execution_bundle.json \
  --command-arg /path/to/python \
  --command-arg=-m \
  --command-arg blueprint_pipeline.measurement_mujoco_adapter
```

The runner appends the standard `--request` and `--output` worker arguments.
Never place tokens, cookies, credentials, shell expressions, or held-out labels
in command arguments, solver settings, case manifests, worker output, or logs.

## Pinocchio/Coal Q-KIN development worker

Install the explicit exact-geometry development pair:

```bash
uv sync --frozen --extra dev --extra exact-geometry-development
```

`blueprint_pipeline.measurement_pinocchio_coal_kinematic_adapter` exact-pins
Pinocchio 4.1.0 and Coal 3.0.3. It solves a bounded synthetic two-link planar
target analytically, verifies the result with Pinocchio forward kinematics,
and checks each link against one primitive obstacle using Coal GJK signed-
distance queries at 101 finite joint-interpolation samples. It runs the full
case twice and requires an identical trace digest.

Plan the checked reachable-clear, reachable-collision, and unreachable corpus:

```bash
python -m blueprint_pipeline.measurement_geometry_kinematic_development_suite \
  --corpus tests/fixtures/measurement_capture_to_geometry_kinematic_v1/corpus.json \
  --qualification-split-digest sha256:<64-hex-preregistered-heldout-split> \
  --controller-scope-digest sha256:<64-hex-controller-scope> \
  --output /path/to/kinematic-development-plan.json
```

Add `--execute` to run all three cases. This worker does not load a captured
mesh or URDF, use site registration, evaluate self-collision, or perform
continuous collision detection. Finite sampling can miss between-sample
collisions. Its output is development evidence only and cannot support a
physical, safety, R5, R6, or R7 claim.

## Qualified geometry and robot/site registration bridge

The routing system can consume qualified site geometry without allowing an
agent or development worker to promote it. Build the site-evidence profile from
four independently checkable artifacts:

1. `metric_geometry_manifest.v1`, produced from the observed source surface;
2. `mesh_collider_candidate_manifest.v1`, preserving the no-fill collider
   candidate and source lineage;
3. `collider_qualification_report.v1`, produced by an independent bounded-
   navigation evaluator with its measurement artifact digest; and
4. `measurement_robot_site_registration.v1`, containing an independently
   measured and signed robot-base-to-site-frame SE(3) transform, task-region
   coverage, sample count, and translation/rotation residual limits.

Call
`blueprint_pipeline.measurement_site_evidence_bridge.build_site_evidence_profile_from_geometry`
with those four artifacts. The bridge revalidates each contract and checks the
source, metric asset, collider asset, qualification report, task regions,
independent evaluators, signature, error thresholds, and every digest join. A
successful result is a `site_evidence_profile.v1` that the deterministic router
and supervisor can audit directly.

An unsigned or development-only registration remains present but unvalidated.
The bridge cannot create a method qualification, R5/R6/R7 admission, physical
task-success claim, deployment-readiness claim, or safety certification. Those
claims require their own separately governed evidence.

## MuJoCo development worker

`blueprint_pipeline.measurement_mujoco_adapter` implements the first real
worker protocol, `mujoco_rigid_drop.v1`. It requires the exact descriptor target
version (currently MuJoCo 3.11.0), verifies its own source digest, builds the
fixed rigid-contact model from bounded public operating-point fields, executes
the simulation twice, and requires identical trace digests. It emits contact
sequence and penetration predictions plus exact runtime observations.

This is development evidence only. It does not establish captured-site
contact accuracy, satisfy the instrumented Geometry-and-Contact benchmark,
create an R5 result, make an R6 decision, admit an R7 route, or prove physical
success.

The checked fixture at
`tests/fixtures/measurement_capture_to_geometry_contact_v1/corpus.json`
contains separate sphere and box drop regimes. Plan both cases with:

```bash
python -m blueprint_pipeline.measurement_geometry_contact_development_suite \
  --corpus tests/fixtures/measurement_capture_to_geometry_contact_v1/corpus.json \
  --qualification-split-digest sha256:<64-hex-preregistered-heldout-split> \
  --controller-scope-digest sha256:<64-hex-controller-scope> \
  --output /path/to/geometry-contact-development-plan.json
```

Add `--execute` to run both cases and produce replay, first-contact-time, and
penetration aggregates. The corpus and suite schemas force `held_out`, physical
measurements, instrumented contact, independent execution, R5, R6, R7,
production eligibility, physical success, and agent promotion false. This is
the complete checked development corpus, not the independently measured
Geometry-and-Contact qualification program.

## Newton/MuJoCo cross-engine rigid-contact development suite

The geometry and development dependency sets exact-pin Newton 1.4.0 and Warp
1.15.0. `blueprint_pipeline.measurement_newton_rigid_adapter` executes the same
method-neutral sphere and box cases as the MuJoCo worker, using CPU XPBD with
`RUN_TO_RUN` deterministic mode. It verifies the exact package pair, its own
implementation bytes, backend, precision, solver settings, and double replay.

```bash
python -m blueprint_pipeline.measurement_geometry_contact_cross_engine_development_suite \
  --corpus tests/fixtures/measurement_capture_to_geometry_contact_v1/corpus.json \
  --qualification-split-digest sha256:<64-hex-preregistered-heldout-split> \
  --controller-scope-digest sha256:<64-hex-controller-scope> \
  --output /path/to/cross-engine-rigid-contact-plan.json
```

Add `--execute` to run both engines against both paired cases. The suite keeps
engine-specific case and execution digests and reports contact-sequence,
contact-generation-time, transient-penetration, and unsafe-prediction deltas.
These quantities use each solver's observed contact path and intentionally
expose disagreement. They are not physical errors until an independent hidden
physical label is joined; engine agreement is likewise not qualification.

## Drake SAP rigid-contact development worker

Drake 1.55 supports current macOS ARM through Python 3.13/3.14, while the main
repository runtime remains Python 3.12. Create a separate exact runtime at a
new explicit path:

```bash
python scripts/bootstrap_measurement_drake_development.py \
  --python /path/to/python3.13 \
  --environment /path/to/new/blueprint-drake-1.55 \
  --output /path/to/drake-bootstrap-receipt.json
```

The bootstrap refuses an existing or broad target path and verifies both the
`drake==1.55.0` distribution and `pydrake` import. Point the suite at the
resulting virtual-environment interpreter without resolving its symlink:

```bash
export BLUEPRINT_DRAKE_PYTHON=/path/to/new/blueprint-drake-1.55/bin/python
python -m blueprint_pipeline.measurement_geometry_contact_drake_development_suite \
  --execute \
  --corpus tests/fixtures/measurement_capture_to_geometry_contact_v1/corpus.json \
  --qualification-split-digest sha256:<64-hex-preregistered-heldout-split> \
  --controller-scope-digest sha256:<64-hex-controller-scope> \
  --output /path/to/drake-rigid-contact-development-suite.json
```

`measurement_drake_rigid_adapter` uses MultibodyPlant on CPU with the SAP
discrete contact approximation and point contact, creates no renderer or Drake
visualizer, and runs each sphere/box case twice. The executor binds the exact
interpreter command, adapter and wrapper bytes, solver settings, cases, logs,
worker results, and receipts. This is synthetic development evidence; it does
not establish hydroelastic-contact accuracy, a captured-site benchmark, R5,
R6, R7, physical success, deployment readiness, or safety.

## PyChrono isolated development runtime

PyChrono 10.0.0 is obtained from Project Chrono's conda release label, not from
an unrelated PyPI distribution. Create a new exact Python 3.12 environment:

```bash
python scripts/bootstrap_measurement_chrono_development.py \
  --conda /path/to/conda \
  --environment /path/to/new/blueprint-chrono-10 \
  --output /path/to/chrono-bootstrap-receipt.json
```

To re-verify an existing environment without mutating it, add
`--inspect-existing`. The receipt reads version, build, channel, and platform
from `conda-meta`, preloads an OpenMP library only when it resides inside the
selected environment, constructs `pychrono.core.ChSystemSMC`, and binds those
observations. It always declares that an environment receipt does not establish
a granular benchmark, and forces `production_route_eligible=false` and
`r7_admission=false`.

The executable development port uses core Chrono NSC rather than the unstable
SMC penalty-contact experiment or the specialized Chrono::Granular GPU module:

```bash
export BLUEPRINT_CHRONO_PYTHON=/path/to/blueprint-chrono-10/bin/python
python -m blueprint_pipeline.measurement_deformation_granular_chrono_development_suite \
  --execute \
  --corpus tests/fixtures/measurement_capture_to_deformation_granular_chrono_v1/corpus.json \
  --qualification-split-digest sha256:<64-hex-preregistered-heldout-split> \
  --controller-scope-digest sha256:<64-hex-controller-scope> \
  --output /path/to/chrono-granular-development-suite.json
```

The two public 27-sphere cases bind CPU/Bullet/PSOR NSC execution and exact
double replay. They measure spread, settling, particle-ground and interparticle
contact, Chrono-reported contact force, and penetration. This is a stable
synthetic development regression, not characterized-material DEM,
Chrono::Granular GPU evidence, pouring/tool interaction, physical accuracy,
R5, R6, or R7.

### Chrono::DEM exact-source CUDA development canary

The CUDA lane is deliberately separate from the PyChrono NSC worker. It builds
the exact Chrono 10.0.0 peeled commit with `CH_ENABLE_MODULE_DEM=ON` and runs two
public synthetic 27-sphere cases. First compile a bundle from a clean immutable
commit:

```bash
python -m blueprint_pipeline.measurement_chrono_dem_vast_bundle \
  --repo-root . \
  --corpus tests/fixtures/measurement_chrono_dem_cuda_v1/corpus.json \
  --qualification-split-digest sha256:<64-hex-preregistered-heldout-split> \
  --controller-scope-digest sha256:<64-hex-controller-scope> \
  --bundle-output /path/to/chrono-dem-input.zip \
  --receipt-output /path/to/chrono-dem-input-receipt.json
```

The compiler refuses a dirty checkout and records no spend authority. Paid
execution must use `python -m blueprint_pipeline.paid_resource_allocator
gpu-canary` with operation `measurement_chrono_dem_canary`, the exact runtime
release, bundle receipt, sensitive URL files, a live independent watchdog using
prefix `blueprint-measurement-chrono-dem-`, retry cap zero, and explicit spend
and hard-TTL bounds. Provider adapters and the canary module are not launchers.

A returned result may establish only the digest-bound synthetic CUDA
development run and exact replay. It cannot establish characterized-material
accuracy, pouring or tool interaction, Q-GRAN, R5-R7, production routing,
physical success, deployment, or safety.

## SAPIEN PhysX rigid-contact development worker

Install the explicit physics-development extra with the frozen environment:

```bash
uv sync --frozen --extra dev --extra sapien-development
```

`blueprint_pipeline.measurement_sapien_rigid_adapter` exact-pins SAPIEN 3.0.3
and runs the same public sphere and box cases used by the method-neutral rigid
corpus. It creates a physics-only scene with no renderer and no ManiSkill task
runtime, uses PhysX CPU with TGS, enhanced determinism, zero CPU workers, and
the request-bound iteration counts, and executes every case twice before
emitting a prediction.

Use the uniform executor shown above, replacing the worker module with:

```text
blueprint_pipeline.measurement_sapien_rigid_adapter
```

Plan the complete shared corpus without executing the worker:

```bash
python -m blueprint_pipeline.measurement_geometry_contact_sapien_development_suite \
  --corpus tests/fixtures/measurement_capture_to_geometry_contact_v1/corpus.json \
  --qualification-split-digest sha256:<64-hex-preregistered-heldout-split> \
  --controller-scope-digest sha256:<64-hex-controller-scope> \
  --output /path/to/sapien-rigid-contact-plan.json
```

Add `--execute` to run both cases and produce the schema-checked aggregate in
`capture_to_geometry_contact_sapien_development_suite.v1`.

The SAPIEN wheel declares the GUI OpenCV distribution. This project globally
excludes that distribution because every repository OpenCV consumer is
headless; `opencv-python-headless` remains the canonical locked `cv2` provider.
The worker does not use a Vulkan renderer, cameras, Pinocchio, or ManiSkill.
Its receipt is development evidence for the narrow PhysX rigid-drop protocol
only. It cannot inherit MuJoCo/Newton qualification or establish SAPIEN sensor,
rendering, task, policy, physical, R5, R6, or R7 validity.

## MuJoCo articulated-joint development worker

`blueprint_pipeline.measurement_mujoco_articulation_adapter` implements the
separately identified `mujoco_articulated_joint_travel.v1` protocol. It runs a
synthetic hinge or slide joint under a fixed effort twice and reports final
travel error and whether the joint limit was reached only after replay digests
match. Its effort and target are public fixture inputs, not measured force or a
physical label.

```bash
python -m blueprint_pipeline.measurement_geometry_contact_articulation_development_suite \
  --corpus tests/fixtures/measurement_capture_to_geometry_contact_articulation_v1/corpus.json \
  --qualification-split-digest sha256:<64-hex-preregistered-heldout-split> \
  --controller-scope-digest sha256:<64-hex-controller-scope> \
  --output /path/to/articulation-development-plan.json
```

Add `--execute` to run the door-hinge and drawer-slide cases. This closes the
checked Q-ART development plumbing for these narrow synthetic regimes. It does
not characterize site geometry, real mass/friction/joints, wrist force/torque,
robot interaction, insertion, held-out performance, or production authority.

## MuJoCo peg-insertion development worker

`blueprint_pipeline.measurement_mujoco_insertion_adapter` implements the
separately identified `mujoco_square_peg_insertion_boundary.v1` protocol. Its
public primitive geometry includes one centered-clearance case and one
lateral-interference case. The worker records signed clearance, side contact,
penetration, and final insertion outcome across two exact replays.

```bash
python -m blueprint_pipeline.measurement_geometry_contact_insertion_development_suite \
  --corpus tests/fixtures/measurement_capture_to_geometry_contact_insertion_v1/corpus.json \
  --qualification-split-digest sha256:<64-hex-preregistered-heldout-split> \
  --controller-scope-digest sha256:<64-hex-controller-scope> \
  --output /path/to/insertion-development-plan.json
```

Add `--execute` to run both cases. The interference case is intentionally
retained as an unsafe regression: it must report side contact, failed insertion,
and negative clearance. These results validate the development port, not a
captured socket, robot controller, wrist force/torque trace, physical success,
or R5-R7 decision.

## Calibrated observation development worker

`blueprint_pipeline.measurement_opencv_observation_adapter` implements
`opencv_calibrated_observation.v1` for the
`direct-captured-observations` candidate. Select it by changing the final
`--command-arg` in the examples above to:

```text
blueprint_pipeline.measurement_opencv_observation_adapter
```

The public case contains a canonical pinhole matrix, distortion coefficients,
non-coplanar metric 3D points, observed pixel correspondences, optional depth
samples, exact nanosecond timestamps, coordinate/unit declarations, and unsafe
thresholds. The worker verifies its source digest and exact OpenCV version,
runs `SOLVEPNP_ITERATIVE` twice, and emits calibrated reprojection RMSE,
missing-depth fraction, and temporal error only when replay digests match.

The checked fixture at
`tests/fixtures/measurement_capture_to_observation_v1/corpus.json` supplies two
synthetic development trials. Its schema forces `held_out`, physical-label,
R5, R6, and R7 flags false. It proves the execution and measurement plumbing,
not sensor fidelity at a customer site. A real Capture-to-Observation
qualification still needs independently captured qualification cases, hidden
physical labels, independent receipts, and the human R6/R7 process.

Plan the complete checked corpus without launching the worker:

```bash
python -m blueprint_pipeline.measurement_observation_development_suite \
  --corpus tests/fixtures/measurement_capture_to_observation_v1/corpus.json \
  --qualification-split-digest sha256:<64-hex-preregistered-heldout-split> \
  --controller-scope-digest sha256:<64-hex-controller-scope> \
  --output /path/to/observation-development-plan.json
```

Add `--execute` to run both cases. The output status can become only
`completed_development_only`; the suite contains receipt/bundle digests and
aggregate regression metrics, while all qualification and production fields
remain false.

## PyElastica cable-deformation development worker

`blueprint_pipeline.measurement_pyelastica_cable_adapter` implements
`pyelastica_cantilever_cable.v1` against exact-pinned PyElastica
0.3.3.post2. The public case describes a fixed-free Cosserat rod with explicit
element count, metric geometry, density, Young's modulus, gravity, damping,
timestep, duration, and displacement/strain envelopes. The worker uses
`PositionVerlet`, samples the tip trajectory, runs the complete simulation
twice, and emits `state_trajectory`, `force`, and `task_outcome` only when the
trace digests match.

Plan the complete two-regime cable corpus:

```bash
python -m blueprint_pipeline.measurement_deformation_cable_development_suite \
  --corpus tests/fixtures/measurement_capture_to_deformation_cable_v1/corpus.json \
  --qualification-split-digest sha256:<64-hex-preregistered-heldout-split> \
  --controller-scope-digest sha256:<64-hex-controller-scope> \
  --output /path/to/cable-development-plan.json
```

Add `--execute` to run both cases. This closes an executable development slice
for the cable lane only. It does not characterize a real cable, rope, or hose;
qualify cloth, granular, contact, topology change, or self-contact; create held-
out R5 evidence; or authorize R6/R7.

## MuJoCo flex-cloth development worker

`blueprint_pipeline.measurement_mujoco_flex_cloth_adapter` implements
`mujoco_flex_cloth_sag.v1` with a distinct implementation identity from the
rigid MuJoCo worker. It uses MuJoCo 3.11's 2D `flexcomp` stretch formulation,
four pinned corners, explicit material and collision parameters, gravity, and
an optional ground-contact regime. It records solver warnings, sampled sag,
edge strain, contact counts, and penetration, and repeats the full trace before
emitting `state_trajectory`, `topology_contact`, `force`, and `task_outcome`.

Plan the complete two-regime cloth corpus:

```bash
python -m blueprint_pipeline.measurement_deformation_cloth_development_suite \
  --corpus tests/fixtures/measurement_capture_to_deformation_cloth_v1/corpus.json \
  --qualification-split-digest sha256:<64-hex-preregistered-heldout-split> \
  --controller-scope-digest sha256:<64-hex-controller-scope> \
  --output /path/to/cloth-development-plan.json
```

Add `--execute` to exercise one free-sag and one ground-contact case. The suite
is scoped only to `mujoco-flex-elastic2d-stretch`. It cannot be generalized to
bending, self-collision, real garments, topology changes, other MuJoCo flex
backends, or physical cloth behavior without separate cases and qualification.

## MuJoCo spherical-granular development worker

`blueprint_pipeline.measurement_mujoco_granular_adapter` implements
`mujoco_spherical_particle_column_collapse.v1` with an implementation identity
separate from the rigid-drop and flex-cloth workers. It creates a bounded,
staggered column of identical noncohesive rigid spheres and records horizontal
spread, settling, particle-ground and interparticle contact, maximum normal
contact force, penetration, and solver warnings. Every trajectory is replayed
before a prediction is emitted.

Plan the complete two-regime corpus:

```bash
python -m blueprint_pipeline.measurement_deformation_granular_development_suite \
  --corpus tests/fixtures/measurement_capture_to_deformation_granular_v1/corpus.json \
  --qualification-split-digest sha256:<64-hex-preregistered-heldout-split> \
  --controller-scope-digest sha256:<64-hex-controller-scope> \
  --output /path/to/granular-development-plan.json
```

Add `--execute` to run both synthetic friction/size regimes. The solver scope
is exactly `mujoco-rigid-monodisperse-sphere-contact`, and the corpus declares
that restitution is not physically measured: contact damping is only a solver
parameter. This development reference is not Chrono DEM, a calibrated
commercial or MPM result, a cohesive or nonspherical material model, or the
required characterized-material pouring/tool-interaction qualification. It
cannot create R5, R6, R7, production, physical-success, deployment, or safety
authority.

## Direct tactile-sequence development worker

`blueprint_pipeline.measurement_direct_tactile_adapter` implements
`direct_tactile_sequence_reduction.v1`. It reduces synchronized optical marker
displacement, contact-intensity, normal-force, and shear-force frames, reports
contact-patch area and force ratios, and applies a public incipient-slip rule
twice before emitting a prediction.

```bash
python -m blueprint_pipeline.measurement_tactile_development_suite \
  --corpus tests/fixtures/measurement_capture_to_tactile_v1/corpus.json \
  --qualification-split-digest sha256:<64-hex-preregistered-heldout-split> \
  --controller-scope-digest sha256:<64-hex-controller-scope> \
  --output /path/to/tactile-development-plan.json
```

Add `--execute` to run the stable-contact and incipient-slip cases. The fixture
is synthetic with identity-only calibration. It proves sequence-reduction and
receipt plumbing, not real tactile calibration, force truth, TacSL or
DiffTactile validity, physical slip, policy ranking, or R5-R7 authority.

## World-model action-fidelity development worker

`blueprint_pipeline.measurement_world_model_action_fidelity_adapter` implements
`world_model_action_fidelity.v1` by materializing Blueprint's strict numeric
WAM action-recovery checks from public synthetic steps. It validates exact
action vectors, units, timing, uncertainty, controller/generated-state
bindings, forward/inverse results, and cross-step motion reuse.

```bash
python -m blueprint_pipeline.measurement_world_model_action_fidelity_suite \
  --corpus tests/fixtures/measurement_world_model_action_fidelity_v1/corpus.json \
  --qualification-split-digest sha256:<64-hex-preregistered-heldout-split> \
  --controller-scope-digest sha256:<64-hex-controller-scope> \
  --output /path/to/world-model-fidelity-plan.json
```

Add `--execute` to run the within/outside action-recovery cases. This adapter
evaluates supplied synthetic checks; it does not generate provider output or
run OSCAR, Cosmos, RoboWorld, IWS, or GigaWorld. It cannot measure policy
ranking, action-motion correlation, physics, task success, or deployment, and
the suite preserves the frozen `thesis_not_supported` verdict.

## Isaac Sim 6.0.1 PhysX rigid-contact development worker

Plan the two-case method-neutral corpus locally without claiming an Isaac run:

```bash
python -m blueprint_pipeline.measurement_geometry_contact_isaac_physx_development_suite \
  --corpus tests/fixtures/measurement_capture_to_geometry_contact_v1/corpus.json \
  --qualification-split-digest sha256:<64-hex-preregistered-heldout-split> \
  --controller-scope-digest sha256:<64-hex-controller-scope> \
  --output /path/to/isaac-physx-development-plan.json
```

On an already configured exact Isaac host, set
`BLUEPRINT_ISAAC_PYTHON=/isaac-sim/python.sh` and add `--execute`. The suite
refuses execution without the explicit external launcher and keeps
`actual_isaac_execution_verified=false` in plan-only output.

For a paid Vast execution, use only the canonical allocator. First, from a
clean immutable commit, build the source/corpus bundle and official runtime
release:

```bash
python -m blueprint_pipeline.measurement_isaac_vast_bundle \
  --repo-root "$CLEAN_CHECKOUT" \
  --corpus "$CLEAN_CHECKOUT/tests/fixtures/measurement_capture_to_geometry_contact_v1/corpus.json" \
  --qualification-split-digest sha256:<64-hex-preregistered-heldout-split> \
  --controller-scope-digest sha256:<64-hex-controller-scope> \
  --bundle-output /path/to/measurement-isaac-input.zip \
  --receipt-output /path/to/measurement-isaac-input-receipt.json

python -m blueprint_pipeline.measurement_isaac_runtime_release \
  --output /path/to/measurement-isaac-runtime-release.json
```

Stage the immutable bundle using expiring file-backed HTTPS GET/PUT URLs, arm
the independent watchdog for the `blueprint-measurement-isaac-` prefix, and
invoke `python -m blueprint_pipeline.paid_resource_allocator gpu-canary` with
`--provider vast`, `--probe-kind reconstruction-worker-smoke`, operation
`measurement_isaac_canary`, the bundle receipt/runtime release, explicit
budget and hard TTL, `--reconstruction-retry-cap 0`, and `--execute`. The lane
uses the exact official
`nvcr.io/nvidia/isaac-sim:6.0.1@sha256:783444c706538aa76cf5126e911ddc5e618779e6105305ad4af4260362a30aa9`
release, retrieves and validates output before teardown, destroys the instance,
and requires both scoped and global provider-zero. Signed URL values must
remain only in mode-0600 files and must never enter reports or logs.

A returned green development receipt proves only that the synthetic sphere and
box cases executed with the bound Isaac/PhysX runtime and replayed. It does not
create held-out R5 evidence, an R6 decision, R7 admission, captured-site
accuracy, policy ranking, physical success, deployment readiness, or safety.

## Routed Evidence Plan attachments and cross-engine development reports

`measurement_routed_execution.execute_routed_development_stage` accepts only a
`route_selected` decision and still requires the caller's explicit execution
flag. After completion, use `attach_routed_development_evidence` to create a
result-side attachment bound to the immutable Evidence Plan, claim, routing
decision, case, receipt, and prediction digests. The plan itself is not
rewritten.

For a comparative development check, put the same
`comparison_case_shape` object in each engine-specific case manifest, execute
each through the routed boundary, and pass the plan-bound attachments to
`build_routed_cross_engine_development_report`. The report rejects plan,
claim, route, case-shape, attachment-digest, or engine-identity drift. Numeric
metric ranges are solver disagreement evidence only; they do not establish
physical error, qualification, or engine interchangeability.

## SimReady preflight before target-simulator admission

`simready_asset_lane.preflight_simready_scene` validates each draft with a
headless MJCF load, body/mass/collider invariants, and a bounded finite-state
dynamics probe. The accompanying `probe_simready_preflight_toolchain` result
records whether the local trimesh, USD/pxr, MuJoCo, Blender, and NVIDIA
Content-Agent Validation surfaces are actually present. Missing optional
Blender or agentic validators are typed and visible; the function performs no
install, network access, or provider call.

The current macOS probe has trimesh, pxr/USD, and MuJoCo, but no `blender` or
`validation-agent` executable. Therefore a green current preflight means the
generated USD/MJCF draft is structurally loadable and numerically stable in the
bounded local reference, not that Blender/NVIDIA validation ran and not that
collider, articulation, material, contact, or sim-to-real validity exists.

### Moving a captured 3DGS object

Do not add an object splat on top of the original whole-scene splat. That leaves
the captured object in the background and creates a visible duplicate when the
new instance moves. Produce `gaussian_object_selection.v1`, run
`partition_gaussian_object`, and verify the resulting manifest/files. Then pass
the partition and the matching SimReady manifest to `build_dynamic_splat_scene`.
The compositor loads the static background once and the object-local splat once;
both visual and collider consume the declared body-pose channel. A legacy
`compose_simready_scene_binding` call without a partition is static-only and
cannot claim duplicate-free dynamic rendering.

## Failure handling

- Version, implementation, request, case, solver-setting, result, or replay
  mismatches fail closed and produce no prediction.
- A timed-out, blocked, failed, or plan-only receipt cannot enter benchmark
  evaluation as a completed prediction.
- Supervisor agents may summarize a receipt and propose human review. They may
  not execute, retry, qualify, promote, or mutate the catalog.
- Paid/provider and physical execution remain false throughout this local
  runner. Use the separately governed provider allocator or physical protocol
  only after the corresponding authorization exists.
