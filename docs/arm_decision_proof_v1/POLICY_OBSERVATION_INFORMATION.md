# Setup knowledge, policy input, and object acquisition

This implements the configuration portion of **ADP-009D / day 14**: a paired
policy test must declare what information the policy receives and whether the
object should already be visible. The existing DROID builder selects its inputs
correctly, but it had no explicit deployment-source contract for object
coordinates or a separately preregistered visual-search protocol. The smallest
change is an explicit binding around that builder, the existing exact-workcell
schedule, and the current canary worker/episode/receipt seams. It does not modify
the current frozen GPU run.

## What is configured

`PolicyObservationInformation` keeps setup coordinates separate from an
optional policy-coordinate source. Setup can use simulator ground truth, known
deployment coordinates, or estimated deployment coordinates, with their source,
coordinate frame, measurement, and context digests preserved. For a staged
matrix, the setup measurement context must equal that cell's resolved reset
digest. Reusing another cell's measurement is rejected. When a coordinate axis
varies directly, `setup_position_dimensions` names its matrix dimension and the
planner compares the coordinate against the resolved value and tolerance.
Registration transforms or other indirect placement rules still need their
existing producer/readback evidence; a digest alone does not prove application.

The current adapters declare these inputs:

| Adapter | Policy inputs | Object XYZ |
| --- | --- | --- |
| `openpi_droid.v1` | Exterior RGB, wrist RGB, seven joints, gripper, language | Unsupported |
| `groot_n17_droid.v1` | Exterior RGB, wrist RGB, joints, gripper, end-effector state and frame provenance, language | Unsupported |

`build_configured_droid_observation` calls the production
`build_droid_observation_from_inputs` builder. Setup coordinates are never
appended to language, robot state, or the returned observation. An explicit
request to feed object coordinates to either current adapter fails before the
observation is built. The GR00T end-effector position is robot proprioception,
not the task object's position.

A future compatible adapter must declare the `object_position_m.v1` schema and
its exact input key in adapter code. `policy_coordinate_fields` then requires a
separate deployment-known or deployment-estimated measurement, an explicit
deployment-availability evidence digest, the expected object/frame/source, and
a finite freshness bound. Simulator ground truth is prohibited as the selected
deployment source. Missing values never fall back to setup knowledge. The
projection receipt stays outside the policy payload. These checks bind declared
evidence; they do not themselves qualify a sensor, establish rights, or prove
that a source is available at a real site. That source's admission remains a
prerequisite for a future execution integration.

The interface choices follow the [OpenPI DROID input implementation](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/droid_policy.py)
and [GR00T DROID instructions](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/DROID/README.md).
Explicitly separating privileged state from visual observations also follows
the distinction documented by [ManiSkill observation modes](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html).

## Visual search is a declared test condition

`ObjectAcquisitionProtocol` supports `visible`, `partially_occluded`, and
`initially_out_of_view`. The baseline requires visibility in every declared
policy camera. Search has a finite time budget and a preregistered resolved
camera/sensor dimension; the canonical anchor must stay in baseline mode.
The policy starts immediately in search mode. This configuration adds no search
controller, privileged hint, pre-roll, or assumption that a checkpoint learned
to search.

`assess_object_acquisition` accepts initial absence only under the explicit
search protocol. It still rejects missing cameras, rendering/freshness failures,
calibration drift, mismatched episodes/objects/resets, non-increasing renderer
frames or physics steps, and changed simulator-time origins. The elapsed time
must equal observation simulator time minus the episode start time.

The result records first **observed** acquisition and the preceding observation
time, so the sampling interval is visible. It preserves a timeout or late
acquisition, and never treats acquiring the object as task success. Measurements
must come from independent simulator segmentation bound to the actual policy
frames; a policy cannot grade its own acquisition.

## Run the local staging example

The committed [request](examples/policy_object_acquisition/staging_request.v1.json)
and [plan](examples/policy_object_acquisition/staged_plan.v1.json) are synthetic
configuration fixtures: an invented workcell/block, placeholder source and
candidate digests, 20 cells, and 80 planned rows. The cells cover one direct
object-position dimension and the three visibility conditions. They are not
the active ten-cell run or evidence of execution. The existing harness retains
its canonical anchor, pairwise coverage, held-out partition, both controls, and
exactly two policies, giving all four subjects identical cell/reset/seed/setup
bindings.

From the repository root, select a new local output filename:

```bash
PYTHONPATH=src python -m blueprint_pipeline.policy_object_acquisition \
  --request docs/arm_decision_proof_v1/examples/policy_object_acquisition/staging_request.v1.json \
  --output /private/tmp/policy-observation-staged-plan.json
```

The command validates inputs and writes the output with create-only semantics.
It refuses to overwrite a prior artifact. It has no launch, upload, model-call,
or provider interface. The same operation is callable through
`stage_observation_protocol(ObservationStagingRequest)`.

## Bind a new native cell

The plan reports `configured_pending_native_binding`, `development_only: true`,
and `execution_authorized: false`. It is separate from paid execution authority.
`bind_native_cell_observation_protocol` consumes that plan, a matching new
native cell, the resolved native scene plan, the task-success contract digest,
and explicit camera setups. Each camera setup carries its parent-relative
OpenCV transform, expected world transform at reset, and source-intrinsics
digest. The binder validates the proposed geometry against the actual native
plan offline, before producing a `native_policy_observation_protocol.v1`
attachment. The staging CLI accepts the same arguments from
`--native-cell-request <json-path>` and writes the attachment create-only.

The synthetic [native-cell request](examples/policy_object_acquisition/native_cell_request.v1.json)
and its [validated binding](examples/policy_object_acquisition/native_binding.v1.json)
exercise this path for an initially out-of-view cell. Reproduce the binding by
adding `--native-cell-request docs/arm_decision_proof_v1/examples/policy_object_acquisition/native_cell_request.v1.json`
to the staging command and choosing a new output filename. Its declared reset
poses are fixture data, not native readback or deployable calibration.

The returned attachment belongs in a newly prepared cell's
`observation_protocol` field. Preparing and authorizing that new manifest
remains the existing launch workflow; this command never edits or re-seals an
active manifest. Cell ID, seed, scenario digest, cell-spec digest, native-plan
digest, source configuration, task-success contract, setup coordinates, camera
geometry, and acquisition condition are all bound together. The native
manifest validator rejects mismatches before runtime construction.

## Runtime behavior

The canary worker applies the bound camera mounts through its existing
scene-plan/CameraCfg path. It preserves the source intrinsics and bypasses the
automatic camera-mount search only for the explicitly configured mounts. Each
reset checks the actual object position, camera mount, intrinsics, and native
world pose. Camera mount/intrinsic checks continue during the episode; a moving
wrist is allowed to change its world pose after reset.
The intrinsic comparison explicitly uses the pinned SDK's aperture-center
coordinates (`width/2`, `height/2`), rather than silently treating the
scene-plan pixel-center convention as the native matrix. Only a new explicit
protocol clears the camera-preservation switches needed to apply its declared
source calibration; the ordinary DROID path retains its original calibration.

The native reader attaches RGB, bounding-box, and reference-time annotations to
the **existing policy render products**. It neither creates a new camera nor
advances simulation. Independent annotation RGB must equal the camera RGB used
by the production policy-input builder. Semantic masks follow that builder's
letterbox geometry; the exact processed RGB is checked before the acquisition
sample is retained. Native sensor counters, physics-step counters, render time,
and episode-relative simulation time must agree. Small pixel area is never
used as a substitute for occlusion evidence: partial occlusion uses the
renderer's bounding-box occlusion ratio, and an entirely occluded in-view
object is distinguished from an object outside the view.

Production construction/reset snapshots also read the counter from
`Camera.frame`, rather than the nonexistent `CameraData.frame`, and retain
`SimulationContext.get_physics_step_count()`. A missing native counter remains
`null`; these counters support freshness review without replacing the actual
image/semantic acceptance gate.

This follows the documented [Replicator annotation outputs](https://docs.omniverse.nvidia.com/kit/docs/omni_replicator/1.13.30/source/extensions/omni.replicator.core/docs/API.html#referencetime).
Structured bounding boxes are attached separately because the
[Isaac Lab camera buffers exclude these structured outputs](https://isaac-sim.github.io/IsaacLab/develop/_modules/isaaclab/sensors/camera/camera.html).

Before inference, the episode retains lossless native segmentation and processed
target masks, their byte digests, the exact policy-frame manifest binding, and
the acquisition sample. The episode receipt includes the acquisition
assessment. The worker re-reads those files and checks the sample/frame/receipt
bindings before accepting a completed episode. Failures preserve the available
acquisition evidence and an explicit blocker.

The initial task-visible predicate changes only for a digest-bound search cell
whose geometry, sensor freshness, and declared initial condition pass. The
receipt reports actual task visibility separately; it does not turn an empty
view into `target_semantic_visibility_passed: true`. Both required controls still
execute the same geometry and seeds. Baseline and qualified-evaluation gates,
renderer checks, complete media, scoring independence, and paid-resource gates
retain their existing requirements.

## Remaining native validation

The code and hermetic runtime wiring are implemented. No new GPU run was
performed for this PR. A newly authorized run must still demonstrate that the
pinned Isaac build supplies synchronized `ReferenceTime` and bounding-box
annotations on these render products, and that the configured camera geometry
produces the intended initial condition. Unsupported annotations, mismatched
RGB, stale time/counters, wrong calibration, or wrong geometry fail closed
before the affected policy query. The implementation does not infer native
success from its fake-Isaac tests.

Any future coordinate-aware policy still needs its own admitted adapter and
qualified deployment source. The two current DROID policy inputs are unchanged.
No camera-aim helper, paid allocator, active matrix/input packet, or deployed
host was changed. Operator-authorized camera aiming remains setup knowledge.
The new protocol rejects a cell that also requests operator camera re-aiming,
so two competing mount definitions cannot silently coexist. It is compatible
with the ordinary native wrist attachment that follows the configured mount.

## Verification

The focused information/acquisition tests cover unsupported coordinates,
source/frame/object/freshness/digest faults, byte-identical DROID observations,
baseline versus search behavior, stale camera rejection, sampled acquisition
timing, reset/placement/time-origin substitutions, paired harness bindings,
and the documented staging CLI's create-only output. Existing DROID-observation
and exact-workcell tests protect compatibility with the reused production seams.
The canary lifecycle rehearsal additionally drives both real client classes
through a bound search cell, including retained acquisition receipts and a
prepolicy invalid-frame refusal. Native reader tests inject only the documented
AOV transport and verify real geometry/frame/time/semantic validators. The
provider import-closure suite verifies the added runtime modules are shipped.
