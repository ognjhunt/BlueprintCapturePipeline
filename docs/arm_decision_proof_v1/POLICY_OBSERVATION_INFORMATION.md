# Setup knowledge, policy input, and object acquisition

This implements the configuration portion of **ADP-009D / day 14**: a paired
policy test must declare what information the policy receives and whether the
object should already be visible. The existing DROID builder selects its inputs
correctly, but it had no explicit deployment-source contract for object
coordinates or a separately preregistered visual-search protocol. The smallest
change is a staging adapter around that builder and the existing exact-workcell
schedule. It does not modify the current frozen GPU run.

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

## Execution integration remains staged

Every generated plan says `staged_runtime_integration_required`,
`development_only: true`, and `execution_authorized: false`. It is a separate
artifact, not an existing launch/execution spec. Before a separately authorized
search experiment, the runtime still must:

1. Apply the newly preregistered camera/occlusion setup and read back its actual
   geometry. The new visibility dimension is deliberately not registered as an
   executable native target by this change.
2. Bind independent visibility samples to the exact lossless frames actually
   passed to the policy, including reset and episode-time origin, and retain
   the acquisition assessment in the episode receipt.
3. Select the explicit search predicate only for those bound search cells.
   Baseline and qualified-evaluation gates, camera validity/freshness, paired
   controls, and full media requirements remain mandatory.
4. Wire any future coordinate-aware adapter through its qualified deployment
   source. The current DROID policies continue to receive their documented
   inputs.

No camera-pose runtime, policy client, provider bundle, paid allocator, live
matrix, or deployed host is changed here. Operator-authorized direct camera aim
remains setup knowledge; it does not imply that object XYZ is a policy input.

## Verification

The focused information/acquisition tests cover unsupported coordinates,
source/frame/object/freshness/digest faults, byte-identical DROID observations,
baseline versus search behavior, stale camera rejection, sampled acquisition
timing, reset/placement/time-origin substitutions, paired harness bindings,
and the documented staging CLI's create-only output. Existing DROID-observation
and exact-workcell tests protect compatibility with the reused production seams.
