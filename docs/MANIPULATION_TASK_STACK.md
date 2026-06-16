# Manipulation Task Stack

Blueprint manipulation tasks use object affordance contracts before any scored
pick/carry/place eval can run. The first supported reference task is:

```text
navigate_to_object -> pregrasp_stance -> reach -> close_grip -> lift -> verify_grasp -> carry_to_return_pose -> place -> release -> verify_placement
```

## Entry Point

```bash
blueprint-build-manipulation-task-stack \
  --capture-root <capture-root> \
  --object-class tote \
  --object-id simready_tote_001
```

To also execute the public Lucky Robots G1 reference assets:

```bash
blueprint-build-manipulation-task-stack \
  --capture-root <capture-root> \
  --run-lucky-reference-adapter \
  --fetch-lucky-reference
```

## Artifacts

The default stack writes:

- `assets/mujoco_tote_asset.xml`
- `assets/mujoco_tote_asset_manifest.json`
- `manipulation_object_contracts.json`
- `manipulation_task_request.json`
- `manipulation_policy_adapter_contract.json`
- `manipulation_policy_tier_matrix.json`
- `default_manipulation_policy_trace.json`
- `mujoco_manipulation_physics/manipulation_physics_output.json`
- `mujoco_manipulation_physics/manipulation_contact_manifest.json`
- `mujoco_manipulation_physics/mujoco_g1_manipulation_model_manifest.json`
- `mujoco_manipulation_physics/mujoco_tote_visual_mesh.obj`
- `mujoco_manipulation_physics/manipulation_video_manifest.json`
- `mujoco_manipulation_physics/manipulation_overview.gif`
- `manipulation_eval_report.json`
- `manipulation_stack_manifest.json`

When `--run-lucky-reference-adapter` is used, the stack also writes:

- `lucky_g1_reference_adapter/lucky_g1_reference_adapter_manifest.json`
- `lucky_g1_reference_adapter/lucky_g1_reference_trace.jsonl`
- `lucky_g1_reference_adapter/lucky_g1_reference_video_manifest.json`
- `lucky_g1_reference_adapter/lucky_g1_reference_overview.gif`

## Object Contract Rule

Random SimReady assets are not automatically scored manipulation targets. A
scored task requires an object contract with:

- asset URI and object pose
- mass, center of mass, friction, and inertia metadata
- collider intent
- grasp affordances and forbidden grasp zones
- lift, carry, drop, tilt, and placement thresholds

Unknown object classes fail closed until a contract or class template exists.
The current default class template is `tote.v1`. When no external tote asset is
provided, the stack materializes a local MJCF tote asset plus OBJ visual mesh
with body/rim contact geoms, mass, friction, center-of-mass metadata, inertia,
and pose so the default run is not a `simready://` placeholder.

## Policy Tiers

The stack supports three policy tiers:

- `default_phase_policy`: Blueprint reference baseline for phase/action trace and evaluator plumbing.
- `lucky_g1_reference_adapter`: public Lucky Robots G1 manipulation challenge adapter. It can fetch or use a local checkout, load `scene.xml`, `g1.xml`, mesh assets, `walker.onnx`, and `right_reacher.onnx`, then run a headless walk/reach/grip/release loop through MuJoCo and ONNX Runtime.
- `policy_api_endpoint`: robot-team VLA endpoint mode. Teams must return normalized attempts, phase traces, action traces, metrics, and artifact paths.

The existing robot-eval policy execution path also accepts:

```json
{
  "default_test_policy": {
    "policy_kind": "mobile_manipulation_pick_carry_place",
    "object_id": "simready_tote_001",
    "object_class": "tote"
  }
}
```

## G1 Manipulation Model

The MuJoCo tote proof command loads a Blueprint G1 manipulation proxy model with:

- actuated base, right-hand Cartesian stage, wrist, and gripper joints
- explicit joint ranges and actuator controls in the model manifest
- right palm and gripper pad contact geoms
- head and right-wrist cameras
- torso, pelvis, arm, foot, and hand geometry for review and collision context

This is a G1-named MuJoCo manipulation proxy for evaluator proof. It is not an
official Unitree dexterous-hand asset and not a physical robot readiness claim.

The Lucky adapter separately verifies the official public challenge assets:

- `walker.onnx` and `right_reacher.onnx` load and execute
- Lucky `g1.xml`, `scene.xml`, mesh assets, 43 actuators, right-arm joints, right-finger actuators, head camera, and wrist camera load
- scripted walk/reach/grip/release loop emits trace/video artifacts

## Physics Proof Boundary

The default phase policy is a reference trace, not grasp physics proof by
itself. Physics claims come from
`mujoco_manipulation_physics/manipulation_physics_output.json`.

When that artifact is `complete`, the stack may claim:

- MuJoCo simulator manipulation physics executed
- Blueprint's G1 manipulation proxy model loaded and drove actuated controls
- object lift occurred
- grasp/carry/place were validated under the configured MuJoCo abstraction
- contact samples and object pose traces were recorded
- a trace-derived review video was generated from physics telemetry

The physics trace records tote pose, controller targets, actuator controls,
joint state, base pose, actual end-effector pose, gripper/tote contacts, a
weld-grasp force proxy, lift height, drop/tilt/slip event flags, and placement
error metrics.

The first physics command uses an actuated G1 manipulation proxy plus a
weld-constraint grasp abstraction. During release, it opens the gripper,
disables hand contact geoms for withdrawal, and stabilizes post-release
xy/angular tote velocity so the placement metric reflects the commanded place
event rather than withdrawal scrape. Those abstractions are recorded in the
physics output claim boundary.

The Lucky adapter may claim only that the official Lucky walker/reacher assets
loaded and executed unless its own trace proves object contacts, lift, carry,
and placement. It currently keeps these stronger claims false:

- contact-only dexterous hand grasp validated
- full Unitree G1 dexterous hand policy proven
- official Lucky pick/place physics validated
- Blueprint tote task validated by Lucky assets
- robot-team policy quality proven
- physical robot readiness

Those require a richer official G1 hand/controller run, a Lucky walker/reacher
run that produces object/contact/lift/place evidence, or a team endpoint
returning execution evidence under the same manipulation task contract.
