# ADP-009D physics-backend comparison

This is a controls-only Day-28 precursor for the sealed InteriorGS/SAGE `840313`
Franka/Robotiq scene. It does not run a learned policy, rank candidates, claim
task success, establish physical truth, or change the production engine.

The immutable design is
[`manifests/adp009d_physics_backend_comparison.v1.json`](manifests/adp009d_physics_backend_comparison.v1.json).
Every run selects exactly one strict `physics_backend` value, `physx` or
`newton`, at simulation construction. Mid-run switching is forbidden. Website
profiles continue to select `physx` explicitly.

## Backend profiles

PhysX is the production baseline. Its existing TGS configuration, native SDF
overlay inspection, collision-cooking diagnostics, contact readback, and
receipt path remain intact.

Newton is an experimental comparison candidate. The sealed profile binds Isaac
Sim `6.0.0-dev2`, IsaacLab-Arena revision
`8b4a3a47fc53de23e8205089d71109a2e2348acd`, Isaac Lab revision
`e57379c634b42db5a0fe9f754341be6e2a7c7c43`, `isaaclab_newton==0.5.9`, Newton
revision `2684d75bfa4bb8b058a93b81c458a74b7701c997`, `warp-lang==1.12.0`,
`mujoco==3.5.0`, and `mujoco-warp==3.5.0.2`. It uses `NewtonCfg` with an
`MJWarpSolverCfg`, including an explicit `nconmax=1024`, and retains the
converted MJCF digest. These pins follow the selected Isaac Lab source tree;
they are not floating recommendations.

Newton loads the sealed generic USD source and never loads the PhysX SDF
overlay. Its contact model and planner allowance are separately explicit. A
Newton probe blocks on any PhysX-only field, unsupported or ignored setting,
missing Franka/Robotiq actuation, missing camera/media path, missing contact
force or partner readback, contact-buffer overflow, schema or digest drift, or
unretained asset conversion.

Isaac Lab documents the backend factory and tensor architecture, Newton's
experimental status, and the MJWarp contact buffer in its
[multi-backend architecture](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/overview/core-concepts/multi_backend_architecture.html),
[backend overview](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/overview/core-concepts/physical-backends/index.html),
and [MJWarp solver reference](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/overview/core-concepts/physical-backends/newton/mjwarp-solver.html).

## Evidence and admission

Provider-free profile validation proves only that the proposed bytes are
internally consistent. A native probe must separately prove the enabled backend,
source and scene bindings, imports, actuation, cameras, contacts, forces,
partners, solver configuration, conversion, and claim boundaries.

The comparison binds the same source scene, object, robot, scenario cell, seed,
intended grasp geometry, and semantic controls plan. Backend execution-plan
digests and contact configurations remain separate. Each terminal run must
retain initialization/reset, target and robot poses, contacts and force vectors,
torque utilization and clipping, closest geometric clearance, action delivery,
phase completion, lossless policy-input-equivalent frames, review media,
teardown, spend, and API-confirmed provider-zero.

A Newton `--execute` request is rejected before provider mutation unless it has
a current explicit Newton canary admission plus the canonical paid-resource
admission, spend cap, TTL, watchdog, artifact storage, teardown, and
provider-inventory gates. The normal gate requires provider-zero. A concurrent
run is admitted only by a fresh explicit authorization that binds every allowed
live Vast instance ID into both the Newton admission and allocator request;
RunPod, DigitalOcean, and all unlisted Vast resources must remain zero. The Vast
transport arms a separate name- and instance-bound hard-TTL watchdog before
object staging or compute allocation, and it validates the current canonical
paid-spend lock before even arming that watchdog. Any mismatch blocks storage
and compute. The contract itself does not authorize or launch a paid canary.

Newton remains comparison evidence only until both backends achieve evidence
parity and an independently meaningful deterministic fidelity result exists.
Even then, the receipt only makes a promotion review eligible; it never promotes
an engine automatically.
