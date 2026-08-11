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
a current explicit, task-scoped Newton GPU authorization plus the canonical
paid-resource admission, spend cap, TTL, watchdog, artifact storage, teardown, and
provider-inventory gates. One authorization may cover the bounded series of GPU
runs genuinely needed for this controls comparison, but every allocation still
requires a fresh immutable admission and zero retries. The normal gate requires
provider-zero. A concurrent run additionally binds every allowed live Vast
instance ID into both the Newton admission and allocator request;
RunPod, DigitalOcean, and all unlisted Vast resources must remain zero. The Vast
transport arms a separate name- and instance-bound hard-TTL watchdog before
object staging or compute allocation, and it validates the current canonical
paid-spend lock before even arming that watchdog. Any mismatch blocks storage
and compute. The contract itself does not authorize or launch a paid canary.

### Observed Newton canaries: 2026-08-11

One explicitly authorized controls-only canary at implementation commit
`c615284b41d151a1885dd38f2608c0ca8e0f4c71` failed closed before environment
construction with `adp009d_backend_runtime_version_mismatch`. The bare
`newton` Isaac Lab installer selector omitted the extension's pinned `all`
extra, allowing its visualizer dependency to resolve newer Newton, MuJoCo, and
MuJoCo-Warp packages. The worker now selects `newton[all]`, which binds the
versions declared by the sealed Isaac Lab source tree.

The terminal receipt is `adp009d_newton_canary_terminal.v1`, digest
`sha256:4cba321ed4d4121783d9435b8178643acdc06969bdc646bf7a0038222b01a694`.
It records a typed pre-controls media gap, zero policy queries, no verdict,
actual provider charge of `$0.321`, successful artifact retention and teardown,
zero retries, a digest for every retained evidence input, and a fresh API
inventory of zero RunPod, Vast, and DigitalOcean resources. This is comparison
evidence only; it is not controls parity, fidelity evidence, or engine-promotion
evidence.

A second controls-only canary at implementation commit
`c77ca07f204f235e3f3a5ea06d2201dd13d33de2` proved the exact pinned runtime
versions and reached Newton environment construction. It then failed closed
before reset because the PhysX regular-expression selector for the two Robotiq
inner fingers was not valid Newton `fnmatch` syntax, so Newton matched zero
contact-sensor bodies. Terminal receipt
`sha256:6fd97b7f7bcbcccc1008e156f345b247933cde3bcb546434526400df898c6fdf`
records the typed pre-controls blocker, zero policy queries, zero retries, a
settled `$0.173` provider charge, 20 retained artifacts, successful teardown, and fresh
API-confirmed provider-zero. The runtime now preserves the PhysX selector and
uses an equivalent Newton-only suffix glob selecting exactly the two terminal
finger bodies.

A third controls-only canary at implementation commit
`fb59e16b4905eca4bcb1e0537a6d191b29adc572` proved that two-body Newton
contact-sensor selection now succeeds. It then failed closed before reset when
the per-finger can filter used the PhysX spawn label `approved_can`, which
matched no Newton counterpart. The sealed USD's sole authored rigid body is its
`canned_beverage` root.
Terminal receipt
`sha256:9816ed8355a38e11111eb87d5adf5d63735470f92f86e8779a1e4fd28412e901`
records the typed blocker, zero policy queries, zero retries, a settled `$0.133`
provider charge, 20 retained artifacts, successful teardown, and fresh
API-confirmed provider-zero.

A provider-free bundle and admission attempt at implementation commit
`b59836ea97963077335cbc178471feb367577f70` was then rejected before storage or
compute mutation because the canonical paid-spend lock path was absent from the
operator environment. Its adapter receipt records
`provider_mutations_performed: 0`; the corrected invocation used a new evidence
root and an explicitly bound lock rather than modifying the failed attempt.

The fourth controls-only canary at that same commit proved the exact source
body label `canned_beverage` also matches no Newton counterpart. Terminal receipt
`sha256:7a420975420b88841ac3b1b1c3dd88699140514888caa4f9f44efaba3b1a138a`
records the typed pre-controls blocker, zero policy queries, zero retries, a
settled `$0.148` provider charge, 20 retained artifacts totaling 277,996 bytes,
successful teardown, and fresh API-confirmed provider-zero. The repair keeps
PhysX's existing body filter unchanged and uses Newton's native shape-level
filter for the sealed can's exact authored `body_collider`; static SAGE remains
separately filtered through the exact 15 shape labels bound by the sealed task
collision manifest. Any subsequent Newton sensor-build failure retains the
finalized model's bounded body/shape label diagnostics, so another label is not
guessed blindly. A new native canary must still prove both partner force
matrices before controls begin.

Newton remains comparison evidence only until both backends achieve evidence
parity and an independently meaningful deterministic fidelity result exists.
Even then, the receipt only makes a promotion review eligible; it never promotes
an engine automatically.
