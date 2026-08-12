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
records the typed blocker, zero policy queries, zero retries, a `$0.133`
provider charge at terminal-receipt time (`$0.149` after final API settlement),
20 retained artifacts, successful teardown, and fresh
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
`$0.148` provider charge at terminal-receipt time (`$0.156` after final API
settlement), 20 retained artifacts totaling 277,996 bytes,
successful teardown, and fresh API-confirmed provider-zero. The repair keeps
PhysX's existing body filter unchanged and uses Newton's native shape-level
filter for the sealed can's exact authored `body_collider`; static SAGE remains
separately filtered through the exact 15 shape labels bound by the sealed task
collision manifest. Any subsequent Newton sensor-build failure retains the
finalized model's bounded body/shape label diagnostics, so another label is not
guessed blindly. A new native canary must still prove both partner force
matrices before controls begin.

A fifth controls-only canary at implementation commit
`22ef94a7ee15466404ea9899a86b2cd12917da7c` reached the pinned Newton/MuJoCo
model conversion before the new shape filter was evaluated. Conversion failed
closed because all nine moving Robotiq bodies in Arena's flattened DROID USD
lacked positive mass and inertia; the first named body was
`left_outer_knuckle`. Terminal receipt
`sha256:c2e7de0cc4a6fc6b8077a5a2971093e5a4c5510ccd13c784143cdc67719cdc73`
records zero controls and policy queries, a typed pre-controls media gap, 20
retained artifacts totaling 280,242 bytes, a `$0.148` charge at
terminal-receipt time (`$0.172` after final API settlement),
successful teardown, and fresh API-confirmed provider-zero.

A sixth controls-only canary at implementation commit
`8acd5b55e0a037ca2744a21ede46bbc8269afdbc` proved the Newton-only inertial
overlay is admitted and reached embodiment configuration in the exact pinned
runtime. It then failed closed before robot spawn because Isaac Lab 4.5.24
stores `UsdFileCfg.func` as a lazy `ResolvableString`; generic `__wrapped__`
inspection intentionally does not resolve that reference. Terminal receipt
`sha256:2da3ea2f428c0b28cfa3c38cf3be88f71797a9ff3bd97c79766b8a5c024dffa2`
records zero controls and policy queries, a typed pre-controls media gap, 20
retained artifacts totaling 238,891 bytes, a `$0.159` charge at terminal-receipt
time (`$0.182` after final API settlement),
successful teardown, and fresh API-confirmed provider-zero. The repair uses
Isaac Lab's public `string_to_callable` resolver only after requiring the exact
pinned `spawn_from_usd` reference, re-verifies the resolved function's module
and name, and then requires the official `@clone` wrapper before unwrapping it.
Any other reference or wrapper shape remains fail closed.

A seventh controls-only canary at implementation commit
`d1af040beda4759efc0451b37f44322f2815b85c` proved the repaired spawn wrapper,
full environment construction, five native contact-sensor instances, all three
320x180 camera paths, two deterministic resets, zero-action delivery, and 40
camera warm-up frames. It then failed closed before scripted controls because
all seven arm-joint positions became non-finite:
`canonical_hold_arm_pose_drift:maximum_error_rad=nan`. The terminal receipt,
digest
`sha256:e614fe638aeaf57e6ab8b6928a3a43c52ac187ce6ed27814b29f0d59aa30be68`,
records zero policy queries, no candidate outcomes, 22 retained artifacts
totaling 4,906,480 bytes, exact settled provider charge `$0.280`, successful
teardown, and fresh API-confirmed provider-zero.

Inspection of the exact digest-bound DROID USD showed that every Franka link
from `panda_link0` through `panda_link7` authors its collision mesh with uniform
scale `0.01`, while its diagonal inertia already contains that scale-squared
factor twice. Newton preserves those near-zero inertias and the initially valid
reset state evolves to NaNs. The Newton-only repair admits exactly one unit
conversion: each of the eight exact source diagonal-inertia vectors is divided
by `0.01^2`. It validates the source mesh paths, collision APIs, scales, masses,
centers of mass, principal-axis authoring, and exact source inertia values
before applying the correction; validates the exact corrected values after;
and rejects any asset, body-set, scale, or value drift. No arbitrary minimum
inertia clamp or replacement robot model is allowed. This conversion remains
comparison-only asset adaptation, not independently meaningful fidelity
evidence.

An eighth authorized controls-only canary at implementation commit
`7f5602359b271e299fa897cf1a0f155da735a991` allocated Vast instance
`47498893`, but the provider container never materialized an on-start log and
the instance exited at the bounded no-progress check. It therefore produced no
native runtime or inertia-conversion evidence. The terminal receipt, digest
`sha256:c342ef55ee175367608ad0cc8cc940cad737c6d9b3a32f1cc32635e19e061ad4`,
retains the typed `pre_runtime_blocked` gap, the observed missing
provider-runtime artifact role, two control-plane/teardown artifacts totaling
17,856 bytes, exact settled provider charge `$0.262`, successful destruction,
zero retries and policy queries, and fresh API-confirmed provider-zero. Machine
`144209` is retained in the run-local avoidlist with reason
`vast_startup_control_plane_did_not_reach_onstart_heartbeat`. This is a provider
startup failure only; it does not confirm or refute the Franka inertia repair.

The terminal compiler accepts this pre-runtime case only when no native result
exists, the allocator explicitly records both missing provider output and
runtime non-completion, the artifact manifest is missing exactly the runtime
evidence role, and teardown, exact billing, and provider-zero still validate.
It never synthesizes a native result to close that gap.

The admitted repair is Newton-only and digest-bound. It verifies the exact
Arena DROID USD bytes, requires the exact nine massless Robotiq rigid bodies and
at least one collider per body, then authors their physically sourced masses in
the live session layer before Newton finalizes its model. Robotiq center of mass
and inertia are derived from the target USD's own collision geometry and scaled
to each mass by the
[pinned Newton importer](https://github.com/newton-physics/newton/tree/2684d75bfa4bb8b058a93b81c458a74b7701c997);
URDF frame-dependent COM or inertia is not copied. The mass source is the
BSD-licensed Robotiq 2F-85 URDF at revision
`a65190bdbb0666609fe7e8c3bb17341e09e81625`, and both source digests, all nine
masses, collider coverage, eight exact Franka source and corrected inertia
vectors, applied properties, and the unmodified source flag are retained in an
immutable runtime receipt. Asset drift, pre-existing Robotiq mass
authoring, missing colliders, extra/missing bodies, arbitrary minimum-inertia
clamps, or receipt drift block the run. The same pre-import receipt retains the
exact PhysX-authored properties that Newton's pinned resolver maps and blocks
every other authored PhysX value before model import; Arena's PhysX solver
iteration/depenetration overrides and `PhysxContactReportAPI` activation are
disabled for Newton. PhysX does not use this overlay.

On 2026-08-11, the ninth bounded Newton controls-only canary ran at clean
implementation commit `ab416507f25afbf604cf1f3d42196082cdacda0b` on Vast
instance `47501029` (L40S). It completed native environment construction, two
resets, zero-action delivery, and all 40 camera-warmup frames with the corrected
Franka inertia overlay. It then failed the canonical post-warmup arm hold, before
any scripted control, policy query, candidate outcome access, or task verdict:
`canonical_hold_arm_pose_drift:maximum_error_rad=0.853033662`. The retained
telemetry identifies the largest deviation at `panda_joint4` (requested
`-2.513274193`, observed `-1.660240531` radians). The exact settled Vast charge
is `$0.194`; the instance was successfully destroyed and its independent
watchdog observed provider absence.

The native log also records that Isaac Lab ignored Arena's legacy implicit
actuator `effort_limit` and `velocity_limit` fields. The next provider-free
profile therefore binds an exact Newton-only mapping of those already-authored
limits to `effort_limit_sim` and `velocity_limit_sim`, clears the ignored legacy
fields, and retains an application receipt. It is a compatibility mapping, not
an actuator retune or a fidelity claim. The revised Newton profile digest is
`sha256:8f46f2c5e68cffb0d5a7b36b9b7b54b49e4d0e1c5e6e3caeb45cbdaf6a12290f` and
the revised provider-free comparison-design digest is
`sha256:f86ea6e006a2a7abf1820696714b976d2a447f1cfa32c06d926b5a367a9e18d0`.

No terminal canary receipt is compiled for this ninth run yet. A fresh,
read-only global post-teardown inventory correctly found two pre-existing active
DigitalOcean production/control-plane droplets, even though Vast and RunPod were
zero. The terminal compiler now requires a digest-bound, API-confirmed global
provider-zero receipt generated after teardown, so it refuses to treat the
earlier pre-launch inventory as closure evidence. Those droplets are outside
this Newton allocation and have not been altered. No replacement GPU may be
admitted until an authorized operator resolves that global provider-zero gate.

Newton remains comparison evidence only until both backends achieve evidence
parity and an independently meaningful deterministic fidelity result exists.
Even then, the receipt only makes a promotion review eligible; it never promotes
an engine automatically.
