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

If a typed runtime blocker prevents a nonterminal measurement, the control-run
receipt retains that field as `not_reached` and binds it to the exact blocker;
it never substitutes a nominal value. Teardown, settled spend, and fresh global
provider-zero remain mandatory and cannot be represented as measurement gaps.
An independently meaningful fidelity result may likewise remain `not_measured`
only when it cites an exact typed backend-run blocker. That yields a valid
blocked comparison receipt, never evidence parity or promotion eligibility.

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
`sha256:6ac1a23917bf009c4a6e406b4914aa18007e0d5fb2dc69ffbc69611114856ec1`.

The first dedicated post-teardown inventory correctly blocked terminal closure
on an active DigitalOcean control droplet. Under explicit task-scope
GPU/spend authority, the exact remaining blocker, DigitalOcean instance
`584554642` (`blueprint-gpu-control-prod-01`), was re-confirmed and terminated.
The fresh post-teardown, API-confirmed global provider-zero receipt covers
DigitalOcean, RunPod, and Vast, records zero live instances and zero hourly
burn, and has digest
`sha256:eccceaf80391bef32c381b7110b94fa9aba7a1052c728c6c24f42c71e63d5146`.

The terminal compiler now seals the retained ninth-run evidence in receipt
`sha256:a29f546e7e38a86e04a6eb1f847b60414c7ec8c1502cc5d06af9a994351b97ef`.
It binds the admitted profile and source bundle, the 22-file artifact manifest,
the successful Vast teardown, the exact `$0.194` settled instance charge, and
the API-confirmed provider-zero receipt. Its status is deliberately
`blocked` with scientific phase `pre_controls_blocked`, zero policy queries,
no candidate outcomes, no task or physical verdict, and an explicit media gap
caused by the failed canonical hold. This closes the allocation lifecycle only:
the ninth run did not reach controls, and its finite drift does not resolve the
earlier Newton NaN result, which remains the scientific blocker requiring a
future independently meaningful retest.

Newton remains comparison evidence only until both backends achieve evidence
parity and an independently meaningful deterministic fidelity result exists.
Even then, the receipt only makes a promotion review eligible; it never promotes
an engine automatically.

## Root cause of the Newton hold drift: the robot is weightless under PhysX

The hold drift is not a Newton solver defect and not an actuator tuning gap. It
is a physics property whose semantics only one backend can express, diagnosed
entirely from retained evidence of the ninth run at no additional GPU cost.

The sealed Franka/Robotiq asset authors `physxRigidBody:disableGravity` on all
eighteen robot bodies — every Panda link and every Robotiq body — and the
runtime configuration confirms `RigidBodyPropertiesCfg(disable_gravity=True)`.
PhysX honours that: its arm carries no weight, which is why the PhysX run at the
bitwise-identical canonical pose, timestep, decimation and forty-frame warmup
held to `4.649162292480469e-06` rad.

The Newton model mjwarp actually simulated is retained as
`newton_converted_model.xml` (the runtime's `save_to_mjcf` target). It contains
no `gravcomp` attribute and no other expression of per-body gravity disable, so
its arm carries the full 18.28 kg. Recomputing from that exact model, the
gravity torque at the canonical pose is `20.070` N·m at `panda_joint4`. The
converted model drives every arm joint with `gainprm=400`, `biasprm="0 -400 -80"`
— an explicit PD at kp=400 — so the steady-state droop alone is
`20.070 / 400 = 0.0502` rad, five times the `1.0e-2` hold gate, before any
transient. Newton could not have passed that gate as configured, by arithmetic.
The observed `0.853033662` rad is that droop plus an underdamped transient
(`kd=80` against zero joint armature), and the earlier NaN is the same mechanism
at a more marginal operating point.

Toggling only gravity on the identical retained model, with contacts and the
Robotiq equality constraints isolated, flips the hold from `0.000000000` rad to
`2.504` rad, and the joints that drift are exactly the joints that drift in the
paid run: `panda_joint4` largest, then `panda_joint2` and `panda_joint6`, with
`panda_joint1/3/5/7` unaffected.

`physxRigidBody:disableGravity` was listed in
`NEWTON_MAPPED_PHYSX_PROPERTY_NAMES`, which asserted that Newton gives the
property semantics. That assertion was false, and it is the specific defect: a
dynamics-changing PhysX property was silently ignored, so the divergence
surfaced as a downstream hold failure that reads like a Newton stability problem
instead of as a typed non-comparability blocker. Blocking the attribute would
not have helped either — it would leave PhysX weightless and Newton loaded, which
is the same divergence with no error.

The property is now classified in `NEWTON_UNREPRESENTABLE_PHYSX_PROPERTY_NAMES`
and `validate_newton_dynamics_representable` fails closed with typed blocker
`adp009d_newton_unrepresentable_physx_property:physxRigidBody:disableGravity`
before a Newton allocation does any work. The revised Newton profile digest is
`sha256:ab6ddfae0dc9fe6e0b9901e3950c2992245f161ee2a1461b9b5db17f957a42de` and
the revised provider-free comparison-design digest is
`sha256:7beac1cf97ab655a5039fc28a4393bfba7f0344d235770f0048ae684ad650a3f`.

Two consequences follow, and neither is a Newton verdict.

First, no bounded Newton retest is scientifically justified until the gravity
semantics are settled, because a retest of the current configuration would only
re-measure the same droop.

Second, and more important for the programme than the Newton question: the
production PhysX baseline simulates a robot arm that does not carry its own
weight. Every joint torque, contact force and clearance measurement collected
under that baseline was measured on a weightless arm. That is a fidelity
property of the current default backend, it is not caused by Newton, and it
bounds what the existing PhysX controls evidence can claim about real
manipulation. Making the two backends comparable requires deciding whether the
arm should carry its weight at all; that decision changes the PhysX baseline,
not only the Newton lane.

## Second, independent Newton defect: the Robotiq drive is explicitly unstable

Separating the arm from the gripper in the retained model isolates a second
defect that the gravity finding was masking. With the gripper actuator frozen,
the arm under full gravity tracks first-principles theory exactly — simulated
steady-state error matches the analytic `gravity_torque / kp` to three
significant figures at every gain tested:

| kp | simulated hold error (rad) | analytic `20.070 / kp` |
| --- | --- | --- |
| 400 | 0.049519 | 0.050175 |
| 1600 | 0.012490 | 0.012544 |
| 2400 | 0.008347 | 0.008363 |
| 8000 | 0.002500 | 0.002509 |

With the gripper actuator live, every arm gain diverges to ~2.5 rad. The
converted model drives `finger_joint` with `gainprm=5729.58` and
`biasprm="0 -5729.58 -0.0114592"` against Robotiq knuckle inertias of
`3.80173e-07` kg·m². That is a natural frequency of
`sqrt(5729.58 / 3.80173e-07) ≈ 1.23e+05` rad/s; at `dt = 1/120` s the
explicit-integration stability ratio `ω·dt` is `≈ 1.0e+03` against a limit of 2 —
roughly five hundred times over. PhysX solves the same declared drive as a
solver constraint and is unconditionally stable, so it never surfaced there.
This is the mechanism behind the earlier canary's `NaN` at the gripper DOF.

Note that `5729.58 = 100 × 180/π` exactly, and `0.0114592 = 0.657 × π/180`, so
the pair is consistent with a per-degree value converted in one direction for
stiffness and the opposite direction for damping. That is the same duplicated
unit-conversion signature already corrected once in this programme for the Franka
diagonal inertias. It is flagged here as a strong indication, not a proven
provenance; the source authoring must be inspected before any correction.

## Gravity-real is the chosen resolution

The programme decision is to make **both** backends gravity-real rather than to
make Newton reproduce PhysX's weightless arm. That is the physically correct
choice for a fidelity comparison, and it makes the existing PhysX baseline —
not only the Newton lane — the thing that has to be re-validated.

Three consequences are already decidable without a provider:

1. `disable_gravity=True` must come off the robot rigid-body properties, which
   changes the PhysX baseline. Every PhysX joint-torque, contact-force and
   clearance measurement collected to date was taken on a weightless arm and
   does not carry over.
2. The shipped `kp=400` cannot satisfy the `1.0e-2` rad canonical hold gate once
   the arm carries weight: droop alone is `0.0502` rad. Either the arm gains
   rise to about `kp≈2400` (which holds at `0.00835` rad with the required
   `20.07` N·m well inside the `87` N·m effort limit) or the hold gate is
   restated as a stability-and-bounded-droop criterion. That is an actuator
   decision affecting both backends and must be declared, not tuned silently.
3. The Robotiq drive must be resolved independently before Newton can execute
   controls at all, because it is unstable with or without gravity.

`validate_newton_explicit_pd_feasibility` now decides all three from the
declared gains, the converted-model inertias and the measured gravity torques,
and fails closed before a provider allocation. Run against the exact ninth-canary
configuration it returns `blocked` with
`adp009d_newton_hold_gate_unreachable_by_droop:panda_joint4`, and against the
Robotiq drive `adp009d_newton_explicit_pd_unstable:finger_joint`. Both paid
canaries were therefore decidable as failures for free, before launch.

No further Newton allocation is justified until the gravity-real actuator
decision and the Robotiq drive are settled, because the outcome of such a run is
already known by arithmetic.

## Gravity-real actuation is now the configured contract

`build_gravity_real_actuation_contract` binds the decision and
`_configure_gravity_real_actuation` applies it for **both** backends before the
backend-specific configuration runs. It clears `disable_gravity` in the spawn
configuration only — the sealed source asset is never mutated — and raises the
arm gains to `kp=2400`, `kd=196`.

The damping follows the stiffness as `sqrt(kp)`: `80 × sqrt(2400/400) = 195.96`,
rounded to `196.0`, which preserves the shipped damping ratio to within 0.02%.
This is therefore a stiffness decision forced by the arm now carrying weight, not
a re-tune of the control character. Predicted worst-joint droop is
`20.070 / 2400 = 0.0083625` rad against the `1.0e-2` rad gate, and the measured
value on the retained model is `0.008347` rad with `22.1` N·m peak against
`panda_joint4`'s `87` N·m limit — `65` N·m of headroom. The superseded
`kp=400` is retained in the contract precisely so the `0.0502` rad droop that
made the gate unreachable stays on the record.

Because the contract sits in the shared section of `build_backend_profile`, both
profile digests moved:

- PhysX profile `sha256:82c22625b895baeffee93482255f29b029faaaac36f4b3ff1939914436749c9f`
- Newton profile `sha256:4d0c5e92e76e174439710146029bd5af08eb148c8f74dc1a9e6002164324f254`
- Comparison design `sha256:9253863575f517863667205cba08a6b879d0b3a29570dc1398351062ce1740d2`

`prior_weightless_evidence_carries_over` is `false` in the contract. The PhysX
controls evidence collected before this change was measured on a weightless arm
and is superseded, not merely re-labelled.

### Execution order

1. **PhysX baseline run.** PhysX is unaffected by the explicit-PD instability, so
   it validates the gravity-real configuration first and re-establishes the
   controls baseline that the previous weightless evidence no longer provides.
2. **Robotiq drive resolution.** Required before Newton can execute controls at
   all; it is unstable with or without gravity and needs the source authoring of
   the `5729.58 / 0.0114592` pair inspected.
3. **Newton run**, once `validate_newton_explicit_pd_feasibility` admits the full
   drive set rather than only the arm.
4. **Comparison**, on the shared gravity-real physical system.

## Limits of the local replica, and what it does not establish

The CPU replica used above loads the exact retained `newton_converted_model.xml`
in MuJoCo 3.11. It is authoritative for what the model *contains* — masses,
inertias, centres of mass, drive gains, effort limits, equality constraints,
gravity — and for analytic quantities derived from it. It is **not** a faithful
reproduction of the paid mjwarp run: the paid Newton canary drifted `0.853` rad,
while the replica diverges to `2.4`–`5.5` rad in every configuration except one.
Its solver settings, contact handling and constraint treatment differ from
mjwarp's, so it must not be used to claim that a given change would or would not
fix Newton on the provider.

What therefore stands independent of the replica:

- The gravity root cause, because it is arithmetic: `20.070 N·m / kp=400 =
  0.0502` rad against a `1.0e-2` rad gate. It is corroborated by PhysX's measured
  `4.649e-06` rad, which is impossible for an arm carrying weight at that gain.
- The Robotiq drive's explicit-integration ratio, also arithmetic:
  `sqrt(5729.58 / 3.80173e-07) ≈ 1.23e+05` rad/s, giving `ω·dt ≈ 1.0e+03` at
  `dt = 1/120` s against a limit of 2.
- `5729.58 / (180/π) = 100.0000` exactly, so the gripper stiffness is a
  per-degree value of 100 correctly converted to per-radian.

What is **not** established, and must not be assumed:

- That the Robotiq drive damping is a unit inversion. `0.0114592 × (180/π)² =
  37.62` looked like the same duplicated-conversion signature as the Franka
  inertias, but restoring it changes nothing in the replica. **Falsified as a
  fix; unresolved as a provenance question.**
- That gripper armature resolves it. Swept `1e-5` … `1e-1`; none passed.
- That reducing gripper stiffness resolves it. Swept `5729.58` → `1.0`; none
  passed.
- That finer integration resolves it. Swept `dt` to `1/7680`, a 64× reduction;
  none passed.

Zeroing the gripper actuator entirely *does* let the gravity-real arm hold at
`0.008347` rad, so the gripper is implicated — but since no physically meaningful
gripper change reproduces that, the replica cannot say which. **The Robotiq
resolution therefore requires the real Newton runtime, not this replica**, and no
Newton allocation should be launched claiming a fix is in hand.

## First gravity-real PhysX baseline: the arm holds its own weight

Vast instance `47519696` (L40S, machine 27268, $0.6794/hr) ran the gravity-real
PhysX controls configuration under a `$2` cap, `5400` s TTL and retry cap `0`.
Settled charge `$0.112`; 36 artifacts retained; teardown completed and
API-confirmed provider-zero across RunPod, Vast and DigitalOcean
(`sha256:85b184c02f9e3fe774e13f0370dd7cfe42e52fddb6a5c349e8b8d59618dda36e`).

**The canonical hold passed at `0.008284330368041992` rad**, inside the
`1.0e-2` rad gate. This is the first ADP-009D run in which the Franka held its
commanded pose while actually carrying its own weight.

The measurement matches the prediction made from the retained model before the
run, to within one percent:

| source | worst-joint hold error (rad) |
| --- | --- |
| analytic `20.070 / 2400` | 0.0083625 |
| CPU replica | 0.008347 |
| **measured, PhysX on GPU** | **0.008284** |

Per joint, the measured error is the analytic `gravity_torque / kp` throughout,
which closes the root-cause argument quantitatively rather than by narrative:

| joint | `τ_g / 2400` | measured |
| --- | --- | --- |
| panda_joint4 | 0.008363 | 0.008284 |
| panda_joint2 | 0.002594 | 0.002805 |
| panda_joint6 | 0.000331 | 0.000351 |
| panda_joint1/3/5/7 | ~0 | 1e-06 … 1e-04 |

The retained `canonical_hold_trace` records `convergence: settled` and
`hold_failure_mode: within_tolerance`, with per-step error rising
`0.00468 → 0.00679 → 0.00767 → 0.00803` and converging. That distinction —
settled at a bounded offset versus still falling — is exactly what the trace was
built for, and it is the first run in which it had a passing hold to describe.

### Controls did not complete, and the cause is a gravity assumption in the replay

The run stopped with `scenario_controls_receipt_missing`, from
`wrist_observable_episode_start_restore_failed:wrist_episode_start_restore_joint_mismatch`.
No policy was queried and no candidate outcome was accessed.

The episode-start replay computed each command as
`achieved + clamp(target - achieved)`. Its fixed point is `command == target`,
therefore `achieved == target - droop`: with a weightless arm the droop is zero
and the law is correct, but a gravity-real joint settles a steady-state droop
below whatever it is commanded, so the replay lands a full droop short of the
pose it is replaying. The early exit at `tolerance / 3 = 1.0e-3` rad can never
trigger against a `0.0083` rad droop, so the loop exhausted its horizon and the
final `3.0e-3` rad check failed at the droop.

Widening the tolerance would have hidden this: the replay would still be
restoring the wrong pose, and the wrist observability it exists to guarantee is
defined on the achieved pose, not the commanded one. The fix is the servo law.
`next_episode_start_restore_command` integrates the command from its own previous
value, making `achieved == target` the fixed point with the command carrying the
droop. With a weightless arm the two laws agree exactly, so previously validated
behaviour is unchanged.

This is the second gravity assumption the change has surfaced, after the arm
gains themselves, and both were invisible while the arm was weightless.

## Second gravity-real PhysX run: the replay fix holds, a third gate remains

Vast instance `47520665` (L40S, machine 137572, $0.7644/hr) ran the same
gravity-real PhysX controls configuration at `0a36cc7c0`, under the same `$2`
cap, `5400` s TTL and retry cap `0`. Settled charge `$0.19`; teardown completed;
API-confirmed provider-zero
(`sha256:8c2a47d58b268e4bb5bfc99d464e52834fd28ed5ae2a884b695aae52cc4dd52a`).

Two results carry forward.

**The hold reproduced bit-identically** at `0.008284330368041992` rad across two
independently allocated instances on different machines. The gravity-real
configuration is deterministic, not a lucky sample.

**The episode-start replay fix works.** The restore receipt returns
`blockers: []`, and the per-joint replay error is now
`[1.85e-05, 2.60e-04, 4.95e-06, 0.0, 2.00e-06, …]` — `panda_joint4` lands
exactly on target where it was previously a full `0.0083` rad droop short. The
integrating command law converges the achieved pose, which is what the wrist
observability guarantee is defined on.

The run then stopped at
`adp009d_backend_native_probe_invalid:adp009d_backend_probe_contact_readback_invalid`,
with no policy queried and no candidate outcome accessed.

The cause is exact and is a contract-versus-emitter mismatch, not a physics
failure. `validate_backend_probe` requires `contact_readback.partner_prim_paths`
to be a non-empty list. The runtime emits `partner_filter.filter_prim_paths_expr`
— the filter that was *requested* — and never the partner prims that were
*resolved*. Those are different claims, and the contract is right to want the
second: `contact_partner_readback` means being able to say which body a measured
force acted against, which a filter expression does not establish.

The measured force vectors themselves are valid and were not the problem. They
are `[[0,0,0],[0,0,0]]` because the closest geometric clearance at probe time is
`0.29880005736181087` m, so nothing is in contact; the validator correctly
accepts finite zeros.

Fixing this requires reading the resolved filtered prim paths back from the
pinned Isaac Lab contact sensor. The exact attribute for the pinned revision was
not verified, and this programme has already spent one paid run on a guessed
prim-path API. No further allocation should be made until that attribute is
confirmed against Arena's own source at the pinned revision, which is the
recorded ground-truth authority for prim paths.

### Where the gravity-real sequence stands

| step | state |
| --- | --- |
| gravity-real configuration, both backends | landed and validated in-run |
| canonical reset | passes, bitwise reproducible |
| canonical hold under real weight | **passes at 0.008284 rad, twice** |
| episode-start replay | **fixed and passing** |
| backend contact-partner probe | blocked: emitter omits resolved partner prims |
| PhysX controls receipt | not yet produced |
| Newton controls receipt | blocked on the Robotiq drive, unresolved |
| dual-backend comparison | not yet possible |

Two paid allocations, `$0.112` and `$0.19`, `$0.302` total, both torn down with
API-confirmed provider-zero and no orphaned spend.
