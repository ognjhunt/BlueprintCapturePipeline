# Goal: ADP-009D Franka Rehearsal On The Sealed Public Scene

Use the text in this file as the complete goal prompt for the next Codex
session. It supersedes the historical `MASTER_GOAL_PROMPT.md` for new public
scene work.

## Objective

Before any fresh-site capture, build and execute one development-only Franka
pick-and-place Task Evaluation Run on the exact project-owner-approved public
scene construction:

- AuraFusion360's selected edit of InteriorGS scene `840313`;
- the exact matching SAGE-3D static collision scene;
- the inactive source canned-beverage collider and preserved cabinet support;
- the exact approved match-v2 SimReady can at its sealed pose; and
- the DROID Franka Panda plus Robotiq 2F-85 embodiment required by the policy
  checkpoints.

Freeze exactly two learned-policy candidates before observing task outcomes.
Run independent negative and scripted/replay positive controls first, then one
media-complete canary per candidate, then the preregistered matrix if and only
if all execution gates pass. Produce visual results that the project owner can
inspect directly in chat.

This is not only a one-off rollout. Materialize the task as a small,
replayable Isaac Lab evaluation harness with one immutable canonical condition
and preregistered diagnostic scenario families for placement, illumination,
camera/sensor perturbations, bounded physics variation, and exact SimReady
object cousins. Keep the canonical sealed scene as the anchor; variations may
measure robustness but may not retroactively redefine the admitted public
scene or the primary task.

This is ADP backlog item
`ADP-009D_public_scene_franka_policy_rehearsal`. It advances the day-28 public
gate. Its completion artifact is a digest-bound two-candidate simulator-only
Task Evaluation Run with complete policy-input media, deterministic
simulator-state grading, replay, teardown, and provider-zero evidence. Existing
code is insufficient because the current sealed receipt explicitly says the
native object-layer review is not a full-scene native render, and no Franka or
learned policy has run in this scene. Make the smallest reversible additions to
the existing external-scene, DROID policy, media, Isaac, and paid-resource
seams; do not build a second simulator or a general policy framework.

## Read first

Read these completely before changing code or allocating compute:

1. `AGENTS.md`
2. `docs/arm_decision_proof_v1/north_star_contract.json`
3. `docs/arm_decision_proof_v1/ADP_009_PUBLIC_SCENE_TRANSITION_DECISION.md`
4. `docs/arm_decision_proof_v1/CURRENT_STATE.md`
5. `docs/arm_decision_proof_v1/IMPLEMENTATION_BACKLOG.md`
6. `docs/arm_decision_proof_v1/PUBLIC_EVIDENCE_LADDER.md`
7. `docs/arm_decision_proof_v1/FOUNDER_SIM_ONLY_PROTOCOL.md`
8. `docs/arm_decision_proof_v1/manifests/adp009b_hybrid_replacement_seal_receipt.v1.json`
9. `docs/arm_decision_proof_v1/manifests/adp009b_aurafusion360_execution_receipt.v1.json`
10. `docs/arm_decision_proof_v1/manifests/adp009b_aura_human_review_receipt.v1.json`
11. `docs/arm_decision_proof_v1/manifests/adp009b_simready_replacement_match_v2_receipt.v1.json`
12. `docs/arm_decision_proof_v1/manifests/adp009a_materialized_suite/exact_simready_object.component_receipt.json`
13. `src/blueprint_pipeline/splat_scene_analysis.py`
14. `src/blueprint_pipeline/perception_views.py`
15. `src/blueprint_pipeline/external_scene_robot_placement.py`
16. `src/blueprint_pipeline/external_scene_task_evaluation.py`
17. `src/blueprint_pipeline/droid_policy_bridge.py`
18. `src/blueprint_pipeline/openpi_droid_policy_runtime.py`
19. `src/blueprint_pipeline/groot_n16_arena_policy_runtime.py`
20. `src/blueprint_pipeline/groot_n17_droid_policy_runtime.py`
21. `src/blueprint_pipeline/franka_droid_closed_loop.py`
22. `src/blueprint_pipeline/franka_droid_control_preflight.py`
23. `src/blueprint_pipeline/adp_isaac_lab_arena_materialization.py`
24. `src/blueprint_pipeline/adp_isaac_lab_arena_vast.py`
25. `src/blueprint_pipeline/scenario_variation_instantiator.py`
26. `src/blueprint_pipeline/evaluation_run_contract.py`
27. `src/blueprint_pipeline/evaluation_run_execution.py`
28. `src/blueprint_pipeline/isaac_review_media.py`
29. `src/blueprint_pipeline/episode_visual_evidence.py`
30. `src/blueprint_pipeline/paid_resource_allocator.py`

### Mandatory fresh Isaac Lab and Omniverse agent-skill audit

Before designing or changing the Isaac Lab environment, use web search and the
official repositories to recheck the latest supported workflows. Agent-skill
instructions are versioned implementation inputs, not timeless advice. Record
the repository, exact commit/tree, skill path, skill status/version, license,
retrieval timestamp, and every directly used reference or example.

The 2026-08-06 starting observations are:

- official Isaac Lab `develop` commit
  `3ea6f7bbf6c7d515aa1f8e8c54bfdfffda2d4857` contains repository-owned
  skills under `skills/` with native `.agents/skills` aliases;
- official NVIDIA verified-skills catalog commit
  `276b9bcce5d1b224769f9b10ad26975c15e0dd4c` contains the current
  Omniverse/Physical-AI workflow skills;
- official `NVIDIA-Omniverse/ovrtx` commit
  `4b9a5fe6f8becf6c5ff031e167cd4201054a96ce` contains renderer skills for
  its current API; and
- official `NVIDIA-Omniverse/PhysX` commit
  `7845321d31fa3619917ebe127ab5e08e73de0bdb` was observed with ovPhysX
  source/tests but no materialized `ovphysx/skills/` directory. Recheck this
  rather than inventing or trusting a catalog link to files that are absent at
  the selected source revision.

At minimum, read each applicable `SKILL.md` completely before acting and follow
its directly linked maintained references/examples:

Isaac Lab:

- `isaaclab-building-environments`;
- `isaaclab-planning-manipulation-tasks`;
- `isaaclab-randomizing-with-events`;
- `isaaclab-using-sensors-actuators`;
- `isaaclab-selecting-backends`; and
- `isaaclab-using-presets` only if a real renderer/backend/domain selector is
  needed. Do not add presets to a one-backend task merely for abstraction.

Omniverse/OVRTX, when that renderer path is actually selected:

- `loading-usd`;
- `renderer-creation`;
- `render-settings`;
- `camera-outputs-rt2`;
- `reading-render-output`;
- `stepping-and-rendering`;
- `warmup`; and
- `semantic-labels`.

NVIDIA verified Physical-AI skills, only when their stage is in scope:

- `omniverse-cad-to-simready` for newly generated cousin assets, not to
  re-author the sealed approved match-v2 can;
- `omniverse-usd-performance-tuning` only after measured stage/render
  bottlenecks and a baseline profile exist; and
- `omniverse-realtime-viewer` only if a viewer/streaming deliverable is needed,
  not as a substitute for lossless policy-input capture.

If installing a mirrored NVIDIA skill, independently verify its detached OMS
signature, skill card, evaluation dataset, benchmark report, and declared
license against the catalog trust anchor before execution. Prefer the official
product repository as the behavioral source of truth when the catalog says it
is a mirror. Do not copy an old global skill over a newer repository-owned
skill or trust an unsigned third-party skill merely because it has stars.

Apply the currently observed guidance explicitly:

- default new tasks to manager-based environments and start from the closest
  maintained source example;
- validate in order: import, small environment instantiate, reset, random or
  scripted step, shapes/devices/timing, then any larger rollout;
- express domain variation through typed EventManager terms and choose
  `prestartup`, `startup`, `reset`, or `interval` based on when the property can
  safely change;
- validate scene physics and reset reachability before interpreting policy or
  reward behavior;
- start camera-based evaluation at a small environment count and measure
  renderer memory before vectorizing;
- keep PhysX-specific behavior in public schema configs and task-local terms,
  not scattered runtime conditionals;
- if OVRTX and ovPhysX share a process, initialize/import OVRTX first;
- in OVRTX attached mode, honor ovstage ordinals and write-floor publication
  before stepping the renderer;
- request only required AOVs. For metric compositing prefer
  `DistanceToCameraSD` or `DistanceToImagePlaneSD` in metres and record their
  geometry; do not confuse unitless `DepthSD` with metric depth;
- write RTX settings on the RenderProduct, then reset/re-warm when settings
  invalidate accumulation;
- for real-time path tracing, treat renderer warmup and texture streaming as a
  measured gate (the current skill recommends 40 warmup frames as a starting
  default); for reference PathTracing, bind the samples-per-pixel and capture
  behavior instead of assuming repeated warmup; and
- author semantic/instance labels in a composed override layer so the sealed
  source assets remain unchanged.

Emit `adp009d_agent_skill_audit.v1` binding the exact skill sources and the
specific instructions that affected architecture, commands, or validation.
If a skill has materially changed since these observations, use the newer
official version, retain a diff/decision receipt, and update the harness plan
before mutation.

Begin from protected `main`. Verify that `main` contains decision ID
`ADP-009-public-scene-transition-2026-08-06`. If it does not, stop with the
smallest blocker `adp009_public_scene_transition_not_on_main`. Inspect all
worktrees, dirty state, active writers, retained outputs, provider state, and
existing checkpoint/model caches. Preserve unrelated work and use a clean
isolated worktree when necessary.

## Immutable starting evidence

Do not rerun AuraFusion360 or alter the approved can. Verify these bytes from
the actual local files before using them; copied receipt values are not enough.
The retained rights-safe data root is:

`/Users/nijelhunt_1/workspace/BlueprintValidation/data/adp009a_tranche1_20260804`

Expected identities at handoff:

- Aura execution: `live_v12_openclip_execute`, exact released source commit
  `f23b26c44ba84608306ba952510533ebf4c7877d`, unmodified.
- Aura result PLY:
  `aura_interiorgs/live_v12_openclip_execute/immutable_execution/artifacts/aurafusion360_840313_ins160_final.ply`;
  `106309429` bytes; `415265` vertices; SHA-256
  `cbb05fc8e6da6ecdb72464f3b115f63e8747e2b67e97c309b4e40952b33000bd`.
- Original InteriorGS PLY:
  `shortlist/InteriorGS/0442_840313/3dgs_compressed.ply`; `33137360` bytes;
  SHA-256
  `57c71edcb450f2323a5b8ad290b5672b437fc73b9283a7485804ce607da12254`.
- SAGE collision:
  `shortlist/SAGE-3D_Collision_Mesh/Collision_Mesh/840313/840313_collision.usd`;
  `42418216` bytes; SHA-256
  `b265706c24f6a8ace3ee6743fd138583c4e21d83f61b99a06fd435e6ac2d6b41`.
- SAGE USDZ:
  `shortlist/SAGE-3D_InteriorGS_usdz/InteriorGS_usdz/840313.usdz`;
  `81806718` bytes; SHA-256
  `bcdc8d36ed88c1a5c4e7cd333479e24c67cde64ed3b3ea135f37028f70d1ebb8`.
- Exact match-v2 replacement:
  `docs/arm_decision_proof_v1/assets/adp009a_840313_canned_beverage_match_v2.usda`;
  `20705128` bytes; SHA-256
  `61c2a03bef425803d82cc5ef24ced5b2ccb4160923c53bb10c6ad0e3f52532ec`.
- Publisher scene ID `840313`; target instance `160` / `ins160`;
  semantic label `canned_beverage`; source collider
  `/Root/ZHQYGJJVAJYEYPTUKY888888` must remain inactive.
- Replacement base pose in the shared metre frame:
  `[3.4681748, -3.3100837, 0.5264650138348479]`.
- Cabinet support height: `0.5264650138348479` m. The support collider must
  remain active.
- Historical hybrid seal digest:
  `sha256:b259532be614098a3830aa9945770a96371968f9c68e8087eb21a2ca00e3c3e3`.

The historical ten-role index remains `7/10`. Do not modify it, its receipts,
or the old hybrid seal. AuraFusion360 is selected by the newer scope decision;
Inpaint360GS and InFusion are nonblocking abstentions; fresh Raw V3.2 capture,
not ScanNet++, supplies the later real metric transfer.

## Claim boundary

This run may establish only that the exact sealed public construction can host
a native simulator task and exercise the two-candidate decision harness. It is
`development_only`, `simulator_only`, and public-dataset-specific.

It does not establish hidden-background truth, measurement-authoritative SAGE
surfaces, fresh-site metrology, observation-domain match, policy quality on a
real robot, sim-to-real fidelity, safety, deployment readiness, or physical
success. InteriorGS/Aura supplies appearance; SAGE supplies a publisher-aligned
static collision proxy; the SimReady USD supplies task-object visual/collision
and physics; none may silently substitute for another.

## Required sequence

### 1. Audit and reproduce the seal

- Recompute all sizes and SHA-256 digests above from actual bytes.
- Recheck rights and access records without printing secrets. Use
  `blueprint_pipeline.model_access_env.normalize_model_access_env()` for
  Hugging Face access.
- Inspect the exact USD layer graph, units, up axis, transforms, active source
  and support colliders, unresolved references/payloads, mass, inertia,
  contact material, and replacement pose.
- Replay the existing static validation and native drop/contact, slide, tip,
  and gripper tests before adding the robot. A changed byte or failed replay is
  a blocker, not permission to regenerate the scene.
- Reuse the splat analyzer and perception-view contracts. Do not return to a
  target-centred orbit or ignore unexplored scene regions.

### 2. Prove a real hybrid policy observation path

This is the first hard gate. The old visual review composited an object layer
over Aura frames and explicitly was not a full native scene render. A learned
policy result is invalid unless both DROID policy cameras receive coherent,
time-synchronized views of the Aura background, moving Franka, moving can, and
occlusions.

Prefer the smallest tested path:

1. render the Aura 2DGS/3DGS appearance from the exact live Isaac camera pose;
2. render the Franka and dynamic task object from Isaac with RGB, metric depth,
   instance segmentation, and alpha;
3. depth-compose the layers in the shared registered frame; and
4. verify static and moving occlusion, camera intrinsics/extrinsics, timestamps,
   colorspace, resolution, and frame hashes.

If an existing native Gaussian/Omniverse renderer can render all required
layers directly and is already admitted, use it. Do not claim that a plain
2D overlay, a frozen Aura frame, or a USD-only render is the approved hybrid
scene. If no depth-correct path exists, implement only the smallest camera
adapter and focused tests. If that still cannot be proven, stop before policy
execution with `sealed_aura_hybrid_policy_observation_renderer_missing`.

Retain a visual preflight montage and short camera-motion video showing the
same frozen camera rendered as: Aura only, Isaac RGB/depth/segmentation, and
final depth-correct composite. Surface these images in chat for project-owner
review.

### 3. Freeze the basic task before policy outcomes

Use the existing SimReady can as the manipulated object. Define a simple
pick-lift-translate-place task on the same cabinet support:

- start at the exact sealed can pose;
- grasp with the DROID Robotiq 2F-85 parallel-jaw gripper;
- lift the can at least `0.08` m above its settled start height;
- translate it to one collision-free, preregistered empty destination patch on
  the same support, at least `0.15` m from the start; and
- release it so it settles upright, with its centre within `0.05` m of the
  destination and tilt no greater than `15` degrees.

Derive the destination from SAGE support triangles plus splat/semantic
occupancy before any learned-policy outcome. Store the selection inputs,
candidate patches, rejection reasons, selected coordinates, clearance, and
digest. Do not choose a destination because one model succeeds there. Do not
add a tray unless a separate exact admitted SimReady tray is genuinely needed.

Register the DROID Franka base with
`external_scene_robot_placement.py`, then prove in native Isaac:

- the base is supported and collision-free;
- the can and destination are in the reachable workspace;
- both external and wrist cameras see the task;
- the Robotiq 2F-85 articulation, joint ordering, limits, reset state, and
  action scaling match DROID; and
- the robot does not start intersecting the SAGE scene.

Do not silently substitute the stock Panda gripper. If the exact DROID Franka
plus Robotiq embodiment cannot be materialized in the sealed scene, retain
`sealed_scene_droid_franka_embodiment_materialization_required`.

Freeze one language instruction, termination horizon, control rate, action
horizon, cameras, reset state, tolerances, collision rules, and success grader.
The primary success metric is deterministic simulator state, never a policy,
VLM, Cosmos, or human aesthetic judgment.

### 3A. Build the bounded Isaac Lab evaluation harness

Use Isaac Lab as the simulator/task substrate and evaluate Isaac Lab Arena as
the smallest upstream evaluation layer before writing new infrastructure. The
official `release/0.2.1` branch was observed on 2026-08-06 at commit
`8b4a3a47fc53de23e8205089d71109a2e2348acd`; independently verify the exact
official commit, tree, license, Isaac Lab/Isaac Sim compatibility, and nested
dependencies before pinning it. Arena is alpha software, so pin exact bytes
and do not depend on a floating `main` branch.

Prefer Arena's native composition and evaluation contracts when they fit:

- one Scene for the sealed Aura/SAGE construction;
- one Embodiment for the exact DROID Franka plus Robotiq 2F-85;
- one Task for the frozen pick-lift-translate-place objective;
- `PolicyBase`-compatible thin adapters for the two frozen candidates and the
  independent controls;
- task-owned metrics and deterministic termination;
- a sequential batch-evaluation jobs file for scenario/policy cells; and
- Isaac Lab observation, action, event, termination, and recorder managers.

Use a manager-based Isaac Lab environment unless a measured incompatibility
requires a direct environment. The manager-based path is preferred here
because variations must remain typed and independently testable:

- `ObservationManager` owns the exact policy and grader observations;
- `ActionManager` owns the DROID-to-Franka command contract;
- `EventManager` applies preregistered reset-time variations;
- `TerminationManager` owns success, invalidity, and timeouts; and
- `RecorderManager` exports episode data and media hooks.

If Arena cannot host the hybrid Gaussian observation path or exact robot
embodiment, retain the incompatibility receipt and implement the smallest
Blueprint adapter around `ManagerBasedRLEnv`. Keep the same external scenario,
episode, metric, and replay contracts; do not fork a second general benchmark
framework.

The harness must materialize evidence-derived, canonical artifacts such as:

- `adp009d_agent_skill_audit.v1`;
- `adp009d_franka_eval_harness_manifest.v1`;
- `adp009d_scenario_suite.v1`;
- `adp009d_scenario_instance.v1` per fully resolved condition/seed;
- `adp009d_episode_receipt.v1` per attempted episode; and
- `adp009d_eval_summary.v1` with stratified paired results.

Names may reuse an existing broader Blueprint schema when that schema already
proves the same fields. Do not add parallel schemas merely for naming. Every
scenario instance must derive its resolved values and digest from the frozen
suite, source assets, seed, and simulator configuration; caller-asserted IDs or
hand-authored success JSON are insufficient.

### 3B. Preregister the scenario and cousin suite

Do not run a Cartesian-product stress campaign. Freeze a compact,
decision-relevant design before any learned-policy task outcome. It must
contain these tiers:

1. **Canonical anchor** — the exact approved Aura/SAGE scene, exact match-v2
   can, sealed start pose, nominal lighting/cameras/physics, and selected
   destination. This is the primary public-scene condition and must never be
   replaced by an averaged randomized condition.
2. **Placement and approach** — bounded start and destination offsets expressed
   in the measured support-plane basis, plus bounded can yaw and robot reset
   perturbations. Derive ranges from collision clearance, camera visibility,
   and reachable-workspace measurements. Include nominal, interior, and
   near-boundary values without moving the can off the admitted support or
   creating an impossible start state.
3. **Illumination and appearance** — preregistered low/nominal/high intensity,
   bounded direction and color-temperature changes, and exposure settings.
   Lighting changes affect rendered observations only; never alter task truth
   or physics. Retain the exact resolved light rigs and rendered preflights.
4. **Camera and sensor robustness** — bounded wrist/external camera extrinsic
   and intrinsic perturbations, exposure/noise, and optional bounded latency.
   Include exact nominal calibration. Perturbations must be explicitly applied
   to both rendering and policy metadata where appropriate; never lie to a
   policy about calibration unless the scenario is specifically labeled as a
   calibration-error test.
5. **Physics robustness** — bounded, physically plausible can mass, friction,
   restitution, centre-of-mass, and support-contact variations around the
   native validated values. Never randomize unmeasured values without recording
   the source or rationale and the allowed claim ceiling.
6. **Object cousins** — the approved match-v2 can plus at least two separately
   packaged and validated SimReady can-family cousins when their exact assets,
   rights, and physics can be admitted before protocol freeze. Prefer one
   visual/material cousin that preserves geometry and collision, and one
   bounded geometric cousin that preserves the cylindrical parallel-jaw grasp
   affordance while varying height/diameter within preregistered limits. Each
   cousin needs its own USD package digest, visual mesh, collider, units, mass,
   inertia, material/texture provenance, validation result, and grasp-clearance
   proof. A color swap is a visual cousin, not geometric generalization. A
   generated or approximate cousin is simulator stimulus, never metric truth.
7. **Composed held-out cases** — a small preregistered set combining two or
   more individually admitted variations to test interaction effects. Freeze
   these before learned outcomes; do not search for attractive successes or
   adversarial failures after the fact.

Optional distractors, partial occlusions, or additional clutter may be added
only from exact rights-admitted SimReady assets with collision-safe placements.
Their absence must not block the required placement/lighting/camera/physics/
cousin suite. Do not delete or move unrelated SAGE geometry to manufacture a
scenario.

For every variation parameter record:

- parameter ID, unit, nominal value, allowed set/range, sampling rule and seed;
- source of the bound: publisher metadata, Blueprint measurement, approved
  engineering tolerance, or explicitly synthetic diagnostic choice;
- assets and USD prims affected;
- whether it changes appearance, observation metadata, dynamics, task geometry,
  or only reset state;
- validity constraints and rejection behavior; and
- whether it belongs to the scored qualification set or an unscored diagnostic
  set.

Use paired scenario instances: both learned policies receive identical resolved
assets, initial states, variations, seeds, horizons, and grading. Freeze a
development/qualification split before outcomes. Development cases may debug
adapters and controls; qualification cases remain untouched until the two
candidate canaries are valid. A failed or invalid qualification episode stays
in the denominator according to the frozen invalidity rules.

Choose episode counts using a recorded power/precision and cost analysis, not
an arbitrary giant sweep. At minimum the design must separately estimate the
canonical success rate, per-family success degradation, and the paired policy
difference. Use one-factor conditions plus the small held-out composition set;
do not average all conditions into one number that hides a concentrated failure
mode.

### 3C. Metrics and eval semantics

The independent simulator-state grader must emit at least:

- binary task success and valid/invalid/timeout state;
- grasp acquired and retained;
- maximum lift height and time to lift;
- transported distance and time to destination;
- final centre error, final tilt, settle duration, and release state;
- robot/environment and object/environment collision counts and impulses;
- joint-limit, action-clipping, controller-stall, drop, and support-edge events;
- completion time, path length, action smoothness, and policy inference latency;
- exact scenario family, parameter values, cousin identity, and paired seed; and
- media completeness and deterministic replay result.

Report canonical and stratified results for every policy and control. Include
paired confidence intervals, scenario-family deltas from canonical, worst-case
and low-quantile performance, invalid/media-incomplete rate, and explicit
failure-mode counts. Sensitivity analysis may identify which frozen factors are
associated with success, but it is secondary analysis and may not rewrite the
primary metric or scenario weights after outcomes.

The result may support a simulator-only select, eliminate, or abstain decision
between the two frozen candidates. It may not establish that the winner will
rank the same on the fresh site or physical robot. Object-cousin performance
establishes only robustness over the exact admitted cousin set.

### 4. Run independent controls

Before any learned model:

- a zero/stationary control must fail;
- a scripted or replay control must complete the exact task;
- camera, action, reset, contact, and media parity must pass; and
- repeated resets must reproduce the preregistered start/destination within
  tolerance.

Run both controls first on the canonical anchor, then across every resolved
scenario/cousin cell. The zero-action control must not acquire false successes;
the scripted/replay control must establish that each scored scenario is
physically solvable and that its grader remains valid. A scenario where the
positive control cannot complete is blocked or diagnostic-only; it may not be
quietly counted as evidence that a learned policy is weak.

Retain the exact lossless observation frames, terminal frame, state/contact
trace, action trace, independent grader result, and review video for both
controls. The existing MuJoCo preflight is useful only as a runner/media oracle;
it is not this native-Isaac result.

### 5. Audit top policy candidates without seeing task outcomes

Independently reverify official repositories, exact source commits, nested
dependencies, checkpoint revisions/objects, licenses, access, environment
locks, and author smoke commands. Use existing exact caches when trustworthy.
Never accept terms on the user's behalf or print tokens.

Candidate pool for compatibility preflight:

- Physical Intelligence OpenPI `pi05_droid`, using the exact byte-inventoried
  `pi05_droid_jointpos_polaris` checkpoint already bound by
  `openpi_droid_policy_runtime.py`; OpenPI source revision
  `15a9616a00943ada6c20a0f158e3adb39df2ccac`. The frozen inventory contains
  `27` objects totalling `12434530837` bytes and has inventory SHA-256
  `492ef95fa2e0ea8c026fda4bf6a2662758e7958ab5223ecb270cde5bc3797063`;
  verify the materialized worker copy against every object row.
- NVIDIA `nvidia/GR00T-N1.7-DROID`, embodiment
  `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`, currently pinned by Blueprint to
  source revision `b9955401d50c92a29258732e3ad6ccd579f1bdc0` and checkpoint
  revision `05e7cc97e40dbd33b0890c35cc0214fcb0547ab5`.
- NVIDIA `nvidia/GR00T-N1.6-DROID`, embodiment `OXE_DROID`, pinned by the
  existing Arena-native seam to source revision
  `e29d8fc50b0e4745120ae3fb72447986fe638aa6` and checkpoint revision
  `ae3ebe8d288971ac53aa30c756ea5cba0f52611b`.
- NVIDIA `nvidia/Cosmos3-Nano-Policy-DROID` as a compatibility candidate only:
  the fresh 2026-08-06 publisher snapshot is public and ungated at revision
  `6706d7680581c255ff61e0f3bb49d90eac55c79e`, contains `43` files totalling
  `32937432846` materialized bytes, and declares OpenMDW 1.1. It is a 16B DROID
  action-policy release, but its 10D Franka/Robotiq state/action contract and
  much larger runtime still require an exact Blueprint adapter and smoke. A
  Cosmos world generator or reasoner is not automatically this robot policy.

Fresh 2026-08-06 Hugging Face probes observed N1.6 at checkpoint revision
`ae3ebe8d288971ac53aa30c756ea5cba0f52611b` (`13` files,
`6573569204` bytes) and N1.7 at
`05e7cc97e40dbd33b0890c35cc0214fcb0547ab5` (`17` files,
`6914267987` bytes), both public and ungated. N1.7's publisher card declares
the NVIDIA Open Model License Agreement; N1.6's card exposed no machine-readable
license declaration. Reverify the authoritative terms and bind them rather
than inferring checkpoint rights from the Apache-2.0 source repository.

Run only author-data inference or interface smokes during this stage. Do not
run any candidate on the sealed task and do not inspect task outcomes.

Freeze exactly two candidates from this pool using only preregistered
non-outcome criteria: exact DROID embodiment compatibility, complete rights and
checkpoint identity, observation/action parity, executable author smoke,
hardware feasibility, and an existing thin Blueprint adapter. Prefer
`pi05_droid_jointpos_polaris` as baseline and the newest GR00T DROID revision
that passes every gate as the alternative. If N1.7 fails parity, retain the
failure and use the already Arena-aligned N1.6; do not patch failed outputs into
success. Cosmos may replace one candidate only if it passes the same action
policy gates before protocol freeze. Never put π0.5, two GR00T versions, and
Cosmos into a multi-policy bakeoff.

The shared policy interface must retain at least:

- lossless external RGB and wrist RGB actually shown to the model;
- seven Franka joint positions and one gripper position;
- DROID end-effector state when required;
- the frozen language instruction;
- raw model action chunks and exact executed joint/gripper targets;
- 15 Hz policy cadence, frozen open-loop horizon, clipping, and joint limits;
  and
- worker-side checkpoint and environment identity receipts.

### 6. Freeze and approve the two-candidate protocol

Create a new digest; do not reuse the maple-table founder protocol digest. The
new protocol must bind the exact public-scene bytes, robot asset, placement,
task, destination, candidates, checkpoints, observation renderer, cameras,
instruction, action translation, reset seeds, horizon, metric, media, timeout,
and claim ceiling.

Recompute the trial-count power analysis for the selected task and minimum
decision-relevant effect. Use paired deterministic reset seeds where valid.
Failed, invalid, timed-out, interrupted, or media-incomplete trials remain
failures in the denominator. Obtain project-owner approval quoting the exact
new protocol digest before candidate execution. Existing authorization for GPU
usage does not approve an unknown or mutable protocol digest.

### 7. Execute canaries and the matrix on authorized compute

Use only the canonical paid-resource path:

`python -m blueprint_pipeline.paid_resource_allocator gpu-canary ...`

Inspect the current CLI help and extend the existing ADP Arena/native-control
probe minimally if the sealed-scene probe kind is missing. Do not call a Vast,
RunPod, AWS, or provider adapter directly.

Compute preference:

- first test whether one 48 GB NVIDIA worker can co-reside Isaac Sim, the
  hybrid renderer, and one policy server without memory pressure;
- otherwise use one L40S-class 48 GB worker for Isaac/rendering and a separate
  RTX 4090-class 24 GB worker for the policy server; and
- select from current measured price/capacity through the canonical allocator,
  not by hard-coding AWS or Vast outside that seam.

Every allocation requires an immutable input bundle, quoted hourly price,
hard per-run and campaign spend caps, hard TTL, independent watchdog, zero
retries, output-return proof, teardown, and API-confirmed provider-zero. The
user has authorized necessary GPU usage for this goal, but silence never
authorizes mutation of the frozen protocol or an unbounded retry.

Execute in this order:

1. native zero/stationary negative control on the canonical condition;
2. native scripted/replay positive control on the canonical condition;
3. both controls across the frozen scored scenario/cousin suite;
4. one media-complete canonical canary for candidate A;
5. one media-complete canonical canary for candidate B;
6. inspect only execution validity, not comparative success, at the canary
   gate; and
7. if both are valid, execute the frozen, powered paired scenario matrix
   exactly once using identical resolved cells and seeds.

If a canary fails, retain it and emit the smallest technical blocker. Do not
switch models based on task performance after outcomes exist.

### 8. Cosmos auxiliary evaluation

After raw simulator outcomes and task-state grades are sealed, an exact
rights-admitted Cosmos reasoner, forward-dynamics model, or action-policy
analysis may be run as a separately labeled auxiliary measurement. Bind its
source/model revision, input episode digests, prompt/config, outputs, runtime,
and cost.

Cosmos may describe visible failures, predict a trajectory, or provide a
secondary plausibility comparison. It may not overwrite simulator state,
grade its own policy, determine primary success, or turn a simulator result
into physical evidence. If no exact compatible Cosmos path is admitted, retain
`cosmos_auxiliary_model_contract_missing`; this does not invalidate the two
learned-policy matrix.

### 9. Required visual and machine-readable results

For every completed episode retain:

- every exact lossless external and wrist image shown to the policy;
- a digest-bound frame manifest and derived review video;
- instruction, proprioception, raw and executed actions;
- object/robot state, contacts, resets, termination, and deterministic grader;
- policy worker identity, checkpoint byte manifest, environment lock;
- stdout, stderr, exit status, duration, GPU identity and peak memory;
- immutable inputs and outputs; and
- spend, watchdog, teardown, and provider-zero receipts.

At handoff, display directly in chat:

1. the sealed scene before robot insertion;
2. Franka at the frozen reset with the can at the exact start;
3. the preregistered destination and scripted positive-control terminal frame;
4. for each policy, external and wrist start, grasp/lift, and terminal frames;
5. per-policy review videos or compact montages; and
6. a scenario contact sheet showing placement, lighting, camera, physics, and
   object-cousin conditions; and
7. a clear table of canonical and per-family attempts, valid episodes,
   successes, timeouts, collisions, object lift/place errors, degradation from
   canonical, and the bounded select/eliminate/abstain result.

Do not call a source clone, checkpoint download, import, author-data smoke,
scripted control, camera composite, prepared job, or failed rollout a learned
policy result.

## Required implementation and tests

Reuse existing production seams. Add only what is missing for:

- exact sealed-scene materialization with Franka;
- depth-correct, pose-synchronized hybrid policy observations;
- deterministic empty-destination selection and task freeze;
- a pinned Isaac Lab/Arena evaluation environment with typed scenario
  materialization, batch execution, metrics, and recording;
- a digest-bound audit proving the applicable official Isaac Lab, OVRTX, and
  NVIDIA Physical-AI agent skills were freshly inspected and followed;
- evidence-derived placement, lighting, camera/sensor, physics, and cousin
  variations plus a small held-out composition set;
- DROID embodiment/action parity in the sealed scene;
- new protocol/approval/schedule receipts;
- candidate execution and media collection; and
- exact evidence compilation and replay.

Focused tests must cover:

- exact scene and model digest derivation from files, not caller assertions;
- rejection of altered Aura, SAGE, can, robot, checkpoint, or camera bytes;
- rejection of active source collider or missing support collider;
- rejection of unresolved USD payloads, unknown units/axes, or missing inverse
  transforms;
- rejection of 2D-only or stale-camera policy observations;
- rejection of stock-gripper substitution and DROID state/action mismatch;
- destination selection before outcomes and collision/clearance mutation tests;
- deterministic scenario resolution from suite digest plus seed;
- rejection of floating, stale, unsigned, license-incompatible, or absent
  agent-skill inputs and unsupported API assumptions;
- rejection of out-of-bounds placement, invalid lighting/camera metadata,
  nonphysical dynamics, unadmitted cousin assets, and post-outcome variations;
- canonical-condition preservation and identical paired cells for both policies;
- positive-control solvability and negative-control false-success rejection for
  every scored scenario;
- per-family metrics, denominator integrity, and failure-mode stratification;
- exactly two candidates and no post-outcome candidate switching;
- independent deterministic success grading;
- lossless policy-input media and review video for every completed episode;
- no caller-asserted success/admission;
- paid-resource price, cap, TTL, watchdog, zero retry, teardown, and
  provider-zero behavior; and
- deterministic replay of the final decision or abstention.

Run claim-linked focused tests, changed-file Ruff, `git diff --check`, then the
impacted-test and sentinel selector. Do not run the full suite unless dependency
analysis proves the change cross-cutting.

## Definition of done

The goal is complete only when the run leaves:

- the exact historical public-scene evidence unchanged;
- a reproduced sealed Aura/SAGE/match-v2 scene identity;
- one native DROID Franka placement and frozen basic task;
- one pinned, replayable Isaac Lab evaluation harness whose canonical condition
  is the exact sealed scene;
- one retained fresh official agent-skill audit binding the exact Isaac Lab,
  Omniverse renderer, and optional cousin-authoring workflows used;
- one preregistered, digest-bound scenario suite covering placement,
  illumination, camera/sensor, bounded physics, at least two admitted object
  cousins, and a small held-out composition set;
- a proven depth-correct hybrid policy observation path;
- passing native negative and scripted/replay positive controls on the
  canonical anchor and every scored scenario/cousin cell;
- exactly two frozen, source/checkpoint/rights/adapter-admitted learned-policy
  candidates and a newly approved protocol digest;
- one valid media-complete canary for each candidate;
- the frozen preregistered paired scenario matrix executed once, or a precise
  retained blocker reached after the canaries;
- an independent simulator-state decision or honest abstention;
- canonical, per-scenario-family, cousin, sensitivity, worst-case, and paired
  policy metrics without post-outcome weighting changes;
- visible before/after and per-policy evidence surfaced in chat;
- a deterministic replay command and concise blocker report;
- protected-main PR publication with required checks and remote/local main tree
  parity; and
- no continuing paid resources, uploads, or secret exposure.

Do not begin the fresh-site capture in this goal. The proposed next work after
this rehearsal is partner/task protocol freeze followed by one rights-cleared
Raw V3.2 object-present plus clean-background capture.
