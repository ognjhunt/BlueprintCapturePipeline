# Unitree G1 policy selection — research and integration status

As of: 2026-09-15. Owner-authorized scope extension: expose G1 alongside Franka
on the website, research two movement and two manipulation candidates, and wire
configuration to actual execution. The owner clarified that there must be two
candidates **per task type** and requested benchmark-based selection.

This extends the ADP-050/day-21 policy/runtime integration seam under explicit
owner direction; it does not change historical ADP qualification receipts.
Completion requires a deployed website robot/hand/task selector, immutable
compatible policy bindings, real runtime dispatch, matched scenario and media
coverage, focused regression/browser tests, and deployed readback. A catalog,
external benchmark, source import, or request receipt alone cannot complete it.

## Decision so far

The initial convenience shortlist (SONIC/Unitree RL Gym for movement and
GR00T/UnifoLM for manipulation) was provisional, **not a benchmark ranking**.
Subsequent research changes the movement pair to Humanoid-GPT and GEAR-SONIC.
The manipulation pair must be scoped to the exact hand and task. No reviewed
source establishes a universal G1 manipulation top two across all candidates.

### Movement: Humanoid-GPT and GEAR-SONIC

The August HumanTracker benchmark provides a shared 29-joint G1 motion-tracking
comparison. These are reference-following controllers, not autonomous visual
navigation policies. Its published completion percentages are:

| Controller | Daily | Highly dynamic | Interaction | Ground |
| --- | ---: | ---: | ---: | ---: |
| GMT | 17.0 | 36.2 | 81.4 | 0.0 |
| TWIST2 | 60.1 | 39.9 | 91.3 | 0.0 |
| SONIC original | 93.8 | 82.1 | 97.6 | 20.1 |
| SONIC 1.1 | 94.4 | 84.7 | 97.9 | 13.4 |
| Humanoid-GPT | 94.4 | 86.9 | 97.2 | 32.9 |

Source: [HumanTracker repository, August 22 update](https://github.com/GalaxyGeneralRobotics/HumanTracker),
[paper Table 3](https://arxiv.org/html/2608.13555v1).

The SONIC 1.1 row is a repository update, not the original paper result.
Humanoid-GPT and the benchmark share authors/organization: this is not an
independent replication. Completion is not task success, and HumanScore is a
learned preference measure, not physical truth. Default version selection must
preserve the original versus 1.1 distinction; 1.1 is not better on every family.

[Humanoid-GPT](https://github.com/GalaxyGeneralRobotics/Humanoid-GPT) releases
G1 inference/deployment and ONNX weights, including `pns_wo_priv264.onnx`, the
checkpoint named by HumanTracker. Hardware revision is selected separately.
[SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) releases code, weights,
and a planner. Its original, low-latency, and 1.1 encoders/decoders/observation
configs must remain paired; benchmark results cannot be transferred between them.

### Manipulation: benchmark-led candidates are task-specific

[HumanoidArena Table 1 and S7](https://arxiv.org/html/2606.17833v1) evaluates ACT,
Diffusion Policy (DP), Flow Matching (FM), and pi0.5 with matched TWIST2 or SONIC
controllers. Under SONIC:

| Candidate | DoubleDesk SR | Pick/place box SR | HOI suite SR | Abnormal fall rate |
| --- | ---: | ---: | ---: | ---: |
| ACT | 18.3 | 56.7 | 30.56 | 8.57 |
| DP | 36.7 | 75.0 | 52.22 | 8.33 |
| FM | 38.3 | 73.3 | 41.67 | 5.71 |
| pi0.5 | 43.3 | 71.7 | 41.67 | 5.24 |

These are percentages from task-trained checkpoints, not generic base models.
DP leads aggregate success; pi0.5 leads DoubleDesk and has the lowest fall rate.
DP + pi0.5 is therefore a defensible *comparison pair* for SONIC-backed G1
manipulation, not an established universal top two. FM slightly exceeds pi0.5
on nominal box placement; outcome variance precludes treating that small gap as
settled. Neither GR00T nor UnifoLM was tested in this table.

The [released code](https://github.com/William-wAng618/HumanoidArena) uses
29-DoF G1 Dex3 tasks. The default intermediate action is 40-dimensional semantic
whole-body control; the optional latent64 route is explicitly experimental.
Do not substitute the 78-dimensional NVIDIA GR00T/SONIC interface. Checkpoint
releases are linked through ModelScope; an exact checkpoint inventory and
weight license still need verification. Source presence is not a working
Blueprint runtime. Persistent versus fresh-process reset changes must be bound.

### Other serious manipulation candidates reviewed

- **GR00T N1.7 + SONIC:** [NVIDIA's official G1 workflow](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_workflow.html)
  is concrete and well documented, but requires G1/task post-training. Its action
  is 64 motion-token values plus seven joints for each hand. A DROID checkpoint
  is not a G1 checkpoint. Strong integration option; not established as best in
  the shared G1 table above.
- **UnifoLM-VLA:** [official source](https://github.com/unitreerobotics/unifolm-vla)
  offers G1 training/deployment and links its Base checkpoint to the official
  [Dex1 dataset collection](https://huggingface.co/collections/unitreerobotics/unifolm-g1-dex1-dataset).
  The [project's 12-task G1 demonstrations](https://unigen-x.github.io/unifolm-vla.github.io/)
  and LIBERO results do not establish a head-to-head G1 win against N1.7.
- **LingBot-VLA 2.0:** [paper Table 1](https://arxiv.org/html/2607.06403v1)
  includes Unitree G1 training data (14 arm + 12 hand dimensions; no controlled
  body dimensions). Its [reported GM-100 wins](https://github.com/Robbyant/lingbot-vla-v2)
  are on Cobot Magic and Galaxea R1Pro. This does not prove G1 locomotion or the
  same G1 hand configuration as Dex3. Exact G1 deployment/checkpoint remains
  to be checked before admission.
- **FetchMan:** [paper](https://arxiv.org/abs/2608.17027) reports 73.3% real G1
  single-object reach-and-pick success with Dex1-1 grippers. The
  [repository](https://github.com/omarrayyann/FetchMan) at inspected commit
  contains only a README promising code. A promise of release is not runnable
  code. Keep it on the research list, not the enabled execution list.
- **GR00T-H N1.7:** [model card](https://huggingface.co/nvidia/GR00T-H-N1.7)
  describes surgical/healthcare post-training; the H is not evidence of a G1
  humanoid-specific checkpoint.
- **World models:** distinguish action-producing stacks from video prediction
  or reasoning auxiliaries. Model family branding cannot establish G1 support.

## The hand is part of the embodiment

No universal G1 hand standard was established by the reviewed sources.
[Unitree's simulation catalog](https://github.com/unitreerobotics/unitree_sim_isaaclab)
explicitly separates G129-Dex1, G129-Dex3, and G129-Inspire tasks. G1 itself has
multiple body configurations. [Unitree's specification](https://www.unitree.com/g1/)
describes Dex3-1 as three fingers with seven active joints per hand.

Recommended initial **benchmark-aligned** profile: **G1 EDU, 29 body joints,
Dex3-1 hands**, because it aligns with HumanoidArena and NVIDIA's 7+7 hand-joint
SONIC manipulation path. Preserve **G1 + Dex1-1** as a separate simple-gripper
profile for its own task-trained checkpoints. Neither choice is an industry
standard, and hands alone do not determine success: body controller, cameras,
object distribution, action normalization, and training data remain material.

Two Dex3 policies can still be incompatible: one may command all finger joints,
another only open/close grasp synergies. Do not silently remap one into another.
The same model family needs the exact hand-specific fine-tune and adapter.

Website contract should select:
`body revision + hand + task family -> compatible checkpoint/controller pair`.
Show movement and manipulation separately. Bind camera/state/action schemas,
control frequency, robot model digest, hand order, checkpoint digest, source
commit, and controller version. Validate these on the server and in Pipeline.
Changing hands invalidates selection and qualification; never inherit Franka
controls, scenario success metrics, or qualification into G1.

## Source revisions observed through public repository APIs

No weights were downloaded and no paid execution was started.

| Repository | Commit observed on/before September 15 |
| --- | --- |
| NVlabs/GR00T-WholeBodyControl | `087f9ac01d46f6d8e4d0b73c01ae64799f292a38` |
| NVIDIA/Isaac-GR00T | `51d4c89f72fda44cbf77285c6a8114b52676b8a1` |
| GalaxyGeneralRobotics/Humanoid-GPT | `9f9e7b74ecadb532abbb34b6a779d87191a9bbb6` |
| GalaxyGeneralRobotics/HumanTracker | `79d637da5e07b8e2c588ca613d3254f9d8acc0f7` |
| William-wAng618/HumanoidArena | `68479287a784a69be9ce6ad739311d2f11f75ef9` |
| unitreerobotics/unifolm-vla | `ff6c39aeb0454cfb95418c66aef40ca777f935c1` |
| unitreerobotics/unitree_rl_gym | `276801e46c5d433564f24658bac64f254b7d2d4b` |
| Robbyant/lingbot-vla-v2 | `bc643d74a0127fab8788da993b261d4d64101138` |
| omarrayyann/FetchMan | `736f8525fc097f35636a6ab63311cecd29808068` |

These revisions locate inspected releases; they are not automatically runtime
admission pins. In particular HumanTracker's SONIC source pin differs from
latest WBC. Reproduction must use the benchmark's recorded pin.

## Implementation audit and next work

Isolated worktrees:
- Pipeline: `/private/tmp/bcp-g1-website-20260915`
- WebApp: `/private/tmp/bw-g1-website-20260915`
- Both branch: `codex/g1-website-policies-20260915`.

The current Pipeline contains G1/SONIC controller bridges and historical G1
policy paths. The website's qualified EvaluationRunSetup is hardcoded Franka;
its canary wizard already renders a robot/policy registry, but the current scene
setup publisher emits only Franka. The autonomous scene intake explicitly
accepts only the DROID pair, and its task schema is pick-and-place only.
Changing UI enums alone would produce a false feature.

Next: verify exact Dex3 candidate checkpoint inventories; extend typed robot,
hand, task and policy configuration through existing submission paths; add G1
movement scoring and correct per-policy adapters; exercise real retained-input
execution and lossless evidence; then publish/deploy and inspect live setup.
The full user goal remains open. Research is progress, not completion.


## Review of the owner's additional September research

Verified against official sources during the same session:

- OpenPI lists both pi05_libero and pi05_droid; these are different distributions.
- NVIDIA publishes GR00T-N1.7-LIBERO and DROID checkpoints under NVIDIA Open
  Model terms, distinct from the repository's code license.
- Ai2 publishes MolmoAct2/Think LIBERO and DROID releases. LeRobot documents
  MolmoAct2 integration; the X-VLA LIBERO model card includes the quoted
  `lerobot-eval` command and an Apache-2.0 label.
- Cosmos-Policy-LIBERO-Predict2-2B explicitly uses the NVIDIA One-Way
  Noncommercial License. It is not a generally commercial default.
- HumanoidBench lists G1 three-finger variants; its original H1/Shadow results
  must not be relabeled G1 results.
- Isaac Lab's named G1 environments exist, but identifiers/import locations are
  versioned: develop documentation already refers to IsaacContrib in places.
  Pin a release instead of treating a task name as stable across versions.

Agreement: the supplied research is a good reproducible-baseline map. It does
not establish industry popularity or an overall performance ranking. Add
Humanoid-GPT to the movement comparison using the benchmark evidence above.
LIBERO-first is useful for adapter validation; it does not replace the requested
captured-site execution path. Keep fixed-base manipulation as a diagnostic;
a whole-body G1 test still requires balance, movement and contact evidence.

## Implementation checkpoint — existing run path (local, not deployed)

The owner clarified: G1 must be an option in the same existing Franka run
configurator, spawn in the scene and try the task each episode. Future
embodiments/policies/options must use the same extensible path. A separate
saved-setup/intake form does not meet this requirement and that draft was
removed from both working trees.

Current changes:
- Existing PolicyCanarySetup can switch among published setup/robot bindings.
  It fetches the selected setup and clears prior policy/confirmation state.
- Server selects the exact published setup by its digest and robot preset,
  preserving scene/team binding; stale or mismatched choices cannot fall back
  to the Franka plan. Available options are derived from published profiles.
- Native scene builder uses a robot adapter registry. Existing Franka spawning
  remains behind its adapter. G1 uses a dedicated direct-joint spawn/reset
  adapter with 29 body + 14 Dex3 joints, explicit per-joint actuator settings,
  limits, an exact local USD hash, and registered head/wrist camera roles.
- Sealed native runtime bundles include the new adapter/import dependencies.
- `configs/g1_humanoidarena_checkpoint_inventory.v1.json` records publisher
  file hashes and sizes for the actual released SONIC box-placement DP and
  pi0.5 checkpoints. Model bytes were not downloaded. Both configurations
  specify RGB 480x640, state width 64 and action width 40. The model collection
  declares Apache 2.0; inherited pi0.5/Gemma and SONIC model terms still apply.

Verification so far:
- 39 existing native runtime/import-scope tests passed.
- The required canary lifecycle rehearsal (18 tests) and provider import-closure
  suite (6 tests) passed in the same run. That run's one failure was an
  incomplete new camera test fixture; corrected G1 suite passes all 12 tests.
- Five targeted server-profile/client-switch tests passed; the existing
  internal policy contract tests also passed in the prior focused run.
- Full WebApp TypeScript check passed (PTY session 61268 exited 0), output at
  `/private/tmp/bw-g1-typecheck-current.log`.
- Graphify refresh could not run because graphifyy is absent from the selected
  interpreter; the script was attempted and its trace is retained in
  `/private/tmp/bw-g1-graphify-trace.log`.

Still required before the original goal is complete:
1. Connect G1 policy clients and the shared episode lifecycle. The worker still
   binds a DROID-specific observation/action episode loop and Franka servo.
2. Materialize G1 scene plans/published launch profiles from admitted robot,
   camera, checkpoint and controller artifacts; no G1 profile is live yet.
3. Verify G1 deterministic per-episode reset, task scoring, exact policy-input
   media, controller timing, provider bundle closure, and a real runtime.
4. Publish/deploy both repos and read back the actual website-to-episode path.

Important implementation facts for the next step:
- Current native source packet pins Arena `8b82dca224f2b5af08f339f987613c59ce9cdbaa`;
  its G1EmbodimentBase and camera classes were verified from that exact source.
- HumanoidArena's semantic_v3 state is heading-canonical orientation6 +
  joint positions29 + velocities29, in its canonical alternating joint order.
  Actions40 are root-relative xy2 + root z1 + root rotation6 + joints29 +
  binary left/right grasp2. They are not direct motor targets or GR00T latents.
- Upstream SonicActionProvider.get_action already steps physics internally.
  Wrapping it with an additional env.step would double-step the simulation.
- HumanoidArena code assumes WXYZ root state; the currently pinned native
  Isaac Beta2 path is XYZW. Any adapter must explicitly reconcile that seam.
- Pinned upstream source inspection copies are in `/private/tmp/g1-*`; use
  those as read-only references, verify source hashes before runtime use.

No provider allocation or paid execution occurred. The goal remains active.

## 2026-09-24 continuation: portable construction packet

The committed G1 spawn adapter was carried onto the current Pipeline branch.
The shared contract now accepts a digest-bound G1/Dex3 robot configuration,
requires head policy and overview review cameras, and leaves its policy
candidate list empty until a G1 episode executor is admitted. The scene
compiler derives protected rigid-body contact paths from the exact verified
robot USD and refuses contact fingertips absent from that asset. The packet
copies the robot USD from evidence-root-relative input, seals source/staged
hashes, publishes a relative path, and re-verifies bytes before container
spawn. The same path works for a local non-container bundle root.

This proves construction packet generation and preflight on a minimal test
asset, not a runnable G1 policy episode. A production robot USD with external
asset references still needs an admitted dependency closure and exact runtime
validation. The DROID-specific episode loop, SONIC execution, checkpoint
inference, scoring, published G1 setup, deployed WebApp path, and review video
remain unverified. Neither simulator nor physical G1 task success is claimed.

The pinned HumanoidArena semantic-v3 boundary now builds its 64-wide policy
state from named joints and explicitly converts native XYZW root quaternions
to the publisher's heading-canonical WXYZ rotation representation. It checks
40-wide policy references and keeps them out of the direct motor-target path.
This is a tested interface adapter, not policy inference or SONIC execution.
