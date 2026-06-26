# Unitree G1 Policy Endpoint Lane

## Purpose

The preferred G1 robot-policy lane is Unitree-native, not generic OpenVLA by
default. Blueprint should use a G1-specific locomotion, hand, gripper, or
world-model-action runtime when the claim is about Unitree G1 behavior.

The WAM/evaluator lane is separate. OSCAR, Cosmos, or Unitree WMA can generate
or score future-world support artifacts, but that is not the same thing as a
robot policy controlling G1 hands.

When no live OSCAR/Cosmos command is configured, the MuJoCo Unitree policy/WAM
loop uses Blueprint's default local OSCAR-style support generator. That default
backend renders action-conditioned next-observation frames and short MP4
segments, carries policy action/proprioception summaries and projected G1
skeleton support into the WAM trace, and re-queries the selected Unitree policy
on those generated frames. It sets `default_local_wam_generator_used=true` and
`learned_oscar_or_cosmos_model_ran=false`; live learned WAM model execution still
requires the explicit gated OSCAR/Cosmos command path.

After each generated next-observation, the loop runs the WAM-derived
perception/observation harness before policy requery. The harness derives
support masks/boxes, tracks, relative depth, pose, contact likelihood,
reviewability, and uncertainty from generated media, combines them with
evaluator-controlled action history, nominal robot state, projected skeletons,
controller limits, and camera calibration when available, then adapts the next
policy input to the selected policy's declared observation schema. Current
Unitree policy commands are treated as RGB plus nominal-state consumers by
default, so harness masks/depth/contact fields are withheld from the policy and
remain diagnostics unless a future policy explicitly declares support.

## Current Policy Roles

- `unitree_g1_policy`: realistic G1 locomotion/control candidate, such as
  Unitree RL Gym or Unitree controller stacks.
- `unitree_lerobot_policy`: G1 Dex1/Dex3/gripper manipulation policy candidate
  through Unitree LeRobot eval/training flows.
- `unitree_unifolm_vla_policy`: Unitree-native VLA policy candidate, with
  public UnifoLM-VLA checkpoints, still requiring a Blueprint action decoder
  and Unitree controller bridge.
- `unitree_unifolm_wma_policy`: Unitree-native world-model-action candidate,
  useful for policy/world-model interaction, still requiring a server/client
  wrapper and Blueprint action mapping.
- `unitree_groot_n17_sonic_policy`: NVIDIA Isaac-GR00T N1.7 policy candidate
  for the `UNITREE_G1_SONIC` embodiment plus GR00T-WholeBodyControl / SONIC.
  This is the current top investigation path for the missing G1
  manipulation/action-command stack. Source/checkpoint runtime configuration is
  separate from proof: the policy is not proven until a PolicyServer/action
  wrapper returns real N1.7/SONIC action chunks and the simulator-only Sim2Sim
  path participates in the bounded loop.
- `openvla_policy`: generic VLA candidate. It can remain available for
  comparison or non-Unitree policies, but it is not the preferred proof path for
  G1 dexterous-hand behavior.
- `oscar_wam` / `cosmos_wam`: evaluator/world-model rollout candidates, not
  the G1 robot policy by themselves.

## Current Implementation Status

Implemented:

- Unitree RL Gym locomotion/controller discovery and MuJoCo proof plumbing.
- Unitree LeRobot and Unitree UnifoLM command-adapter boundaries that read
  Blueprint observation packets and emit Blueprint-compatible actions.
- Provider-smoke/import artifacts for Unitree UnifoLM, with OpenVLA kept as a
  comparison path rather than the selected G1 policy.
- GR00T N1.7 + UNITREE_G1_SONIC runtime audit, provider-smoke bundle, and
  action-command adapter contracts. These write
  `unitree_groot_n17_sonic_installation_audit.json`,
  `unitree_groot_n17_sonic_policy_runtime_truth_boundary.json`, and
  `unitree_groot_n17_sonic_policy_runtime_summary.json`, but runtime
  configuration artifacts are not model execution.
- Local GR00T/SONIC source and SONIC checkpoint setup under
  `robot_eval_jobs/unitree_groot_n17_sonic_runtime_setup_20260622T152243Z/`.
  The setup cloned official Isaac-GR00T and GR00T-WholeBodyControl sources,
  downloaded bounded GEAR-SONIC deployment/training/sample assets, created the
  `.venv_sim` MuJoCo environment, and started `run_sim_loop.py` until a bounded
  harness timeout. This is Sim2Sim startup evidence only, not an action-command
  proof or manipulation success.
- A concrete Blueprint PolicyServer command wrapper,
  `blueprint-unitree-groot-n17-sonic-policy-server-command`, now converts the
  simulated egocentric frame plus MuJoCo-derived UNITREE_G1_SONIC joint-group
  state into the GR00T PolicyServer request schema. The current bounded proof
  under
  `robot_eval_jobs/unitree_groot_n17_sonic_policy_server_timeout_blocked_20260622T160942Z/`
  reaches the wrapper and then blocks at `gr00t_policy_server_timeout:ping`
  because no GR00T PolicyServer is listening at the configured endpoint.
- Vast and RunPod provider-bundle plumbing for
  `provider_bundle_kind=unitree_unifolm`. This runs a Unitree UnifoLM provider
  bundle as a Unitree policy smoke, not as a WAM bundle, and expects
  `unitree_unifolm_policy_provider_output.json` as the returned policy result.
- A completed RunPod Unitree UnifoLM provider smoke at
  `robot_eval_jobs/unitree_unifolm_runpod_live_probe_20260622T173653Z_framefix_default_retry/`.
  The poll manifest reports `status=completed`,
  `provider_bundle_kind=unitree_unifolm`, `runtime_result_status=completed`,
  no runtime blockers, and no continuing spend from that run. Its imported
  output at
  `robot_eval_jobs/unitree_unifolm_provider_import_20260622T173653Z_framefix_default_retry/`
  proves one Unitree model-backed action-command execution:
  `unitree_unifolm_model_executed=true`,
  `unitree_unifolm_policy_action_command_ran=true`, and a
  Blueprint-compatible `manipulation_contact` action with a 25-step, 23-value
  Unitree UnifoLM action chunk.
- A local authenticated endpoint replay of that Unitree provider output through
  MuJoCo at
  `robot_eval_jobs/mujoco_g1_unitree_unifolm_endpoint_replay_2s_20260622T175441Z_labels/`.
  That run used the endpoint path with `endpoint_policy_used=true`,
  `fixture_policy_used=false`, 20 endpoint invocations, zero rejected actions,
  and a 2-second head-POV video. It is provider-output replay, not fresh
  per-observation inference.
- A strict Unitree UnifoLM endpoint smoke at
  `robot_eval_jobs/mujoco_g1_unitree_unifolm_live_endpoint_1ep_every_step_20260622T210403Z/`.
  The local authenticated Blueprint endpoint was invoked four times with
  `fixture_policy_used=false` and zero rejected actions. The endpoint observed a
  Unitree-family action chunk, but the run is replay-backed:
  `unitree_endpoint_hand_policy_output_observed=true`,
  `unitree_endpoint_hand_policy_used=false`,
  `unitree_endpoint_fresh_policy_action_command_ran=false`, and
  `unitree_endpoint_provider_output_replay_used=true`. This is endpoint/action
  plumbing evidence, not fresh Unitree hand-policy inference.
- A reusable Unitree UnifoLM GPU image-context generator,
  `blueprint-build-unitree-unifolm-gpu-image`, based on Unitree's CUDA 12.4
  install path. The image context does not bake checkpoints or credentials.
- A pushed Unitree UnifoLM SDPA fallback provider image:
  `docker.io/nijelhunt/blueprint-unitree-unifolm:20260622-cu124-sdpa3`.
  Its build-time healthcheck passes on the CPU builder and public registry
  manifest access succeeds, but that is image/runtime packaging evidence only,
  not live G1 hand-policy execution.
- WAM evaluator requery artifacts that require a Unitree G1 hand/manipulation
  policy endpoint before marking the policy/WAM loop as proven. The default
  no-live-WAM path now writes action-conditioned support frames, short MP4
  segments, and `wam_generated_next_observations.jsonl` with projected skeleton
  conditioning, then re-queries the selected Unitree policy on those generated
  observations. This proves simulator-only policy/WAM loop plumbing, not learned
  OSCAR/Cosmos model execution or generated-world rank fidelity.

## Latest Proof Snapshot: 2026-06-22

The current G1 robot-policy direction is Unitree-native. OpenVLA is not selected
as the G1 robot policy, and OSCAR/Cosmos/WAM rollouts are not selected as the
G1 robot policy. The latest local endpoint smoke at
`robot_eval_jobs/mujoco_g1_unitree_unifolm_live_endpoint_1ep_every_step_20260622T210403Z/`
proves authenticated endpoint invocation, normalized action flow, head-POV
observation capture, and MuJoCo/controller execution. It does not prove a fresh
Unitree hand/manipulation policy inference because the endpoint response is
provider-output replay.

The current OSCAR WAM provider proof against that same Unitree endpoint job is
under
`robot_eval_jobs/oscar_wam_unitree_unifolm_live_endpoint_1ep_2s_vast_budgeted_20260622T200023Z/`
and the short-horizon retry is under
`robot_eval_jobs/oscar_wam_unitree_unifolm_live_endpoint_1ep_2s_vast_short9_20260622T200739Z/`.
Both runs executed the learned provider model and generated MP4s, and both Vast
instances were torn down with `continuing_spend_from_this_run=false`. The
49-frame run produced a valid 3.27s MP4 but failed visual quality because later
frames were flat/dark and lost scene structure. The 9-frame run produced a
valid 0.60s MP4 and no longer failed the flat/dark check, but it still lost
scene structure after the first generated frame. Therefore
`action_conditioned_video_rollout_generated=true`, while
`wam_success_label_from_generated_video=false` and
`forward_inverse_consistency_proven=false` remain the correct full-rollout claim
boundary.

The latest no-spend strict re-import artifact is
`robot_eval_jobs/oscar_wam_unitree_unifolm_every_step_requery_strict_20260622T211219Z/`.
It binds the failed short-horizon rollout back to
`contact_or_push_light_object`, spawn `doorway`, task prompt
`Approach the lightweight object and push it forward slightly.`, and camera
`head_pov`. Its `wam_policy_loop_manifest.json` reports
`learned_wam_model_ran=true`, `provider_output_replay_used=true`,
`action_conditioned_video_rollout_generated=true`,
`single_step_policy_requery_frame_useful=true`,
`endpoint_action_returned_for_wam_generated_next_observation=true`,
`unitree_g1_hand_policy_output_observed=true`,
`single_step_wam_policy_requery_proven=false`,
`policy_observes_wam_generated_next_observation=false`,
`unitree_g1_hand_policy_endpoint_used=false`,
`policy_requery_policy_id=unitree_unifolm_vla_policy_provider_replay`,
`policy_requery_provider_replay_used=true`,
`policy_requery_provider_replay_is_not_fresh_policy_observation=true`,
`g1_robot_policy_selected_family=null`,
`openvla_selected_as_g1_robot_policy=false`, and
`wam_rollout_selected_as_g1_robot_policy=false`. This proves endpoint plumbing
and a Unitree-family action shape can flow through the WAM requery path, but it
does not prove a fresh GPU policy inference on the WAM-generated observation, a
full closed-loop episode, every-frame policy/WAM exchange, dexterous
manipulation success, or generated-video success scoring.

Its `wam_manipulation_loop_readiness_manifest.json` also records
`source_unitree_endpoint_hand_policy_used=false`,
`source_unitree_endpoint_fresh_policy_action_command_ran=false`, and
`source_unitree_endpoint_provider_output_replay_used=true`. Its new
`policy_requery_endpoint_readiness_manifest.json` records
`status=ready_for_policy_requery`, `live_policy_requery_endpoint_ready=true`,
and `generated_rollout_visually_useful_for_policy_requery=true`, while
`full_rollout_visually_useful_for_success_review=false`. This artifact is useful
for audit and next-step debugging, not for success scoring, because the generated
rollout still loses scene structure after the first frame.

The aggregate installation gate is
`unitree_policy_stack_installation_audit.json`. It reports
`whole_unitree_policy_stack_installed=true` only when all required components
are configured:

- official Unitree RL Gym locomotion/control root plus checkpoint
- a Unitree-specific manipulation runtime, such as GR00T N1.7 + SONIC,
  LeRobot sim-eval, or UnifoLM VLA/WMA
- a Unitree-specific action-command adapter with a runnable command and
  checkpoint
- for the GR00T/SONIC lane, the simulator-only Sim2Sim command required to
  connect returned SONIC action chunks back into MuJoCo

An RL Gym locomotion provider may be configured while the whole Unitree policy
stack remains `not_installed`. Do not treat `selected_provider:
official_unitree_rl_gym` as proof that Unitree manipulation or action-command
stacks are installed.

The provider registry keeps the legacy `selected_provider` field for older
consumers, but new consumers must read the explicit G1 fields:

- `selected_locomotion_provider`
- `selected_unitree_manipulation_runtime`
- `selected_unitree_action_command`
- `selected_unitree_hand_policy`
- `unitree_hand_manipulation_policy_in_place`
- `openvla_selected_for_g1_policy`
- `wam_selected_for_g1_policy`
- `g1_robot_policy_family_decision`

For the current lane, `unitree_hand_manipulation_policy_in_place=true` requires
both a Unitree manipulation runtime and a Unitree action-command adapter. A
configured OpenVLA endpoint, OSCAR rollout, Cosmos rollout, or Unitree RL Gym
locomotion provider must not flip that field.

Only a fresh Unitree-specific endpoint invocation can set
`unitree_endpoint_hand_policy_used=true` and
`unitree_endpoint_fresh_policy_action_command_ran=true`. A replay-backed
endpoint response may set `unitree_endpoint_hand_policy_output_observed=true`,
but it must also set `unitree_endpoint_provider_output_replay_used=true` and
must keep `unitree_endpoint_hand_policy_used=false`. Even a fresh action-command
proof remains separate from task success, dexterous grasp success, WAM
closed-loop proof, generated-world rank fidelity, and safety
validation.

When `BLUEPRINT_UNITREE_LEROBOT_ROOT` exists and `eval_g1_sim.py` is present,
the audit still separates file availability from runtime execution readiness.
It runs a bounded `eval_g1_sim.py --help` smoke probe and reports
`source_runtime_files_configured`, `source_runtime_execution_ready`, and
`source_runtime_dependency_smoke`. If the files are present but the smoke probe
fails, the candidate reports
`configuration_stage=source_runtime_files_ready_dependency_smoke_failed` and
blocks on `blocked_unitree_lerobot_eval_script_smoke_failed`.

If the smoke probe passes but no LeRobot policy/checkpoint or action command is
configured, the audit reports the LeRobot candidate as a partial source-runtime
setup: `configuration_stage=source_runtime_ready_policy_missing`,
`source_runtime_configured=true`, and `partial_configuration=true`. The whole
stack still remains `whole_unitree_policy_stack_installed=false`.

Not yet proven:

- A trusted or official Unitree G1 Dex1/Dex3/gripper manipulation model command
  returning real action chunks from a local or provider-mounted checkpoint.
  GR00T N1.7 + SONIC remains the preferred next candidate for this gap, and its
  Blueprint action-command wrapper now exists. The current preflight verifies
  the `UNITREE_G1_SONIC` source/schema contract. A bounded RTX 3090 provider
  probe loaded `nvidia/GR00T-N1.7-3B` on CUDA and confirmed the sharper blocker:
  the public base N1.7 checkpoint does not support `UNITREE_G1_SONIC`.
  NVIDIA's PolicyServer reports that `UNITREE_G1_SONIC` is a posttrain tag that
  requires a finetuned checkpoint. The SONIC WBC/ONNX deploy assets are
  controller assets, not a GR00T PolicyServer `--model-path` checkpoint.
  A second bounded RTX 3090 Ti provider probe used the third-party Hugging Face
  checkpoint `LucaFrat/groot-bs16/checkpoint-4000` and produced a real
  Blueprint action command: `unitree_groot_n17_sonic_policy_action_command_ran=true`,
  `unitree_policy_action_command_ran=true`, and a 3120-value SONIC action chunk
  with `left_hand_joints`, `right_hand_joints`, and `motion_token` fields.
  This is still a simulator-only third-party checkpoint proof, not official
  generated-world rank fidelity or task success. A later bounded RTX 3090 Ti provider
  run authenticated to `nvidia/Cosmos-Reason2-2B`, loaded the same third-party
  GR00T/SONIC checkpoint, produced a fresh 3120-value action chunk, consumed it
  through Blueprint's simulator-only MuJoCo Sim2Sim bridge as 40 x 78D SONIC
  frames, and ran a WAM re-observation loop with repeated policy calls. This is
  action-command and simulator-consumption proof only: the GR00T/SONIC chunk is
  not yet integrated into the same contact-task rollout controller, so object
  placement/contact success remains unproven.
- A full closed-loop manipulation episode where a Unitree hand policy observes
  WAM-generated next observations and repeatedly emits hand/arm action chunks.
- Dexterous grasp or task success. Those require simulator traces plus
  generated/executed video review from a VLM or human scorer.

## Source Mapping Checked 2026-06-22

Current public source mapping supports the Unitree-native split:

- Unitree `unitree_lerobot` documents `eval_g1.py`, `eval_g1_sim.py`, and
  `eval_g1_dataset.py` with `--policy.path=.../pretrained_model`,
  `--repo_id=unitreerobotics/G1_Dex3_ToastedBread_Dataset`, `--arm=G1_29`, and
  `--ee=dex3`. In Blueprint, that means a real hand/manipulation proof still
  needs `BLUEPRINT_UNITREE_LEROBOT_POLICY_PATH` and a runnable
  `BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND`.
- Hugging Face LeRobot's Unitree G1 documentation describes G1 simulation,
  teleoperation, training, locomanipulation policies, and controller choices.
  This is relevant to a Unitree-specific runtime, not a generic OpenVLA proof.
- Unitree `unifolm-vla` publishes Unitree-native VLA code/checkpoints for
  humanoid manipulation. Blueprint should treat it as
  `unitree_unifolm_vla_policy` only after a command/checkpoint is configured and
  the action output is decoded into Blueprint/Unitree G1 actions.
- Unitree `unifolm-world-model-action` publishes WMA checkpoints and G1/Z1
  datasets. It is useful for world-model-action or evaluator integration, but a
  generated WMA rollout is not a G1 hand policy unless a Unitree-specific policy
  endpoint consumes the observation and emits normalized G1 actions.
- NVIDIA Isaac-GR00T N1.7 publishes a VLA policy path and the
  GR00T-WholeBodyControl docs describe a G1 `UNITREE_G1_SONIC` inference stack
  with a 78-dimensional action space and MuJoCo Sim2Sim setup. Blueprint treats
  this as `unitree_groot_n17_sonic_policy` only after the configured command
  returns real N1.7/SONIC action chunks.

## Required Env Contract

For the currently verified local G1 locomotion/control stack, source the
machine-local env file:

```bash
source .env.unitree.local
```

That file points the G1 policy root and checkpoint envs at the verified local
Unitree RL Gym snapshot:

```bash
export BLUEPRINT_UNITREE_G1_POLICY_ROOT=/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/robot_eval_jobs/unitree_g1_runtime_setup_20260620T214055Z/runtime_sources/unitree_rl_gym
export BLUEPRINT_UNITREE_G1_POLICY_SOURCE_ROOT="$BLUEPRINT_UNITREE_G1_POLICY_ROOT"
export BLUEPRINT_UNITREE_RL_GYM_ROOT="$BLUEPRINT_UNITREE_G1_POLICY_ROOT"
export BLUEPRINT_UNITREE_G1_POLICY_CHECKPOINT="$BLUEPRINT_UNITREE_G1_POLICY_ROOT/deploy/pre_train/g1/motion.pt"
```

It also points `BLUEPRINT_UNITREE_LEROBOT_ROOT` at the local official Unitree
LeRobot source checkout used for probe/dry-run command construction:

```bash
export BLUEPRINT_UNITREE_LEROBOT_ROOT=/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/robot_eval_jobs/unitree_lerobot_runtime_setup_20260622T000000Z/runtime_sources/unitree_lerobot
```

`BLUEPRINT_UNITREE_LEROBOT_PYTHON` may point at a dedicated Python 3.10/venv
interpreter for the Unitree LeRobot checkout. The audit records that
interpreter and prepends the local Unitree LeRobot checkout plus its pinned
`unitree_lerobot/lerobot/src` submodule to `PYTHONPATH` for the bounded
`eval_g1_sim.py --help` smoke probe and sim-eval subprocess.
`BLUEPRINT_UNITREE_LEROBOT_SMOKE_TIMEOUT_SECONDS` can raise the import-smoke
bound for fresh Python 3.10/Pinocchio/Torch environments; the default is 60
seconds. A passing smoke probe only means the source runtime can import and
start; it is still not a trained manipulation checkpoint, action-command
adapter, task-success, or deployment proof.

That checkout is pinned to Unitree `unitree_lerobot` commit
`41c2805742de879ddab2d8d6beaeaf215f876395` and contains
`unitree_lerobot/eval_robot/eval_g1_sim.py`. The initial full checkout hit an
upstream Git LFS quota blocker on mesh assets, so it was recovered with LFS
smudge disabled. Treat this as source/script availability for Blueprint runtime
probing, not full asset or checkpoint availability.

`BLUEPRINT_UNITREE_G1_POLICY_COMMAND` stays unset until there is a real
Unitree-specific stdin/stdout action wrapper. Do not point it at the direct
`unitree_g1_policy_execution` runner, because that runner takes explicit MuJoCo
execution arguments and does not implement the policy endpoint command contract.

A real Unitree hand/manipulation endpoint requires a trained policy plus a
runnable Unitree-specific action command:

```bash
export BLUEPRINT_UNITREE_LEROBOT_POLICY_PATH='<local pretrained_model path or trusted policy repo id>'
export BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND='<runnable adapter command>'
export BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_CHECKPOINT='<local checkpoint path>'
```

Do not point `BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_CHECKPOINT` at a dataset
repo. Public Unitree Hugging Face entries such as
`unitreerobotics/G1_Dex3_ToastedBread_Dataset` are datasets for training or
validation, not trained policy checkpoints.

For GR00T N1.7 + UNITREE_G1_SONIC:

```bash
export BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT='<Isaac-GR00T checkout>'
export BLUEPRINT_UNITREE_GROOT_N17_SONIC_WBC_ROOT='<GR00T-WholeBodyControl checkout>'
export BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT='<finetuned GR00T N1.7 UNITREE_G1_SONIC checkpoint path or repo id>'
export BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT='<nvidia/GEAR-SONIC or local SONIC checkpoint path>'
export BLUEPRINT_UNITREE_GROOT_N17_SONIC_HF_TOKEN_FILE='<optional file containing an HF token for private or gated checkpoint access>'
export BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND='blueprint-unitree-groot-n17-sonic-policy-server-command'
export BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL='tcp://127.0.0.1:5550'
export BLUEPRINT_UNITREE_GROOT_N17_SONIC_SIM2SIM_COMMAND='<MuJoCo SONIC Sim2Sim command>'
```

The Blueprint adapter command is still available when an outer command-adapter
artifact is needed:

```bash
blueprint-unitree-groot-n17-sonic-policy-command-adapter
```

The runtime audit and provider-smoke helpers are:

```bash
blueprint-run-unitree-groot-n17-sonic-policy-runtime --job-dir robot_eval_jobs/<job_id>
blueprint-run-unitree-groot-n17-sonic-policy-server-preflight --job-dir robot_eval_jobs/<job_id>
blueprint-run-unitree-groot-n17-sonic-policy-provider-smoke \
  --job-dir robot_eval_jobs/<job_id> \
  --frame-path <simulated-policy-frame>
```

These commands are simulator/proof-lane commands. They must use the configured
simulation/provider lane, and a dry run or provider-output import is not fresh
model execution.

When using repository checkpoints, keep Hugging Face auth file-based. The runtime
and preflight audits record whether `BLUEPRINT_UNITREE_GROOT_N17_SONIC_HF_TOKEN_FILE`
exists without writing the token value. The current setup can authenticate to
`nvidia/GR00T-N1.7-3B` and `nvidia/Cosmos-Reason2-2B`, but the base N1.7 repo is
not a `UNITREE_G1_SONIC` manipulation checkpoint. The third-party
`LucaFrat/groot-bs16/checkpoint-4000` checkpoint has run through the Blueprint
action-command contract, simulator-only SONIC action consumption, and WAM loop
in simulator-only mode. Do not promote that to trusted Unitree deployment proof,
generated-world rank fidelity, or task success until a trusted checkpoint and a
policy-chunk-integrated successful contact/manipulation rollout are proven.

## Unitree LeRobot G1 Sim-Eval Provider

The job-level LeRobot lane is separate from the official RL Gym locomotion lane.
Official RL Gym G1 locomotion policy execution remains proven only by the
official RL Gym artifacts, such as `unitree_policy_runtime_truth_boundary.json`
and `official_g1_policy_handoff/robot_team_handoff_manifest.json`.

The LeRobot provider writes its own artifact family:

- `unitree_lerobot_g1_runtime_probe.json`
- `unitree_lerobot_g1_policy_runtime_truth_boundary.json`
- `unitree_lerobot_g1_policy_runtime_summary.json`
- `unitree_lerobot_g1_policy_handoff/command.json`
- `unitree_lerobot_g1_policy_handoff/stdout.log`
- `unitree_lerobot_g1_policy_handoff/stderr.log`
- `unitree_lerobot_g1_policy_handoff/robot_team_handoff_manifest.json`
- `unitree_lerobot_g1_policy_handoff/task_data/...` when the runtime writes data

Supported modes:

- `probe`: inspect env vars and filesystem only.
- `dry_run`: build the exact `eval_g1_sim.py` command without subprocess inference.
- `sim_eval`: run the configured Unitree LeRobot sim eval and capture logs/artifacts.
- `not_configured`: structured missing-requirements output when root, script, or policy config is absent.

Required or supported env vars:

```bash
export BLUEPRINT_UNITREE_LEROBOT_ROOT="/path/to/unitree_lerobot"
export BLUEPRINT_UNITREE_LEROBOT_POLICY_PATH="/path/to/pretrained_model"
export BLUEPRINT_UNITREE_LEROBOT_DATASET_REPO_ID="unitreerobotics/G1_Dex3_ToastedBread_Dataset"
export BLUEPRINT_UNITREE_LEROBOT_DATASET_ROOT=""
export BLUEPRINT_UNITREE_LEROBOT_TASK="pick up the target object"
export BLUEPRINT_UNITREE_POLICY_FAMILY="pi05"
export BLUEPRINT_UNITREE_G1_ARM="G1_29"
export BLUEPRINT_UNITREE_G1_EE="dex3"
export BLUEPRINT_UNITREE_LEROBOT_FREQUENCY="30"
export BLUEPRINT_UNITREE_LEROBOT_EPISODES="0"
export BLUEPRINT_UNITREE_LEROBOT_MAX_EPISODES="1200"
export BLUEPRINT_UNITREE_LEROBOT_VISUALIZATION="true"
export BLUEPRINT_UNITREE_LEROBOT_SAVE_DATA="true"
export BLUEPRINT_UNITREE_LEROBOT_SEND_REAL_ROBOT="false"
export BLUEPRINT_UNITREE_ALLOW_REAL_ROBOT_COMMANDS="false"
export BLUEPRINT_UNITREE_ALLOW_DOWNLOADS="false"
```

Probe:

```bash
blueprint-run-unitree-lerobot-g1-policy-eval \
  --job-dir robot_eval_jobs/unitree_lerobot_g1_probe \
  --mode probe
```

Dry run:

```bash
blueprint-run-unitree-lerobot-g1-policy-eval \
  --job-dir robot_eval_jobs/unitree_lerobot_g1_dry_run \
  --mode dry_run
```

Sim eval:

```bash
blueprint-run-unitree-lerobot-g1-policy-eval \
  --job-dir robot_eval_jobs/unitree_lerobot_g1_sim_eval \
  --mode sim_eval
```

The command builder prefers `eval_g1_sim.py` under these local layouts:

- `unitree_lerobot/eval_robot/eval_g1_sim.py`
- `eval_robot/eval_g1_sim.py`

It detects `eval_g1.py` paths for reporting, but the Blueprint provider does not
select a real-robot script by default. Any path that would send real robot
commands is blocked unless `BLUEPRINT_UNITREE_ALLOW_REAL_ROBOT_COMMANDS=true`;
generated-world rank fidelity and generated-world rank fidelity remain false.

`BLUEPRINT_UNITREE_ALLOW_DOWNLOADS=false` is the default. Tests use fake local
roots and scripts and must not download Hugging Face models, datasets, videos,
or checkpoints.

Pi0.5, GR00T, and Unitree VLA can be represented through
`BLUEPRINT_UNITREE_POLICY_FAMILY`, but LeRobot/VLA execution is proven only when
the configured LeRobot provider successfully runs a sim eval and produces
captured logs/artifacts. OpenVLA remains a comparison or endpoint family unless
an explicit Unitree G1 action adapter is configured and validated. UnifoLM-VLA
and UnifoLM-WMA remain optional providers until local commands/checkpoints or
endpoints exist; `wam_world_model_used` stays false unless a WMA runtime is
actually invoked.

For Unitree UnifoLM VLA:

```bash
export BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND='<runnable adapter command>'
export BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT='<local checkpoint path>'
export BLUEPRINT_UNITREE_UNIFOLM_POLICY_CHECKPOINT='<provider-facing alias accepted by image/provider runners>'
export BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT='<local UnifoLM VLM backbone path>'
```

`BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT` is the source-specific VLA env.
`BLUEPRINT_UNITREE_UNIFOLM_POLICY_CHECKPOINT` is accepted by the Blueprint
adapter, provider bundle, and generated GPU image as a stable provider-facing
alias. If both are set, source-specific launcher paths may prefer the VLA env;
do not point either value at a dataset repo.

For Unitree UnifoLM WMA:

```bash
export BLUEPRINT_UNITREE_UNIFOLM_WMA_COMMAND='<runnable adapter command>'
export BLUEPRINT_UNITREE_UNIFOLM_WMA_CHECKPOINT='<local checkpoint path>'
```

The repo-local wrapper command is:

```bash
blueprint-unitree-unifolm-policy-command-adapter --mode vla
blueprint-unitree-unifolm-policy-command-adapter --mode wma
```

For the Unitree UnifoLM-VLA FastAPI server path, the source server command from
the cloned Unitree repo is:

```bash
python deployment/model_server/run_real_eval_server.py \
  --ckpt_path /path/to/UnifoLM-VLA-Base/checkpoints/pytorch_model.pt \
  --port 8777 \
  --unnorm_key g1_stack_block \
  --vlm_pretrained_path /path/to/UnifoLM-VLM-Base
```

The public Unitree checkpoint route checked on 2026-06-22 is:

- VLA checkpoint: `unitreerobotics/UnifoLM-VLA-Base/checkpoints/pytorch_model.pt`
  (`pytorch_model.pt` is roughly 19 GB)
- VLM backbone: `unitreerobotics/UnifoLM-VLM-Base`
- server action key: `--unnorm_key g1_stack_block`

Creating the Blueprint HTTP endpoint is not enough to make this true. The
endpoint only wraps a command. A live Unitree hand-policy proof requires the GPU
worker to download or mount both model repos, start the UnifoLM server, load the
checkpoint, and answer `/act` for each observation. Provider-output replay may
prove a prior action result, but it must keep current-invocation fields such as
`unitree_unifolm_policy_action_command_ran=false`.

That server expects `POST /act` with Unitree-shaped `observations`. Blueprint
uses a bridge command to translate `{"observation": ...}` packets into the
Unitree server payload:

```bash
blueprint-unitree-unifolm-vla-server-bridge \
  --server-url http://127.0.0.1:8777/act
```

Set `BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND` to that bridge command only after a
real UnifoLM server is running. If the server is not listening, the adapter must
emit `blocked_unitree_unifolm_vla_server_call_failed:URLError`; that is not a
policy proof.

The provider-smoke/import command is:

```bash
blueprint-run-unitree-unifolm-policy-provider-smoke \
  --job-dir /path/to/job \
  --frame-path /path/to/head_or_torso_frame.png \
  --mode vla
```

The Vast adapter must use the Unitree-specific bundle kind:

```bash
python -m blueprint_pipeline.vast_provider_adapter \
  --provider-bundle-kind unitree_unifolm \
  --enable-blueprint-bundle \
  --provider-bundle /path/to/unitree_unifolm_policy_provider_runtime_bundle.zip
```

That path proves remote bundle download, Unitree entrypoint execution, and
returned policy JSON only after a live provider run returns
`unitree_unifolm_policy_provider_output.json`. It does not by itself install
UnifoLM dependencies or start a checkpoint-backed `/act` server.

The RunPod async runner supports the same Unitree bundle kind:

```bash
python -m blueprint_pipeline.runpod_wam_async_runner create \
  --provider-bundle-kind unitree_unifolm \
  --image-name docker.io/<user>/blueprint-unitree-unifolm:<versioned-tag> \
  --bundle-path /path/to/unitree_unifolm_policy_provider_runtime_bundle.zip \
  --provider-bundle-url-file /path/to/provider_bundle_url.txt \
  --provider-output-put-url-file /path/to/provider_output_put_url.txt \
  --provider-output-get-url-file /path/to/provider_output_get_url.txt \
  --allow-paid-runpod-launch
```

Despite the legacy module name, `provider_bundle_kind=unitree_unifolm` selects
the Unitree entrypoint, forwards Unitree UnifoLM runtime env defaults, and does
not execute the WAM provider runner. The provider bundle must include
`provider_runtime/input_frame.png`; the remote runner rewrites the observation's
`camera_frame_path` to that pod-local frame before invoking the policy command.
If the returned JSON says `unitree_unifolm_policy_action_command_ran=false`, the
hand/manipulation policy is still not in place.
Provider-output replay is intentionally reported separately:
`provider_output_replay_used=true` and
`provider_unitree_unifolm_policy_action_command_ran=true` can prove that a prior
provider job returned a Unitree action, but replay keeps
`unitree_unifolm_policy_action_command_ran=false` for the current invocation and
does not set `unitree_hand_manipulation_policy_used`.

Build the reusable image context with:

```bash
blueprint-build-unitree-unifolm-gpu-image \
  --job-dir /path/to/unitree_unifolm_gpu_image_context \
  --image-ref docker.io/<user>/blueprint-unitree-unifolm:<versioned-tag>
```

The default generated image uses `--dependency-profile inference`. That profile
keeps the Unitree server/runtime dependencies needed for a bounded policy call
including TensorFlow image preprocessing, Qwen/Transformers, Diffusers action
modules, FastAPI, and the Blueprint bridge, while excluding training/data-stack
packages such as `datasets`, `deepspeed`, W&B, `tensorflow_datasets`, and
`tensorflow_graphics`. Use `--dependency-profile full` only when intentionally
building Unitree's broader training/data image; it is not required for the fast
fresh-policy proof lane.

Then build/push using the generated `build_image.sh` and `push_image.sh`, and
use the pushed image as `--public-image` for the Vast adapter. Image build/push
is still not policy execution proof; the proof starts only when the live provider
loads checkpoints and returns policy JSON.

The generated image also includes `run_unitree_unifolm_vla_policy_once`. That
command bridges the Blueprint policy-command contract to Unitree's `/act`
server: it starts `run_unitree_unifolm_vla_server`, waits for the server port,
then calls `blueprint-unitree-unifolm-vla-server-bridge`. If checkpoint paths are
not mounted, the image launcher can download the public
`unitreerobotics/UnifoLM-VLA-Base` and `unitreerobotics/UnifoLM-VLM-Base` repos
on the GPU worker when `BLUEPRINT_UNITREE_UNIFOLM_ALLOW_HF_DOWNLOAD=true`.
Those weights are runtime material, not image-baked artifacts.

For a long-lived RunPod `/act` server, use:

```bash
export BLUEPRINT_ALLOW_RUNPOD_API_CALLS=true
export BLUEPRINT_ALLOW_RUNPOD_POD_LAUNCH=true

blueprint-launch-unitree-unifolm-runpod-server launch \
  --job-dir robot_eval_jobs/unitree_unifolm_runpod_server_<timestamp> \
  --image-name docker.io/nijelhunt/blueprint-unitree-unifolm:20260622-cu124-sdpa3 \
  --allow-paid-runpod-launch
```

The launch manifest writes a proxy URL shaped like
`https://<pod_id>-8777.proxy.runpod.net/act`. Point the local command adapter at
that URL:

```bash
export BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND="python -m blueprint_pipeline.unitree_unifolm_vla_server_bridge --server-url https://<pod_id>-8777.proxy.runpod.net/act"
export BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT=unitreerobotics/UnifoLM-VLA-Base
export BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT=unitreerobotics/UnifoLM-VLM-Base
```

This is the path that can make `unitree_endpoint_hand_policy_used=true`, but
only after the remote server is actually reachable and returns a fresh action
for the current observation. Delete the pod when finished:

```bash
blueprint-launch-unitree-unifolm-runpod-server poll \
  --job-dir robot_eval_jobs/unitree_unifolm_runpod_server_<timestamp>

blueprint-launch-unitree-unifolm-runpod-server probe \
  --job-dir robot_eval_jobs/unitree_unifolm_runpod_server_<timestamp>

blueprint-launch-unitree-unifolm-runpod-server delete \
  --job-dir robot_eval_jobs/unitree_unifolm_runpod_server_<timestamp>
```

`poll` reads RunPod pod state. `probe` reads the public status proxy and records
whether the backend `run_unitree_unifolm_vla_server` process is running, whether
a backend startup error occurred, and the redacted server log tail. A running
pod without a reachable proxy or running backend is not a hand-policy proof.

When a provider smoke completes, point the evaluator at it with:

```bash
export BLUEPRINT_UNITREE_UNIFOLM_PROVIDER_SMOKE_JOB_DIR=/path/to/unitree_unifolm_policy_provider_smoke_job
```

The wrapper should be pointed at a GPU-side UnifoLM runner command through the
matching `BLUEPRINT_UNITREE_UNIFOLM_*_COMMAND` env var. The wrapper forwards the
Blueprint observation packet, checkpoint path, source root, policy mode, and
camera frame path to the runner and expects a Blueprint-compatible `action` or
action chunk JSON in return.

These commands must read Blueprint observation packets and write
Blueprint-compatible policy actions. A downloaded checkpoint alone does not
prove endpoint execution.

OpenVLA, OSCAR, Cosmos, and fixture WAM envs should remain unset for Unitree G1
policy proof unless the run is explicitly a comparison/evaluator job. They are
not fallback G1 policy paths.

## Proof Boundary

The following are true only after a real Unitree policy command runs through the
HTTP endpoint path and emits normalized actions for the episode:

- `real_model_endpoint_ready`
- `single_step_wam_policy_requery_proven`
- `unitree_g1_hand_policy_endpoint_used`
- `unitree_hand_policy_requery_used`
- `unitree_g1_dexterous_manipulation_proven`

In endpoint-eval artifacts, `unitree_endpoint_hand_policy_output_observed=true`
means the endpoint saw Unitree-family action output, including replayed provider
output. `unitree_endpoint_hand_policy_used=true` is stricter: it means a fresh
per-observation Unitree hand-policy command ran for that episode. Replayed
provider output must keep `unitree_endpoint_hand_policy_used=false` and
`unitree_endpoint_provider_output_replay_used=true`.

`real_vla_or_unitree_hand_policy_endpoint_used` can still appear in artifacts as
a backward-compatible alias, but for G1 claims it mirrors the stricter
Unitree-specific endpoint decision. A generic OpenVLA action does not satisfy
that field.

The following do not prove G1 hand policy success by themselves:

- an OpenVLA provider smoke action
- an OSCAR/Cosmos/Unitree WMA generated rollout
- a local reference adapter action
- a replayed provider output
- a Unitree locomotion-only controller run
- a public checkpoint existing on Hugging Face

## Evaluator Loop

The intended closed-loop shape is:

```text
scene + G1 observation
  -> Unitree-specific policy endpoint emits action chunk
  -> WAM/evaluator predicts or scores next world observation
  -> WAM-derived perception harness emits derived support observations and checks
  -> policy observation adapter withholds unsupported masks/depth/contact fields
  -> policy re-observes a useful generated next observation
  -> repeat until task end
  -> separate VLM/human scorer labels success from generated or executed video
```

The current repo has adapter contracts, a default local action/skeleton
conditioned support generator, and fail-closed manifests for this loop. It must
continue to block rather than fake success when a runnable Unitree hand-policy
command or checkpoint is missing. A default local WAM completion is not a live
learned OSCAR/Cosmos completion, generated-world rank fidelity, off-scope
validation, or task-success proof.

The WAM-derived harness can recommend early termination when generated media is
too weak for reliable policy requery, target identity is lost, the target is
offscreen, relative depth jumps, or the action trace no longer matches visual
motion. That recommendation blocks success scoring unless an explicit review
path accepts the artifact. Harness outputs are derived observations, not real
sensors; inferred depth is not sensor depth, object masks are not physical truth,
and contact likelihood is not physical contact proof.

The default Unitree loop keeps the policy interface RGB plus declared nominal
state. Optional perception backends such as SAM-style segmentation, tracking, or
depth commands can be enabled only through
`BLUEPRINT_WAM_PERCEPTION_HARNESS_BACKEND_KIND`,
`BLUEPRINT_WAM_PERCEPTION_HARNESS_BACKEND_COMMAND`, and
`BLUEPRINT_ALLOW_WAM_PERCEPTION_HARNESS_EXTERNAL_BACKEND=true`; if the gate or
command is missing, the backend result is blocked and the fixture/object-index
path remains the local default. Even when a real backend runs, the policy
receives masks/depth/contact fields only if its declared observation schema
supports them. Otherwise those fields are limited to diagnostics, validation
metrics, false-success reduction analysis, review reports, and
early-termination/scoring gates.

For a sim-only architecture proof of the provider path, run
`python -m blueprint_pipeline.wam_sim_provider_e2e --provider-mode real` with a
generated frame and optional SAM3/depth/pose configuration. That proves the
generated-frame -> provider -> harness -> adapter -> gated-requery artifact
path, but it does not change the default Unitree policy interface or prove
perception accuracy, generated-world rank fidelity, off-scope validation, or
real-world task success.
