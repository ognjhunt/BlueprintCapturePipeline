# G1 asset for a shared Task Evaluation Run

The IsaacLab-Arena `G1_WITH_DEX3` embodiment at commit
`8b82dca224f2b5af08f339f987613c59ce9cdbaa` points to
`Samples/Groot/Robots/g1_29dof_with_hand_rev_1_0.usd` in the Isaac 6.0
asset root. The manifest in `configs/g1_arena_robot_asset_isaac_6.sha256manifest`
pins the observed 38,195,671 byte file and its SHA-256. Download its exact
bytes into the evaluation evidence root with the existing digest-checking fetcher:

```bash
python deploy/docker/robot_eval_worker/groot_oscar_closed_loop/fetch_pinned_isaac_assets.py \
  --manifest configs/g1_arena_robot_asset_isaac_6.sha256manifest \
  --base-url https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Samples/Groot/Robots/ \
  --output-dir "$EVIDENCE_ROOT/robot"
```

Pass `robot/g1_29dof_with_hand_rev_1_0.usd` as the robot asset source in the
same native task arena packet as the site scene and task. The packet copies the
verified USD and local dependencies. This USD has no local file dependencies;
it references `OmniPBR.mdl`, which must resolve in the Isaac/Kit runtime. The
packet records that exact symbolic dependency, and the runtime rejects other
unlisted material assets. A native GPU scene construction run must still verify
the material and robot articulation; a local USD parse is not that proof.

The USD's default prim is `/g1_29dof_with_hand_rev_1_0`. For the right Dex3
hand, the observed rigid contact bodies are
`right_hand_index_1_link`, `right_hand_thumb_2_link`, and
`right_hand_middle_1_link` under that prim. Configure the task contact paths
against the selected hand and validate them against the same pinned USD before
running an episode. These are robot bodies inside the shared site scene, not a
separate G1 scene.

`build_pinned_g1_arena_robot_configuration` in
`src/blueprint_pipeline/native_g1_arena_robot_configuration.py` authors the
packet robot row from this exact USD and the pinned Arena G1 configuration.
Pass the USD path inside the evidence root, the evidence root itself, a
`base_pose_world` with `position_world_m` and `orientation_xyzw`, and the task
hand (`left` or `right`). It checks the USD bytes, reads and converts all 43
joint limits to radians, and binds the chosen Dex3 contact bodies. The returned
row is ready for the existing native task packet builder.

For a rigid pick-and-place task, set `robot.grasp_frame` to
`{"kind":"body_midpoint","body_names":["right_hand_index_1_link","right_hand_thumb_2_link"]}`
and include both exact paths in `robot.task_contact_body_paths`. The shared
rigid readback checks this relation before an episode and derives the grasp
midpoint from those two live Dex3 bodies. The G1 spawn adapter applies all 43
configured reset joint positions to the Arena articulation's initial state.

`native_g1_shared_scene_episode.py` can sequence pinned HumanoidArena
manipulation and vision-navigation candidates through the same Arena scene. It
binds the prompt to the scene task or explicit navigation goal, uses the
observed head camera and 64-value state for each policy
query, sends every returned action through the pinned SONIC target bridge, and
retains calibrated lossless head input and head/overview review PNGs plus a task
sample at each step. The shared media finalizer seals derived H.264 review
videos for both cameras, linked to the immutable frame manifest. Its result is
still an unscored development trace: the production worker must attest the
executing checkpoint and controller, score the bounded task, and retain a
terminal episode receipt before any candidate is offered as runnable. The
navigation candidates use the same semantic-v3 action and SONIC controller
binding, but exact checkpoint inference remains unobserved.

`run_g1_built_scene_policy_episode` now connects that loop to the existing
native G1 joint environment, live rigid-task scene readback, and shared task
scorer. It rechecks the offline scene/model/source preflight, retains the
reset sample before the first policy query, then writes a trace and a scored
development receipt after the episode. A scored failure is still a valid
observed outcome. The receipt keeps `ranking_eligible` and policy-runtime
identity false: the qualified worker has not yet bound the running server to
the checkpoint or admitted G1 in the published bundle/run path. The separate
navigation-goal score has the same development-only ceiling.

The two box checkpoints are separately pinned in
`configs/g1_humanoidarena_checkpoint_inventory.v1.json`. A team can stage one
candidate at a time, then recheck exact bytes without network access:

```bash
python scripts/fetch_g1_humanoidarena_checkpoint.py \
  --candidate humanoidarena_dp_g1_dex3_sonic \
  --output-dir "$CHECKPOINT_ROOT"
python scripts/fetch_g1_humanoidarena_checkpoint.py \
  --candidate humanoidarena_dp_g1_dex3_sonic \
  --output-dir "$CHECKPOINT_ROOT" --verify-only
```

The other candidate id is `humanoidarena_pi05_g1_dex3_sonic`. The fetcher
streams each file from ModelScope to a temporary local file and publishes it
only after the inventory SHA-256 and byte size match. This is checkpoint
storage proof, not a running or licensed policy. Verify the applicable model
terms and bind the executing server process to these bytes before admitting
either candidate to an evaluation run.

The same inventory also pins two `HSI_vision_navi` movement candidates:
`humanoidarena_dp_g1_dex3_sonic_vision_navi` and
`humanoidarena_pi05_g1_dex3_sonic_vision_navi`. Their published configurations
use the same front image, 64-value observation state, 40-value semantic action,
and SONIC controller interface as the box candidates. Stage either with the
same fetch command and its candidate id. These are navigation checkpoints for
moving to a marked area; their file identities do not establish a runnable
navigation task in Blueprint's captured site. The explicit goal and its own
visible marker below supply a development task contract; checkpoint/server attestation
and live episode evidence are still required before offering them as runnable.

Before a local or container attempt, run the same offline preflight against
the staged packet and runtime inputs. `--bundle-root` is the root holding the
packet's relative `assets/` paths; `--checkpoint-root` is the fetcher's output
directory. Supply the checked-out official HumanoidArena
`serve_lerobot_vla_http.py` and `action_provider_sonic.py` source files and
the SONIC encoder/decoder ONNX files. The command checks the G1 scene packet,
its USD closure and articulation, each candidate checkpoint file, the pinned
server and SONIC source hashes, and the caller-declared SONIC model hashes:

```bash
PYTHONPATH=src python scripts/preflight_g1_shared_scene_run.py \
  --scene-plan "$BUNDLE_ROOT/native_task_arena_scene_plan.v1.json" \
  --bundle-root "$BUNDLE_ROOT" \
  --inventory configs/g1_humanoidarena_checkpoint_inventory.v1.json \
  --candidate humanoidarena_dp_g1_dex3_sonic \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --policy-server-source "$POLICY_SERVER_SOURCE" \
  --sonic-provider-source "$SONIC_PROVIDER_SOURCE" \
  --sonic-encoder "$SONIC_ENCODER" --sonic-encoder-sha256 "$SONIC_ENCODER_SHA256" \
  --sonic-decoder "$SONIC_DECODER" --sonic-decoder-sha256 "$SONIC_DECODER_SHA256"
```

The receipt says `staged_inputs_verified` and records the exact selected
candidate, scene digest, and file identities. It explicitly reports that no
server or SONIC process was attested and no episode or score exists. It does
not make a movement candidate runnable: the shared episode loop and
navigation score still need a live checkpoint/controller run. The π0.5 upstream configs also
contain absolute base-model references that need a verified loader mapping
before their servers can be admitted.

For an authorized development run, `start_g1_policy_server` in
`native_g1_policy_server_supervisor.py` owns the pinned HumanoidArena HTTP
server as a child process. It reruns the staged-input check, requires the
official checkout at the inventory's exact source revision with no local
changes, refuses an unverified absolute π0.5 base-model reference, launches
the selected checkpoint on loopback, checks that the child owns the listener,
and requires the official `/reset` acknowledgement. Its lease must be closed
in `finally`; closing it waits for that child to exit. The receipt records
the launch command, interpreter hash, candidate, scene, and listener PID.
The server's response does not expose loaded-weight identity, so the receipt
keeps `loaded_checkpoint_identity_observed=false` and makes no inference or
task-success claim. `run_g1_supervised_built_scene_episode` in
`native_g1_runtime_assembly.py` now owns this lease and the pinned SONIC
controller through one built-scene development episode. It records the scored
episode digest and child teardown in a terminal receipt. The production policy
worker does not invoke this assembly yet, and no GPU run has verified it.

`python -m blueprint_pipeline.native_g1_development_worker --request
<sealed-request.json> --output-dir <new-attempt-directory>` is the local
development worker for that assembly. It consumes the same sealed
`native_task_arena_packet` that the Franka path uses. Its request schema is
`native_g1_development_episode_request.v1`; it names the packet root, selected
candidate, checkpoint inventory/root, pinned server and SONIC sources,
encoder/decoder paths and hashes, policy-server Python executable, loopback
port, `cuda:0`, maximum steps, and the existing Isaac runtime provisioning
receipt. The request and embedded rights review both carry canonical digests.
The review must name a human reviewer, bind the candidate, scene and exact
inventory file, and record review of the checkpoint's inherited model terms
plus source and SONIC terms for development simulation. A receipt with those
fields is a recorded decision; the validator cannot establish that the
reviewer actually had authority or that publication rights are granted.

Before starting Isaac, the worker rechecks the packet receipt and every
staged source/model byte, then compares the preflight scene digest with the
sealed packet. It refuses missing rights review. It builds the selected G1
in that packet, checks the device readback, runs the supervised episode, and
closes the environment and simulator even after failure. The worker writes
`native_g1_development_worker_result.v1.json` with the episode receipt and
teardown states. This is a development simulator path. The published policy
bundle, published launch profile, live checkpoint inference, movement
clearance, and public video approval remain separate gates.

The same worker can be staged for the pinned Isaac Sim 6.0.1 container. Use a
Linux x86-64 NVIDIA Docker host with the digest-pinned image already present,
the verified native runtime source packet and its source receipt, and a request
whose scene, candidate, model hashes, and human rights review are sealed. The
request's `python_executable` must name a separately provisioned Linux LeRobot
policy interpreter inside `--policy-runtime-root`. The pinned HumanoidArena
release uses separate Isaac and LeRobot environments because their Python
dependencies differ. Build the policy environment with its own pinned package
closure and a copied interpreter inside the same container filesystem layout;
symlinks from the policy environment to a host Python outside that root are
rejected. Its LeRobot import must resolve to the exact clean HumanoidArena
checkout at the pinned revision. The launcher mounts that full monorepo, the
policy environment, packet, models, and runtime source packet read-only. It
probes policy imports and CUDA before provisioning Isaac Lab/Arena without
network, then invokes the existing worker with the same
scene/candidate/rights fields. Its output directory is the only writable bind
mount. Before writing a plan, the launcher verifies the sealed packet and
runtime source archive, checks the selected checkpoint, server and SONIC bytes,
and validates the candidate-specific rights review. Navigation candidates also
need a team-confirmed goal authority bound to this scene. The plan records
those host checks under `host_preflight`; USD articulation, policy imports,
CUDA, and the actual episode still require the container worker. Planning
without `--execute` starts no container:

```bash
PYTHONPATH=src python -m blueprint_pipeline.native_g1_container_run \
  --request "$SEALED_G1_REQUEST" \
  --source-receipt "$NATIVE_RUNTIME_SOURCE_RECEIPT" \
  --source-packet "$NATIVE_RUNTIME_SOURCE_PACKET" \
  --policy-runtime-root "$POLICY_RUNTIME_ROOT" \
  --output-dir "$NEW_G1_ATTEMPT_DIR"
```

Review `native_g1_container_run_plan.v1.json`, then use the same command with
`--execute` and a **new** output directory. The launcher requires the pinned
image locally (`--pull never`), GPU 0, and loopback-only container networking;
it retains `container.log`, the runtime provisioning receipt, and the worker's
terminal receipt in the output directory. A staged plan proves only a transport
recipe. A successful Docker exit still needs the worker receipt, media, scorer,
checkpoint runtime identity, and teardown inspected before any runnable WebApp
profile or video claim is published.

For the two pinned `HSI_vision_navi` candidates, the same rigid-task packet
may include a `task_spec.g1_navigation_goal` side objective. The task still
uses its original site, object, scene plan, and manipulation prompt. The
movement candidate receives its published instruction, “Avoid obstacles and
move to the yellow marked area.” The goal contract requires a measured world
center, acceptance radius, maximum root-height drift, and settle window. It
must exactly match a non-colliding `flat_yellow_disc` marker nested under
`task_spec.g1_navigation_goal.visible_target_marker`. The Arena builder renders
that cue alongside the task's `task_spec.visible_target_marker` when present, so
the manipulation target and captured site assets remain in the same scene.

For example, add this field to the existing rigid task spec before sealing the
packet (the coordinates must come from that site's measured goal). Retain the
existing manipulation marker and task prompt as they are:

```json
{
  "g1_navigation_goal": {
    "schema_version": "native_g1_navigation_goal.v1",
    "center_world_m": [2.0, 0.0, 0.0],
    "acceptance_radius_m": 0.3,
    "max_root_height_drift_m": 0.2,
    "settle_window_samples": 2,
    "task_instruction": "Avoid obstacles and move to the yellow marked area.",
    "visible_target_marker": {
      "schema_version": "native_task_target_marker.v1",
      "shape": "flat_yellow_disc",
      "non_colliding": true,
      "surface_position_world_m": [2.0, 0.0, 0.0],
      "radius_m": 0.4
    }
  }
}
```

The navigation scorer reads G1 root position from Isaac at reset and after
each SONIC-controlled action. It requires the robot to start outside the goal
and hold inside the visible target for the terminal settle window without a
large root-height drop. The receipt reports measured distance, first settled
step, and terminal hold. It explicitly leaves obstacle-clearance scoring
false, so it cannot establish the full “avoid obstacles” behavior or physical
navigation readiness. Navigation remains development-only until a compatible
checkpoint runs and the goal/clearance evidence is inspected.

Before a navigation candidate can launch, the site/robot team must confirm
that exact measured goal for this task. After that decision is recorded, use
`seal_g1_navigation_goal_authority` in `native_g1_navigation_goal.py` with the
sealed scene plan, confirming team id, and human reviewer. Put the resulting
`native_g1_navigation_goal_authority.v1` object in the worker request as
`navigation_goal_authority` and reseal the request digest. The worker checks
the confirmed rigid-task contract, team, site, task, scene-plan digest, and
goal bytes before Isaac starts. The authority explicitly confirms goal arrival
and terminal hold only; it records that obstacle clearance and physical
outcome are unscored. A caller-supplied reviewer name records a decision but
does not itself prove the reviewer had authority. The separate rights review
and live checkpoint/episode gates still apply.

To derive both worker requests from the same published Task Evaluation Run
setup catalog, use the no-spend selection stage. The runtime template contains
the shared worker fields (packet, inventory, checkpoint root, pinned sources,
SONIC models and hashes, policy Python, port, device, and step cap), but no
candidate, rights review, or request digest. Supply a separate already-reviewed
rights receipt for each selected candidate:

```bash
PYTHONPATH=src python -m blueprint_pipeline.native_g1_development_selection \
  --setup "$PUBLISHED_POLICY_SETUP" --objective task_success \
  --runtime-template "$G1_SHARED_RUNTIME_TEMPLATE" \
  --rights-review "humanoidarena_dp_g1_dex3_sonic=$G1_DP_RIGHTS" \
  --rights-review "humanoidarena_pi05_g1_dex3_sonic=$G1_PI05_RIGHTS" \
  --output-dir "$NEW_G1_SELECTION_DIR"
```

Use `--objective g1_navigation_goal`, the corresponding two `_vision_navi`
rights receipts, and `--navigation-authority` for the movement pair. The stage
checks that the selected pair is in the published G1 catalog and belongs to
the same objective, that the packet matches the selected task and site, and
that each rights review binds its candidate, scene, and inventory. It writes
two sealed request JSON files and a digest-bound plan. The `--selection` option
accepts a sealed selection record carrying the same setup, robot, candidate,
objective, and scene identities; it is the input seam for the existing WebApp
configurator. Staging never runs a policy or changes catalog readiness.

To attempt both policies for one objective on the same sealed task/site, create
one worker request per candidate. Keep every runtime and scene field identical;
only `candidate_id`, its candidate-specific `rights_review`, and the resulting
`request_digest` may differ. For navigation, both requests must carry the same
team-confirmed goal authority. Validate the pair without starting Isaac or a
container:

```bash
PYTHONPATH=src python -m blueprint_pipeline.native_g1_development_pair \
  --request "$SEALED_G1_DP_REQUEST" --request "$SEALED_G1_PI05_REQUEST"
```

On a provisioned local Isaac/LeRobot host, add `--execute --output-dir
"$NEW_G1_PAIR_DIR"`. On a Linux NVIDIA Docker host, also add `--mode container`
with `--source-receipt`, `--source-packet`, and `--policy-runtime-root` using the
same verified inputs described above. The wrapper executes the DP candidate
first, then π0.5 only if the first worker completes; it stops after an
infrastructure block to avoid a second unproductive GPU attempt. It keeps the
two worker receipts, scores, media trees, and any container logs under separate
candidate directories, plus a digest-bound pair receipt at the output root.
For each completed candidate the pair receipt indexes the exact head and
overview MP4 review files by relative path and SHA-256, bound to the episode
trace and frame-manifest digest. These videos are review conveniences; the
receipt does not authorize public redistribution or replace the retained
lossless policy inputs.
Scored task failure still counts as a completed development episode. Neither
this comparison nor a successful simulator score makes a candidate qualified
or available in the public Task Evaluation Run.
