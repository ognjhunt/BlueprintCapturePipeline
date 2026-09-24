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

`native_g1_shared_scene_episode.py` can sequence either pinned HumanoidArena
box-manipulation candidate through the same Arena scene. It binds the prompt to
the scene task, uses the observed head camera and 64-value state for each policy
query, sends every returned action through the pinned SONIC target bridge, and
retains calibrated lossless head input and head/overview review PNGs plus a task
sample at each step. The shared media finalizer seals derived H.264 review
videos for both cameras, linked to the immutable frame manifest. Its result is
still an unscored development trace: the production worker must attest the
executing checkpoint and controller, score the bounded task, and retain a
terminal episode receipt before either candidate is offered as runnable.
Movement policies require their own verified action/controller binding; the
box-manipulation candidates do not prove movement-policy support.

`run_g1_built_scene_policy_episode` now connects that loop to the existing
native G1 joint environment, live rigid-task scene readback, and shared task
scorer. It rechecks the offline scene/model/source preflight, retains the
reset sample before the first policy query, then writes a trace and a scored
development receipt after the episode. A scored failure is still a valid
observed outcome. The receipt keeps `ranking_eligible` and policy-runtime
identity false: the qualified worker has not yet bound the running server to
the checkpoint or admitted G1 in the published bundle/run path. Navigation
remains outside this box-task scorer.

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
navigation task in Blueprint's captured site. A site-grounded movement task
contract, checkpoint/server attestation, and live episode evidence are still
required before offering them in the run configurator.

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
not make a movement candidate runnable: the shared episode loop currently
accepts only the box-manipulation candidates, and navigation still needs a
site-grounded task/goal and score contract. The π0.5 upstream configs also
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
box candidate, checkpoint inventory/root, pinned server and SONIC sources,
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
bundle, container launch profile, live checkpoint inference, movement task,
and public video approval remain separate gates.
