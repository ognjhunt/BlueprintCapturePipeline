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
