# MolmoAct2 bimanual YAM public integration

The publisher provides a fine-tuned, directly loadable YAM checkpoint and an
explicit Apache 2.0 model license statement. This change adds an opt-in local
adapter and host; it does not register or enable a candidate. CPU contract tests
are development evidence. No weights were downloaded, model inference performed,
GPU allocated, task scene changed, or native YAM performance measured.

## Exact sources and license

- Model: [`allenai/MolmoAct2-BimanualYAM`](https://huggingface.co/allenai/MolmoAct2-BimanualYAM/tree/8dcbed66f2380e4393189c303ea72488eb9e63c2), revision `8dcbed66f2380e4393189c303ea72488eb9e63c2`.
- Official code: [`allenai/molmoact2`](https://github.com/allenai/molmoact2/tree/6070080a20321b4f498ab30f28e1d09ac465edb7), revision `6070080a20321b4f498ab30f28e1d09ac465edb7`.
- The [pinned README license section](https://github.com/allenai/molmoact2/blob/6070080a20321b4f498ab30f28e1d09ac465edb7/README.md#7-license) states: **“This model is licensed under Apache 2.0.”** The same README lists the exact BimanualYAM checkpoint for inference. It describes research/educational intended use and links Ai2's Responsible Use Guidelines. The repository also contains the Apache 2.0 license text.
- The HF model card has no license metadata field and no separate LICENSE file. Retain the explicit publisher model statement and its pinned hash; do not misrepresent HF metadata as populated.
- This is distinct from RLinf's dual-YAM clean-desk checkpoint, whose weight license remains unresolved. The Molmo statement supplies no rights to RLinf weights or to third-party simulation assets.

The packaged `molmoact2_yam_release.v1.json` records all 25 checkpoint files,
source references, license evidence and hashes. Total repository size is
**21,782,347,229 bytes**; five weight shards total **21,768,955,912 bytes**.
Fourteen small local files were verified against their HF git blob IDs and hashed
with SHA256. Weight and tokenizer digests are publisher LFS declarations, not
claims that these bytes were downloaded or verified. The shard index references
exactly the five listed shards. `verify_checkpoint_directory` verifies all 20
required local inference files, including full weight bytes and custom Python,
before the loader executes checkpoint code. It refuses every unlisted file,
including tokenizer/config sidecars, and verifies any supplied optional publisher
files too. Runtime caches and other ephemeral state belong outside the snapshot.

## Publisher API and exact robot contract

The [official YAM host](https://github.com/allenai/molmoact2/blob/6070080a20321b4f498ab30f28e1d09ac465edb7/examples/yam/host_server_yam.py)
uses `POST /act` with JSON-NumPy arrays: `top_cam`, `left_cam`, `right_cam`
(uint8 RGB), `instruction`, and a float32 `state[14]`. Optional `num_steps`
is the flow solver count, not the action horizon. The response is float32
`actions[N,14]` and `dt_ms`. Its health endpoint reports model ID and dimensions
but does not verify exact checkpoint bytes or isolate episodes.

The [checkpoint normalization file](https://huggingface.co/allenai/MolmoAct2-BimanualYAM/blob/8dcbed66f2380e4393189c303ea72488eb9e63c2/norm_stats.json)
defines `norm_tag=yam_dual_molmoact2` and the following order for both state and
actions:

| Indices | Physical meaning | Native binding |
| --- | --- | --- |
| 0–5 | Left arm absolute joint angles, radians | `left_joint1` … `left_joint6` |
| 6 | Left gripper opening: 0 closed, 1 open | `left_gripper` |
| 7–12 | Right arm absolute joint angles, radians | `right_joint1` … `right_joint6` |
| 13 | Right gripper opening: 0 closed, 1 open | `right_gripper` |

State and actions use q01/q99 normalization for arm channels only. The gripper
mask is false on channels 6 and 13. The model normalizes input state, clips the
normalized range, and unnormalizes model output exactly once. The adapter passes
raw radians and opening fractions and does not apply normalization again, add
the current state, or convert angles to deltas. The pinned q01/q99 output ranges
fit all twelve current source arm limits; native bounds remain authoritative.

The model's internal action width is 32; the supported prediction API returns
the 14 channels for this normalization tag. Prediction horizon is **30 rows**;
continuous mode uses **10 solver steps**, `n_obs_steps=1`, and depth reasoning
disabled. The publisher physical example runs at **30 Hz** and executes the
first **25** rows before observing again. Its ManiSkill client can execute all
30. The binding explicitly chooses 1–30 rows, defaulting to that 25-row physical
example. This is unrelated to RLinf's five-row execution window.

## Current I2RT body, tools and camera assumptions

Blueprint's `i2rt_yam_bimanual` recipe pins I2RT revision
`120c3c81400171174604e503943f8d1ebc891058`, source
`i2rt/robot_models/station/yam_station_linear_4310_d405/yam_station_linear_4310_d405.urdf`.
It supplies two six-joint YAM arms and LINEAR_4310 grippers. The
[publisher physical wrapper](https://github.com/allenai/molmoact2/blob/6070080a20321b4f498ab30f28e1d09ac465edb7/examples/yam/gello_min/yam.py)
selects that same gripper type. This is a concrete body/tool match, not proof of
site calibration or controller performance.

Native arrays are gathered and scattered by exact joint/tool names. The runtime
already converts normalized closure fractions into each tool's configured
open/closed finger positions. Current I2RT fingers `left_joint7/8` and
`right_joint7/8` travel **+0.04695 m** when open. The publisher's
[ManiSkill adapter](https://github.com/allenai/molmoact2/blob/6070080a20321b4f498ab30f28e1d09ac465edb7/sim_eval/inference/common.py)
instead uses interleaved arm arrays and **−0.0475 m** finger travel. Those indices,
signs and ManiSkill's normalized gripper actions must not be copied to Isaac.
The adapter independently maps each output opening to `1 - opening`, retaining
raw values and any finite saturation to the physical opening interval.

The [publisher simulation camera reference](https://github.com/allenai/molmoact2/blob/6070080a20321b4f498ab30f28e1d09ac465edb7/sim_eval/robots/bimanual_yam.py)
uses **640×360**, overhead **D435i / 69.4° horizontal FOV**, and two wrist
**D405 / 87°** views. Current I2RT CAD has an overhead D405; its generated generic
camera defaults are 320×176/90°. `publisher_camera_reference()` preserves the
publisher poses, mount frames, FOVs and Sapien wxyz convention. Applying those
poses to Isaac requires a declared frame conversion for the current station;
they are not measured training or site extrinsics. Use simulated D435i overhead
optics as an explicit supported assumption, retain the CAD discrepancy, and
bind the resulting camera configuration digest. No public camera source requires
vendor permission. The adapter requires the declared 640×360 three-view inputs.

## Implementation and remaining integration

`molmoact2_yam_policy.py` supplies `molmoact2_yam_joint14.v1`, preserving scientific
request bytes and raw responses. Its loopback transport has finite timeouts,
no implicit inference retry, exact identity checks, episode IDs and monotonic
sequence numbers. Failure requires an explicit episode reset. The local host
adds reset/identity metadata around the publisher API; this client therefore
expects the Blueprint wrapper, not an unchanged publisher `/act` process.

`molmoact2_yam_policy_server.py` verifies a staged local snapshot and loads it
with `AutoProcessor` and `AutoModelForImageTextToText`, `trust_remote_code=True`
and `local_files_only=True`. FP32 with CUDA graphs disabled follows the public
API without the publisher example's mutable BF16 source patches. No training is
needed to attempt inference. Publisher memory figures are approximately 24 GB
for FP32 without CUDA graphs and 26 GB with them; these are not measurements in
Blueprint. Exact dependency compatibility and allocated native execution remain
to be tested. The official server pins Transformers 4.57 while checkpoint
metadata names 5.3; do not silently call the dependency environment validated.

Next actions, using the existing local-team policy framework:

1. Register this fixed local interface in `TEAM_LOCAL_INTERFACES` with the exact
   YAM manifest, checkpoint and retained publisher rights evidence. Keep candidate
   admission separate from adapter tests.
2. Stage the exact model inventory and isolated inference dependencies through
   the existing `fetch_exact` / transfer budget / model-volume allocator path.
   No API key or paid hosted-policy account is required.
3. Bind the policy-specific three-camera simulation rig, 30 Hz action cadence,
   execution horizon and independent tools to the current body/configuration
   receipt. Reuse current team runtime resets and bounds handling.
4. Run a bounded allocated model-load and native controller/camera check, then
   evaluate the same prepared task and scorer as other policies. Retain each
   candidate's result, budget deferral or execution failure individually.

This patch does not duplicate provisioning, change the candidate catalog, or
turn CPU tests into evidence of a runnable model or successful simulated task.
