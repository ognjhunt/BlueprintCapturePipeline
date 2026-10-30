# Public TRLC policy audit — 2026-10-30

Public TRLC-specific checkpoints exist. The precise current limitation is that
Blueprint has not installed and executed a rights-complete, byte-verified TRLC
policy integration. Neither “no checkpoint exists” nor “vendor access is always
required” describes the public evidence.

The best documented candidate found is the community model
[`djkesu/pi05_trlc_rabc_3cam_alpha025_20260319_v2_step10000_merged`](https://huggingface.co/djkesu/pi05_trlc_rabc_3cam_alpha025_20260319_v2_step10000_merged/tree/10466a133c74018465c6839b7003b700775e4f69),
pinned at `10466a133c74018465c6839b7003b700775e4f69`. Its public, ungated repository
contains a merged checkpoint, the policy configuration, processor definitions,
normalization tensors, training configuration and merge provenance. No affiliation
between this uploader and The Robot Learning Company was established.

This audit fetched only bounded public metadata, source and normalization files;
it downloaded no policy weights or gated tokenizer assets and performed no model
inference, GPU allocation, native task run or physical test. The companion
`trlc-public-policy-audit.evidence.json` retains the metadata identities and checks.

## Exact action and observation contract

The pinned [training configuration](https://huggingface.co/djkesu/pi05_trlc_rabc_3cam_alpha025_20260319_v2_step10000_merged/blob/10466a133c74018465c6839b7003b700775e4f69/train_config.json)
selects `Gongsta/trlc_tshirt_folding`, but does not pin its revision. The observed
public dataset revision is `2fb972f237989ff01ed59a818834fdf3e1537734`. Its
[`meta/info.json`](https://huggingface.co/datasets/Gongsta/trlc_tshirt_folding/blob/2fb972f237989ff01ed59a818834fdf3e1537734/meta/info.json)
identifies `bi_dk1_follower`, 30 Hz, 532 episodes, and the following named ordering:

| Vector indices, zero based | Meaning |
| --- | --- |
| State 0–5; action 0–5 | `left_joint_1.pos` through `left_joint_6.pos` |
| State 6; action 6 | `left_gripper.pos` |
| State 7–12; action 7–12 | `right_joint_1.pos` through `right_joint_6.pos` |
| State 13; action 13 | `right_gripper.pos` |
| Action 14–20 | Left end-effector `x,y,z,wx,wy,wz,gripper_pos` |
| Action 21–27 | Right end-effector `x,y,z,wx,wy,wz,gripper_pos` |

The 28 outputs are consequently two representations of the same two arms, not
four arms or 28 independent actuators. A joint-control adapter can explicitly
consume the first 14 **unnormalized** outputs and retain the unused EEF outputs
for diagnostics. It need not invent an EEF rotation convention or inverse
kinematics for those last 14 values.

The official [DK1 follower source](https://github.com/robot-learning-co/trlc-dk1/blob/12f5368aefd0381461f2c7ffbb5611b4e8c90de9/lerobot_robot_trlc_dk1/follower.py)
and [control API](https://github.com/robot-learning-co/trlc-dk1/blob/12f5368aefd0381461f2c7ffbb5611b4e8c90de9/trlc_dk1_control/robot.py)
at the **same source pin as Blueprint's body recipe** establish absolute arm
joint targets in radians and normalized gripper closure, `0=open`, `1=closed`.
The publisher dataset's exact collection fork/calibration is not recorded, so
identity between its hardware zero calibration and the public URDF remains an
explicit simulation assumption, not measured calibration evidence.

For Blueprint's corrected TRLC profile, the source-backed mapping is:

- Dataset `left_joint_N.pos` / `right_joint_N.pos` maps to native
  `left_jointN` / `right_jointN`, for N=1…6. Do not convert radians to degrees.
- Each closure maps through its two-finger channel. Current source configuration
  has both prismatic fingers at `0 m` when open and `−0.045 m` when closed:
  `finger_target = −0.045 × closure`. Pre-correction reversed profiles must not
  be reused.
- Native declared target ordering places all 12 arm joints before the two
  closure channels. Reorder the policy's interleaved prefix as
  `[a[0:6], a[7:13], a[6], a[13]]`; construct observed state in the inverse order.
- Preserve native position, velocity and closure bounds. Dataset gripper values
  occasionally exceed 1; do not silently reinterpret their quantile range as
  a new physical scale. Any command clipping must be explicit and reported.

The [policy configuration](https://huggingface.co/djkesu/pi05_trlc_rabc_3cam_alpha025_20260319_v2_step10000_merged/blob/10466a133c74018465c6839b7003b700775e4f69/config.json)
uses 50-row action chunks, 14-state values, and three RGB cameras: `top`
(600×800), `right_wrist` and `left_wrist` (540×960 each, CHW). The policy's model
image size is 224×224. Future integration must preserve its actual preprocessing
and camera roles; the public training metadata supplies no camera extrinsics or
intrinsics. Blueprint's simulated mounts remain declared approximations. The
dataset's 30 Hz cadence can be represented by holding commands for two ticks
of the existing 60 Hz native loop. This is a supported candidate implementation
choice, not proof of the publisher's deployment timing.

## Saved normalization is available and checked

The exact [preprocessor](https://huggingface.co/djkesu/pi05_trlc_rabc_3cam_alpha025_20260319_v2_step10000_merged/blob/10466a133c74018465c6839b7003b700775e4f69/policy_preprocessor.json)
and [postprocessor](https://huggingface.co/djkesu/pi05_trlc_rabc_3cam_alpha025_20260319_v2_step10000_merged/blob/10466a133c74018465c6839b7003b700775e4f69/policy_postprocessor.json)
select `QUANTILES` for state and action, identity for images, and epsilon `1e-8`.
Both saved processor tensor files are 9,920 bytes and have SHA-256
`be756f317fa7659062d7745ea831097ebf15c7689efd89a8d75fd8dd0ee4b740`.

The retained tensors contain the complete 14-state/28-action statistics.
For both features, all six checked arrays (`min`, `max`, `mean`, `std`, `q01`,
`q99`) equal the pinned dataset's [`meta/stats.json`](https://huggingface.co/datasets/Gongsta/trlc_tshirt_folding/blob/2fb972f237989ff01ed59a818834fdf3e1537734/meta/stats.json)
after float32 conversion: 12 comparisons, maximum absolute difference zero.
This supports the named dataset schema despite the unpinned training dataset
reference. An implementation should load these processor states exactly,
round-trip test the quantile transform, then select the joint prefix after
unnormalization. It must not substitute another dataset's statistics.

## Runtime source and rights still need their own closure

The [merge metadata](https://huggingface.co/djkesu/pi05_trlc_rabc_3cam_alpha025_20260319_v2_step10000_merged/blob/10466a133c74018465c6839b7003b700775e4f69/merge_metadata.json)
names base model `lerobot/pi05_base`, but supplies no base revision or source
commit. The training file records RABC and LoRA training settings, but no
reproducible LeRobot fork/commit or dependency lock. A public author log shows
this exact checkpoint being selected with `--action-mode joint_positions`,
which corroborates the prefix interpretation but is not execution proof:
[TRLC Logs](https://stevengong.co/notes/TRLC-Logs).
Public Gongsta LeRobot/OpenPI branches and indexed source searches did not
establish the exact `serve_trlc_openpi.py` / TRLC training fork. This does not
establish that such code is private. A future implementation can pin a compatible
public LeRobot runtime and verify checkpoint parameter names/shapes and strict
loading, instead of assuming a particular unseen fork.

Rights must be assessed across the complete dependency chain:

| Input | Pinned public evidence | Status established by this audit |
| --- | --- | --- |
| Community merged checkpoint | [Card at `10466a1…`](https://huggingface.co/djkesu/pi05_trlc_rabc_3cam_alpha025_20260319_v2_step10000_merged/blob/10466a133c74018465c6839b7003b700775e4f69/README.md) | Declares Apache-2.0; public and ungated |
| Named base `lerobot/pi05_base` | [Card at observed revision `22c30e5…`](https://huggingface.co/lerobot/pi05_base/blob/22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0/README.md) | Declares **Gemma**; current public base revision is not proven to be the training revision |
| Configured tokenizer `google/paligemma-3b-pt-224` | [Repository at observed revision `35e4f46…`](https://huggingface.co/google/paligemma-3b-pt-224/tree/35e4f46485b4d07967e7e9935bc3786aad50687c) | Declares **Gemma** and currently requires manual access approval; no gated files fetched |
| Training dataset | [Card at `2fb972f…`](https://huggingface.co/datasets/Gongsta/trlc_tshirt_folding/blob/2fb972f237989ff01ed59a818834fdf3e1537734/README.md) | Declares Apache-2.0 |

The community Apache-2.0 declaration does **not** clear inherited Gemma terms.
The existing G1 terms review covers its exact four policies and is not approval
for this checkpoint. A legitimately retained tokenizer may be reused only after
its rights, exact byte identity and compatibility are verified for this policy.
Otherwise the tokenizer's authorized access path is a concrete dependency.
No new implementation or registration is authorized by this audit alone.

The model file is **7,473,096,344 bytes**, with Hub LFS SHA-256
`1b9086d511dc534ce8ceb0a119912fbf27325cb44643b3dd6821d8135d3db2ad`.
That identity comes from public repository metadata; the model bytes have not
been downloaded or locally verified. A later bounded installation must verify
them, stage tokenizer/processor/runtime dependencies, and account for disk,
transfer, GPU memory and execution through the canonical allocator.

## Are the 14-action siblings cleaner?

Two public converted siblings have 14 outputs and one top camera:

| Candidate | Exact revision | Published action semantics | Rights evidence |
| --- | --- | --- | --- |
| [`djkesu/pi05_trlc_abs_a100x2_fast_v1`](https://huggingface.co/djkesu/pi05_trlc_abs_a100x2_fast_v1/tree/60a92af48f47b52a832746129a70fa55dc9871f7) | `60a92af48f47b52a832746129a70fa55dc9871f7` | Absolute EEF targets, not 14 joint targets | No license declaration found in its pinned card, metadata or file inventory |
| [`djkesu/pi05_trlc_delta_a100x2_fast_v1`](https://huggingface.co/djkesu/pi05_trlc_delta_a100x2_fast_v1/tree/f1617fc1ae3d0790b9c3e3655fc84a9bc0cafdc8) | `f1617fc1ae3d0790b9c3e3655fc84a9bc0cafdc8` | Delta EEF; exported mask changes six coordinates per arm, leaves each gripper absolute | No license declaration found in its pinned card, metadata or file inventory |

Both export metadata files name the same shirt-folding dataset and require
further EEF frame/rotation reconstruction evidence. They are not a cleaner
licensed joint-control substitute. Other discovered DAgger/mixed10/OpenPI
checkpoints remain shirt-folding variants with missing license declarations or
unresolved serving-source details. The discovered MolmoAct2 TRLC repository
contains only `.gitattributes` at its observed pin; its name is not evidence of
available model weights. This bounded search found no licensed 14-action
TRLC-specific rigid-task checkpoint; it does not prove none exists elsewhere.

## Recommended next step after rights/tokenizer verification

Prefer the merged, documented 28-action candidate with an explicit 14-value
joint-control projection. Complete the pinned inference runtime, strict load,
normalization round trips, named joint/closure mapping, three camera inputs,
50-row/30 Hz cadence, and bounded controller checks. Then run an individually
reported diagnostic against a supported task and scorer.

The dataset's sole task is “Fold the t-shirt.” That training scope does not make
the joint interface mechanically incompatible with rigid pick/place. A rigid
scene can be a legitimate **out-of-distribution diagnostic**, with no promise
of competence or transfer. Conversely, this audit does not provide cloth
physics, garment-state observations or a folding scorer; a rigid task result
must never be presented as cloth-folding evidence. Any excluded candidate still
needs its individual reason, such as rights/tokenizer incomplete, runtime not
installed, allocation budget unavailable, or actual execution failure.
