# OpenWAM Alpha single-FR3 simulation integration

The existing `openwam_alpha_franka` candidate now has a concrete EEF10 adapter,
local server, sealed artifact/bootstrap route and fixed registration operation.
A valid registration replaces that candidate's old unavailable row; it does not
add a duplicate Franka candidate. Without a retained imported robot registration,
the row remains unavailable with `prepared_robot_policy_registration_missing`.
No native checkpoint loading, GPU inference or site-task success is asserted.

The distinct body is `franka_research_3_robotiq_2f85_sim`: one official FR3 arm,
with the public PickNik Robotiq2F85 tool, six mechanically coupled gripper joints
and one commanded closure channel. It is separate from Panda DROID and dual FR3
with Franka Hand. The public paper identifies FR3 and a Robotiq parallel gripper,
but not the exact tool SKU, mounting transform or recorded EEF/TCP definition.
The source assembly therefore labels its zero-offset mount at `fr3_link8`,
nominal grasp frame at tool Z=0.134m with yaw π, source reset/controller settings,
and camera geometry as simulation assumptions. It claims no publisher hardware
or calibration match. This is a nineteenth public preparation variant, not an
additional deployment-company coverage claim.

## Exact public evidence and loader identity

- Model: [`OpenWAM/OpenWAM-Alpha-Real-Single-Arm-Franka`](https://huggingface.co/OpenWAM/OpenWAM-Alpha-Real-Single-Arm-Franka/tree/e75522eb38f9b4148d5ecb985ab80c490ddb3854), revision `e75522eb38f9b4148d5ecb985ab80c490ddb3854`.
- The pinned [model card](https://huggingface.co/OpenWAM/OpenWAM-Alpha-Real-Single-Arm-Franka/blob/e75522eb38f9b4148d5ecb985ab80c490ddb3854/README.md) explicitly declares Apache-2.0; 1,384 bytes, SHA256 `456ba724d1f54e86ea92a9349042d5b4548af59cce39845038fe6883e8812461`.
- Weight: `checkpoint_step_9860.safetensors`, 24,813,767,464 bytes, publisher LFS SHA256 `76e53e963525d9eebdb70e5285908c6497082270de66f9227874844e96bc43bb`.
- `config.yaml`: 3,407 bytes, SHA256 `704222ea8504bb9332b6492a83054dcdeaa9f7b810836eec6dc9436bb971adc2`.
- `normalization_stats.npy`: 1,368 bytes, SHA256 `d54b7b411f545de021a8ffd04f35637db6588d36bbcbd4463755271b91859d9a`.
- [Serving source](https://github.com/OpenWAM-Official/OpenWAM/tree/3950b6d9ea848dc980a82a89b4554af2c082544a) is the same pinned OpenWAM revision as PiPER. Source archive SHA256 is `58e3901bc1b15643828e17350fe27d0f4d2493a0b7076251d3d6b5354c980ab7`.
- The [historical training reader](https://github.com/OpenWAM-Official/OpenWAM/blob/29967d4bbd023e711ecaac4d201d09a52d695d3d/openwam/dataloader/muka_franka.py) and [dataset schema](https://huggingface.co/datasets/ewykric/muka_franka_lerobot_v3/blob/702524adccdd6c667420952d46d7590a28d8e81c/meta/muka_schema.json) establish achieved EEF targets, signed gripper convention, raw prompt and 15Hz sampling. The removed training reader is not required by the current generic deployment path.

`openwam_franka_assets.EXPECTED_FILES` binds exactly eight staged files: the
seven inference inputs plus the license-bearing card. It rejects unlisted
sidecars, extra checkpoint files, symlinks and extra directories, and hashes all
local bytes with the existing worker deadline before upstream loading. Receipts
stay outside the checkpoint directory. Checkpoint-wide verification remains a
worker action; knowing publisher hashes does not mean weights were downloaded.

The server reuses PiPER's pinned CPython3.12.12, Torch2.7.1+cu128/CUDA12.8 and
75-wheel dependency closure. The dependency manifest digest is
`sha256:92558c8efed216e0ccf103c2194ead8e7d521ee40b3d6ffd1200afbe5735b68d`.
The thin simulator client uses the same explicit websockets16 dependency.
Dependencies/source are staged before offline model loading; the bootstrap uses
only bounded fixed URLs, sizes and SHA256 values inside an already admitted
canonical worker. The 80GB combined GPU memory floor is a conservative allocation
setting inherited from PiPER, not measured peak VRAM. Download accounting includes
the exact checkpoint, dependencies, sealed bundle and cold-image allowance.

## Observation, frame and controller contract

`openwam_franka_eef10.v1` observes one named arm `arm`, seven native FR3 joints,
its measured TCP pose and one measured closure. The model sees ten physical
values: absolute base-frame xyz, the first two rotation-matrix columns
concatenated, and signed opening `1 - 2*closure` (−1 closed, +1 open). Output is an
absolute next achieved EEF target, not a joint command, velocity or delta pose.
The source rotation convention is extrinsic XYZ Euler radians, converted to
rotation columns by training. The adapter composes explicit full base/TCP
transforms and inverts them for the generic named-arm DLS controller.

The initial policy/native frame maps are explicitly labeled identity assumptions
between the publisher's unreported recorded EEF and the source nominal grasp TCP.
The real body and every transform are sealed into the candidate configuration.
Raw task instructions are sent unchanged; PiPER's video-description prefix is
not used. Only sensor state, two RGB images and the instruction cross transport;
privileged task targets or scorer state are not policy inputs.

Two actual 640×480 images map `overhead` to `head_camera` and `arm_wrist` to
`left_wrist_camera`. Missing wrist frames are errors; no synthetic camera is
substituted. The pinned configuration has a null third slot. Upstream builds its
384×320 composite with external image on top, wrist at bottom-left, and intentional
black structural padding at bottom-right. Camera poses and the source 90° simulated
optics are retained assumptions, not the paper's D455 calibration.

The exact checkpoint normalizer handles raw10 → normalized unified80 slots0–9
and inverse action mapping once. State normalization clips to recorded ranges:
full-open +1 is clipped to the recorded maximum approximately +0.75351274. The
adapter does not substitute dataset statistics or normalize a second time.
Diffusion outputs can exceed the gripper range; finite signed opening is projected
to [−1,1], then converted to closure `(1-opening)/2`. Raw response and projection
are retained in inference evidence. Degenerate rotations and nonfinite actions
remain genuine returned-policy failures.

Policy rate is the source dataset's 15Hz; native controller rate is 60Hz, with
four native ticks per EEF target. Upstream predicts32 rows; its synchronous
executor buffers an explicitly chosen first8, returning one row per query.
The native client requests/executes one row at a time. Per-episode reset clears
upstream temporal state, sequence and the client evidence state. A lost response
poisons that episode until reset; no inference retry silently consumes another
buffered action. Physics, contact and the independent task scorer remain in the
existing generic runtime.

## Protected installation

`install-public-franka-policy` accepts the existing exact public-policy request
fields described in [the PiPER installation contract](openwam-piper-registration-2026-10-29.md),
with these fixed values:

| Field | Value |
| --- | --- |
| `schema_version` | `openwam_franka_policy_install_request.v1` |
| `candidate_id` | `openwam_alpha_franka` |
| `team_id` | `blueprint-public-models` |
| `checkpoint_id` | `openwam-alpha-franka-e75522eb` |
| `policy_profile` | `openwam_alpha_real_franka.v1` |

Bind a real retained preparation publication, exact current release and registry
digest, and the user's actual request to run public simulation policies:

```sh
python3 scripts/operator_door.py install-public-franka-policy \
  --request-file openwam-franka-install.json --wait
```

It uses the existing authenticated `operate` scope and networkless transient.
The service user verifies retained source/import/model closure; the independently
installed stdlib root verifier reconstructs the exact profile, manifest, license
and scoped rights before the shared serialized compare/replace/audit operation.
It changes only this candidate in registry defaults and existing explicit
profiles. It accepts no arbitrary paths, URLs, commands, code or credentials and
starts no provider, simulator or task. Upgrading the release/door creates no
request. Exact retries retain their original installation identity and receipt.
Noncooperating administrator writers must not race the final registry comparison
and atomic replacement.

The body binding changes only native60Hz, one response row, and the two640×480
camera dimensions; source tool, mount, camera poses, optics and reset are retained.
Rights record the verified public Apache2 model card and user instruction, without
private vendor permission, self-grading, official ranking, ADP or physical claims.
Capture rights, task authority and the site screening budget still apply.

## Validation boundary

Focused tests cover signed-state/action/frame round trips, raw prompt, required
real views, checkpoint closure/sidecars, source/dependency/profile identity,
sequence/reset/raw response evidence, fixed bootstrap URLs and deadline accounting,
registration → existing catalog row → task planning → factory → sealed bundle →
canonical dry transport. The real native loop ran against explicitly fake Cartesian
physics: three queries, twelve native ticks, hold stride4, six coupled Robotiq
commands, and an independent false task score. It is not Isaac execution.

An offline CPU check passed the actual new adapter's wire data through the pinned
upstream preprocessor and normalizer, including two-view structural padding,
raw instruction and signed gripper conversion. Another check prepared the retained
public FR3/Robotiq sources and used a development converter producing real USD
structure through both actual registration stages and idempotent retry. Neither
loaded weights, called a provider, allocated compute or executed a native task.
Actual checkpoint loading, simulator interaction, memory/latency and task results
remain to be measured through the normal budget-authorized candidate run.
