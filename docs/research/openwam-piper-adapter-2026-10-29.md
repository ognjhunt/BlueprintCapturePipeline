# OpenWAM real PiPER adapter: pinned public baseline

The public **OpenWAM-Alpha-Real-RoboDojo-Piper** checkpoint supplies a concrete
learned-policy path for our `agilex_piper_bimanual` body. That body is associated
with Dream Labs in the deployment-company catalog; this is an independent
OpenWAM baseline, not Dream Labs' policy. The adapter in this change does not
register a runnable candidate, launch compute, or claim native/GPU execution.

## Release and license evidence

| Item | Retained identity / evidence |
| --- | --- |
| Checkpoint | [`OpenWAM/OpenWAM-Alpha-Real-RoboDojo-Piper`](https://huggingface.co/OpenWAM/OpenWAM-Alpha-Real-RoboDojo-Piper/tree/12bfca6fb289174ab5a0f4f7eb363dc77ed265f0), revision `12bfca6fb289174ab5a0f4f7eb363dc77ed265f0` |
| Weight license | The [pinned model card](https://huggingface.co/OpenWAM/OpenWAM-Alpha-Real-RoboDojo-Piper/blob/12bfca6fb289174ab5a0f4f7eb363dc77ed265f0/README.md) declares `license: apache-2.0`; the repository was public and ungated when inspected on 2026-10-29. The card's exact bytes are included in the required deployment inventory. |
| Source | [`OpenWAM-Official/OpenWAM`](https://github.com/OpenWAM-Official/OpenWAM/tree/3950b6d9ea848dc980a82a89b4554af2c082544a), commit `3950b6d9ea848dc980a82a89b4554af2c082544a`; [Apache-2.0 LICENSE](https://github.com/OpenWAM-Official/OpenWAM/blob/3950b6d9ea848dc980a82a89b4554af2c082544a/LICENSE) |
| Weights | `checkpoint_step_30000.safetensors`, 24,813,767,464 bytes; publisher LFS SHA256 `ba9b2696624b12676610ac7bdc18882d088f819d9993b37d05d5d13f6466f004` |
| Other deploy inputs | Pinned `config.yaml`, `normalization_stats.npy`, model card, and four UMT5 tokenizer files; exact sizes/hashes in `openwam_piper_assets.EXPECTED_FILES` |

Only the small deployment inputs and safetensors header were fetched during this
investigation. The weight hash above is publisher metadata, not a claim that we
downloaded and verified all weights. `verify_checkpoint_directory` streams and
checks every required file before NumPy deserialization or model loading. It
refuses ambiguous additional numbered checkpoints. The standalone source path
requires the exact clean upstream Git checkout.

The safetensors header contains 2,089 BF16 tensors, including the video backbone,
action network, UMT5 text encoder and VAE. The publisher's checkpoint builder
constructs these from the retained configuration: it does not require a second
Wan/UMT5/VAE weight download. File size is **not measured peak VRAM**. No minimum
inference GPU memory or native runtime performance has been established here.

## Observation, action and simulator contract

Sources: the [checkpoint configuration](https://huggingface.co/OpenWAM/OpenWAM-Alpha-Real-RoboDojo-Piper/blob/12bfca6fb289174ab5a0f4f7eb363dc77ed265f0/config.yaml),
[RoboDojo real frame contract](https://github.com/OpenWAM-Official/OpenWAM/blob/3950b6d9ea848dc980a82a89b4554af2c082544a/openwam/dataloader/robodojo_contract.py),
[pose conversions](https://github.com/OpenWAM-Official/OpenWAM/blob/3950b6d9ea848dc980a82a89b4554af2c082544a/openwam/dataloader/utils/poses.py),
[observation preprocessor](https://github.com/OpenWAM-Official/OpenWAM/blob/3950b6d9ea848dc980a82a89b4554af2c082544a/openwam/deploy/obs_preprocess.py), and
[sync executor](https://github.com/OpenWAM-Official/OpenWAM/blob/3950b6d9ea848dc980a82a89b4554af2c082544a/openwam/deploy/executors/sync_executor.py).

| Boundary | Exact behavior |
| --- | --- |
| Cameras | Three declared RGB views, ordered head / left wrist / right wrist. The thin client sends source-sized PNGs. Upstream stretches/composes a 384-high × 320-wide canvas: head 256×320 above two 128×160 wrist views. No task/scorer channels enter the payload. |
| Prompt | Client prepends the publisher's training prefix: `A video recorded from a robot's point of view executing the following instruction: `. The server forwards it verbatim. |
| State | Physical 20-vector: left `xyz[3], rotation_columns[6], gripper_opening[1]`, then right. Position is metres in each arm's base frame. Rotation is the first two matrix **columns**, not rows. Opening is 0 closed, 1 open. The simulator's normalized closure is inverted once. |
| Frame binding | Explicit per-arm `T_policy_base_native_base` and `T_native_tcp_policy_tcp`. Observation uses `T_policy_base_native_base × T_native_base_native_tcp × T_native_tcp_policy_tcp`; commands use the inverse mapping. Nonidentity base and TCP transforms are tested. |
| Model space | Upstream alone applies checkpoint min-max normalization and the 20→80 mapping (indices 0–9 and 34–43), then gathers/unnormalizes output once. The thin client never repeats it. |
| Actions | Physical EEF20 absolute poses. The checkpoint predicts 32 rows; the sync executor retains the chosen execution horizon (adapter default 8) and returns **one row per query** with a 1-based step. Reset clears buffer and step. |
| Gripper saturation | Upstream min-max inverse is affine and unbounded; this checkpoint declares no binary action dimensions. Finite predicted openings saturate to [0,1] at the native tool boundary. Exact raw response plus per-arm raw/applied/clipped values remain in inference evidence. Nonfinite vectors and degenerate rotations remain malformed policy outputs. |
| Native command | Existing `blueprint.named_arm_pose_chunk.v1`, one row, two arm-base poses and two independent gripper closures. `run_team_episode` already routes this through native Jacobian pose control and existing tool actuators. No DROID padding or joint-action reinterpretation is needed. |
| Clock | `control_hz` is explicit and recorded. The checkpoint's video stride 4 is not a control-rate specification. A chosen simulation rate must be labeled as an assumption and compatible with the existing native hold-step timing. |

The [publisher normalizer](https://github.com/OpenWAM-Official/OpenWAM/blob/3950b6d9ea848dc980a82a89b4554af2c082544a/openwam/dataloader/transforms/normalize.py)
clips training/proprio inputs but does not clip its inverse. The
[policy facade](https://github.com/OpenWAM-Official/OpenWAM/blob/3950b6d9ea848dc980a82a89b4554af2c082544a/openwam/deploy/policy.py)
only projects dimensions explicitly designated binary, which this release does
not declare. Gripper saturation is therefore our explicit actuator projection,
not a claimed upstream guarantee.

## Endpoint evidence and supported simulation assumptions

Our PiPER asset derives from Agilex-College commit
`c2688be41e1bc99a9addd555f237a16bc839d936`. Its fixed `gripper_base_joint`
places `gripper_base` at `link6` with zero translation and rotation. The
[official SDK forward kinematics](https://github.com/agilexrobotics/piper_sdk/blob/b847e5393a7ce6ff8508d713eca7bdd3a42421b4/piper_sdk/kinematics/piper_fk.py)
returns the final link6 pose; it also exposes a DH-offset option. The retained
OpenWAM real-data contract calls the endpoint `recorded_ee_pose`, with per-arm
base coordinates, and does not establish the original recording's firmware/TCP
offset selection.

Identity between recorded EE and the current `gripper_base` is a supported
**simulation assumption** for an initial diagnostic candidate, grounded in the
terminal-link geometry. It is not publisher calibration. Retain that assumption
with the configuration digest, camera poses, and selected control cadence so
its effect can be evaluated. This uncertainty is not a requirement to acquire
private access. Do not import the unrelated dual-X5 simulator's base transforms
or convert SDK millimetres again: checkpoint EEF coordinates are already SI.

## Runtime and evidence behavior

The client uses NumPy, Pillow and a separate `websockets==16.0` requirement.
Version 16.x is checked before connecting because the no-proxy connection
argument is unavailable in older project-wide WebSocket clients. Only loopback
addresses are accepted. A connection owns one temporal session; competing
clients are rejected, disconnect resets the backend, and ambiguous transport or
response failures require an explicit episode reset rather than replaying a
potentially consumed action.

The policy process separately loads the pinned upstream code and verified local
checkpoint, with Hugging Face and Transformers offline. It calls the upstream
builder with CUDA, sync inference, 10 denoising steps, no video decode, no
compiler or DiT cache, and bounded prompt caching. Upstream's dependency ranges
include Python ≥3.10, Torch ≥2.5,<3, Transformers ≥5.5,<6, Diffusers ≥0.38,<1,
WebSockets ≥16,<17, plus torchvision, OmegaConf, Hydra, safetensors, Pillow,
NumPy, einops, sentencepiece, protobuf, scipy and its other runtime imports.
These ranges still need a resolved, sealed worker environment before allocation;
the adapter does not claim an installable production image already exists.

Every inference persists the exact allowed request through the existing request
evidence sink before transmission. Completed-response truth is retained even
when response JSON or actions are malformed. Raw responses precede frame or
gripper projection. Current evidence explicitly reports
`native_execution_validated=false` and `catalog_registered=false`.

## Remaining production integration, in dependency order

| Owner seam | Concrete next work |
| --- | --- |
| Prepared body and candidate binding | Resolve the retained PiPER native USD/configuration receipt, declare three real simulator cameras, and bind both arm bases, gripper TCP assumptions, independent tools and clock to that configuration digest. |
| `policy_interface_binding.py`, `team_candidate_policy_client.py` | Add the explicit local OpenWAM named-pose interface and factory; instantiate this client only from the sealed binding. It currently cannot be selected through the production factory. |
| `beta_policy_screening_team.py`, capability preparation | Build a local checkpoint-backed policy spec and registration for exactly `agilex_piper_bimanual`; carry named-pose/one-row semantics, public-license evidence and scoped team identity. Preserve an individual outcome or reason for every candidate. |
| Source and model-volume preparation | Retain a complete pinned source archive with a verified inventory for workers without Git metadata. Download the eight exact checkpoint files once within authorized preparation; fully hash them, retain inventory receipts and reuse the immutable model cache. |
| Runtime/bootstrap | Resolve exact CUDA/Torch/OpenWAM dependencies in a separate policy environment and stage the WebSockets16 client dependency independently of Isaac. No credential is required for this public release. |
| Sealed bundles | Extend `native_task_arena_policy_canary_bundle.py` and `policy_canary_session_bundle_identity.py` local-policy allowlists and required file closure; include adapter modules, exact binding, source/checkpoint inventory, runtime environment and the generic team's USD dependency closure. |
| Operator registration | Add a fixed authenticated installation operation that resolves retained prepared PiPER assets and this fixed policy profile, checks expected registry digest, and records public Apache-2.0 license/authorization provenance. It must not accept arbitrary code/paths/URLs or launch compute. |
| Budgeted validation | Use the normal allocator for actual model load/memory/latency and native camera/controller checks, then an authorized prepared task. Record allocation or budget deferral separately from policy failure. The checkpoint's size does not justify inventing a VRAM requirement or successful episode. |

## Other deployment-company leads

The catalog's 14 public-source-covered bodies do not imply 14 matching learned
policies. The strongest additional leads remain distinct:

- Dream-associated YAM has an exact public dual-YAM checkpoint,
  [`RLinf/RLinf-Pi05-RealWorld-DualYAM-Clean_Desk`](https://huggingface.co/RLinf/RLinf-Pi05-RealWorld-DualYAM-Clean_Desk/tree/15a037b444fa36618a56729fd55a69e09c9983f5),
  with matching 14D joint/gripper and three-view contracts. Its inspected card
  and inventory establish no weight license. RPent's Apache code license is not
  a substitute; weight rights remain unresolved.
- [OpenWAM single-arm Franka](https://huggingface.co/OpenWAM/OpenWAM-Alpha-Real-Single-Arm-Franka/tree/e75522eb38f9b4148d5ecb985ab80c490ddb3854)
  declares Apache-2.0, but is a single-arm 10D EEF release. It cannot be relabeled
  as the catalog's bimanual FR3 policy without a distinct body/controller setup.
- Extend Robotics' training workflows and TRLC's LeRobot hardware integration
  establish frameworks, not released company task checkpoints. Almond's Pi
  access, Jacobi's planner service, GrayMatter's GMR-AI and Unchained's LUNA
  remain separate account or proprietary-policy paths; public robot geometry
  does not establish checkpoint/API access.
- `DREAM_API_KEY` was absent in a boolean-only process check after the managed
  environment restart on 2026-10-29. No key value was read or printed and no
  provider inference was attempted.

## Local validation

Focused tests exercise source layout and nonidentity transforms, exact request
capture, completed/malformed response evidence, finite gripper overshoot,
checkpoint tamper rejection, builder settings, reset and sequence handling,
actual WebSockets16 loopback session isolation, and the real `run_team_episode`
generator with an explicitly fake Cartesian robot/backend. These are
`development_only` contract tests, not native physics or learned-inference proof.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q \
  tests/test_openwam_piper_policy.py tests/test_team_arm_kinematics.py \
  tests/test_policy_request_evidence.py
```
