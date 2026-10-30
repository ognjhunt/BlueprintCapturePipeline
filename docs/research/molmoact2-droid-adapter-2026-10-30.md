# MolmoAct2 DROID local adapter — 2026-10-30

This adds a thin, opt-in local client/server profile for the public DROID
checkpoint. It does not enable a candidate or supply a provisioning, native
execution, or task-success receipt. No model weights were downloaded and no GPU
was allocated during this work.

## Exact checkpoint, source and rights evidence

- Model: [allenai/MolmoAct2-DROID](https://huggingface.co/allenai/MolmoAct2-DROID/tree/d8c1abd8a27d8e859455bbe514df2bcc617db0fb), revision `d8c1abd8a27d8e859455bbe514df2bcc617db0fb`.
- Publisher code: [allenai/molmoact2](https://github.com/allenai/molmoact2/tree/6070080a20321b4f498ab30f28e1d09ac465edb7), revision `6070080a20321b4f498ab30f28e1d09ac465edb7`.
- The pinned [README license section](https://github.com/allenai/molmoact2/blob/6070080a20321b4f498ab30f28e1d09ac465edb7/README.md#7-license), which explicitly lists this DROID checkpoint, says: “This model is licensed under Apache 2.0. It is intended for research and educational use in accordance with Ai2's Responsible Use Guidelines.”
- The HF card does not populate a license metadata field or include a separate
  LICENSE file. The inventory retains the publisher's explicit model statement,
  URL and SHA256. This does not grant rights to unrelated models or robot assets.

The exact release inventory records 24 files / **21,781,603,004 bytes**; five
weight shards total **21,768,955,912 bytes**. Nineteen files are required by this
continuous-inference path. Fourteen non-weight inputs, including the tokenizer,
were verified against HF git blob IDs or publisher LFS SHA256. Weight hashes
remain publisher declarations until the production loader reads their bytes.
The loader verifies the full permitted file closure and refuses extra tokenizer
sidecars, Python, symlinks, changed files and expired verification deadlines.

## Robot and policy contract

`molmoact2_droid_joint8.v1` uses the existing
`droid_franka_panda_robotiq_2f85_v1` simulator profile. The publisher recommends
the original [DROID implementation](https://github.com/droid-dataset/droid/tree/33ae6a67274f36d2e29525b86f23a56616ef43a7), which includes Panda and FR3 configurations.
Its [RobotEnv](https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/robot_env.py)
uses **15 Hz** and the familiar DROID reset posture. Its
[gripper interface](https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/franka/robot.py)
defines closure as `1 - width / max_width`: **0 open, 1 closed**.

- State: seven raw absolute joint angles followed by one normalized closure.
- Output: **15×8**, seven absolute joint targets plus raw closure. No addition
  of current state, delta conversion, second normalization or YAM opening
  inversion occurs. Native joint limits and measured gripper mapping still apply.
- Norm tag: `franka_droid`; norm SHA256
  `9cc29b5018604be4b785dd6519be3c86d0420675da7364755c58155afccb131d`.
  The model applies q01/q99 normalization once; the gripper mask is false.
- Prediction horizon: 15. Execution defaults to the first **8** rows, an explicit
  harness configuration that can be set from 1 through 15. Eight is not a
  publisher recommendation. All predicted rows remain available as evidence.
- Continuous action mode, 10 solver steps, one observation frame, depth reasoning
  disabled, language normalization enabled, FP32 and CUDA graphs disabled.

The profile accepts two **real 640×360 RGB** views in exterior-then-wrist order.
The [exact model card](https://huggingface.co/allenai/MolmoAct2-DROID/blob/d8c1abd8a27d8e859455bbe514df2bcc617db0fb/README.md)
explicitly supports these two views. Training used exterior 1, exterior 2, wrist;
the card also describes three-view and duplicated-exterior inference. This
adapter uses neither a fabricated second view nor a different camera ordering.
The existing scene's camera configuration digest and its calibration/assumptions
must be bound by the runtime. Camera dimensions are a declared simulation input,
not proof of measured site calibration.

The publisher's separate ManiSkill demo uses FR3, raw gripper-knuckle values,
and 30 Hz. Those example-specific mappings are not copied into the existing
DROID training-contract path. A future reproduction of that demo would be a
different configuration, with its own declared robot and controller.

## Memory, dependencies and executed CPU checks

The exact HF card reports FP32 approximately **24 GB without CUDA graphs** and
26 GB with graphs, and BF16 under 16 GB. The repository README still says
88/96 GB for an older DROID path. The exact checkpoint card is the applicable
current statement. These are publisher figures, not Blueprint measurements.
A 48 GB allocation floor can conservatively allow simulator headroom; the
actual combined peak remains unmeasured. This adapter uses the unpatched FP32
path; it does not edit a mutable Hugging Face cache to enable BF16.

All six DROID checkpoint Python files are byte-identical to the audited YAM
release. The processor, tokenizer and shard-index bytes also match. The same
CPython 3.12.12 / Torch 2.11.0 / Transformers 4.57.6 dependency closure is
reusable; checkpoint identity, config, normalization and inventory remain
distinct. Provisioning must bind the DROID model and its exact inventory,
rather than substitute a YAM checkpoint identity.

The committed [CPU receipt](molmoact2-droid-cpu-import-audit-2026-10-30.json)
uses the existing pinned CPU-only audit environment. It verifies the 14 small
files, loads the actual config/processor and five runtime-imported custom
modules, validates the `predict_action` call signature, and exercises tokenizer,
prompt construction, image processing and state/action normalizers. Two views
produce `pixel_values[2,729,588]`; state and actions are `[8]` and `[15,8]`, with
the closure channel preserved. A separate three-view audit also passed. Both
blocked socket connections and recorded zero connection attempts.

Reproduce without weights or GPU:

```bash
PYTHONPATH=src python scripts/audit_molmoact2_droid_inference_imports.py \
  --checkpoint-metadata /path/to/verified/non-weight-inputs \
  --cache /tmp/molmoact2-droid-cache --output /tmp/molmoact2-droid-cpu.json
```

Transport tests cover exact scientific request bytes, real loopback HTTP,
checkpoint identity, episode reset/sequence isolation, no implicit inference
retry, failure invalidation, finite shapes and complete checkpoint verification.
The wrapper's `/reset` metadata makes it distinct from the publisher's unchanged
`/act` server. The next runtime integration owns sealed provisioning, allocator
admission, candidate routing and actual same-task simulation evidence.
