# MolmoAct2 DROID in ordinary beta screening

The existing `molmoact2_droid` catalog candidate now has a fixed local runtime
for `allenai/MolmoAct2-DROID` at
`d8c1abd8a27d8e859455bbe514df2bcc617db0fb`. It uses the existing sealed Panda
and Robotiq DROID station. This adds one public policy on that body; it does
not add another deployment-company relationship or change the 74 published
robot–route pairs. The original pi0.5, GR00T and Cosmos Edge configuration
digests remain unchanged.

For an ordinary prepared rigid task, the harness derives the exact scene and
camera binding, retains public-license evidence, and produces a candidate
capability. Existing task rights, aggregate site budget, candidate reservation,
canonical offer admission and watchdog still decide whether it can run.
Missing inputs, capacity refusal, budget deferral, provisioning failure and
individual policy failures retain a result for this candidate. There is no
operator body installation step for this existing DROID station, and no
compute is launched by catalog construction or bundle preparation.

## Publisher contract and native execution

`molmoact2_droid_joint8.v1` accepts two distinct current RGB images: external
and wrist. The ordinary DROID observation builder preserves aspect ratio and
produces 640×360 RGB uint8. These are the publisher-supported original DROID
views; the second exterior camera is optional for the publisher and is not
fabricated here. Existing station mounts and their site adjustments remain
explicit simulation configuration, not a claim of training-camera calibration.

State is seven Panda joint angles in radians followed by normalized gripper
closure (`0` open, `1` closed). The checkpoint returns 15 rows of eight
absolute values at 15 Hz. The initial harness executes eight rows before a
fresh observation and records the seven unused rows. That execution horizon
is a harness choice, not a publisher recommendation. Normalization occurs
once in the model API. Raw gripper closure is preserved, without the opening
inversion used by YAM. Existing DROID action bounds and measured Robotiq
command calibration remain authoritative.

Native preflight performs an exact local session/identity reset without model
inference. Each query retains the real wire request and response. Wrong-shaped
or nonfinite returned actions remain individual candidate outcomes with raw
response evidence. A connection failure receives no implicit retry, and a
failed session must be explicitly reset before reuse.

## Exact runtime and accounting

The model inventory binds 19 required files totaling 21,781,401,931 bytes:
five complete weight shards plus configuration, six custom Python files,
tokenizer, normalization and shard metadata. Full file closure and hashes are
verified before checkpoint Python can run. The model card's absent HF license
field is preserved. Rights evidence records the official Ai2 README's Apache
2.0 statement, its pinned hash, research/educational intended-use note and this
user's instruction to screen public policies; no private permission is invented.

All six custom Python file hashes match YAM. The exact 50-wheel inference set
and standalone CPython 3.12.12 distribution are reused byte for byte, with a
separate DROID dependency manifest, model identity and CPU audit. The audited
AutoConfig/AutoProcessor/model-class path imports five of those modules; the
publisher's convenience `inference.py` is retained and verified but unused by
the local server. The DROID CPU audit verifies two-camera preprocessing,
tokenization, 8D normalization and the real prediction API signature without
constructing a model, loading weights or connecting to a network.

Transfers stream to disk with exact per-file and aggregate bounds, full
SHA256 checks, no retries and the canonical absolute deadline. Dependency
installation is offline, uses hashes and performs no resolution or source
build. The standalone interpreter is isolated from Isaac's Python paths.
Inference uses only local checkpoint files and separate module/cache folders
with HF and Transformers offline; no HF token is forwarded.

The allocator quote counts the actual bundle, any external native runtime
packet, required checkpoint bytes, 4,045,566,007 dependency bytes, a 64 GiB
cold-image allowance and up to 8 GB of output. Existing 200 GB disk pricing,
independent watchdog, zero-retry allocation and server teardown apply.
The 48 GB GPU floor is a conservative policy-plus-simulator request based on
publisher memory guidance. It is not measured peak usage.

## Verification limits

Tests exercise automatic catalog selection through sealed archive validation
and the actual dry canonical transport, plus native and RoboLab episode loops
with fake physics and the real client/server adapter. Both loops consume
four 15-row model responses, execute 32 actions at 15 Hz, retain the two real
views and preserve normalized closure. Dependency, checkpoint, rights,
endpoint, transfer, startup and deadline contracts have focused tests.

Complete checkpoint weights were not downloaded during this work. Native
Isaac GPU execution, model inference on a GPU, measured VRAM and task success
remain unverified. The first admitted run must supply its own result; this
integration does not manufacture a success or physical-validation receipt.
