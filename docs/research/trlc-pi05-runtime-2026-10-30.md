# Public community TRLC Pi0.5 runtime

This adds a distinct opt-in community checkpoint to the existing TRLC-DK1 body.
It is not the policy used by a deployment company. The catalog initially records
it as unavailable; the fixed operator registration consumes retained native body
and tokenizer evidence without starting compute. Screening still needs the
current task/source/rights binding, the site budget and canonical allocator.

## Exact inputs and control contract

- Checkpoint: [djkesu/pi05_trlc_rabc_3cam_alpha025_20260319_v2_step10000_merged](https://huggingface.co/djkesu/pi05_trlc_rabc_3cam_alpha025_20260319_v2_step10000_merged/tree/10466a133c74018465c6839b7003b700775e4f69).
  Six inference files total 7,473,120,997 bytes; the single weight file accounts
  for 7,473,096,344 bytes. Inventory hashes bind every file and reject sidecars.
- Inference implementation: the verified LeRobot subtree of
  [HumanoidArena 68479287a784a69be9ce6ad739311d2f11f75ef9](https://github.com/William-wAng618/HumanoidArena/tree/68479287a784a69be9ce6ad739311d2f11f75ef9/lerobot).
  This is a compatible inference implementation, not a claim about the training
  fork. The sealed 1,203,039-byte archive contains 363 files, with 4,099,808
  expanded bytes. No runtime Git clone or source lookup occurs.
- State is `[left six joints, left closure, right six joints, right closure]`.
  Joint units are radians; closure is zero open and one closed. The corrected
  prepared body maps closure to the two fingers on each arm, from 0 to -0.045m.
- The model returns 50 absolute 28-value rows. The saved postprocessor
  unnormalizes all 28 values once; native columns are
  `[0,1,2,3,4,5,7,8,9,10,11,12,6,13]`. Remaining EEF diagnostic values remain
  in the raw response/evidence. Only finite closure channels are saturated to
  `[0,1]`, with raw/applied values recorded; joint limits remain the native
  controller's responsibility.
- Input camera order is top, right wrist, left wrist. Top is 800×600; wrists
  are 960×540, RGB uint8. The host converts to float `[0,1]` once. Exact upstream
  preprocessing pads/resizes to 224×224 and applies SigLIP `[-1,1]` scaling.
- Policy cadence is 30Hz, native cadence 60Hz, execution horizon 50. The native
  loop holds each row for two ticks. Prepared camera mounts/optics and the
  dataset-to-URDF joint zero correspondence are explicit simulation assumptions.

The saved pre/postprocessor JSON and safetensors are loaded directly. Constructing
new processors from current defaults would add relative-action behavior absent
from this checkpoint. The strict loader does not use the upstream
`PI05Policy.from_pretrained` catch-and-return fallback. It requires the exact
812 saved names/shapes and adds only the tied embedding alias declared by the
safetensors header, then calls `load_state_dict(strict=True)`. Missing/extra keys
or load errors prevent a ready handshake.

## Terms and retained tokenizer

The community card declares Apache-2.0. Its Pi0.5/PaliGemma ancestry also carries
Gemma terms and use restrictions; the Apache label does not erase those terms.
The public terms permit ordinary internal simulator use within their conditions.
The bundle retains the exact community/base cards, Apache text, complete Gemma
terms and prohibited-use text, NOTICE and source provenance. See the companion
[registration note](trlc-pi05-registration-2026-10-30.md).

The six tokenizer files are already legitimately retained for the exact reviewed
PaliGemma revision `35e4f46485b4d07967e7e9935bc3786aad50687c`. The installer verifies
the fixed protected asset directory; no old G1 task authority is reused. Runtime
preparation honors the existing explicit directory override only when the same
six files verify. It copies those six files only and rejects any extra staged
sidecar. Missing/inaccessible/tampered files produce an individual unavailable
reason. No gated download, ambient Hugging Face credential or fallback tokenizer
is used. Current production tokenizer availability is not asserted by this work.

## Bounded runtime and evidence limits

The existing native Python3.12/Torch2.10+CUDA12.8 packet supplies exact large CUDA
wheels. The separate inference environment uses the existing immutable 96-wheel
inventory: 4,132,469,228 reused native wheel bytes and 261,777,389 externally
fetched bytes. Every wheel and expansion is verified before offline
`--no-index --no-deps --require-hashes` installation. No apt or resolver runs.
The provider's isolated Python startup uses `-I -B`, selects only verified code,
and enables hub offline flags. Model/tokenizer loading is local-only.

Transfer admission includes bundle and external native packet bytes, the exact
7.473GB checkpoint, exact external wheels, a conservative 64GiB cold-image
allowance and the existing 8GB output ceiling. A pre-allocation disk check counts
compressed/expanded native and scene inputs, checkpoint, wheels, the 16GiB
inference installation ceiling, source/tokenizer, image/output allowance and an
8GiB reserve against the existing 200GB provider disk. The 48GB GPU floor includes
unmeasured simulator headroom; it is a configuration assumption, not measured
peak VRAM. Native task deadlines, provider watchdog, teardown and no implicit
paid retry are unchanged.

The checkpoint was trained for community T-shirt folding. A current rigid-object
task is explicitly an out-of-distribution diagnostic. This integration does not
claim deformable-cloth task support, company endorsement, physical performance,
GPU inference success or native task success.

## Verification and next proof

Focused tests cover named-joint permutation, full28-to-physical14 mapping, closure
direction, malformed replies, strict loading, missing/altered tokenizer assets,
exact source extraction, registration/planning, a real extracted sealed bundle,
provider dry preflight, transfer/disk accounting, offline bootstrap flags and
startup failure isolation. A fake-physics run exercises the actual native team
loop with 102 ticks and two model requests; its scorer reports false and it is
not native/GPU evidence.

The [retained CPU receipt](evidence/trlc-pi05-cpu-adapter-2026-10-30.json) comes from
`scripts/audit_trlc_pi05_cpu.py` in the reviewed CPU dependency environment. It
verifies all small checkpoint files and the bounded weight header, extracts the
actual shipped source, constructs the 813-entry model on the meta device, checks
the exact tied alias, and executes the actual saved normalization/prompt/image
pipeline through the new backend. The tokenizer and model outputs are explicit
stubs; zero network attempts, no tensor weight download/load, and no inference
are claimed. Native proof still requires a verified prepared body and tokenizer,
canonical admission, strict full-weight loading, a real simulator episode,
independent scoring and scoped teardown receipts.
