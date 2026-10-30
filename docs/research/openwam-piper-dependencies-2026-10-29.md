# OpenWAM PiPER inference dependency lock — 2026-10-29

The pinned dependency closure installs offline and passes actual CPU imports,
checkpoint config resolution, tokenizer loading, normalization and observation
composition. No checkpoint weights were loaded and no CUDA inference occurred.

The target is OpenWAM source `3950b6d9ea848dc980a82a89b4554af2c082544a` and
`OpenWAM/OpenWAM-Alpha-Real-RoboDojo-Piper` checkpoint revision
`12bfca6fb289174ab5a0f4f7eb363dc77ed265f0`. The retained public source archive's
SHA-256 is `bd7daa2b9af724a8917e2a608462576bb25bfb44840b7c7071f8fc2f4813580d`.

## Concrete installation inventory

The packaged `_vendor/openwam-piper-dependencies.v1.json` records 75 exact
server wheels, one separate CPython 3.11 client wheel, and an exact CPython
3.12.12 standalone distribution. Every external input has a fixed URL, byte
size and SHA-256. Total unique external dependency downloads are
**4,119,355,954 bytes**, excluding model/source downloads and the bundled
143,402-byte ANTLR wheel. Expanded interpreter files measured 97,446,341 bytes;
the extraction cap is 128 MiB. The target is x86-64 Linux with glibc >=2.28.

Server pins follow the publisher's CUDA 12.8 lock: Torch 2.7.1+cu128 and
TorchVision 0.22.1+cu128. Python 3.12 is required by the pinned NumPy 2.5.3 and
SciPy 1.18.1 wheels. The server therefore owns its interpreter; Isaac retains
its existing Python 3.11 and receives only WebSockets 16.0. Headless OpenCV
4.14.0.94 replaces the same-version GUI wheel; no GUI functions are used.

The helper exports `DEPENDENCY_MANIFEST_FILENAME`, raw
`DEPENDENCY_MANIFEST_SHA256`, canonical `DEPENDENCY_MANIFEST_DIGEST`, and
`TOTAL_DOWNLOAD_BYTES`. Per-file hashes are plain 64-character hexadecimal.
`verified_dependency_manifest(path=None)` validates the fixed release bytes
using only the standard library. `stage_dependency_manifest(runtime)` copies
the manifest and verified local wheel into `runtime/openwam_dependencies/`.

After bounded downloads and full hash verification, install the listed wheel
paths using the pinned interpreter's pip with `--no-index --no-deps
--require-hashes --only-binary=:all:`. The standalone interpreter includes
bootstrap pip; pip 26.2.1 is also in the server inventory. Do not install the
OpenWAM project through its broad training dependency list: import the verified
source tree directly. Install the separate client wheel with Isaac's interpreter.

## Why these dependencies

The actual `openwam.deploy.server.build_server_from_config` path imports the
model registry and Wan components. `HuggingfaceTokenizer` uses Transformers,
SentencePiece and the locally retained tokenizer. Wan's shared loader eagerly
imports ModelScope and Hugging Face helpers even when the checkpoint is
self-contained. Those packages and their declared base dependencies are present;
runtime hub downloads remain disabled.

Observation preprocessing imports `openwam.dataloader.transforms`, whose package
initializers eagerly import RoboDojo, RoboTwin, EBench and LeRobot readers.
Consequently HDF5, OpenCV, Pandas and Arrow are necessary for imports even though
the inference server reads no training dataset. SciPy supports the shared
rotation transforms. Optional PyAV import failures are caught by the publisher;
video-file decoding is not used for PNG observations and PyAV is excluded.

DeepSpeed, W&B, Hydra, Accelerate, Diffusers, timm, torchaudio, imageio, flash
attention, SageAttention and xformers are absent. The selected Wan checkpoint
does not execute their optional paths. Supported attention settings are
`WAM_ATTENTION_IMPL=sdpa` and `DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch`.
Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and
`DIFFSYNTH_SKIP_DOWNLOAD=true` before importing the publisher.

## Executed validation

The isolated local environment used the exact standalone interpreter and the
same non-CUDA pins. Torch 2.7.1+cpu and TorchVision 0.22.1+cpu replaced the CUDA
pair solely for this check; NVIDIA and Triton wheels were omitted. Sixty files,
417,432,029 bytes in total, were downloaded or copied with full hash checks.
Offline pip installation succeeded and `pip check` reported no broken requirements.

`scripts/audit_openwam_piper_inference_imports.py` disables socket connections
and imports the real publisher server, loader, policy, synchronous executor,
component builder, DiT, VAE and T5 modules. It then reads the retained checkpoint's
small config/tokenizer/stats files. Results:

- 129 publisher source files imported; the receipt records each SHA-256.
- Real tokenizer output has shape `[1, 512]`.
- Real normalizer maps state `[1, 20]` to `[1, 80]` and action `[1, 80]` to
  physical `[1, 20]`, with finite values.
- Actual multiview preprocessing produces a 320×384 RGB canvas.
- Zero network connection attempts occurred.

Reproduce with the prepared CPU environment:

```bash
python scripts/audit_openwam_piper_inference_imports.py \
  --source /path/to/verified/OpenWAM \
  --checkpoint-metadata /path/to/retained/piper-small-files \
  --output /tmp/openwam-piper-cpu-audit.json
```

The retained [receipt](openwam-piper-cpu-import-audit-2026-10-29.json) explicitly
limits its claim to CPU imports, config, tokenizer and preprocessing. The model
was not constructed or queried. Production CUDA wheel loading, full model
loading, memory use and simulation behavior still require a bounded real run.

Eight focused tests validate transitive dependency versions, wheel interpreter
and platform tags, exact download accounting, staging, retained BSD license,
and rejection of changed manifests, changed bundled wheels and symlink targets.
