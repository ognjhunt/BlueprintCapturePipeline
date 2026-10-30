# MolmoAct2 bimanual YAM inference dependencies — 2026-10-29

The exact publisher checkpoint code loads and preprocesses on the official
Transformers 4.57 server range. The isolated CPU audit does not construct the
model, load checkpoint weights, query a policy or measure GPU memory.

## Fixed inventory

`_vendor/molmoact2-yam-dependencies.v1.json` contains 50 inference wheels and an
exact standalone CPython 3.12.12 interpreter. Total unique external dependency
downloads are **4,045,566,007 bytes**. Checkpoint inputs are separately budgeted
by the model inventory; they are not included in this dependency total. No
source packages are compiled and no local repackaged wheel is needed.

The publisher's [pyproject.toml](https://github.com/allenai/molmoact2/blob/6070080a20321b4f498ab30f28e1d09ac465edb7/pyproject.toml)
specifies Python 3.11–3.12, Torch 2.11.0, TorchVision 0.26.0, CUDA 12.8 and
Transformers >=4.57,<4.58. The inventory fixes Torch 2.11.0+cu128,
TorchVision 0.26.0+cu128 and Transformers 4.57.6. The checkpoint config's saved
`transformers_version=5.3.0` is retained as metadata, not treated as an override
of the publisher's working server range. Other publisher dependency ranges are
resolved to exact inference-only versions and their full transitive closure.

Torch 2.11 expresses most CUDA dependencies as `cuda-toolkit` extras. The lock
includes **all eleven requested extras** and their actual NVIDIA wheels:
cublas, cudart, cufft, cufile, cupti, curand, cusolver, cusparse, nvjitlink,
nvrtc and nvtx. It additionally includes required cuDNN, cuSPARSELt, NCCL,
NVSHMEM, CUDA bindings/pathfinder and Triton. Dropping extras during metadata
resolution would incorrectly omit essential runtime libraries.

Every wheel has an exact HTTPS URL, filename, size, SHA-256, Python requirement
and active dependency requirements. PyPI or the official PyTorch index supplies
large CUDA-wheel metadata; these large files were **not downloaded here** and
must be verified in full when staged. Target platform is x86-64 Linux,
glibc >=2.28. The exact interpreter archive has already been downloaded and
hashed; its expanded regular files total 97,446,341 bytes under a 128 MiB cap.

The server environment remains independent from OpenWAM. Isaac's Python 3.11
client uses existing NumPy and standard-library HTTP/JSON/base64; it requires
no new client wheel. The wrapper therefore needs no FastAPI, Uvicorn,
json-numpy, ManiSkill, Gymnasium, tyro or robot-hardware packages. Training,
vLLM, flash-attention builds, video-file decoders and CUDA-graph assumptions
are absent. Accelerate is unnecessary for this wrapper's explicit local
FP32 loading path without `device_map`; actual complete loading remains untested.

## Staging interface

`molmoact2_yam_dependencies.py` exposes the same standalone APIs as the PiPER
dependency package: `verified_dependency_manifest(path=None)` and
`stage_dependency_manifest(runtime)`, plus `DEPENDENCY_MANIFEST_FILENAME`,
`DEPENDENCY_MANIFEST_SHA256`, `DEPENDENCY_MANIFEST_DIGEST` and
`TOTAL_DOWNLOAD_BYTES`. The SHA256 constant identifies exact JSON bytes; the
canonical digest omits the manifest's own digest field. Per-file SHA-256 values
are plain hexadecimal. Validation uses only the standard library.

After bounded downloads and full hash checks, install the exact wheel paths
with the isolated interpreter using pip `--no-index --no-deps --require-hashes
--only-binary=:all:`. Do not run the publisher's broad `uv sync` on the provider.
The interpreter includes bootstrap pip and the wheel inventory pins pip 26.2.1.
Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before loading verified local
checkpoint code. Keep all caches outside the exact checkpoint inventory.

The manifest also binds all six checkpoint Python sources to exact bytes at
model revision `8dcbed66f2380e4393189c303ea72488eb9e63c2`, plus the fifteen
non-weight inputs used by the audit. These sources travel in the verified
checkpoint directory, not a mutable Git checkout. Model/source license evidence
and full weight verification belong to the separate release inventory and
adapter. This dependency inventory does not itself register a candidate.

## Executed CPU proof

The local CPU environment used the exact CPython distribution and production
non-CUDA pins. Torch 2.11.0+cpu and TorchVision 0.26.0+cpu replaced the CUDA pair
solely for this check; CUDA/Triton wheels were omitted. All 31 CPU wheels,
253,283,226 bytes in total, were verified, installed offline with hashes, and
passed `pip check`. The 11,983,605-byte tokenizer was downloaded separately and
matched its publisher SHA-256. No model weight shards were downloaded.

`scripts/audit_molmoact2_yam_inference_imports.py` verifies exactly the fifteen
non-weight files before executing checkpoint code and blocks socket connections.
It loads real `AutoConfig`, `AutoProcessor` and the custom model class, then
exercises the publisher's actual tokenizer, prompt builder, image processor
and state/action normalizers on synthetic diagnostic inputs. Results:

- All six loaded checkpoint modules match the retained source hashes.
- `predict_action` accepts every keyword passed by the Blueprint wrapper.
- Three 640×360 RGB views produce `pixel_values[3,729,588]`, image grids
  `[3,4]`, and input token IDs `[1,694]` for the fixed diagnostic prompt.
- State `[14]` and diagnostic action `[30,14]` normalization preserve distinct
  gripper values in channels 6 and 13; all values remain finite.
- Zero socket connection attempts occurred.

Reproduce in the prepared CPU environment:

```bash
PYTHONPATH=src python scripts/audit_molmoact2_yam_inference_imports.py \
  --checkpoint-metadata /path/to/verified/fifteen-small-files \
  --cache /tmp/molmoact2-cpu-cache \
  --output /tmp/molmoact2-cpu-audit.json
```

The retained [CPU receipt](molmoact2-yam-cpu-import-audit-2026-10-29.json)
records exact versions, source hashes, tensor shapes and explicit false flags
for model construction, weight loading, inference and task execution. This
establishes import/config/tokenizer/API compatibility, not complete CUDA model
loading or native YAM performance. FP32 GPU residency, allocator sizing and
simulation execution still require a bounded real run.

Seven focused tests cover transitive requirements including CUDA extras,
platform tags, source identity, transfer accounting, immutable staging and
rejection of changed bytes or symlink destinations.
