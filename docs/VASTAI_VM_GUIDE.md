# Vast.ai VM Guide (CUDA COLMAP + Offline-Ready Snapshot)

This guide documents the current production path for running `scripts/nurec_shim.py` on Vast.ai with:

- CUDA-enabled COLMAP
- SAM3 + DA3 integration
- Optional Fixer (local or H100 stage offload)
- Snapshot image path with no runtime model/dependency downloads

## 1) Quick Start (Snapshot Image)

Build and push snapshot image:

```bash
./scripts/build_vast_snapshot.sh --tag cuda-snapshot
```

Provision/test instance from snapshot:

```bash
./scripts/vastai_bootstrap.sh
```

Defaults:

- `VASTAI_IMAGE=nijelhunt/blueprint-capture-pipeline:cuda-snapshot`
- Onstart command is `sleep infinity` (no apt/pip runtime installs)

If you want to override image:

```bash
VASTAI_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04 ./scripts/vastai_bootstrap.sh
```

## 2) One-Time Install on Existing Base VM

If you started from a plain CUDA image, install full stack once:

```bash
./scripts/vastai_bootstrap.sh --instance-id <ID> --install-ml
```

Optional local Fixer install:

```bash
./scripts/vastai_bootstrap.sh --instance-id <ID> --install-ml --with-fixer
```

Installer used: `scripts/install_ml_stack.sh`

What it does:

- Installs CUDA COLMAP from source (`scripts/install_colmap_cuda.sh`)
- Installs 3DGRUT, SAM3, DA3
- Downloads DA3 weights
- Prewarms SAM3 + DA3 caches
- Runs offline load validation (`HF_HUB_OFFLINE=1`)

## 3) Validate VM State

Run on VM:

```bash
which colmap
colmap help | head -n 4
```

Expected: `/usr/local/bin/colmap` and banner includes `with CUDA`.

Check no first-run model pulls required:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -c "from sam3 import build_sam3_image_model; build_sam3_image_model(); from depth_anything_3.api import DepthAnything3; DepthAnything3.from_pretrained('/opt/da3/weights/metric_large'); print('offline-ok')"
```

## 4) Run NuRec Shim End-to-End

```bash
python3 /app/scripts/nurec_shim.py \
  --job-spec /workspace/pipeline/job_spec.json \
  --output-dir /workspace/pipeline/output \
  --raw-prefix /workspace/pipeline/input/video.MOV \
  --environment warehouse
```

Notes:

- Dense stage is enabled by default (remove `--skip-dense` unless debugging).
- SAM3 defaults are now auto-tuned by clip length/environment:
  - warehouse: more sampled frames, `min_frame_detections=1` by default
- Override if needed:
  - `--sam3-n-frames <N>`
  - `--sam3-min-frame-detections <N>`

## 5) Fixer Modes

`nurec_shim.py` supports:

- `--skip-fixer`
- `--fixer-mode local`
- `--fixer-mode h100`
- `--fixer-mode auto` (H100 first, then local fallback)

H100 stage script:

- `scripts/fixer_h100_stage.sh`

## 6) Resolved First-Run Issues

| Issue | Current State |
|---|---|
| apt COLMAP without CUDA | Fixed via `scripts/install_colmap_cuda.sh` |
| Headless COLMAP crashes | Fixed (CUDA build + shim option compatibility for old/new COLMAP flags) |
| Dense mesh skipped | No longer required by default; dense stage runs when COLMAP CUDA is present |
| SAM3 too few surviving objects | Improved defaults (more frames, lower warehouse filter threshold) |
| DA3 runtime downloads | Avoided via local `DA3_MODEL_PATH` and prewarmed cache |
| Fixer flash-attn mismatch on 12.4 | Kept optional; use H100 mode for reliable Fixer stage |

## 7) Snapshot Requirements (No Runtime Downloads)

To satisfy "no realtime downloads":

1. Build snapshot with `scripts/build_vast_snapshot.sh`.
2. Use snapshot image in Vast instance creation.
3. Verify offline load command in section 3.

At that point, runtime pipeline execution does not need internet downloads for COLMAP/SAM3/DA3 dependencies.
