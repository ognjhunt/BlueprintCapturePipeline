# HumanoidArena G1 beta inference dependencies

This profile serves the DP and Pi0.5 policy server in public HumanoidArena commit
[`68479287a784a69be9ce6ad739311d2f11f75ef9`](https://github.com/William-wAng618/HumanoidArena/tree/68479287a784a69be9ce6ad739311d2f11f75ef9).
It is separate from the legacy G1 environment. The legacy requirements file stays
unchanged at SHA-256 `584a48b4c42f7911780d5cde20c2c216e35a4ba2d0680a8805d8327e53536e2a`.

The JSON inventory seals 96 CPython 3.12 Linux x86_64 wheels, including their
versions, filenames, exact SHA-256, sizes, active dependency metadata, public
license metadata and source URLs. It requires glibc 2.28 or newer. New downloads
total **261,777,389 bytes**; **4,132,469,228 bytes** are reused from the verified
native source packet's `runtime_dependencies/wheels/`. Missing native wheels
must produce a source-packet error, rather than an additional unquoted download.
The helper stages JSON only and uses the already pinned Isaac image interpreter;
it downloads neither a second interpreter nor another copy of the Torch/CUDA wheels.
The separate policy venv installs the verified reused wheels and new wheels
without inheriting Isaac's site-packages.

| Package | Version | Wheel bytes | Delivery |
| --- | --- | ---: | --- |
| Torch | 2.10.0+cu128 | 916,856,347 | Native source packet |
| TorchVision | 0.25.0+cu128 | 8,072,400 | External pinned wheel |
| Transformers | 5.3.0 | 10,661,827 | External pinned wheel |
| Diffusers | 0.35.2 | 4,121,649 | External pinned wheel |
| NumPy | 2.2.6 | 16,527,618 | External pinned wheel |

Installation is offline, `--no-deps --require-hashes`, with no apt packages or
source builds. `verify_wheelhouse` verifies every byte, refuses unsafe ZIP
members, and limits aggregate expanded wheel contents to 16 GiB while honoring
the absolute bootstrap deadline. This is a conservative installation ceiling,
not a measured GPU memory requirement. The runtime builder must quote runtime
packet transfer only once and reserve its own complete workspace overhead.

The closure follows the actual publisher import graph. Factory/configuration
imports bring in dataset metadata, PyArrow, pandas, PyAV, imageio and pyserial,
even though the policy server does not open a motor bus. Those packages remain.
Unused evdev, pynput/Xlib, cmake, wandb, rerun-sdk, torchcodec and accelerate are
excluded. In particular, the old evdev source build and build-essential install
are unnecessary for this policy-server path. The source remains unmodified.

Package versions were selected from the existing legacy pins and native packet
where compatible. Fsspec is 2026.2.0 because datasets 4.8.5 caps it there, and
typing-extensions 4.16.0 satisfies anyio's published requirement. Active extras
(including fsspec HTTP support) are included. The manifest tests check complete
version constraints and CPython/platform wheel compatibility.

Public package metadata is evidence of the declared terms, not a separate
permission or model-rights approval. The exact wheels retain their complete
licenses and notices, including bundled third-party libraries. The MIT text in
mypy-extensions is verified from its exact wheel member where release metadata
is blank. HumanoidArena's root source is MIT; inherited LeRobot source retains
its Apache-2.0 notices and bundled license file. No third-party source was copied
into the standard-library profile helper, which uses this repository's license.
Checkpoint, tokenizer and auxiliary initializer rights remain separate runtime
inputs; this inventory contains none of their weights.

## CPU validation and limits

The retained audit at `/workspace/work/g1-beta-audit/` used all 363 selected
publisher Python/package/license files (4,099,808 bytes), individually checked
against the pinned Git tree blob hashes. It is a selected source tree, not a
complete checkout. A fresh CPython 3.12 environment installed 78 wheels offline
from exact hashes, using Torch 2.10.0+cpu and TorchVision 0.25.0+cpu solely for this
CPU audit. All other versions match the applicable inference closure. `pip check`
passed. Network socket connections were refused during publisher imports and
construction.

Actual imports passed for the publisher HTTP server, policy factory, Diffusion
Policy and Pi0.5. Both retained DP configurations constructed 262,730,568
parameters on CPU, using the separately verified retained ResNet18 initializer.
Both Pi0.5 configurations constructed 4,143,421,208-parameter graphs on the meta
device, with no parameter storage. Only the device was overridden in the exact
publisher configurations. The companion audit JSON records config digests,
CPU-wheel identities, source inventory digest and the individual observations.

No policy checkpoints were downloaded or loaded, no prediction was performed,
and no GPU, SONIC control, native task or physical performance was established.
Production still needs the verified full publisher source, exact checkpoint and
auxiliary assets, the native runtime packet, and a bounded allocator-approved
GPU execution. CUDA wheel bytes are sealed from public release metadata and
verified during production staging; this CPU check does not test CUDA loading.
