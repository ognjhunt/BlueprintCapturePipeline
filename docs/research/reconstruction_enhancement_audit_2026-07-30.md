# Reconstruction Enhancement Audit — 2026-07-30

Status: deterministic rejection pending license/runtime qualification and real
held-out evidence. These methods are generated visual support only.

## ArtiFixer

The official source repository is Apache-2.0 at commit
`a392c4dfe17459ef9952407accdb9fcdcdddba98`, but the released NVIDIA model card
says the model is for research and development only and governs checkpoint use
under a separate NVIDIA License. The official CUDA 12 image currently starts
from `nvcr.io/nvidia/pytorch:25.01-py3`, upgrades to PyTorch 2.11/CUDA 12.8,
installs unpinned FlashAttention/MoGe sources, and requires a 6.72 GB 1.3B or
67.6 GB 14B pickled checkpoint plus a named Wan2.1 base model. That is not the
current Blueprint worker lock.

Decision: retain the adapter, but reject execution unless the exact source,
container, checkpoint, base model, license receipt, baseline reconstruction,
and frozen split are digest-bound. Require the real held-out evaluator after
every run. Those bindings cannot override the current rejected license/runtime
audit; that audit must be independently qualified first. The preparation
workflow's learned MoGe scale estimate is never a
metric-scale authority.

Primary sources: [official code](https://github.com/nv-tlabs/artifixer),
[official model card](https://huggingface.co/nvidia/ArtiFixer).

## Difix3D+

The official source commit is
`c76edc595586e16732c91ddee82f3a6d83a8a9cc`. Its NVIDIA license limits the work
and derivatives to non-commercial research or evaluation. The included
requirements leave Torch/Torchvision/Xformers floating while pinning an older
Diffusers/Transformers stack. The quickstart also loads model repository code
with `trust_remote_code=True`. The project provides both single-frame
postprocessing and iterative Nerfstudio/gsplat distillation, but single-frame
output does not establish temporal consistency and neither mode establishes
pose, scale, or collision truth.

Decision: deterministic non-commercial rejection for Blueprint's default
commercial product lane. A separate commercial license, reviewed dependency
lock, safe vendored model code, exact checkpoint, worker smoke, and frozen real
held-out baseline comparison would be required to reconsider it.

Primary sources: [official code and usage](https://github.com/nv-tlabs/Difix3D),
[official license](https://github.com/nv-tlabs/Difix3D/blob/main/LICENSE.txt).

## NVIDIA DiffusionHarmonizer

The official source commit is
`d9a817c8376f82000721a52f9d740ef5c24f47bd`. NVIDIA publishes temporal and
non-temporal checkpoints under the NVIDIA Open Model License and documents a
576x1024 PyTorch/CUDA runtime for Ampere, Hopper, and Blackwell. It is an online
or offline RGB enhancement model; it does not solve camera pose, metric scale,
or collision geometry.

Decision: candidate registered but deterministically rejected until source and
dependency licensing is reviewed, the Harmonizer and Cosmos-Predict2 base model
digests are pinned, a compatible Blueprint worker is built and smoke-tested,
and temporal/non-temporal modes beat the preserved baseline on frozen real
held-out views.

Primary sources: [official model card](https://huggingface.co/nvidia/Harmonizer),
[official NVIDIA NuRec documentation](https://docs.nvidia.com/nurec/fixer/model.html),
[official project](https://research.nvidia.com/labs/sil/projects/diffusion-harmonizer/).

## Shared proof boundary

No candidate may read hidden held-out frames, replace the unenhanced baseline,
promote generated pixels to captured observations, or alter metric, geometry,
collision, physics, physical-success, or deployment qualification. Subjective
visual preference is insufficient; improvement must be independently measured
and operationally meaningful.
