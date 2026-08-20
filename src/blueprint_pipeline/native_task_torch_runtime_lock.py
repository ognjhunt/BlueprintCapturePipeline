"""Exact official PyTorch CUDA wheel closure for Isaac Sim 6.0 Arena runs."""

from __future__ import annotations

from typing import Any


PYTORCH_CU128_INDEX = "https://download.pytorch.org/whl/cu128"


def _wheel(
    filename: str,
    package: str,
    version: str,
    license_spdx: str,
    *,
    pure_python: bool = True,
    wheel_tag: str = "py3-none-any",
) -> dict[str, Any]:
    return {
        "filename": filename,
        "package": package,
        "version": version,
        "license_spdx": license_spdx,
        "pure_python": pure_python,
        "wheel_tag": wheel_tag,
        "source": PYTORCH_CU128_INDEX,
    }


# Resolved from the official cu128 index for CPython 3.12 / Linux x86_64.
# typing-extensions is already present in the base runtime closure at a version
# satisfying Torch's >=4.10 requirement, so it is intentionally not duplicated.
TORCH_RUNTIME_DEPENDENCY_WHEELS = (
    _wheel(
        "torch-2.10.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl",
        "torch",
        "2.10.0+cu128",
        "BSD-3-Clause",
        pure_python=False,
        wheel_tag="cp312-cp312-manylinux_2_28_x86_64",
    ),
    _wheel(
        "triton-3.6.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        "triton",
        "3.6.0",
        "MIT",
        pure_python=False,
        wheel_tag="cp312-cp312-manylinux_2_28_x86_64",
    ),
    _wheel("cuda_pathfinder-1.6.0-py3-none-any.whl", "cuda-pathfinder", "1.6.0", "Apache-2.0"),
    _wheel(
        "cuda_bindings-12.9.4-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
        "cuda-bindings",
        "12.9.4",
        "LicenseRef-NVIDIA-SOFTWARE-LICENSE",
        pure_python=False,
        wheel_tag="cp312-cp312-manylinux_2_28_x86_64",
    ),
    _wheel("filelock-3.32.3-py3-none-any.whl", "filelock", "3.32.3", "MIT"),
    _wheel("fsspec-2026.7.0-py3-none-any.whl", "fsspec", "2026.7.0", "BSD-3-Clause"),
    _wheel("jinja2-3.1.6-py3-none-any.whl", "Jinja2", "3.1.6", "BSD-3-Clause"),
    _wheel(
        "markupsafe-3.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
        "MarkupSafe",
        "3.0.3",
        "BSD-3-Clause",
        pure_python=False,
        wheel_tag="cp312-cp312-manylinux_2_28_x86_64",
    ),
    _wheel("mpmath-1.3.0-py3-none-any.whl", "mpmath", "1.3.0", "BSD-3-Clause"),
    _wheel("networkx-3.6.1-py3-none-any.whl", "networkx", "3.6.1", "BSD-3-Clause"),
    _wheel("setuptools-78.1.0-py3-none-any.whl", "setuptools", "78.1.0", "MIT"),
    _wheel("sympy-1.14.0-py3-none-any.whl", "sympy", "1.14.0", "BSD-3-Clause"),
    _wheel(
        "nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl",
        "nvidia-cublas-cu12",
        "12.8.4.1",
        "LicenseRef-NVIDIA-Proprietary",
        wheel_tag="py3-none-manylinux_2_27_x86_64",
    ),
    _wheel(
        "nvidia_cuda_cupti_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        "nvidia-cuda-cupti-cu12",
        "12.8.90",
        "LicenseRef-NVIDIA-Proprietary",
        wheel_tag="py3-none-manylinux_2_17_x86_64",
    ),
    _wheel(
        "nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl",
        "nvidia-cuda-nvrtc-cu12",
        "12.8.93",
        "LicenseRef-NVIDIA-Proprietary",
        wheel_tag="py3-none-manylinux_2_12_x86_64",
    ),
    _wheel(
        "nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        "nvidia-cuda-runtime-cu12",
        "12.8.90",
        "LicenseRef-NVIDIA-Proprietary",
        wheel_tag="py3-none-manylinux_2_17_x86_64",
    ),
    _wheel(
        "nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl",
        "nvidia-cudnn-cu12",
        "9.10.2.21",
        "LicenseRef-NVIDIA-Proprietary",
        wheel_tag="py3-none-manylinux_2_27_x86_64",
    ),
    _wheel(
        "nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        "nvidia-cufft-cu12",
        "11.3.3.83",
        "LicenseRef-NVIDIA-Proprietary",
        wheel_tag="py3-none-manylinux_2_17_x86_64",
    ),
    _wheel(
        "nvidia_cufile_cu12-1.13.1.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        "nvidia-cufile-cu12",
        "1.13.1.3",
        "LicenseRef-NVIDIA-Proprietary",
        wheel_tag="py3-none-manylinux_2_17_x86_64",
    ),
    _wheel(
        "nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl",
        "nvidia-curand-cu12",
        "10.3.9.90",
        "LicenseRef-NVIDIA-Proprietary",
        wheel_tag="py3-none-manylinux_2_27_x86_64",
    ),
    _wheel(
        "nvidia_cusolver_cu12-11.7.3.90-py3-none-manylinux_2_27_x86_64.whl",
        "nvidia-cusolver-cu12",
        "11.7.3.90",
        "LicenseRef-NVIDIA-Proprietary",
        wheel_tag="py3-none-manylinux_2_27_x86_64",
    ),
    _wheel(
        "nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        "nvidia-cusparse-cu12",
        "12.5.8.93",
        "LicenseRef-NVIDIA-Proprietary",
        wheel_tag="py3-none-manylinux_2_17_x86_64",
    ),
    _wheel(
        "nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl",
        "nvidia-cusparselt-cu12",
        "0.7.1",
        "LicenseRef-NVIDIA-Proprietary",
        wheel_tag="py3-none-manylinux2014_x86_64",
    ),
    _wheel(
        "nvidia_nccl_cu12-2.27.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        "nvidia-nccl-cu12",
        "2.27.5",
        "BSD-3-Clause",
        wheel_tag="py3-none-manylinux_2_17_x86_64",
    ),
    _wheel(
        "nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl",
        "nvidia-nvjitlink-cu12",
        "12.8.93",
        "LicenseRef-NVIDIA-Proprietary",
        wheel_tag="py3-none-manylinux_2_12_x86_64",
    ),
    _wheel(
        "nvidia_nvshmem_cu12-3.4.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        "nvidia-nvshmem-cu12",
        "3.4.5",
        "LicenseRef-NVIDIA-Proprietary",
        wheel_tag="py3-none-manylinux_2_17_x86_64",
    ),
    _wheel(
        "nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        "nvidia-nvtx-cu12",
        "12.8.90",
        "Apache-2.0",
        wheel_tag="py3-none-manylinux_2_17_x86_64",
    ),
)


__all__ = ["PYTORCH_CU128_INDEX", "TORCH_RUNTIME_DEPENDENCY_WHEELS"]
