from __future__ import annotations

import subprocess

from blueprint_pipeline.vast_cuda_runtime_probe import (
    cuda_runtime_probe_shell_fragment,
    gpu_sanity_from_log,
)


def test_cuda_runtime_probe_covers_runtime_image_locations_and_pytorch_fallback() -> None:
    fragment = cuda_runtime_probe_shell_fragment(required=True)

    for location in (
        "/usr/local/cuda*/lib64/libcudart.so*",
        "/usr/lib/*-linux-gnu/libcudart.so*",
        "/opt/conda/lib/libcudart.so*",
        "/opt/conda/lib/python*/site-packages/nvidia/cuda_runtime/lib/libcudart.so*",
    ):
        assert location in fragment
    assert "torch.cuda.is_available()" in fragment
    assert "torch.cuda.device_count()" in fragment
    assert "backend=pytorch" in fragment
    subprocess.run(
        ["bash", "-n"],
        input=fragment,
        text=True,
        check=True,
        capture_output=True,
    )


def test_pytorch_cuda_fallback_marker_qualifies_gpu_sanity() -> None:
    result = gpu_sanity_from_log(
        "\n".join(
            (
                "BLUEPRINT_VAST_GPU_SANITY_OK",
                "BLUEPRINT_VAST_CUDA_RUNTIME_API_OK:devices=1:backend=pytorch",
                "BLUEPRINT_VAST_CUDA_RUNTIME_OK",
            )
        ),
        require_cuda_runtime=True,
    )

    assert result["gpu_ok"] is True
    assert result["blockers"] == []
