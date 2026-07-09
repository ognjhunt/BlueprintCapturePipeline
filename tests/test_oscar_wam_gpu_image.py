from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from blueprint_pipeline import oscar_wam_gpu_image as image_module
from blueprint_pipeline.oscar_official_release import (
    OFFICIAL_OSCAR_SOURCE_COMMIT,
    OFFICIAL_OSCAR_WAM_IMAGE_DIGEST,
    OFFICIAL_OSCAR_WAM_IMAGE_REF,
)


def _purge_transformer_engine_modules() -> None:
    for name in list(sys.modules):
        if name == "transformer_engine" or name.startswith("transformer_engine."):
            sys.modules.pop(name, None)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_oscar_wam_gpu_image_context_writes_cuda128_shim_contract(
    tmp_path: Path,
) -> None:
    manifest = image_module.build_oscar_wam_gpu_image_context(
        job_dir=tmp_path / "image-context",
        image_ref="registry.example/blueprint/oscar-wam:20260621-cu128",
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert manifest["status"] == "ready_for_image_build"
    assert manifest["configured_image_ref_is_versioned"] is True
    dockerfile = Path(str(manifest["artifact_paths"]["dockerfile"])).read_text(
        encoding="utf-8"
    )
    assert "nvidia/cuda:12.8.0-devel-ubuntu22.04" in dockerfile
    assert "FROM --platform=linux/amd64" in dockerfile
    assert "https://download.pytorch.org/whl/cu128" in dockerfile
    assert "torch==2.10.0" in dockerfile
    assert "torchvision==0.25.0" in dockerfile
    assert "nvidia-cudnn-cu12" in dockerfile
    requirements = Path(str(manifest["artifact_paths"]["requirements"])).read_text(
        encoding="utf-8"
    )
    assert "nvidia-ml-py" in requirements
    assert "loguru" in requirements
    assert "matplotlib" in requirements
    assert "megatron-core" in requirements
    assert "pytest>=8.0.0" in requirements
    assert "transformers>=4.45,<5" in requirements
    assert "CUDNN_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn" in dockerfile
    assert "CPATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/include" in dockerfile
    assert "$CPATH" not in dockerfile
    assert "BLUEPRINT_OSCAR_WAM_SKIP_RUNTIME_PIP_INSTALL=true" in dockerfile
    assert "install_transformer_engine_shim.py" in dockerfile
    assert "transformer_engine[pytorch]" in dockerfile
    assert "BLUEPRINT_TRANSFORMER_ENGINE_MODE=shim" in dockerfile
    assert "BLUEPRINT_OSCAR_WAM_SOURCE_ROOT=/opt/oscar-public" in dockerfile
    assert OFFICIAL_OSCAR_SOURCE_COMMIT in dockerfile
    assert "BLUEPRINT_OSCAR_WAM_CHECKPOINT" not in dockerfile
    assert "HF_TOKEN" not in dockerfile
    assert "dckr_pat_" not in dockerfile

    shim_script = Path(str(manifest["artifact_paths"]["transformer_engine_shim"])).read_text(
        encoding="utf-8"
    )
    assert '"pytorch" / "tensor" / "__init__.py"' in shim_script
    assert "class QuantizedTensor" in shim_script
    assert "class Float8Tensor" in shim_script
    assert '"pytorch" / "fp8.py"' in shim_script
    assert "class FP8GlobalStateManager" in shim_script
    assert '"common" / "recipe.py"' in shim_script
    assert "class DelayedScaling" in shim_script
    assert "class Linear(torch.nn.Linear)" in shim_script
    assert "class LayerNormLinear" in shim_script
    assert "from . import distributed, ops" in shim_script
    assert '"pytorch" / "ops" / "__init__.py"' in shim_script
    assert '"pytorch" / "distributed" / "__init__.py"' in shim_script

    healthcheck = Path(str(manifest["artifact_paths"]["image_healthcheck"])).read_text(
        encoding="utf-8"
    )
    assert "cudnn.h" in healthcheck
    assert "torch_version_not_2_10_0" in healthcheck
    assert "torch_not_built_for_cu128" in healthcheck
    assert "transformer_engine_or_shim_not_importable" in healthcheck
    assert "transformer_engine_tensor_api_not_importable" in healthcheck
    assert "QuantizedTensor" in healthcheck
    assert "Float8Tensor" in healthcheck
    assert "FP8GlobalStateManager" in healthcheck
    assert "DelayedScaling" in healthcheck
    assert "transformer_engine_module_api_importable" in healthcheck
    assert "LayerNormLinear" in healthcheck
    assert "transformer_engine_ops_api_importable" in healthcheck
    assert "pynvml_not_importable" in healthcheck
    assert "loguru_not_importable" in healthcheck
    assert "matplotlib" in healthcheck
    assert "megatron.core" in healthcheck
    assert '"pytest": "pytest"' in healthcheck
    assert '{label}_not_importable' in healthcheck
    assert "worldsim_runtime_imports" in healthcheck

    persisted = _read_json(tmp_path / "image-context" / "oscar_wam_gpu_image_manifest.json")
    assert persisted["truth_boundary"]["no_raw_tokens_or_hashes_written"] is True
    assert persisted["registry_auth"]["docker_pat_file"]["path_redacted"] is True
    assert "path" not in persisted["registry_auth"]["docker_pat_file"]
    assert persisted["registry_auth"]["secret_artifact_policy"]["local_secret_file_paths_recorded"] is False
    assert persisted["platform"] == "linux/amd64"
    assert persisted["oscar_source_ref"] == OFFICIAL_OSCAR_SOURCE_COMMIT
    assert (
        persisted["official_oscar_release"]["source_ref_pinned_to_reviewed_commit"]
        is True
    )
    assert (
        persisted["official_checked_image"]["digest_ref"]
        == OFFICIAL_OSCAR_WAM_IMAGE_REF
    )
    assert (
        persisted["official_checked_image"]["index_digest"]
        == OFFICIAL_OSCAR_WAM_IMAGE_DIGEST
    )
    assert persisted["runtime_contract"]["model_checkpoint_baked_into_image"] is False
    assert persisted["runtime_contract"]["raw_credentials_baked_into_image"] is False
    build_script = Path(str(manifest["artifact_paths"]["build_command"])).read_text(
        encoding="utf-8"
    )
    assert 'SCRIPT_DIR="' in build_script
    assert "--platform linux/amd64" in build_script
    assert str(tmp_path) not in build_script


def test_gpu_image_transformer_engine_shim_rope_matches_te_split_half_semantics(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    script = tmp_path / "install_transformer_engine_shim.py"
    source_root = tmp_path / "source"
    script.write_text(image_module.transformer_engine_shim_script_text(), encoding="utf-8")
    source_root.mkdir()

    subprocess.run(
        [sys.executable, str(script), str(source_root)],
        check=True,
        capture_output=True,
        text=True,
    )

    def _rotate_half(tensor: object, interleaved: bool) -> object:
        if not interleaved:
            first, second = torch.chunk(tensor, 2, dim=-1)
            return torch.cat((-second, first), dim=-1)
        first = tensor[..., ::2]
        second = tensor[..., 1::2]
        return torch.stack((-second, first), dim=-1).flatten(-2)

    def _reference_rope(tensor: object, freqs: object, *, interleaved: bool = False) -> object:
        freqs = freqs.transpose(0, 1)
        cos = torch.cos(freqs).to(tensor.dtype)
        sin = torch.sin(freqs).to(tensor.dtype)
        rot_dim = freqs.shape[-1]
        tensor_rot = tensor[..., :rot_dim]
        tensor_pass = tensor[..., rot_dim:]
        tensor_rot = tensor_rot * cos + _rotate_half(tensor_rot, interleaved) * sin
        return torch.cat((tensor_rot, tensor_pass), dim=-1)

    sys.path.insert(0, str(source_root))
    try:
        _purge_transformer_engine_modules()
        from transformer_engine.pytorch.attention import apply_rotary_pos_emb

        tensor = torch.arange(36, dtype=torch.float32).reshape(1, 3, 2, 6) / 10.0
        freqs = torch.linspace(0.01, 0.24, steps=12, dtype=torch.float32).reshape(
            3, 1, 1, 4
        )

        assert torch.allclose(
            apply_rotary_pos_emb(tensor, freqs, tensor_format="bshd", fused=True),
            _reference_rope(tensor, freqs),
        )
        assert torch.allclose(
            apply_rotary_pos_emb(
                tensor,
                freqs,
                tensor_format="bshd",
                interleaved=True,
                fused=True,
            ),
            _reference_rope(tensor, freqs, interleaved=True),
        )
    finally:
        _purge_transformer_engine_modules()
        sys.path.remove(str(source_root))


def test_oscar_wam_gpu_image_context_blocks_unversioned_ref(tmp_path: Path) -> None:
    manifest = image_module.build_oscar_wam_gpu_image_context(
        job_dir=tmp_path / "image-context",
        image_ref="registry.example/blueprint/oscar-wam:latest",
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert manifest["status"] == "context_written_blocked"
    assert "blocked_oscar_wam_gpu_image_ref_not_versioned" in manifest["blockers"]


def test_oscar_wam_gpu_image_real_transformer_engine_mode(tmp_path: Path) -> None:
    manifest = image_module.build_oscar_wam_gpu_image_context(
        job_dir=tmp_path / "image-context",
        image_ref="registry.example/blueprint/oscar-wam:20260621-real-te",
        transformer_engine_mode="real",
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert manifest["transformer_engine_strategy"] == (
        "real_transformer_engine_pip_no_build_isolation"
    )
    dockerfile = Path(str(manifest["artifact_paths"]["dockerfile"])).read_text(
        encoding="utf-8"
    )
    assert "BLUEPRINT_TRANSFORMER_ENGINE_MODE=real" in dockerfile
    assert "NVTE_FRAMEWORK=pytorch" in dockerfile
    assert "--no-build-isolation" in dockerfile


def test_oscar_wam_gpu_image_rejects_unknown_transformer_engine_mode(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="transformer_engine_mode"):
        image_module.build_oscar_wam_gpu_image_context(
            job_dir=tmp_path / "image-context",
            image_ref="registry.example/blueprint/oscar-wam:20260621",
            transformer_engine_mode="unsupported",
        )
