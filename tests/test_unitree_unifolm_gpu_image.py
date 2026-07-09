from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline import unitree_unifolm_gpu_image as image_module


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_unitree_unifolm_gpu_image_context_writes_cuda124_contract(
    tmp_path: Path,
) -> None:
    manifest = image_module.build_unitree_unifolm_gpu_image_context(
        job_dir=tmp_path / "image-context",
        image_ref="docker.io/nijelhunt/blueprint-unitree-unifolm:20260622-cu124",
        generated_at="2026-06-22T00:00:00+00:00",
    )

    assert manifest["status"] == "ready_for_image_build"
    assert manifest["configured_image_ref_is_versioned"] is True
    dockerfile = Path(str(manifest["artifact_paths"]["dockerfile"])).read_text(
        encoding="utf-8"
    )
    assert "nvidia/cuda:12.4.1-devel-ubuntu22.04" in dockerfile
    assert "FROM --platform=linux/amd64" in dockerfile
    assert "https://download.pytorch.org/whl/cu124" in dockerfile
    assert "torch==2.5.1" in dockerfile
    assert "torchvision==0.20.1" in dockerfile
    assert "flash-attn==2.5.6" in dockerfile
    assert "patch_unitree_unifolm_attention.py" in dockerfile
    assert "BLUEPRINT_UNITREE_UNIFOLM_VLA_ATTENTION_IMPLEMENTATION" in dockerfile
    assert "BLUEPRINT_UNITREE_UNIFOLM_DEPENDENCY_PROFILE=inference" in dockerfile
    assert "https://github.com/unitreerobotics/unifolm-vla.git" in dockerfile
    assert "huggingface/lerobot.git@${LEROBOT_REF}" in dockerfile
    assert "run_unitree_unifolm_vla_server" in dockerfile
    assert "run_unitree_unifolm_vla_policy_once" in dockerfile
    assert "BLUEPRINT_UNITREE_UNIFOLM_ALLOW_HF_DOWNLOAD=true" in dockerfile
    assert "unitreerobotics/UnifoLM-VLA-Base" in dockerfile
    assert "unitreerobotics/UnifoLM-VLM-Base" in dockerfile
    assert 'CMD ["bash", "-lc", "sleep infinity"]' in dockerfile
    assert "BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT" not in dockerfile
    assert "HF_TOKEN" not in dockerfile
    assert "dckr_pat_" not in dockerfile

    requirements = Path(str(manifest["artifact_paths"]["requirements"])).read_text(
        encoding="utf-8"
    )
    assert "transformers==4.52.3" in requirements
    assert "tensorflow-cpu==2.15.0" in requirements
    assert "diffusers==0.35.1" in requirements
    assert "json_numpy" in requirements
    assert "qwen-vl-utils" in requirements
    assert "datasets==3.6.0" not in requirements
    assert "deepspeed==0.16.9" not in requirements
    assert "tensorflow_datasets==4.9.3" not in requirements
    assert "tensorflow_graphics==2021.12.3" not in requirements
    assert "wandb" not in requirements
    assert "torch==" not in requirements
    assert "torchvision==" not in requirements

    launcher = Path(str(manifest["artifact_paths"]["server_launcher"])).read_text(
        encoding="utf-8"
    )
    assert "run_real_eval_server.py" in launcher
    assert "--ckpt_path" in launcher
    assert "--vlm_pretrained_path" in launcher
    assert "g1_stack_block" in launcher
    assert "huggingface-cli download" in launcher
    assert "BLUEPRINT_UNITREE_UNIFOLM_POLICY_CHECKPOINT" in launcher
    assert "checkpoints/pytorch_model.pt" in launcher
    policy_once = Path(str(manifest["artifact_paths"]["policy_once_launcher"])).read_text(
        encoding="utf-8"
    )
    assert "run_unitree_unifolm_vla_server" in policy_once
    assert "unitree_unifolm_vla_server_bridge" in policy_once
    assert "blocked_unitree_unifolm_vla_server_startup_timeout" in policy_once

    healthcheck = Path(str(manifest["artifact_paths"]["image_healthcheck"])).read_text(
        encoding="utf-8"
    )
    assert "unitree_unifolm_real_eval_server_missing" in healthcheck
    assert "torch_version_not_2_5_1" in healthcheck
    assert "torch_not_built_for_cu124" in healthcheck
    assert '"unifolm_vla"' in healthcheck
    attention_patch = Path(str(manifest["artifact_paths"]["attention_patch"])).read_text(
        encoding="utf-8"
    )
    assert "attn_implementation=os.getenv" in attention_patch
    assert "flash_attention_2" in attention_patch

    persisted = _read_json(
        tmp_path / "image-context" / "unitree_unifolm_gpu_image_manifest.json"
    )
    assert persisted["truth_boundary"]["no_raw_tokens_or_hashes_written"] is True
    assert persisted["registry_auth"]["docker_pat_file"]["path_redacted"] is True
    assert "path" not in persisted["registry_auth"]["docker_pat_file"]
    assert persisted["registry_auth"]["secret_artifact_policy"]["local_secret_file_paths_recorded"] is False
    assert persisted["runtime_contract"]["model_checkpoint_baked_into_image"] is False
    assert persisted["runtime_contract"]["vlm_checkpoint_baked_into_image"] is False
    assert persisted["runtime_contract"]["unitree_qwen_attention_patch_applied"] is True
    assert persisted["dependency_profile"] == "inference"
    assert persisted["runtime_contract"]["dependency_profile"] == "inference"
    assert (
        persisted["runtime_contract"][
            "inference_profile_keeps_tensorflow_cpu_for_server_preprocessing"
        ]
        is True
    )
    assert "deepspeed==0.16.9" in persisted["dependency_profile_excluded_training_packages"]
    assert (
        persisted["runtime_contract"]["single_action_policy_command"]
        == "/usr/local/bin/run_unitree_unifolm_vla_policy_once"
    )
    assert persisted["commands"]["vast_usage"].endswith(
        '--provider-bundle-kind unitree_unifolm --public-image "${BLUEPRINT_UNITREE_UNIFOLM_GPU_IMAGE_REF}"'
    )


def test_unitree_unifolm_gpu_image_context_can_skip_flash_attn_for_sdpa_fallback(
    tmp_path: Path,
) -> None:
    manifest = image_module.build_unitree_unifolm_gpu_image_context(
        job_dir=tmp_path / "image-context",
        image_ref="docker.io/nijelhunt/blueprint-unitree-unifolm:20260622-cu124-sdpa",
        install_flash_attn=False,
        attention_implementation="sdpa",
        generated_at="2026-06-22T00:00:00+00:00",
    )

    dockerfile = Path(str(manifest["artifact_paths"]["dockerfile"])).read_text(
        encoding="utf-8"
    )
    assert "ARG INSTALL_FLASH_ATTN=false" in dockerfile
    assert "ARG UNITREE_UNIFOLM_ATTENTION_IMPLEMENTATION=sdpa" in dockerfile
    assert "BLUEPRINT_UNITREE_UNIFOLM_FLASH_ATTN_INSTALL_SKIPPED" in dockerfile
    assert 'python3 -m pip install "flash-attn==2.5.6"' in dockerfile
    assert manifest["install_flash_attn"] is False
    assert manifest["attention_implementation"] == "sdpa"


def test_unitree_unifolm_gpu_image_context_can_request_full_dependency_profile(
    tmp_path: Path,
) -> None:
    manifest = image_module.build_unitree_unifolm_gpu_image_context(
        job_dir=tmp_path / "image-context",
        image_ref="docker.io/nijelhunt/blueprint-unitree-unifolm:20260622-cu124-full",
        dependency_profile="full",
        generated_at="2026-06-22T00:00:00+00:00",
    )

    dockerfile = Path(str(manifest["artifact_paths"]["dockerfile"])).read_text(
        encoding="utf-8"
    )
    requirements = Path(str(manifest["artifact_paths"]["requirements"])).read_text(
        encoding="utf-8"
    )
    assert "BLUEPRINT_UNITREE_UNIFOLM_DEPENDENCY_PROFILE=full" in dockerfile
    assert "datasets==3.6.0" in requirements
    assert "deepspeed==0.16.9" in requirements
    assert "tensorflow_datasets==4.9.3" in requirements
    assert manifest["dependency_profile"] == "full"
    assert manifest["dependency_profile_excluded_training_packages"] == []


def test_unitree_unifolm_gpu_image_context_blocks_unversioned_ref(
    tmp_path: Path,
) -> None:
    manifest = image_module.build_unitree_unifolm_gpu_image_context(
        job_dir=tmp_path / "image-context",
        image_ref="docker.io/nijelhunt/blueprint-unitree-unifolm:latest",
        generated_at="2026-06-22T00:00:00+00:00",
    )

    assert manifest["status"] == "context_written_blocked"
    assert "blocked_unitree_unifolm_gpu_image_ref_not_versioned" in manifest["blockers"]
