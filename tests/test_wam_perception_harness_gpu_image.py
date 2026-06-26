from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline import wam_perception_harness_gpu_image as image_module


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_wam_perception_harness_gpu_image_context_writes_provider_contract(
    tmp_path: Path,
) -> None:
    manifest = image_module.build_wam_perception_harness_gpu_image_context(
        job_dir=tmp_path / "image-context",
        image_ref="docker.io/nijelhunt/blueprint-wam-perception-harness:20260626-cu126",
        generated_at="2026-06-26T00:00:00+00:00",
    )

    assert manifest["status"] == "ready_for_image_build"
    assert manifest["configured_image_ref_is_versioned"] is True
    assert manifest["runtime_contract"]["bakes_python_provider_dependencies"] is True
    assert manifest["runtime_contract"]["bakes_sam3_weights"] is False
    assert manifest["runtime_contract"]["sam3_weights_mount_or_fetch_required"] is True
    assert manifest["truth_boundary"]["no_raw_tokens_or_hashes_written"] is True
    assert Path(str(manifest["artifact_paths"]["context_pyproject"])).is_file()
    assert Path(str(manifest["artifact_paths"]["context_readme"])).is_file()
    assert Path(str(manifest["artifact_paths"]["context_src"])).is_dir()
    assert (
        Path(str(manifest["artifact_paths"]["context_src"]))
        / "blueprint_pipeline"
        / "wam_sim_provider_e2e.py"
    ).is_file()

    dockerfile = Path(str(manifest["artifact_paths"]["dockerfile"])).read_text(
        encoding="utf-8"
    )
    assert "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04" in dockerfile
    assert "FROM --platform=linux/amd64" in dockerfile
    assert "https://download.pytorch.org/whl/cu126" in dockerfile
    assert "torch==2.7.0" in dockerfile
    assert "torchvision==0.22.0" in dockerfile
    assert "python3 -m venv /opt/blueprint/venv" in dockerfile
    assert "VIRTUAL_ENV=/opt/blueprint/venv" in dockerfile
    assert 'python -m pip install -e ".[cloud,runtime,retrieval,validation]"' in dockerfile
    assert "huggingface_hub[cli]" in dockerfile
    assert "SAM3_WEIGHTS_PATH=/models/sam3/sam3.pt" in dockerfile
    assert "BLUEPRINT_WAM_POSE_MODEL_PATH=/models/yolo/yolo11n-pose.pt" in dockerfile
    assert "Depth-Anything-V2-Small-hf" in dockerfile
    assert "AutoModelForDepthEstimation.from_pretrained" in dockerfile
    assert "YOLO(\"yolo11n-pose.pt\")" in dockerfile
    assert "ByteDance-Seed/depth-anything-3" in dockerfile
    assert "HF_TOKEN" not in dockerfile
    assert "dckr_pat_" not in dockerfile
    assert "dop_v1_" not in dockerfile

    healthcheck = Path(str(manifest["artifact_paths"]["image_healthcheck"])).read_text(
        encoding="utf-8"
    )
    assert "wam_perception_harness_gpu_image_healthcheck.v1" in healthcheck
    assert "_run_fixture_smoke" in healthcheck
    assert "wam_sim_provider_e2e_fixture_smoke_failed" in healthcheck
    assert "torch_not_built_for_cu126" in healthcheck
    assert "sam3_weights_missing" in healthcheck
    assert "depth_anything_3" in healthcheck
    assert "image_healthcheck_is_not_provider_accuracy_validation" in healthcheck

    build_script = Path(str(manifest["artifact_paths"]["build_command"])).read_text(
        encoding="utf-8"
    )
    assert 'SCRIPT_DIR="' in build_script
    assert "--platform linux/amd64" in build_script
    assert "--build-arg INSTALL_DA3=false" in build_script
    assert "--build-arg PREFETCH_WAM_PERCEPTION_MODELS=true" in build_script
    assert str(tmp_path) not in build_script

    push_script = Path(str(manifest["artifact_paths"]["push_command"])).read_text(
        encoding="utf-8"
    )
    assert "DOCKER_USERNAME_FILE" in push_script
    assert "DOCKER_PAT_FILE" in push_script
    assert "--password-stdin" in push_script
    assert "dckr_pat_" not in push_script

    prepare_script = Path(
        str(manifest["artifact_paths"]["prepare_model_mounts_command"])
    ).read_text(encoding="utf-8")
    assert "Put SAM3 weights at" in prepare_script
    assert "Do not paste raw HF, Docker, or object-store secrets" in prepare_script

    persisted = _read_json(
        tmp_path / "image-context" / "wam_perception_harness_gpu_image_manifest.json"
    )
    assert persisted["runtime_contract"]["default_depth_provider"] == (
        "transformers_depth_anything_v2"
    )
    assert persisted["registry_auth"]["registry_auth_secret_values_written"] is False
    assert persisted["object_store_auth"]["object_store_secret_values_written"] is False
    run_healthcheck_script = Path(
        str(manifest["artifact_paths"]["run_healthcheck_command"])
    ).read_text(encoding="utf-8")
    assert "--entrypoint python" in run_healthcheck_script


def test_wam_perception_harness_gpu_image_context_blocks_unversioned_ref(
    tmp_path: Path,
) -> None:
    manifest = image_module.build_wam_perception_harness_gpu_image_context(
        job_dir=tmp_path / "image-context",
        image_ref="docker.io/nijelhunt/blueprint-wam-perception-harness:latest",
        generated_at="2026-06-26T00:00:00+00:00",
    )

    assert manifest["status"] == "context_written_blocked"
    assert "blocked_wam_perception_harness_gpu_image_ref_not_versioned" in manifest[
        "blockers"
    ]


def test_wam_perception_harness_gpu_image_context_can_request_da3_without_prefetch(
    tmp_path: Path,
) -> None:
    manifest = image_module.build_wam_perception_harness_gpu_image_context(
        job_dir=tmp_path / "image-context",
        image_ref="docker.io/nijelhunt/blueprint-wam-perception-harness:20260626-da3",
        install_da3=True,
        prefetch_models=False,
        generated_at="2026-06-26T00:00:00+00:00",
    )

    assert manifest["install_da3"] is True
    assert manifest["prefetch_models"] is False
    dockerfile = Path(str(manifest["artifact_paths"]["dockerfile"])).read_text(
        encoding="utf-8"
    )
    build_script = Path(str(manifest["artifact_paths"]["build_command"])).read_text(
        encoding="utf-8"
    )
    assert "ARG INSTALL_DA3=true" in dockerfile
    assert "ARG PREFETCH_WAM_PERCEPTION_MODELS=false" in dockerfile
    assert "--build-arg INSTALL_DA3=true" in build_script
    assert "--build-arg PREFETCH_WAM_PERCEPTION_MODELS=false" in build_script
