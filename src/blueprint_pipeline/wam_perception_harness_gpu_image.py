"""Build context generator for the WAM perception harness GPU image."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .wam_real_provider_validation_probe import (
    DEFAULT_DA3_MODEL_ID,
    DEFAULT_DEPTH_MODEL_ID,
    DEFAULT_POSE_MODEL_PATH,
)


WAM_PERCEPTION_HARNESS_GPU_IMAGE_SCHEMA_VERSION = (
    "wam_perception_harness_gpu_image_context.v1"
)
DEFAULT_BASE_IMAGE = "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04"
DEFAULT_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu126"
DEFAULT_TORCH_VERSION = "2.7.0"
DEFAULT_TORCHVISION_VERSION = "0.22.0"
DEFAULT_PLATFORM = "linux/amd64"
DEFAULT_CONTEXT_FILENAME = "Dockerfile.wam-perception-harness-gpu"
DEFAULT_IMAGE_REF = "docker.io/nijelhunt/blueprint-wam-perception-harness:20260626-cu126"
IMAGE_REF_ENV = "BLUEPRINT_WAM_PERCEPTION_HARNESS_GPU_IMAGE_REF"
LEGACY_IMAGE_REF_ENV = "BLUEPRINT_WAM_PROVIDER_IMAGE_REF"
DEFAULT_SAM3_WEIGHTS_PATH = "/models/sam3/sam3.pt"
DEFAULT_POSE_MODEL_IMAGE_PATH = "/models/yolo/yolo11n-pose.pt"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _image_ref_is_versioned(image_ref: str) -> bool:
    if not image_ref or image_ref.endswith(":latest"):
        return False
    last = image_ref.rsplit("/", maxsplit=1)[-1]
    return ":" in last or "@" in last


def _secret_file_status(env_name: str, default_path: str) -> dict[str, Any]:
    configured = _string(os.getenv(env_name))
    path = Path(configured or default_path).expanduser()
    mode = oct(path.stat().st_mode & 0o777) if path.exists() else None
    return {
        "env_name": env_name,
        "path": str(path),
        "configured_by_env": bool(configured),
        "present": path.is_file(),
        "mode": mode,
        "mode_is_0600": mode == "0o600",
        "raw_secret_values_recorded": False,
        "secret_hash_recorded": False,
    }


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def image_healthcheck_text() -> str:
    return r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import tempfile
from pathlib import Path


def _probe(label: str, module: str) -> dict[str, object]:
    try:
        imported = importlib.import_module(module)
        return {
            "label": label,
            "module": module,
            "status": "importable",
            "version": getattr(imported, "__version__", None),
        }
    except Exception as exc:
        return {
            "label": label,
            "module": module,
            "status": "blocked",
            "error_type": type(exc).__name__,
            "error_preview": str(exc)[:300],
        }


def _run_fixture_smoke() -> dict[str, object]:
    try:
        import argparse as argparse_module
        from PIL import Image

        from blueprint_pipeline.wam_sim_provider_e2e import run_sim_provider_e2e

        with tempfile.TemporaryDirectory(prefix="wam_perception_harness_image_") as tmp:
            root = Path(tmp)
            frame = root / "generated_start.jpg"
            Image.new("RGB", (320, 240), (42, 48, 52)).save(frame)
            args = argparse_module.Namespace(
                output_dir=root / "sim_e2e",
                generated_frame=frame,
                step_count=1,
                target_prompt="robot arm",
                policy_id="image_healthcheck_policy",
                policy_schema="rgbd_mask_pose",
                provider_mode="fixture",
                sam3_weights=None,
                sam3_confidence=0.01,
                pose_model=os.getenv("BLUEPRINT_WAM_POSE_MODEL_PATH") or "yolo11n-pose.pt",
                depth_provider="v2",
                depth_model_id=os.getenv("BLUEPRINT_WAM_DEPTH_MODEL_ID")
                or "depth-anything/Depth-Anything-V2-Small-hf",
                da3_model_id=os.getenv("BLUEPRINT_WAM_DA3_MODEL_ID") or "depth-anything/DA3-BASE",
                backend_timeout_seconds=60,
            )
            manifest = run_sim_provider_e2e(args)
            return {
                "status": manifest.get("status"),
                "sim_only_provider_harness_e2e_completed": manifest.get(
                    "sim_only_provider_harness_e2e_completed"
                ),
                "perception_accuracy_validated": manifest.get("perception_accuracy_validated"),
                "step_count_completed": manifest.get("step_count_completed"),
            }
    except Exception as exc:
        return {
            "status": "blocked",
            "error_type": type(exc).__name__,
            "error_preview": str(exc)[:500],
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-time", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-sam3-weights", action="store_true")
    parser.add_argument("--skip-fixture-smoke", action="store_true")
    args = parser.parse_args()

    probes = [
        _probe("torch", "torch"),
        _probe("torchvision", "torchvision"),
        _probe("transformers", "transformers"),
        _probe("ultralytics", "ultralytics"),
        _probe("cv2", "cv2"),
        _probe("PIL", "PIL"),
        _probe("huggingface_hub", "huggingface_hub"),
        _probe("blueprint_pipeline", "blueprint_pipeline"),
        _probe("wam_real_provider_validation_probe", "blueprint_pipeline.wam_real_provider_validation_probe"),
        _probe("wam_sim_provider_e2e", "blueprint_pipeline.wam_sim_provider_e2e"),
    ]
    da3_probe = _probe("depth_anything_3", "depth_anything_3")
    blockers = [f"{row['label']}_not_importable" for row in probes if row["status"] != "importable"]

    cuda_available = False
    torch_cuda = None
    torch_probe = next(row for row in probes if row["label"] == "torch")
    if torch_probe["status"] == "importable":
        import torch

        cuda_available = bool(torch.cuda.is_available())
        torch_cuda = torch.version.cuda
        if not str(torch.__version__).split("+", 1)[0].startswith("2.7."):
            blockers.append("torch_version_not_2_7_x")
        if torch_cuda and not str(torch_cuda).startswith("12.6"):
            blockers.append("torch_not_built_for_cu126")
    if args.require_cuda and not cuda_available:
        blockers.append("cuda_not_available")

    sam3_weights_path = Path(os.getenv("SAM3_WEIGHTS_PATH") or "/models/sam3/sam3.pt")
    pose_model_path = Path(os.getenv("BLUEPRINT_WAM_POSE_MODEL_PATH") or "/models/yolo/yolo11n-pose.pt")
    hf_home = Path(os.getenv("HF_HOME") or "/models/hf")
    if args.require_sam3_weights and not sam3_weights_path.is_file():
        blockers.append("sam3_weights_missing")

    fixture_smoke = {"status": "skipped"}
    if not args.skip_fixture_smoke:
        fixture_smoke = _run_fixture_smoke()
        if fixture_smoke.get("status") != "completed":
            blockers.append("wam_sim_provider_e2e_fixture_smoke_failed")

    payload = {
        "schema_version": "wam_perception_harness_gpu_image_healthcheck.v1",
        "status": "completed" if not blockers else "blocked",
        "build_time": bool(args.build_time),
        "require_cuda": bool(args.require_cuda),
        "require_sam3_weights": bool(args.require_sam3_weights),
        "cuda_available": cuda_available,
        "torch_cuda": torch_cuda,
        "sam3_weights_path": str(sam3_weights_path),
        "sam3_weights_present": sam3_weights_path.is_file(),
        "pose_model_path": str(pose_model_path),
        "pose_model_present": pose_model_path.is_file(),
        "hf_home": str(hf_home),
        "hf_home_present": hf_home.exists(),
        "probes": probes,
        "optional_probes": [da3_probe],
        "fixture_smoke": fixture_smoke,
        "blockers": blockers,
        "claim_boundary": {
            "image_healthcheck_is_not_provider_accuracy_validation": True,
            "fixture_smoke_is_not_real_provider_execution": True,
            "sam3_masks_are_not_physical_truth": True,
            "inferred_depth_is_not_sensor_depth": True,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
        },
        "raw_secret_values_recorded": False,
        "secret_hashes_recorded": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
'''


def dockerfile_text(
    *,
    base_image: str = DEFAULT_BASE_IMAGE,
    platform: str = DEFAULT_PLATFORM,
    torch_index_url: str = DEFAULT_TORCH_INDEX_URL,
    torch_version: str = DEFAULT_TORCH_VERSION,
    torchvision_version: str = DEFAULT_TORCHVISION_VERSION,
    install_da3: bool = False,
    prefetch_models: bool = True,
) -> str:
    prefetch = "true" if prefetch_models else "false"
    da3_default = _bool_text(install_da3)
    return f"""# syntax=docker/dockerfile:1.7
FROM --platform={platform} {base_image}

ARG INSTALL_DA3={da3_default}
ARG PREFETCH_WAM_PERCEPTION_MODELS={prefetch}

ENV DEBIAN_FRONTEND=noninteractive \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1 \\
    HF_HOME=/models/hf \\
    TRANSFORMERS_CACHE=/models/hf \\
    SAM3_WEIGHTS_PATH={DEFAULT_SAM3_WEIGHTS_PATH} \\
    BLUEPRINT_WAM_POSE_MODEL_PATH={DEFAULT_POSE_MODEL_IMAGE_PATH} \\
    BLUEPRINT_WAM_DEPTH_MODEL_ID={DEFAULT_DEPTH_MODEL_ID} \\
    BLUEPRINT_WAM_DA3_MODEL_ID={DEFAULT_DA3_MODEL_ID} \\
    BLUEPRINT_WAM_DEPTH_PROVIDER_KIND=transformers_depth_anything_v2 \\
    BLUEPRINT_ALLOW_WAM_AUTO_DEPTH_PROVIDER=true \\
    BLUEPRINT_ALLOW_WAM_AUTO_POSE_PROVIDER=true \\
    BLUEPRINT_ALLOW_WAM_PERCEPTION_HARNESS_EXTERNAL_BACKEND=true

WORKDIR /workspace/BlueprintCapturePipeline

RUN apt-get update && apt-get install -y --no-install-recommends \\
    ca-certificates \\
    curl \\
    ffmpeg \\
    git \\
    git-lfs \\
    libgl1 \\
    libglib2.0-0 \\
    python3 \\
    python3-dev \\
    python3-pip \\
    python3-venv \\
  && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python
RUN python3 -m venv /opt/blueprint/venv

ENV VIRTUAL_ENV=/opt/blueprint/venv \\
    PATH="/opt/blueprint/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \\
    PYTHONPATH="/workspace/BlueprintCapturePipeline/src"

COPY pyproject.toml README.md ./
COPY src ./src
COPY wam_perception_harness_image_healthcheck.py /opt/blueprint/wam_perception_harness_image_healthcheck.py

RUN python -m pip install --upgrade pip setuptools wheel \\
  && python -m pip install --index-url {torch_index_url} \\
     torch=={torch_version} torchvision=={torchvision_version} \\
  && python -m pip install -e ".[cloud,runtime,retrieval,validation]" \\
  && python -m pip install "huggingface_hub[cli]" safetensors timm

RUN if [ "$INSTALL_DA3" = "true" ]; then \\
      python -m pip install "git+https://github.com/ByteDance-Seed/depth-anything-3.git"; \\
    else \\
      echo "Depth Anything 3 install skipped; enable with --build-arg INSTALL_DA3=true"; \\
    fi

RUN mkdir -p /models/hf /models/sam3 /models/yolo

RUN if [ "$PREFETCH_WAM_PERCEPTION_MODELS" = "true" ]; then \\
      python3 -c 'from pathlib import Path; from transformers import AutoImageProcessor, AutoModelForDepthEstimation; from ultralytics import YOLO; AutoImageProcessor.from_pretrained("{DEFAULT_DEPTH_MODEL_ID}"); AutoModelForDepthEstimation.from_pretrained("{DEFAULT_DEPTH_MODEL_ID}"); YOLO("{DEFAULT_POSE_MODEL_PATH}"); source = Path("{DEFAULT_POSE_MODEL_PATH}"); target = Path("{DEFAULT_POSE_MODEL_IMAGE_PATH}"); target.parent.mkdir(parents=True, exist_ok=True); source.is_file() and (not target.is_file()) and target.write_bytes(source.read_bytes())'; \\
    fi

RUN python /opt/blueprint/wam_perception_harness_image_healthcheck.py --build-time

ENTRYPOINT ["python", "-m", "blueprint_pipeline.wam_sim_provider_e2e"]
"""


def _build_command_text(
    *,
    platform: str,
    image_ref_default: str,
    install_da3: bool,
    prefetch_models: bool,
) -> str:
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        f'IMAGE_REF="${{{IMAGE_REF_ENV}:-{image_ref_default}}}"\n'
        f'docker build --platform {platform} \\\n'
        f'  --build-arg INSTALL_DA3={_bool_text(install_da3)} \\\n'
        f'  --build-arg PREFETCH_WAM_PERCEPTION_MODELS={_bool_text(prefetch_models)} \\\n'
        f'  -f "$SCRIPT_DIR/{DEFAULT_CONTEXT_FILENAME}" \\\n'
        '  -t "$IMAGE_REF" "$SCRIPT_DIR"\n'
    )


def _push_command_text(*, image_ref_default: str) -> str:
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'IMAGE_REF="${{{IMAGE_REF_ENV}:-{image_ref_default}}}"\n'
        'if [[ -n "${DOCKER_USERNAME_FILE:-}" && -n "${DOCKER_PAT_FILE:-}" ]]; then\n'
        '  docker login -u "$(cat "$DOCKER_USERNAME_FILE")" --password-stdin < "$DOCKER_PAT_FILE"\n'
        "fi\n"
        'docker push "$IMAGE_REF"\n'
    )


def _run_healthcheck_command_text(*, image_ref_default: str) -> str:
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'IMAGE_REF="${{{IMAGE_REF_ENV}:-{image_ref_default}}}"\n'
        'docker run --rm --gpus all \\\n'
        '  -e SAM3_WEIGHTS_PATH="${SAM3_WEIGHTS_PATH:-/models/sam3/sam3.pt}" \\\n'
        '  --entrypoint python \\\n'
        '  "$IMAGE_REF" \\\n'
        "  /opt/blueprint/wam_perception_harness_image_healthcheck.py --require-cuda\n"
    )


def _prepare_models_command_text() -> str:
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'MODEL_DIR="${1:-$PWD/model_mounts}"\n'
        'mkdir -p "$MODEL_DIR/sam3" "$MODEL_DIR/yolo" "$MODEL_DIR/hf"\n'
        "cat <<'MSG'\n"
        "This script prepares host-side model mount directories only.\n"
        "Put SAM3 weights at: $MODEL_DIR/sam3/sam3.pt\n"
        "Do not paste raw HF, Docker, or object-store secrets into generated artifacts.\n"
        "Use local secret files or provider-native secrets when fetching gated weights.\n"
        "MSG\n"
    )


def build_wam_perception_harness_gpu_image_context(
    *,
    job_dir: Path | None = None,
    image_ref: str | None = None,
    base_image: str = DEFAULT_BASE_IMAGE,
    platform: str = DEFAULT_PLATFORM,
    torch_index_url: str = DEFAULT_TORCH_INDEX_URL,
    torch_version: str = DEFAULT_TORCH_VERSION,
    torchvision_version: str = DEFAULT_TORCHVISION_VERSION,
    install_da3: bool = False,
    prefetch_models: bool = True,
    generated_at: str | None = None,
) -> dict[str, Any]:
    root = _repo_root()
    generated = generated_at or utc_now_iso()
    output = Path(
        job_dir or root / "robot_eval_jobs" / f"wam_perception_harness_gpu_image_{_timestamp()}"
    ).expanduser().resolve()
    ensure_dir(output)
    configured_image_ref = (
        _string(image_ref)
        or _string(os.getenv(IMAGE_REF_ENV))
        or _string(os.getenv(LEGACY_IMAGE_REF_ENV))
        or DEFAULT_IMAGE_REF
    )

    dockerfile_path = output / DEFAULT_CONTEXT_FILENAME
    healthcheck_path = output / "wam_perception_harness_image_healthcheck.py"
    build_command_path = output / "build_image.sh"
    push_command_path = output / "push_image.sh"
    run_healthcheck_command_path = output / "run_image_healthcheck.sh"
    prepare_models_command_path = output / "prepare_model_mounts.sh"
    context_pyproject_path = output / "pyproject.toml"
    context_readme_path = output / "README.md"
    context_src_path = output / "src"

    shutil.copy2(root / "pyproject.toml", context_pyproject_path)
    shutil.copy2(root / "README.md", context_readme_path)
    if context_src_path.exists():
        shutil.rmtree(context_src_path)
    shutil.copytree(
        root / "src",
        context_src_path,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    dockerfile_path.write_text(
        dockerfile_text(
            base_image=base_image,
            platform=platform,
            torch_index_url=torch_index_url,
            torch_version=torch_version,
            torchvision_version=torchvision_version,
            install_da3=install_da3,
            prefetch_models=prefetch_models,
        ),
        encoding="utf-8",
    )
    healthcheck_path.write_text(image_healthcheck_text(), encoding="utf-8")
    build_command_path.write_text(
        _build_command_text(
            platform=platform,
            image_ref_default=configured_image_ref,
            install_da3=install_da3,
            prefetch_models=prefetch_models,
        ),
        encoding="utf-8",
    )
    push_command_path.write_text(
        _push_command_text(image_ref_default=configured_image_ref),
        encoding="utf-8",
    )
    run_healthcheck_command_path.write_text(
        _run_healthcheck_command_text(image_ref_default=configured_image_ref),
        encoding="utf-8",
    )
    prepare_models_command_path.write_text(_prepare_models_command_text(), encoding="utf-8")
    for path in (
        healthcheck_path,
        build_command_path,
        push_command_path,
        run_healthcheck_command_path,
        prepare_models_command_path,
    ):
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    blockers: list[str] = []
    if not configured_image_ref:
        blockers.append(f"missing_env_{IMAGE_REF_ENV}")
    elif not _image_ref_is_versioned(configured_image_ref):
        blockers.append("blocked_wam_perception_harness_gpu_image_ref_not_versioned")

    docker_auth = {
        "docker_username_file": _secret_file_status(
            "DOCKER_USERNAME_FILE",
            "~/.blueprint-secrets/docker_username",
        ),
        "docker_pat_file": _secret_file_status("DOCKER_PAT_FILE", "~/.blueprint-secrets/docker_pat"),
        "registry_auth_secret_values_written": False,
        "registry_auth_secret_hashes_written": False,
    }
    object_store_auth = {
        "digitalocean_api_token_file": _secret_file_status(
            "DIGITALOCEAN_API_TOKEN_FILE",
            "~/.blueprint-secrets/digitalocean_api_token",
        ),
        "spaces_access_key_id_file": _secret_file_status(
            "DIGITALOCEAN_SPACES_ACCESS_KEY_ID_FILE",
            "~/.blueprint-secrets/digitalocean_spaces_access_key_id",
        ),
        "spaces_secret_access_key_file": _secret_file_status(
            "DIGITALOCEAN_SPACES_SECRET_ACCESS_KEY_FILE",
            "~/.blueprint-secrets/digitalocean_spaces_secret_access_key",
        ),
        "object_store_secret_values_written": False,
        "object_store_secret_hashes_written": False,
    }
    manifest = {
        "schema_version": WAM_PERCEPTION_HARNESS_GPU_IMAGE_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "ready_for_image_build" if not blockers else "context_written_blocked",
        "job_dir": str(output),
        "image_ref_env": IMAGE_REF_ENV,
        "legacy_image_ref_env": LEGACY_IMAGE_REF_ENV,
        "configured_image_ref_present": bool(configured_image_ref),
        "configured_image_ref": configured_image_ref or None,
        "configured_image_ref_is_versioned": _image_ref_is_versioned(configured_image_ref),
        "base_image": base_image,
        "platform": platform,
        "torch_index_url": torch_index_url,
        "torch_version": torch_version,
        "torch_cuda_wheel_family": "cu126",
        "torchvision_version": torchvision_version,
        "install_da3": bool(install_da3),
        "prefetch_models": bool(prefetch_models),
        "runtime_contract": {
            "image_family": "wam_perception_harness",
            "bakes_blueprint_harness_code": True,
            "bakes_python_provider_dependencies": True,
            "bakes_depth_anything_v2_cache": bool(prefetch_models),
            "bakes_yolo_pose_model_cache": bool(prefetch_models),
            "bakes_sam3_weights": False,
            "sam3_weights_expected_path": DEFAULT_SAM3_WEIGHTS_PATH,
            "sam3_weights_mount_or_fetch_required": True,
            "raw_credentials_baked_into_image": False,
            "supports_fixture_mode": True,
            "supports_real_provider_probe_mode": True,
            "supports_optional_da3": True,
            "default_depth_provider": "transformers_depth_anything_v2",
            "default_depth_model_id": DEFAULT_DEPTH_MODEL_ID,
            "default_pose_model_path": DEFAULT_POSE_MODEL_IMAGE_PATH,
        },
        "artifact_paths": {
            "dockerfile": str(dockerfile_path),
            "context_pyproject": str(context_pyproject_path),
            "context_readme": str(context_readme_path),
            "context_src": str(context_src_path),
            "image_healthcheck": str(healthcheck_path),
            "build_command": str(build_command_path),
            "push_command": str(push_command_path),
            "run_healthcheck_command": str(run_healthcheck_command_path),
            "prepare_model_mounts_command": str(prepare_models_command_path),
            "manifest": str(output / "wam_perception_harness_gpu_image_manifest.json"),
        },
        "commands": {
            "build": str(build_command_path),
            "push": str(push_command_path),
            "run_gpu_healthcheck": str(run_healthcheck_command_path),
            "prepare_model_mounts": str(prepare_models_command_path),
            "run_fixture_e2e": (
                f"docker run --rm ${{{IMAGE_REF_ENV}}} --provider-mode fixture --step-count 1"
            ),
            "run_real_provider_e2e": (
                "docker run --rm --gpus all "
                "-v /host/models/sam3:/models/sam3:ro "
                f"${{{IMAGE_REF_ENV}}} --provider-mode real --step-count 2 --depth-provider v2"
            ),
        },
        "registry_auth": docker_auth,
        "object_store_auth": object_store_auth,
        "blockers": blockers,
        "truth_boundary": {
            "image_build_is_not_provider_execution": True,
            "image_push_is_not_wam_rollout_generation": True,
            "image_healthcheck_is_not_perception_accuracy_validation": True,
            "model_outputs_remain_derived_observations": True,
            "sam3_masks_are_not_physical_truth": True,
            "inferred_depth_is_not_sensor_depth": True,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
            "no_raw_tokens_or_hashes_written": True,
        },
        "raw_secret_values_recorded": False,
        "secret_hashes_recorded": False,
    }
    write_json(output / "wam_perception_harness_gpu_image_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write a reusable WAM perception harness GPU image build context."
    )
    parser.add_argument("--job-dir")
    parser.add_argument("--image-ref", default=None)
    parser.add_argument("--base-image", default=DEFAULT_BASE_IMAGE)
    parser.add_argument("--platform", default=DEFAULT_PLATFORM)
    parser.add_argument("--torch-index-url", default=DEFAULT_TORCH_INDEX_URL)
    parser.add_argument("--torch-version", default=DEFAULT_TORCH_VERSION)
    parser.add_argument("--torchvision-version", default=DEFAULT_TORCHVISION_VERSION)
    parser.add_argument("--install-da3", action="store_true")
    parser.add_argument("--no-prefetch-models", action="store_true")
    args = parser.parse_args(argv)
    manifest = build_wam_perception_harness_gpu_image_context(
        job_dir=Path(args.job_dir) if args.job_dir else None,
        image_ref=args.image_ref,
        base_image=args.base_image,
        platform=args.platform,
        torch_index_url=args.torch_index_url,
        torch_version=args.torch_version,
        torchvision_version=args.torchvision_version,
        install_da3=args.install_da3,
        prefetch_models=not args.no_prefetch_models,
    )
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["status"] == "ready_for_image_build" else 2


if __name__ == "__main__":
    raise SystemExit(main())
