"""Build context generator for a reusable Unitree UnifoLM VLA GPU image."""

from __future__ import annotations

import argparse
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json


UNITREE_UNIFOLM_GPU_IMAGE_SCHEMA_VERSION = "unitree_unifolm_gpu_image_context.v1"
DEFAULT_BASE_IMAGE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"
DEFAULT_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"
DEFAULT_TORCH_VERSION = "2.5.1"
DEFAULT_TORCHVISION_VERSION = "0.20.1"
DEFAULT_FLASH_ATTN_VERSION = "2.5.6"
DEFAULT_INSTALL_FLASH_ATTN = True
DEFAULT_ATTENTION_IMPLEMENTATION = "flash_attention_2"
DEFAULT_UNITREE_SOURCE_URL = "https://github.com/unitreerobotics/unifolm-vla.git"
DEFAULT_UNITREE_SOURCE_REF = "main"
DEFAULT_LEROBOT_REF = "0878c68"
DEFAULT_PLATFORM = "linux/amd64"
DEFAULT_CONTEXT_FILENAME = "Dockerfile.unitree-unifolm-vla-gpu"
DEFAULT_DEPENDENCY_PROFILE = "inference"
SUPPORTED_DEPENDENCY_PROFILES = ("inference", "full")
IMAGE_REF_ENV = "BLUEPRINT_UNITREE_UNIFOLM_GPU_IMAGE_REF"


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
        "raw_secret_value_recorded": False,
        "secret_hash_recorded": False,
    }


def _full_requirements() -> list[str]:
    return [
        "accelerate==1.5.2",
        "albumentations==1.4.18",
        "av",
        "datasets==3.6.0",
        "decord==0.6.0",
        "deepspeed==0.16.9",
        "diffusers==0.35.1",
        "einops",
        "eva-decord==0.6.1",
        "fastapi",
        "fastparquet==2024.11.0",
        "h5py",
        "huggingface_hub==0.34.4",
        "json_numpy",
        "jsonlines==4.0.0",
        "matplotlib",
        "mujoco==3.3.5",
        "numpy==1.26.4",
        "numpydantic==1.6.9",
        "omegaconf",
        "pillow==11.3.0",
        "pydantic==2.10.6",
        "pyarrow==15.0.1",
        "qwen-vl-utils",
        "rich",
        "scipy",
        "tensorboard",
        "tensorflow==2.15.0",
        "tensorflow_datasets==4.9.3",
        "tensorflow_graphics==2021.12.3",
        "tiktoken",
        "transformers==4.52.3",
        "transformers_stream_generator==0.0.4",
        "tyro==0.9.35",
        "uvicorn",
        "wandb",
        "websocket-client==1.8.0",
    ]


def _inference_requirements() -> list[str]:
    return [
        "accelerate==1.5.2",
        "av",
        "decord==0.6.0",
        "diffusers==0.35.1",
        "einops",
        "eva-decord==0.6.1",
        "fastapi",
        "huggingface_hub==0.34.4",
        "json_numpy",
        "jsonlines==4.0.0",
        "numpy==1.26.4",
        "omegaconf",
        "pillow==11.3.0",
        "pydantic==2.10.6",
        "qwen-vl-utils",
        "rich",
        "scipy",
        "tensorflow-cpu==2.15.0",
        "tiktoken",
        "tqdm",
        "transformers==4.52.3",
        "transformers_stream_generator==0.0.4",
        "tyro==0.9.35",
        "uvicorn",
        "websocket-client==1.8.0",
    ]


def _dependency_packages(profile: str) -> list[str]:
    normalized = profile.strip().lower()
    if normalized == "full":
        return _full_requirements()
    if normalized == "inference":
        return _inference_requirements()
    raise ValueError(f"unsupported Unitree UnifoLM dependency profile: {profile}")


def requirements_text(profile: str = DEFAULT_DEPENDENCY_PROFILE) -> str:
    packages = _dependency_packages(profile)
    return "\n".join(packages) + "\n"


def image_healthcheck_text() -> str:
    return r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-time", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    probes = [
        _probe("torch", "torch"),
        _probe("torchvision", "torchvision"),
        _probe("transformers", "transformers"),
        _probe("fastapi", "fastapi"),
        _probe("uvicorn", "uvicorn"),
        _probe("json_numpy", "json_numpy"),
        _probe("tensorflow", "tensorflow"),
        _probe("qwen_vl_utils", "qwen_vl_utils"),
        _probe("unifolm_vla", "unifolm_vla"),
    ]
    blockers: list[str] = [
        f"{row['label']}_not_importable" for row in probes if row["status"] != "importable"
    ]
    torch_probe = next(row for row in probes if row["label"] == "torch")
    cuda_available = False
    torch_cuda = None
    if torch_probe["status"] == "importable":
        import torch

        cuda_available = bool(torch.cuda.is_available())
        torch_cuda = torch.version.cuda
        if torch.__version__.split("+", 1)[0] != "2.5.1":
            blockers.append("torch_version_not_2_5_1")
        if torch_cuda and not str(torch_cuda).startswith("12.4"):
            blockers.append("torch_not_built_for_cu124")
    if args.require_cuda and not cuda_available:
        blockers.append("cuda_not_available")

    source_root = os.getenv("BLUEPRINT_UNITREE_UNIFOLM_SOURCE_ROOT", "/opt/unifolm-vla")
    server_path = os.path.join(source_root, "deployment", "model_server", "run_real_eval_server.py")
    if not os.path.isfile(server_path):
        blockers.append("unitree_unifolm_real_eval_server_missing")

    payload = {
        "schema_version": "unitree_unifolm_gpu_image_healthcheck.v1",
        "status": "completed" if not blockers else "blocked",
        "build_time": bool(args.build_time),
        "require_cuda": bool(args.require_cuda),
        "cuda_available": cuda_available,
        "torch_cuda": torch_cuda,
        "source_root": source_root,
        "real_eval_server_path": server_path,
        "probes": probes,
        "blockers": blockers,
        "raw_secret_values_recorded": False,
        "secret_hashes_recorded": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
	'''


def attention_patch_text() -> str:
    return r'''#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


TARGET = Path("/opt/unifolm-vla/src/unifolm_vla/model/modules/vlm/QWen2_5.py")


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    if "import os" not in text:
        text = text.replace("import torch\n", "import os\nimport torch\n", 1)
    text = text.replace(
        'attn_implementation="flash_attention_2",',
        (
            'attn_implementation=os.getenv('
            '"BLUEPRINT_UNITREE_UNIFOLM_VLA_ATTENTION_IMPLEMENTATION", '
            '"flash_attention_2"'
            "),"
        ),
        1,
    )
    TARGET.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def server_launcher_text() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${BLUEPRINT_UNITREE_UNIFOLM_SOURCE_ROOT:-/opt/unifolm-vla}"
PORT="${BLUEPRINT_UNITREE_UNIFOLM_VLA_SERVER_PORT:-8777}"
UNNORM_KEY="${BLUEPRINT_UNITREE_UNIFOLM_UNNORM_KEY:-g1_stack_block}"
MODEL_CACHE_ROOT="${BLUEPRINT_UNITREE_UNIFOLM_MODEL_CACHE_ROOT:-/mnt/models}"
ALLOW_HF_DOWNLOAD="${BLUEPRINT_UNITREE_UNIFOLM_ALLOW_HF_DOWNLOAD:-true}"
VLA_REPO="${BLUEPRINT_UNITREE_UNIFOLM_VLA_REPO:-unitreerobotics/UnifoLM-VLA-Base}"
VLM_REPO="${BLUEPRINT_UNITREE_UNIFOLM_VLM_REPO:-unitreerobotics/UnifoLM-VLM-Base}"
VLA_INPUT="${BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT:-${BLUEPRINT_UNITREE_UNIFOLM_POLICY_CHECKPOINT:-}}"
VLM_INPUT="${BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT:-}"

download_repo() {
  local repo="$1"
  local dest="$2"
  mkdir -p "$dest"
  echo "BLUEPRINT_UNITREE_UNIFOLM_DOWNLOAD_REPO:${repo}->${dest}" >&2
  huggingface-cli download "$repo" --local-dir "$dest" --local-dir-use-symlinks False >/tmp/blueprint_unitree_hf_download.log 2>&1 || {
    cat /tmp/blueprint_unitree_hf_download.log >&2 || true
    return 1
  }
}

resolve_vla_checkpoint() {
  local candidate="$1"
  if [ -n "$candidate" ] && [ -f "$candidate" ]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  if [ -n "$candidate" ] && [ -d "$candidate" ] && [ -f "$candidate/checkpoints/pytorch_model.pt" ]; then
    printf '%s\n' "$candidate/checkpoints/pytorch_model.pt"
    return 0
  fi
  if [ "$ALLOW_HF_DOWNLOAD" != "true" ]; then
    echo "blocked_unitree_unifolm_vla_checkpoint_missing_and_hf_download_disabled" >&2
    return 2
  fi
  local repo="$VLA_REPO"
  if [[ "$candidate" == unitreerobotics/* ]]; then
    repo="$candidate"
  fi
  local dest="$MODEL_CACHE_ROOT/${repo##*/}"
  if [ ! -f "$dest/checkpoints/pytorch_model.pt" ]; then
    download_repo "$repo" "$dest"
  fi
  printf '%s\n' "$dest/checkpoints/pytorch_model.pt"
}

resolve_vlm_checkpoint() {
  local candidate="$1"
  if [ -n "$candidate" ] && [ -d "$candidate" ]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  if [ "$ALLOW_HF_DOWNLOAD" != "true" ]; then
    echo "blocked_unitree_unifolm_vlm_checkpoint_missing_and_hf_download_disabled" >&2
    return 2
  fi
  local repo="$VLM_REPO"
  if [[ "$candidate" == unitreerobotics/* ]]; then
    repo="$candidate"
  fi
  local dest="$MODEL_CACHE_ROOT/${repo##*/}"
  if [ ! -f "$dest/config.json" ]; then
    download_repo "$repo" "$dest"
  fi
  printf '%s\n' "$dest"
}

VLA_CHECKPOINT="$(resolve_vla_checkpoint "$VLA_INPUT")"
VLM_CHECKPOINT="$(resolve_vlm_checkpoint "$VLM_INPUT")"

exec python3 "$SOURCE_ROOT/deployment/model_server/run_real_eval_server.py" \
  --ckpt_path "$VLA_CHECKPOINT" \
  --port "$PORT" \
  --unnorm_key "$UNNORM_KEY" \
  --vlm_pretrained_path "$VLM_CHECKPOINT"
'''


def policy_once_launcher_text() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail

HOST="${BLUEPRINT_UNITREE_UNIFOLM_VLA_SERVER_HOST:-127.0.0.1}"
PORT="${BLUEPRINT_UNITREE_UNIFOLM_VLA_SERVER_PORT:-8777}"
SERVER_URL="${BLUEPRINT_UNITREE_UNIFOLM_VLA_SERVER_URL:-http://${HOST}:${PORT}/act}"
STARTUP_TIMEOUT="${BLUEPRINT_UNITREE_UNIFOLM_VLA_SERVER_STARTUP_TIMEOUT_SECONDS:-900}"
LOG_PATH="${BLUEPRINT_UNITREE_UNIFOLM_VLA_SERVER_LOG:-/tmp/blueprint_unitree_unifolm_vla_server.log}"

port_ready() {
python3 - "$HOST" "$PORT" <<'PY'
import socket
import sys
host = sys.argv[1]
port = int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(1.0)
    raise SystemExit(0 if sock.connect_ex((host, port)) == 0 else 1)
PY
}

if ! port_ready; then
  run_unitree_unifolm_vla_server >"$LOG_PATH" 2>&1 &
  server_pid=$!
  deadline=$((SECONDS + STARTUP_TIMEOUT))
  until port_ready; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "blocked_unitree_unifolm_vla_server_exited_before_ready" >&2
      tail -n 200 "$LOG_PATH" >&2 || true
      exit 2
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "blocked_unitree_unifolm_vla_server_startup_timeout" >&2
      tail -n 200 "$LOG_PATH" >&2 || true
      exit 2
    fi
    sleep 5
  done
fi

exec python3 -m blueprint_pipeline.unitree_unifolm_vla_server_bridge \
  --server-url "$SERVER_URL"
'''


def dockerfile_text(
    *,
    base_image: str = DEFAULT_BASE_IMAGE,
    platform: str = DEFAULT_PLATFORM,
    torch_index_url: str = DEFAULT_TORCH_INDEX_URL,
    torch_version: str = DEFAULT_TORCH_VERSION,
    torchvision_version: str = DEFAULT_TORCHVISION_VERSION,
    flash_attn_version: str = DEFAULT_FLASH_ATTN_VERSION,
    install_flash_attn: bool = DEFAULT_INSTALL_FLASH_ATTN,
    attention_implementation: str = DEFAULT_ATTENTION_IMPLEMENTATION,
    unitree_source_url: str = DEFAULT_UNITREE_SOURCE_URL,
    unitree_source_ref: str = DEFAULT_UNITREE_SOURCE_REF,
    lerobot_ref: str = DEFAULT_LEROBOT_REF,
    dependency_profile: str = DEFAULT_DEPENDENCY_PROFILE,
) -> str:
    install_flash = "true" if install_flash_attn else "false"
    return f"""# syntax=docker/dockerfile:1
# Blueprint reusable Unitree UnifoLM VLA provider GPU image.
# This image intentionally excludes raw credentials and model checkpoints.
FROM --platform={platform} {base_image}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG UNITREE_UNIFOLM_SOURCE_URL={unitree_source_url}
ARG UNITREE_UNIFOLM_SOURCE_REF={unitree_source_ref}
ARG LEROBOT_REF={lerobot_ref}
ARG INSTALL_FLASH_ATTN={install_flash}
ARG UNITREE_UNIFOLM_ATTENTION_IMPLEMENTATION={attention_implementation}
ARG BLUEPRINT_UNITREE_UNIFOLM_DEPENDENCY_PROFILE={dependency_profile}

ENV PIP_NO_CACHE_DIR=1 \\
    PYTHONUNBUFFERED=1 \\
    BLUEPRINT_UNITREE_UNIFOLM_SOURCE_ROOT=/opt/unifolm-vla \\
    BLUEPRINT_UNITREE_UNIFOLM_VLA_SERVER_PORT=8777 \\
    BLUEPRINT_UNITREE_UNIFOLM_ALLOW_HF_DOWNLOAD=true \\
    BLUEPRINT_UNITREE_UNIFOLM_VLA_REPO=unitreerobotics/UnifoLM-VLA-Base \\
    BLUEPRINT_UNITREE_UNIFOLM_VLM_REPO=unitreerobotics/UnifoLM-VLM-Base \\
    BLUEPRINT_UNITREE_UNIFOLM_VLA_ATTENTION_IMPLEMENTATION=${{UNITREE_UNIFOLM_ATTENTION_IMPLEMENTATION}} \\
    BLUEPRINT_UNITREE_UNIFOLM_DEPENDENCY_PROFILE=${{BLUEPRINT_UNITREE_UNIFOLM_DEPENDENCY_PROFILE}} \\
    PYTHONPATH=/opt/unifolm-vla/src:/workspace/provider_runtime

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    ca-certificates \\
    cmake \\
    curl \\
    ffmpeg \\
    git \\
    libgl1 \\
    libglib2.0-0 \\
    ninja-build \\
    pkg-config \\
    python-is-python3 \\
    python3 \\
    python3-dev \\
    python3-pip \\
    python3-venv \\
    unzip \\
  && rm -rf /var/lib/apt/lists/*

COPY requirements_blueprint_unitree_unifolm.txt /opt/blueprint/requirements_blueprint_unitree_unifolm.txt
COPY unitree_unifolm_image_healthcheck.py /opt/blueprint/unitree_unifolm_image_healthcheck.py
COPY patch_unitree_unifolm_attention.py /opt/blueprint/patch_unitree_unifolm_attention.py
COPY run_unitree_unifolm_vla_server.sh /usr/local/bin/run_unitree_unifolm_vla_server
COPY run_unitree_unifolm_vla_policy_once.sh /usr/local/bin/run_unitree_unifolm_vla_policy_once

RUN chmod +x /usr/local/bin/run_unitree_unifolm_vla_server \\
  && chmod +x /usr/local/bin/run_unitree_unifolm_vla_policy_once \\
  && python3 -m pip install --upgrade pip setuptools wheel packaging ninja \\
  && python3 -m pip install --index-url {torch_index_url} \\
      torch=={torch_version} torchvision=={torchvision_version} \\
  && python3 -m pip install -r /opt/blueprint/requirements_blueprint_unitree_unifolm.txt

RUN python3 -m pip install --no-deps "lerobot @ git+https://github.com/huggingface/lerobot.git@${{LEROBOT_REF}}" \\
  && git clone --depth 1 "$UNITREE_UNIFOLM_SOURCE_URL" "$BLUEPRINT_UNITREE_UNIFOLM_SOURCE_ROOT" \\
  && if [[ "$UNITREE_UNIFOLM_SOURCE_REF" != "main" && "$UNITREE_UNIFOLM_SOURCE_REF" != "HEAD" ]]; then \\
       cd "$BLUEPRINT_UNITREE_UNIFOLM_SOURCE_ROOT" \\
       && git fetch --depth 1 origin "$UNITREE_UNIFOLM_SOURCE_REF" \\
       && git checkout FETCH_HEAD; \\
     fi \\
  && python3 -m pip install --no-deps -e "$BLUEPRINT_UNITREE_UNIFOLM_SOURCE_ROOT" \\
  && python3 /opt/blueprint/patch_unitree_unifolm_attention.py \\
  && if [[ "$INSTALL_FLASH_ATTN" == "true" ]]; then \\
       python3 -m pip install "flash-attn=={flash_attn_version}" --no-build-isolation; \\
     else \\
       echo "BLUEPRINT_UNITREE_UNIFOLM_FLASH_ATTN_INSTALL_SKIPPED attention=${{BLUEPRINT_UNITREE_UNIFOLM_VLA_ATTENTION_IMPLEMENTATION}}"; \\
     fi

RUN python3 /opt/blueprint/unitree_unifolm_image_healthcheck.py --build-time

WORKDIR /workspace
CMD ["bash", "-lc", "sleep infinity"]
"""


def build_unitree_unifolm_gpu_image_context(
    *,
    job_dir: Path | None = None,
    image_ref: str | None = None,
    base_image: str = DEFAULT_BASE_IMAGE,
    platform: str = DEFAULT_PLATFORM,
    torch_index_url: str = DEFAULT_TORCH_INDEX_URL,
    torch_version: str = DEFAULT_TORCH_VERSION,
    torchvision_version: str = DEFAULT_TORCHVISION_VERSION,
    flash_attn_version: str = DEFAULT_FLASH_ATTN_VERSION,
    install_flash_attn: bool = DEFAULT_INSTALL_FLASH_ATTN,
    attention_implementation: str = DEFAULT_ATTENTION_IMPLEMENTATION,
    unitree_source_url: str = DEFAULT_UNITREE_SOURCE_URL,
    unitree_source_ref: str = DEFAULT_UNITREE_SOURCE_REF,
    lerobot_ref: str = DEFAULT_LEROBOT_REF,
    dependency_profile: str = DEFAULT_DEPENDENCY_PROFILE,
    generated_at: str | None = None,
) -> dict[str, Any]:
    root = _repo_root()
    generated = generated_at or utc_now_iso()
    output = Path(
        job_dir or root / "robot_eval_jobs" / f"unitree_unifolm_gpu_image_{_timestamp()}"
    ).expanduser().resolve()
    ensure_dir(output)
    configured_image_ref = _string(image_ref) or _string(os.getenv(IMAGE_REF_ENV))
    normalized_dependency_profile = _string(dependency_profile).lower() or DEFAULT_DEPENDENCY_PROFILE
    dependency_profile_blockers: list[str] = []
    if normalized_dependency_profile not in SUPPORTED_DEPENDENCY_PROFILES:
        dependency_profile_blockers.append(
            "blocked_unitree_unifolm_dependency_profile_unsupported"
        )
        normalized_dependency_profile = DEFAULT_DEPENDENCY_PROFILE
    dockerfile_path = output / DEFAULT_CONTEXT_FILENAME
    requirements_path = output / "requirements_blueprint_unitree_unifolm.txt"
    healthcheck_path = output / "unitree_unifolm_image_healthcheck.py"
    attention_patch_path = output / "patch_unitree_unifolm_attention.py"
    launcher_path = output / "run_unitree_unifolm_vla_server.sh"
    policy_once_launcher_path = output / "run_unitree_unifolm_vla_policy_once.sh"
    dockerfile_path.write_text(
        dockerfile_text(
            base_image=base_image,
            platform=platform,
            torch_index_url=torch_index_url,
            torch_version=torch_version,
            torchvision_version=torchvision_version,
            flash_attn_version=flash_attn_version,
            install_flash_attn=install_flash_attn,
            attention_implementation=attention_implementation,
            unitree_source_url=unitree_source_url,
            unitree_source_ref=unitree_source_ref,
            lerobot_ref=lerobot_ref,
            dependency_profile=normalized_dependency_profile,
        ),
        encoding="utf-8",
    )
    requirements_path.write_text(
        requirements_text(normalized_dependency_profile),
        encoding="utf-8",
    )
    healthcheck_path.write_text(image_healthcheck_text(), encoding="utf-8")
    attention_patch_path.write_text(attention_patch_text(), encoding="utf-8")
    launcher_path.write_text(server_launcher_text(), encoding="utf-8")
    policy_once_launcher_path.write_text(policy_once_launcher_text(), encoding="utf-8")
    for script_path in (healthcheck_path, launcher_path, policy_once_launcher_path):
        script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR)

    build_command_path = output / "build_image.sh"
    push_command_path = output / "push_image.sh"
    run_healthcheck_command_path = output / "run_image_healthcheck.sh"
    build_command = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        f"docker build --platform {platform} -f \"$SCRIPT_DIR/{DEFAULT_CONTEXT_FILENAME}\" "
        f"-t \"${{{IMAGE_REF_ENV}}}\" \"$SCRIPT_DIR\"\n"
    )
    push_command = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"docker push \"${{{IMAGE_REF_ENV}}}\"\n"
    )
    run_healthcheck_command = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"docker run --rm --gpus all \"${{{IMAGE_REF_ENV}}}\" "
        "python3 /opt/blueprint/unitree_unifolm_image_healthcheck.py --require-cuda\n"
    )
    build_command_path.write_text(build_command, encoding="utf-8")
    push_command_path.write_text(push_command, encoding="utf-8")
    run_healthcheck_command_path.write_text(run_healthcheck_command, encoding="utf-8")
    for path in (build_command_path, push_command_path, run_healthcheck_command_path):
        path.chmod(0o755)

    blockers: list[str] = list(dependency_profile_blockers)
    if not configured_image_ref:
        blockers.append(f"missing_env_{IMAGE_REF_ENV}")
    elif not _image_ref_is_versioned(configured_image_ref):
        blockers.append("blocked_unitree_unifolm_gpu_image_ref_not_versioned")

    docker_auth = {
        "docker_username_file": _secret_file_status(
            "DOCKER_USERNAME_FILE",
            "~/.blueprint-secrets/docker_username",
        ),
        "docker_pat_file": _secret_file_status("DOCKER_PAT_FILE", "~/.blueprint-secrets/docker_pat"),
        "registry_auth_secret_values_written": False,
        "registry_auth_secret_hashes_written": False,
    }
    manifest = {
        "schema_version": UNITREE_UNIFOLM_GPU_IMAGE_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "ready_for_image_build" if not blockers else "context_written_blocked",
        "job_dir": str(output),
        "image_ref_env": IMAGE_REF_ENV,
        "configured_image_ref_present": bool(configured_image_ref),
        "configured_image_ref": configured_image_ref or None,
        "configured_image_ref_is_versioned": _image_ref_is_versioned(configured_image_ref),
        "base_image": base_image,
        "platform": platform,
        "torch_index_url": torch_index_url,
        "torch_version": torch_version,
        "torch_cuda_wheel_family": "cu124",
        "torchvision_version": torchvision_version,
        "flash_attn_version": flash_attn_version,
        "install_flash_attn": install_flash_attn,
        "attention_implementation": attention_implementation,
        "unitree_source_url": unitree_source_url,
        "unitree_source_ref": unitree_source_ref,
        "lerobot_ref": lerobot_ref,
        "dependency_profile": normalized_dependency_profile,
        "dependency_profile_supported": not dependency_profile_blockers,
        "dependency_profile_excluded_training_packages": (
            [
                "albumentations==1.4.18",
                "datasets==3.6.0",
                "deepspeed==0.16.9",
                "fastparquet==2024.11.0",
                "matplotlib",
                "mujoco==3.3.5",
                "pyarrow==15.0.1",
                "tensorflow_datasets==4.9.3",
                "tensorflow_graphics==2021.12.3",
                "wandb",
            ]
            if normalized_dependency_profile == "inference"
            else []
        ),
        "runtime_contract": {
            "sets_BLUEPRINT_UNITREE_UNIFOLM_SOURCE_ROOT": "/opt/unifolm-vla",
            "server_launcher": "/usr/local/bin/run_unitree_unifolm_vla_server",
            "single_action_policy_command": "/usr/local/bin/run_unitree_unifolm_vla_policy_once",
            "server_port_env": "BLUEPRINT_UNITREE_UNIFOLM_VLA_SERVER_PORT",
            "model_checkpoint_baked_into_image": False,
            "vlm_checkpoint_baked_into_image": False,
            "raw_credentials_baked_into_image": False,
            "provider_bundle_still_supplies_observation_inputs": True,
            "unitree_qwen_attention_patch_applied": True,
            "dependency_profile": normalized_dependency_profile,
            "inference_profile_keeps_tensorflow_cpu_for_server_preprocessing": (
                normalized_dependency_profile == "inference"
            ),
        },
        "artifact_paths": {
            "dockerfile": str(dockerfile_path),
            "requirements": str(requirements_path),
            "image_healthcheck": str(healthcheck_path),
            "attention_patch": str(attention_patch_path),
            "server_launcher": str(launcher_path),
            "policy_once_launcher": str(policy_once_launcher_path),
            "build_command": str(build_command_path),
            "push_command": str(push_command_path),
            "run_healthcheck_command": str(run_healthcheck_command_path),
            "manifest": str(output / "unitree_unifolm_gpu_image_manifest.json"),
        },
        "commands": {
            "build": f"{build_command_path}",
            "push": f"{push_command_path}",
            "run_gpu_healthcheck": f"{run_healthcheck_command_path}",
            "vast_usage": (
                "python -m blueprint_pipeline.vast_provider_adapter ... "
                f"--provider-bundle-kind unitree_unifolm --public-image \"${{{IMAGE_REF_ENV}}}\""
            ),
        },
        "registry_auth": docker_auth,
        "blockers": blockers,
        "truth_boundary": {
            "image_build_is_not_model_execution": True,
            "image_push_is_not_unitree_policy_execution": True,
            "no_raw_tokens_or_hashes_written": True,
            "model_checkpoints_not_baked": True,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
        },
        "raw_secret_values_recorded": False,
        "secret_hashes_recorded": False,
    }
    write_json(output / "unitree_unifolm_gpu_image_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write a reusable Unitree UnifoLM VLA CUDA GPU image build context."
    )
    parser.add_argument("--job-dir")
    parser.add_argument("--image-ref")
    parser.add_argument("--base-image", default=DEFAULT_BASE_IMAGE)
    parser.add_argument("--platform", default=DEFAULT_PLATFORM)
    parser.add_argument("--torch-index-url", default=DEFAULT_TORCH_INDEX_URL)
    parser.add_argument("--torch-version", default=DEFAULT_TORCH_VERSION)
    parser.add_argument("--torchvision-version", default=DEFAULT_TORCHVISION_VERSION)
    parser.add_argument("--flash-attn-version", default=DEFAULT_FLASH_ATTN_VERSION)
    parser.add_argument(
        "--skip-flash-attn",
        action="store_true",
        help="Write an sdpa/eager fallback image context without compiling flash-attn.",
    )
    parser.add_argument(
        "--attention-implementation",
        default=DEFAULT_ATTENTION_IMPLEMENTATION,
        help="Qwen attention implementation injected into the Unitree wrapper.",
    )
    parser.add_argument("--unitree-source-url", default=DEFAULT_UNITREE_SOURCE_URL)
    parser.add_argument("--unitree-source-ref", default=DEFAULT_UNITREE_SOURCE_REF)
    parser.add_argument("--lerobot-ref", default=DEFAULT_LEROBOT_REF)
    parser.add_argument(
        "--dependency-profile",
        choices=SUPPORTED_DEPENDENCY_PROFILES,
        default=DEFAULT_DEPENDENCY_PROFILE,
        help=(
            "Python dependency profile. Use 'inference' for the bounded provider "
            "proof image; 'full' preserves Unitree's broader training/data deps."
        ),
    )
    args = parser.parse_args(argv)
    manifest = build_unitree_unifolm_gpu_image_context(
        job_dir=Path(args.job_dir) if args.job_dir else None,
        image_ref=args.image_ref,
        base_image=args.base_image,
        platform=args.platform,
        torch_index_url=args.torch_index_url,
        torch_version=args.torch_version,
        torchvision_version=args.torchvision_version,
        flash_attn_version=args.flash_attn_version,
        install_flash_attn=not args.skip_flash_attn,
        attention_implementation=args.attention_implementation,
        unitree_source_url=args.unitree_source_url,
        unitree_source_ref=args.unitree_source_ref,
        lerobot_ref=args.lerobot_ref,
        dependency_profile=args.dependency_profile,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
