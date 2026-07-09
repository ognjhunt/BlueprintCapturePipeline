"""Bootstrap package for real WAM/VLA model runtime wiring.

This module does not fake model output. It writes the concrete command,
checkpoint, auth, and provider artifacts needed before
``oscar_cosmos_wam_evaluator`` may truthfully run a learned model.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .oscar_official_release import (
    OFFICIAL_OSCAR_HF_REPO,
    OFFICIAL_OSCAR_HF_REVISION,
    OFFICIAL_OSCAR_MODEL_URL,
    OFFICIAL_OSCAR_POLICY_ROLLOUT_DATASET_URL,
    OFFICIAL_OSCAR_PROJECT_PAGE_URL,
    OFFICIAL_OSCAR_SOURCE_COMMIT,
    OFFICIAL_OSCAR_SOURCE_URL,
    OFFICIAL_OSCAR_SOURCE_WEB_URL,
    official_release_contract,
)
from .secret_artifact_policy import (
    redacted_secret_file_status_from_env,
    secret_path_disclosure_policy,
)


BOOTSTRAP_SCHEMA_VERSION = "wam_model_runtime_bootstrap.v1"
DEFAULT_CANDIDATE = "oscar_wam"
BOOTSTRAP_CANDIDATES = {
    "oscar_wam": {
        "model_repo_id": OFFICIAL_OSCAR_HF_REPO,
        "model_revision": OFFICIAL_OSCAR_HF_REVISION,
        "model_url": OFFICIAL_OSCAR_MODEL_URL,
        "source_repo_url": OFFICIAL_OSCAR_SOURCE_WEB_URL,
        "source_repo_git_url": OFFICIAL_OSCAR_SOURCE_URL,
        "source_repo_commit": OFFICIAL_OSCAR_SOURCE_COMMIT,
        "project_url": OFFICIAL_OSCAR_PROJECT_PAGE_URL,
        "paper_url": "https://arxiv.org/abs/2606.04463",
        "policy_rollout_dataset_url": OFFICIAL_OSCAR_POLICY_ROLLOUT_DATASET_URL,
        "policy_rollout_dataset_is_reference_data_not_runtime": True,
        "checkpoint_env": "BLUEPRINT_OSCAR_WAM_CHECKPOINT",
        "command_env": "BLUEPRINT_OSCAR_WAM_COMMAND",
        "source_root_env": "BLUEPRINT_OSCAR_WAM_SOURCE_ROOT",
        "checkpoint_allow_patterns": ["model/**", "case_map.json", "README.md"],
        "minimum_vram_gb": 24,
        "expected_checkpoint_bytes": 4_245_460_687,
        "runtime_kind": "action_conditioned_world_model_rollout_generator",
        "claim_boundary": (
            "OSCAR requires the reviewed oscar-public source commit plus pinned "
            "OSCAR-2B checkpoint revision; a checkpoint download alone is not "
            "WAM execution proof."
        ),
    },
    "cosmos_wam": {
        "model_repo_id": "nvidia/Cosmos-Predict2.5-2B",
        "model_url": "https://huggingface.co/nvidia/Cosmos-Predict2.5-2B",
        "source_repo_url": "https://github.com/nvidia-cosmos/cosmos-predict2.5",
        "project_url": "https://github.com/nvidia-cosmos/cosmos-predict2.5",
        "paper_url": "https://research.nvidia.com/labs/cosmos-lab/cosmos-predict2.5/",
        "checkpoint_env": "BLUEPRINT_COSMOS_WAM_CHECKPOINT",
        "command_env": "BLUEPRINT_COSMOS_WAM_COMMAND",
        "source_root_env": "BLUEPRINT_COSMOS_WAM_SOURCE_ROOT",
        "checkpoint_allow_patterns": [
            "robot/action-cond/**",
            "base/distilled/**",
            "README.md",
        ],
        "minimum_vram_gb": 24,
        "expected_checkpoint_bytes": 75_069_894_477,
        "runtime_kind": "world_video_rollout_or_review_substrate",
        "claim_boundary": "Cosmos source/checkpoints require a compatible NVIDIA runtime adapter before Blueprint can claim action-conditioned WAM rollouts.",
    },
    "openvla_policy": {
        "model_repo_id": "openvla/openvla-7b",
        "model_url": "https://huggingface.co/openvla/openvla-7b",
        "source_repo_url": "https://github.com/openvla/openvla",
        "project_url": "https://openvla.github.io/",
        "paper_url": "https://huggingface.co/papers/2406.09246",
        "checkpoint_env": "BLUEPRINT_OPENVLA_POLICY_CHECKPOINT",
        "command_env": "BLUEPRINT_OPENVLA_POLICY_COMMAND",
        "source_root_env": "BLUEPRINT_OPENVLA_POLICY_SOURCE_ROOT",
        "checkpoint_allow_patterns": ["*.safetensors", "*.json", "*.py", "tokenizer.*"],
        "minimum_vram_gb": 16,
        "expected_checkpoint_bytes": 15_085_153_882,
        "runtime_kind": "vla_policy_endpoint_candidate",
        "claim_boundary": "OpenVLA predicts robot actions from images/instructions; it is not a generated-video WAM unless paired with a WAM evaluator.",
    },
}
GPU_PROVIDER_IDS = ("runpod", "vast", "digitalocean_gpu")
WAM_PROVIDER_IMAGE_REF_ENV = "BLUEPRINT_WAM_PROVIDER_IMAGE_REF"
DOCKER_PAT_FILE_ENV = "DOCKER_PAT_FILE"
DOCKER_USERNAME_FILE_ENV = "DOCKER_USERNAME_FILE"
CHECKPOINT_FILE_SUFFIXES = (
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".bin",
    ".gguf",
    ".distcp",
)
CHECKPOINT_MINIMUM_READY_BYTES_RATIO = 0.01
CHECKPOINT_MINIMUM_READY_BYTES_FLOOR = 1024
CHECKPOINT_MINIMUM_READY_BYTES_CAP = 256 * 1024 * 1024


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _secret_file_status(env_name: str, default_path: str) -> dict[str, Any]:
    status = redacted_secret_file_status_from_env(
        env_name,
        default_path,
        raw_secret_field="raw_secret_written_to_artifacts",
    )
    status["secret_hash_written_to_artifacts"] = False
    return status


def _path_status(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"configured": False, "path": None, "exists": False, "is_dir": False}
    expanded = Path(path).expanduser()
    return {
        "configured": True,
        "path": str(expanded),
        "exists": expanded.exists(),
        "is_dir": expanded.is_dir(),
    }


def _candidate_source_ready(candidate_id: str, path: Path) -> bool:
    if candidate_id == "oscar_wam":
        return (path / "inference" / "inference_oscar.py").is_file()
    return path.exists()


def _checkpoint_minimum_ready_bytes(expected_bytes: int) -> int:
    if expected_bytes <= 0:
        return CHECKPOINT_MINIMUM_READY_BYTES_FLOOR
    return max(
        CHECKPOINT_MINIMUM_READY_BYTES_FLOOR,
        min(
            int(expected_bytes * CHECKPOINT_MINIMUM_READY_BYTES_RATIO),
            CHECKPOINT_MINIMUM_READY_BYTES_CAP,
        ),
    )


def _checkpoint_inventory(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "file_count": 0,
            "checkpoint_file_count": 0,
            "total_bytes": 0,
            "checkpoint_total_bytes": 0,
            "largest_checkpoint_files": [],
        }
    files = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
    checkpoint_files = [
        item for item in files if item.suffix.lower() in CHECKPOINT_FILE_SUFFIXES
    ]
    largest = sorted(
        (
            {
                "relative_path": str(item.relative_to(path.parent if path.is_file() else path)),
                "size": item.stat().st_size,
            }
            for item in checkpoint_files
        ),
        key=lambda row: int(row["size"]),
        reverse=True,
    )[:12]
    return {
        "file_count": len(files),
        "checkpoint_file_count": len(checkpoint_files),
        "total_bytes": sum(item.stat().st_size for item in files),
        "checkpoint_total_bytes": sum(item.stat().st_size for item in checkpoint_files),
        "largest_checkpoint_files": largest,
    }


def _candidate_checkpoint_ready(path: Path, *, minimum_bytes: int | None = None) -> bool:
    if path.is_file():
        if path.suffix.lower() not in CHECKPOINT_FILE_SUFFIXES:
            return False
        return minimum_bytes is None or path.stat().st_size >= minimum_bytes
    if not path.is_dir():
        return False
    inventory = _checkpoint_inventory(path)
    if not inventory["checkpoint_file_count"]:
        return False
    return minimum_bytes is None or int(inventory["checkpoint_total_bytes"]) >= minimum_bytes


def _newest_ready_path(paths: Sequence[Path], *, ready: Callable[[Path], bool]) -> Path | None:
    ready_paths = [path for path in paths if ready(path)]
    if not ready_paths:
        return None
    return max(ready_paths, key=lambda path: path.stat().st_mtime)


def _runtime_roots(job_root: Path, candidate_id: str, leaf: str) -> list[Path]:
    return [
        path
        for path in sorted(job_root.glob(f"*/runtime_sources/{candidate_id}/{leaf}"))
        if path.exists()
    ]


def _discover_runtime_path(
    *,
    candidate_id: str,
    candidate: Mapping[str, Any],
    job_root: Path,
    explicit_path: Path | None,
    env_name: str,
    leaf: str,
    default_path: Path,
) -> tuple[Path, dict[str, Any]]:
    env_value = _string(os.getenv(env_name))
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        source = "explicit_argument"
    elif env_value:
        path = Path(env_value).expanduser().resolve()
        source = env_name
    elif leaf == "source":
        path = (
            _newest_ready_path(
                _runtime_roots(job_root, candidate_id, leaf),
                ready=lambda item: _candidate_source_ready(candidate_id, item),
            )
            or default_path
        )
        source = "auto_discovered_local_runtime" if path != default_path else "default_output_path"
    else:
        path = (
            _newest_ready_path(
                _runtime_roots(job_root, candidate_id, leaf),
                ready=_candidate_checkpoint_ready,
            )
            or default_path
        )
        source = "auto_discovered_local_runtime" if path != default_path else "default_output_path"
    return path, {
        "candidate_id": candidate_id,
        "path": str(path),
        "selection_source": source,
        "env_name": env_name,
        "model_repo_id": candidate.get("model_repo_id"),
    }


def _command_available(command: str | None) -> bool:
    if not command:
        return False
    try:
        import shlex

        parts = shlex.split(command)
    except ValueError:
        return False
    if not parts:
        return False
    return bool(Path(parts[0]).expanduser().is_file() or shutil.which(parts[0]))


def _git_ls_remote_probe(url: str, *, timeout_seconds: float = 15.0) -> dict[str, Any]:
    result = subprocess.run(
        ["git", "ls-remote", url, "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    head = ""
    if result.returncode == 0:
        head = (result.stdout.strip().split() or [""])[0]
    return {
        "url": url,
        "status": "reachable" if result.returncode == 0 else "blocked",
        "head_sha": head or None,
        "returncode": result.returncode,
        "stderr_size_bytes": len(result.stderr or ""),
        "stderr_omitted": bool(result.stderr),
    }


def _hf_repo_metadata(repo_id: str) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, files_metadata=True)
    except Exception as exc:
        return {
            "repo_id": repo_id,
            "status": "blocked",
            "error_type": type(exc).__name__,
            "error": str(exc)[:300],
        }
    siblings = list(info.siblings or [])
    total_bytes = sum((getattr(item, "size", None) or 0) for item in siblings)
    largest = sorted(
        [
            {
                "rfilename": item.rfilename,
                "size": getattr(item, "size", None) or 0,
            }
            for item in siblings
        ],
        key=lambda row: int(row["size"]),
        reverse=True,
    )[:12]
    return {
        "repo_id": repo_id,
        "status": "completed",
        "file_count": len(siblings),
        "total_bytes": total_bytes,
        "largest_files": largest,
    }


def _disk_status(path: Path, *, required_bytes: int) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "free_bytes": int(usage.free),
        "required_bytes": int(required_bytes),
        "free_gib": round(usage.free / (1024**3), 3),
        "required_gib": round(required_bytes / (1024**3), 3),
        "has_required_space": usage.free > required_bytes,
    }


def _provider_gate_status() -> dict[str, Any]:
    providers = [
        {
            "provider_id": "runpod",
            "api_key_file": _secret_file_status("RUNPOD_API_KEY_FILE", "~/.blueprint-secrets/runpod_api_key"),
            "api_gate_env": "BLUEPRINT_ALLOW_RUNPOD_API_CALLS",
            "api_gate_enabled": _env_truthy("BLUEPRINT_ALLOW_RUNPOD_API_CALLS"),
            "launch_gate_env": "BLUEPRINT_ALLOW_GPU_PROVIDER_LAUNCH",
            "launch_gate_enabled": _env_truthy("BLUEPRINT_ALLOW_GPU_PROVIDER_LAUNCH"),
            "adapter_available": True,
        },
        {
            "provider_id": "vast",
            "api_key_file": _secret_file_status("VAST_API_KEY_FILE", "~/.blueprint-secrets/vast_api_key"),
            "api_gate_env": "BLUEPRINT_ALLOW_VAST_API_CALLS",
            "api_gate_enabled": _env_truthy("BLUEPRINT_ALLOW_VAST_API_CALLS"),
            "launch_gate_env": "BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH",
            "launch_gate_enabled": _env_truthy("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH"),
            "adapter_available": True,
        },
        {
            "provider_id": "digitalocean_gpu",
            "api_key_file": _secret_file_status(
                "DIGITALOCEAN_API_TOKEN_FILE",
                "~/.blueprint-secrets/digitalocean_api_token",
            ),
            "api_gate_env": "BLUEPRINT_ALLOW_DIGITALOCEAN_API_CALLS",
            "api_gate_enabled": _env_truthy("BLUEPRINT_ALLOW_DIGITALOCEAN_API_CALLS"),
            "launch_gate_env": "BLUEPRINT_ALLOW_DIGITALOCEAN_GPU_DROPLET_LAUNCH",
            "launch_gate_enabled": _env_truthy("BLUEPRINT_ALLOW_DIGITALOCEAN_GPU_DROPLET_LAUNCH"),
            "adapter_available": False,
            "adapter_blocker": "blocked_no_digitalocean_gpu_adapter_implemented",
            "object_store_note": (
                "DigitalOcean Spaces can be used as S3-compatible bundle/output staging; "
                "a CPU Droplet is not sufficient for OSCAR/Cosmos/OpenVLA inference."
            ),
        },
    ]
    for row in providers:
        row["status"] = (
            "ready_for_gated_launch"
            if row["api_key_file"]["present"]
            and row["api_gate_enabled"]
            and row["launch_gate_enabled"]
            and row["adapter_available"]
            else "blocked"
        )
        row["blockers"] = [
            blocker
            for blocker in [
                f"missing_file_based_secret_{row['api_key_file']['env_name']}"
                if not row["api_key_file"]["present"]
                else None,
                f"missing_env_{row['api_gate_env']}" if not row["api_gate_enabled"] else None,
                f"missing_env_{row['launch_gate_env']}"
                if not row["launch_gate_enabled"]
                else None,
                row.get("adapter_blocker") if not row["adapter_available"] else None,
            ]
            if blocker
        ]
    return {
        "schema_version": "wam_model_provider_gate_status.v1",
        "status": "ready_for_gated_launch"
        if any(row["status"] == "ready_for_gated_launch" for row in providers)
        else "blocked",
        "providers": providers,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "secret_artifact_policy": secret_path_disclosure_policy(),
    }


def _image_ref_is_versioned(image_ref: str) -> bool:
    if not image_ref or image_ref.endswith(":latest"):
        return False
    last = image_ref.rsplit("/", maxsplit=1)[-1]
    return ":" in last or "@" in last


def _provider_image_plan(*, candidate_id: str, output_dir: Path) -> dict[str, Any]:
    image_ref = _string(os.getenv(WAM_PROVIDER_IMAGE_REF_ENV))
    docker_pat = _secret_file_status(DOCKER_PAT_FILE_ENV, "~/.blueprint-secrets/docker_pat")
    docker_username = _secret_file_status(
        DOCKER_USERNAME_FILE_ENV,
        "~/.blueprint-secrets/docker_username",
    )
    dockerfile_path = output_dir / "Dockerfile.wam-provider"
    blockers: list[str] = []
    if not image_ref:
        blockers.append(f"missing_env_{WAM_PROVIDER_IMAGE_REF_ENV}")
    elif not _image_ref_is_versioned(image_ref):
        blockers.append("blocked_wam_provider_image_ref_not_versioned")
    if image_ref and not docker_pat["present"]:
        blockers.append(f"missing_file_based_secret_{DOCKER_PAT_FILE_ENV}")
    if image_ref and not docker_username["present"]:
        blockers.append(f"missing_file_based_secret_{DOCKER_USERNAME_FILE_ENV}")
    return {
        "schema_version": "wam_provider_reusable_image_plan.v1",
        "status": "ready_for_manual_image_build_and_push" if not blockers else "blocked",
        "candidate_id": candidate_id,
        "daily_reusable_image_recommended": True,
        "image_ref_env": WAM_PROVIDER_IMAGE_REF_ENV,
        "configured_image_ref_present": bool(image_ref),
        "configured_image_ref": image_ref or None,
        "configured_image_ref_is_versioned": _image_ref_is_versioned(image_ref),
        "dockerfile_path": str(dockerfile_path),
        "registry_auth": {
            "docker_username_file": docker_username,
            "docker_pat_file": docker_pat,
            "chat_pasted_tokens_must_be_rotated_before_use": True,
            "raw_secret_values_recorded": False,
            "secret_hashes_recorded": False,
            "secret_artifact_policy": secret_path_disclosure_policy(),
        },
        "commands": {
            "build": (
                f"docker build -f {dockerfile_path} -t \"$"
                f"{WAM_PROVIDER_IMAGE_REF_ENV}\" ."
            ),
            "login": (
                f"cat \"${{{DOCKER_PAT_FILE_ENV}}}\" | docker login "
                f"--username \"$(cat \"${{{DOCKER_USERNAME_FILE_ENV}}}\")\" --password-stdin"
            ),
            "push": f"docker push \"${WAM_PROVIDER_IMAGE_REF_ENV}\"",
        },
        "provider_usage": {
            "runpod_image_env": "BLUEPRINT_RUNPOD_WAM_IMAGE_NAME",
            "vast_image_arg": "--public-image",
            "why": "Provider pulls a fixed image daily instead of rebuilding dependencies inside each paid GPU job.",
        },
        "blockers": blockers,
        "claim_boundary": {
            "image_build_is_not_model_execution": True,
            "image_push_is_not_generated_rollout": True,
            "cpu_droplet_is_not_gpu_wam_runtime": True,
            "raw_credentials_written_to_artifacts": False,
        },
    }


def _provider_dockerfile(candidate_id: str) -> str:
    if candidate_id == "oscar_wam":
        from .oscar_wam_gpu_image import dockerfile_text

        return dockerfile_text()
    return f"""# Blueprint reusable WAM provider image for {candidate_id}.
# This image contains the Blueprint adapter boundary. Mount or bake real model
# source/checkpoints separately according to wam_model_runtime_env.sh.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \\
    PIP_NO_CACHE_DIR=1 \\
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \\
    ca-certificates \\
    ffmpeg \\
    git \\
    python3 \\
    python3-pip \\
    python3-venv \\
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/BlueprintCapturePipeline
COPY pyproject.toml README.md ./
COPY src ./src
RUN python3 -m pip install --upgrade pip \\
  && python3 -m pip install -e . \\
  && python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 \\
  && python3 -m pip install opencv-python-headless imageio imageio-ffmpeg decord einops diffusers transformers accelerate safetensors huggingface_hub

ENTRYPOINT ["python3", "-m", "blueprint_pipeline.oscar_wam_command_adapter"]
"""


def _env_template(
    *,
    candidate_id: str,
    candidate: Mapping[str, Any],
    output_dir: Path,
    source_root: Path,
    checkpoint_root: Path,
    adapter_command: str,
) -> str:
    lines = [
        "# Source this file after the real runtime and checkpoint exist.",
        "# It contains no raw secrets.",
        "export BLUEPRINT_ALLOW_LOCAL_WAM_MODEL=true",
        f"export {candidate['source_root_env']}=\"{source_root}\"",
        f"export {candidate['checkpoint_env']}=\"{checkpoint_root}\"",
        f"export {candidate['command_env']}=\"{adapter_command}\"",
    ]
    if candidate_id == "oscar_wam":
        lines.extend(
            [
                f"export BLUEPRINT_OSCAR_WAM_SOURCE_URL=\"{candidate['source_repo_git_url']}\"",
                f"export BLUEPRINT_OSCAR_WAM_SOURCE_REF=\"{candidate['source_repo_commit']}\"",
                f"export BLUEPRINT_OSCAR_WAM_HF_REPO=\"{candidate['model_repo_id']}\"",
                f"export BLUEPRINT_OSCAR_WAM_HF_REVISION=\"{candidate['model_revision']}\"",
            ]
        )
    lines.extend(
        [
            "",
            "# Verification command:",
            (
                "# PYTHONDONTWRITEBYTECODE=1 python -m blueprint_pipeline.oscar_cosmos_wam_evaluator "
                "--input-job-dir <mujoco_endpoint_eval_job_dir> "
                f"--job-dir \"{output_dir / 'verify_wam_model_run'}\" "
                f"--model-candidate {candidate_id} --allow-wam-model-run"
            ),
            "",
            "# Optional reusable GPU-provider image, set only after pushing a versioned image:",
            f"# export {WAM_PROVIDER_IMAGE_REF_ENV}=registry.example/blueprint/{candidate_id}-provider:YYYYMMDD",
            "",
        ]
    )
    return "\n".join(lines)


def build_bootstrap_package(
    *,
    candidate_id: str = DEFAULT_CANDIDATE,
    output_dir: Path | None = None,
    job_root: Path | None = None,
    source_root: Path | None = None,
    checkpoint_root: Path | None = None,
    adapter_command: str | None = None,
    generated_at: str | None = None,
    refresh_hf_metadata: bool = False,
    verify_source_repo: bool = False,
) -> dict[str, Any]:
    if candidate_id not in BOOTSTRAP_CANDIDATES:
        raise ValueError(f"candidate_id must be one of {', '.join(sorted(BOOTSTRAP_CANDIDATES))}")
    generated = generated_at or utc_now_iso()
    root = Path(job_root or (_repo_root() / "robot_eval_jobs"))
    output = Path(
        output_dir or (root / f"wam_model_runtime_bootstrap_{candidate_id}_{_timestamp()}")
    ).resolve()
    ensure_dir(output)
    candidate = dict(BOOTSTRAP_CANDIDATES[candidate_id])
    default_source = output / "runtime_sources" / candidate_id / "source"
    default_checkpoint = output / "runtime_sources" / candidate_id / "checkpoint"
    source, source_selection = _discover_runtime_path(
        candidate_id=candidate_id,
        candidate=candidate,
        job_root=root,
        explicit_path=source_root,
        env_name=_string(candidate["source_root_env"]),
        leaf="source",
        default_path=default_source,
    )
    checkpoint, checkpoint_selection = _discover_runtime_path(
        candidate_id=candidate_id,
        candidate=candidate,
        job_root=root,
        explicit_path=checkpoint_root,
        env_name=_string(candidate["checkpoint_env"]),
        leaf="checkpoint",
        default_path=default_checkpoint,
    )
    default_adapter_commands = {
        "oscar_wam": "/usr/bin/env python -m blueprint_pipeline.oscar_wam_command_adapter",
        "cosmos_wam": "/usr/bin/env python -m blueprint_pipeline.oscar_cosmos_wam_command_adapter",
        "openvla_policy": "/usr/bin/env python -m blueprint_pipeline.openvla_policy_command_adapter",
    }
    adapter = adapter_command or default_adapter_commands.get(
        candidate_id,
        f"<set after installing {candidate_id} inference adapter; must write Blueprint WAM JSON>",
    )
    hf_metadata = (
        _hf_repo_metadata(_string(candidate["model_repo_id"]))
        if refresh_hf_metadata
        else {
            "repo_id": candidate["model_repo_id"],
            "status": "not_refreshed",
            "expected_checkpoint_bytes": candidate["expected_checkpoint_bytes"],
        }
    )
    source_probe = (
        _git_ls_remote_probe(
            _string(candidate.get("source_repo_git_url") or candidate["source_repo_url"])
        )
        if verify_source_repo
        else {
            "url": candidate.get("source_repo_git_url") or candidate["source_repo_url"],
            "status": "not_checked",
        }
    )
    disk = _disk_status(output.parent, required_bytes=int(candidate["expected_checkpoint_bytes"]))
    checkpoint_status = _path_status(checkpoint)
    source_status = _path_status(source)
    checkpoint_inventory = _checkpoint_inventory(checkpoint)
    minimum_checkpoint_bytes = _checkpoint_minimum_ready_bytes(
        int(candidate["expected_checkpoint_bytes"])
    )
    checkpoint_contains_weight_file = bool(checkpoint_inventory["checkpoint_file_count"])
    checkpoint_bytes_ready = (
        int(checkpoint_inventory["checkpoint_total_bytes"]) >= minimum_checkpoint_bytes
    )
    checkpoint_ready = bool(
        checkpoint_status["exists"]
        and checkpoint_contains_weight_file
        and checkpoint_bytes_ready
    )
    checkpoint_status.update(
        {
            "ready": checkpoint_ready,
            "contains_checkpoint_weight_file": checkpoint_contains_weight_file,
            "checkpoint_file_count": checkpoint_inventory["checkpoint_file_count"],
            "file_count": checkpoint_inventory["file_count"],
            "total_bytes": checkpoint_inventory["total_bytes"],
            "checkpoint_total_bytes": checkpoint_inventory["checkpoint_total_bytes"],
            "largest_checkpoint_files": checkpoint_inventory["largest_checkpoint_files"],
            "expected_checkpoint_bytes": candidate["expected_checkpoint_bytes"],
            "minimum_ready_checkpoint_bytes": minimum_checkpoint_bytes,
            "checkpoint_bytes_ready": checkpoint_bytes_ready,
        }
    )
    command_ready = _command_available(adapter)
    blockers: list[str] = []
    if not source_status["exists"]:
        blockers.append("blocked_missing_model_source_runtime")
    if not checkpoint_status["exists"]:
        blockers.append("blocked_missing_model_checkpoint")
    elif not checkpoint_contains_weight_file:
        blockers.append("blocked_incomplete_or_unusable_model_checkpoint")
    elif not checkpoint_bytes_ready:
        blockers.append("blocked_checkpoint_bytes_below_minimum_ready_threshold")
    if not command_ready:
        blockers.append("blocked_missing_runnable_adapter_command")
    if not checkpoint_ready and not disk["has_required_space"]:
        blockers.append("blocked_insufficient_disk_for_checkpoint_download")
    status = "ready_for_wam_evaluator_configuration" if not blockers else "blocked"
    model_revision = _string(candidate.get("model_revision"))
    revision_kwarg = f", revision={model_revision!r}" if model_revision else ""
    download_plan = {
        "schema_version": "wam_model_checkpoint_download_plan.v1",
        "generated_at": generated,
        "candidate_id": candidate_id,
        "model_repo_id": candidate["model_repo_id"],
        "model_revision": model_revision or None,
        "model_url": candidate["model_url"],
        "target_checkpoint_root": str(checkpoint),
        "allow_patterns": list(candidate["checkpoint_allow_patterns"]),
        "expected_checkpoint_bytes": candidate["expected_checkpoint_bytes"],
        "disk_status": disk,
        "checkpoint_status": checkpoint_status,
        "download_not_started_by_this_artifact": True,
        "download_command": (
            "python - <<'PY'\n"
            "from huggingface_hub import snapshot_download\n"
            f"snapshot_download(repo_id={candidate['model_repo_id']!r}{revision_kwarg}, local_dir={str(checkpoint)!r}, "
            f"allow_patterns={list(candidate['checkpoint_allow_patterns'])!r})\n"
            "PY"
        ),
    }
    adapter_contract = {
        "schema_version": "wam_model_adapter_command_contract.v1",
        "generated_at": generated,
        "candidate_id": candidate_id,
        "command_env": candidate["command_env"],
        "checkpoint_env": candidate["checkpoint_env"],
        "source_root_env": candidate["source_root_env"],
        "required_stdin_or_env_contract": {
            "BLUEPRINT_WAM_ROLLOUT_INPUT": "Path to rollout input JSON.",
            "BLUEPRINT_WAM_ROLLOUT_OUTPUT": "Path where adapter must write rollout JSON.",
            "BLUEPRINT_WAM_MODEL_CANDIDATE": candidate_id,
            "BLUEPRINT_WAM_MODEL_CHECKPOINT": "Configured checkpoint path.",
        },
        "required_output_json": {
            "rollouts": [
                {
                    "rollout_id": "string",
                    "scenario_eval_run_id": "string",
                    "generated_video_path": "path to generated video",
                    "model_rollout_confidence": "number in [0, 1]",
                }
            ]
        },
        "no_fixture_or_synthetic_rollout_allowed": True,
        "raw_credentials_written_to_artifacts": False,
    }
    provider_request = {
        "schema_version": "wam_model_provider_launch_request.v1",
        "generated_at": generated,
        "candidate_id": candidate_id,
        "status": "ready_for_provider_request_after_runtime_bundle_upload",
        "provider_options": list(GPU_PROVIDER_IDS),
        "minimum_vram_gb": candidate["minimum_vram_gb"],
        "provider_gate_status": _provider_gate_status(),
        "worker_command_contract": adapter_contract["required_stdin_or_env_contract"],
        "cost_controls": {
            "max_active_workers": 1,
            "scale_to_zero_required": True,
            "hard_timeout_minutes_recommended": 60,
        },
        "blockers_before_launch": [
            "package_source_runtime_and_adapter_command",
            "download_or_mount_checkpoint",
            "upload_runtime_bundle_to_provider_accessible_storage",
        ],
    }
    dockerfile_path = output / "Dockerfile.wam-provider"
    dockerfile_path.write_text(_provider_dockerfile(candidate_id), encoding="utf-8")
    provider_image_plan = _provider_image_plan(candidate_id=candidate_id, output_dir=output)
    provider_request["provider_image_plan"] = provider_image_plan
    official_oscar_release = (
        official_release_contract(
            source_url=_string(candidate.get("source_repo_git_url")),
            source_ref=_string(candidate.get("source_repo_commit")),
            hf_repo=_string(candidate.get("model_repo_id")),
            hf_revision=_string(candidate.get("model_revision")),
        )
        if candidate_id == "oscar_wam"
        else None
    )
    manifest = {
        "schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "generated_at": generated,
        "status": status,
        "candidate_id": candidate_id,
        "candidate": candidate,
        "source_status": source_status,
        "source_selection": source_selection,
        "source_repo_probe": source_probe,
        "official_oscar_release": official_oscar_release,
        "checkpoint_status": checkpoint_status,
        "checkpoint_selection": checkpoint_selection,
        "adapter_command": {
            "configured": bool(adapter),
            "value_redacted": "<configured>" if adapter else None,
            "available": command_ready,
        },
        "hf_model_metadata": hf_metadata,
        "disk_status": disk,
        "provider_gate_status": provider_request["provider_gate_status"],
        "blockers": blockers,
        "next_truthful_steps": [
            "Install or clone the real upstream inference runtime for the selected candidate.",
            "Download or mount the selected model checkpoint.",
            "Set the generated env template values to a runnable adapter command and checkpoint path.",
            "Run blueprint-run-oscar-cosmos-wam-evaluator with --allow-wam-model-run and verify generated rollout artifacts.",
        ],
        "claim_boundary": {
            "bootstrap_package_is_not_model_execution": True,
            "checkpoint_download_is_not_generated_rollout": True,
            "provider_gate_ready_is_not_gpu_model_run": True,
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        },
    }
    write_json(output / "wam_model_runtime_bootstrap_manifest.json", manifest)
    write_json(output / "wam_model_checkpoint_download_plan.json", download_plan)
    write_json(output / "wam_model_adapter_command_contract.json", adapter_contract)
    write_json(output / "wam_model_provider_launch_request.json", provider_request)
    write_json(output / "wam_provider_reusable_image_plan.json", provider_image_plan)
    write_json(output / "wam_model_source_probe.json", source_probe)
    env_file_content = _env_template(
        candidate_id=candidate_id,
        candidate=candidate,
        output_dir=output,
        source_root=source,
        checkpoint_root=checkpoint,
        adapter_command=adapter,
    )
    (output / "wam_model_runtime_env_template.sh").write_text(
        env_file_content,
        encoding="utf-8",
    )
    (output / "wam_model_runtime_env.sh").write_text(
        env_file_content,
        encoding="utf-8",
    )
    summary = {
        "schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "generated_at": generated,
        "status": status,
        "output_dir": str(output),
        "candidate_id": candidate_id,
        "blockers": blockers,
        "artifact_paths": {
            "manifest": str(output / "wam_model_runtime_bootstrap_manifest.json"),
            "checkpoint_download_plan": str(output / "wam_model_checkpoint_download_plan.json"),
            "adapter_command_contract": str(output / "wam_model_adapter_command_contract.json"),
            "provider_launch_request": str(output / "wam_model_provider_launch_request.json"),
            "provider_image_plan": str(output / "wam_provider_reusable_image_plan.json"),
            "provider_dockerfile": str(dockerfile_path),
            "source_probe": str(output / "wam_model_source_probe.json"),
            "env_template": str(output / "wam_model_runtime_env_template.sh"),
            "env_file": str(output / "wam_model_runtime_env.sh"),
        },
    }
    write_json(output / "wam_model_runtime_bootstrap_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=sorted(BOOTSTRAP_CANDIDATES), default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--job-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--adapter-command")
    parser.add_argument("--refresh-hf-metadata", action="store_true")
    parser.add_argument("--verify-source-repo", action="store_true")
    args = parser.parse_args(argv)
    summary = build_bootstrap_package(
        candidate_id=args.candidate,
        output_dir=args.output_dir,
        job_root=args.job_root,
        source_root=args.source_root,
        checkpoint_root=args.checkpoint_root,
        adapter_command=args.adapter_command,
        refresh_hf_metadata=args.refresh_hf_metadata,
        verify_source_repo=args.verify_source_repo,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
