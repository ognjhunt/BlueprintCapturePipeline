"""Build a Vast-runnable OSCAR WAM provider runtime bundle.

The bundle is intentionally small: it contains Blueprint's WAM rollout input
manifest, the materialized first-frame and skeleton-conditioning inputs, and a
remote runner that acquires OSCAR source/checkpoint material inside the GPU
runtime before attempting inference.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .oscar_wam_command_adapter import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_NUM_FRAMES,
    DEFAULT_WIDTH,
    _materialize_oscar_input_package,
)


OSCAR_WAM_PROVIDER_BUNDLE_SCHEMA_VERSION = "oscar_wam_provider_bundle_manifest.v1"
DEFAULT_OSCAR_SOURCE_URL = "https://github.com/wuzy2115/oscar-public.git"
DEFAULT_OSCAR_HF_REPO = "zywu2115/OSCAR-2B"
DEFAULT_BUNDLE_FILENAME = "oscar_wam_provider_runtime_bundle.zip"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def _materialized_package_from_existing(
    *,
    oscar_input_dir: Path,
    package_manifest_path: Path | None,
    rollout_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    first_frame = oscar_input_dir / "first_frame.png"
    skeleton = oscar_input_dir / "blueprint_proxy_skeleton_conditioning.mp4"
    if not first_frame.is_file():
        raise FileNotFoundError("oscar_input_first_frame_missing")
    if not skeleton.is_file():
        raise FileNotFoundError("oscar_input_skeleton_conditioning_video_missing")
    source_manifest_path = package_manifest_path
    if source_manifest_path is None:
        candidate = oscar_input_dir.parent / "oscar_wam_input_package_manifest.json"
        source_manifest_path = candidate if candidate.is_file() else None
    if source_manifest_path and source_manifest_path.is_file():
        manifest = _read_json(source_manifest_path)
    else:
        manifest = {
            "schema_version": "blueprint_oscar_wam_input_package.v1",
            "status": "completed",
            "prompt": "Predict the next robot-scene frames from Blueprint action conditioning.",
            "num_frames": DEFAULT_NUM_FRAMES,
            "fps": DEFAULT_FPS,
            "height": DEFAULT_HEIGHT,
            "width": DEFAULT_WIDTH,
            "source_mujoco_endpoint_eval_job_dir": rollout_manifest.get(
                "source_mujoco_endpoint_eval_job_dir"
            ),
        }
    manifest["first_frame"] = {**_mapping(manifest.get("first_frame")), "path": str(first_frame)}
    manifest["skeleton_video"] = {**_mapping(manifest.get("skeleton_video")), "path": str(skeleton)}
    manifest["claim_boundary"] = {
        **_mapping(manifest.get("claim_boundary")),
        "skeleton_conditioning_is_proxy_from_mujoco_trace": True,
        "true_robot_proprioceptive_skeleton_available": False,
        "generated_input_is_not_model_output": True,
    }
    return manifest


REMOTE_RUNNER = r'''#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import textwrap
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "wam_runtime_result.v1"
OSCAR_SOURCE_URL = os.environ.get("BLUEPRINT_OSCAR_WAM_SOURCE_URL", "https://github.com/wuzy2115/oscar-public.git")
OSCAR_HF_REPO = os.environ.get("BLUEPRINT_OSCAR_WAM_HF_REPO", "zywu2115/OSCAR-2B")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _redacted_tail(text: str, *, limit: int = 4000) -> str:
    if not text:
        return ""
    redacted = text[-limit:]
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            redacted = redacted.replace(value, "<redacted-secret>")
    return redacted


def _run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 3600,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        argv,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
        env=dict(env) if env is not None else None,
    )
    return {
        "argv_redacted": [
            "<hf_token_env>" if "HF_TOKEN" in item or item.startswith("--token") else item
            for item in argv
        ],
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 6),
        "stdout_size_bytes": len(completed.stdout or ""),
        "stderr_size_bytes": len(completed.stderr or ""),
        "stdout_tail_redacted": _redacted_tail(completed.stdout or ""),
        "stderr_tail_redacted": _redacted_tail(completed.stderr or ""),
        "raw_secret_values_recorded": False,
    }


def _find_python() -> str:
    return os.environ.get("BLUEPRINT_WAM_PROVIDER_PYTHON") or sys.executable


def _bootstrap_python(work_dir: Path) -> tuple[str, dict[str, Any]]:
    base_python = _find_python()
    configured = os.environ.get("BLUEPRINT_WAM_PROVIDER_VENV_PYTHON", "").strip()
    if configured and Path(configured).is_file():
        return configured, {
            "status": "completed",
            "source": "configured_venv_python",
            "base_python": base_python,
            "python": configured,
        }
    venv_dir = work_dir / ".blueprint_wam_venv"
    venv_python = venv_dir / "bin" / "python"
    if not venv_python.is_file():
        detail = _run(
            [base_python, "-m", "venv", "--system-site-packages", str(venv_dir)],
            timeout=300,
        )
        if detail.get("returncode") != 0 or not venv_python.is_file():
            return base_python, {
                "status": "blocked",
                "source": "venv_create_failed",
                "base_python": base_python,
                "fallback_python": base_python,
                "venv_dir": str(venv_dir),
                "blockers": ["wam_provider_venv_create_failed"],
                "subprocess": detail,
            }
    return str(venv_python), {
        "status": "completed",
        "source": "venv_with_system_site_packages",
        "base_python": base_python,
        "python": str(venv_python),
        "venv_dir": str(venv_dir),
    }


def _clone_source(work_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    configured = os.environ.get("BLUEPRINT_OSCAR_WAM_SOURCE_ROOT", "").strip()
    if configured and (Path(configured) / "inference" / "inference_oscar.py").is_file():
        return Path(configured).resolve(), {
            "status": "completed",
            "source": "configured_path",
            "path": str(Path(configured).resolve()),
        }
    target = work_dir / "external" / "oscar-public"
    if (target / "inference" / "inference_oscar.py").is_file():
        return target, {"status": "completed", "source": "existing_cache", "path": str(target)}
    if not shutil.which("git"):
        return None, {"status": "blocked", "blockers": ["git_missing_for_oscar_source_clone"]}
    target.parent.mkdir(parents=True, exist_ok=True)
    detail = _run(["git", "clone", "--depth", "1", OSCAR_SOURCE_URL, str(target)], timeout=900)
    blockers = []
    if detail["returncode"] != 0:
        blockers.append("oscar_source_clone_failed")
    if not (target / "inference" / "inference_oscar.py").is_file():
        blockers.append("oscar_inference_entrypoint_missing_after_clone")
    return (target if not blockers else None), {
        "status": "completed" if not blockers else "blocked",
        "source": "git_clone",
        "path": str(target),
        "blockers": blockers,
        "subprocess": detail,
    }


def _python_env_for_source(source_root: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if source_root is not None:
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(source_root)
            if not existing_pythonpath
            else str(source_root) + os.pathsep + existing_pythonpath
        )
    return env


def _write_text_if_changed(path: Path, text: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def _apply_oscar_source_compatibility(source_root: Path) -> dict[str, Any]:
    strategy = os.environ.get(
        "BLUEPRINT_OSCAR_WAM_TRANSFORMER_ENGINE_STRATEGY",
        "torch_sdpa_compat_shim",
    ).strip() or "torch_sdpa_compat_shim"
    if strategy in {"require_real_transformer_engine", "none", "disabled"}:
        return {
            "status": "skipped",
            "strategy": strategy,
            "files_written": [],
            "raw_secret_values_recorded": False,
    }
    shim_root = source_root / "transformer_engine"
    files = {
        shim_root / "__init__.py": """
# Blueprint-local TransformerEngine compatibility shim for OSCAR inference.
# This is only written when the real transformer_engine package is not required.
from . import pytorch

BLUEPRINT_COMPAT_SHIM = True
""",
        shim_root / "pytorch" / "__init__.py": """
# PyTorch SDPA fallback surface for OSCAR's optional TransformerEngine imports.
import torch

from .attention import DotProductAttention, apply_rotary_pos_emb

BLUEPRINT_COMPAT_SHIM = True
RMSNorm = torch.nn.RMSNorm
""",
        shim_root / "pytorch" / "attention" / "__init__.py": """
# Minimal TransformerEngine attention shim backed by torch SDPA.
from __future__ import annotations

from typing import Any

import torch
from torch import nn

BLUEPRINT_COMPAT_SHIM = True


def _flatten_bshd(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2] * tensor.shape[3])


class DotProductAttention(nn.Module):
    def __init__(
        self,
        num_attention_heads: int | None = None,
        kv_channels: int | None = None,
        *,
        attention_dropout: float = 0.0,
        qkv_format: str = "bshd",
        **_: Any,
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.kv_channels = kv_channels
        self.attention_dropout = float(attention_dropout or 0.0)
        self.qkv_format = qkv_format

    def set_context_parallel_group(self, *_: Any, **__: Any) -> None:
        return None

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
            raise ValueError("DotProductAttention shim expects q/k/v rank-4 tensors")
        if self.qkv_format == "sbhd":
            q_bhsd = query.permute(1, 2, 0, 3)
            k_bhsd = key.permute(1, 2, 0, 3)
            v_bhsd = value.permute(1, 2, 0, 3)
            out = torch.nn.functional.scaled_dot_product_attention(
                q_bhsd,
                k_bhsd,
                v_bhsd,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,
            )
            return out.permute(2, 0, 1, 3).reshape(query.shape[0], query.shape[1], -1)
        q_bhsd = query.permute(0, 2, 1, 3)
        k_bhsd = key.permute(0, 2, 1, 3)
        v_bhsd = value.permute(0, 2, 1, 3)
        out = torch.nn.functional.scaled_dot_product_attention(
            q_bhsd,
            k_bhsd,
            v_bhsd,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        return _flatten_bshd(out.permute(0, 2, 1, 3).contiguous())


def apply_rotary_pos_emb(
    tensor: torch.Tensor,
    freqs: torch.Tensor,
    *,
    tensor_format: str = "bshd",
    fused: bool = True,
) -> torch.Tensor:
    del fused
    if tensor.dim() != 4:
        raise ValueError("apply_rotary_pos_emb shim expects a rank-4 tensor")
    half = tensor.shape[-1] // 2
    freqs = freqs.to(device=tensor.device, dtype=torch.float32)
    if freqs.shape[-1] >= tensor.shape[-1]:
        freqs = freqs[..., :half]
    if freqs.shape[-1] != half:
        raise ValueError(f"rotary freqs last dim {freqs.shape[-1]} does not match half head dim {half}")
    while freqs.dim() > 2 and freqs.shape[1] == 1:
        freqs = freqs.squeeze(1)
    while freqs.dim() > 2 and freqs.shape[-2] == 1:
        freqs = freqs.squeeze(-2)
    if freqs.dim() != 2:
        freqs = freqs.reshape(freqs.shape[0], half)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    if tensor_format == "sbhd":
        cos = cos[:, None, None, :]
        sin = sin[:, None, None, :]
    else:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    even = tensor[..., 0::2].to(torch.float32)
    odd = tensor[..., 1::2].to(torch.float32)
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2).to(tensor.dtype)
""",
        shim_root / "pytorch" / "attention" / "rope.py": """
from . import apply_rotary_pos_emb

BLUEPRINT_COMPAT_SHIM = True
""",
    }
    written: list[str] = []
    for path, content in files.items():
        changed = _write_text_if_changed(path, textwrap.dedent(content).lstrip())
        written.append(str(path.relative_to(source_root)))
        if changed:
            path.chmod(0o644)
    return {
        "status": "completed",
        "strategy": strategy,
        "compatibility_basis": "OSCAR README states inference can fall back to PyTorch SDPA without TransformerEngine",
        "files_written": written,
        "raw_secret_values_recorded": False,
    }


def _framework_probe(python: str, source_root: Path | None = None) -> dict[str, Any]:
    code = (
        "import importlib.util, json\n"
        "payload={'torch_importable': False, 'torch_cuda_available': False, "
        "'cuda_device_count': 0, 'transformer_engine_importable': False, "
        "'transformer_engine_blueprint_compat_shim': False}\n"
        "try:\n"
        " import torch\n"
        " payload['torch_importable']=True\n"
        " payload['torch_version']=getattr(torch, '__version__', None)\n"
        " payload['torch_cuda_available']=bool(torch.cuda.is_available())\n"
        " payload['cuda_device_count']=torch.cuda.device_count()\n"
        "except Exception as exc:\n"
        " payload['torch_error_type']=type(exc).__name__\n"
        "spec = importlib.util.find_spec('transformer_engine')\n"
        "payload['transformer_engine_importable'] = spec is not None\n"
        "payload['transformer_engine_origin'] = getattr(spec, 'origin', None) if spec is not None else None\n"
        "try:\n"
        " import transformer_engine as te\n"
        " payload['transformer_engine_blueprint_compat_shim']=bool(getattr(te, 'BLUEPRINT_COMPAT_SHIM', False))\n"
        "except Exception as exc:\n"
        " payload['transformer_engine_error_type']=type(exc).__name__\n"
        "print(json.dumps(payload))\n"
    )
    detail = _run([python, "-c", code], timeout=120, env=_python_env_for_source(source_root))
    payload: dict[str, Any] = {}
    try:
        payload = json.loads(detail.get("stdout_tail_redacted") or "{}")
    except Exception:
        payload = {}
    return {"status": "completed", "payload": payload, "subprocess": detail}


def _ensure_dependencies(python: str, source_root: Path) -> dict[str, Any]:
    commands: list[dict[str, Any]] = []
    framework_before = _framework_probe(python, source_root)
    framework_before_payload = _mapping(framework_before.get("payload"))
    system_torch_available = framework_before_payload.get("torch_cuda_available") is True
    transformer_engine_available = framework_before_payload.get("transformer_engine_importable") is True
    transformer_engine_is_compat_shim = (
        framework_before_payload.get("transformer_engine_blueprint_compat_shim") is True
    )
    base_packages = [
        "huggingface_hub",
        "opencv-python-headless",
        "imageio",
        "imageio-ffmpeg",
        "ffmpegcv",
        "peft",
    ]
    commands.append(_run([python, "-m", "pip", "install", "--upgrade", "pip"], timeout=600))
    commands.append(_run([python, "-m", "pip", "install", *base_packages], timeout=900))
    req = source_root / "requirements.txt"
    if not req.is_file():
        req = source_root / "requirements_minimal.txt"
    if req.is_file():
        torch_req = source_root / "requirements_torch_cuda128.txt"
        filtered_req = source_root / "requirements_blueprint_without_torch.txt"
        torch_lines: list[str] = []
        filtered_lines: list[str] = []
        for line in req.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            package_name = stripped.split("==", maxsplit=1)[0].split(">=", maxsplit=1)[0]
            if package_name in {"torch", "torchvision"}:
                torch_lines.append(stripped)
            else:
                filtered_lines.append(line)
        torch_req.write_text(
            "\n".join(torch_lines or ["torch", "torchvision"]) + "\n",
            encoding="utf-8",
        )
        if system_torch_available:
            commands.append(
                {
                    "argv_redacted": [python, "-m", "pip", "install", "<torch_requirements_skipped>"],
                    "returncode": 0,
                    "duration_seconds": 0.0,
                    "stdout_size_bytes": 0,
                    "stderr_size_bytes": 0,
                    "stdout_tail_redacted": "skipped because CUDA torch is already importable",
                    "stderr_tail_redacted": "",
                    "raw_secret_values_recorded": False,
                }
            )
        else:
            commands.append(
                _run(
                    [
                        python,
                        "-m",
                        "pip",
                        "install",
                        "--index-url",
                        "https://download.pytorch.org/whl/cu128",
                        "-r",
                        str(torch_req),
                    ],
                    cwd=source_root,
                    timeout=1800,
                )
            )
        filtered_req.write_text("\n".join(filtered_lines) + "\n", encoding="utf-8")
        commands.append(
            _run([python, "-m", "pip", "install", "-r", str(filtered_req)], cwd=source_root, timeout=2400)
        )
    framework_after_requirements = _framework_probe(python, source_root)
    framework_after_requirements_payload = _mapping(framework_after_requirements.get("payload"))
    should_attempt_real_te_install = os.environ.get(
        "BLUEPRINT_OSCAR_WAM_ATTEMPT_TRANSFORMER_ENGINE_INSTALL",
        "",
    ).strip().lower() in {"1", "true", "yes"}
    if (
        framework_after_requirements_payload.get("transformer_engine_importable") is not True
        or (
            framework_after_requirements_payload.get("transformer_engine_blueprint_compat_shim") is True
            and should_attempt_real_te_install
        )
    ):
        te_env = os.environ.copy()
        te_env["NVTE_FRAMEWORK"] = "pytorch"
        commands.append(
            _run(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--no-build-isolation",
                    "transformer_engine[pytorch]",
                ],
                cwd=source_root,
                timeout=3600,
                env=te_env,
            )
        )
    framework_after_transformer_engine = _framework_probe(python, source_root)
    blockers = [f"dependency_command_failed:{index}" for index, row in enumerate(commands) if row.get("returncode") != 0]
    framework_after_transformer_engine_payload = _mapping(framework_after_transformer_engine.get("payload"))
    if framework_after_transformer_engine_payload.get("transformer_engine_importable") is not True:
        blockers.append("transformer_engine_or_compat_shim_not_importable_after_dependencies")
    return {
        "status": "completed" if not blockers else "blocked",
        "source_requirements_file": str(req) if req.is_file() else None,
        "framework_probe_before_install": framework_before,
        "framework_probe_after_requirements": framework_after_requirements,
        "framework_probe_after_transformer_engine": framework_after_transformer_engine,
        "system_torch_reused": system_torch_available,
        "transformer_engine_available_before_install": transformer_engine_available,
        "transformer_engine_compat_shim_available_before_install": transformer_engine_is_compat_shim,
        "attempted_real_transformer_engine_install": should_attempt_real_te_install,
        "commands": commands,
        "blockers": blockers,
    }


def _checkpoint(work_dir: Path, python: str) -> tuple[Path | None, dict[str, Any]]:
    configured = os.environ.get("BLUEPRINT_OSCAR_WAM_CHECKPOINT", "").strip()
    if configured and Path(configured).exists():
        return Path(configured).resolve(), {
            "status": "completed",
            "source": "configured_path",
            "path": str(Path(configured).resolve()),
        }
    target = work_dir / "checkpoints" / "oscar_2b"
    if target.exists() and any(target.rglob("*")):
        return target, {"status": "completed", "source": "existing_cache", "path": str(target)}
    code = (
        "from huggingface_hub import snapshot_download\n"
        "import os, sys\n"
        "repo=os.environ['BLUEPRINT_OSCAR_WAM_HF_REPO']\n"
        "target=os.environ['BLUEPRINT_OSCAR_WAM_CHECKPOINT_TARGET']\n"
        "snapshot_download(repo_id=repo, local_dir=target, local_dir_use_symlinks=False, token=os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN'))\n"
    )
    env = os.environ.copy()
    env["BLUEPRINT_OSCAR_WAM_HF_REPO"] = OSCAR_HF_REPO
    env["BLUEPRINT_OSCAR_WAM_CHECKPOINT_TARGET"] = str(target)
    target.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    completed = subprocess.run(
        [python, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=3600,
    )
    detail = {
        "argv_redacted": [python, "-c", "<huggingface_snapshot_download>"],
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 6),
        "stdout_size_bytes": len(completed.stdout or ""),
        "stderr_size_bytes": len(completed.stderr or ""),
        "stderr_omitted_to_avoid_secret_leakage": bool(completed.stderr),
    }
    blockers = []
    if completed.returncode != 0:
        blockers.append("oscar_checkpoint_download_failed")
    if not any(target.rglob("*")):
        blockers.append("oscar_checkpoint_directory_empty_after_download")
    return (target if not blockers else None), {
        "status": "completed" if not blockers else "blocked",
        "source": "huggingface_snapshot_download",
        "repo_id": OSCAR_HF_REPO,
        "path": str(target),
        "hf_token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
        "raw_hf_token_recorded": False,
        "blockers": blockers,
        "subprocess": detail,
    }


def _cuda_probe(python: str) -> dict[str, Any]:
    code = "import json, torch; print(json.dumps({'torch_cuda_available': bool(torch.cuda.is_available()), 'cuda_device_count': torch.cuda.device_count()}))"
    detail = _run([python, "-c", code], timeout=120)
    payload: dict[str, Any] = {}
    try:
        completed = subprocess.run([python, "-c", code], text=True, capture_output=True, check=False, timeout=120)
        payload = json.loads(completed.stdout or "{}")
        detail = {
            "argv_redacted": [python, "-c", "<torch_cuda_probe>"],
            "returncode": completed.returncode,
            "stdout_size_bytes": len(completed.stdout or ""),
            "stderr_size_bytes": len(completed.stderr or ""),
            "stderr_omitted_to_avoid_secret_leakage": bool(completed.stderr),
        }
    except Exception:
        payload = {}
    blockers = []
    if payload.get("torch_cuda_available") is not True:
        blockers.append("blocked_oscar_requires_cuda_gpu_runtime")
    return {"status": "completed" if not blockers else "blocked", "payload": payload, "blockers": blockers, "subprocess": detail}


def main() -> int:
    started = time.monotonic()
    bundle_dir = Path(os.environ.get("BLUEPRINT_WAM_PROVIDER_BUNDLE_DIR", Path.cwd())).resolve()
    output_dir = Path(os.environ.get("BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR", bundle_dir / "runtime_output")).resolve()
    work_dir = Path(
        os.environ.get("BLUEPRINT_WAM_PROVIDER_WORK_DIR", bundle_dir / "runtime_work")
    ).resolve()
    runtime_manifest_path = bundle_dir / "provider_runtime" / "wam_provider_runtime_manifest.json"
    rollout_input_path = Path(os.environ.get("BLUEPRINT_WAM_ROLLOUT_INPUT", bundle_dir / "provider_runtime" / "wam_rollout_input_manifest.json")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "wam_runtime_result.json"
    provider_output_path = output_dir / "wam_provider_output.json"
    generated_video = output_dir / "oscar_generated_rollout.mp4"
    python, python_bootstrap = _bootstrap_python(work_dir)
    blockers: list[str] = []
    if python_bootstrap.get("status") != "completed":
        blockers.extend(python_bootstrap.get("blockers") or ["wam_provider_python_bootstrap_failed"])
    runtime_manifest = _mapping(json.loads(runtime_manifest_path.read_text(encoding="utf-8"))) if runtime_manifest_path.is_file() else {}
    rollout_input = _mapping(json.loads(rollout_input_path.read_text(encoding="utf-8"))) if rollout_input_path.is_file() else {}
    source_root, source_detail = _clone_source(work_dir)
    if source_root is None:
        blockers.extend(source_detail.get("blockers") or ["oscar_source_unavailable"])
    source_compatibility_detail: dict[str, Any] = {"status": "not_run"}
    if source_root is not None and not blockers:
        source_compatibility_detail = _apply_oscar_source_compatibility(source_root)
        if source_compatibility_detail.get("status") == "blocked":
            blockers.extend(
                source_compatibility_detail.get("blockers")
                or ["oscar_source_compatibility_patch_failed"]
            )
    dependency_detail: dict[str, Any] = {"status": "not_run"}
    checkpoint_path: Path | None = None
    checkpoint_detail: dict[str, Any] = {"status": "not_run"}
    if source_root is not None and not blockers:
        dependency_detail = _ensure_dependencies(python, source_root)
        blockers.extend(dependency_detail.get("blockers") or [])
    cuda: dict[str, Any] = {"status": "not_run"}
    if not blockers:
        cuda = _cuda_probe(python)
        if cuda.get("status") != "completed":
            blockers.extend(cuda.get("blockers") or [])
    if not blockers:
        checkpoint_path, checkpoint_detail = _checkpoint(work_dir, python)
        if checkpoint_path is None:
            blockers.extend(checkpoint_detail.get("blockers") or ["oscar_checkpoint_unavailable"])
    inference_detail: dict[str, Any] = {"status": "not_run"}
    if not blockers and source_root is not None and checkpoint_path is not None:
        inference_checkpoint_path = (
            checkpoint_path / "model"
            if checkpoint_path.is_dir() and (checkpoint_path / "model").exists()
            else checkpoint_path
        )
        checkpoint_detail["inference_checkpoint_path"] = str(inference_checkpoint_path)
        checkpoint_detail["inference_checkpoint_source"] = (
            "model_subdirectory" if inference_checkpoint_path != checkpoint_path else "checkpoint_path"
        )
        first_frame = bundle_dir / "provider_runtime" / "oscar_input" / "first_frame.png"
        skeleton_video = bundle_dir / "provider_runtime" / "oscar_input" / "blueprint_proxy_skeleton_conditioning.mp4"
        prompt = runtime_manifest.get("prompt") or "Predict the next robot-scene frames from Blueprint action conditioning."
        argv = [
            python,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            "inference/inference_oscar.py",
            "--checkpoint",
            str(inference_checkpoint_path),
            "--first-frame",
            str(first_frame),
            "--skeleton-video",
            str(skeleton_video),
            "--start-frame",
            "0",
            "--prompt",
            str(prompt),
            "--num-steps",
            str(runtime_manifest.get("num_steps") or 35),
            "--guidance",
            str(runtime_manifest.get("guidance") or 6.0),
            "--seed",
            str(runtime_manifest.get("seed") or 42),
            "--num-frames",
            str(runtime_manifest.get("num_frames") or 81),
            "--height",
            str(runtime_manifest.get("height") or 480),
            "--width",
            str(runtime_manifest.get("width") or 640),
            "--fps",
            str(runtime_manifest.get("fps") or 15.0),
            "--output",
            str(generated_video),
        ]
        inference_env = os.environ.copy()
        existing_pythonpath = inference_env.get("PYTHONPATH", "")
        inference_env["PYTHONPATH"] = (
            str(source_root)
            if not existing_pythonpath
            else str(source_root) + os.pathsep + existing_pythonpath
        )
        inference_detail = _run(
            argv,
            cwd=source_root,
            timeout=int(runtime_manifest.get("timeout_seconds") or 3600),
            env=inference_env,
        )
        inference_detail["argv_redacted"] = [
            "<checkpoint_path_configured>" if item == str(inference_checkpoint_path) else item
            for item in inference_detail["argv_redacted"]
        ]
        if inference_detail.get("returncode") != 0:
            blockers.append("oscar_inference_command_nonzero")
        if not generated_video.is_file():
            discovered = sorted(output_dir.rglob("*.mp4"))
            if discovered:
                shutil.copy2(discovered[0], generated_video)
        if not generated_video.is_file():
            blockers.append("blocked_no_generated_oscar_mp4")
    rollouts = []
    if generated_video.is_file() and not blockers:
        rollouts.append(
            {
                "rollout_id": "oscar_wam_rollout_0001",
                "policy_id": "oscar_wam_provider_runtime",
                "model_candidate": "oscar_wam",
                "generated_video_path": str(generated_video),
                "model_rollout_confidence": None,
                "generated_rollout_termination_reason": "oscar_inference_command_completed",
                "success_label_source": "generated_video_requires_review",
            }
        )
    status = "completed" if rollouts and not blockers else "blocked"
    provider_payload = {
        "schema_version": "oscar_wam_command_adapter.v1",
        "status": status,
        "adapter_id": "oscar_wam_provider_runtime",
        "rollouts": rollouts,
        "generated_video_count": len(rollouts),
        "model_provenance": {
            "candidate": "oscar_wam",
            "source_url": OSCAR_SOURCE_URL,
            "checkpoint_repo": OSCAR_HF_REPO,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_exists": bool(checkpoint_path and checkpoint_path.exists()),
        },
        "input_package": runtime_manifest.get("input_package"),
        "blockers": blockers,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    _write_json(provider_output_path, provider_payload)
    runtime_result = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "runtime": "oscar_wam_provider_runtime",
        "provider": "vast_or_compatible_cuda",
        "model_candidate": "oscar_wam",
        "model_name": "OSCAR-2B",
        "action_conditioned_video_rollout_generated": bool(rollouts),
        "learned_wam_model_ran": bool(rollouts),
        "generated_video_path": str(generated_video) if generated_video.is_file() else None,
        "rollout_input_manifest_path": str(rollout_input_path),
        "rollout_input_loaded": bool(rollout_input),
        "source_detail": source_detail,
        "source_compatibility_detail": source_compatibility_detail,
        "python_bootstrap": python_bootstrap,
        "dependency_detail": dependency_detail,
        "checkpoint_detail": checkpoint_detail,
        "cuda_probe": cuda,
        "inference_detail": inference_detail,
        "duration_seconds": round(time.monotonic() - started, 6),
        "blockers": blockers,
        "truth_boundary": {
            "generated_video_is_model_output": bool(rollouts),
            "wam_success_label_from_generated_video": False,
            "forward_inverse_consistency_proven": False,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    _write_json(result_path, runtime_result)
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
'''


REMOTE_ENTRYPOINT = r'''#!/usr/bin/env bash
set +e
write_missing_result() {
  mkdir -p "${BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR:-runtime_output}"
  cat > "${BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR:-runtime_output}/wam_runtime_result.json" <<'JSON'
{
  "schema_version": "wam_runtime_result.v1",
  "status": "blocked",
  "runtime": "oscar_wam_provider_runtime",
  "blockers": [
    "wam_runner_process_exited_without_runtime_result",
    "blocked_wam_process_exited_without_result"
  ],
  "action_conditioned_video_rollout_generated": false,
  "learned_wam_model_ran": false,
  "raw_credentials_written_to_artifacts": false,
  "secret_hashes_written_to_artifacts": false
}
JSON
}
PYTHON_BIN="${BLUEPRINT_WAM_PROVIDER_PYTHON:-python3}"
"$PYTHON_BIN" "$(dirname "$0")/wam_provider_runtime_runner.py"
rc=$?
if [ ! -f "${BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR:-runtime_output}/wam_runtime_result.json" ]; then
  write_missing_result
fi
exit $rc
'''


def _write_runtime_files(
    *,
    runtime_dir: Path,
    rollout_manifest: Mapping[str, Any],
    input_package: Mapping[str, Any],
    oscar_source_url: str,
    oscar_hf_repo: str,
    timeout_seconds: int,
    num_steps: int,
    guidance: float,
    seed: int,
) -> None:
    ensure_dir(runtime_dir)
    oscar_input_dir = runtime_dir / "oscar_input"
    ensure_dir(oscar_input_dir)
    first_frame = Path(_string(_mapping(input_package.get("first_frame")).get("path"))).expanduser()
    skeleton = Path(_string(_mapping(input_package.get("skeleton_video")).get("path"))).expanduser()
    _copy_file(first_frame, oscar_input_dir / "first_frame.png")
    _copy_file(skeleton, oscar_input_dir / "blueprint_proxy_skeleton_conditioning.mp4")
    write_json(runtime_dir / "wam_rollout_input_manifest.json", dict(rollout_manifest))
    runtime_manifest = {
        "schema_version": "wam_provider_runtime_manifest.v1",
        "runtime": "oscar_wam_provider_runtime",
        "model_candidate": "oscar_wam",
        "model_name": "OSCAR-2B",
        "oscar_source_url": oscar_source_url,
        "oscar_hf_repo": oscar_hf_repo,
        "prompt": input_package.get("prompt"),
        "input_package": dict(input_package),
        "num_frames": input_package.get("num_frames") or DEFAULT_NUM_FRAMES,
        "fps": input_package.get("fps") or DEFAULT_FPS,
        "height": input_package.get("height") or DEFAULT_HEIGHT,
        "width": input_package.get("width") or DEFAULT_WIDTH,
        "num_steps": num_steps,
        "guidance": guidance,
        "seed": seed,
        "timeout_seconds": timeout_seconds,
        "remote_secret_contract": {
            "hf_token_env_supported": ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"],
            "raw_tokens_written_to_artifacts": False,
            "token_hashes_written_to_artifacts": False,
        },
        "truth_boundary": {
            "model_backend_replaceable": True,
            "generated_rollout_not_physical_robot_proof": True,
            "generated_success_label_requires_review_or_evaluator": True,
        },
    }
    write_json(runtime_dir / "wam_provider_runtime_manifest.json", runtime_manifest)
    runner = runtime_dir / "wam_provider_runtime_runner.py"
    runner.write_text(REMOTE_RUNNER, encoding="utf-8")
    runner.chmod(runner.stat().st_mode | stat.S_IXUSR)
    entrypoint = runtime_dir / "run_wam_provider_runtime.sh"
    entrypoint.write_text(REMOTE_ENTRYPOINT, encoding="utf-8")
    entrypoint.chmod(entrypoint.stat().st_mode | stat.S_IXUSR)


def build_oscar_wam_provider_bundle(
    *,
    job_dir: str | Path,
    wam_rollout_input_manifest: str | Path,
    oscar_input_dir: str | Path | None = None,
    oscar_input_package_manifest: str | Path | None = None,
    oscar_source_url: str = DEFAULT_OSCAR_SOURCE_URL,
    oscar_hf_repo: str = DEFAULT_OSCAR_HF_REPO,
    timeout_seconds: int = 3600,
    num_steps: int = 35,
    guidance: float = 6.0,
    seed: int = 42,
    bundle_filename: str = DEFAULT_BUNDLE_FILENAME,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    resolved_job_dir = Path(job_dir).expanduser().resolve()
    resolved_rollout_input = Path(wam_rollout_input_manifest).expanduser().resolve()
    ensure_dir(resolved_job_dir)
    bundle_root = resolved_job_dir / "oscar_wam_provider_bundle"
    runtime_dir = bundle_root / "provider_runtime"
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    ensure_dir(runtime_dir)
    blockers: list[str] = []
    rollout_manifest: dict[str, Any] = {}
    input_package: dict[str, Any] = {}
    if not resolved_rollout_input.is_file():
        blockers.append("wam_rollout_input_manifest_missing")
    else:
        rollout_manifest = _read_json(resolved_rollout_input)
    try:
        if not blockers and oscar_input_dir:
            resolved_input_dir = Path(oscar_input_dir).expanduser().resolve()
            resolved_package_manifest = (
                Path(oscar_input_package_manifest).expanduser().resolve()
                if oscar_input_package_manifest
                else None
            )
            input_package = _materialized_package_from_existing(
                oscar_input_dir=resolved_input_dir,
                package_manifest_path=resolved_package_manifest,
                rollout_manifest=rollout_manifest,
            )
        elif not blockers:
            workspace = resolved_job_dir / "local_input_materialization"
            input_package = _materialize_oscar_input_package(
                rollout_manifest=rollout_manifest,
                work_dir=workspace,
                width=DEFAULT_WIDTH,
                height=DEFAULT_HEIGHT,
                fps=DEFAULT_FPS,
                num_frames=DEFAULT_NUM_FRAMES,
            )
    except Exception as exc:
        blockers.append(f"oscar_wam_input_package_materialization_failed:{type(exc).__name__}")
    if not blockers:
        _write_runtime_files(
            runtime_dir=runtime_dir,
            rollout_manifest=rollout_manifest,
            input_package=input_package,
            oscar_source_url=oscar_source_url,
            oscar_hf_repo=oscar_hf_repo,
            timeout_seconds=timeout_seconds,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )
    bundle_path = resolved_job_dir / bundle_filename
    zip_entries: list[str] = []
    if not blockers:
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(bundle_root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(bundle_root).as_posix())
        with zipfile.ZipFile(bundle_path) as archive:
            zip_entries = sorted(archive.namelist())
            if archive.testzip() is not None:
                blockers.append("provider_runtime_bundle_zip_integrity_failed")
    manifest = {
        "schema_version": OSCAR_WAM_PROVIDER_BUNDLE_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "completed" if not blockers else "blocked",
        "job_dir": str(resolved_job_dir),
        "bundle_path": str(bundle_path),
        "bundle_present": bundle_path.is_file(),
        "bundle_size_bytes": bundle_path.stat().st_size if bundle_path.is_file() else 0,
        "local_bundle_ready_for_remote_staging": not blockers,
        "wam_rollout_input_manifest": str(resolved_rollout_input),
        "provider_bundle_kind": "wam",
        "runtime_dir": str(runtime_dir),
        "zip_entry_count": len(zip_entries),
        "zip_entries": zip_entries,
        "oscar_source_url": oscar_source_url,
        "oscar_hf_repo": oscar_hf_repo,
        "blockers": blockers,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "truth_boundary": {
            "bundle_build_is_not_model_execution": True,
            "provider_runtime_must_generate_mp4_before_wam_model_ran_true": True,
        },
    }
    write_json(resolved_job_dir / "oscar_wam_provider_bundle_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--wam-rollout-input-manifest", required=True)
    parser.add_argument("--oscar-input-dir")
    parser.add_argument("--oscar-input-package-manifest")
    parser.add_argument("--oscar-source-url", default=DEFAULT_OSCAR_SOURCE_URL)
    parser.add_argument("--oscar-hf-repo", default=DEFAULT_OSCAR_HF_REPO)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--num-steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bundle-filename", default=DEFAULT_BUNDLE_FILENAME)
    args = parser.parse_args(argv)
    manifest = build_oscar_wam_provider_bundle(
        job_dir=args.job_dir,
        wam_rollout_input_manifest=args.wam_rollout_input_manifest,
        oscar_input_dir=args.oscar_input_dir,
        oscar_input_package_manifest=args.oscar_input_package_manifest,
        oscar_source_url=args.oscar_source_url,
        oscar_hf_repo=args.oscar_hf_repo,
        timeout_seconds=args.timeout_seconds,
        num_steps=args.num_steps,
        guidance=args.guidance,
        seed=args.seed,
        bundle_filename=args.bundle_filename,
    )
    print(f"[oscar-wam-provider-bundle] manifest={Path(args.job_dir).resolve() / 'oscar_wam_provider_bundle_manifest.json'}")
    print(f"[oscar-wam-provider-bundle] status={manifest.get('status')}")
    blockers = manifest.get("blockers") or []
    if blockers:
        print("[oscar-wam-provider-bundle] blockers=" + ",".join(str(item) for item in blockers))
    return 0 if manifest.get("status") == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
