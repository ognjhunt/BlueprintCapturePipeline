"""Require the reviewed ONNX Runtime GPU build before SONIC construction."""

from __future__ import annotations

import importlib
import hashlib
from pathlib import Path
from collections.abc import Mapping
from typing import Any


PINNED_ONNXRUNTIME_GPU_VERSION = "1.24.4"


def require_sonic_cuda_runtime(module: Any | None = None) -> dict[str, Any]:
    runtime = module if module is not None else importlib.import_module("onnxruntime")
    version = str(getattr(runtime, "__version__", ""))
    if version != PINNED_ONNXRUNTIME_GPU_VERSION:
        raise RuntimeError("g1_sonic_onnxruntime_gpu_version_mismatch")
    preload = getattr(runtime, "preload_dlls", None)
    providers = getattr(runtime, "get_available_providers", None)
    if not callable(preload) or not callable(providers):
        raise RuntimeError("g1_sonic_onnxruntime_gpu_api_missing")
    preload()
    available = providers()
    if not isinstance(available, list) or "CUDAExecutionProvider" not in available:
        raise RuntimeError("g1_sonic_cuda_execution_provider_unavailable")
    return {"version": version, "available_providers": available,
            "cuda_execution_provider_available": True}


def preflight_sonic_cuda_models(
    sonic_assets: Mapping[str, Any], module: Any | None = None,
) -> dict[str, Any]:
    """Load both sealed SONIC models with CUDA before large policy downloads."""

    runtime = module if module is not None else importlib.import_module("onnxruntime")
    availability = require_sonic_cuda_runtime(runtime)
    files = sonic_assets.get("files")
    if not isinstance(files, list) or len(files) != 2:
        raise ValueError("g1_sonic_cuda_model_inventory_invalid")
    by_role = {row.get("role"): row for row in files if isinstance(row, Mapping)}
    if set(by_role) != {"encoder", "decoder"}:
        raise ValueError("g1_sonic_cuda_model_inventory_invalid")
    rows: list[dict[str, Any]] = []
    for role in ("encoder", "decoder"):
        row = by_role[role]
        path = Path(str(row.get("path") or ""))
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError("g1_sonic_cuda_model_path_invalid:" + role)
        with path.open("rb") as stream:
            identity = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
        if identity != row.get("sha256") or path.stat().st_size != row.get("size_bytes"):
            raise ValueError("g1_sonic_cuda_model_identity_invalid:" + role)
        session = runtime.InferenceSession(str(path), providers=["CUDAExecutionProvider"])
        active = session.get_providers()
        if not isinstance(active, list) or not active or active[0] != "CUDAExecutionProvider":
            raise RuntimeError("g1_sonic_cuda_model_session_fell_back:" + role)
        rows.append({"role": role, "sha256": identity, "session_providers": active})
        del session
    return {"status": "sonic_cuda_sessions_ready_no_inference",
            "onnxruntime_version": availability["version"], "models": rows,
            "policy_query_performed": False}
