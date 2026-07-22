"""Strategy-aware readiness calculation for the hosted native runtime."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .native_runtime_strategy import NativeRuntimeStrategy


def optional_existing_path(raw_value: Any) -> Path | None:
    """Resolve a local existing path while rejecting remote URI values."""

    text = str(raw_value or "").strip()
    if not text or text.startswith(("gs://", "http://", "https://")):
        return None
    path = Path(text).expanduser().resolve()
    return path if path.exists() else None


def find_configured_cosmos_repo(
    environ: Mapping[str, str],
) -> tuple[Path, Path] | None:
    """Resolve only an explicit, readable and executable legacy Cosmos checkout."""

    explicit = str(environ.get("COSMOS_OFFICIAL_REPO_ROOT") or "").strip()
    if not explicit:
        return None
    root = Path(explicit).expanduser()
    inference_script = root / "examples" / "inference.py"
    python_bin = root / ".venv" / "bin" / "python"
    try:
        if not inference_script.is_file() or not python_bin.is_file():
            return None
        if not os.access(inference_script, os.R_OK) or not os.access(
            python_bin, os.X_OK
        ):
            return None
        return root, python_bin
    except OSError:
        return None


def build_native_runtime_readiness(
    *,
    strategy: NativeRuntimeStrategy,
    packages: Mapping[str, bool],
    model_dir: Path | None,
    checkpoint_path: Path | None,
    cosmos_repo: tuple[Path, Path] | None,
    model_ready_override: bool,
    checkpoint_ready_override: bool,
) -> dict[str, Any]:
    """Return readiness for the selected strategy without ambient backend selection."""

    package_flags = dict(packages)
    cosmos_package_ready = bool(package_flags.get("torch")) and (
        bool(package_flags.get("diffusers"))
        or bool(package_flags.get("cosmos_predict2_5"))
        or bool(cosmos_repo)
    )
    cosmos_model_ready = bool(model_dir) or model_ready_override or bool(cosmos_repo)
    cosmos_checkpoint_ready = (
        bool(checkpoint_path) or checkpoint_ready_override or bool(cosmos_repo)
    )
    cosmos_notes: list[str] = []
    if not cosmos_package_ready:
        cosmos_notes.append("missing_native_runtime_packages")
    if not cosmos_model_ready:
        cosmos_notes.append("native_model_not_provisioned")
    if not cosmos_checkpoint_ready:
        cosmos_notes.append("native_checkpoint_not_provisioned")
    cosmos_ready = (
        cosmos_package_ready and cosmos_model_ready and cosmos_checkpoint_ready
    )
    site_splat_package_ready = bool(package_flags.get("numpy")) and bool(
        package_flags.get("PIL")
    )
    if strategy.requires_model_runtime:
        package_ready = cosmos_package_ready
        model_ready = cosmos_model_ready
        checkpoint_ready = cosmos_checkpoint_ready
        notes = list(cosmos_notes)
    else:
        package_ready = site_splat_package_ready
        model_ready = True
        checkpoint_ready = True
        notes = [] if site_splat_package_ready else ["missing_site_splat_runtime_packages"]
    return {
        "ready": package_ready and model_ready and checkpoint_ready,
        "package_ready": package_ready,
        "model_ready": model_ready,
        "checkpoint_ready": checkpoint_ready,
        "packages": package_flags,
        "model_dir": str(model_dir) if model_dir else "",
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
        "cosmos_repo": str(cosmos_repo[0]) if cosmos_repo else "",
        "cosmos_package_ready": cosmos_package_ready,
        "cosmos_model_ready": cosmos_model_ready,
        "cosmos_checkpoint_ready": cosmos_checkpoint_ready,
        "cosmos_ready": cosmos_ready,
        "cosmos_notes": cosmos_notes,
        "selected_strategy": strategy.to_dict(),
        "selected_runtime_path": strategy.synthesis_mode,
        "notes": notes,
    }
