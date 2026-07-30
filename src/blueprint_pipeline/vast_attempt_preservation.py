"""Recoverable preservation of evidence before a new Vast live attempt."""

from __future__ import annotations

import re
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .common import ensure_dir, write_json


VAST_LIVE_ATTEMPT_ARTIFACT_NAMES = (
    "vast_runtime_phase_log.jsonl",
    "vast_offer_selection_manifest.json",
    "vast_budget_ledger.json",
    "vast_all_in_cost_binding.json",
    "vast_startup_probe_manifest.json",
    "vast_gpu_sanity_report.json",
    "vast_isaac_smoke_result.json",
    "vast_provider_command_result.json",
    "vast_video_smoke_result.json",
    "vast_teardown_manifest.json",
    "vast_retained_instance_decision.json",
    "retained_gpu_session_lifecycle.jsonl",
    "retained_gpu_session_manifest.json",
    "vast_final_validation.json",
    "vast_provider_adapter_result.json",
    "vast_session_budget_guard.json",
    "vast_blueprint_bundle_preflight.json",
    "vast_launch_lock_manifest.json",
    "vast_prelaunch_inventory_guard.json",
    "provider_worker_endpoint_manifest.json",
    "vast_provider_runtime_output.zip",
    "vast_onstart_container.log",
)


def attempt_preservation_slug(generated_at: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "", generated_at)
    return slug[:32] or str(int(time.time()))


def preserve_existing_live_attempt_artifacts(
    *,
    job_dir: Path,
    generated_at: str,
    reason: str,
    artifact_names: Sequence[str] = VAST_LIVE_ATTEMPT_ARTIFACT_NAMES,
    additional_artifact_paths: Sequence[Path] = (),
) -> dict[str, Any] | None:
    """Copy any prior live-attempt evidence before refreshing output paths."""

    candidates = [job_dir / name for name in artifact_names]
    candidates.extend(Path(path).expanduser().resolve() for path in additional_artifact_paths)
    present_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for candidate in candidates:
        resolved_candidate = candidate.expanduser().resolve()
        if resolved_candidate in seen_paths or not resolved_candidate.is_file():
            continue
        seen_paths.add(resolved_candidate)
        present_paths.append(resolved_candidate)
    if not present_paths:
        return None
    preserve_dir = job_dir / f"attempt_preserved_{attempt_preservation_slug(generated_at)}"
    suffix = 1
    while preserve_dir.exists():
        suffix += 1
        preserve_dir = (
            job_dir / f"attempt_preserved_{attempt_preservation_slug(generated_at)}_{suffix}"
        )
    ensure_dir(preserve_dir)
    copied: list[str] = []
    copy_errors: list[dict[str, Any]] = []
    for source in present_paths:
        target = preserve_dir / source.name
        suffix = 1
        while target.exists():
            suffix += 1
            target = preserve_dir / f"{source.stem}_{suffix}{source.suffix}"
        try:
            shutil.copy2(source, target)
            copied.append(target.name)
        except Exception as exc:
            copy_errors.append(
                {
                    "artifact": source.name,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:300],
                }
            )
    manifest = {
        "schema_version": "vast_live_attempt_preservation_manifest.v1",
        "generated_at": generated_at,
        "status": "completed" if not copy_errors else "blocked_copy_errors",
        "reason": reason,
        "source_job_dir": str(job_dir),
        "preserve_dir": str(preserve_dir),
        "copied_artifacts": copied,
        "copy_errors": copy_errors,
        "artifact_count": len(copied),
        "raw_secret_values_recorded": False,
    }
    write_json(preserve_dir / "vast_attempt_preservation_manifest.json", manifest)
    write_json(job_dir / "vast_latest_attempt_preservation_manifest.json", manifest)
    return manifest


__all__ = [
    "VAST_LIVE_ATTEMPT_ARTIFACT_NAMES",
    "attempt_preservation_slug",
    "preserve_existing_live_attempt_artifacts",
]
