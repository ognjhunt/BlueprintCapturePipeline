"""Stage raw local bundles into a capture tree and optionally run offline pipeline steps."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .capture_orchestrator import run_capture_pipeline
from .common import PipelineError
from .evaluation_prep_stage import run_evaluation_prep_stage
from .local_capture import resolve_local_capture_context
from .materialization import materialize_capture_bundle
from .preflight_capture import build_capture_preflight_report
from .capture_orchestrator import PipelineConfig


@dataclass(frozen=True)
class BundleIdentity:
    scene_id: str
    capture_id: str


def _read_json_object(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PipelineError(f"Required bundle file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PipelineError(f"Invalid JSON in bundle file: {path}") from exc
    if not isinstance(payload, Mapping):
        raise PipelineError(f"Expected JSON object in bundle file: {path}")
    return dict(payload)


def _id_candidate(payload: Mapping[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = payload.get(key)
        text = str(value or "").strip()
        if text:
            return text
    return None


def detect_bundle_identity(source_bundle: str | Path) -> BundleIdentity:
    source_root = Path(source_bundle).resolve()
    raw_root = source_root / "raw"
    if not raw_root.is_dir():
        raise PipelineError(f"Source bundle is missing raw/: {source_root}")

    manifest = _read_json_object(raw_root / "manifest.json")
    capture_context = _read_json_object(raw_root / "capture_context.json")
    upload_complete = _read_json_object(raw_root / "capture_upload_complete.json")

    scene_candidates = {
        "raw/manifest.json": _id_candidate(manifest, "scene_id", "sceneId"),
        "raw/capture_context.json": _id_candidate(capture_context, "scene_id", "sceneId"),
        "raw/capture_upload_complete.json": _id_candidate(upload_complete, "scene_id", "sceneId"),
    }
    capture_candidates = {
        "raw/manifest.json": _id_candidate(manifest, "capture_id", "captureId"),
        "raw/capture_context.json": _id_candidate(capture_context, "capture_id", "captureId"),
        "raw/capture_upload_complete.json": _id_candidate(upload_complete, "capture_id", "captureId"),
    }

    scene_values = {value for value in scene_candidates.values() if value}
    capture_values = {value for value in capture_candidates.values() if value}
    if not scene_values:
        raise PipelineError(f"Could not determine scene_id from raw bundle metadata: {source_root}")
    if not capture_values:
        raise PipelineError(f"Could not determine capture_id from raw bundle metadata: {source_root}")
    if len(scene_values) != 1:
        raise PipelineError(f"Conflicting scene IDs in raw bundle metadata: {scene_candidates}")
    if len(capture_values) != 1:
        raise PipelineError(f"Conflicting capture IDs in raw bundle metadata: {capture_candidates}")

    return BundleIdentity(
        scene_id=next(iter(scene_values)),
        capture_id=next(iter(capture_values)),
    )


def stage_local_bundle(
    *,
    source_bundle: str | Path,
    storage_root: str | Path,
    bucket: str = "local-blueprint",
    mode: str = "link",
    force: bool = False,
) -> Path:
    source_root = Path(source_bundle).resolve()
    raw_source = source_root / "raw"
    if not raw_source.is_dir():
        raise PipelineError(f"Source bundle is missing raw/: {source_root}")
    if mode not in {"link", "copy"}:
        raise PipelineError(f"Unsupported staging mode: {mode}")

    identity = detect_bundle_identity(source_root)
    storage_path = Path(storage_root).resolve()
    capture_root = storage_path / bucket / "scenes" / identity.scene_id / "captures" / identity.capture_id
    target_raw = capture_root / "raw"

    if capture_root.exists() or capture_root.is_symlink():
        if not force:
            raise PipelineError(
                f"Capture root already exists: {capture_root}. Re-run with --force to replace it."
            )
        if capture_root.is_symlink() or capture_root.is_file():
            capture_root.unlink()
        else:
            shutil.rmtree(capture_root)

    capture_root.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copytree(raw_source, target_raw)
    else:
        os.symlink(raw_source, target_raw, target_is_directory=True)
    return capture_root


def build_local_commands(*, capture_root: str | Path, storage_root: str | Path) -> Dict[str, str]:
    resolved_capture = Path(capture_root).resolve()
    resolved_storage = Path(storage_root).resolve()
    return {
        "preflight": (
            f"PYTHONPATH=src python3 -m blueprint_pipeline.preflight_capture "
            f"--capture-root {resolved_capture}"
        ),
        "qualification": (
            f"GCS_ROOT={resolved_storage} PYTHONPATH=src python3 -m blueprint_pipeline.capture_orchestrator "
            f"--descriptor-gcs-uri {resolve_local_capture_context(resolved_capture).descriptor_uri} --lane qualification"
        ),
        "evaluation_prep": (
            f"PYTHONPATH=src python3 -m blueprint_pipeline.evaluation_prep_stage "
            f"--capture-root {resolved_capture} --provider manual"
        ),
        "agent_review_openai": (
            f"PYTHONPATH=src python3 -m blueprint_pipeline.run_e2e "
            f"--capture-root {resolved_capture} --provider openai --run-evaluation-prep"
        ),
    }


def remaining_runtime_requirements() -> Dict[str, list[str]]:
    return {
        "neoverse_runtime": ["NEOVERSE_RUNTIME_SERVICE_URL"],
        "agent_review_openai": [
            "codex CLI installed",
            "Codex login via local OAuth/session",
            "optional: OPENAI_PHASE2_MODEL / OPENAI_PHASE2_REASONING_EFFORT overrides",
        ],
        "agent_review_claude": ["ANTHROPIC_API_KEY"],
    }


def run_local_bundle_workflow(
    *,
    source_bundle: str | Path,
    storage_root: str | Path,
    bucket: str = "local-blueprint",
    mode: str = "link",
    force: bool = False,
    run_qualification: bool = False,
    run_evaluation_prep: bool = False,
) -> Dict[str, Any]:
    if run_evaluation_prep and not run_qualification:
        raise PipelineError("--run-evaluation-prep requires --run-qualification")

    capture_root = stage_local_bundle(
        source_bundle=source_bundle,
        storage_root=storage_root,
        bucket=bucket,
        mode=mode,
        force=force,
    )
    context = resolve_local_capture_context(capture_root)
    preflight = build_capture_preflight_report(capture_root)
    missing_required = preflight.get("missing_required_inputs")
    if isinstance(missing_required, list) and missing_required:
        missing_text = ", ".join(str(item) for item in missing_required)
        raise PipelineError(f"Staged bundle failed preflight; missing required inputs: {missing_text}")

    materialization_result: Optional[Dict[str, Any]] = None
    if context.raw_complete_path.is_file():
        materialization_result = materialize_capture_bundle(
            bucket=context.bucket,
            scene_id=context.scene_id,
            capture_id=context.capture_id,
            gcs_root=context.storage_root,
            raw_prefix_uri=context.raw_prefix_uri,
        )

    qualification_result: Optional[Dict[str, Any]] = None
    evaluation_prep_result: Optional[Dict[str, Any]] = None
    if run_qualification:
        qualification_result = run_capture_pipeline(
            descriptor_gcs_uri=context.descriptor_uri,
            lane="qualification",
            config=PipelineConfig(
                gcs_root=context.storage_root,
            ),
        )
        if run_evaluation_prep:
            evaluation_prep_result = run_evaluation_prep_stage(
                capture_root=context.capture_root,
                provider_name="manual",
            )

    return {
        "source_bundle": str(Path(source_bundle).resolve()),
        "capture_root": str(capture_root),
        "storage_root": str(Path(storage_root).resolve()),
        "bucket": bucket,
        "mode": mode,
        "preflight": preflight,
        "materialization": materialization_result,
        "qualification": qualification_result,
        "evaluation_prep": evaluation_prep_result,
        "commands": build_local_commands(capture_root=capture_root, storage_root=storage_root),
        "remaining_runtime_requirements": remaining_runtime_requirements(),
    }
