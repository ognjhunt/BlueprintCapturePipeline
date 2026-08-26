#!/usr/bin/env python3
"""Execute one portable six-stage configuration envelope on an allocated host."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.task_evaluation_scene_configuration_provider_runtime import (
    execute_scene_configuration_stage_chain,
)


RESULT_SCHEMA_VERSION = "task_evaluation_scene_configuration_provider_result.v1"


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if path.is_symlink() or not isinstance(value, dict):
        raise ValueError("scene_configuration_provider_input_invalid")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _runtime_file(
    runtime: Path, relative: object, *, digest: object, size_bytes: object
) -> Path:
    value = str(relative or "")
    if not value or value.startswith("/") or ".." in Path(value).parts:
        raise ValueError("scene_configuration_provider_relative_path_invalid")
    path = (runtime / value).resolve()
    try:
        path.relative_to(runtime)
    except ValueError as exc:
        raise ValueError("scene_configuration_provider_relative_path_invalid") from exc
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != size_bytes
        or _sha256(path) != digest
    ):
        raise ValueError("scene_configuration_provider_bound_file_invalid")
    return path


def _hydrate_envelope(runtime: Path, portable: dict) -> dict:
    if portable.get("envelope_digest") != canonical_digest(
        portable, digest_field="envelope_digest"
    ):
        raise ValueError("scene_configuration_provider_envelope_digest_invalid")
    envelope = json.loads(json.dumps(portable))
    for row in envelope.get("materialized_references") or []:
        path = _runtime_file(
            runtime,
            row.get("provider_relative_path"),
            digest=row.get("digest"),
            size_bytes=row.get("size_bytes"),
        )
        row["materialized_path"] = str(path)
    render = envelope.get("render_inputs_result") or {}
    for key in ("camera_calibration", "render_manifest"):
        row = render.get(key)
        if (
            key == "render_manifest"
            and row is None
            and render.get("status")
            == "derived_method_inputs_pending_provider_render"
        ):
            continue
        row = row or {}
        path = _runtime_file(
            runtime,
            row.get("path"),
            digest=row.get("digest"),
            size_bytes=row.get("size_bytes"),
        )
        row["path"] = str(path)
    for row in render.get("derived_frames") or []:
        path = _runtime_file(
            runtime,
            row.get("path"),
            digest=row.get("digest"),
            size_bytes=row.get("size_bytes"),
        )
        row["path"] = str(path)
        mask = row.get("source_object_mask") or {}
        mask_path = _runtime_file(
            runtime,
            mask.get("path"),
            digest=mask.get("digest"),
            size_bytes=mask.get("size_bytes"),
        )
        mask["path"] = str(mask_path)
    cutout = render.get("derived_gaussian_cutout") or {}
    for key in (
        "source_object_candidate",
        "retained_scene_without_source_object",
    ):
        row = cutout.get(key) or {}
        path = _runtime_file(
            runtime,
            row.get("path"),
            digest=row.get("digest"),
            size_bytes=row.get("size_bytes"),
        )
        row["path"] = str(path)
    for row in envelope.get("stage_configuration_references") or []:
        path = _runtime_file(
            runtime,
            row.get("relative_path"),
            digest=row.get("digest"),
            size_bytes=row.get("size_bytes"),
        )
        row["materialized_path"] = str(path)
    envelope["portable_envelope_digest"] = portable["envelope_digest"]
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    return envelope


def main() -> int:
    runtime = Path(
        os.environ.get(
            "BLUEPRINT_SCENE_CONFIGURATION_RUNTIME_ROOT",
            "/workspace/task_evaluation_scene_configuration_provider_bundle/provider_runtime",
        )
    ).resolve()
    output = Path(
        os.environ.get(
            "BLUEPRINT_SCENE_CONFIGURATION_OUTPUT_ROOT",
            str(runtime.parent / "runtime_output"),
        )
    ).resolve()
    result_path = Path(
        os.environ.get(
            "BLUEPRINT_SCENE_CONFIGURATION_PROVIDER_RESULT",
            str(output / f"{RESULT_SCHEMA_VERSION}.json"),
        )
    ).resolve()
    portable = _read(runtime / "input/portable_construction_envelope.v1.json")
    portable_envelope_digest = str(portable.get("envelope_digest") or "")
    envelope = _hydrate_envelope(runtime, portable)
    configurations = {}
    for row in envelope["stage_configuration_references"]:
        stage_id = str(row["stage_id"])
        path = Path(row["materialized_path"]).resolve()
        configurations[stage_id] = (_read(path), path)
    output.mkdir(parents=True, exist_ok=True)
    stages_root = output / "stages"
    stages_root.mkdir(mode=0o750)
    try:
        chain = execute_scene_configuration_stage_chain(
            envelope=envelope,
            configurations=configurations,
            output_root=stages_root,
        )
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "completed",
            "run_id": envelope["run_id"],
            "source_commit": envelope["expected_production_commit"],
            "construction_envelope_digest": portable_envelope_digest,
            "source_construction_envelope_digest": envelope.get(
                "control_plane_envelope_digest"
            ),
            "stage_chain": chain,
            "first_stage_started": True,
            "evaluation_episode_executed": False,
            "candidate_policy_queried": False,
            "provider_zero_required_after_return": True,
            "blockers": [],
            "result_digest": "",
        }
    except Exception as exc:
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "blocked",
            "run_id": envelope.get("run_id"),
            "source_commit": envelope.get("expected_production_commit"),
            "construction_envelope_digest": portable_envelope_digest,
            "source_construction_envelope_digest": envelope.get(
                "control_plane_envelope_digest"
            ),
            "first_stage_started": stages_root.exists(),
            "evaluation_episode_executed": False,
            "candidate_policy_queried": False,
            "provider_zero_required_after_return": True,
            "blockers": [f"scene_configuration_provider_failed:{type(exc).__name__}:{exc}"],
            "result_digest": "",
        }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    result_path.write_text(canonical_json(result) + "\n", encoding="utf-8")
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
