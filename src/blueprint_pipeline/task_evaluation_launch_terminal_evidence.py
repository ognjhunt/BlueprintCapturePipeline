"""Terminal evidence projection for canonical Task Evaluation launches.

The dispatcher owns queueing, authorization, and allocator invocation.  This
module owns the narrower read-only boundary that turns an allocator result and
its artifacts into the terminal evidence embedded in the launch receipt.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .core.common import redacted_failure_text
from .decision_evidence_contracts import canonical_digest


_URI_SCHEMES = ("gs://", "s3://", "r2://", "https://")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _scene_configuration_terminal_projection(
    result: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    if (
        result.get("schema_version")
        != "task_evaluation_scene_configuration_vast_result.v1"
    ):
        return None, []

    def valid_reference(value: Any) -> bool:
        return (
            isinstance(value, Mapping)
            and str(value.get("uri") or "").startswith(_URI_SCHEMES)
            and re.fullmatch(
                r"sha256:[0-9a-f]{64}", str(value.get("digest") or "")
            )
            is not None
            and isinstance(value.get("size_bytes"), int)
            and not isinstance(value.get("size_bytes"), bool)
            and value.get("size_bytes") > 0
        )

    revision_reference = result.get("configured_scene_revision_reference")
    bundle_reference = result.get("configured_scene_bundle_reference")
    queue_finalization = _mapping(
        result.get("scene_construction_queue_finalization")
    )
    result_blockers: list[str] = []
    for item in result.get("blockers") or []:
        if not isinstance(item, str) or not item.strip():
            continue
        detail = " ".join(redacted_failure_text(item).split())
        if detail:
            result_blockers.append("scene_configuration_result:" + detail[:512])
    valid = (
        result.get("status") == "completed"
        and result.get("configuration_completed") is True
        and result.get("configured_scene_published") is True
        and result.get("full_byte_service_account_readback_passed") is True
        and queue_finalization.get("schema_version")
        == "task_evaluation_scene_construction_finalization.v1"
        and queue_finalization.get("status") == "completed"
        and queue_finalization.get("queue_state") == "completed"
        and queue_finalization.get("finalization_performed") is True
        and queue_finalization.get("run_id") == result.get("run_id")
        and queue_finalization.get("source_commit") == result.get("source_commit")
        and queue_finalization.get("result_digest")
        == canonical_digest(queue_finalization, digest_field="result_digest")
        and valid_reference(revision_reference)
        and valid_reference(bundle_reference)
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(result.get("configured_scene_revision_digest") or ""),
        )
        is not None
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(result.get("publication_result_digest") or ""),
        )
        is not None
    )
    if not valid:
        return None, sorted(
            set(
                ["scene_configuration_terminal_publication_evidence_invalid"]
                + result_blockers
            )
        )
    return {
        "schema_version": "task_evaluation_scene_configuration_terminal_evidence.v1",
        "configuration_completed": True,
        "configured_scene_published": True,
        "configured_scene_revision_digest": result[
            "configured_scene_revision_digest"
        ],
        "configured_scene_revision_reference": dict(revision_reference),
        "configured_scene_bundle_reference": dict(bundle_reference),
        "publication_result_digest": result["publication_result_digest"],
        "scene_construction_queue_finalization_digest": queue_finalization[
            "result_digest"
        ],
        "full_byte_service_account_readback_passed": True,
    }, []


def terminal_evidence(
    profile: Mapping[str, Any],
    *,
    execute: bool,
    run_root: Path,
    render_launch_path: Callable[..., str],
    artifact: Callable[[Path], dict[str, Any]],
    read_json: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    """Project one validated allocator result into a terminal receipt."""

    terminal = _mapping(profile.get("terminal_contract"))
    result_path = Path(
        render_launch_path(
            str(terminal.get("result_path") or ""), run_root=run_root
        )
    ).expanduser().resolve()
    if not execute:
        return {
            "status": "not_required_for_dry_run",
            "result": artifact(result_path),
            "blockers": [],
        }
    blockers: list[str] = []
    result: dict[str, Any] = {}
    if not result_path.is_file():
        blockers.append("allocator_terminal_result_missing")
    else:
        try:
            result = read_json(result_path)
        except (OSError, json.JSONDecodeError, ValueError):
            blockers.append("allocator_terminal_result_invalid")
    if result:
        if result.get("status") not in terminal.get("success_statuses", []):
            blockers.append("allocator_terminal_status_not_success")
        for field, expected in _mapping(terminal.get("required_values")).items():
            if result.get(field) != expected:
                blockers.append(f"allocator_terminal_value_mismatch:{field}")
        artifacts: dict[str, Any] = {}
        for field in terminal.get("required_path_fields") or []:
            raw = str(result.get(field) or "").strip()
            if not raw:
                # Path("").resolve() names the process cwd. Missing allocator
                # output must remain an absent artifact, never a checkout path.
                artifacts[str(field)] = {
                    "path": None,
                    "exists": False,
                    "digest": None,
                }
                blockers.append(f"allocator_terminal_artifact_missing:{field}")
                continue
            artifact_path = Path(raw).expanduser().resolve()
            artifacts[str(field)] = artifact(artifact_path)
            if not artifact_path.is_file():
                blockers.append(f"allocator_terminal_artifact_missing:{field}")
    else:
        artifacts = {}
    evidence = {
        "status": "passed" if not blockers else "blocked",
        "result": artifact(result_path),
        "artifacts": artifacts,
        "blockers": sorted(set(blockers)),
    }
    visual_evidence = _mapping(result.get("visual_evidence"))
    if visual_evidence:
        evidence["visual_evidence"] = visual_evidence
    scene_configuration, scene_configuration_blockers = (
        _scene_configuration_terminal_projection(result)
    )
    blockers.extend(scene_configuration_blockers)
    if scene_configuration is not None:
        evidence["scene_configuration"] = scene_configuration
    evidence["status"] = "passed" if not blockers else "blocked"
    evidence["blockers"] = sorted(set(blockers))
    return evidence


__all__ = ["terminal_evidence"]
