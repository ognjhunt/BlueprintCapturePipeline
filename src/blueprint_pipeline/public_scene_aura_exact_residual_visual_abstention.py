"""Seal reject-only visual abstentions for 1--5 Aura residual tasks.

The exact-mask compositor proves pixel locality and shared native 2DGS geometry;
it intentionally does not pronounce the completed pixels visually correct.  This
module binds an evidence-operator rejection to those exact bytes.  It has no
acceptance path and cannot authorize native simulation or policy execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_rehearsal_contract import MAX_REPLACEMENT_OBJECTS
from .public_scene_aura_exact_residual_compositor import (
    SCHEMA_VERSION as COMPOSITE_SCHEMA,
)


REQUEST_SCHEMA = "public_scene_aura_exact_residual_visual_abstention_request.v1"
RECEIPT_SCHEMA = "public_scene_aura_exact_residual_visual_abstention.v1"

REJECT_CODES = {
    "background_geometry_not_reconstructed",
    "multiview_background_inconsistency",
    "semantic_hallucination_in_exact_residual",
    "source_object_residual_visible",
}
DEPENDENCY_CODE = "shared_scene_dependency_failed"


class AuraExactResidualVisualAbstentionError(ValueError):
    """Stable failure for an unbound or admission-seeking review request."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _file(value: str | Path, *, code: str) -> Path:
    unresolved = Path(value).expanduser()
    if unresolved.is_symlink():
        raise AuraExactResidualVisualAbstentionError([code])
    path = unresolved.resolve()
    if not path.is_file():
        raise AuraExactResidualVisualAbstentionError([code])
    return path


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuraExactResidualVisualAbstentionError([code]) from exc
    if not isinstance(value, dict):
        raise AuraExactResidualVisualAbstentionError([code])
    return value


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _bound_relative(root: Path, value: Any, *, code: str) -> Path:
    if not isinstance(value, Mapping):
        raise AuraExactResidualVisualAbstentionError([code])
    relative = str(value.get("relative_path") or "")
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise AuraExactResidualVisualAbstentionError([code])
    unresolved = root / relative
    if unresolved.is_symlink():
        raise AuraExactResidualVisualAbstentionError([code])
    path = unresolved.resolve()
    if (
        (path != root and root not in path.parents)
        or not path.is_file()
        or path.stat().st_size != value.get("size_bytes")
        or _sha256(path) != value.get("sha256")
    ):
        raise AuraExactResidualVisualAbstentionError([code])
    return path


def _bound_absolute(value: Any, *, code: str) -> Path:
    if not isinstance(value, Mapping):
        raise AuraExactResidualVisualAbstentionError([code])
    path = _file(str(value.get("path") or ""), code=code)
    if path.stat().st_size != value.get("size_bytes") or _sha256(path) != value.get(
        "sha256"
    ):
        raise AuraExactResidualVisualAbstentionError([code])
    return path


def materialize_aura_exact_residual_visual_abstention(
    *,
    request_path: str | Path,
    composite_receipt_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Bind observed visual defects without creating an admission path."""

    request_file = _file(
        request_path, code="aura_exact_residual_visual_request_missing"
    )
    composite_file = _file(
        composite_receipt_path, code="aura_exact_residual_composite_missing"
    )
    request = _read(
        request_file, code="aura_exact_residual_visual_request_unreadable"
    )
    composite = _read(
        composite_file, code="aura_exact_residual_composite_unreadable"
    )
    if (
        composite.get("schema_version") != COMPOSITE_SCHEMA
        or composite.get("status") != "exact_mask_composites_materialized_unqualified"
        or composite.get("composite_digest")
        != canonical_digest(composite, digest_field="composite_digest")
        or composite.get("outside_mask_changed_pixels_total") != 0
        or composite.get("multi_view_consistency_required") is not True
        or (composite.get("multi_view_consistency_measurement") or {}).get(
            "visual_semantic_consistency_passed"
        )
        is not False
        or (composite.get("claim_boundary") or {}).get(
            "inpainting_result_qualified"
        )
        is not False
    ):
        raise AuraExactResidualVisualAbstentionError(
            ["aura_exact_residual_composite_invalid"]
        )

    task_manifests = composite.get("task_render_manifests") or []
    task_ids = {
        str(row.get("task_id") or "")
        for row in task_manifests
        if isinstance(row, Mapping)
    }
    replacement_count = composite.get("replacement_object_count")
    if (
        not isinstance(replacement_count, int)
        or isinstance(replacement_count, bool)
        or not 1 <= replacement_count <= MAX_REPLACEMENT_OBJECTS
        or len(task_manifests) != replacement_count
        or len(task_ids) != replacement_count
        or "" in task_ids
    ):
        raise AuraExactResidualVisualAbstentionError(
            ["aura_exact_residual_visual_task_set_invalid"]
        )

    forbidden = {"accept", "admit", "qualified", "quality_pass_claimed"}
    findings = request.get("findings")
    if (
        request.get("schema_version") != REQUEST_SCHEMA
        or request.get("decision") != "reject_or_block_shared_scene_visual_candidate"
        or request.get("reviewer_role") != "evidence_operator"
        or request.get("learned_policy_outcomes_accessed") is not False
        or forbidden.intersection(request)
        or not isinstance(findings, list)
    ):
        raise AuraExactResidualVisualAbstentionError(
            ["aura_exact_residual_visual_request_invalid"]
        )

    frames_by_task: dict[str, dict[str, Mapping[str, Any]]] = {
        task_id: {} for task_id in task_ids
    }
    for row in composite.get("frames") or []:
        if not isinstance(row, Mapping):
            raise AuraExactResidualVisualAbstentionError(
                ["aura_exact_residual_visual_frame_invalid"]
            )
        task_id = str(row.get("task_id") or "")
        camera_id = str(row.get("camera_id") or "")
        if (
            task_id not in frames_by_task
            or not camera_id
            or camera_id in frames_by_task[task_id]
        ):
            raise AuraExactResidualVisualAbstentionError(
                ["aura_exact_residual_visual_frame_invalid"]
            )
        frames_by_task[task_id][camera_id] = row

    consistency_tasks = (composite["multi_view_consistency_measurement"] or {}).get(
        "tasks"
    )
    expected_cameras_by_task: dict[str, set[str]] = {}
    if isinstance(consistency_tasks, list):
        for row in consistency_tasks:
            if not isinstance(row, Mapping):
                break
            task_id = str(row.get("task_id") or "")
            camera_ids = row.get("exact_camera_ids")
            if (
                task_id in expected_cameras_by_task
                or not isinstance(camera_ids, list)
                or not camera_ids
                or len(camera_ids) != len(set(camera_ids))
                or row.get("exact_camera_count") != len(camera_ids)
                or row.get("all_raw_frames_bind_same_native_aura_point_cloud")
                is not True
                or row.get("visual_semantic_consistency_passed") is not False
            ):
                break
            expected_cameras_by_task[task_id] = {
                str(camera_id) for camera_id in camera_ids
            }
    if (
        set(expected_cameras_by_task) != task_ids
        or any(
            set(frames_by_task[task_id]) != expected_cameras_by_task[task_id]
            for task_id in task_ids
        )
    ):
        raise AuraExactResidualVisualAbstentionError(
            ["aura_exact_residual_visual_frame_set_incomplete"]
        )

    dispositions: list[dict[str, Any]] = []
    reviewed_frames: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    rejected_tasks: set[str] = set()
    for finding in findings:
        if not isinstance(finding, Mapping):
            raise AuraExactResidualVisualAbstentionError(
                ["aura_exact_residual_visual_finding_invalid"]
            )
        task_id = str(finding.get("task_id") or "")
        disposition = str(finding.get("disposition") or "")
        codes = finding.get("observed_artifact_codes")
        camera_ids = finding.get("reviewed_camera_ids")
        if (
            task_id not in task_ids
            or task_id in seen_tasks
            or not isinstance(codes, list)
            or not codes
            or not isinstance(camera_ids, list)
            or not camera_ids
            or len(camera_ids) != len(set(camera_ids))
            or any(camera_id not in frames_by_task[task_id] for camera_id in camera_ids)
        ):
            raise AuraExactResidualVisualAbstentionError(
                ["aura_exact_residual_visual_finding_invalid"]
            )
        code_set = {str(code) for code in codes}
        if disposition == "reject_visual_candidate":
            if not code_set or not code_set <= REJECT_CODES:
                raise AuraExactResidualVisualAbstentionError(
                    ["aura_exact_residual_visual_artifact_codes_invalid"]
                )
            rejected_tasks.add(task_id)
        elif disposition == "blocked_by_shared_scene_dependency":
            if code_set != {DEPENDENCY_CODE}:
                raise AuraExactResidualVisualAbstentionError(
                    ["aura_exact_residual_visual_artifact_codes_invalid"]
                )
        else:
            raise AuraExactResidualVisualAbstentionError(
                ["aura_exact_residual_visual_disposition_invalid"]
            )
        seen_tasks.add(task_id)
        dispositions.append(
            {
                "task_id": task_id,
                "disposition": disposition,
                "observed_artifact_codes": sorted(code_set),
                "reviewed_camera_ids": sorted(str(value) for value in camera_ids),
            }
        )
        task_root = composite_file.parent / task_id
        for camera_id in sorted(str(value) for value in camera_ids):
            frame = frames_by_task[task_id][camera_id]
            reviewed_frames.append(
                {
                    "task_id": task_id,
                    "camera_id": camera_id,
                    "retained_scene_before": _record(
                        _bound_relative(
                            task_root,
                            frame.get("retained_scene_before"),
                            code="aura_exact_residual_visual_before_frame_changed",
                        )
                    ),
                    "exact_residual_mask": _record(
                        _bound_relative(
                            task_root,
                            frame.get("exact_residual_mask"),
                            code="aura_exact_residual_visual_mask_changed",
                        )
                    ),
                    "native_aura_frame": _record(
                        _bound_absolute(
                            frame.get("native_aura_frame"),
                            code="aura_exact_residual_visual_native_frame_changed",
                        )
                    ),
                    "exact_mask_composited_frame": _record(
                        _bound_relative(
                            task_root,
                            frame.get("exact_mask_composited_frame"),
                            code="aura_exact_residual_visual_composite_frame_changed",
                        )
                    ),
                }
            )
    if seen_tasks != task_ids or not rejected_tasks:
        raise AuraExactResidualVisualAbstentionError(
            ["aura_exact_residual_visual_task_dispositions_incomplete"]
        )

    blockers = [
        (
            f"released_code_exact_residual_visual_quality_rejected:{task_id}"
            if task_id in rejected_tasks
            else f"shared_scene_visual_admission_missing:{task_id}"
        )
        for task_id in sorted(task_ids)
    ]
    provider = composite.get("provider_closeout") or {}
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009D",
        "status": "typed_visual_abstention_before_native_simulator",
        "decision": request["decision"],
        "reviewer_role": request["reviewer_role"],
        "bindings": {
            "review_request": _record(request_file),
            "exact_composite_receipt": {
                **_record(composite_file),
                "composite_digest": composite["composite_digest"],
            },
            "reviewed_frames": reviewed_frames,
        },
        "replacement_object_count": replacement_count,
        "task_dispositions": sorted(dispositions, key=lambda row: row["task_id"]),
        "outside_mask_changed_pixels_total": 0,
        "provider_closeout": {
            "actual_cost_usd": provider.get("actual_cost_usd"),
            "destroyed_vast_instance_ids": provider.get("destroyed_vast_instance_ids"),
            "continuing_spend_from_this_run": False,
        },
        "smallest_missing_capability": (
            "released_code_multiview_completion_that_reconstructs_consistent_"
            "background_inside_exact_residual_masks"
        ),
        "blockers": blockers,
        "controls_executed": False,
        "learned_candidate_episodes_executed": False,
        "automatic_paid_retry_executed": False,
        "next_action": (
            "preserve the rejected candidate; do not retry paid inference or run "
            "native controls/policies without a new file-backed authority and a "
            "newly qualified appearance-completion method"
        ),
        "claim_boundary": {
            "visual_rejection_only": True,
            "inpainting_result_qualified": False,
            "gaussian_source_removal_qualified": False,
            "native_simulator_qualified": False,
            "policy_or_physical_claim": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    output = Path(output_path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise AuraExactResidualVisualAbstentionError(
            ["aura_exact_residual_visual_abstention_output_exists"]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--composite-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = materialize_aura_exact_residual_visual_abstention(
        request_path=args.request,
        composite_receipt_path=args.composite_receipt,
        output_path=args.output,
    )
    print(canonical_json(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AuraExactResidualVisualAbstentionError",
    "RECEIPT_SCHEMA",
    "REQUEST_SCHEMA",
    "main",
    "materialize_aura_exact_residual_visual_abstention",
]
