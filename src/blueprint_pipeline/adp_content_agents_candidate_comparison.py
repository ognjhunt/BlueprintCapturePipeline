"""Digest-bound terminal comparison for the 1--5 object CAD matrix.

This join is deliberately stricter than a spreadsheet assembled from remembered
run IDs.  Each row derives its task/backend identity from the sealed Content
Agents bundle receipt and reopens the terminal allocator, artifact manifest,
cleanup, teardown, provider-zero, and retained review-frame bytes.  It reports
within-task evidence only; it does not turn generated CAD or provider output
into SimReady, native-import, policy, or physical evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import utc_now_iso, write_json
from .content_agents_execution_route import MAX_REPLACEMENT_OBJECTS
from .decision_evidence_contracts import canonical_digest
from .simready_cad_agent_contract import ADMITTED_BACKENDS


SCHEMA_VERSION = "adp_content_agents_candidate_comparison.v1"
_TEXTURED_OUTPUT_SUFFIX = "/texture_workdir/output/textured_output.usd"
_PHYSICS_OUTPUT_SUFFIX = "/physics_workdir/physics/source_asset_physics.usda"
_REVIEW_FRAME_PREFIX = "immutable_execution/texture_workdir/renders/render_"


class ContentAgentsCandidateComparisonError(ValueError):
    """Fail-closed comparison construction/validation error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path, code: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ContentAgentsCandidateComparisonError(code)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise ContentAgentsCandidateComparisonError(code) from exc
    if not isinstance(value, Mapping):
        raise ContentAgentsCandidateComparisonError(code)
    return dict(value)


def _file_record(path: Path, code: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ContentAgentsCandidateComparisonError(code)
    return {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _path(value: Any, code: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ContentAgentsCandidateComparisonError(code)
    return Path(text).expanduser().resolve()


def _canonical_receipt(
    path: Path,
    *,
    schema_version: str,
    digest_field: str | None,
    code: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _read_json(path, code)
    if value.get("schema_version") != schema_version:
        raise ContentAgentsCandidateComparisonError(code)
    if digest_field and value.get(digest_field) != canonical_digest(
        value, digest_field=digest_field
    ):
        raise ContentAgentsCandidateComparisonError(code)
    return value, _file_record(path, code)


def _one_output(
    files: Sequence[Any], *, suffix: str, code: str
) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in files
        if isinstance(row, Mapping)
        and str(row.get("relative_path") or "").endswith(suffix)
    ]
    if len(matches) != 1:
        raise ContentAgentsCandidateComparisonError(code)
    row = matches[0]
    if not str(row.get("sha256") or "").startswith("sha256:") or not isinstance(
        row.get("size_bytes"), int
    ):
        raise ContentAgentsCandidateComparisonError(code)
    return row


def _normalize_candidate(spec: Mapping[str, Any]) -> dict[str, Any]:
    receipt_path = _path(spec.get("bundle_receipt_path"), "comparison_bundle_receipt_invalid")
    receipt, receipt_record = _canonical_receipt(
        receipt_path,
        schema_version="adp_content_agents_provider_bundle.v1",
        digest_field=None,
        code="comparison_bundle_receipt_invalid",
    )
    bindings = receipt.get("input_variant_bindings")
    normalization = receipt.get("input_usd_normalization")
    if (
        receipt.get("status") != "ready"
        or receipt.get("blockers") not in ([], None)
        or receipt.get("input_variant") != "agent_cad_v1"
        or not isinstance(bindings, Mapping)
        or not isinstance(normalization, Mapping)
    ):
        raise ContentAgentsCandidateComparisonError("comparison_bundle_receipt_invalid")
    slot = bindings.get("replacement_slot")
    backend = str(bindings.get("cad_agent_backend_id") or "")
    task_id = str(bindings.get("task_id") or "")
    asset_id = str(bindings.get("asset_id") or "")
    mesh_count = normalization.get("mesh_count")
    if (
        isinstance(slot, bool)
        or not isinstance(slot, int)
        or not 1 <= slot <= MAX_REPLACEMENT_OBJECTS
        or backend not in ADMITTED_BACKENDS
        or not task_id
        or not asset_id
        or isinstance(mesh_count, bool)
        or not isinstance(mesh_count, int)
        or mesh_count < 1
    ):
        raise ContentAgentsCandidateComparisonError("comparison_candidate_identity_invalid")

    profile_path = _path(spec.get("launch_profile_path"), "comparison_launch_profile_invalid")
    profile, profile_record = _canonical_receipt(
        profile_path,
        schema_version="task_evaluation_launch_profile.v1",
        digest_field="profile_digest",
        code="comparison_launch_profile_invalid",
    )
    immutable_inputs = profile.get("immutable_inputs")
    source_bundle_rows = (
        [
            row
            for row in immutable_inputs
            if isinstance(row, Mapping) and row.get("name") == "source_bundle_manifest"
        ]
        if isinstance(immutable_inputs, list)
        else []
    )
    allocator_profile = profile.get("allocator")
    argv = allocator_profile.get("argv") if isinstance(allocator_profile, Mapping) else None
    try:
        probe_kind = argv[argv.index("--probe-kind") + 1] if isinstance(argv, list) else None
    except (ValueError, IndexError):
        probe_kind = None
    if (
        not isinstance(allocator_profile, Mapping)
        or len(source_bundle_rows) != 1
        or source_bundle_rows[0].get("digest") != receipt_record["sha256"]
        or allocator_profile.get("subcommand") != "gpu-canary"
        or probe_kind != "adp-usd-content-agents"
    ):
        raise ContentAgentsCandidateComparisonError("comparison_launch_profile_invalid")

    allocator_path = _path(spec.get("allocator_result_path"), "comparison_allocator_result_invalid")
    allocator, allocator_record = _canonical_receipt(
        allocator_path,
        schema_version="adp_content_agents_vast_run.v1",
        digest_field=None,
        code="comparison_allocator_result_invalid",
    )
    estimated_cost = allocator.get("estimated_cost_usd")
    if (
        allocator.get("status") != "completed"
        or allocator.get("blockers") not in ([], None)
        or allocator.get("bundle_sha256") != receipt.get("bundle_sha256")
        or allocator.get("retry_cap") != 0
        or allocator.get("continuing_spend_from_this_run") is not False
        or allocator.get("all_staged_objects_absent") is not True
        or allocator.get("raw_secret_values_recorded") is not False
        or isinstance(estimated_cost, bool)
        or not isinstance(estimated_cost, (int, float))
        or estimated_cost < 0
        or estimated_cost > allocator.get("hard_cap_usd", -1)
    ):
        raise ContentAgentsCandidateComparisonError("comparison_allocator_result_invalid")

    manifest_path = _path(spec.get("artifact_manifest_path"), "comparison_artifact_manifest_invalid")
    manifest, manifest_record = _canonical_receipt(
        manifest_path,
        schema_version="task_evaluation_artifact_manifest.v1",
        digest_field="manifest_digest",
        code="comparison_artifact_manifest_invalid",
    )
    files = manifest.get("files")
    if (
        manifest.get("status") != "completed"
        or manifest.get("blockers") not in ([], None)
        or not isinstance(files, list)
    ):
        raise ContentAgentsCandidateComparisonError("comparison_artifact_manifest_invalid")
    textured = _one_output(
        files, suffix=_TEXTURED_OUTPUT_SUFFIX, code="comparison_textured_output_invalid"
    )
    physics = _one_output(
        files, suffix=_PHYSICS_OUTPUT_SUFFIX, code="comparison_physics_output_invalid"
    )
    review_rows = sorted(
        [
            dict(row)
            for row in files
            if isinstance(row, Mapping)
            and str(row.get("relative_path") or "").startswith(_REVIEW_FRAME_PREFIX)
            and str(row.get("relative_path") or "").endswith(".png")
        ],
        key=lambda row: str(row.get("relative_path") or ""),
    )
    review_paths = spec.get("review_frame_paths")
    if (
        len(review_rows) < 2
        or not isinstance(review_paths, Sequence)
        or isinstance(review_paths, (str, bytes))
        or len(review_paths) != len(review_rows)
    ):
        raise ContentAgentsCandidateComparisonError("comparison_review_frames_invalid")
    retained_frames: list[dict[str, Any]] = []
    for source, expected in zip(review_paths, review_rows, strict=True):
        record = _file_record(
            _path(source, "comparison_review_frames_invalid"),
            "comparison_review_frames_invalid",
        )
        if record["sha256"] != expected.get("sha256") or record["size_bytes"] != expected.get(
            "size_bytes"
        ):
            raise ContentAgentsCandidateComparisonError("comparison_review_frames_invalid")
        retained_frames.append({**record, "manifest_relative_path": expected["relative_path"]})

    cleanup_path = _path(spec.get("object_store_cleanup_path"), "comparison_cleanup_invalid")
    cleanup, cleanup_record = _canonical_receipt(
        cleanup_path,
        schema_version="wam_provider_object_store_cleanup.v1",
        digest_field=None,
        code="comparison_cleanup_invalid",
    )
    if (
        cleanup.get("status") != "completed"
        or cleanup.get("blockers") not in ([], None)
        or cleanup.get("all_objects_absent") is not True
        or cleanup.get("raw_secret_values_recorded") is not False
    ):
        raise ContentAgentsCandidateComparisonError("comparison_cleanup_invalid")

    teardown_path = _path(spec.get("teardown_manifest_path"), "comparison_teardown_invalid")
    teardown, teardown_record = _canonical_receipt(
        teardown_path,
        schema_version="vast_teardown_manifest.v1",
        digest_field=None,
        code="comparison_teardown_invalid",
    )
    if (
        teardown.get("status") != "completed"
        or teardown.get("continuing_spend_from_this_run") is not False
        or teardown.get("runner_gpu_teardown_completed") is not True
        or teardown.get("raw_secret_values_recorded") is not False
    ):
        raise ContentAgentsCandidateComparisonError("comparison_teardown_invalid")

    zero_path = _path(spec.get("provider_zero_path"), "comparison_provider_zero_invalid")
    zero, zero_record = _canonical_receipt(
        zero_path,
        schema_version="task_evaluation_post_teardown_provider_zero.v1",
        digest_field="provider_zero_receipt_digest",
        code="comparison_provider_zero_invalid",
    )
    if (
        zero.get("status") != "provider_zero_confirmed"
        or zero.get("blockers") not in ([], None)
        or zero.get("provider_zero_verified") is not True
        or zero.get("continuing_spend_from_this_run") is not False
        or zero.get("automatic_retry_performed") is not False
    ):
        raise ContentAgentsCandidateComparisonError("comparison_provider_zero_invalid")

    website = {"website_trigger_proven": False, "webapp_sync": None}
    if spec.get("webapp_sync_path"):
        sync_path = _path(spec["webapp_sync_path"], "comparison_webapp_sync_invalid")
        sync, sync_record = _canonical_receipt(
            sync_path,
            schema_version="task_evaluation_launch_webapp_sync_result.v1",
            digest_field="sync_result_digest",
            code="comparison_webapp_sync_invalid",
        )
        response = sync.get("response")
        launch_fields = ("launch_id", "run_id", "request_digest")
        if (
            sync.get("status") != "succeeded"
            or not isinstance(response, Mapping)
            or response.get("schema_version")
            != "task_evaluation_launch_web_sync_receipt.v1"
            or any(sync.get(field) != response.get(field) for field in launch_fields)
            or sync.get("receipt_digest") != response.get("receipt_digest")
            or any(sync.get(field) != zero.get(field) for field in launch_fields)
        ):
            raise ContentAgentsCandidateComparisonError("comparison_webapp_sync_invalid")
        website = {
            "website_trigger_proven": True,
            "webapp_sync": sync_record,
        }

    return {
        "replacement_slot": slot,
        "task_id": task_id,
        "asset_id": asset_id,
        "cad_agent_backend_id": backend,
        "mesh_count": mesh_count,
        "estimated_cost_usd": round(float(estimated_cost), 6),
        "bundle_sha256": receipt["bundle_sha256"],
        "bundle_receipt": receipt_record,
        "launch_profile": profile_record,
        "allocator_result": allocator_record,
        "artifact_manifest": manifest_record,
        "textured_output": textured,
        "physics_output": physics,
        "review_frames": retained_frames,
        "object_store_cleanup": cleanup_record,
        "teardown_manifest": teardown_record,
        "provider_zero": zero_record,
        **website,
        "claim_boundary": {
            "content_agents_executed": True,
            "generated_candidate_only": True,
            "simready_qualified": False,
            "native_simulator_import_qualified": False,
            "physical_equivalence": False,
        },
    }


def _expected(candidates: Sequence[Mapping[str, Any]], generated_at: str) -> dict[str, Any]:
    if not candidates or len(candidates) > MAX_REPLACEMENT_OBJECTS * len(ADMITTED_BACKENDS):
        raise ContentAgentsCandidateComparisonError("comparison_capacity_invalid")
    items = [_normalize_candidate(spec) for spec in candidates]
    seen: set[tuple[int, str]] = set()
    per_slot: dict[int, tuple[str, str, set[str]]] = {}
    for item in items:
        key = (item["replacement_slot"], item["cad_agent_backend_id"])
        if key in seen:
            raise ContentAgentsCandidateComparisonError("comparison_duplicate_candidate")
        seen.add(key)
        prior = per_slot.setdefault(
            item["replacement_slot"], (item["task_id"], item["asset_id"], set())
        )
        if prior[0] != item["task_id"] or prior[1] != item["asset_id"]:
            raise ContentAgentsCandidateComparisonError("comparison_slot_identity_mismatch")
        prior[2].add(item["cad_agent_backend_id"])
    slots = sorted(per_slot)
    if slots != list(range(1, len(slots) + 1)) or any(
        row[2] != set(ADMITTED_BACKENDS) for row in per_slot.values()
    ):
        raise ContentAgentsCandidateComparisonError("comparison_backend_coverage_invalid")
    ordered = sorted(items, key=lambda row: (row["replacement_slot"], row["cad_agent_backend_id"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed_candidates_ready_for_within_task_visual_review",
        "candidate_count": len(ordered),
        "replacement_slot_count": len(slots),
        "aggregate_estimated_cost_usd": round(
            sum(row["estimated_cost_usd"] for row in ordered), 6
        ),
        "items": ordered,
        "comparison_boundary": {
            "compare_backends_within_same_task_only": True,
            "cross_task_winner_permitted": False,
            "human_visual_review_required": True,
            "generated_outputs_are_physical_evidence": False,
        },
    }


def materialize_content_agents_candidate_comparison(
    *,
    candidates: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Seal terminal candidate evidence into one comparison receipt."""

    timestamp = str(generated_at or "").strip() or utc_now_iso()
    comparison = _expected(candidates, timestamp)
    comparison["receipt_digest"] = canonical_digest(
        comparison, digest_field="receipt_digest"
    )
    # The signature accepts `str | Path`, and `write_json` reaches for
    # `.parent`. Every caller so far handed it a `Path` from a test, so the
    # declared `str` half raised `AttributeError` the first time this was called
    # from a command line -- which is the only way an operator ever calls it.
    write_json(Path(output_path).expanduser(), comparison)
    return comparison


def validate_content_agents_candidate_comparison(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Reopen every referenced byte and reject comparison drift."""

    if not isinstance(value, Mapping) or value.get("schema_version") != SCHEMA_VERSION:
        raise ContentAgentsCandidateComparisonError("comparison_schema_invalid")
    records = value.get("items")
    if not isinstance(records, list) or not str(value.get("generated_at") or "").strip():
        raise ContentAgentsCandidateComparisonError("comparison_items_invalid")
    specs = []
    for row in records:
        if not isinstance(row, Mapping):
            raise ContentAgentsCandidateComparisonError("comparison_items_invalid")
        specs.append(
            {
                "bundle_receipt_path": (row.get("bundle_receipt") or {}).get("path"),
                "launch_profile_path": (row.get("launch_profile") or {}).get("path"),
                "allocator_result_path": (row.get("allocator_result") or {}).get("path"),
                "artifact_manifest_path": (row.get("artifact_manifest") or {}).get("path"),
                "object_store_cleanup_path": (row.get("object_store_cleanup") or {}).get("path"),
                "teardown_manifest_path": (row.get("teardown_manifest") or {}).get("path"),
                "provider_zero_path": (row.get("provider_zero") or {}).get("path"),
                "review_frame_paths": [
                    frame.get("path") for frame in row.get("review_frames", [])
                ],
                "webapp_sync_path": (row.get("webapp_sync") or {}).get("path"),
            }
        )
    expected = _expected(specs, str(value["generated_at"]))
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise ContentAgentsCandidateComparisonError("comparison_content_invalid")
    if value.get("receipt_digest") != canonical_digest(
        value, digest_field="receipt_digest"
    ):
        raise ContentAgentsCandidateComparisonError("comparison_digest_invalid")
    return json.loads(json.dumps(value))


__all__ = [
    "ContentAgentsCandidateComparisonError",
    "SCHEMA_VERSION",
    "materialize_content_agents_candidate_comparison",
    "validate_content_agents_candidate_comparison",
]
