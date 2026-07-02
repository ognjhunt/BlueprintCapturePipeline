"""Post-Training Data Package export, checksum, and archive builder."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tarfile
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .local_capture import resolve_local_capture_context
from .rl_post_training_handoff import build_rl_post_training_handoff_packet


POST_TRAINING_DATA_PACKAGE_EXPORT_SCHEMA_VERSION = "post_training_data_package_export.v1"
CUSTOMER_HANDOFF_REPORT_SCHEMA_VERSION = "post_training_customer_handoff_report.v1"
DELIVERY_MANIFEST_SCHEMA_VERSION = "post_training_delivery_manifest.v1"
SIGNED_ACCESS_MANIFEST_SCHEMA_VERSION = "post_training_signed_access_manifest.v1"
HANDOFF_SUMMARY_SCHEMA_VERSION = "post_training_data_package_handoff_summary.v1"

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "post_training_data_package_export",
    "export_manifest_only": False,
    "export_files_written": True,
    "review_acceptance_proven": False,
    "rights_privacy_scope_proven": False,
    "signed_delivery_access_proven": False,
    "customer_handoff_ready": False,
    "delivery_access_is_deployment_approval": False,
    "package_delivery_is_deployment_approval": False,
    "deployment_approval_proven": False,
    "physical_robot_readiness_proven": False,
    "safety_validation_proven": False,
    "simulator_execution_proven": False,
    "robot_policy_execution_proven": False,
    "rank_fidelity_result_proven": False,
    "training_completed": False,
    "public_claim_upgrade_allowed": False,
}


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> List[str]:
    if value is None:
        values: Sequence[Any] = []
    elif isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence):
        values = value
    else:
        values = [value]
    out: List[str] = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _rows(payload: Mapping[str, Any], key: str) -> List[Dict[str, Any]]:
    values = payload.get(key)
    if isinstance(values, list):
        return [dict(item) for item in values if isinstance(item, Mapping)]
    return []


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    content = "\n".join(json.dumps(dict(row), sort_keys=True) for row in rows)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to(base_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base_dir.resolve()).replace("\\", "/")


def _artifact(base_dir: Path, path: Path) -> Dict[str, Any]:
    return {
        "path": _relative_to(base_dir, path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": _sha_file(path) if path.is_file() else None,
    }


def _job_artifact(job_dir: Path, name: str) -> str | None:
    path = job_dir / name
    return name.replace("\\", "/") if path.is_file() else None


def _pipeline_artifact(pipeline_dir: Path, relative_path: str) -> str | None:
    path = pipeline_dir / relative_path
    return relative_path if path.is_file() else None


def _optional_export_formats() -> Dict[str, Dict[str, Any]]:
    formats: Dict[str, Dict[str, Any]] = {}
    dependencies = {
        "rlds": ("rlds",),
        "lerobot": ("lerobot",),
        "hdf5": ("h5py",),
        "parquet": ("pyarrow", "pandas"),
    }
    for name, packages in dependencies.items():
        available = all(importlib.util.find_spec(package) is not None for package in packages)
        formats[name] = {
            "status": "available_not_written" if available else "blocked_optional_dependency_missing",
            "dependencies": list(packages),
            "format_written": False,
        }
    formats["video_bundle"] = {
        "status": "degraded_manifest_only",
        "dependencies": ["clips_manifest.json", "clip files when present"],
        "format_written": False,
    }
    return formats


def _live_closure_gate_reference(
    live_closure: Mapping[str, Any],
    gate_id: str,
) -> Dict[str, Any]:
    gate = _mapping(_mapping(live_closure.get("gates")).get(gate_id))
    evidence = _mapping(gate.get("evidence"))
    return {
        "gate_id": gate_id,
        "present": bool(gate),
        "passed": bool(gate.get("passed")),
        "blockers": _string_list(gate.get("blockers")),
        "evidence_keys": sorted(evidence),
    }


def _gate_blockers(
    gate_reference: Mapping[str, Any],
    gate_id: str,
    fallback_blocker: str,
) -> List[str]:
    if not gate_reference.get("present"):
        return [f"{gate_id}_gate_missing"]
    if gate_reference.get("passed"):
        return []
    blockers = _string_list(gate_reference.get("blockers"))
    if not blockers:
        blockers = [fallback_blocker]
    return [f"{gate_id}:{blocker}" for blocker in blockers]


def _handoff_claim_boundary(
    *,
    export_ready: bool,
    customer_handoff_ready: bool,
    review_acceptance_proven: bool,
    rights_privacy_scope_proven: bool,
    signed_delivery_access_proven: bool,
) -> Dict[str, Any]:
    return {
        **dict(CLAIM_BOUNDARY),
        "post_training_package_export_ready": bool(export_ready),
        "customer_handoff_ready": bool(customer_handoff_ready),
        "hosted_access_ready": bool(customer_handoff_ready),
        "review_acceptance_proven": bool(review_acceptance_proven),
        "rights_privacy_scope_proven": bool(rights_privacy_scope_proven),
        "signed_delivery_access_proven": bool(signed_delivery_access_proven),
        "delivery_approval_proven": False,
        "delivery_access_is_deployment_approval": False,
        "package_delivery_is_deployment_approval": False,
        "deployment_approval_proven": False,
        "physical_robot_readiness_proven": False,
        "field_readiness_proven": False,
        "safety_validation_proven": False,
        "simulator_execution_proven": False,
        "robot_policy_execution_proven": False,
        "rank_fidelity_result_proven": False,
        "training_completed": False,
        "public_claim_upgrade_allowed": False,
    }


def _merge_claim_boundary(
    existing: Mapping[str, Any],
    boundary: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        **dict(existing),
        **dict(boundary),
        "delivery_approval_proven": False,
        "delivery_access_is_deployment_approval": False,
        "package_delivery_is_deployment_approval": False,
        "deployment_approval_proven": False,
        "physical_robot_readiness_proven": False,
        "field_readiness_proven": False,
        "safety_validation_proven": False,
        "public_claim_upgrade_allowed": False,
    }


def _handoff_status(*, export_ready: bool, customer_handoff_ready: bool) -> str:
    if customer_handoff_ready:
        return "customer_handoff_ready"
    if export_ready:
        return "export_ready_handoff_blocked"
    return "blocked_missing_package_export_inputs"


def _handoff_summary(
    *,
    generated_at: str,
    export_ready: bool,
    customer_handoff_ready: bool,
    handoff_blockers: Sequence[str],
    gate_blockers: Mapping[str, Sequence[str]],
    live_gate_references: Mapping[str, Mapping[str, Any]],
    webapp_ids: Mapping[str, Any],
    included_artifacts: Mapping[str, str],
    boundary: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": HANDOFF_SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": _handoff_status(
            export_ready=export_ready,
            customer_handoff_ready=customer_handoff_ready,
        ),
        "post_training_package_export_ready": bool(export_ready),
        "customer_handoff_ready": bool(customer_handoff_ready),
        "handoff_ready": bool(customer_handoff_ready),
        "blockers": _string_list(handoff_blockers),
        "gate_blockers": {
            key: _string_list(value) for key, value in gate_blockers.items()
        },
        "webapp_ids": dict(webapp_ids),
        "artifact_paths": {
            "post_training_data_package_export_manifest": (
                "post_training_data_package_export_manifest.json"
            ),
            "customer_handoff_report": "customer_handoff_report.json",
            "delivery_manifest": "delivery_manifest.json",
            "signed_access_manifest": "signed_access_manifest.json",
            "proof_boundary": included_artifacts.get("proof_boundary"),
            "live_eval_closure_manifest": included_artifacts.get("live_eval_closure_manifest"),
            "rights_packet": included_artifacts.get("rights_packet"),
            "review_resolution_ledger": included_artifacts.get("review_resolution_ledger"),
            "accepted_failure_labels": included_artifacts.get("accepted_failure_labels"),
        },
        "live_closure_gate_references": {
            key: dict(value) for key, value in live_gate_references.items()
        },
        "claim_boundary": dict(boundary),
    }


def _read_existing_handoff_payload(
    *,
    output_dir: Path,
    job_dir: Path | None,
    name: str,
) -> Dict[str, Any]:
    output_payload = _read_optional_mapping(output_dir / name)
    if output_payload:
        return output_payload
    if job_dir and (job_dir / name).resolve() != (output_dir / name).resolve():
        return _read_optional_mapping(job_dir / name)
    return {}


def _write_customer_handoff_markdown(path: Path, report: Mapping[str, Any]) -> None:
    summary = _mapping(report.get("post_training_data_package_handoff"))
    blockers = _string_list(summary.get("blockers") or report.get("blockers"))
    lines = [
        "# Post-Training Data Package Handoff",
        "",
        f"- Status: `{summary.get('status') or report.get('status')}`",
        f"- Export ready: `{bool(summary.get('post_training_package_export_ready'))}`",
        f"- Customer handoff ready: `{bool(summary.get('customer_handoff_ready'))}`",
        "",
        "## Blockers",
        "",
    ]
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "Package export readiness does not prove hosted access, delivery approval, deployment approval, safety validation, or field readiness.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_handoff_manifests(
    *,
    output_dir: Path,
    job_dir: Path | None,
    generated_at: str,
    scene_id: str,
    capture_id: str,
    export_ready: bool,
    included_artifacts: Mapping[str, str],
    live_closure: Mapping[str, Any],
    live_gate_references: Mapping[str, Mapping[str, Any]],
    trace: Mapping[str, Any],
    labels: Mapping[str, Any],
    clips: Mapping[str, Any],
) -> Dict[str, Any]:
    gate_blockers = {
        "webapp_upstream_truth": _gate_blockers(
            live_gate_references["webapp_upstream_truth"],
            "webapp_upstream_truth",
            "webapp_upstream_truth_not_proven",
        ),
        "rights_privacy_scope": _gate_blockers(
            live_gate_references["rights_privacy_scope"],
            "rights_privacy_scope",
            "rights_privacy_scope_not_proven",
        ),
        "review_acceptance": _gate_blockers(
            live_gate_references["review_acceptance"],
            "review_acceptance",
            "review_acceptance_not_proven",
        ),
        "signed_delivery_access": _gate_blockers(
            live_gate_references["signed_delivery_access"],
            "signed_delivery_access",
            "signed_delivery_access_not_proven",
        ),
    }
    handoff_blockers = _string_list(
        [
            *(["post_training_data_package_export_not_ready"] if not export_ready else []),
            *gate_blockers["webapp_upstream_truth"],
            *gate_blockers["rights_privacy_scope"],
            *gate_blockers["review_acceptance"],
            *gate_blockers["signed_delivery_access"],
        ]
    )
    customer_handoff_ready = bool(export_ready and not handoff_blockers)
    boundary = _handoff_claim_boundary(
        export_ready=export_ready,
        customer_handoff_ready=customer_handoff_ready,
        review_acceptance_proven=bool(live_gate_references["review_acceptance"]["passed"]),
        rights_privacy_scope_proven=bool(
            live_gate_references["rights_privacy_scope"]["passed"]
        ),
        signed_delivery_access_proven=bool(
            live_gate_references["signed_delivery_access"]["passed"]
        ),
    )
    webapp_evidence = _mapping(
        _mapping(_mapping(live_closure.get("gates")).get("webapp_upstream_truth")).get(
            "evidence"
        )
    )
    webapp_ids = _mapping(webapp_evidence.get("ids"))
    summary = _handoff_summary(
        generated_at=generated_at,
        export_ready=export_ready,
        customer_handoff_ready=customer_handoff_ready,
        handoff_blockers=handoff_blockers,
        gate_blockers=gate_blockers,
        live_gate_references=live_gate_references,
        webapp_ids=webapp_ids,
        included_artifacts=included_artifacts,
        boundary=boundary,
    )

    existing_report = _read_existing_handoff_payload(
        output_dir=output_dir,
        job_dir=job_dir,
        name="customer_handoff_report.json",
    )
    report = {
        **existing_report,
        "schema_version": existing_report.get(
            "schema_version",
            CUSTOMER_HANDOFF_REPORT_SCHEMA_VERSION,
        ),
        "generated_at": generated_at,
        "status": existing_report.get("status")
        if existing_report and existing_report.get("status")
        else summary["status"],
        "scene_id": scene_id,
        "capture_id": capture_id,
        "blockers": _string_list(existing_report.get("blockers") or handoff_blockers),
        "post_training_data_package_export_path": (
            "post_training_data_package_export_manifest.json"
        ),
        "delivery_manifest_path": "delivery_manifest.json",
        "signed_access_manifest_path": "signed_access_manifest.json",
        "buyer_summary": {
            **_mapping(existing_report.get("buyer_summary")),
            "post_training_package_export_ready": bool(export_ready),
            "customer_handoff_ready": bool(customer_handoff_ready),
            "attempt_count": int(trace.get("attempt_count") or 0),
            "failure_label_count": int(labels.get("label_count") or 0),
            "clip_count": int(clips.get("clip_count") or 0),
        },
        "known_limits": _string_list(existing_report.get("known_limits"))
        or [
            "Post-Training Data Package export readiness is not hosted access.",
            "Signed delivery access is package access only, not deployment approval.",
            "This handoff does not prove safety validation or physical field readiness.",
        ],
        "post_training_data_package_handoff": summary,
        "claim_boundary": _merge_claim_boundary(
            _mapping(existing_report.get("claim_boundary")),
            boundary,
        ),
    }
    write_json(output_dir / "customer_handoff_report.json", report)
    _write_customer_handoff_markdown(output_dir / "customer_handoff_report.md", report)

    existing_signed_access = _read_existing_handoff_payload(
        output_dir=output_dir,
        job_dir=job_dir,
        name="signed_access_manifest.json",
    )
    signed_access_ready = bool(live_gate_references["signed_delivery_access"]["passed"])
    signed_access_blockers = gate_blockers["signed_delivery_access"]
    signed_access = {
        **existing_signed_access,
        "schema_version": existing_signed_access.get(
            "schema_version",
            SIGNED_ACCESS_MANIFEST_SCHEMA_VERSION,
        ),
        "generated_at": generated_at,
        "status": existing_signed_access.get("status")
        if existing_signed_access and existing_signed_access.get("status")
        else ("signed_access_ready" if signed_access_ready else "blocked_signed_delivery_access"),
        "blockers": _string_list(
            existing_signed_access.get("blockers") or signed_access_blockers
        ),
        "signed_delivery_access_proven": bool(signed_access_ready),
        "signed_access_ready": bool(signed_access_ready),
        "customer_handoff_ready": bool(customer_handoff_ready),
        "handoff_blockers": handoff_blockers,
        "delivery_access_is_deployment_approval": False,
        "package_delivery_is_deployment_approval": False,
        "claim_boundary": _merge_claim_boundary(
            _mapping(existing_signed_access.get("claim_boundary")),
            boundary,
        ),
    }
    write_json(output_dir / "signed_access_manifest.json", signed_access)

    existing_delivery = _read_existing_handoff_payload(
        output_dir=output_dir,
        job_dir=job_dir,
        name="delivery_manifest.json",
    )
    delivery = {
        **existing_delivery,
        "schema_version": existing_delivery.get(
            "schema_version",
            DELIVERY_MANIFEST_SCHEMA_VERSION,
        ),
        "generated_at": generated_at,
        "status": existing_delivery.get("status")
        if existing_delivery and existing_delivery.get("status")
        else summary["status"],
        "blockers": _string_list(existing_delivery.get("blockers") or handoff_blockers),
        "post_training_data_package_export_path": (
            "post_training_data_package_export_manifest.json"
        ),
        "customer_handoff_report_path": "customer_handoff_report.json",
        "signed_access_manifest_path": "signed_access_manifest.json",
        "local_package_index_path": "package_index.json",
        "archive_manifest_path": "archive_manifest.json",
        "post_training_data_package_handoff": summary,
        "claim_boundary": _merge_claim_boundary(
            _mapping(existing_delivery.get("claim_boundary")),
            boundary,
        ),
    }
    write_json(output_dir / "delivery_manifest.json", delivery)

    return {
        "customer_handoff_report": report,
        "delivery_manifest": delivery,
        "signed_access_manifest": signed_access,
        "summary": summary,
    }


def _annotate_live_closure_with_handoff(
    *,
    job_dir: Path | None,
    handoff_summary: Mapping[str, Any],
) -> None:
    if not job_dir:
        return
    path = job_dir / "live_eval_closure_manifest.json"
    live_closure = _read_optional_mapping(path)
    if not live_closure:
        return
    boundary = _mapping(handoff_summary.get("claim_boundary"))
    live_closure["post_training_data_package_handoff"] = dict(handoff_summary)
    proof_boundary = _mapping(live_closure.get("proof_boundary"))
    proof_boundary.update(
        {
            "post_training_package_export_ready": bool(
                handoff_summary.get("post_training_package_export_ready")
            ),
            "customer_handoff_ready": bool(handoff_summary.get("customer_handoff_ready")),
            "hosted_access_ready": bool(handoff_summary.get("customer_handoff_ready")),
            "delivery_approval_proven": False,
            "delivery_access_is_deployment_approval": False,
            "package_delivery_is_deployment_approval": False,
            "deployment_approval_proven": False,
            "physical_robot_readiness_proven": False,
            "field_readiness_proven": False,
            "safety_validation_proven": bool(boundary.get("safety_validation_proven")),
            "public_claim_upgrade_allowed": False,
        }
    )
    live_closure["proof_boundary"] = proof_boundary
    claim_boundary = _mapping(live_closure.get("claim_boundary"))
    claim_boundary.update(
        {
            "post_training_package_export_ready": bool(
                handoff_summary.get("post_training_package_export_ready")
            ),
            "customer_handoff_ready": bool(handoff_summary.get("customer_handoff_ready")),
            "hosted_access_ready": bool(handoff_summary.get("customer_handoff_ready")),
            "delivery_approval_proven": False,
            "delivery_access_is_deployment_approval": False,
            "package_delivery_is_deployment_approval": False,
            "deployment_approval_proven": False,
            "physical_robot_readiness_proven": False,
            "field_readiness_proven": False,
            "safety_validation_proven": False,
            "public_claim_upgrade_allowed": False,
        }
    )
    live_closure["claim_boundary"] = claim_boundary
    write_json(path, live_closure)


def _rows_for_optional_exports(
    *,
    attempts: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    scene_id: str,
    capture_id: str,
) -> List[Dict[str, Any]]:
    labels_by_attempt: Dict[str, List[Dict[str, Any]]] = {}
    labels_by_scenario: Dict[str, List[Dict[str, Any]]] = {}
    for label in label_rows:
        label_payload = dict(label)
        attempt_id = str(label.get("attempt_id") or "").strip()
        scenario_id = str(label.get("scenario_id") or "").strip()
        if attempt_id:
            labels_by_attempt.setdefault(attempt_id, []).append(label_payload)
        if scenario_id:
            labels_by_scenario.setdefault(scenario_id, []).append(label_payload)

    rows: List[Dict[str, Any]] = []
    for index, attempt in enumerate(attempts, start=1):
        attempt_id = str(attempt.get("attempt_id") or f"attempt_{index}").strip()
        scenario_id = str(attempt.get("scenario_id") or "").strip()
        labels = labels_by_attempt.get(attempt_id) or labels_by_scenario.get(scenario_id) or []
        rows.append(
            {
                "episode_id": attempt_id,
                "episode_index": index - 1,
                "scene_id": scene_id,
                "capture_id": capture_id,
                "task_id": attempt.get("task_id"),
                "scenario_id": scenario_id or None,
                "policy_id": attempt.get("policy_id"),
                "success": bool(attempt.get("success")),
                "status": attempt.get("status") or "unknown",
                "metrics": dict(_mapping(attempt.get("metrics"))),
                "actions": attempt.get("actions") or attempt.get("action_trace") or [],
                "observations": attempt.get("observations") or attempt.get("observation_refs") or [],
                "failure_labels": labels,
                "package_metrics": dict(metrics),
                "source_format": "blueprint_normalized_attempt_trace.v1",
                "claim_boundary": dict(CLAIM_BOUNDARY),
            }
        )
    if rows:
        return rows
    return [
        {
            "episode_id": "missing_attempts",
            "episode_index": 0,
            "scene_id": scene_id,
            "capture_id": capture_id,
            "task_id": None,
            "scenario_id": None,
            "policy_id": None,
            "success": False,
            "status": "missing_source_attempts",
            "metrics": dict(metrics),
            "actions": [],
            "observations": [],
            "failure_labels": list(label_rows),
            "package_metrics": dict(metrics),
            "source_format": "blueprint_normalized_attempt_trace.v1",
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
    ]


def _flat_export_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "episode_id": row.get("episode_id"),
                "episode_index": row.get("episode_index"),
                "scene_id": row.get("scene_id"),
                "capture_id": row.get("capture_id"),
                "task_id": row.get("task_id"),
                "scenario_id": row.get("scenario_id"),
                "policy_id": row.get("policy_id"),
                "success": bool(row.get("success")),
                "status": row.get("status"),
                "metrics_json": json.dumps(row.get("metrics") or {}, sort_keys=True),
                "actions_json": json.dumps(row.get("actions") or [], sort_keys=True),
                "observations_json": json.dumps(row.get("observations") or [], sort_keys=True),
                "failure_labels_json": json.dumps(
                    row.get("failure_labels") or [],
                    sort_keys=True,
                ),
            }
        )
    return out


def _write_native_hdf5(path: Path, rows: Sequence[Mapping[str, Any]]) -> bool:
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError:
        return False
    ensure_dir(path.parent)
    payloads = [json.dumps(dict(row), sort_keys=True) for row in rows]
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "blueprint_post_training_hdf5.v1"
        handle.attrs["source_format"] = "blueprint_normalized_attempt_trace.v1"
        string_dtype = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset("episodes_json", data=payloads, dtype=string_dtype)
    return True


def _write_native_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> bool:
    if importlib.util.find_spec("pyarrow") is None or importlib.util.find_spec("pandas") is None:
        return False
    import pandas as pd  # type: ignore[import-not-found]

    ensure_dir(path.parent)
    pd.DataFrame(_flat_export_rows(rows)).to_parquet(path, index=False)
    return True


def _write_optional_exports(
    *,
    output_dir: Path,
    attempts: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    clips: Mapping[str, Any],
    generated_at: str,
    scene_id: str,
    capture_id: str,
) -> Dict[str, Any]:
    rows = _rows_for_optional_exports(
        attempts=attempts,
        label_rows=label_rows,
        metrics=metrics,
        scene_id=scene_id,
        capture_id=capture_id,
    )
    files: Dict[str, str] = {}
    formats = _optional_export_formats()

    rlds_path = output_dir / "exports" / "rlds" / "episodes.jsonl"
    _write_jsonl(rlds_path, rows)
    files["rlds_episodes"] = _relative_to(output_dir, rlds_path)
    formats["rlds"] = {
        **formats["rlds"],
        "status": "written_jsonl",
        "format_written": True,
        "path": files["rlds_episodes"],
        "episode_count": len(rows),
        "native_package_required": False,
    }

    lerobot_rows = [
        {
            "episode_index": row.get("episode_index"),
            "episode_id": row.get("episode_id"),
            "task": row.get("task_id"),
            "scenario": row.get("scenario_id"),
            "observation": row.get("observations") or [],
            "action": row.get("actions") or [],
            "reward_or_success": 1.0 if row.get("success") else 0.0,
            "metadata": {
                "scene_id": scene_id,
                "capture_id": capture_id,
                "policy_id": row.get("policy_id"),
                "failure_labels": row.get("failure_labels") or [],
                "claim_boundary": dict(CLAIM_BOUNDARY),
            },
        }
        for row in rows
    ]
    lerobot_path = output_dir / "exports" / "lerobot" / "episodes.jsonl"
    _write_jsonl(lerobot_path, lerobot_rows)
    files["lerobot_episodes"] = _relative_to(output_dir, lerobot_path)
    formats["lerobot"] = {
        **formats["lerobot"],
        "status": "written_jsonl",
        "format_written": True,
        "path": files["lerobot_episodes"],
        "episode_count": len(lerobot_rows),
        "native_package_required": False,
    }

    hdf5_path = output_dir / "exports" / "hdf5" / "episodes.hdf5"
    if _write_native_hdf5(hdf5_path, rows):
        files["hdf5_episodes"] = _relative_to(output_dir, hdf5_path)
        formats["hdf5"] = {
            **formats["hdf5"],
            "status": "written_native",
            "format_written": True,
            "path": files["hdf5_episodes"],
            "episode_count": len(rows),
        }
    else:
        hdf5_fallback = output_dir / "exports" / "hdf5" / "episodes.hdf5.jsonl"
        _write_jsonl(hdf5_fallback, rows)
        files["hdf5_episodes"] = _relative_to(output_dir, hdf5_fallback)
        formats["hdf5"] = {
            **formats["hdf5"],
            "status": "written_jsonl_fallback",
            "format_written": True,
            "path": files["hdf5_episodes"],
            "episode_count": len(rows),
            "fallback_reason": "optional_dependency_h5py_missing",
        }

    parquet_path = output_dir / "exports" / "parquet" / "episodes.parquet"
    if _write_native_parquet(parquet_path, rows):
        files["parquet_episodes"] = _relative_to(output_dir, parquet_path)
        formats["parquet"] = {
            **formats["parquet"],
            "status": "written_native",
            "format_written": True,
            "path": files["parquet_episodes"],
            "episode_count": len(rows),
        }
    else:
        parquet_fallback = output_dir / "exports" / "parquet" / "episodes.parquet.jsonl"
        _write_jsonl(parquet_fallback, _flat_export_rows(rows))
        files["parquet_episodes"] = _relative_to(output_dir, parquet_fallback)
        formats["parquet"] = {
            **formats["parquet"],
            "status": "written_jsonl_fallback",
            "format_written": True,
            "path": files["parquet_episodes"],
            "episode_count": len(rows),
            "fallback_reason": "optional_dependency_pyarrow_or_pandas_missing",
        }

    video_bundle_path = output_dir / "exports" / "video_bundle" / "clips_manifest.json"
    write_json(
        video_bundle_path,
        {
            "schema_version": "post_training_video_bundle_manifest.v1",
            "generated_at": generated_at,
            "status": "written_manifest",
            "source_clips": dict(clips),
            "clip_count": int(clips.get("clip_count") or 0) if clips else 0,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        },
    )
    files["video_bundle_manifest"] = _relative_to(output_dir, video_bundle_path)
    formats["video_bundle"] = {
        **formats["video_bundle"],
        "status": "written_manifest",
        "format_written": True,
        "path": files["video_bundle_manifest"],
        "clip_count": int(clips.get("clip_count") or 0) if clips else 0,
    }

    return {
        "schema_version": "post_training_data_package_optional_exports.v1",
        "generated_at": generated_at,
        "formats": formats,
        "files": files,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _clip_curation_summary(capture_root: Path) -> Dict[str, Any]:
    """Summarize clip curation + semantic dedup state for the package.

    Absence of the manifests is an explicit QA state, never silently green;
    coverage counts, when present, are post-dedup.
    """
    curation = _read_optional_mapping(
        capture_root / "derived" / "clip_curation" / "clip_curation_manifest.json"
    )
    rejections = _read_optional_mapping(
        capture_root / "derived" / "clip_curation" / "clip_rejection_manifest.json"
    )
    dedup = _read_optional_mapping(
        capture_root / "derived" / "semantic_dedup" / "semantic_dedup_manifest.json"
    )
    dedup_coverage = _mapping(dedup.get("coverage"))
    return {
        "curation_status": "run" if curation else "not_run",
        "dedup_status": "run" if dedup else "not_run",
        "accepted_clip_count": curation.get("accepted_clip_count"),
        "rejected_clip_count": (
            rejections.get("rejected_count")
            if rejections
            else curation.get("rejected_clip_count")
        ),
        "post_dedup_clip_count": dedup_coverage.get("kept_clip_count"),
        "dedup_dropped_clip_count": dedup_coverage.get("dropped_clip_count"),
        "embedding_provider": _mapping(dedup.get("embedding_provider")) or None,
        "qa_note": (
            "clip curation and semantic dedup manifests included"
            if curation and dedup
            else "clip curation/dedup not run for this bundle; coverage counts are uncurated"
        ),
    }


def _write_package_files(
    *,
    output_dir: Path,
    included_artifacts: Mapping[str, str],
    trace: Mapping[str, Any],
    labels: Mapping[str, Any],
    metrics: Mapping[str, Any],
    clips: Mapping[str, Any],
    generated_at: str,
    scene_id: str,
    capture_id: str,
    visual_augmentation_packet: Mapping[str, Any] | None = None,
    rl_post_training_handoff: Mapping[str, Any] | None = None,
    clip_curation: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    data_dir = output_dir / "data"
    attempts = _rows(trace, "attempts")
    label_rows = _rows(labels, "labels")
    _write_jsonl(data_dir / "attempts.jsonl", attempts)
    _write_jsonl(data_dir / "failure_labels.jsonl", label_rows)
    write_json(
        data_dir / "metrics.json",
        dict(metrics)
        if metrics
        else {
            "schema_version": "post_training_package_metrics.v1",
            "generated_at": generated_at,
            "status": "missing_source_metrics",
            "attempt_count": len(attempts),
            "failure_count": len(label_rows),
        },
    )
    write_json(
        output_dir / "clips_manifest.json",
        dict(clips)
        if clips
        else {
            "schema_version": "post_training_package_clips_manifest.v1",
            "generated_at": generated_at,
            "status": "missing_source_clips",
            "clip_count": 0,
            "clips": [],
        },
    )
    dataset_card = {
        "schema_version": "post_training_data_package_dataset_card.v1",
        "generated_at": generated_at,
        "scene_id": scene_id,
        "capture_id": capture_id,
        "dataset_type": "real_site_robot_eval_post_training_package",
        "attempt_count": len(attempts),
        "failure_label_count": len(label_rows),
        "clip_curation": dict(clip_curation or {"curation_status": "not_run", "dedup_status": "not_run"}),
        "source_artifacts": dict(included_artifacts),
        "proof_boundary": dict(CLAIM_BOUNDARY),
    }
    license_manifest = {
        "schema_version": "post_training_data_package_license_manifest.v1",
        "generated_at": generated_at,
        "status": "review_required",
        "rights_privacy_review_required": True,
        "commercial_use_requires_package_scope_clearance": True,
        "included_artifacts": dict(included_artifacts),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    optional_exports = _write_optional_exports(
        output_dir=output_dir,
        attempts=attempts,
        label_rows=label_rows,
        metrics=metrics,
        clips=clips,
        generated_at=generated_at,
        scene_id=scene_id,
        capture_id=capture_id,
    )
    visual_augmentation_support: Dict[str, Any] | None = None
    if visual_augmentation_packet:
        packet_boundary = _mapping(visual_augmentation_packet.get("claim_boundary"))
        visual_augmentation_support = {
            "schema_version": "post_training_visual_augmentation_support_manifest.v1",
            "generated_at": generated_at,
            "status": "included_model_derived_support_packet",
            "source_packet_manifest": included_artifacts.get(
                "oscar_visual_augmentation_packet_manifest"
            ),
            "source_packet_status": visual_augmentation_packet.get("status"),
            "source_packet_type": visual_augmentation_packet.get("packet_type"),
            "variant_count": int(visual_augmentation_packet.get("variant_count") or 0),
            "generated_video_count": int(
                visual_augmentation_packet.get("generated_video_count") or 0
            ),
            "selected_backend_id": visual_augmentation_packet.get("selected_backend_id"),
            "requires_human_or_vlm_review_before_training_use": True,
            "generated_videos_model_derived": True,
            "raw_capture_evidence": False,
            "physical_robot_episode_evidence": False,
            "claim_boundary": {
                **packet_boundary,
                "artifact_purpose": "post_training_visual_augmentation_support",
                "included_in_post_training_data_package": True,
                "generated_videos_are_model_derived_support_assets": True,
                "generated_videos_are_raw_capture_evidence": False,
                "contact_physics_proven": False,
                "real_robot_readiness_proven": False,
                "deployment_safety_proven": False,
                "public_claim_upgrade_allowed": False,
            },
        }
        write_json(
            output_dir / "visual_augmentation_support_manifest.json",
            visual_augmentation_support,
        )
    rl_handoff_payload = dict(rl_post_training_handoff or {})
    if rl_handoff_payload:
        write_json(output_dir / "rl_post_training_handoff_packet.json", rl_handoff_payload)
    write_json(output_dir / "dataset_card.json", dataset_card)
    write_json(output_dir / "license_manifest.json", license_manifest)
    write_json(output_dir / "optional_export_manifest.json", optional_exports)
    package_file_index = {
        "attempts_jsonl": "data/attempts.jsonl",
        "failure_labels_jsonl": "data/failure_labels.jsonl",
        "metrics_json": "data/metrics.json",
        "clips_manifest": "clips_manifest.json",
        "dataset_card": "dataset_card.json",
        "license_manifest": "license_manifest.json",
        "optional_export_manifest": "optional_export_manifest.json",
        **dict(optional_exports.get("files") or {}),
    }
    if visual_augmentation_support:
        package_file_index["visual_augmentation_support_manifest"] = (
            "visual_augmentation_support_manifest.json"
        )
    if rl_handoff_payload:
        package_file_index["rl_post_training_handoff_packet"] = (
            "rl_post_training_handoff_packet.json"
        )
    package_index = {
        "schema_version": "post_training_data_package_index.v1",
        "generated_at": generated_at,
        "files": package_file_index,
        "source_artifacts": dict(included_artifacts),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(output_dir / "package_index.json", package_index)
    package_files = {
        key: _artifact(output_dir, output_dir / path)
        for key, path in package_index["files"].items()
    }
    checksums = {
        "schema_version": "post_training_data_package_checksums.v1",
        "generated_at": generated_at,
        "files": package_files,
    }
    write_json(output_dir / "checksums.json", checksums)
    return {
        "dataset_card": dataset_card,
        "license_manifest": license_manifest,
        "visual_augmentation_support": visual_augmentation_support,
        "rl_post_training_handoff": rl_handoff_payload,
        "package_index": package_index,
        "checksums": checksums,
        "package_files": package_files,
    }


def _write_archive(output_dir: Path, generated_at: str) -> Dict[str, Any]:
    archive_dir = output_dir / "archives"
    ensure_dir(archive_dir)
    archive_path = archive_dir / "post_training_data_package.tar.gz"
    archive_inputs = [
        output_dir / "data" / "attempts.jsonl",
        output_dir / "data" / "failure_labels.jsonl",
        output_dir / "data" / "metrics.json",
        output_dir / "clips_manifest.json",
        output_dir / "customer_handoff_report.json",
        output_dir / "customer_handoff_report.md",
        output_dir / "delivery_manifest.json",
        output_dir / "signed_access_manifest.json",
        output_dir / "dataset_card.json",
        output_dir / "license_manifest.json",
        output_dir / "optional_export_manifest.json",
        output_dir / "visual_augmentation_support_manifest.json",
        output_dir / "rl_post_training_handoff_packet.json",
        output_dir / "package_index.json",
        output_dir / "checksums.json",
    ]
    exports_dir = output_dir / "exports"
    if exports_dir.is_dir():
        archive_inputs.extend(
            path for path in sorted(exports_dir.rglob("*")) if path.is_file()
        )
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in archive_inputs:
            if path.is_file():
                tar.add(path, arcname=_relative_to(output_dir, path))
    archive_manifest = {
        "schema_version": "post_training_data_package_archive_manifest.v1",
        "generated_at": generated_at,
        "status": "created",
        "archive": _artifact(output_dir, archive_path),
        "included_files": [
            _relative_to(output_dir, path) for path in archive_inputs if path.is_file()
        ],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(output_dir / "archive_manifest.json", archive_manifest)
    return archive_manifest


def build_post_training_data_package_export(
    *,
    capture_root: str | Path,
    job_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    pipeline_dir = context.pipeline_root
    resolved_job_dir = Path(job_dir).resolve() if job_dir else None
    resolved_output_dir = (
        Path(output_dir).resolve()
        if output_dir
        else resolved_job_dir
        if resolved_job_dir
        else pipeline_dir / "post_training_data_package"
    )
    ensure_dir(resolved_output_dir)
    generated_at = utc_now_iso()

    included_artifacts: Dict[str, str] = {}
    if resolved_job_dir:
        for key, name in (
            ("job_request", "job_request.json"),
            ("normalized_attempt_trace", "normalized_attempt_trace.json"),
            ("failure_labels", "failure_labels.json"),
            ("policy_package_manifest", "policy_package_manifest.json"),
            ("task_eval_rl_post_training_handoff_packet", "rl_post_training_handoff_packet.json"),
            ("visual_review_ledger", "visual_review_ledger.json"),
            ("arena_eval_metrics", "arena_eval_metrics.json"),
            ("simulator_provider_adapter_manifest", "simulator_provider_adapter_manifest.json"),
            ("simulator_command_artifacts_manifest", "simulator_command_artifacts_manifest.json"),
            (
                "remote_cloud_execution_closure_manifest",
                "remote_cloud_execution_closure_manifest.json",
            ),
            (
                "robot_team_grade_eval_closure_manifest",
                "robot_team_grade_eval_closure_manifest.json",
            ),
            (
                "simulator_command_batch_trace_package_manifest",
                "simulator_command_batch_trace_package_manifest.json",
            ),
            (
                "simulator_command_batch_attempt_trace",
                "simulator_command_batch_attempt_trace.jsonl",
            ),
            (
                "simulator_command_batch_contact_stream",
                "simulator_command_batch_contact_stream.jsonl",
            ),
            (
                "simulator_command_batch_planner_state",
                "simulator_command_batch_planner_state.jsonl",
            ),
            (
                "simulator_command_batch_control_stream",
                "simulator_command_batch_control_stream.jsonl",
            ),
            (
                "simulator_command_batch_metrics",
                "simulator_command_batch_metrics.json",
            ),
            (
                "simulator_command_batch_failure_labels",
                "simulator_command_batch_failure_labels.json",
            ),
            (
                "simulator_command_batch_visual_media_coverage",
                "simulator_command_batch_visual_media_coverage.json",
            ),
            (
                "simulator_command_batch_visual_review_ledger",
                "simulator_command_batch_visual_review_ledger.json",
            ),
            (
                "simulator_command_digital_twin_fidelity_qa",
                "simulator_command_digital_twin_fidelity_qa.json",
            ),
            (
                "simulator_command_batch_artifact_checksums",
                "simulator_command_batch_artifact_checksums.json",
            ),
            (
                "simulator_command_batch_closure_manifest",
                "simulator_command_batch_closure_manifest.json",
            ),
            (
                "webapp_robot_eval_status_projection",
                "webapp_robot_eval_status_projection.json",
            ),
            ("scenario_eval_matrix", "scenario_eval_matrix.json"),
            ("robot_pov_observation_manifest", "robot_pov_observation_manifest.json"),
            ("robot_pov_observation_candidate_set", "robot_pov_observation_candidate_set.json"),
            ("selected_initial_policy_observation", "selected_initial_policy_observation.json"),
            ("robot_pov_observations", "robot_pov_observations.jsonl"),
            ("robot_pov_frame_sequence_manifest", "robot_pov_frame_sequence_manifest.json"),
            ("robot_pov_render_storyboard", "robot_pov_render_storyboard.json"),
            ("policy_execution_manifest", "policy_execution_manifest.json"),
            ("policy_execution_trace", "policy_execution_trace.json"),
            ("intervention_safety_ledger", "intervention_safety_ledger.json"),
            ("safety_events_ledger", "safety_events_ledger.json"),
            ("clips_manifest", "clips_manifest.json"),
            ("accepted_failure_labels", "accepted_failure_labels.json"),
            ("review_resolution_ledger", "review_resolution_ledger.json"),
            ("customer_handoff_report", "customer_handoff_report.json"),
            ("delivery_manifest", "delivery_manifest.json"),
            ("signed_access_manifest", "signed_access_manifest.json"),
            ("live_operator_ledger", "live_operator_ledger.json"),
            ("arena_rerun_plan", "arena_rerun_plan.json"),
            ("policy_adapter_manifest", "policy_adapter_manifest.json"),
            ("arena_result_ingest_ledger", "arena_result_ingest_ledger.json"),
            ("prediction_outcome_ledger", "prediction_outcome_ledger.json"),
            ("calibration_report", "calibration_report.json"),
            ("deployment_outcome_intake_manifest", "deployment_outcome_intake_manifest.json"),
            ("deployment_outcome_ledger", "deployment_outcome_ledger.json"),
            ("sim_vs_real_calibration_report", "sim_vs_real_calibration_report.json"),
            (
                "prediction_vs_actual_deployment_summary",
                "prediction_vs_actual_deployment_summary.json",
            ),
            (
                "real_world_validation_followup_plan",
                "real_world_validation_followup_plan.json",
            ),
            (
                "real_world_validation_followup_request_queue",
                "real_world_validation_followup_request_queue.json",
            ),
            ("live_eval_closure_manifest", "live_eval_closure_manifest.json"),
            ("live_eval_closure_evidence", "live_eval_closure_evidence.json"),
            ("breakage_library", "breakage_library.json"),
            ("evaluation_result", "evaluation_result.json"),
            ("robot_eval_report", "robot_eval_report.json"),
            ("robot_eval_report_markdown", "robot_eval_report.md"),
            ("proof_boundary", "proof_boundary.json"),
            (
                "policy_improvement_rl_post_training_handoff_packet",
                "policy_improvement_run/rl_post_training_handoff_packet.json",
            ),
            (
                "policy_autoresearch_report",
                "policy_autoresearch/policy_autoresearch_report.json",
            ),
            (
                "policy_candidate_package",
                "policy_autoresearch/policy_candidate_package.json",
            ),
            ("heldout_eval_result", "policy_autoresearch/heldout_eval_result.json"),
            (
                "oscar_visual_augmentation_packet_manifest",
                "oscar_visual_augmentation_packet/oscar_visual_augmentation_packet_manifest.json",
            ),
            (
                "oscar_visual_augmentation_variant_requests",
                "oscar_visual_augmentation_packet/visual_augmentation_variant_requests.jsonl",
            ),
            (
                "oscar_visual_augmentation_backend_registry",
                "oscar_visual_augmentation_packet/model_backend_registry.json",
            ),
            (
                "oscar_visual_distribution_shift_eval_protocol",
                "oscar_visual_augmentation_packet/visual_distribution_shift_eval_protocol.json",
            ),
            (
                "oscar_visual_augmentation_claim_boundary",
                "oscar_visual_augmentation_packet/claim_boundary.json",
            ),
            (
                "oscar_visual_augmentation_generation_run_manifest",
                "oscar_visual_augmentation_packet/visual_augmentation_generation_run_manifest.json",
            ),
            (
                "oscar_visual_augmentation_generation_results",
                "oscar_visual_augmentation_packet/visual_augmentation_generation_results.jsonl",
            ),
            (
                "oscar_visual_augmentation_generation_qa",
                "oscar_visual_augmentation_packet/visual_augmentation_generation_qa_manifest.json",
            ),
            (
                "oscar_visual_augmentation_training_readiness",
                "oscar_visual_augmentation_packet/visual_augmentation_training_readiness_manifest.json",
            ),
            (
                "oscar_visual_augmentation_training_dataset",
                "oscar_visual_augmentation_packet/visual_augmentation_training_dataset_manifest.json",
            ),
            (
                "oscar_visual_augmentation_training_episodes",
                "oscar_visual_augmentation_packet/exports/visual_augmentation/episodes.jsonl",
            ),
        ):
            value = _job_artifact(resolved_job_dir, name)
            if value:
                included_artifacts[key] = value

    for key, relative_path in (
        ("site_card", "robot_eval_dataset/site_card.json"),
        ("task_cards", "robot_eval_dataset/task_cards.json"),
        ("scenario_cards", "robot_eval_dataset/scenario_cards.json"),
        ("eval_cards", "robot_eval_dataset/eval_cards.json"),
        ("rights_packet", "robot_eval_dataset/rights_packet.json"),
        ("proof_boundaries", "robot_eval_dataset/proof_boundaries.json"),
        ("robot_eval_dataset_manifest", "robot_eval_dataset/robot_eval_dataset_manifest.json"),
        ("worldlabs_export_manifest", "worldlabs_export_manifest.json"),
        ("arena_environment_packet", "simulation_automation/arena_environment_packet.json"),
        ("gpu_handoff_packet", "simulation_automation/gpu_handoff_packet.json"),
    ):
        value = _pipeline_artifact(pipeline_dir, relative_path)
        if value:
            included_artifacts[key] = _relative_to(resolved_output_dir, pipeline_dir / value)

    for key, relative_path in (
        ("clip_curation_manifest", "derived/clip_curation/clip_curation_manifest.json"),
        ("clip_rejection_manifest", "derived/clip_curation/clip_rejection_manifest.json"),
        ("semantic_dedup_manifest", "derived/semantic_dedup/semantic_dedup_manifest.json"),
    ):
        candidate = context.capture_root / relative_path
        if candidate.is_file():
            included_artifacts[key] = _relative_to(resolved_output_dir, candidate)

    required = (
        "normalized_attempt_trace",
        "failure_labels",
        "prediction_outcome_ledger",
        "calibration_report",
        "breakage_library",
        "site_card",
        "task_cards",
        "scenario_cards",
        "eval_cards",
        "proof_boundaries",
    )
    missing = [key for key in required if key not in included_artifacts]
    status = "blocked_missing_inputs" if missing else "export_ready_review_required"
    live_closure = (
        _read_optional_mapping(resolved_job_dir / "live_eval_closure_manifest.json")
        if resolved_job_dir
        else {}
    )
    proof_boundary = (
        _read_optional_mapping(resolved_job_dir / "proof_boundary.json")
        if resolved_job_dir
        else {}
    )
    trace = (
        _read_optional_mapping(resolved_job_dir / "normalized_attempt_trace.json")
        if resolved_job_dir
        else {}
    )
    labels = (
        _read_optional_mapping(resolved_job_dir / "failure_labels.json")
        if resolved_job_dir
        else {}
    )
    metrics = (
        _read_optional_mapping(resolved_job_dir / "arena_eval_metrics.json")
        if resolved_job_dir
        else {}
    )
    clips = (
        _read_optional_mapping(resolved_job_dir / "clips_manifest.json")
        if resolved_job_dir
        else {}
    )
    job_request = (
        _read_optional_mapping(resolved_job_dir / "job_request.json")
        if resolved_job_dir
        else {}
    )
    scenario_matrix = (
        _read_optional_mapping(resolved_job_dir / "scenario_eval_matrix.json")
        if resolved_job_dir
        else {}
    )
    evaluation_result = (
        _read_optional_mapping(resolved_job_dir / "evaluation_result.json")
        if resolved_job_dir
        else {}
    )
    policy_package = (
        _read_optional_mapping(resolved_job_dir / "policy_package_manifest.json")
        if resolved_job_dir
        else {}
    )
    policy_report = (
        _read_optional_mapping(
            resolved_job_dir / "policy_autoresearch" / "policy_autoresearch_report.json"
        )
        if resolved_job_dir
        else {}
    )
    candidate_package = (
        _read_optional_mapping(
            resolved_job_dir / "policy_autoresearch" / "policy_candidate_package.json"
        )
        if resolved_job_dir
        else {}
    )
    heldout_result = (
        _read_optional_mapping(resolved_job_dir / "policy_autoresearch" / "heldout_eval_result.json")
        if resolved_job_dir
        else {}
    )
    policy_execution_trace = (
        _read_optional_mapping(resolved_job_dir / "policy_execution_trace.json")
        if resolved_job_dir
        else {}
    )
    safety_events = (
        _read_optional_mapping(resolved_job_dir / "intervention_safety_ledger.json")
        if resolved_job_dir
        else {}
    )
    if not safety_events and resolved_job_dir:
        safety_events = _read_optional_mapping(resolved_job_dir / "safety_events_ledger.json")
    visual_augmentation_packet = (
        _read_optional_mapping(
            resolved_job_dir
            / "oscar_visual_augmentation_packet"
            / "oscar_visual_augmentation_packet_manifest.json"
        )
        if resolved_job_dir
        else {}
    )
    rl_post_training_handoff = build_rl_post_training_handoff_packet(
        scene_id=context.scene_id,
        capture_id=context.capture_id,
        job_id=resolved_job_dir.name if resolved_job_dir else None,
        generated_at=generated_at,
        job_request=job_request,
        scenario_matrix=scenario_matrix,
        trace=trace,
        labels=labels,
        evaluation_result=evaluation_result,
        policy_package=policy_package,
        policy_report=policy_report,
        candidate_package=candidate_package,
        heldout_result=heldout_result,
        policy_execution_trace=policy_execution_trace,
        safety_events=safety_events,
        source_artifacts=included_artifacts,
    )
    package_files = _write_package_files(
        output_dir=resolved_output_dir,
        included_artifacts=included_artifacts,
        trace=trace,
        labels=labels,
        metrics=metrics,
        clips=clips,
        generated_at=generated_at,
        scene_id=context.scene_id,
        capture_id=context.capture_id,
        visual_augmentation_packet=visual_augmentation_packet,
        rl_post_training_handoff=rl_post_training_handoff,
        clip_curation=_clip_curation_summary(context.capture_root),
    )
    live_gate_references = {
        gate_id: _live_closure_gate_reference(live_closure, gate_id)
        for gate_id in (
            "webapp_upstream_truth",
            "rights_privacy_scope",
            "review_acceptance",
            "signed_delivery_access",
        )
    }
    handoff_payloads = _write_handoff_manifests(
        output_dir=resolved_output_dir,
        job_dir=resolved_job_dir,
        generated_at=generated_at,
        scene_id=context.scene_id,
        capture_id=context.capture_id,
        export_ready=status == "export_ready_review_required",
        included_artifacts=included_artifacts,
        live_closure=live_closure,
        live_gate_references=live_gate_references,
        trace=trace,
        labels=labels,
        clips=clips,
    )
    included_artifacts["customer_handoff_report"] = "customer_handoff_report.json"
    included_artifacts["customer_handoff_report_markdown"] = "customer_handoff_report.md"
    included_artifacts["delivery_manifest"] = "delivery_manifest.json"
    included_artifacts["signed_access_manifest"] = "signed_access_manifest.json"
    archive_manifest = _write_archive(resolved_output_dir, generated_at)
    delivery_manifest = _mapping(handoff_payloads.get("delivery_manifest"))
    signed_access_manifest = _mapping(handoff_payloads.get("signed_access_manifest"))
    handoff_records = {
        "proof_boundary_path": included_artifacts.get("proof_boundary"),
        "live_eval_closure_manifest_path": included_artifacts.get(
            "live_eval_closure_manifest"
        ),
        "live_eval_closure_evidence_path": included_artifacts.get(
            "live_eval_closure_evidence"
        ),
        "rights_packet_path": included_artifacts.get("rights_packet"),
        "review_resolution_ledger_path": included_artifacts.get(
            "review_resolution_ledger"
        ),
        "accepted_failure_labels_path": included_artifacts.get(
            "accepted_failure_labels"
        ),
        "customer_handoff_report_path": included_artifacts.get("customer_handoff_report"),
        "delivery_manifest_path": included_artifacts.get("delivery_manifest"),
        "signed_access_manifest_path": included_artifacts.get("signed_access_manifest"),
        "delivery_manifest_status": delivery_manifest.get("status"),
        "signed_access_manifest_status": signed_access_manifest.get("status"),
        "post_training_package_export_ready": status == "export_ready_review_required",
        "customer_handoff_ready": bool(
            _mapping(handoff_payloads.get("summary")).get("customer_handoff_ready")
        ),
        "customer_handoff_blockers": _string_list(
            _mapping(handoff_payloads.get("summary")).get("blockers")
        ),
        "live_eval_closure_status": live_closure.get("status"),
        "proof_boundary_status": proof_boundary.get("status"),
        "live_closure_gate_references": live_gate_references,
    }
    manifest_claim_boundary = {
        **dict(CLAIM_BOUNDARY),
        "post_training_package_export_ready": status == "export_ready_review_required",
        "review_acceptance_proven": live_gate_references["review_acceptance"]["passed"],
        "rights_privacy_scope_proven": live_gate_references["rights_privacy_scope"][
            "passed"
        ],
        "signed_delivery_access_proven": live_gate_references["signed_delivery_access"][
            "passed"
        ],
        "customer_handoff_ready": bool(
            _mapping(handoff_payloads.get("summary")).get("customer_handoff_ready")
        ),
        "hosted_access_ready": bool(
            _mapping(handoff_payloads.get("summary")).get("customer_handoff_ready")
        ),
        "delivery_approval_proven": False,
        "delivery_access_is_deployment_approval": False,
        "package_delivery_is_deployment_approval": False,
        "deployment_approval_proven": False,
        "physical_robot_readiness_proven": False,
        "field_readiness_proven": False,
        "safety_validation_proven": False,
    }

    manifest = {
        "schema_version": POST_TRAINING_DATA_PACKAGE_EXPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "package_type": "post_training_data_package",
        "status": status,
        "blockers": [f"missing_{key}" for key in missing],
        "included_artifacts": included_artifacts,
        "handoff_records": handoff_records,
        "manifest_counts": {
            "attempt_count": int(trace.get("attempt_count") or 0),
            "failure_label_count": int(labels.get("label_count") or 0),
            "clip_count": int(clips.get("clip_count") or 0),
            "visual_augmentation_variant_count": int(
                visual_augmentation_packet.get("variant_count") or 0
            ),
            "visual_augmentation_generated_video_count": int(
                visual_augmentation_packet.get("generated_video_count") or 0
            ),
            "rl_handoff_recoverable_failure_label_count": int(
                _mapping(
                    rl_post_training_handoff.get("recoverable_failure_labels")
                ).get("label_count")
                or 0
            ),
            "rl_handoff_intervention_event_count": int(
                _mapping(
                    rl_post_training_handoff.get("intervention_safety_ledger")
                ).get("event_count")
                or 0
            ),
        },
        "export_policy": {
            "curated_robot_pov_clips_required_for_richer_exports": True,
            "robot_pov_observations_included": "robot_pov_observation_manifest"
            in included_artifacts,
            "scenario_eval_matrix_included": "scenario_eval_matrix" in included_artifacts,
            "policy_execution_trace_included": "policy_execution_trace" in included_artifacts,
            "normalized_eval_attempts_included": "normalized_attempt_trace" in included_artifacts,
            "failure_labels_included": "failure_labels" in included_artifacts,
            "visual_review_ledger_included": "visual_review_ledger" in included_artifacts
            or "simulator_command_batch_visual_review_ledger" in included_artifacts,
            "arena_metrics_included": bool(metrics),
            "clips_manifest_included": bool(clips),
            "calibration_included": "calibration_report" in included_artifacts,
            "simulator_provider_adapter_included": "simulator_provider_adapter_manifest"
            in included_artifacts,
            "simulator_command_batch_trace_streams_included": all(
                key in included_artifacts
                for key in (
                    "simulator_command_batch_attempt_trace",
                    "simulator_command_batch_contact_stream",
                    "simulator_command_batch_planner_state",
                    "simulator_command_batch_control_stream",
                )
            ),
            "sim_vs_real_calibration_included": "sim_vs_real_calibration_report"
            in included_artifacts,
            "deployment_outcome_intake_included": "deployment_outcome_intake_manifest"
            in included_artifacts,
            "deployment_outcomes_included": "deployment_outcome_ledger" in included_artifacts,
            "real_world_validation_followup_plan_included": (
                "real_world_validation_followup_plan" in included_artifacts
            ),
            "real_world_validation_followup_queue_included": (
                "real_world_validation_followup_request_queue" in included_artifacts
            ),
            "live_eval_closure_included": "live_eval_closure_manifest"
            in included_artifacts,
            "breakage_library_included": "breakage_library" in included_artifacts,
            "robot_eval_report_included": "robot_eval_report" in included_artifacts,
            "visual_augmentation_packet_included": bool(visual_augmentation_packet),
            "visual_augmentation_is_model_derived_support": bool(
                visual_augmentation_packet
            ),
            "visual_augmentation_generated_videos_are_raw_capture_evidence": False,
            "rl_post_training_handoff_included": True,
            "rl_sparse_reward_signal_included": True,
            "concurrent_baseline_ab_plan_included": True,
            "bottleneck_stage_detection_included": True,
            "speed_curriculum_plan_included": True,
            "action_chunk_continuity_qa_included": True,
            "intervention_safety_ledger_included": True,
        },
        "package_files": package_files["package_files"],
        "dataset_card_path": "dataset_card.json",
        "license_manifest_path": "license_manifest.json",
        "package_index_path": "package_index.json",
        "checksums_path": "checksums.json",
        "archive_manifest_path": "archive_manifest.json",
        "archive": archive_manifest["archive"],
        "optional_export_manifest_path": "optional_export_manifest.json",
        "visual_augmentation_support_manifest_path": (
            "visual_augmentation_support_manifest.json"
            if package_files.get("visual_augmentation_support")
            else None
        ),
        "rl_post_training_handoff_packet_path": "rl_post_training_handoff_packet.json",
        "rl_post_training_handoff_summary": {
            "concurrent_baseline_ab_status": _mapping(
                rl_post_training_handoff.get("concurrent_baseline_ab")
            ).get("status"),
            "dominant_bottleneck_stage": _mapping(
                rl_post_training_handoff.get("bottleneck_stage_detection")
            ).get("dominant_stage"),
            "speed_curriculum_status": _mapping(
                rl_post_training_handoff.get("speed_curriculum_plan")
            ).get("status"),
            "action_chunk_qa_status": _mapping(
                rl_post_training_handoff.get("action_chunk_continuity_qa")
            ).get("status"),
            "safety_validation_proven": False,
        },
        "claim_boundary": manifest_claim_boundary,
    }
    write_json(resolved_output_dir / "post_training_data_package_export_manifest.json", manifest)
    _annotate_live_closure_with_handoff(
        job_dir=resolved_job_dir,
        handoff_summary=_mapping(handoff_payloads.get("summary")),
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a Post-Training Data Package export, checksum, and archive"
    )
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--job-dir")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    result = build_post_training_data_package_export(
        capture_root=args.capture_root,
        job_dir=args.job_dir,
        output_dir=args.output_dir,
    )
    default_output_dir = Path(args.capture_root) / "pipeline" / "post_training_data_package"
    manifest_dir = Path(args.output_dir or args.job_dir or default_output_dir)
    manifest_path = manifest_dir / "post_training_data_package_export_manifest.json"
    print(f"[post-training-data-package] manifest={manifest_path}")
    print(f"[post-training-data-package] status={result['status']}")
    return 0 if result["status"] == "export_ready_review_required" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
