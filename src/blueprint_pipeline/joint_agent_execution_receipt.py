"""Seal one observed Joint Agent execution without promoting it to SimReady."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adp_joint_agent_vast import REQUIRED_RETAINED_ARTIFACT_ROLES
from .decision_evidence_contracts import canonical_digest, canonical_json


SCHEMA_VERSION = "adp_joint_agent_execution_receipt.v1"
RUNTIME_SCHEMA_VERSION = "adp_joint_agent_result.v1"
RUN_SCHEMA_VERSION = "adp_joint_agent_vast_run.v1"
PACKET_SCHEMA_VERSION = "usd_content_joint_agent_packet.v1"


class JointAgentExecutionReceiptError(ValueError):
    """Stable fail-closed Joint Agent execution evidence errors."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _under(path: str | Path, root: Path, code: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_relative_to(root) or resolved.is_symlink():
        raise JointAgentExecutionReceiptError([code])
    return resolved


def _read(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JointAgentExecutionReceiptError([code]) from exc
    if not isinstance(value, dict):
        raise JointAgentExecutionReceiptError([code])
    return value


def _file_record(path: Path, root: Path, *, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def materialize_joint_agent_execution_receipt(
    *,
    packet_path: str | Path,
    bundle_receipt_path: str | Path,
    runtime_result_path: str | Path,
    run_result_path: str | Path,
    evidence_root: str | Path,
    repo_root: str | Path,
    receipt_output: str | Path | None = None,
) -> dict[str, Any]:
    """Verify returned files, exact source joins, zero retry, and teardown."""

    evidence = Path(evidence_root).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve()
    if not evidence.is_dir() or not repo.is_dir():
        raise JointAgentExecutionReceiptError(["joint_agent_approved_root_missing"])
    packet_file = _under(packet_path, evidence, "joint_agent_packet_outside_evidence_root")
    bundle_file = _under(
        bundle_receipt_path, evidence, "joint_agent_bundle_receipt_outside_evidence_root"
    )
    runtime_file = _under(
        runtime_result_path, evidence, "joint_agent_runtime_result_outside_evidence_root"
    )
    run_file = _under(run_result_path, evidence, "joint_agent_run_result_outside_evidence_root")
    packet = _read(packet_file, "joint_agent_packet_invalid")
    bundle = _read(bundle_file, "joint_agent_bundle_receipt_invalid")
    runtime = _read(runtime_file, "joint_agent_runtime_result_invalid")
    run = _read(run_file, "joint_agent_run_result_invalid")

    if (
        packet.get("schema_version") != PACKET_SCHEMA_VERSION
        or packet.get("packet_digest")
        != canonical_digest(packet, digest_field="packet_digest")
    ):
        raise JointAgentExecutionReceiptError(["joint_agent_packet_invalid"])
    source = packet.get("source_asset") or {}
    if (
        bundle.get("status") != "ready"
        or bundle.get("packet_digest") != packet.get("packet_digest")
        or bundle.get("input_usd_sha256") != source.get("sha256")
        or bundle.get("completion_retries") != 0
        or bundle.get("automatic_paid_retry_allowed") is not False
    ):
        raise JointAgentExecutionReceiptError(["joint_agent_bundle_binding_invalid"])
    provider_bundle = _under(
        str(bundle.get("bundle_path") or ""),
        evidence,
        "joint_agent_provider_bundle_outside_evidence_root",
    )
    if (
        not provider_bundle.is_file()
        or provider_bundle.stat().st_size != bundle.get("bundle_size_bytes")
        or _sha256(provider_bundle) != bundle.get("bundle_sha256")
    ):
        raise JointAgentExecutionReceiptError(["joint_agent_provider_bundle_changed"])
    if (
        runtime.get("schema_version") != RUNTIME_SCHEMA_VERSION
        or runtime.get("status") != "completed"
        or runtime.get("blockers")
        or runtime.get("joint_agent_inference_executed") is not True
        or runtime.get("owned_core_publication_executed") is not True
        or runtime.get("retry_cap") != 0
    ):
        raise JointAgentExecutionReceiptError(["joint_agent_runtime_not_completed"])
    if (
        run.get("schema_version") != RUN_SCHEMA_VERSION
        or run.get("status") != "completed"
        or run.get("blockers")
        or run.get("bundle_sha256") != bundle.get("bundle_sha256")
        or run.get("retry_cap") != 0
        or run.get("continuing_spend_from_this_run") is not False
        or run.get("all_staged_objects_absent") is not True
        or Path(str(run.get("execution_result_path") or "")).resolve() != runtime_file
    ):
        raise JointAgentExecutionReceiptError(["joint_agent_provider_run_not_completed"])

    teardown_file = _under(
        str(run.get("teardown_manifest_path") or ""),
        evidence,
        "joint_agent_teardown_outside_evidence_root",
    )
    teardown = _read(teardown_file, "joint_agent_teardown_invalid")
    if teardown.get("continuing_spend_from_this_run") is not False:
        raise JointAgentExecutionReceiptError(["joint_agent_provider_zero_not_proven"])

    root = runtime_file.parent
    rows = runtime.get("retained_artifacts")
    if not isinstance(rows, list):
        raise JointAgentExecutionReceiptError(["joint_agent_retained_artifacts_missing"])
    retained: list[dict[str, Any]] = []
    observed_roles: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise JointAgentExecutionReceiptError(["joint_agent_retained_artifact_invalid"])
        role = str(row.get("role") or "")
        relative = Path(str(row.get("relative_path") or ""))
        path = (root / relative).resolve()
        if (
            role in observed_roles
            or role not in REQUIRED_RETAINED_ARTIFACT_ROLES
            or relative.is_absolute()
            or not path.is_relative_to(root)
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != row.get("size_bytes")
            or _sha256(path) != row.get("sha256")
        ):
            raise JointAgentExecutionReceiptError(
                [f"joint_agent_retained_artifact_invalid:{role or 'missing_role'}"]
            )
        observed_roles.add(role)
        retained.append(_file_record(path, root, role=role))
    if observed_roles != set(REQUIRED_RETAINED_ARTIFACT_ROLES):
        raise JointAgentExecutionReceiptError(["joint_agent_retained_artifact_set_incomplete"])
    retained_by_role = {row["role"]: row for row in retained}
    expected_role_digests = {
        "articulation_candidates": runtime.get("candidates_sha256"),
        "candidate_bounds": runtime.get("candidate_bounds_sha256"),
        "deterministic_articulation_review": runtime.get("review_receipt_sha256"),
        "owned_core_rigged_asset": runtime.get("rigged_usdz_sha256"),
    }
    if any(
        not isinstance(expected, str)
        or retained_by_role[role]["sha256"] != expected
        for role, expected in expected_role_digests.items()
    ):
        raise JointAgentExecutionReceiptError(
            ["joint_agent_runtime_retained_artifact_cross_join_invalid"]
        )

    authored_joint_paths = runtime.get("authored_joint_paths")
    if (
        not isinstance(authored_joint_paths, list)
        or not authored_joint_paths
        or any(not str(path).startswith("/") for path in authored_joint_paths)
    ):
        raise JointAgentExecutionReceiptError(["joint_agent_authored_joint_readback_invalid"])

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "status": "executed_topology_candidate_not_simready",
        "source": {
            "packet_digest": packet["packet_digest"],
            "source_receipt_digest": source.get("source_receipt_digest"),
            "source_asset_sha256": source.get("sha256"),
            "connected_component_count": source.get("connected_component_count"),
        },
        "released_code": bundle.get("released_code"),
        "bundle": {
            "bundle_sha256": bundle.get("bundle_sha256"),
            "provider_bundle": _file_record(
                provider_bundle, evidence, role="provider_bundle"
            ),
            "bundle_receipt": _file_record(
                bundle_file, evidence, role="bundle_receipt"
            ),
            "freeze_digest": bundle.get("freeze_digest"),
            "review_contract_digest": bundle.get("review_contract_digest"),
        },
        "provider_run": {
            "estimated_cost_usd": run.get("estimated_cost_usd"),
            "hard_cap_usd": run.get("hard_cap_usd"),
            "hard_ttl_seconds": run.get("hard_ttl_seconds"),
            "retry_cap": 0,
            "continuing_spend_from_this_run": False,
            "all_staged_objects_absent": True,
            "teardown_manifest": _file_record(
                teardown_file, evidence, role="teardown_manifest"
            ),
        },
        "execution": {
            "runtime_result": _file_record(runtime_file, evidence, role="runtime_result"),
            "retained_artifacts": retained,
            "authored_joint_paths": [str(path) for path in authored_joint_paths],
            "candidates_sha256": runtime.get("candidates_sha256"),
            "candidate_bounds_sha256": runtime.get("candidate_bounds_sha256"),
            "review_receipt_sha256": runtime.get("review_receipt_sha256"),
        },
        "claim_boundary": {
            "joint_agent_model_output_is_not_ground_truth": True,
            "owned_core_topology_is_not_simready_qualification": True,
            "physical_equivalence_proven": False,
        },
        "blockers": [
            "independent_physics_authoring_and_native_simulator_qualification_missing"
        ],
        "raw_secret_values_recorded": False,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    if receipt_output is not None:
        output = _under(
            receipt_output, repo, "joint_agent_execution_receipt_output_outside_repo"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt
