from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.adp_joint_agent_vast import REQUIRED_RETAINED_ARTIFACT_ROLES
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.joint_agent_execution_receipt import (
    JointAgentExecutionReceiptError,
    materialize_joint_agent_execution_receipt,
)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, *, scene_id: str, target_id: str) -> dict[str, Path]:
    repo = tmp_path / "repo"
    evidence = tmp_path / "evidence"
    runtime_root = evidence / "run/immutable_execution"
    repo.mkdir()
    runtime_root.mkdir(parents=True)
    source_sha = "sha256:" + "a" * 64
    packet = {
        "schema_version": "usd_content_joint_agent_packet.v1",
        "scene": {"publisher_scene_id": scene_id, "target_instance_id": target_id},
        "source_asset": {
            "sha256": source_sha,
            "source_receipt_digest": "sha256:" + "b" * 64,
            "connected_component_count": 28,
        },
        "packet_digest": "",
    }
    packet["packet_digest"] = canonical_digest(packet, digest_field="packet_digest")
    packet_path = evidence / "packet.json"
    _write(packet_path, packet)
    bundle = {
        "status": "ready",
        "packet_digest": packet["packet_digest"],
        "input_usd_sha256": source_sha,
        "bundle_sha256": "sha256:" + "c" * 64,
        "freeze_digest": "sha256:" + "d" * 64,
        "review_contract_digest": "sha256:" + "e" * 64,
        "completion_retries": 0,
        "automatic_paid_retry_allowed": False,
        "released_code": {"version": "0.5.2", "commit": "f" * 40},
    }
    bundle_path = evidence / "bundle.json"
    _write(bundle_path, bundle)
    rows = []
    for role in sorted(REQUIRED_RETAINED_ARTIFACT_ROLES):
        path = runtime_root / "retained" / f"{role}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(role.encode())
        rows.append(
            {
                "role": role,
                "relative_path": path.relative_to(runtime_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
        )
    runtime = {
        "schema_version": "adp_joint_agent_result.v1",
        "status": "completed",
        "blockers": [],
        "joint_agent_inference_executed": True,
        "owned_core_publication_executed": True,
        "retry_cap": 0,
        "retained_artifacts": rows,
        "authored_joint_paths": ["/World/fridge/upper_hinge"],
        "candidates_sha256": "sha256:" + "1" * 64,
        "candidate_bounds_sha256": "sha256:" + "2" * 64,
        "review_receipt_sha256": "sha256:" + "3" * 64,
    }
    runtime_path = runtime_root / "adp_joint_agent_result.json"
    _write(runtime_path, runtime)
    teardown = {"status": "completed", "continuing_spend_from_this_run": False}
    teardown_path = evidence / "run/vast_provider_run/vast_teardown_manifest.json"
    _write(teardown_path, teardown)
    run = {
        "schema_version": "adp_joint_agent_vast_run.v1",
        "status": "completed",
        "blockers": [],
        "bundle_sha256": bundle["bundle_sha256"],
        "retry_cap": 0,
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "execution_result_path": str(runtime_path),
        "teardown_manifest_path": str(teardown_path),
        "estimated_cost_usd": 0.5,
        "hard_cap_usd": 2.0,
        "hard_ttl_seconds": 7200,
    }
    run_path = evidence / "run/adp_joint_agent_vast_result.json"
    _write(run_path, run)
    return {
        "repo": repo,
        "evidence": evidence,
        "packet": packet_path,
        "bundle": bundle_path,
        "runtime": runtime_path,
        "run": run_path,
        "output": repo / "receipt.json",
    }


@pytest.mark.parametrize(
    ("scene_id", "target_id"), [("840313", "160"), ("840796", "123")]
)
def test_seals_scene_neutral_topology_candidate(
    tmp_path: Path, scene_id: str, target_id: str
) -> None:
    paths = _fixture(tmp_path, scene_id=scene_id, target_id=target_id)
    receipt = materialize_joint_agent_execution_receipt(
        packet_path=paths["packet"],
        bundle_receipt_path=paths["bundle"],
        runtime_result_path=paths["runtime"],
        run_result_path=paths["run"],
        evidence_root=paths["evidence"],
        repo_root=paths["repo"],
        receipt_output=paths["output"],
    )

    assert receipt["status"] == "executed_topology_candidate_not_simready"
    assert len(receipt["execution"]["retained_artifacts"]) == 5
    assert receipt["provider_run"]["retry_cap"] == 0
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def test_rejects_changed_returned_asset(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, scene_id="840796", target_id="123")
    retained = paths["runtime"].parent / "retained/owned_core_rigged_asset.bin"
    retained.write_bytes(b"changed")

    with pytest.raises(
        JointAgentExecutionReceiptError, match="joint_agent_retained_artifact_invalid"
    ):
        materialize_joint_agent_execution_receipt(
            packet_path=paths["packet"],
            bundle_receipt_path=paths["bundle"],
            runtime_result_path=paths["runtime"],
            run_result_path=paths["run"],
            evidence_root=paths["evidence"],
            repo_root=paths["repo"],
        )


def test_rejects_nonzero_provider_state(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, scene_id="840796", target_id="123")
    run = json.loads(paths["run"].read_text(encoding="utf-8"))
    run["continuing_spend_from_this_run"] = True
    _write(paths["run"], run)

    with pytest.raises(
        JointAgentExecutionReceiptError, match="joint_agent_provider_run_not_completed"
    ):
        materialize_joint_agent_execution_receipt(
            packet_path=paths["packet"],
            bundle_receipt_path=paths["bundle"],
            runtime_result_path=paths["runtime"],
            run_result_path=paths["run"],
            evidence_root=paths["evidence"],
            repo_root=paths["repo"],
        )
