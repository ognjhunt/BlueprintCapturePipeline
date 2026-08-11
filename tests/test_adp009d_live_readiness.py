from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from blueprint_pipeline.adp009d_live_readiness import build_live_readiness
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


COMMIT = "a" * 40


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _fixtures(tmp_path: Path) -> dict[str, object]:
    pair_path = tmp_path / "controls/cell/adp009d_control_pair.v1.json"
    pair_path.parent.mkdir(parents=True)
    pair = {
        "schema_version": "adp009d_control_pair.v1",
        "instance_digest": "sha256:243c0e62697da0298081a53c6530cee16cf94cde5a73df08f3773629b52c3001",
        "candidate_policy_queried": False,
        "cell_admitted_for_policy_execution": True,
        "policy_execution_blockers": [],
        "controls": [
            {
                "control_id": "zero_action_negative",
                "control_passed": True,
                "observed_outcome": "never_moved",
            },
            {
                "control_id": "deterministic_scripted_positive",
                "control_passed": True,
                "observed_outcome": "placed_in_target",
            },
        ],
    }
    pair["pair_digest"] = canonical_digest(pair, digest_field="pair_digest")
    pair_path.write_text(json.dumps(pair, sort_keys=True) + "\n", encoding="utf-8")
    release = {
        "schema_version": "task_evaluation_pipeline_release_evidence.v1",
        "status": "passed",
        "source_commit": COMMIT,
        "source_ref": "main",
        "tracked_state": "clean",
    }
    release["release_digest"] = canonical_digest(release, digest_field="release_digest")
    bundle = {
        "schema_version": "adp009d_native_microcheck_bundle.v1",
        "status": "ready",
        "implementation_commit": COMMIT,
        "controls_requested": True,
        "policy_candidate_id": None,
        "scenario_instance_digest": "sha256:243c0e62697da0298081a53c6530cee16cf94cde5a73df08f3773629b52c3001",
        "retry_cap": 0,
        "bundle_sha256": "sha256:" + "b" * 64,
        "asset_bindings": [
            {
                "role": "aura_appearance",
                "sha256": "sha256:4b73dd13e6044b00b59da7737989d79d891ccac157b33411b30ef59542f3e6a2",
                "visual_only": True,
                "collision_authority": False,
            }
        ],
    }
    artifact = {
        "schema_version": "task_evaluation_artifact_manifest.v1",
        "status": "completed",
        "blockers": [],
        "files": [
            {
                "relative_path": "immutable_execution/controls/cell/adp009d_control_pair.v1.json",
                "sha256": _sha(pair_path),
            }
        ],
    }
    artifact["manifest_digest"] = canonical_digest(artifact, digest_field="manifest_digest")
    now = datetime.now(timezone.utc)
    teardown = {
        "schema_version": "vast_teardown_manifest.v1",
        "generated_at": now.isoformat(),
        "status": "completed",
        "runner_gpu_teardown_completed": True,
        "continuing_spend_from_this_run": False,
    }
    guard = {
        "schema_version": "gpu_spend_guard.v1",
        "generated_at": (now + timedelta(seconds=1)).isoformat(),
        "live_instance_count": 0,
        "total_burn_per_hour_usd": 0.0,
        "inventory_results": [
            {"provider": provider, "status": "succeeded", "row_count": 0}
            for provider in ("digitalocean", "runpod", "vast")
        ],
    }
    allocator = {
        "status": "completed",
        "retry_cap": 0,
        "continuing_spend_from_this_run": False,
        "artifact_manifest_path": "artifact_manifest.json",
        "teardown_manifest_path": "vast_teardown_manifest.json",
    }
    return {
        "source_commit": COMMIT,
        "release_evidence": release,
        "bundle_receipt": bundle,
        "allocator_result": allocator,
        "control_pair": pair,
        "control_pair_path": pair_path,
        "artifact_manifest": artifact,
        "teardown_manifest": teardown,
        "provider_zero_guard": guard,
    }


def test_live_readiness_passes_only_with_controls_manifest_teardown_and_zero(tmp_path: Path) -> None:
    receipt = build_live_readiness(**_fixtures(tmp_path))

    assert receipt["status"] == "passed"
    assert receipt["live_execution_enabled"] is True
    assert receipt["blockers"] == []
    assert all(receipt["observations"].values())
    assert receipt["provider_mutation_performed"] is False
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def test_live_readiness_blocks_failed_positive_and_nonzero_provider(tmp_path: Path) -> None:
    values = _fixtures(tmp_path)
    pair = values["control_pair"]
    assert isinstance(pair, dict)
    pair["controls"][1]["control_passed"] = False
    pair["controls"][1]["observed_outcome"] = "never_moved"
    pair["pair_digest"] = canonical_digest(pair, digest_field="pair_digest")
    pair_path = values["control_pair_path"]
    assert isinstance(pair_path, Path)
    pair_path.write_text(json.dumps(pair, sort_keys=True) + "\n", encoding="utf-8")
    artifact = values["artifact_manifest"]
    assert isinstance(artifact, dict)
    artifact["files"][0]["sha256"] = _sha(pair_path)
    artifact["manifest_digest"] = canonical_digest(artifact, digest_field="manifest_digest")
    guard = values["provider_zero_guard"]
    assert isinstance(guard, dict)
    guard["live_instance_count"] = 1
    guard["total_burn_per_hour_usd"] = 0.7
    guard["inventory_results"][2]["row_count"] = 1

    receipt = build_live_readiness(**values)

    assert receipt["status"] == "blocked"
    assert receipt["live_execution_enabled"] is False
    assert "live_readiness_controls_not_passed" in receipt["blockers"]
    assert "live_readiness_provider_zero_invalid" in receipt["blockers"]
