from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.decision_evidence_contracts import (
    canonical_digest,
    cross_runtime_canonical_digest,
)
from blueprint_pipeline import (
    task_evaluation_scene_configuration_publication_recovery as recovery,
)
from blueprint_pipeline.task_evaluation_scene_construction_queue import (
    finalize_scene_construction,
)
from tests.test_task_evaluation_scene_configuration_orchestrator import (
    staged_configuration,
)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _reference(seed: str) -> dict[str, object]:
    return {
        "uri": f"https://example.test/{seed}",
        "digest": "sha256:" + seed * 64,
        "size_bytes": 1,
    }


def test_recovers_completed_provider_output_without_repeating_provider(
    tmp_path: Path, monkeypatch
) -> None:
    request, queue, _inputs = staged_configuration(tmp_path)
    pending = next((queue / "pending").glob("*.json"))
    envelope = json.loads(pending.read_text(encoding="utf-8"))
    portable = {
        **envelope,
        "control_plane_envelope_digest": envelope["envelope_digest"],
    }
    original_finalization = finalize_scene_construction(
        queue_root=queue,
        envelope=portable,
        terminal_result={
            "status": "blocked",
            "run_id": envelope["run_id"],
            "source_commit": request["expected_production_commit"],
            "configuration_completed": True,
            "configured_scene_published": False,
            "full_byte_service_account_readback_passed": False,
            "continuing_spend_from_this_run": False,
            "blockers": ["scene_configuration_configured_revision_not_published"],
        },
    )
    provider_result = {
        "schema_version": "task_evaluation_scene_configuration_provider_result.v1",
        "status": "completed",
        "run_id": envelope["run_id"],
        "source_commit": request["expected_production_commit"],
        "blockers": [],
        "stage_chain": {
            "result_digest": "sha256:" + "9" * 64,
            "stage_results": [],
        },
        "result_digest": "",
    }
    provider_result["result_digest"] = canonical_digest(
        provider_result, digest_field="result_digest"
    )
    provider_path = tmp_path / "immutable_execution" / "provider-result.json"
    _write(provider_path, provider_result)
    teardown_path = tmp_path / "vast_teardown_manifest.json"
    teardown_path.write_text("teardown", encoding="utf-8")
    original_result = {
        "schema_version": "task_evaluation_scene_configuration_vast_result.v1",
        "status": "blocked",
        "run_id": envelope["run_id"],
        "source_commit": request["expected_production_commit"],
        "bundle_sha256": "sha256:" + "8" * 64,
        "execution_result_path": str(provider_path.resolve()),
        "stage_chain_result_digest": "sha256:" + "9" * 64,
        "teardown_manifest_path": str(teardown_path.resolve()),
        "configuration_completed": True,
        "configured_scene_published": False,
        "full_byte_service_account_readback_passed": False,
        "continuing_spend_from_this_run": False,
        "independent_watchdog": {"provider_absence_confirmed": True},
        "object_store_cleanup": {"all_objects_absent": True},
        "scene_construction_queue_finalization": original_finalization,
        "blockers": ["scene_configuration_configured_revision_not_published"],
        "result_digest": "",
    }
    original_result["result_digest"] = canonical_digest(
        original_result, digest_field="result_digest"
    )
    original_result_path = tmp_path / "original-result.json"
    _write(original_result_path, original_result)
    original_launch = {
        "schema_version": "task_evaluation_launch_receipt.v1",
        "status": "blocked",
        "launch_id": "launch-1",
        "run_id": "launch-1",
        "request_digest": "sha256:" + "7" * 64,
        "launch_profile_digest": "sha256:" + "6" * 64,
        "source_commit": request["expected_production_commit"],
        "binding_digest": "sha256:" + "5" * 64,
        "canonical_allocator": (
            "python -m blueprint_pipeline.paid_resource_allocator gpu-canary"
        ),
        "allocator_exit_code": 2,
        "execute_requested": True,
        "provider_mutation_attempted": True,
        "terminal_evidence": {"status": "blocked", "blockers": ["publication"]},
        "blockers": ["publication"],
        "raw_secret_values_recorded": False,
        "agent_operator_used": False,
        "claim_ceiling": "development_only",
        "receipt_digest": "",
    }
    original_launch["receipt_digest"] = cross_runtime_canonical_digest(
        original_launch, digest_field="receipt_digest"
    )
    original_launch_path = tmp_path / "launch_receipt.json"
    _write(original_launch_path, original_launch)
    bundle_receipt = {
        "source_commit": request["expected_production_commit"],
        "run_id": envelope["run_id"],
        "bundle_sha256": "sha256:" + "8" * 64,
    }
    monkeypatch.setattr(
        recovery,
        "load_scene_configuration_provider_bundle_receipt",
        lambda _path: bundle_receipt,
    )
    monkeypatch.setattr(
        recovery, "_portable_construction_envelope", lambda _receipt: portable
    )

    def publish(**kwargs):
        root = kwargs["output_root"]
        root.mkdir()
        revision_path = root / "revision.json"
        publication_path = root / f"{recovery.PUBLICATION_RESULT_SCHEMA_VERSION}.json"
        revision_path.write_text("revision", encoding="utf-8")
        publication_path.write_text("publication", encoding="utf-8")
        revision = _reference("a")
        bundle = _reference("b")
        thumbnail = _reference("c")
        selection = _reference("d")
        offering = {
            "schema_version": "task_evaluation_configured_scene_offering.v1",
            "status": "configured_controls_pending",
            "configuration_run_id": envelope["run_id"],
            "catalog_visibility": "team_only",
            "presentation": {
                "task_thumbnail": thumbnail,
                "selection_receipt": selection,
            },
            "evaluation_preparation_binding": {
                "configured_scene_revision": revision,
                "configured_scene_revision_digest": "sha256:" + "e" * 64,
                "configured_scene_bundle": bundle,
            },
            "offering_digest": "",
        }
        offering["offering_digest"] = canonical_digest(
            offering, digest_field="offering_digest"
        )
        return {
            "status": "configured_scene_published",
            "full_byte_service_account_readback_passed": True,
            "configured_scene_revision": {"path": str(revision_path)},
            "configured_scene_revision_reference": revision,
            "configured_scene_revision_digest": "sha256:" + "e" * 64,
            "configured_scene_bundle_reference": bundle,
            "task_thumbnail_reference": thumbnail,
            "task_thumbnail_selection": {"camera_id": "camera-1"},
            "task_thumbnail_selection_receipt_reference": selection,
            "configured_scene_offering": offering,
            "result_digest": "sha256:" + "f" * 64,
        }

    monkeypatch.setattr(recovery, "_publish_completed_configuration", publish)
    output = tmp_path / "recovery"
    result = recovery.recover_completed_configuration_publication(
        bundle_receipt_path=tmp_path / "unused-bundle-receipt.json",
        provider_result_path=provider_path,
        original_result_path=original_result_path,
        original_launch_receipt_path=original_launch_path,
        queue_root=queue,
        output_root=output,
        recovery_source_commit="a" * 40,
    )

    assert result["status"] == "completed"
    assert result["provider_execution_repeated"] is False
    recovered_result = json.loads(
        Path(result["recovered_result"]["path"]).read_text(encoding="utf-8")
    )
    recovered_launch = json.loads(
        Path(result["recovered_launch_receipt"]["path"]).read_text(encoding="utf-8")
    )
    assert recovered_result["configured_scene_published"] is True
    assert recovered_result["publication_recovery"]["provider_execution_repeated"] is False
    assert recovered_launch["status"] == "completed"
    assert recovered_launch["allocator_exit_code"] == 2
    assert recovered_launch["publication_recovery"]["original_terminal_receipt_digest"] == (
        original_launch["receipt_digest"]
    )
    assert original_result_path.read_text(encoding="utf-8")
    assert original_launch_path.read_text(encoding="utf-8")
