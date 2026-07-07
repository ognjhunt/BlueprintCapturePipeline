from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import live_robot_eval_closure as lrec


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_live_evidence_loads_signed_access_manifest_delivery_proof(tmp_path: Path) -> None:
    capture_root = tmp_path / "storage" / "scenes" / "scene-1" / "captures" / "capture-1"
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-signed-access"
    _write_json(
        job_dir / "signed_access_manifest.json",
        {
            "schema_version": "post_training_signed_access_manifest.v1",
            "status": "signed_access_ready",
            "storage_upload_performed": True,
            "signed_urls": ["https://signed.example.test/package.zip"],
            "entitlement_verified": True,
            "buyer_access_check": {
                "entitlement_verified": True,
                "buyer_access_checked": True,
                "buyer_accessible": True,
                "status": "signed_url_minted",
            },
            "operator_attestation": "delivery owner accepted signed buyer access",
        },
    )

    evidence, sources = lrec._load_live_evidence(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request={},
    )
    gate = lrec._signed_delivery_access_gate(evidence)

    assert str(job_dir / "signed_access_manifest.json") in sources
    assert gate["passed"] is True
    assert gate["blockers"] == []
    assert gate["evidence"]["entitlement_verified"] is True
    assert gate["evidence"]["buyer_access_checked"] is True


def test_live_evidence_does_not_upgrade_local_delivery_manifest(tmp_path: Path) -> None:
    capture_root = tmp_path / "storage" / "scenes" / "scene-1" / "captures" / "capture-1"
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-local-access"
    _write_json(
        job_dir / "signed_access_manifest.json",
        {
            "schema_version": "post_training_signed_access_manifest.v1",
            "status": "local_delivery_ready_review_required",
            "storage_upload_performed": False,
            "local_access_paths": ["local/package.zip"],
            "entitlement_verified": False,
            "operator_attestation": "delivery owner reviewed local package copy",
        },
    )

    evidence, _sources = lrec._load_live_evidence(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request={},
    )
    gate = lrec._signed_delivery_access_gate(evidence)

    assert gate["passed"] is False
    assert "signed_delivery_access_not_proven" in gate["blockers"]
    assert "signed_delivery_entitlement_not_verified" in gate["blockers"]


def test_low_level_validation_helpers_cover_edge_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_root = tmp_path / "storage" / "scenes" / "scene-1" / "captures" / "capture-1"
    raw_root = capture_root / "raw"
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-1"
    raw_root.mkdir(parents=True)
    job_dir.mkdir(parents=True)

    assert lrec._string_list(7) == ["7"]
    assert not lrec._valid_cycle_time({"sample_count": -1, "mean_seconds": 1})
    assert lrec._valid_recovery_success(
        {"attempt_count": 2, "success_count": 1, "success_rate": 0.5}
    )
    assert not lrec._valid_uncertainty({"status": "", "sample_count": 1, "mean_score": 0.5})

    missing = []
    for modality in lrec.SUPPORTED_POLICY_MODALITIES:
        missing.extend(lrec._policy_modality_missing_inputs(modality, {}))
    assert "policy_package.policy_api_endpoint.endpoint_url" in missing
    assert "policy_package.docker_container.image_ref" in missing
    assert "policy_package.docker_container.digest" in missing
    assert "policy_package.recorded_action_trace.trace_manifest_uri" in missing
    assert "policy_package.recorded_action_trace.timestamp_alignment" in missing
    assert "policy_package.high_level_skill_trace.ordered_skill_sequence" in missing
    assert "policy_package.teleop_demo.demo_artifact_uri" in missing
    assert "policy_package.teleop_demo.rights_privacy_attestation" in missing
    assert "policy_package.sim_controller_plugin.simulator_framework" in missing
    assert "policy_package.sim_controller_plugin.plugin_uri" in missing

    artifacts, missing_keys, missing_inputs = lrec._policy_modality_local_reference_audit(
        modality="docker_container",
        reference={"output_manifest_uri": "missing/output.json"},
        capture_root=capture_root,
        job_dir=job_dir,
    )
    assert "output_manifest_uri" in artifacts
    assert missing_keys == ["output_manifest_uri"]
    assert missing_inputs == ["policy_package.docker_container.output_manifest_uri_local_file_missing"]
    monkeypatch.setattr(lrec, "_local_reference_path", lambda *args, **kwargs: None)
    artifacts, missing_keys, missing_inputs = lrec._policy_modality_local_reference_audit(
        modality="docker_container",
        reference={"output_manifest_uri": "unresolvable/output.json"},
        capture_root=capture_root,
        job_dir=job_dir,
    )
    assert artifacts == {}
    assert missing_keys == []
    assert missing_inputs == []

    invalid_json = job_dir / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    assert lrec._read_optional_mapping(invalid_json) == {}
    assert lrec._read_optional_any(job_dir / "missing.json") is None
    assert lrec._read_optional_any(invalid_json) is None

    local_video = raw_root / "walkthrough.mov"
    local_video.write_bytes(b"video")
    assert lrec._raw_capture_pointer_path(
        capture_root=capture_root, raw_root=raw_root, value="https://example.invalid/video.mov"
    ) is None
    assert lrec._raw_capture_pointer_path(
        capture_root=capture_root, raw_root=raw_root, value=f"file://{local_video}"
    ) == local_video
    assert lrec._raw_capture_pointer_path(
        capture_root=capture_root, raw_root=raw_root, value=str(local_video)
    ) == local_video
    assert lrec._raw_capture_pointer_path(
        capture_root=capture_root, raw_root=raw_root, value="raw/walkthrough.mov"
    ) == local_video

    summary = lrec._raw_capture_evidence_summary(
        capture_root=capture_root,
        raw_manifest={
            "video_uri": [
                "https://example.invalid/video.mov",
                "walkthrough.mov",
                "missing.mov",
            ],
            "exposure_samples": [{"iso": 100}],
        },
    )
    assert summary["has_capture_evidence"]
    assert summary["remote_pointer_uris"]
    assert summary["local_pointer_files"]
    assert summary["missing_local_pointer_files"]
    assert summary["positive_counts"]["exposure_sample_count"] == 1
    monkeypatch.setattr(lrec, "_raw_capture_pointer_path", lambda *args, **kwargs: None)
    summary = lrec._raw_capture_evidence_summary(
        capture_root=capture_root,
        raw_manifest={"video_uri": ["unresolvable.mov"]},
    )
    assert summary["pointer_fields"]["video_uri"] == ["unresolvable.mov"]

    assert not lrec._engine_mutation_operations_present({})
    assert not lrec._engine_mutation_operations_present(
        {"operations": [{"parameters": {"x": 1}}], "operation_count": 0}
    )
    assert lrec._variation_instance_detail_index([{}, {"instance_id": "i1"}])["i1"] == {
        "concrete_mutation": False,
        "engine_mutations": False,
    }
    assert (
        lrec._scenario_eval_runs_missing_concrete_variation_details(
            rows=[{"baseline_capture_layout": True}],
            variation_instance_details={},
        )
        == []
    )
    assert lrec._scenario_family_variation_names({"variations": "not-a-list"}) == set()
    missing_by_scenario = lrec._missing_required_variations_by_scenario(
        coverage_rows=[
            {"baseline_capture_layout": True, "scenario_id": "s1", "variation_name": "lighting"},
            {"scenario_id": "", "variation_name": ""},
        ],
        required_variation_names=["lighting"],
    )
    assert missing_by_scenario == [
        {"task_id": "", "scenario_id": "s1", "missing_variation_names": ["lighting"]}
    ]
    assert lrec._job_local_artifact_path(job_dir, str(local_video)) == local_video
    assert lrec._cards_count({"cards": [{"id": "1"}, "bad"]}) == 1
    assert lrec._cards_count({"scenarios": [{"id": "1"}, "bad"]}) == 1
    assert lrec._attestation_ok("operator accepted")

    owner_audit = lrec._owner_gpu_proof_manifest_audit(
        {
            "status": "accepted",
            "owner_gpu_simulator_execution_proven": True,
            "owner_system_id": "owner-1",
            "simulator_backend": "mujoco",
            "simulator_version": "1",
            "gpu_model": "a10",
            "proof_path": "proof.json",
            "exit_code": 0,
            "blockers": ["manual_blocker"],
            "missing_inputs": ["trace"],
            "evidence": {field: True for field in lrec.OWNER_GPU_PROOF_REQUIRED_EVIDENCE_FLAGS},
        }
    )
    assert "owner_gpu_proof_manifest_has_blockers" in owner_audit["blockers"]
    assert "owner_gpu_proof_manifest_missing_inputs" in owner_audit["blockers"]


def test_reference_loading_and_webapp_helpers_cover_edge_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_root = tmp_path / "storage" / "scenes" / "scene-1" / "captures" / "capture-1"
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-1"
    automation_dir = capture_root / "pipeline" / "simulation_automation"
    job_dir.mkdir(parents=True)
    automation_dir.mkdir(parents=True)
    monkeypatch.setenv("GCS_ROOT", str(tmp_path / "gcs"))

    local_ref = job_dir / "ref.json"
    _write_json(local_ref, {"ok": True})
    capture_ref = capture_root / "pipeline" / "capture_ref.json"
    _write_json(capture_ref, {"capture": True})
    auto_ref = automation_dir / "auto.json"
    _write_json(auto_ref, {"auto": True})

    assert lrec._local_reference_path("", capture_root=capture_root, job_dir=job_dir) is None
    assert lrec._local_reference_path(
        f"file://{local_ref}", capture_root=capture_root, job_dir=job_dir
    ) == local_ref
    assert str(
        lrec._local_reference_path("gs://bucket/path.json", capture_root=capture_root, job_dir=job_dir)
    ).endswith("path.json")
    assert lrec._local_reference_path(
        "https://example.invalid/ref.json", capture_root=capture_root, job_dir=job_dir
    ) is None
    assert lrec._local_reference_path("ref.json", capture_root=capture_root, job_dir=job_dir) == local_ref

    assert lrec._automation_local_reference_path(
        "", capture_root=capture_root, automation_dir=automation_dir
    ) is None
    assert lrec._automation_local_reference_path(
        f"file://{auto_ref}", capture_root=capture_root, automation_dir=automation_dir
    ) == auto_ref
    assert str(
        lrec._automation_local_reference_path(
            "gs://bucket/auto.json", capture_root=capture_root, automation_dir=automation_dir
        )
    ).endswith("auto.json")
    assert lrec._automation_local_reference_path(
        "https://example.invalid/auto.json",
        capture_root=capture_root,
        automation_dir=automation_dir,
    ) is None
    assert lrec._automation_local_reference_path(
        str(auto_ref), capture_root=capture_root, automation_dir=automation_dir
    ) == auto_ref
    assert lrec._automation_local_reference_path(
        "pipeline/capture_ref.json", capture_root=capture_root, automation_dir=automation_dir
    ) == capture_ref

    assert lrec._load_reference_mapping("ref.json", capture_root=capture_root, job_dir=job_dir) == {
        "ok": True
    }
    assert lrec._merge_evidence({"outer": {"a": 1}}, {"outer": {"b": 2}, "plain": 3}) == {
        "outer": {"a": 1, "b": 2},
        "plain": 3,
    }

    ref_payload = job_dir / "owner-evidence.json"
    _write_json(
        ref_payload,
        {
            "schema_version": lrec.LIVE_ROBOT_EVAL_EVIDENCE_SCHEMA_VERSION,
            "job_id": "job-1",
            "review_acceptance": {"accepted": True},
        },
    )
    evidence, sources = lrec._load_live_evidence(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request={
            "live_eval_closure_evidence": {
                "schema_version": "wrong-schema",
                "job_id": "job-1",
            },
            "owner_rank_fidelity_evidence": {
                "schema_version": lrec.LIVE_ROBOT_EVAL_EVIDENCE_SCHEMA_VERSION,
                "job_id": "other-job",
            },
            "owner_evidence_manifest_uri": "owner-evidence.json",
        },
    )
    assert [item["blocker"] for item in evidence["_input_blockers"]] == [
        "live_closure_evidence_schema_mismatch",
        "live_closure_evidence_job_id_mismatch",
    ]
    assert any(source.startswith("job_request_ref:") for source in sources)
    assert evidence["review_acceptance"]["accepted"] is True

    assert lrec._capture_lineage_from_path_text("") is None
    assert lrec._capture_lineage_from_path_text(
        f"file://{capture_root}"
    ) == ("scene-1", "capture-1")
    assert lrec._capture_lineage_from_path_text(
        "gs://bucket/scenes/scene-2/captures/capture-2"
    ) == ("scene-2", "capture-2")
    assert lrec._capture_lineage_from_path_text("plain/path") is None
    assert lrec._capture_lineage_from_path_text("no/scenes/here") is None
    assert lrec._capture_lineage_from_path_text("scenes/scene-1/not-captures/capture-1") is None

    source_fields = lrec._webapp_route_forwarding_id_source_fields(
        proof={"request_id": "request-1"},
        job_request={
            "source": {"selection_state": {"buyer_request_id": "buyer-1"}},
            "site_submission_id": "site-sub-1",
        },
    )
    assert source_fields["site_submission_id"] == "job_request.site_submission_id"
    assert source_fields["buyer_request_id"] == "job_request.source.selection_state.buyer_request_id"
    assert source_fields["request_id"] == "request_id"

    proof_dir = capture_root / "pipeline" / "webapp_route_forwarding_proof"
    proof_dir.mkdir(parents=True)
    (proof_dir / "bad.json").write_text("{", encoding="utf-8")
    payloads, audits, grounded = lrec._webapp_route_forwarding_source_payloads(
        capture_root=capture_root,
        job_dir=job_dir,
        scene_id="scene-1",
        capture_id="capture-1",
    )
    assert payloads == []
    assert audits == []
    assert grounded == set()

    assert lrec._request_capture_root_matches(
        {"site_package": {"capture_root": f"file://{capture_root}"}},
        capture_root,
    )
    assert not lrec._request_capture_root_matches(
        {"site_package": {"capture_root": "gs://bucket/scenes/scene-1/captures/capture-1"}},
        capture_root,
    )

    real_path_class = lrec.Path

    class RaisingPath:
        def __init__(self, value: str) -> None:
            self.value = value

        def expanduser(self) -> "RaisingPath":
            return self

        def resolve(self) -> Path:
            raise OSError("resolve failed")

    monkeypatch.setattr(lrec, "Path", RaisingPath)
    try:
        assert not lrec._request_capture_root_matches(
            {"site_package": {"capture_root": "/bad/path"}},
            capture_root,
        )
    finally:
        monkeypatch.setattr(lrec, "Path", real_path_class)

    webapp_gate = lrec._webapp_upstream_gate(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request={
            "source": {"system": "webapp", "route": "/sites/scene-1"},
            "site_package": {"capture_root": str(capture_root)},
            "site_submission_id": "placeholder-site",
            "request_id": "scene-1:capture-1",
            "buyer_request_id": "buyer-1",
            "capture_job_id": "capture-1",
        },
        evidence={},
        scene_id="scene-1",
        capture_id="capture-1",
    )
    assert "placeholder_webapp_site_submission_id" in webapp_gate["blockers"]
    assert "generated_capture_id_used_for_webapp_request_id" in webapp_gate["blockers"]
    assert lrec._bool_field_from_sources({"externalUseAllowed": False}, snake_key="x", camel_key="externalUseAllowed") is False


def test_owner_evidence_gates_cover_reference_failure_branches(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    job_dir = capture_root / "jobs" / "job-1"
    job_dir.mkdir(parents=True)

    rights_ref_audit = lrec._rights_evidence_ref_audit(
        scope={
            "evidence_uri_or_path": "missing-rights.json",
            "proof_uri": "https://trusted.invalid/proof.json",
            "clearance_uri_or_path": "https://example.invalid/placeholder-proof.json",
        },
        evidence_scope={},
        capture_root=capture_root,
        job_dir=job_dir,
    )
    assert "job_request.evidence_uri_or_path" in rights_ref_audit["missing_local_ref_keys"]
    assert "job_request.proof_uri" in rights_ref_audit["proven_ref_keys"]
    assert "job_request.clearance_uri_or_path" in rights_ref_audit["invalid_remote_ref_keys"]

    rights_gate = lrec._rights_gate(
        job_request={
            "rights_privacy_scope": {
                "status": "accepted",
                "externalUseAllowed": True,
                "evidence_uri_or_path": "missing-rights.json",
                "proof_uri": "https://example.invalid/placeholder-proof.json",
            }
        },
        evidence={},
        capture_root=capture_root,
        job_dir=job_dir,
    )
    assert "rights_privacy_local_evidence_refs_missing" in rights_gate["blockers"]
    assert "rights_privacy_evidence_refs_invalid_or_placeholder" in rights_gate["blockers"]

    ref_audit = lrec._named_local_ref_audit(
        section={
            "evidence_uri_or_path": "https://trusted.invalid/review.json",
            "methodology_uri_or_path": "https://example.invalid/placeholder-method.json",
        },
        aliases_by_name={
            "evidence_uri_or_path": ("evidence_uri_or_path",),
            "methodology_uri_or_path": ("methodology_uri_or_path",),
        },
        capture_root=capture_root,
        job_dir=job_dir,
    )
    assert "evidence_uri_or_path" in ref_audit["proven_ref_keys"]
    assert "methodology_uri_or_path" in ref_audit["invalid_remote_ref_keys"]

    review_gate = lrec._review_acceptance_gate(
        evidence={"review_acceptance": {"accepted": False}},
        capture_root=capture_root,
        job_dir=job_dir,
    )
    assert "review_acceptance_not_accepted" in review_gate["blockers"]
    review_gate = lrec._review_acceptance_gate(
        evidence={"review_acceptance": {"accepted": True}},
        capture_root=capture_root,
        job_dir=job_dir,
    )
    assert "review_acceptance_owner_evidence_missing" in review_gate["blockers"]
    review_gate = lrec._review_acceptance_gate(
        evidence={
            "review_acceptance": {
                "accepted": True,
                "evidence_uri_or_path": "https://example.invalid/placeholder-review.json",
            }
        },
        capture_root=capture_root,
        job_dir=job_dir,
    )
    assert "review_acceptance_evidence_refs_invalid_or_placeholder" in review_gate["blockers"]

    delivery_gate = lrec._signed_delivery_access_gate(
        {"delivery_access": {"signed_urls": ["https://signed.example"], "entitlementVerified": False}}
    )
    assert "signed_delivery_entitlement_not_verified" in delivery_gate["blockers"]
    missing_entitlement = lrec._signed_delivery_access_gate(
        {
            "delivery_access": {
                "signed_urls": ["https://signed.example"],
                "operator_attestation": "owner accepted signed delivery access",
                "buyer_access_check": {
                    "buyer_access_checked": True,
                    "buyer_accessible": True,
                },
            }
        }
    )
    assert "signed_delivery_entitlement_not_verified" in missing_entitlement["blockers"]
    attestation_only = lrec._signed_delivery_access_gate(
        {
            "delivery_access": {
                "signed_urls": ["https://signed.example"],
                "entitlement_verified": True,
                "operator_attestation": "owner accepted signed delivery access",
            }
        }
    )
    assert (
        "signed_delivery_buyer_access_check_not_executed"
        in attestation_only["blockers"]
    )

    safety_gate = lrec._safety_contact_physics_gate(
        evidence={
            "safety_contact_physics": {
                "physicsContactValidated": False,
                "safetyValidated": False,
                "robotReadinessProven": True,
                "methodology_uri_or_path": "https://example.invalid/placeholder-method.json",
            }
        },
        capture_root=capture_root,
        job_dir=job_dir,
    )
    assert safety_gate["blockers"] == []
    assert "physics_contact_validation_not_proven" in safety_gate["evidence"]["diagnostic_blockers"]
    assert (
        "non_ranking_operational_claim_not_proven"
        in safety_gate["evidence"]["diagnostic_blockers"]
    )
    assert (
        "safety_contact_physics_operator_attestation_missing"
        in safety_gate["evidence"]["diagnostic_blockers"]
    )
    assert (
        "safety_contact_physics_evidence_refs_invalid_or_placeholder"
        in safety_gate["evidence"]["diagnostic_blockers"]
    )


def test_repo_local_gates_cover_missing_and_malformed_artifact_edges(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    job_dir = capture_root / "jobs" / "job-1"
    dataset_dir = capture_root / "pipeline" / "robot_eval_dataset"
    automation_dir = capture_root / "pipeline" / "simulation_automation"
    job_dir.mkdir(parents=True)

    site_gate = lrec._site_capture_gate(
        capture_root=capture_root,
        base_dir=tmp_path,
        scene_id="scene-1",
        capture_id="capture-1",
    )
    assert "missing_capture_descriptor" in site_gate["blockers"]
    assert "missing_raw_manifest" in site_gate["blockers"]

    _write_json(capture_root / "capture_descriptor.json", {"scene_id": "wrong", "capture_id": "bad"})
    _write_json(
        capture_root / "raw" / "manifest.json",
        {"scene_id": "wrong", "capture_id": "bad", "video_uri": "missing.mov"},
    )
    _write_json(
        capture_root / "raw" / "capture_upload_complete.json",
        {"sceneId": "wrong", "captureId": "bad", "status": "failed"},
    )
    site_gate = lrec._site_capture_gate(
        capture_root=capture_root,
        base_dir=tmp_path,
        scene_id="scene-1",
        capture_id="capture-1",
    )
    assert "raw_capture_upload_completion_scene_id_mismatch" in site_gate["blockers"]
    assert "raw_capture_upload_completion_capture_id_mismatch" in site_gate["blockers"]
    assert "raw_capture_upload_completion_status_not_complete" in site_gate["blockers"]
    assert "raw_capture_evidence_local_files_missing" in site_gate["blockers"]
    assert "capture_descriptor_scene_id_mismatch" in site_gate["blockers"]
    assert "raw_manifest_capture_id_mismatch" in site_gate["blockers"]

    _write_json(capture_root / "capture_descriptor.json", {"scene_id": "scene-1", "capture_id": "capture-1"})
    _write_json(capture_root / "raw" / "manifest.json", {"scene_id": "scene-1", "capture_id": "capture-1"})
    _write_json(capture_root / "raw" / "capture_upload_complete.json", {"status": "complete"})
    site_gate = lrec._site_capture_gate(
        capture_root=capture_root,
        base_dir=tmp_path,
        scene_id="scene-1",
        capture_id="capture-1",
    )
    assert "raw_capture_upload_completion_scene_id_missing" in site_gate["blockers"]
    assert "raw_capture_upload_completion_capture_id_missing" in site_gate["blockers"]

    cards_only_path = dataset_dir / "task_cards.json"
    _write_json(cards_only_path, {"task_card_count": 1})
    dataset_gate = lrec._robot_eval_dataset_gate(
        capture_root=capture_root,
        job_dir=job_dir,
        gate_id="task_definitions",
        filename="task_cards.json",
        count_fields=("task_card_count",),
        required_card_fields=lrec.TASK_CARD_REQUIRED_FIELDS,
    )
    assert "task_definitions_missing_card_rows" in dataset_gate["blockers"]

    _write_json(dataset_dir / "scenario_cards.json", {"scenario_card_count": 1, "cards": []})
    _write_json(dataset_dir / "scenario_family_library.json", {"family_count": 1, "families": []})
    _write_json(automation_dir / "scenario_variation_instances.json", {"status": "pending", "instance_count": 1})
    scenario_gate = lrec._scenario_library_gate(capture_root=capture_root, job_dir=job_dir)
    assert "scenario_family_library_empty" in scenario_gate["blockers"]
    assert "scenario_variation_instances_not_completed" in scenario_gate["blockers"]
    _write_json(dataset_dir / "scenario_family_library.json", {"family_count": 0, "families": [{"task_id": "task-1", "variations": []}]})
    _write_json(automation_dir / "scenario_variation_instances.json", {"status": "completed", "instance_count": 0})
    scenario_gate = lrec._scenario_library_gate(capture_root=capture_root, job_dir=job_dir)
    assert "scenario_variation_instances_empty" in scenario_gate["blockers"]

    _write_json(job_dir / "scenario_eval_matrix.json", {"status": "completed", "scenario_eval_run_count": 1, "runs": [{"scenario_eval_run_id": "run-1"}]})
    _write_json(
        job_dir / "robot_pov_observation_manifest.json",
        {
            "status": "completed",
            "observation_count": 1,
            "observations": [
                {
                    "camera": "front",
                    "generated_frame_path": "frames/missing-observation.png",
                    "observation_id": "obs-1",
                    "render_sequence_id": "seq-1",
                    "render_storyboard_id": "story-1",
                    "scenario_id": "scenario-1",
                    "task_id": "task-1",
                    "scenario_eval_run_id": "run-1",
                }
            ],
        },
    )
    _write_json(
        job_dir / "robot_pov_frame_sequence_manifest.json",
        {
            "status": "completed",
            "sequence_count": 2,
            "total_frame_count": 2,
            "sequences": [
                {"sequence_id": "seq-1", "scenario_eval_run_id": "run-1", "frame_paths": ["frames/missing-seq.png"]},
                {"sequence_id": "seq-empty", "scenario_eval_run_id": "run-2", "frame_paths": []},
            ],
        },
    )
    _write_json(
        job_dir / "robot_pov_render_storyboard.json",
        {
            "status": "completed",
            "storyboard_count": 2,
            "storyboards": [
                {"storyboard_id": "story-empty", "scenario_eval_run_id": "run-1", "frames": []},
                {
                    "storyboard_id": "story-2",
                    "scenario_eval_run_id": "run-2",
                    "frames": ["bad", {}, {"frame_path": "frames/missing-story.png"}],
                },
            ],
        },
    )
    pov_gate = lrec._robot_pov_gate(job_dir)
    assert "robot_pov_observation_local_generated_frames_missing" in pov_gate["blockers"]
    assert "robot_pov_frame_sequences_missing_frame_paths" in pov_gate["blockers"]
    assert "robot_pov_frame_sequence_local_files_missing" in pov_gate["blockers"]
    assert "robot_pov_storyboards_missing_frames" in pov_gate["blockers"]
    assert "robot_pov_storyboard_local_frame_files_missing" in pov_gate["blockers"]

    _write_json(job_dir / "eval_cards.json", {"eval_card_count": 1, "cards": []})
    _write_json(job_dir / "scenario_eval_matrix.json", {"status": "completed", "scenario_eval_run_count": 0})
    eval_gate = lrec._scenario_eval_suite_gate(capture_root=capture_root, job_dir=job_dir)
    assert "scenario_eval_matrix_empty" in eval_gate["blockers"]

    _write_json(job_dir / "normalized_attempt_trace.json", {"attempts": ["bad"]})
    labels_gate = lrec._failure_labels_gate(job_dir)
    assert "missing_failure_labels" in labels_gate["blockers"]


def test_policy_plugin_report_and_simulator_gates_cover_remaining_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_root = tmp_path / "capture"
    job_dir = capture_root / "jobs" / "job-1"
    automation_dir = capture_root / "pipeline" / "simulation_automation"
    job_dir.mkdir(parents=True)
    automation_dir.mkdir(parents=True)

    _write_json(
        job_dir / "policy_package_manifest.json",
        {
            "status": "blocked",
            "missing_inputs": ["policy_package.manual_blocker"],
            "selected_modalities": ["unsupported_policy"],
            "modalities": {},
        },
    )
    policy_gate = lrec._policy_interface_gate(capture_root=capture_root, job_dir=job_dir)
    assert "policy_interface_unknown_selected_modalities" in policy_gate["blockers"]
    assert "policy_package.manual_blocker" in policy_gate["blockers"]

    monkeypatch.setattr(lrec, "_automation_local_reference_path", lambda *args, **kwargs: None)
    _write_json(
        automation_dir / "simulator_engine_plugin_registry.json",
        {
            "plugin_count": 1,
            "world_model_plugin_count": 1,
            "plugins": {
                "mujoco": {
                    "adapter_contract_status": "ready",
                    "managed_execution_supported": True,
                    "inputs": {"config": "local-config.json"},
                }
            },
            "world_model_plugins": {
                "worldlabs_world_model": {
                    "adapter_contract_status": "blocked",
                    "managed_execution_supported": False,
                    "source_status": "ready",
                    "inputs": {"config": "local-config.json"},
                }
            },
        },
    )
    plugin_gate = lrec._simulator_plugins_gate(capture_root=capture_root, job_dir=job_dir)
    assert "world_model_engine_plugins_not_ready" in plugin_gate["blockers"]

    _write_json(
        automation_dir / "simulator_engine_plugin_registry.json",
        {
            "plugin_count": 1,
            "world_model_plugin_count": 1,
            "plugins": {},
            "world_model_plugins": {
                "worldlabs_world_model": {
                    "adapter_contract_status": "ready",
                    "managed_execution_supported": True,
                    "source_status": "ready",
                    "inputs": {"config": "https://trusted.invalid/config.json"},
                }
            },
        },
    )
    plugin_gate = lrec._simulator_plugins_gate(capture_root=capture_root, job_dir=job_dir)
    assert "simulator_engine_plugin_registry_missing_required_engines" in plugin_gate["blockers"]

    _write_json(job_dir / "scenario_eval_matrix.json", {"status": "completed", "scenario_eval_run_count": 1, "variation_names_covered": ["lighting"]})
    _write_json(job_dir / "evaluation_result.json", {"status": "completed", "standard_policy_scorecard": {}})
    _write_json(job_dir / "policy_execution_manifest.json", {"status": "completed", "selected_modalities": ["docker_container"], "robot_policy_execution_proven": False})
    _write_json(job_dir / "proof_boundary.json", {"simulator_execution_proven": False})
    report = {
        "status": "draft",
        "scenario_eval": {
            "status": "wrong",
            "scenario_eval_run_count": 2,
            "variation_names_covered": ["glare"],
        },
        "policy_interface": {
            "policy_execution_status": "wrong",
            "selected_modalities": [],
            "robot_policy_execution_proven": True,
        },
        "evaluation_status": "wrong",
        "evaluator_scores": {},
        "live_eval_closure": {},
        "requirement_coverage": {},
        "proof_boundary": {"simulator_execution_proven": True},
        "artifact_paths": {
            "scenario_eval_matrix": "scenario_eval_matrix.json",
            "evaluation_result": "evaluation_result.json",
            "policy_execution_manifest": "policy_execution_manifest.json",
            "policy_execution_trace": "missing_policy_trace.json",
            "deployment_outcome_ledger": "missing_ledger.json",
            "prediction_vs_actual_deployment_summary": "missing_summary.json",
            "proof_boundary": "proof_boundary.json",
        },
        "neutral_eval_harness_flow": ["report_generated"],
    }
    audit = lrec._report_referenced_artifact_audit(report=report, job_dir=job_dir)
    assert audit["artifact_mismatches"]
    assert any(item.get("field") == "evaluator_scores" for item in audit["artifact_mismatches"])
    assert any(
        item.get("field") == "evaluation_result.standard_policy_scorecard"
        for item in audit["artifact_mismatches"]
    )
    _write_json(job_dir / "robot_eval_report.json", report)
    (job_dir / "robot_eval_report.md").write_text("# Report\n", encoding="utf-8")
    report_gate = lrec._report_generation_gate(job_dir)
    assert "robot_eval_report_not_generated" in report_gate["blockers"]
    assert "robot_eval_report_artifact_mismatches" in report_gate["blockers"]

    _write_json(job_dir / "job_request.json", {"simulator_preference": "mujoco_first"})
    _write_json(job_dir / "simulator_service_result.json", {"status": "failed", "framework": "mujoco", "simulators_run": True})
    _write_json(
        job_dir / "normalized_attempt_trace.json",
        {
            "status": "completed",
            "attempt_count": 0,
            "required_scenario_eval_run_count": 1,
            "scenario_eval_run_coverage_complete": False,
            "attempts": [],
        },
    )
    _write_json(job_dir / "scenario_eval_matrix.json", {"scenario_eval_run_count": 1, "runs": [{"scenario_eval_run_id": "run-1"}]})
    sim_gate = lrec._live_simulator_gate(capture_root=capture_root, job_dir=job_dir)
    assert sim_gate["evidence"]["expected_simulator"] == "mujoco"
    assert "simulator_execution_incomplete_scenario_eval_run_coverage" in sim_gate["blockers"]

    _write_json(job_dir / "job_request.json", {"simulator_preference": "isaac"})
    sim_gate = lrec._live_simulator_gate(capture_root=capture_root, job_dir=job_dir)
    assert sim_gate["evidence"]["expected_simulator"] == "isaac_sim"


def test_beta_readiness_schema_edges_and_main_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    job_dir = tmp_path / "job-1"
    job_dir.mkdir()
    _write_json(job_dir / "deployment_outcome_intake_manifest.json", {"schema_version": "wrong"})
    _write_json(job_dir / "deployment_outcome_ledger.json", {"schema_version": "wrong"})
    _write_json(job_dir / "prediction_vs_actual_deployment_summary.json", {"schema_version": "wrong"})
    _write_json(job_dir / "sim_vs_real_calibration_report.json", {"schema_version": "wrong"})

    summary = lrec._robot_team_beta_readiness_summary(
        gates={},
        job_dir=job_dir,
        repo_local_ready=False,
        live_external_ready=False,
        live_end_to_end_verified=False,
    )
    deployment_check = next(
        check for check in summary["checks"] if check["check_id"] == "deployment_outcome_joins"
    )
    assert deployment_check["blockers"] == []
    assert (
        "deployment_outcome_intake_manifest_schema_invalid"
        in deployment_check["evidence"]["diagnostic_blockers"]
    )
    assert (
        "deployment_outcome_ledger_schema_invalid"
        in deployment_check["evidence"]["diagnostic_blockers"]
    )
    assert (
        "prediction_vs_actual_deployment_summary_schema_invalid"
        in deployment_check["evidence"]["diagnostic_blockers"]
    )
    assert (
        "sim_vs_real_calibration_report_schema_invalid"
        in deployment_check["evidence"]["diagnostic_blockers"]
    )

    _write_json(job_dir / "deployment_outcome_intake_manifest.json", {"schema_version": "deployment_outcome_intake_manifest.v1", "status": "weird"})
    _write_json(job_dir / "deployment_outcome_ledger.json", {"schema_version": "deployment_outcome_ledger.v1", "status": "weird"})
    _write_json(job_dir / "prediction_vs_actual_deployment_summary.json", {"schema_version": "prediction_vs_actual_deployment_summary.v1", "status": "weird"})
    _write_json(job_dir / "sim_vs_real_calibration_report.json", {"schema_version": "sim_vs_real_calibration_report.v1", "status": "weird"})
    summary = lrec._robot_team_beta_readiness_summary(
        gates={},
        job_dir=job_dir,
        repo_local_ready=False,
        live_external_ready=False,
        live_end_to_end_verified=False,
    )
    deployment_check = next(
        check for check in summary["checks"] if check["check_id"] == "deployment_outcome_joins"
    )
    assert deployment_check["blockers"] == []
    assert (
        "deployment_outcome_intake_manifest_status_invalid"
        in deployment_check["evidence"]["diagnostic_blockers"]
    )
    assert (
        "deployment_outcome_ledger_status_invalid"
        in deployment_check["evidence"]["diagnostic_blockers"]
    )
    assert (
        "prediction_vs_actual_deployment_summary_status_invalid"
        in deployment_check["evidence"]["diagnostic_blockers"]
    )
    assert (
        "sim_vs_real_calibration_report_status_invalid"
        in deployment_check["evidence"]["diagnostic_blockers"]
    )

    request_path = job_dir / "job_request.json"
    _write_json(request_path, {"job_id": "job-1"})

    def fake_build(**kwargs: object) -> dict[str, object]:
        assert kwargs["job_request"] == {"job_id": "job-1"}
        return {"status": "blocked", "blockers": ["missing"]}

    monkeypatch.setattr(lrec, "build_live_robot_eval_closure_manifest", fake_build)
    assert (
        lrec.main(
            [
                "--capture-root",
                str(tmp_path / "capture"),
                "--job-dir",
                str(job_dir),
                "--job-request",
                str(request_path),
            ]
        )
        == 1
    )
    output = capsys.readouterr().out
    assert "[live-robot-eval-closure] manifest=" in output
    assert "[live-robot-eval-closure] status=blocked" in output
    assert "[live-robot-eval-closure] blockers=1" in output
