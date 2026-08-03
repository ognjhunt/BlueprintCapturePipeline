from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import jsonschema

from blueprint_pipeline.arkit_raw_contract_validation import (
    build_arkit_raw_contract_validation,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.post_capture_evidence_spine import (
    GEOMETRY_QUALIFICATION_SCHEMA,
    PLACEMENT_MEASUREMENT_SCHEMA,
    PostCaptureEvidenceError,
    build_derived_site_geometry,
    build_native_3dgs_candidate,
    build_native_3dgs_candidate_from_canonical,
    build_native_3dgs_candidate_from_teleport,
    build_policy_execution_decision,
    build_qualified_robot_placement,
    build_qualified_site_geometry,
    build_registered_site_reconstruction,
    build_registration_qualification_from_canonical,
    build_scene_composition_decision,
    build_source_profile,
    build_task_robot_selection,
    run_post_capture_evidence_spine,
)


def _sha(value: str) -> str:
    return "sha256:" + value * 64


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _finalize(value: dict, field: str) -> dict:
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _raw_source(tmp_path: Path, *, payload: bytes = b"retained raw source") -> tuple[Path, dict]:
    root = tmp_path / ("source-" + hashlib.sha256(payload).hexdigest()[:8])
    root.mkdir()
    source_path = root / "capture.bin"
    source_path.write_bytes(payload)
    receipt = build_arkit_raw_contract_validation(
        intake_id="capture-raw-v32",
        source_capture_digest=canonical_digest(
            {"source_file_digest": _file_digest(source_path)}
        ),
        source_artifact_digests={"capture.bin": _file_digest(source_path)},
        implementation_digest=_sha("1"),
        source_commit_sha="a" * 40,
        runtime_identity="test-runtime",
        runtime_digest=_sha("2"),
        frozen_split_digest=_sha("3"),
        metric_scaffold_digest=_sha("4"),
        reconstruction_dataset_export_digest=_sha("5"),
        coordinate_frame_declaration={
            "frame": "arkit_world",
            "units": "meters",
            "up_axis": "Y",
            "handedness": "right_handed",
        },
        retained_frame_count=2,
        dropped_attempt_count=0,
        depth_confidence_pair_count=2,
        authority_used={"local_processing_authorized": True},
        timestamp="2026-08-03T13:00:00Z",
    )
    return root, receipt


def _depth_result(tmp_path: Path, source_capture_digest: str) -> tuple[Path, dict]:
    root = tmp_path / "depth"
    root.mkdir()
    surface = root / "surface.json"
    surface.write_text('{"observed":true,"holes":["unseen"]}\n', encoding="utf-8")
    value = {
        "schema_version": "arkit_depth_surface_compilation_result.v1",
        "source_capture_digest": source_capture_digest,
        "surface_asset": {
            "relative_path": "surface.json",
            "digest": _file_digest(surface),
        },
        "coordinate_frame_declaration": {
            "frame": "arkit_world",
            "units": "meters",
        },
        "metric_scale_status": "sensor_metric_unvalidated",
        "observed_region_ids": ["observed-frusta"],
        "unsupported_region_ids": ["unseen-regions"],
        "accepted_high_confidence_pixel_count": 20,
        "rejected_or_missing_pixel_count": 12,
        "discontinuity_rejected_triangle_count": 3,
        "generated_fill_used": False,
    }
    return root, _finalize(value, "arkit_depth_surface_compilation_result_digest")


def _qualified_geometry(tmp_path: Path, source: dict) -> dict:
    depth_root, depth = _depth_result(tmp_path, source["source_capture_digest"])
    candidate = build_derived_site_geometry(
        source_profile=source,
        depth_surface_result=depth,
        artifact_root=depth_root,
    )
    qualification = _finalize(
        {
            "schema_version": GEOMETRY_QUALIFICATION_SCHEMA,
            "status": "qualified",
            "derived_site_geometry_digest": candidate["derived_site_geometry_digest"],
            "geometry_asset_digest": candidate["geometry_asset_digest"],
            "collider_candidate_digest": candidate["collider_candidate_digest"],
            "metric_scale_qualified": True,
            "collision_geometry_qualified": True,
            "isaac_contact_qualified": False,
            "qualifier_identity": "independent-geometry-gate",
            "candidate_may_self_qualify": False,
            "blockers": [],
        },
        "geometry_qualification_digest",
    )
    return build_qualified_site_geometry(
        geometry_candidate=candidate,
        independent_qualification=qualification,
    )


def _appearance(source: dict) -> dict:
    receipt = _finalize(
        {
            "schema_version": "canonical_3dgs_quality_comparison.v1",
            "status": "quality_winner_selected",
            "source_capture_digest": source["source_capture_digest"],
        },
        "canonical_3dgs_quality_comparison_digest",
    )
    return build_native_3dgs_candidate(
        source_profile=source,
        provider_receipt=receipt,
        appearance_asset_digest=_sha("6"),
        provider_identity="canonical-3dgs-worker",
        provider_receipt_digest_field="canonical_3dgs_quality_comparison_digest",
        full_resolution_appearance_preserved=True,
    )


def _registered(tmp_path: Path, source: dict) -> dict:
    geometry = _qualified_geometry(tmp_path, source)
    appearance = _appearance(source)
    qualification = _finalize(
        {
            "schema_version": "scene_registration_qualification.v1",
            "status": "qualified",
            "source_profile_digest": source["source_profile_digest"],
            "native_3dgs_candidate_digest": appearance[
                "native_3dgs_candidate_digest"
            ],
            "appearance_asset_digest": appearance["appearance_asset_digest"],
            "derived_site_geometry_digest": geometry["derived_site_geometry_digest"],
            "geometry_asset_digest": geometry["geometry_asset_digest"],
            "scene_registration_digest": _sha("7"),
            "registration_transform_digest": _sha("8"),
            "residual_measurement_digest": _sha("9"),
            "qualifier_identity": "independent-registration-evaluator",
            "candidate_may_self_qualify": False,
        },
        "registration_qualification_digest",
    )
    return build_registered_site_reconstruction(
        source_profile=source,
        appearance_candidate=appearance,
        site_geometry=geometry,
        registration_qualification=qualification,
    )


def _target(reconstruction: dict, *, interaction: bool = False) -> dict:
    target = {
        "schema_version": "automatic_task_target_orchestration.v1",
        "status": "selected_target_ready",
        "reconstruction_digest": reconstruction["reconstruction_digest"],
        "analysis_appearance_digest": reconstruction["appearance_asset_digest"],
        "selected_target": {
            "proposal_id": "sink-target",
            "task_family": "rigid_object_pick_place" if interaction else "inspection",
            "task_class": "rigid_pick_place" if interaction else "visual_perception",
            "target_binding_digest": _sha("a"),
            "candidate_self_authorized": False,
        },
        "task_zone_asset_requirement": {
            "verified_simready_asset_required": interaction,
        },
    }
    return _finalize(target, "target_orchestration_digest")


def _placement(target: dict) -> tuple[dict, dict, dict]:
    selection = build_task_robot_selection(target)
    candidate = _finalize(
        {
            "schema_version": "external_scene_robot_placement_candidate.v1",
            "robot_id": selection["robot_id"],
            "target_binding_digest": selection["target_binding_digest"],
            "robot_pose_xyzyaw_collision_stage": [0.0, 0.0, 0.0, 0.0],
            "producing_method": "analytic-placement-producer",
        },
        "placement_proposal_digest",
    )
    measurement = _finalize(
        {
            "schema_version": PLACEMENT_MEASUREMENT_SCHEMA,
            "status": "qualified",
            "placement_candidate_digest": candidate["placement_proposal_digest"],
            "robot_selection_digest": selection["robot_selection_digest"],
            "target_binding_digest": selection["target_binding_digest"],
            "robot_id": selection["robot_id"],
            "reachable": True,
            "footprint_clear": True,
            "collision_aware": True,
            "source_collider_qualified": True,
            "qualifier_identity": "independent-placement-gate",
            "candidate_may_self_qualify": False,
            "blockers": [],
        },
        "placement_qualification_digest",
    )
    qualified = build_qualified_robot_placement(
        placement_candidate=candidate,
        robot_selection=selection,
        independent_qualification=measurement,
    )
    return selection, candidate, qualified


def test_source_profile_verifies_raw_bytes_and_detects_tampering(tmp_path: Path) -> None:
    root, receipt = _raw_source(tmp_path)
    source = build_source_profile(source_artifact=receipt, source_root=root)
    assert source["status"] == "admitted_blueprint_raw_contract"
    assert source["source_bytes_verified"] is True
    assert source["claim_boundary"]["blueprint_raw_contract_truth"] is True

    (root / "capture.bin").write_bytes(b"tampered")
    with pytest.raises(PostCaptureEvidenceError, match="source_byte_0_digest_mismatch"):
        build_source_profile(source_artifact=receipt, source_root=root)


def test_geometry_preserves_holes_and_requires_independent_qualification(
    tmp_path: Path,
) -> None:
    root, receipt = _raw_source(tmp_path)
    source = build_source_profile(source_artifact=receipt, source_root=root)
    depth_root, depth = _depth_result(tmp_path, source["source_capture_digest"])
    geometry = build_derived_site_geometry(
        source_profile=source,
        depth_surface_result=depth,
        artifact_root=depth_root,
    )
    assert geometry["coverage_and_uncertainty"]["unsupported_region_ids"] == [
        "unseen-regions"
    ]
    assert geometry["coverage_and_uncertainty"]["generated_fill_used"] is False
    assert geometry["qualification_state"]["collision_geometry"] == "unqualified"
    assert geometry["smallest_missing_measurement"]["code"] == (
        "independent_metric_scale_measurement_missing"
    )

    (depth_root / "surface.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(PostCaptureEvidenceError, match="site_geometry_surface_asset_digest_mismatch"):
        build_derived_site_geometry(
            source_profile=source,
            depth_surface_result=depth,
            artifact_root=depth_root,
        )


def test_registration_requires_exact_independent_residual_join(tmp_path: Path) -> None:
    root, receipt = _raw_source(tmp_path)
    source = build_source_profile(source_artifact=receipt, source_root=root)
    geometry = _qualified_geometry(tmp_path, source)
    appearance = _appearance(source)
    missing = build_registered_site_reconstruction(
        source_profile=source,
        appearance_candidate=appearance,
        site_geometry=geometry,
        registration_qualification=None,
    )
    assert missing["status"] == "abstained"
    assert missing["smallest_missing_measurement"]["code"] == (
        "splat_metric_frame_registration_missing"
    )

    registration = _finalize(
        {
            "schema_version": "scene_registration_qualification.v1",
            "status": "qualified",
            "source_profile_digest": source["source_profile_digest"],
            "native_3dgs_candidate_digest": appearance[
                "native_3dgs_candidate_digest"
            ],
            "appearance_asset_digest": _sha("f"),
            "derived_site_geometry_digest": geometry["derived_site_geometry_digest"],
            "geometry_asset_digest": geometry["geometry_asset_digest"],
            "scene_registration_digest": _sha("7"),
            "registration_transform_digest": _sha("8"),
            "residual_measurement_digest": _sha("9"),
            "qualifier_identity": "independent-registration-evaluator",
            "candidate_may_self_qualify": False,
        },
        "registration_qualification_digest",
    )
    with pytest.raises(PostCaptureEvidenceError, match="scene_registration_exact_join_mismatch"):
        build_registered_site_reconstruction(
            source_profile=source,
            appearance_candidate=appearance,
            site_geometry=geometry,
            registration_qualification=registration,
        )


def test_canonical_registered_appearance_is_adapted_without_claim_upgrade(
    tmp_path: Path,
) -> None:
    root, receipt = _raw_source(tmp_path)
    source = build_source_profile(source_artifact=receipt, source_root=root)
    geometry = _qualified_geometry(tmp_path, source)
    measurement = _finalize(
        {
            "schema_version": "canonical_3dgs_registration_measurement.v1",
            "status": "qualified",
            "source_capture_digest": source["source_capture_digest"],
            "appearance_asset_digest": _sha("6"),
            "method_id": "independent-correspondence-measurement-v1",
            "transform_appearance_to_site": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "residual_summary": {"rmse_m": 0.001, "p95_m": 0.002},
            "thresholds_m": {"maximum_rmse_m": 0.01, "maximum_p95_m": 0.01},
            "registration_gate_passed": True,
        },
        "canonical_3dgs_registration_measurement_digest",
    )
    registered_appearance = _finalize(
        {
            "schema_version": "canonical_registered_appearance.v1",
            "status": "qualified",
            "source_profile_digest": _sha("c"),
            "source_capture_digest": source["source_capture_digest"],
            "appearance_format": "native_3dgs",
            "appearance_asset_digest": _sha("6"),
            "full_resolution_appearance_preserved": True,
            "registration_status": "qualified",
            "scene_registration_digest": measurement[
                "canonical_3dgs_registration_measurement_digest"
            ],
        },
        "canonical_registered_appearance_digest",
    )
    appearance = build_native_3dgs_candidate_from_canonical(
        source_profile=source,
        registered_appearance=registered_appearance,
    )
    assert appearance["appearance_is_geometry_authority"] is False
    qualification = build_registration_qualification_from_canonical(
        source_profile=source,
        appearance_candidate=appearance,
        site_geometry=geometry,
        registered_appearance=registered_appearance,
        registration_measurement=measurement,
    )
    reconstruction = build_registered_site_reconstruction(
        source_profile=source,
        appearance_candidate=appearance,
        site_geometry=geometry,
        registration_qualification=qualification,
    )
    assert reconstruction["status"] == "qualified"
    assert reconstruction["claim_boundary"]["appearance_used_as_dynamics_authority"] is False

    tampered = dict(measurement)
    tampered["registration_gate_passed"] = False
    with pytest.raises(PostCaptureEvidenceError, match="canonical_registration_measurement_invalid"):
        build_registration_qualification_from_canonical(
            source_profile=source,
            appearance_candidate=appearance,
            site_geometry=geometry,
            registered_appearance=registered_appearance,
            registration_measurement=tampered,
        )


def test_teleport_receipts_bind_the_exact_native_ply_without_qualification(
    tmp_path: Path,
) -> None:
    root, receipt = _raw_source(tmp_path)
    source = build_source_profile(source_artifact=receipt, source_root=root)
    imported = _finalize(
        {
            "schema_version": "provider_splat_import_receipt.v1",
            "status": "imported_provider_appearance_candidate_only",
            "provider_identity": "teleport",
            "source_capture_digest": source["source_capture_digest"],
            "provider_native_output_preserved_unchanged": True,
            "imported_assets": [
                {
                    "artifact_kind": "splat_ply",
                    "digest": _sha("e"),
                    "relative_path": "provider-import/native.ply",
                }
            ],
        },
        "provider_splat_import_receipt_digest",
    )
    run = _finalize(
        {
            "schema_version": "teleport_provider_run_receipt.v1",
            "status": "succeeded_unqualified",
            "provider_identity": "teleport",
            "provider_splat_import_receipt_digest": imported[
                "provider_splat_import_receipt_digest"
            ],
            "metric_scale_proven": False,
            "collision_geometry_validated": False,
        },
        "teleport_provider_run_receipt_digest",
    )
    candidate = build_native_3dgs_candidate_from_teleport(
        source_profile=source,
        run_receipt=run,
        import_receipt=imported,
    )
    assert candidate["appearance_asset_digest"] == _sha("e")
    assert candidate["claim_boundary"]["appearance_quality_qualified"] is False
    assert candidate["appearance_is_geometry_authority"] is False

    wrong_source = dict(imported)
    wrong_source["source_capture_digest"] = _sha("f")
    wrong_source = _finalize(wrong_source, "provider_splat_import_receipt_digest")
    rebound_run = dict(run)
    rebound_run["provider_splat_import_receipt_digest"] = wrong_source[
        "provider_splat_import_receipt_digest"
    ]
    rebound_run = _finalize(rebound_run, "teleport_provider_run_receipt_digest")
    with pytest.raises(PostCaptureEvidenceError, match="native_3dgs_source_capture_mismatch"):
        build_native_3dgs_candidate_from_teleport(
            source_profile=source,
            run_receipt=rebound_run,
            import_receipt=wrong_source,
        )


def test_target_robot_placement_scene_and_authorization_are_independent(
    tmp_path: Path,
) -> None:
    root, receipt = _raw_source(tmp_path)
    source = build_source_profile(source_artifact=receipt, source_root=root)
    reconstruction = _registered(tmp_path, source)
    target = _target(reconstruction)
    selection, _, placement = _placement(target)
    assert selection["robot_id"] == "franka_panda"
    assert placement["status"] == "qualified"

    composition = build_scene_composition_decision(
        target_orchestration=target,
        qualified_placement=placement,
    )
    assert composition["status"] == "qualified"
    assert composition["task_zone_replacement"]["status"] == "not_required"

    metric = _finalize(
        {
            "schema_version": "task_outcome_metric_spec.v1",
            "metric_id": "inspection-distance",
        },
        "metric_spec_digest",
    )
    candidates = [
        _finalize(
            {
                "schema_version": "learned_policy_candidate_identity.v1",
                "candidate_id": f"policy-{index}",
            },
            "policy_identity_digest",
        )
        for index in range(5)
    ]
    route = _finalize(
        {
            "schema_version": "task_site_measurement_routing_decision.v1",
            "status": "route_selected",
            "agent_selected_route": False,
            "agent_qualified_method": False,
            "selected_route": {"stages": []},
        },
        "routing_decision_digest",
    )
    authorization = build_policy_execution_decision(
        routing_decision=route,
        qualified_placement=placement,
        scene_composition=composition,
        task_metric=metric,
        policy_candidates=candidates,
        authorizer_identity="independent-policy-admission",
    )
    assert authorization["policy_execution_authorized"] is True
    assert authorization["physical_robot_execution_authorized"] is False
    assert authorization["agent_or_provider_self_authorized"] is False

    self_authorizing_route = dict(route)
    self_authorizing_route["selected_route"] = {
        "stages": [{"method_id": "provider-method"}]
    }
    _finalize(self_authorizing_route, "routing_decision_digest")
    with pytest.raises(PostCaptureEvidenceError, match="policy_authorizer_independence_invalid"):
        build_policy_execution_decision(
            routing_decision=self_authorizing_route,
            qualified_placement=placement,
            scene_composition=composition,
            task_metric=metric,
            policy_candidates=candidates,
            authorizer_identity="provider-method",
        )


def test_interaction_requires_exact_simready_task_zone(tmp_path: Path) -> None:
    root, receipt = _raw_source(tmp_path)
    source = build_source_profile(source_artifact=receipt, source_root=root)
    target = _target(_registered(tmp_path, source), interaction=True)
    _, _, placement = _placement(target)
    result = build_scene_composition_decision(
        target_orchestration=target,
        qualified_placement=placement,
    )
    assert result["status"] == "abstained"
    assert result["smallest_missing_measurement"]["code"] == (
        "qualified_simready_task_zone_missing"
    )


def test_content_addressed_run_is_idempotent_and_upstream_change_invalidates(
    tmp_path: Path,
) -> None:
    root, receipt = _raw_source(tmp_path, payload=b"source-a")
    first = run_post_capture_evidence_spine(
        run_id="site-task-run",
        source_artifact=receipt,
        source_root=root,
        output_root=tmp_path / "runs",
    )
    repeated = run_post_capture_evidence_spine(
        run_id="site-task-run",
        source_artifact=receipt,
        source_root=root,
        output_root=tmp_path / "runs",
    )
    assert repeated["run_root"] == first["run_root"]
    assert repeated["manifest"] == first["manifest"]
    assert first["terminal"]["smallest_missing_measurement"]["code"] == (
        "native_3dgs_appearance_missing"
    )

    changed_root, changed_receipt = _raw_source(tmp_path, payload=b"source-b")
    changed = run_post_capture_evidence_spine(
        run_id="site-task-run",
        source_artifact=changed_receipt,
        source_root=changed_root,
        output_root=tmp_path / "runs",
    )
    assert changed["run_root"] != first["run_root"]
    assert changed["manifest"]["source_profile_digest"] != first["manifest"][
        "source_profile_digest"
    ]


def test_one_run_executes_producer_chain_through_routing_gate(tmp_path: Path) -> None:
    source_root, receipt = _raw_source(tmp_path)
    source = build_source_profile(source_artifact=receipt, source_root=source_root)
    depth_root, depth = _depth_result(tmp_path, source["source_capture_digest"])
    geometry_candidate = build_derived_site_geometry(
        source_profile=source,
        depth_surface_result=depth,
        artifact_root=depth_root,
    )
    geometry_measurement = _finalize(
        {
            "schema_version": GEOMETRY_QUALIFICATION_SCHEMA,
            "status": "qualified",
            "derived_site_geometry_digest": geometry_candidate[
                "derived_site_geometry_digest"
            ],
            "geometry_asset_digest": geometry_candidate["geometry_asset_digest"],
            "collider_candidate_digest": geometry_candidate[
                "collider_candidate_digest"
            ],
            "metric_scale_qualified": True,
            "collision_geometry_qualified": True,
            "isaac_contact_qualified": False,
            "qualifier_identity": "independent-geometry-gate",
            "candidate_may_self_qualify": False,
            "blockers": [],
        },
        "geometry_qualification_digest",
    )
    geometry = build_qualified_site_geometry(
        geometry_candidate=geometry_candidate,
        independent_qualification=geometry_measurement,
    )
    appearance = _appearance(source)
    registration_measurement = _finalize(
        {
            "schema_version": "scene_registration_qualification.v1",
            "status": "qualified",
            "source_profile_digest": source["source_profile_digest"],
            "native_3dgs_candidate_digest": appearance[
                "native_3dgs_candidate_digest"
            ],
            "appearance_asset_digest": appearance["appearance_asset_digest"],
            "derived_site_geometry_digest": geometry["derived_site_geometry_digest"],
            "geometry_asset_digest": geometry["geometry_asset_digest"],
            "scene_registration_digest": _sha("7"),
            "registration_transform_digest": _sha("8"),
            "residual_measurement_digest": _sha("9"),
            "qualifier_identity": "independent-registration-evaluator",
            "candidate_may_self_qualify": False,
        },
        "registration_qualification_digest",
    )
    reconstruction = build_registered_site_reconstruction(
        source_profile=source,
        appearance_candidate=appearance,
        site_geometry=geometry,
        registration_qualification=registration_measurement,
    )
    target = _target(reconstruction)
    selection, placement_candidate, _ = _placement(target)
    placement_measurement = _finalize(
        {
            "schema_version": PLACEMENT_MEASUREMENT_SCHEMA,
            "status": "qualified",
            "placement_candidate_digest": placement_candidate[
                "placement_proposal_digest"
            ],
            "robot_selection_digest": selection["robot_selection_digest"],
            "target_binding_digest": selection["target_binding_digest"],
            "robot_id": selection["robot_id"],
            "reachable": True,
            "footprint_clear": True,
            "collision_aware": True,
            "source_collider_qualified": True,
            "qualifier_identity": "independent-placement-gate",
            "candidate_may_self_qualify": False,
            "blockers": [],
        },
        "placement_qualification_digest",
    )
    result = run_post_capture_evidence_spine(
        run_id="producer-chain",
        source_artifact=receipt,
        source_root=source_root,
        output_root=tmp_path / "producer-chain-runs",
        appearance_candidate=appearance,
        depth_surface_result=depth,
        depth_surface_root=depth_root,
        geometry_qualification=geometry_measurement,
        registration_qualification=registration_measurement,
        target_orchestration=target,
        placement_candidate=placement_candidate,
        placement_qualification=placement_measurement,
    )
    run_root = Path(result["run_root"])
    assert (run_root / "01_source_profile.json").is_file()
    assert (run_root / "02_native_3dgs_candidate.json").is_file()
    assert (run_root / "03_derived_site_geometry.json").is_file()
    assert (run_root / "04_registered_site_reconstruction.json").is_file()
    assert (run_root / "05_target_orchestration.json").is_file()
    assert (run_root / "06_task_robot_selection.json").is_file()
    assert (run_root / "07_robot_placement.json").is_file()
    assert (run_root / "08_scene_composition.json").is_file()
    assert result["terminal"]["terminal_stage"] == "task_site_engine_routing"
    assert result["terminal"]["smallest_missing_measurement"]["code"] == (
        "task_site_engine_route_inputs_missing"
    )


def test_produced_artifacts_validate_against_spine_schema(tmp_path: Path) -> None:
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/post_capture_evidence_spine.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    reconstruction_schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/registered_site_reconstruction.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    root, receipt = _raw_source(tmp_path)
    source = build_source_profile(source_artifact=receipt, source_root=root)
    reconstruction = _registered(tmp_path, source)
    target = _target(reconstruction)
    selection, _, placement = _placement(target)
    composition = build_scene_composition_decision(
        target_orchestration=target,
        qualified_placement=placement,
    )
    result = run_post_capture_evidence_spine(
        run_id="schema-validation",
        source_artifact=receipt,
        source_root=root,
        output_root=tmp_path / "schema-run",
    )
    for artifact in (source, reconstruction, selection, placement, composition, result["manifest"]):
        jsonschema.validate(artifact, schema)
    jsonschema.validate(reconstruction, reconstruction_schema)
    abstained_reconstruction = json.loads(
        (
            Path(result["run_root"]) / "04_registered_site_reconstruction.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(abstained_reconstruction, reconstruction_schema)


@pytest.mark.slow
@pytest.mark.external_data
def test_real_retained_arkitscenes_40958756_reaches_scientific_abstention(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).parents[1]
    source_root = repo / "output/public_dataset_smokes/arkitscenes/40958756"
    receipt_path = repo / "docs/evidence/arkitscenes_raw_proxy_40958756_b2d7297f.json"
    if not (source_root / "source/40958756.mov").is_file():
        pytest.skip("retained real ARKitScenes 40958756 source bytes are not installed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    result = run_post_capture_evidence_spine(
        run_id="arkitscenes-40958756-real-post-capture",
        source_artifact=receipt,
        source_root=source_root,
        output_root=tmp_path / "real-runs",
    )
    source = json.loads(
        (Path(result["run_root"]) / "01_source_profile.json").read_text(encoding="utf-8")
    )
    assert len(source["verified_source_files"]) == 6
    assert source["source_kind"] == "arkitscenes_public_dataset_proxy"
    assert source["claim_boundary"]["blueprint_raw_contract_truth"] is False
    assert result["terminal"]["terminal_stage"] == "reconstruction_registration"
    assert result["terminal"]["smallest_missing_measurement"]["code"] == (
        "native_3dgs_appearance_missing"
    )
    assert result["manifest"]["fixture_evidence_used"] is False
    retained_root = (
        repo / "docs/evidence/arkitscenes_40958756_post_capture_2f8e5921"
    )
    assert result["manifest"] == json.loads(
        (retained_root / "post_capture_evidence_run.json").read_text(encoding="utf-8")
    )
    assert result["terminal"] == json.loads(
        (retained_root / "terminal_new_site_task_evaluation_run.json").read_text(
            encoding="utf-8"
        )
    )
