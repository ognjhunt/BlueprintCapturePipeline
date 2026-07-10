from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import robot_eval_dataset as red


class _BlankTruthy:
    def __bool__(self) -> bool:
        return True

    def __str__(self) -> str:
        return "   "


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_scalar_normalizers_cover_string_scalar_and_invalid_values() -> None:
    assert red._string_list("alpha") == ["alpha"]
    assert red._string_list(42) == ["42"]
    assert red._float_triplet(["2.5", None, "bad"], fallback=[1.0, 2.0, 3.0]) == [
        2.5,
        0.0,
        0.0,
    ]
    assert red._float_triplet(object(), fallback=[1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
    assert red._pose_triplet_or_none({"xyz": [1, 2]}) == [1.0, 2.0, 0.0]
    assert red._pose_triplet_or_none({"x": "3", "y": "4"}) == [3.0, 4.0, 0.0]
    assert red._pose_triplet_or_none({}) is None
    assert red._pose_triplet_or_none([1]) is None
    assert red._pose_triplet_or_none([1, "bad"]) is None
    assert red._pose_triplet_or_none([1, float("inf")]) is None
    assert red._number("bad", default=9.0) == 9.0
    assert red._int("bad", default=7) == 7
    assert red._bool(2) is True
    assert red._deterministic_generated_at({}) == red.DETERMINISTIC_DEFAULT_GENERATED_AT
    assert red._site_id_from_inputs({}, {}) is None


def test_scene_frame_helpers_reject_invalid_bounds_and_emit_odd_anchor_pair(
    tmp_path: Path,
) -> None:
    pipeline_dir = tmp_path / "capture" / "pipeline"

    assert red._scene_frame_anchor_pair(pipeline_dir, index=0) == (None, None)

    _write_json(
        pipeline_dir / "simulation_automation" / "scene_frame_estimate.json",
        {"frame": {"bounds": {"min": [0.0, 0.0, 0.0]}}},
    )
    assert red._scene_frame_bounds(pipeline_dir) is None

    _write_json(
        pipeline_dir / "simulation_automation" / "scene_frame_estimate.json",
        {"frame": {"bounds": {"min": [2.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}}},
    )
    assert red._scene_frame_bounds(pipeline_dir) is None

    _write_json(
        pipeline_dir / "simulation_automation" / "scene_frame_estimate.json",
        {"frame": {"bounds": {"min": [0.0, 0.0, 0.0], "max": [10.0, 6.0, 3.0]}}},
    )
    assert red._scene_frame_anchor_pair(pipeline_dir, index=1) == (
        [8.0, 4.8, 0.793],
        [2.0, 1.2, 0.793],
    )


def test_scene_asset_geometry_handles_empty_assets_hints_duplicates_and_navigation(
    tmp_path: Path,
) -> None:
    pipeline_dir = tmp_path / "capture" / "pipeline"
    _write_json(
        pipeline_dir / "simulation_automation" / "scene_asset_inspection.json",
        {"assets": {"not": "a-list"}},
    )
    assert red._object_geometry_from_scene_assets(pipeline_dir) == {}

    empty_pipeline = tmp_path / "empty" / "pipeline"
    _write_json(
        empty_pipeline / "simulation_automation" / "scene_asset_inspection.json",
        {"assets": ["skip-me"]},
    )
    assert red._object_geometry_from_scene_assets(empty_pipeline) == {}

    _write_json(
        pipeline_dir.parent / "raw" / "manifest.json",
        {"taskSteps": ["navigate to selected waypoint"], "zone": "north aisle"},
    )
    _write_json(
        pipeline_dir / "simulation_automation" / "scene_asset_inspection.json",
        {
            "assets": [
                "skip-me",
                {
                    "path": "scene.glb",
                    "semantic_hints": [
                        "navigation_workspace",
                        "navigation_workspace",
                        {"label": "Loose Label"},
                    ],
                    "bounds": {"min": [0.0, 0.0, 0.0], "max": [10.0, 6.0, 2.0]},
                    "collision_evidence": {"proxy_estimated": True},
                },
                {
                    "path": "no-hints.glb",
                    "bounds": {"min": [2.0, 2.0, 0.0], "max": [4.0, 6.0, 2.0]},
                },
            ]
        },
    )

    manifest = red._object_geometry_from_scene_assets(pipeline_dir)

    object_ids = {item["object_id"] for item in manifest["objects"]}
    assert {"navigation_workspace", "loose_label", "scene_asset_2", "selected_waypoint"} <= object_ids
    scene_asset = next(item for item in manifest["objects"] if item["object_id"] == "scene_asset_2")
    assert scene_asset["center_xyz"] == [3.0, 4.0, 1.0]


def test_object_index_and_center_helpers_cover_fallback_and_invalid_geometry() -> None:
    assert red._objects_by_id({"objects": "not-a-list"}) == {}

    objects = red._objects_by_id(
        {
            "objects": [
                "skip",
                {"object_id": _BlankTruthy()},
                {"id": "nested", "placement_bbox": [[0, 0, 0], [2, 4, 6]], "metric_placement_ready": True},
                {"id": "flat", "bbox": [0, 1, 2, 2, 3, 4], "metric_placement_ready": True},
                {"id": "bad", "bbox": [0, "bad", 2, 2, 3, 4], "metric_placement_ready": True},
                {
                    "id": "two_d_proxy",
                    "placement_bbox": [[0, 0, 0], [2, 4, 6]],
                    "collision_hulls": [{"kind": "box"}],
                    "support_surfaces": [{"kind": "plane"}],
                    "metric_placement_ready": False,
                    "physics_ready": False,
                },
            ]
        }
    )

    assert set(objects) == {"nested", "flat", "bad", "two_d_proxy"}
    assert objects["nested"]["center_xyz"] == [1.0, 2.0, 3.0]
    assert objects["flat"]["center_xyz"] == [1.0, 2.0, 3.0]
    assert objects["bad"]["center_xyz"] is None
    assert objects["two_d_proxy"]["center_xyz"] is None
    assert objects["two_d_proxy"]["has_collision_hulls"] is False
    assert objects["two_d_proxy"]["has_support_surfaces"] is False
    index = red._object_index({"objects": [{"id": "two_d_proxy", "metric_placement_ready": False}]})
    assert index["metric_placement_object_count"] == 0
    assert index["physics_covered_object_count"] == 0


@pytest.mark.parametrize(
    ("metadata", "tasks", "object_geometry", "expected"),
    [
        ({"site_type": "customer lab"}, [], {}, "customer lab"),
        ({}, [{"task_text": "assist nurse in hospital hallway"}], {}, "hospital hallway"),
        ({}, [{"task_text": "move pallet at loading dock"}], {}, "loading dock"),
        ({}, [{"task_text": "deliver tote to assembly line station"}], {}, "factory line-side station"),
        ({}, [{"task_text": "navigate warehouse rack aisle"}], {}, "warehouse aisle"),
        ({}, [], {}, "unknown_site_type"),
    ],
)
def test_site_type_inference_covers_explicit_and_keyword_fallbacks(
    metadata: dict[str, object],
    tasks: list[dict[str, object]],
    object_geometry: dict[str, object],
    expected: str,
) -> None:
    assert (
        red._infer_site_type(
            metadata=metadata,
            tasks=tasks,
            object_geometry_manifest=object_geometry,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("task_id", "task_text", "task_category", "expected"),
    [
        ("", "move from conveyor to cart", "generic", "cart_to_conveyor_transfer"),
        ("", "deliver kit to the line", "generic", "line_side_delivery"),
        ("", "open the door", "generic", "open_door_enter_room"),
        ("", "yield while human crosses", "generic", "human_crossing_safety_response"),
        ("", "recover from blocked obstacle", "generic", "blocked_path_recovery"),
        ("", "inspect shelf label", "generic", "inspect_shelf"),
        ("", "move tote to cart", "generic", "move_tote"),
        ("", "pick loose object", "generic", "pick_known_object"),
        ("", "route to station", "generic", "navigate_to_station"),
    ],
)
def test_canonical_task_id_keyword_fallbacks(
    task_id: str,
    task_text: str,
    task_category: str,
    expected: str,
) -> None:
    assert (
        red._canonical_task_id_for_task(
            task_id=task_id,
            task_text=task_text,
            task_category=task_category,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("task_text", "expected"),
    [
        ("inspect an unlabeled shelving bay", "inspect_shelf"),
        ("carry a tote to the staging area", "move_tote"),
        ("go to the marked station", "navigate_to_station"),
    ],
)
def test_canonical_task_id_keyword_fallbacks_without_ontology_aliases(
    monkeypatch: pytest.MonkeyPatch,
    task_text: str,
    expected: str,
) -> None:
    monkeypatch.setattr(red, "_ontology_by_id", lambda: {})

    assert (
        red._canonical_task_id_for_task(
            task_id="",
            task_text=task_text,
            task_category="generic",
        )
        == expected
    )


def test_task_library_and_simulation_task_anchor_skip_invalid_entries(tmp_path: Path) -> None:
    task_library = red._task_library(
        task_anchor_manifest={"tasks": ["skip", {"task_id": _BlankTruthy()}]},
        object_geometry_manifest={},
        generated_at="2026-06-20T00:00:00Z",
    )
    assert task_library["task_count"] == 0
    assert red._success_criteria("custom")[-1] == "failure_mode_ids reference failure_taxonomy.json"

    pipeline_dir = tmp_path / "capture" / "pipeline"
    _write_json(
        pipeline_dir / "simulation_automation" / "task_anchor_proposal_manifest.json",
        {"proposals": {"not": "a-list"}},
    )
    assert red._task_anchor_from_simulation_automation(pipeline_dir) == {}

    _write_json(
        pipeline_dir / "simulation_automation" / "task_anchor_proposal_manifest.json",
        {"proposals": ["skip"]},
    )
    assert red._task_anchor_from_simulation_automation(pipeline_dir) == {}

    _write_json(
        pipeline_dir / "simulation_automation" / "task_anchor_proposal_manifest.json",
        {"proposals": [{"task_id": "task-a"}, {"task_id": "task-a"}]},
    )
    anchored = red._task_anchor_from_simulation_automation(pipeline_dir)
    assert [task["task_id"] for task in anchored["tasks"]] == ["task-a"]


def test_robot_profiles_prediction_sources_and_submission_statuses_cover_alternate_shapes() -> None:
    assert red._robot_profiles(
        site_world_spec={"robot_profiles": ["skip", {"robot_profile_id": "arm"}]},
        hosted_session_runtime_manifest={},
    ) == [
        {
            "robot_profile_id": "arm",
            "display_name": "arm",
            "embodiment_type": "unknown",
            "action_space": {},
            "source": "site_world_spec_or_hosted_session_manifest",
            "claim_boundary": "robot_profile_requirement_only_not_robot_asset_ready",
        }
    ]

    sources = red._available_prediction_sources(
        simready_scene_manifest={},
        marble_simready_bridge={},
        cosmos3_readiness={"status": "ready"},
    )
    assert sources == [{"source": "cosmos_preflight", "status": "ready", "execution_proven": False}]

    first_modality = red.ROBOT_TEAM_TEST_MODALITY_REQUIREMENTS[0]["modality_id"]
    list_statuses = red._robot_team_submission_modality_statuses(
        {"modalities": [{"modality": first_modality, "selected": True}]}
    )
    assert red.ROBOT_TEAM_TEST_MODALITY_REQUIREMENTS[0]["missing_evidence_status"] not in list_statuses
    assert red._robot_team_submission_modality_statuses({"modalities": "not-a-list"}) == [
        item["missing_evidence_status"] for item in red.ROBOT_TEAM_TEST_MODALITY_REQUIREMENTS
    ]


def test_prediction_threshold_publication_and_failure_helpers_cover_blocked_edges() -> None:
    ledger = red._prediction_outcome_ledger(
        scenario_library={"scenarios": [{"scenario_id": "s1", "task_id": "t1"}]},
        prediction_sources=[{"source": ""}, {"source": "simready_review"}],
        source_artifacts={},
        generated_at="2026-06-20T00:00:00Z",
    )
    assert ledger["record_count"] == 1

    assert red._threshold_template_for_task(
        {"task_category": "custom", "ontology_task_id": "pick_known_object"}
    )["threshold_profile_id"] == "pick_place_default_v1"
    assert red._threshold_template_for_task(
        {"task_category": "custom", "ontology_task_id": "delivery_route"}
    )["threshold_profile_id"] == "navigation_default_v1"
    assert red._threshold_template_for_task(
        {"task_category": "custom", "ontology_task_id": "unknown"}
    )["threshold_profile_id"] == "general_task_default_v1"

    thresholds = red._task_thresholds(
        task_library={"tasks": [{"task_id": ""}, {"task_id": "t1", "task_category": "custom"}]},
        scoring_methodology={"metrics": []},
        generated_at="2026-06-20T00:00:00Z",
    )
    assert thresholds["task_threshold_count"] == 1

    blockers = red._worldlabs_simready_asset_quality_blockers(
        worldlabs_world_manifest={
            "collider_glb_uri": "gs://bucket/collider.glb",
            "metric_scale_proven": True,
            "ground_plane_proven": True,
            "usd_scene_uri": "gs://bucket/scene.usd",
            "physics_ready": True,
        },
        marble_validation={},
    )
    assert blockers == []

    readiness = red._publication_readiness(
        dataset_state="capture_grounded_review_ready",
        dataset_statuses=["capture_grounded_ready", "needs_robot_pov"],
        output_paths={artifact: f"{artifact}.json" for artifact in red.PUBLICATION_REQUIRED_ARTIFACTS},
        task_thresholds={"task_threshold_count": 0},
        rights_privacy={"rights_status": "approved"},
        worldlabs_world_manifest={},
        marble_validation={},
        generated_at="2026-06-20T00:00:00Z",
    )
    assert "task_thresholds" in readiness["missing_required_artifacts"]

    assert red._record_metric({"cycle_time_seconds": "12.5"}, "cycle_time_seconds") == 12.5
    assert red._record_metric({}, "cycle_time_seconds") == 0.0
    assert red._infer_recorded_failure_modes(
        success=True,
        intervention_count=1,
        unsafe_proximity_count=1,
        collision_count=1,
        object_drop_count=1,
        wrong_object_count=1,
        timeout_count=1,
    ) == []
    assert red._infer_recorded_failure_modes(
        success=False,
        intervention_count=1,
        unsafe_proximity_count=1,
        collision_count=1,
        object_drop_count=1,
        wrong_object_count=0,
        timeout_count=1,
    ) == [
        "failure_intervention_required",
        "failure_safety_threshold_violation",
        "failure_contact_collision",
        "failure_manipulation_miss",
        "failure_cycle_time_exceeded",
    ]
    assert red._infer_recorded_failure_modes(
        success=False,
        intervention_count=0,
        unsafe_proximity_count=0,
        collision_count=0,
        object_drop_count=0,
        wrong_object_count=0,
        timeout_count=0,
    ) == ["failure_task_not_attempted"]


def test_actual_outcome_records_cover_manifest_records_and_trace_fallback() -> None:
    records = red._actual_outcome_records(
        actual_outcome_input={
            "records": [
                {
                    "attempt_id": "actual-1",
                    "scenario_id": "s1",
                    "task_id": "t1",
                    "success": "yes",
                    "cycle_time_seconds": "bad",
                    "interventions": "2",
                    "artifact_paths": {"video": "trace.mp4"},
                }
            ]
        },
        recorded_trace_eval_report={},
    )
    assert records[0]["outcome_id"] == "actual-1"
    assert records[0]["actual_success"] is True
    assert records[0]["cycle_time_seconds"] == 0.0
    assert records[0]["intervention_count"] == 2

    trace_records = red._actual_outcome_records(
        actual_outcome_input={},
        recorded_trace_eval_report={
            "attempts": [
                "skip",
                {
                    "attempt_id": "trace-1",
                    "scenario_id": "s1",
                    "task_id": "t1",
                    "success": False,
                    "intervention_count": "3",
                    "evidence_refs": {"trace": "trace.json"},
                },
            ]
        },
    )
    assert trace_records == [
        {
            "outcome_id": "trace-1",
            "scenario_id": "s1",
            "task_id": "t1",
            "actual_source": "recorded_action_trace",
            "actual_success": False,
            "cycle_time_seconds": None,
            "intervention_count": 3,
            "failure_mode_ids": [],
            "tuning_notes": [],
            "evidence_refs": {"trace": "trace.json"},
            "claim_boundary": "recorded_trace_actual_is_advisory_until_owner_system_review",
        }
    ]


def test_card_helpers_cover_missing_mapping_and_skip_paths() -> None:
    assert red._metadata_from({"metadata": "not-a-mapping"}) == {}
    mapped_condition = red._condition_card(
        key="lighting",
        value={"label": "bright", "source": "operator", "confidence": "observed"},
    )
    assert mapped_condition["ground_truth_status"] == "observed_or_operator_supplied"
    assert red._task_zone_cards([{"task_id": ""}, {"task_id": "task-1"}])[0]["task_id"] == "task-1"
    assert red._object_location_cards({"objects": "not-a-list"}) == []
    object_cards = red._object_location_cards(
        {"objects": ["skip", {"object_id": _BlankTruthy()}, {"id": "obj", "task_role": "target"}]}
    )
    assert [card["object_id"] for card in object_cards] == ["obj"]

    assert red._task_cards(task_library={"tasks": ["skip"]}, generated_at="now")["task_card_count"] == 0
    assert red._scenario_cards(
        scenario_library={"scenarios": ["skip"]}, generated_at="now"
    )["scenario_card_count"] == 0
    assert red._eval_cards(ledger={"records": ["skip"]}, generated_at="now")["eval_card_count"] == 0

    backlog = red._annotation_backlog(
        site_card={"geometry": {"collider": {"backend_blockers": []}}},
        task_cards={
            "cards": [
                {
                    "task_card_id": "task-card",
                    "semantic_grounding": {
                        "object_semantics_status": "missing",
                        "validated_spawn_target_pair": False,
                    },
                    "required_missing_annotations": ["robot_pov_attempt_alignment"],
                }
            ]
        },
        scenario_cards={
            "cards": [
                {
                    "scenario_card_id": "scenario-card",
                    "semantic_spawn_target": {"validated_spawn_target_pair": False},
                }
            ]
        },
        dataset_statuses=[],
        generated_at="now",
    )
    backlog_ids = {item["backlog_id"] for item in backlog["items"]}
    assert {
        "task_card_target_object_semantics",
        "task_card_validated_spawn_target_pair",
        "scenario_card_semantic_spawn_target",
    } <= backlog_ids


def test_collider_and_site_card_helpers_cover_nested_assets_and_scale_branches() -> None:
    assert red._collider_available(
        marble_validation={"collider_mesh_available": True},
        marble_bridge={},
        worldlabs_world_manifest={},
    ) is True
    assert red._collider_available(
        marble_validation={},
        marble_bridge={"mesh": {}, "assets": {"mesh": {"collider_url": "gs://bucket/collider.glb"}}},
        worldlabs_world_manifest={},
    ) is True
    assert red._portable_collider_glb_present(
        marble_validation={"mesh": {"collider_url": "gs://bucket/portable.glb"}},
        marble_bridge={},
        worldlabs_world_manifest={},
    ) is True

    labels = red._collision_backend_labels(
        simready_scene_manifest={},
        marble_validation={},
        marble_bridge={"assets": {"mesh": {"collider_mesh_url": "gs://bucket/collider.glb"}}},
        worldlabs_world_manifest={},
        cpu_preflight_scorecard={
            "isaac_usd_collision_verified": True,
            "cpu_proxy_collision_estimated": True,
        },
    )
    assert "isaac_usd_collision_verified" in labels["labels"]
    assert "cpu_proxy_collision_estimated" in labels["labels"]

    context = SimpleNamespace(scene_id="scene-1", capture_id="capture-1")
    present_scale = red._site_card(
        context=context,
        descriptor={},
        raw_manifest={},
        site_world_spec={"site_type": "customer lab"},
        object_geometry_manifest={},
        task_library={"tasks": []},
        source_artifacts={},
        simready_scene_manifest={},
        marble_validation={},
        marble_bridge={},
        worldlabs_world_manifest={"assets": {"semantics_metadata": {"metric_scale_factor": 2.0}}},
        cpu_preflight_scorecard={},
        protected_regions_manifest={},
        rights_privacy={},
        generated_at="now",
    )
    missing_scale = red._site_card(
        context=context,
        descriptor={},
        raw_manifest={},
        site_world_spec={"site_type": "customer lab"},
        object_geometry_manifest={},
        task_library={"tasks": []},
        source_artifacts={},
        simready_scene_manifest={},
        marble_validation={},
        marble_bridge={},
        worldlabs_world_manifest={},
        cpu_preflight_scorecard={},
        protected_regions_manifest={},
        rights_privacy={},
        generated_at="now",
    )

    assert present_scale["geometry"]["scale"]["status"] == "present"
    assert missing_scale["geometry"]["scale"]["status"] == "needs_scale_review"


def test_rights_field_and_main_cover_cli_success_failure_and_module_guard(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert red._rights_field({"evidence_uri": "owner://approval"}, key="evidence_uri") == (
        "owner://approval"
    )

    monkeypatch.setattr(
        red,
        "build_real_site_robot_eval_dataset",
        lambda *, capture_root: {
            "manifest_path": f"{capture_root}/manifest.json",
            "status": "capture_grounded_review_ready",
        },
    )
    assert red.main(["--capture-root", "/tmp/capture"]) == 0
    assert "manifest=/tmp/capture/manifest.json" in capsys.readouterr().out


def test_rights_packet_carries_revocation_and_revenue_share_review(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rights_privacy = red._rights_privacy_status(
        rights_summary={},
        rights_review={},
        privacy_manifest={"status": "no_people_detected"},
        descriptor={
            "metadata": {
                "capture_rights": {
                    "consent_status": "revoked",
                    "consent_revoked_at": "2026-07-04T12:00:00Z",
                    "consent_scope": [
                        "mujoco_g1_simulator_evaluation_for_this_staged_capture"
                    ],
                    "permission_document_uri": "owner://consent",
                }
            }
        },
        raw_manifest={},
    )
    rights_packet = red._rights_packet(
        rights_summary={},
        rights_review={},
        privacy_manifest={"status": "no_people_detected"},
        rights_privacy=rights_privacy,
        source_artifacts={},
        generated_at="2026-07-04T12:00:00Z",
    )

    assert rights_privacy["blocked"] is True
    assert rights_privacy["consent_revoked"] is True
    assert rights_packet["status"] == "blocked"
    assert rights_packet["revocation_takedown"]["status"] == "takedown_required"
    assert rights_packet["revenue_share_review"]["required_before_paid_reuse_or_resale"] is True
    assert rights_packet["revenue_share_review"]["revenue_share_commitment_made"] is False

    def _raise_failure(*, capture_root: str) -> dict[str, object]:
        raise RuntimeError(f"bad capture: {capture_root}")

    monkeypatch.setattr(red, "build_real_site_robot_eval_dataset", _raise_failure)
    assert red.main(["--capture-root", "/tmp/bad-capture"]) == 1
    assert "FAILED: bad capture: /tmp/bad-capture" in capsys.readouterr().out

    guard = compile("raise SystemExit(main())", red.__file__ or "<robot_eval_dataset>", "exec").replace(
        co_firstlineno=4262
    )
    with pytest.raises(SystemExit) as exc:
        exec(guard, {"main": lambda: 5})
    assert exc.value.code == 5


def test_robot_pov_requirements_require_camera_calibration_metadata() -> None:
    from blueprint_pipeline.robot_eval_dataset import _robot_pov_requirements

    requirements = _robot_pov_requirements(generated_at="2026-07-04T00:00:00+00:00")

    for field in (
        "camera_intrinsics",
        "camera_extrinsics_or_mount_pose",
        "camera_calibration_status",
    ):
        assert field in requirements["required_fields"]
    contract = requirements["camera_metadata_contract"]
    assert "uncalibrated" in contract["calibration_status"]
    assert "never metric geometry claims" in contract["uncalibrated_footage_downgrade"]
