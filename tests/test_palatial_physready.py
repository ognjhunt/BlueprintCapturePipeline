from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from blueprint_pipeline.evaluation_prep_stage import palatial_physready_evaluation_prep_surface
from blueprint_pipeline.palatial_physready import build_palatial_physready_assets, main


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload)), encoding="utf-8")


def _build_capture_root(tmp_path: Path) -> Path:
    capture_root = tmp_path / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    raw_root = capture_root / "raw"
    raw_root.mkdir(parents=True)
    _write_json(
        raw_root / "manifest.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "site_identity": {"site_id": "site-1", "site_id_source": "fixture"},
        },
    )
    _write_json(
        capture_root / "capture_descriptor.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "metadata": {"site_identity": {"site_id": "site-1", "site_id_source": "fixture"}},
        },
    )
    image_root = raw_root / "object_images"
    image_root.mkdir(parents=True)
    (image_root / "microwave_front.jpg").write_bytes(b"fake-microwave-jpeg")
    (image_root / "tote_front.jpg").write_bytes(b"fake-tote-jpeg")
    return capture_root


def _object_geometry_manifest() -> dict[str, object]:
    return {
        "schema_version": "object_geometry_manifest.v1",
        "objects": [
            {
                "object_id": "microwave_001",
                "label": "microwave",
                "placement_bbox": {
                    "center": [1.0, 0.5, 0.65],
                    "extents": [0.55, 0.42, 0.32],
                },
                "provenance": {"reference_image": "raw/object_images/microwave_front.jpg"},
            },
            {
                "object_id": "blue_tote_001",
                "label": "blue warehouse tote",
                "placement_bbox": {
                    "center": [2.0, 0.4, 0.25],
                    "extents": [0.6, 0.4, 0.3],
                },
                "image_paths": ["raw/object_images/tote_front.jpg"],
                "provenance": {"grounding_level": "observed"},
            },
            {
                "object_id": "floor_001",
                "label": "floor",
                "placement_bbox": {
                    "center": [0.0, 0.0, 0.0],
                    "extents": [4.0, 4.0, 0.02],
                },
            },
        ],
    }


def _task_anchor_manifest() -> dict[str, object]:
    return {
        "schema_version": "task_anchor_manifest.v1",
        "tasks": [
            {
                "task_id": "open_microwave",
                "task_text": "Open the microwave door and place a container inside",
                "target_object_ids": ["microwave_001"],
                "articulation_required_ids": ["microwave_001"],
                "task_critical": True,
            },
            {
                "task_id": "move_tote",
                "task_text": "Move the blue tote from the shelf to the cart",
                "target_object_ids": ["blue_tote_001"],
                "task_critical": True,
            },
        ],
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_palatial_physready_default_builds_microwave_and_tote_plan_without_live_call(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)

    result = build_palatial_physready_assets(
        capture_root=capture_root,
        object_geometry_manifest=_object_geometry_manifest(),
        task_anchor_manifest=_task_anchor_manifest(),
        env={},
    )

    palatial_dir = capture_root / "pipeline" / "palatial_physready"
    candidates = _read_json(palatial_dir / "twin_candidate_manifest.json")
    request = _read_json(palatial_dir / "palatial_request_manifest.json")
    run = _read_json(palatial_dir / "palatial_physready_run_manifest.json")

    labels = {candidate["label"] for candidate in candidates["candidates"]}
    microwave = next(
        candidate for candidate in candidates["candidates"] if candidate["source_object_id"] == "microwave_001"
    )
    tote = next(
        candidate for candidate in candidates["candidates"] if candidate["source_object_id"] == "blue_tote_001"
    )

    assert result["status"] == "ready_for_manual_or_live_submission"
    assert labels == {"microwave", "blue warehouse tote"}
    assert microwave["desired_articulation"]["type"] == "hinged_door_with_handle"
    assert tote["desired_articulation"]["type"] == "rigid_container_with_optional_lid_or_grasp_handles"
    assert microwave["reference_image_count"] == 1
    assert tote["reference_image_count"] == 1
    assert request["pricing_estimate"]["estimated_marginal_cost_usd"] == 20.0
    assert request["live_execution_gate"]["live_provider_calls_allowed"] is False
    assert run["live_provider_calls_performed"] is False
    assert run["claim_boundary"]["generated_assets_are_capture_truth"] is False
    assert run["easy_on_off_switch"]["off_default"] is True


class _ExplodingClient:
    def generate_asset(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        raise AssertionError("client should not be called when gates are missing")


def test_palatial_physready_live_call_requires_env_flag_and_api_key(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)

    result = build_palatial_physready_assets(
        capture_root=capture_root,
        object_geometry_manifest=_object_geometry_manifest(),
        task_anchor_manifest=_task_anchor_manifest(),
        allow_live_palatial=True,
        client=_ExplodingClient(),
        env={},
    )

    run = _read_json(
        capture_root
        / "pipeline"
        / "palatial_physready"
        / "palatial_physready_run_manifest.json"
    )

    assert result["status"] == "blocked_missing_live_gates"
    assert "BLUEPRINT_ENABLE_PALATIAL_PHYSREADY=true" in run["blockers"]
    assert "PALATIAL_API_KEY=<secret>" in run["blockers"]
    assert run["live_provider_calls_performed"] is False


class _FakeClient:
    def __init__(self) -> None:
        self.requests: list[Mapping[str, Any]] = []

    def generate_asset(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        self.requests.append(dict(request))
        return {
            "asset_id": f"asset-{len(self.requests)}",
            "status": "queued",
        }


def test_palatial_physready_live_gate_submits_with_fake_client(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)
    client = _FakeClient()

    result = build_palatial_physready_assets(
        capture_root=capture_root,
        object_geometry_manifest=_object_geometry_manifest(),
        task_anchor_manifest=_task_anchor_manifest(),
        allow_live_palatial=True,
        client=client,
        env={
            "BLUEPRINT_ENABLE_PALATIAL_PHYSREADY": "true",
            "PALATIAL_API_KEY": "test-secret-not-written",
        },
    )

    run = _read_json(
        capture_root
        / "pipeline"
        / "palatial_physready"
        / "palatial_physready_run_manifest.json"
    )

    assert result["status"] == "submitted_waiting_for_exports"
    assert len(client.requests) == 2
    assert any("microwave" in request["prompt"] for request in client.requests)
    assert all(request["image_paths"] for request in client.requests)
    assert run["live_provider_calls_allowed"] is True
    assert run["live_provider_calls_performed"] is True
    assert "test-secret-not-written" not in json.dumps(run)


def test_palatial_physready_materializes_provider_response_for_local_review(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    export_root = tmp_path / "provider-exports"
    export_root.mkdir()
    export_path = export_root / "microwave.usda"
    export_path.write_text(
        "\n".join(
            [
                "#usda 1.0",
                "(",
                '    defaultPrim = "Microwave"',
                "    metersPerUnit = 1",
                '    upAxis = "Z"',
                ")",
                'def Xform "Microwave" {}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    response_path = tmp_path / "palatial-response.json"
    _write_json(
        response_path,
        {
            "candidate_id": "palatial_microwave_001",
            "asset_id": "asset-microwave",
            "exports": {"isaac_sim": str(export_path)},
        },
    )

    result = build_palatial_physready_assets(
        capture_root=capture_root,
        object_geometry_manifest=_object_geometry_manifest(),
        task_anchor_manifest=_task_anchor_manifest(),
        provider_response_paths=[response_path],
        env={},
    )

    palatial_dir = capture_root / "pipeline" / "palatial_physready"
    materialization = _read_json(palatial_dir / "materialization_manifest.json")
    validation = _read_json(palatial_dir / "validation_manifest.json")

    assert result["status"] == "materialized_for_review"
    assert materialization["materialized_export_count"] == 1
    assert Path(materialization["exports"][0]["local_path"]).is_file()
    assert validation["status"] == "prepared_for_review"
    assert validation["simulator_execution_proven"] is False
    assert validation["robot_readiness_proven"] is False


def test_evaluation_prep_surfaces_existing_palatial_artifacts(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)
    eval_dir = capture_root / "pipeline" / "evaluation_prep"
    eval_dir.mkdir(parents=True)

    build_palatial_physready_assets(
        capture_root=capture_root,
        object_geometry_manifest=_object_geometry_manifest(),
        task_anchor_manifest=_task_anchor_manifest(),
        env={},
    )

    surface = palatial_physready_evaluation_prep_surface(
        capture_root=capture_root,
        eval_dir=eval_dir,
    )

    assert surface["status"] == "ready_for_manual_or_live_submission"
    assert surface["live_provider_calls_performed"] is False
    assert surface["robot_readiness_proven"] is False
    assert surface["public_claim_upgrade_allowed"] is False
    assert surface["artifacts"]["palatial_physready_run_manifest"].endswith(
        "../palatial_physready/palatial_physready_run_manifest.json"
    )
    assert surface["artifact_uris"]["palatial_physready_run_manifest_uri"].endswith(
        "/pipeline/palatial_physready/palatial_physready_run_manifest.json"
    )


def test_palatial_physready_cli_writes_default_manifests(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)
    eval_dir = capture_root / "pipeline" / "evaluation_prep"
    _write_json(eval_dir / "object_geometry_manifest.json", _object_geometry_manifest())
    _write_json(eval_dir / "task_anchor_manifest.json", _task_anchor_manifest())

    exit_code = main(["--capture-root", str(capture_root), "--label", "microwave"])

    palatial_dir = capture_root / "pipeline" / "palatial_physready"
    run = _read_json(palatial_dir / "palatial_physready_run_manifest.json")
    candidates = _read_json(palatial_dir / "twin_candidate_manifest.json")

    assert exit_code == 0
    assert run["status"] == "ready_for_manual_or_live_submission"
    assert run["live_provider_calls_performed"] is False
    assert candidates["candidate_count"] == 1
    assert candidates["candidates"][0]["source_object_id"] == "microwave_001"
