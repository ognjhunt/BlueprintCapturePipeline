from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import jsonschema

from blueprint_pipeline.scene_object_discovery import (
    SceneObjectDiscoveryError,
    build_full_scene_camera_plan,
    compile_scene_object_discovery,
    materialize_scene_object_discovery_renders,
)


DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
DIGEST_C = "sha256:" + "c" * 64
DIGEST_D = "sha256:" + "d" * 64


def _geometry() -> dict[str, object]:
    return {
        "aabb_min": [-2.0, -1.0, 0.0],
        "aabb_max": [2.0, 3.0, 2.5],
        "up_axis": 2,
        "up_sign": 1.0,
        "unseen_regions": ["behind_uncaptured_partition"],
    }


def _plan() -> dict[str, object]:
    return build_full_scene_camera_plan(
        scene_geometry=_geometry(),
        source_splat_digest=DIGEST_A,
        retained_gaussian_count=1234,
        registration_digest=DIGEST_B,
        width=320,
        height=240,
        n_azimuths=4,
        elevations_deg=(10.0, 35.0),
    )


def _source() -> dict[str, object]:
    return {
        "source_splat_digest": DIGEST_A,
        "retained_gaussian_count": 1234,
        "registration_digest": DIGEST_B,
    }


def _render(plan: dict[str, object]) -> dict[str, object]:
    return {
        "source_splat_digest": DIGEST_A,
        "camera_plan_digest": plan["camera_plan_digest"],
        "render_manifest_digest": DIGEST_C,
    }


def test_full_scene_plan_is_digest_bound_and_records_unseen_regions() -> None:
    plan = _plan()

    assert len(plan["cameras"]) == 8
    assert plan["camera_plan_digest"].startswith("sha256:")
    assert plan["coverage"]["known_scene_bounds_covered"] is True
    assert plan["coverage"]["unseen_regions"] == ["behind_uncaptured_partition"]
    first = plan["cameras"][0]
    assert first["camera_id"] == "survey_000"
    assert len(first["T_world_camera_provider_frame"]) == 4
    assert first["intrinsics"]["width"] == 320


def test_non_z_up_plan_requires_explicit_normalization_binding() -> None:
    geometry = _geometry()
    geometry["up_axis"] = 1

    with pytest.raises(SceneObjectDiscoveryError) as exc:
        build_full_scene_camera_plan(
            scene_geometry=geometry,
            source_splat_digest=DIGEST_A,
            retained_gaussian_count=1,
            registration_digest=DIGEST_B,
        )

    assert "scene_discovery_non_z_up_normalization_required" in exc.value.codes
    assert "scene_discovery_normalized_geometry_required" in exc.value.codes


def test_malformed_camera_plan_values_fail_with_typed_errors() -> None:
    geometry = _geometry()
    geometry["up_sign"] = "not-a-number"

    with pytest.raises(SceneObjectDiscoveryError) as exc:
        build_full_scene_camera_plan(
            scene_geometry=geometry,
            source_splat_digest=DIGEST_A,
            retained_gaussian_count=None,
            registration_digest=DIGEST_B,
        )

    assert "scene_discovery_normalized_geometry_required" in exc.value.codes
    assert "scene_discovery_retained_count_invalid" in exc.value.codes


def test_render_adapter_passes_exact_production_method_input_contract(tmp_path: Path) -> None:
    splat = tmp_path / "scene.ply"
    splat.write_bytes(b"exact splat")
    source_digest = "sha256:" + hashlib.sha256(splat.read_bytes()).hexdigest()
    plan = build_full_scene_camera_plan(
        scene_geometry=_geometry(),
        source_splat_digest=source_digest,
        retained_gaussian_count=42,
        registration_digest=DIGEST_B,
        width=320,
        height=240,
        n_azimuths=4,
        elevations_deg=(20.0,),
    )
    observed: dict[str, object] = {}

    def runtime_resolver(**kwargs):
        observed["runtime"] = kwargs
        return {
            "node": "/runtime/node",
            "renderer_root": "/runtime/renderer",
            "browser_executable": "/runtime/chromium",
            "identity": {"runtime_digest": DIGEST_D},
        }

    def renderer(**kwargs):
        observed["renderer"] = kwargs
        return {
            "schema_version": "sealed_camera_render_manifest.v1",
            "sealed_camera_render_manifest_digest": DIGEST_C,
        }

    result = materialize_scene_object_discovery_renders(
        source_splat_path=splat,
        camera_plan=plan,
        output_root=tmp_path / "out",
        runtime_resolver=runtime_resolver,
        renderer=renderer,
    )

    call = observed["renderer"]
    assert call["authorization_class"] == "method_input"
    assert call["purpose"] == "scene_object_discovery_full_scene_method_inputs"
    assert call["retained_gaussian_count"] == 42
    assert call["source_splat_digest"] == source_digest
    assert Path(call["calibrated_camera_file"]).is_file()
    assert result["renderer_capabilities"]["depth"] is False
    assert result["renderer_capabilities"]["per_gaussian_contributions"] is False
    assert result["render_binding"] == {
        "source_splat_digest": source_digest,
        "camera_plan_digest": plan["camera_plan_digest"],
        "render_manifest_digest": DIGEST_C,
    }


def test_visual_candidates_remain_unselected_without_metric_refinement() -> None:
    plan = _plan()
    output = compile_scene_object_discovery(
        source_binding=_source(),
        camera_plan=plan,
        render_binding=_render(plan),
        analyzer_runs=[
            {
                "backend": "splat_analyzer",
                "run_digest": DIGEST_D,
                "source_splat_digest": DIGEST_A,
                "render_manifest_digest": DIGEST_C,
                "candidates": [
                    {
                        "candidate_id": "splat_box_1",
                        "label": "red tote",
                        "confidence": 0.9,
                        "supporting_view_ids": ["survey_000"],
                        "metric_geometry": {
                            "authority": "rough_splat_analyzer_box",
                            "validated": True,
                            "evidence_digest": DIGEST_D,
                            "bounds_min": [0, 0, 0],
                            "bounds_max": [1, 1, 1],
                        },
                    }
                ],
            }
        ],
        task_context={"task_statement": "pick the red tote"},
    )

    assert output["status"] == "metric_refinement_required"
    assert output["source_object"] is None
    assert output["candidates"][0]["eligible_for_automatic_source_object"] is False
    assert output["claim_boundary"]["splat_analyzer_boxes_are_metric_geometry"] is False


def test_candidate_identity_and_public_label_are_bounded() -> None:
    plan = _plan()
    with pytest.raises(SceneObjectDiscoveryError) as exc:
        compile_scene_object_discovery(
            source_binding=_source(),
            camera_plan=plan,
            render_binding=_render(plan),
            analyzer_runs=[
                {
                    "backend": "splat_analyzer",
                    "run_digest": DIGEST_D,
                    "source_splat_digest": DIGEST_A,
                    "render_manifest_digest": DIGEST_C,
                    "candidates": [
                        {
                            "candidate_id": "unsafe candidate/id",
                            "label": "x" * 513,
                            "confidence": 0.9,
                            "supporting_view_ids": ["survey_000"],
                        }
                    ],
                }
            ],
            task_context={"task_statement": "pick the target"},
        )

    assert "scene_discovery_candidate_identity_invalid_or_duplicate" in exc.value.codes


def test_unique_production_semantic_obb_auto_selects_source_object() -> None:
    plan = _plan()
    output = compile_scene_object_discovery(
        source_binding=_source(),
        camera_plan=plan,
        render_binding=_render(plan),
        analyzer_runs=[
            {
                "backend": "sam31",
                "run_digest": DIGEST_D,
                "source_splat_digest": DIGEST_A,
                "render_manifest_digest": DIGEST_C,
                "candidates": [
                    {
                        "candidate_id": "sam31_tote_1",
                        "label": "red tote",
                        "confidence": 0.94,
                        "supporting_view_ids": ["survey_000", "survey_003"],
                        "metric_geometry": {
                            "authority": "production_semantic_gaussian_obb",
                            "validated": True,
                            "production_large_scene_ready": True,
                            "independent_deterministic_validation_passed": True,
                            "evidence_digest": DIGEST_D,
                            "bounds_min": [0.1, 0.2, 0.3],
                            "bounds_max": [0.8, 0.9, 1.0],
                        },
                    }
                ],
            }
        ],
        task_context={"task_statement": "pick the red tote"},
    )

    assert output["status"] == "ready_auto_selected"
    assert output["selected_candidate_id"] == "sam31_tote_1"
    assert output["source_object"]["bounds_min"] == [0.1, 0.2, 0.3]
    assert output["discovery_digest"].startswith("sha256:")
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "schemas"
        / "scene_object_discovery.v1.schema.json"
    )
    jsonschema.Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(
        output
    )


def test_multiple_metric_candidates_require_selection_and_digest_mismatch_fails() -> None:
    plan = _plan()
    candidate = {
        "label": "red tote",
        "confidence": 0.9,
        "supporting_view_ids": ["survey_000"],
        "metric_geometry": {
            "authority": "publisher_metric_label",
            "validated": True,
            "evidence_digest": DIGEST_D,
            "bounds_min": [0, 0, 0],
            "bounds_max": [1, 1, 1],
        },
    }
    output = compile_scene_object_discovery(
        source_binding=_source(),
        camera_plan=plan,
        render_binding=_render(plan),
        analyzer_runs=[
            {
                "backend": "publisher_semantics",
                "run_digest": DIGEST_D,
                "source_splat_digest": DIGEST_A,
                "candidates": [
                    {"candidate_id": "publisher_tote_1", **candidate},
                    {"candidate_id": "publisher_tote_2", **candidate},
                ],
            }
        ],
        task_context={"task_statement": "pick the red tote"},
    )
    assert output["status"] == "selection_required"
    assert output["source_object"] is None

    bad_render = _render(plan)
    bad_render["camera_plan_digest"] = DIGEST_D
    with pytest.raises(SceneObjectDiscoveryError) as exc:
        compile_scene_object_discovery(
            source_binding=_source(),
            camera_plan=plan,
            render_binding=bad_render,
            analyzer_runs=[],
            task_context={},
        )
    assert "scene_discovery_render_camera_plan_mismatch" in exc.value.codes


def test_camera_plan_digest_detects_tampering() -> None:
    plan = _plan()
    plan["cameras"][0]["camera_id"] = "tampered"
    splat = Path(__file__)

    with pytest.raises(SceneObjectDiscoveryError) as exc:
        materialize_scene_object_discovery_renders(
            source_splat_path=splat,
            camera_plan=plan,
            output_root=splat.parent / "unused",
            runtime_resolver=lambda **_kwargs: {},
            renderer=lambda **_kwargs: {},
        )

    assert "scene_discovery_camera_plan_digest_mismatch" in exc.value.codes
