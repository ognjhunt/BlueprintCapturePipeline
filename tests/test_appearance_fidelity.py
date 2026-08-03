from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.appearance_fidelity import (
    AppearanceFidelityContractError,
    build_appearance_fidelity_qualification,
    build_presentation_derivative_contract,
    build_robot_appearance_composite_contract,
    select_best_fidelity_render_route,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _bounds() -> dict:
    return {
        "aabb_min": [-2.0, -1.0, -4.0],
        "aabb_max": [6.0, 2.0, 4.0],
        "robust_min": [-1.9, -0.9, -3.9],
        "robust_max": [5.9, 1.9, 3.9],
        "robust_percentile": 0.999,
    }


def _qualification(*, ssim: float = 0.96, retained: int = 980, decimated: bool = False) -> dict:
    removed = 1000 - retained
    return {
        "source_appearance": {
            "asset_digest": _digest("a"),
            "coordinate_basis_digest": _digest("7"),
            "representation": "spz",
            "splat_count": 1000,
            "sh_degree": 3,
            "bounds": _bounds(),
        },
        "render_input": {
            "asset_digest": _digest("b"),
            "coordinate_basis_digest": _digest("7"),
            "representation": "standard_3dgs_ply",
            "splat_count": retained,
            "sh_degree": 3,
            "bounds": _bounds(),
            "global_decimation_applied": decimated,
            "removal_reasons": [
                {
                    "reason": "low_opacity",
                    "count": removed,
                    "independently_qualified": True,
                    "qualification_digest": _digest("c"),
                    "policy_digest": _digest("d"),
                }
            ],
        },
        "renderer": {
            "renderer_id": "native-gsplat-v1",
            "implementation_digest": _digest("e"),
            "runtime_digest": _digest("f"),
            "native_3dgs": True,
            "full_anisotropic_gaussians": True,
            "maximum_sh_degree": 3,
        },
        "reference_frame_comparison": {
            "status": "completed",
            "source_frame_digest": _digest("1"),
            "rendered_frame_digest": _digest("2"),
            "camera_spec_digest": _digest("3"),
            "camera_basis_digest": _digest("7"),
            "metrics": {"ssim": ssim, "psnr_db": 31.0, "lpips": 0.08},
        },
        "qualification_policy": {
            "minimum_retained_fraction": 0.95,
            "minimum_ssim": 0.9,
            "minimum_psnr_db": 25.0,
            "maximum_lpips": 0.15,
        },
    }


def _schema(name: str) -> dict:
    return json.loads(
        (Path(__file__).parents[1] / "docs" / "schemas" / name).read_text(encoding="utf-8")
    )


def test_qualifies_only_auditable_full_fidelity_derivative() -> None:
    result = build_appearance_fidelity_qualification(_qualification())

    assert result["status"] == "qualified"
    assert result["retained_splat_fraction"] == 0.98
    assert result["source_appearance"]["immutable_appearance_truth"] is True
    assert result["evaluation_render_authorized"] is True
    assert result["appearance_is_metric_or_collision_truth"] is False
    jsonschema.validate(result, _schema("appearance_fidelity_qualification.v1.schema.json"))


def test_global_decimation_can_never_qualify() -> None:
    result = build_appearance_fidelity_qualification(_qualification(retained=250, decimated=True))

    assert result["status"] == "blocked"
    assert "global_splat_decimation_forbidden" in result["blockers"]
    assert "retained_splat_fraction_below_threshold" in result["blockers"]
    assert result["evaluation_render_authorized"] is False


def test_camera_must_bind_the_exact_render_coordinate_basis() -> None:
    value = _qualification()
    value["reference_frame_comparison"]["camera_basis_digest"] = _digest("8")
    result = build_appearance_fidelity_qualification(value)
    assert result["status"] == "blocked"
    assert "camera_coordinate_basis_mismatch" in result["blockers"]


@pytest.mark.parametrize("field", ["measurement", "threshold"])
def test_ssim_values_above_one_are_rejected(field: str) -> None:
    value = _qualification()
    if field == "measurement":
        value["reference_frame_comparison"]["metrics"]["ssim"] = 1.01
    else:
        value["qualification_policy"]["minimum_ssim"] = 1.01

    with pytest.raises(
        AppearanceFidelityContractError,
        match="appearance_fidelity_ssim_measurement_invalid",
    ):
        build_appearance_fidelity_qualification(value)


def test_route_selects_best_measured_fidelity_independently_from_isaac() -> None:
    lower = build_appearance_fidelity_qualification(_qualification(ssim=0.93))
    higher_input = _qualification(ssim=0.98)
    higher_input["renderer"]["renderer_id"] = "native-gsplat-hq-v2"
    higher_input["renderer"]["runtime_digest"] = _digest("9")
    higher = build_appearance_fidelity_qualification(higher_input)
    route = select_best_fidelity_render_route(
        source_appearance_digest=_digest("a"),
        site_id="506-lenox",
        task_family="franka_sink_inspection",
        appearance_candidates=[lower, higher],
        dynamics_engine={
            "engine_id": "isaac-sim-6.0",
            "qualification_digest": _digest("8"),
            "status": "qualified",
        },
    )

    assert route["status"] == "qualified_route_selected"
    assert route["selected_appearance"]["renderer_id"] == "native-gsplat-hq-v2"
    assert route["dynamics_engine"]["engine_id"] == "isaac-sim-6.0"
    assert route["appearance_renderer_selected_independently_from_dynamics"] is True
    jsonschema.validate(route, _schema("appearance_render_route.v1.schema.json"))


def test_composite_requires_qualified_appearance_official_robot_and_depth() -> None:
    value = {
        "robot_id": "franka_panda",
        "appearance_fidelity_status": "qualified",
        "appearance_fidelity_qualification_digest": _digest("a"),
        "appearance_frame_digest": _digest("b"),
        "robot_rgba_frame_digest": _digest("c"),
        "robot_depth_frame_digest": _digest("d"),
        "camera_spec_digest": _digest("e"),
        "robot_asset_digest": _digest("f"),
        "dynamics_runtime_result_digest": _digest("1"),
        "compositor_implementation_digest": _digest("2"),
        "output_frame_digest": _digest("3"),
        "robot_asset_source": "official_simulator_asset",
        "exact_camera_binding": True,
        "depth_aware_occlusion": True,
        "presentation_derivative_used": False,
    }
    result = build_robot_appearance_composite_contract(value)

    assert result["status"] == "authorized_composite"
    assert result["physical_success_proven"] is False
    jsonschema.validate(result, _schema("robot_appearance_composite_contract.v1.schema.json"))
    with pytest.raises(
        AppearanceFidelityContractError, match="presentation_derivative_forbidden_in_composite"
    ):
        build_robot_appearance_composite_contract({**value, "presentation_derivative_used": True})


@pytest.mark.parametrize("method_id", ["difix", "artifixer"])
def test_generative_enhancers_are_presentation_only(method_id: str) -> None:
    value = {
        "source_frame_digest": _digest("a"),
        "output_frame_digest": _digest("b"),
        "implementation_digest": _digest("c"),
        "method_id": method_id,
        "presentation_only": True,
        "evaluation_input_allowed": False,
        "policy_observation_allowed": False,
        "target_binding_allowed": False,
        "metric_geometry_allowed": False,
        "collision_geometry_allowed": False,
        "qualification_routing_allowed": False,
    }
    result = build_presentation_derivative_contract(value)

    assert result["claim_ceiling"] == "presentation_only"
    jsonschema.validate(result, _schema("appearance_presentation_derivative.v1.schema.json"))
    with pytest.raises(
        AppearanceFidelityContractError, match="presentation_derivative_authority_invalid"
    ):
        build_presentation_derivative_contract({**value, "evaluation_input_allowed": True})
