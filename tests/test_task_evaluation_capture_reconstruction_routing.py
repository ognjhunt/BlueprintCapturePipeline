from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_supervisor import (
    AutonomyMode,
    CaptureReconstructionRouteError,
    SupervisorContext,
    ToolRegistry,
    build_capture_reconstruction_route,
    deterministic_baseline_capabilities,
    load_capture_build_ingress,
    validate_capture_reconstruction_route,
    validate_tool_observation_binding,
)
from blueprint_pipeline.task_evaluation_supervisor.supervisor import (
    default_authority_envelope,
)
from blueprint_pipeline.task_evaluation_supervisor.tools import non_spend_tool_bindings


def _capture_build(tmp_path: Path, *, profile: str, has_lidar: bool | None = None) -> dict:
    capture_root = tmp_path / profile
    capture_root.mkdir(parents=True)
    manifest = {
        "schema_version": "blueprint_raw_capture_manifest.v1",
        "scene_id": f"scene-{profile}",
        "capture_id": f"capture-{profile}",
        "capture_authority_profile": profile,
        "capture_modality": profile,
    }
    if has_lidar is not None:
        manifest["has_lidar"] = has_lidar
    (capture_root / "capture_descriptor.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return load_capture_build_ingress(capture_root)


def test_iphone_lidar_and_360_capture_receive_different_reconstruction_routes(
    tmp_path: Path,
) -> None:
    iphone = build_capture_reconstruction_route(
        _capture_build(tmp_path, profile="iphone_arkit_lidar", has_lidar=True),
        requested_claim_types=["reachability"],
    )
    panorama = build_capture_reconstruction_route(
        _capture_build(tmp_path, profile="camera_360_equirectangular"),
        requested_claim_types=["reachability"],
    )

    assert iphone["status"] == panorama["status"] == "route_proposed"
    assert iphone["capture_authority_profile"] == "iphone_arkit_lidar"
    assert panorama["capture_authority_profile"] == "camera_360_equirectangular"
    assert "local://arkit-metric-scaffold-v1" in iphone["currently_registered_adapters"]
    assert "local://arkit-metric-scaffold-v1" not in panorama["currently_registered_adapters"]
    assert [row["stage_id"] for row in iphone["stages"]] != [
        row["stage_id"] for row in panorama["stages"]
    ]
    assert "project_spherical_to_perspective_views" in {
        row["stage_id"] for row in panorama["stages"]
    }
    assert "metric_reference_layer" in panorama["required_representations"]
    assert panorama["execution_authorized_by_route"] is False
    assert panorama["appearance_layer_is_metric_or_physics_truth"] is False
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs"
            / "schemas"
            / "task_evaluation_capture_reconstruction_route.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(iphone, schema)
    jsonschema.validate(panorama, schema)


@pytest.mark.parametrize(
    ("profile", "required_stage"),
    [
        ("camera_360_native", "normalize_native_360_container"),
        ("monocular_video", "estimate_camera_poses_with_sfm"),
        ("precomputed_external_reconstruction", "verify_source_capture_binding"),
    ],
)
def test_each_supported_capture_family_has_a_profile_specific_route(
    tmp_path: Path,
    profile: str,
    required_stage: str,
) -> None:
    route = build_capture_reconstruction_route(_capture_build(tmp_path, profile=profile))

    assert route["status"] == "route_proposed"
    assert required_stage in {row["stage_id"] for row in route["stages"]}
    assert route["agent_selected_capture_profile"] is False
    assert route["proof_effect"] == "none"


def test_lidar_hint_alone_cannot_grant_arkit_lidar_route(tmp_path: Path) -> None:
    capture_root = tmp_path / "undeclared"
    capture_root.mkdir()
    (capture_root / "capture_descriptor.json").write_text(
        json.dumps(
            {
                "schema_version": "blueprint_capture_descriptor.v1",
                "scene_id": "scene-undeclared",
                "capture_id": "capture-undeclared",
                "has_lidar": True,
            }
        ),
        encoding="utf-8",
    )

    route = build_capture_reconstruction_route(load_capture_build_ingress(capture_root))

    assert route["status"] == "capture_profile_required"
    assert route["capture_authority_profile"] is None
    assert route["stages"] == []
    assert route["currently_registered_adapters"] == []
    assert route["blockers"] == ["validated_capture_authority_profile_missing"]


def test_conflicting_capture_profiles_fail_closed(tmp_path: Path) -> None:
    capture_root = tmp_path / "conflicting"
    (capture_root / "raw").mkdir(parents=True)
    (capture_root / "capture_descriptor.json").write_text(
        json.dumps(
            {
                "schema_version": "blueprint_capture_descriptor.v1",
                "capture_authority_profile": "iphone_arkit_lidar",
            }
        ),
        encoding="utf-8",
    )
    (capture_root / "raw" / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "blueprint_raw_capture_manifest.v1",
                "capture_modality": "camera_360_equirectangular",
            }
        ),
        encoding="utf-8",
    )

    route = build_capture_reconstruction_route(load_capture_build_ingress(capture_root))

    assert route["status"] == "ambiguous_capture_profile"
    assert route["capture_authority_profile"] is None
    assert route["stages"] == []
    assert route["blockers"] == ["conflicting_capture_authority_profiles"]


def test_route_validator_rejects_agent_execution_or_proof_upgrade(tmp_path: Path) -> None:
    route = build_capture_reconstruction_route(
        _capture_build(tmp_path, profile="camera_360_equirectangular")
    )
    compromised = json.loads(json.dumps(route))
    compromised["execution_authorized_by_route"] = True
    compromised["capture_reconstruction_route_digest"] = canonical_digest(
        compromised,
        digest_field="capture_reconstruction_route_digest",
    )

    with pytest.raises(
        CaptureReconstructionRouteError,
        match="capture_reconstruction_route_contract_invalid",
    ):
        validate_capture_reconstruction_route(compromised)


def test_capture_agent_proposes_route_when_only_capture_build_exists(tmp_path: Path) -> None:
    capture_build = _capture_build(tmp_path, profile="camera_360_equirectangular")
    capture_agent = next(
        capability
        for capability in deterministic_baseline_capabilities()
        if capability.kind.value == "capture_testbed_supervisor"
    )

    result = capture_agent.propose(
        SupervisorContext(
            run_id="capture-route-only",
            customer_question="Build the appropriate site reconstruction.",
            capture_build=capture_build,
        )
    ).to_mapping()

    assert result["status"] == "blocked"
    route = result["artifact"]["capture_reconstruction_route"]
    assert route["capture_authority_profile"] == "camera_360_equirectangular"
    assert result["proposals"][0]["tool_id"] == "plan_capture_reconstruction_route"
    assert "maintained_site_task_testbed_missing" in result["blockers"]


def test_agents_sdk_receives_only_digest_bound_deterministic_route_tool(
    tmp_path: Path,
) -> None:
    capture_build = _capture_build(tmp_path, profile="iphone_arkit_lidar", has_lidar=True)
    registry = ToolRegistry.default()
    context = SupervisorContext(
        run_id="capture-route-tool",
        customer_question="Which reconstruction route applies?",
        capture_build=capture_build,
        supervisor_output_dir=str(tmp_path / "run"),
    )
    authority = default_authority_envelope(
        run_id=context.run_id,
        mode=AutonomyMode.EXECUTE_NON_SPEND,
        tool_registry=registry,
        immutable_input_digests=[capture_build["capture_build_digest"]],
    ).to_mapping()
    bindings = {
        binding.tool_id: binding
        for binding in non_spend_tool_bindings(
            capability="capture_testbed_supervisor",
            context=context,
            registry=registry,
            authority=authority,
        )
    }

    assert set(bindings) == {
        "inspect_site_task_testbed",
        "plan_capture_reconstruction_route",
        "propose_targeted_recapture",
    }
    observation = bindings["plan_capture_reconstruction_route"].invoke(
        {
            "capture_build_digest": capture_build["capture_build_digest"],
            "requested_claim_types": [],
        }
    )
    assert observation["status"] == "completed"
    assert observation["typed_result"]["capture_authority_profile"] == "iphone_arkit_lidar"
    assert observation["typed_result"]["execution_authorized_by_route"] is False
    assert observation["proof_effect"] == "none"

    invented_claim = bindings["plan_capture_reconstruction_route"].invoke(
        {
            "capture_build_digest": capture_build["capture_build_digest"],
            "requested_claim_types": ["collision_contact"],
        }
    )
    assert invented_claim["status"] == "refused"
    assert "registered_tool_claim_scope_mismatch" in invented_claim["typed_failure"]["reason"]

    compromised_observation = json.loads(json.dumps(observation))
    compromised_route = compromised_observation["typed_result"]["route"]
    compromised_route["stages"][0]["stage_id"] = "agent_selected_unregistered_route"
    compromised_route["capture_reconstruction_route_digest"] = canonical_digest(
        compromised_route,
        digest_field="capture_reconstruction_route_digest",
    )
    compromised_observation["typed_result"]["route_digest"] = compromised_route[
        "capture_reconstruction_route_digest"
    ]
    compromised_observation["output_digest"] = canonical_digest(
        compromised_observation["typed_result"]
    )
    compromised_observation["observation_digest"] = canonical_digest(
        compromised_observation,
        digest_field="observation_digest",
    )
    with pytest.raises(
        CaptureReconstructionRouteError,
        match="capture_reconstruction_route_profile_invalid",
    ):
        validate_tool_observation_binding(
            compromised_observation,
            run_id=context.run_id,
            capability="capture_testbed_supervisor",
            registry=registry,
            authority=authority,
        )

    refused = bindings["plan_capture_reconstruction_route"].invoke(
        {
            "capture_build_digest": "sha256:" + "0" * 64,
            "requested_claim_types": [],
        }
    )
    assert refused["status"] == "refused"
    assert refused["typed_result"] == {}
