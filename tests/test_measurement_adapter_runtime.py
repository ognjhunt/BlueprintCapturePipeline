from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.measurement_adapter_runtime import (
    ADAPTER_RECIPES,
    MeasurementAdapterError,
    build_adapter_admission_packet,
    build_capability_draft,
    build_measurement_adapter_descriptor,
    priority_adapter_descriptors,
    probe_measurement_adapter,
    validate_capability_draft,
    validate_measurement_adapter_descriptor,
)
from blueprint_pipeline.task_site_measurement_routing import ALL_CAPABILITY_FIELDS


SHA_A = "sha256:" + "a" * 64


def test_priority_descriptors_cover_named_adapter_program_without_authorization() -> None:
    descriptors = priority_adapter_descriptors()
    assert {row["candidate_id"] for row in descriptors} == set(ADAPTER_RECIPES)
    assert {
        "mujoco-3",
        "isaac-sim-6-physx",
        "newton-1-4",
        "drake-1-55",
        "sapien-maniskill-3",
        "project-chrono-10",
        "flash",
        "garmentdynamics-rgbench",
        "simweaver-sim1",
        "sofa-26-06",
        "tacsl",
        "difftactile",
    } <= set(ADAPTER_RECIPES)
    for descriptor in descriptors:
        assert descriptor["production_execution_authorized"] is False
        assert descriptor["production_route_eligible"] is False
        assert descriptor["physical_robot_execution_authorized"] is False
        assert descriptor["agent_may_authorize"] is False
        assert descriptor["adapter_descriptor_digest"].startswith("sha256:")
    chrono = build_measurement_adapter_descriptor("project-chrono-10")
    assert chrono["probe_contract"]["python_distributions"] == []
    assert chrono["probe_contract"]["executables"] == []
    assert chrono["execution_mode"] == "isolated_external_conda"
    assert chrono["target_version"] == "10.0.0"


def test_probe_is_side_effect_free_and_version_mismatch_is_observable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = build_measurement_adapter_descriptor("mujoco-3")

    def fake_version(name: str) -> str:
        assert name == "mujoco"
        return "3.10.0"

    monkeypatch.setattr(
        "blueprint_pipeline.measurement_adapter_runtime.importlib.metadata.version",
        fake_version,
    )
    result = probe_measurement_adapter(descriptor)
    assert result["status"] == "available"
    assert result["observed_versions"] == ["3.10.0"]
    assert result["target_version"] == "3.11.0"
    assert result["target_version_observed"] is False
    assert result["package_imported"] is False
    assert result["process_launched"] is False
    assert result["capabilities_established"] is False
    assert result["qualification_established"] is False
    assert result["production_route_eligible"] is False


def test_provider_probe_requires_access_without_inspecting_credentials() -> None:
    descriptor = build_measurement_adapter_descriptor("world-labs-marble")
    result = probe_measurement_adapter(descriptor)
    assert result["status"] == "access_required"
    assert result["credentials_inspected"] is False
    assert result["execution_authorized"] is False


def test_capability_draft_preserves_unknown_and_requires_independent_evidence() -> None:
    descriptor = build_measurement_adapter_descriptor("mujoco-3")
    probe = probe_measurement_adapter(descriptor)
    draft = build_capability_draft(
        descriptor,
        probe,
        {
            "dynamic_collision_supported": {
                "state": "supported",
                "evidence_refs": ["fixture://independent-adapter-smoke"],
                "independently_verified": True,
            }
        },
    )
    assert set(draft["capabilities"]) == set(ALL_CAPABILITY_FIELDS)
    assert draft["capabilities"]["dynamic_collision_supported"]["state"] == ("supported")
    assert draft["capabilities"]["hydroelastic_contact_supported"]["state"] == ("unknown")
    assert draft["unknown_is_wildcard"] is False
    assert draft["install_probe_is_qualification"] is False
    assert draft["production_route_eligible"] is False

    with pytest.raises(MeasurementAdapterError, match="evidence_missing"):
        build_capability_draft(
            descriptor,
            probe,
            {"dynamic_collision_supported": {"state": "supported"}},
        )


def test_admission_packet_binds_inputs_but_completes_no_governance_stage() -> None:
    descriptor = build_measurement_adapter_descriptor("drake-1-55")
    probe = probe_measurement_adapter(descriptor)
    draft = build_capability_draft(descriptor, probe)
    packet = build_adapter_admission_packet(descriptor, probe, draft, source_snapshot_digest=SHA_A)
    assert packet["adapter_descriptor_digest"] == descriptor["adapter_descriptor_digest"]
    assert packet["adapter_probe_digest"] == probe["adapter_probe_digest"]
    assert packet["capability_draft_digest"] == draft["capability_draft_digest"]
    for stage in range(1, 8):
        suffix = {
            1: "source_verification_complete",
            2: "rights_review_complete",
            3: "adapter_feasibility_complete",
            4: "benchmark_preregistered",
            5: "independent_heldout_complete",
            6: "human_decision_complete",
            7: "catalog_admitted",
        }[stage]
        assert packet[f"r{stage}_{suffix}"] is False
    assert packet["production_route_eligible"] is False
    assert packet["execution_authorized"] is False


def test_adapter_runtime_contracts_match_checked_schema() -> None:
    schema = json.loads(
        (
            Path(__file__).parents[1] / "docs/schemas/measurement_adapter_runtime.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    descriptor = build_measurement_adapter_descriptor("mujoco-3")
    probe = probe_measurement_adapter(descriptor)
    draft = build_capability_draft(descriptor, probe)
    packet = build_adapter_admission_packet(descriptor, probe, draft, source_snapshot_digest=SHA_A)
    for artifact in (descriptor, probe, draft, packet):
        jsonschema.validate(artifact, schema)


def test_tampering_and_unknown_candidates_fail_closed() -> None:
    with pytest.raises(MeasurementAdapterError, match="candidate_unknown"):
        build_measurement_adapter_descriptor("vibes-physics")
    descriptor = build_measurement_adapter_descriptor("mujoco-3")
    tampered = copy.deepcopy(descriptor)
    tampered["production_route_eligible"] = True
    with pytest.raises(MeasurementAdapterError, match="production_route_eligible_must_be_false"):
        validate_measurement_adapter_descriptor(tampered)
    probe = probe_measurement_adapter(descriptor)
    draft = build_capability_draft(descriptor, probe)
    draft["capabilities"].pop("dynamic_collision_supported")
    with pytest.raises(MeasurementAdapterError, match="fields_incomplete"):
        validate_capability_draft(draft)
