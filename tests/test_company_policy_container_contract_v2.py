from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from blueprint_pipeline.company_policy_container_contract_v2 import (
    ACTION_ROUTE,
    BLOCKER_IMAGE,
    BLOCKER_INVALID,
    BLOCKER_SECURITY,
    CLAIM_CEILING,
    LIVE_HANDSHAKE_KIND,
    LIVE_PROTOCOL_VERSION,
    SCHEMA_VERSION,
    SECURITY_PROFILE,
    CompanyPolicyContainerContractV2Error,
    validate_company_policy_container_contract_v2,
)
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest


SCHEMA_PATH = (
    Path(__file__).parents[1]
    / "docs/webapp_handoff/company-policy-container.v2/contract.schema.json"
)
HANDOFF_ROOT = SCHEMA_PATH.parent


def _contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "policy_id": "acme_widget_grasp_v3",
        "company_id": "acme_robotics",
        "display_name": "ACME Widget Grasp v3",
        "checkpoint_identity": {
            "repository": "registry.acme.example/models/widget-grasp",
            "revision": "2026.08.1",
            "inventory_digest": "sha256:" + "a" * 64,
        },
        "claim_ceiling": CLAIM_CEILING,
        "rights": {
            "license": "ACME evaluation license 2026-08",
            "rights_provenance": "acme_msa_2026_07_appendix_b",
            "rights_evidence_uri": "blueprint-rights://acme/widget-grasp/2026-08",
            "rights_evidence_digest": "sha256:" + "b" * 64,
            "provider_use_status": "permitted_for_this_evaluation",
            "redistribution_status": "weights_remain_in_company_container",
            "rights_ready": True,
        },
        "container": {
            "image": "registry.acme.example/widget-grasp@sha256:" + "c" * 64,
            "visibility": "private",
            "serve_command": ["python", "-m", "acme_policy.serve", "--port", "8600"],
            "port": 8600,
            "handshake": {
                "kind": LIVE_HANDSHAKE_KIND,
                "protocol_version": LIVE_PROTOCOL_VERSION,
                "action_route": ACTION_ROUTE,
            },
            "run_as_uid": 65532,
            "run_as_gid": 65532,
            "gpu_required": True,
            "resources": {
                "cpus": 8.0,
                "memory_mib": 32768,
                "pids_limit": 512,
                "tmpfs_mib": 2048,
                "startup_timeout_seconds": 300,
                "request_timeout_ms": 2500,
            },
        },
        "robot": {
            "embodiment_id": "franka_panda_robotiq_2f85_v1",
            "definition_uri": "blueprint-robot://franka-panda-robotiq-2f85/v1",
            "definition_digest": "sha256:" + "d" * 64,
            "joint_names": [f"panda_joint{i}" for i in range(1, 8)],
            "joint_limits": [
                {
                    "name": f"panda_joint{i}",
                    "lower": -2.0,
                    "upper": 2.0,
                    "unit": "radian",
                }
                for i in range(1, 8)
            ],
            "gripper": {
                "name": "gripper",
                "command_interval": [0.0, 1.0],
                "unit": "normalized_fraction",
                "executed_semantics": "clip_then_map_to_parallel_jaw_width",
            },
        },
        "observation_schema": {
            "cameras": [
                {
                    "name": "external_rgb",
                    "width": 320,
                    "height": 180,
                    "color_space": "rgb",
                    "dtype": "uint8",
                    "layout": "hwc",
                    "encoding": "lossless_png",
                    "calibration_uri": "blueprint-calibration://scene/external/v1",
                    "calibration_digest": "sha256:" + "e" * 64,
                },
                {
                    "name": "wrist_rgb",
                    "width": 320,
                    "height": 180,
                    "color_space": "rgb",
                    "dtype": "uint8",
                    "layout": "hwc",
                    "encoding": "lossless_png",
                    "calibration_uri": "blueprint-calibration://scene/wrist/v1",
                    "calibration_digest": "sha256:" + "f" * 64,
                },
            ],
            "state_fields": [
                {
                    "name": "joint_position",
                    "shape": [7],
                    "dtype": "float32",
                    "unit": "radian",
                },
                {
                    "name": "gripper_position",
                    "shape": [1],
                    "dtype": "float32",
                    "unit": "normalized_fraction",
                },
            ],
            "prompt": {"mode": "text", "required": True},
            "control_frequency_hz": 15.0,
        },
        "action_schema": {
            "adapter_id": "absolute_joint_position_gripper_v1",
            "chunk_rows": 15,
            "channels": [
                *(
                    {
                        "name": f"panda_joint{i}",
                        "kind": "bounded_continuous",
                        "command_interval": [-2.0, 2.0],
                        "raw_accepted_bounds": [-2.0, 2.0],
                        "unit": "radian",
                        "executed_semantics": "absolute_joint_position",
                    }
                    for i in range(1, 8)
                ),
                {
                    "name": "gripper",
                    "kind": "threshold_scalar",
                    "command_interval": [0.0, 1.0],
                    "raw_accepted_bounds": [-0.25, 1.25],
                    "unit": "normalized_fraction",
                    "executed_semantics": "clip_then_map_to_parallel_jaw_width",
                },
            ],
            "normalization": {
                "observation": "none",
                "action": "none",
                "gripper": "raw_envelope_then_clip_to_command_interval",
            },
        },
    }


def _refuses(contract: dict[str, Any], prefix: str) -> None:
    with pytest.raises(CompanyPolicyContainerContractV2Error) as excinfo:
        validate_company_policy_container_contract_v2(contract)
    assert any(error.startswith(prefix) for error in excinfo.value.errors)


def test_golden_contract_is_a_digest_stable_fixed_point() -> None:
    normalized = validate_company_policy_container_contract_v2(_contract())

    assert normalized["security_profile"] == SECURITY_PROFILE
    assert normalized["contract_digest"] == cross_runtime_canonical_digest(
        normalized, digest_field="contract_digest"
    )
    assert normalized["contract_digest"] == (
        "sha256:1337317266b54bade7e6b78511f8da522db8aa284d71425e8cd744035cde10d3"
    )
    assert validate_company_policy_container_contract_v2(normalized) == normalized


def test_checked_in_webapp_schema_accepts_declared_and_normalized_forms() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)

    validator.validate(_contract())
    validator.validate(validate_company_policy_container_contract_v2(_contract()))


def test_webapp_handoff_manifest_binds_every_shared_artifact() -> None:
    manifest = json.loads((HANDOFF_ROOT / "artifact-manifest.json").read_text(encoding="utf-8"))
    assert manifest["manifest_self_excluded"] is True
    assert {row["path"] for row in manifest["artifacts"]} == {
        "README.md",
        "contract.schema.json",
    }
    for row in manifest["artifacts"]:
        payload = (HANDOFF_ROOT / row["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == row["sha256"]


@pytest.mark.parametrize(
    "image",
    [
        "registry.acme.example/widget-grasp:latest",
        "registry.acme.example/widget-grasp",
        "registry.acme.example/widget-grasp@sha256:beef",
    ],
)
def test_only_digest_pinned_images_are_admitted(image: str) -> None:
    contract = _contract()
    contract["container"]["image"] = image
    _refuses(contract, BLOCKER_IMAGE)


def test_checkpoint_inventory_is_digest_bound() -> None:
    contract = _contract()
    del contract["checkpoint_identity"]["inventory_digest"]
    _refuses(contract, f"{BLOCKER_INVALID}:checkpoint_inventory_digest")


def test_only_the_versioned_blueprint_action_route_is_admitted() -> None:
    contract = _contract()
    contract["container"]["handshake"]["action_route"] = "/custom/actions"
    _refuses(contract, f"{BLOCKER_INVALID}:handshake_action_route")


@pytest.mark.parametrize(
    "field,value",
    [
        ("registry_token", "secret"),
        ("credential_files", ["token"]),
        ("environment", {"TOKEN": "secret"}),
        ("mounts", ["/scene"]),
        ("network_mode", "host"),
    ],
)
def test_credentials_mounts_environment_and_network_are_not_contract_fields(
    field: str, value: Any
) -> None:
    contract = _contract()
    contract["container"][field] = value
    _refuses(contract, f"{BLOCKER_INVALID}:unknown_field:container.{field}")


def test_company_cannot_weaken_the_injected_security_profile() -> None:
    contract = _contract()
    contract["security_profile"] = {
        **SECURITY_PROFILE,
        "network_mode": "host",
    }
    _refuses(contract, BLOCKER_SECURITY)


def test_robot_joint_order_and_limits_must_match_exactly() -> None:
    contract = _contract()
    contract["robot"]["joint_limits"][0]["name"] = "panda_joint2"
    _refuses(contract, f"{BLOCKER_INVALID}:robot_joint_limit:0")


@pytest.mark.parametrize("field", ["calibration_uri", "calibration_digest"])
def test_every_policy_camera_is_calibration_bound(field: str) -> None:
    contract = _contract()
    del contract["observation_schema"]["cameras"][0][field]
    _refuses(contract, f"{BLOCKER_INVALID}:observation_camera")


def test_raw_action_envelope_must_cover_the_executed_command_interval() -> None:
    contract = _contract()
    contract["action_schema"]["channels"][-1]["raw_accepted_bounds"] = [0.1, 0.9]
    _refuses(contract, f"{BLOCKER_INVALID}:action_channel_raw_narrower")


def test_contract_digest_tampering_refuses() -> None:
    normalized = validate_company_policy_container_contract_v2(_contract())
    normalized["contract_digest"] = "sha256:" + "0" * 64
    _refuses(normalized, f"{BLOCKER_INVALID}:contract_digest_mismatch")


def test_unknown_fields_refuse_instead_of_disappearing_from_the_digest() -> None:
    contract = copy.deepcopy(_contract())
    contract["robot"]["undocumented_adapter"] = "magic"
    _refuses(
        contract,
        f"{BLOCKER_INVALID}:unknown_field:robot.undocumented_adapter",
    )
