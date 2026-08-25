from __future__ import annotations

import copy
from typing import Any

import pytest

from blueprint_pipeline.company_policy_container_contract import (
    BLOCKER_CONTRACT_INVALID,
    BLOCKER_IMAGE_NOT_DIGEST_PINNED,
    BLOCKER_RAW_BOUNDS_NARROWER,
    BLOCKER_REMOTE_ENDPOINT_FORBIDDEN,
    SCHEMA_VERSION,
    CompanyPolicyContractError,
    company_policy_channel_contracts,
    validate_company_policy_container_contract,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _golden_contract() -> dict[str, Any]:
    """A complete, admissible company contract; each test mutates one field."""

    return {
        "schema_version": SCHEMA_VERSION,
        "policy_id": "acme_widget_grasp_v3",
        "company_id": "acme_robotics",
        "display_name": "ACME Widget Grasp v3",
        "checkpoint_identity": {
            "repository": "https://models.acme.example/widget-grasp",
            "revision": "2026.08.1",
            "inventory_digest": "sha256:" + "a" * 64,
        },
        "claim_ceiling": "development_only",
        "rights": {
            "license": "ACME Evaluation License 2026-08",
            "rights_provenance": "acme_msa_2026_07_appendix_b",
            "provider_use_status": "permitted_on_rented_gpu_for_this_evaluation",
            "redistribution_status": "no_redistribution_weights_stay_in_container",
            "rights_ready": True,
        },
        "container": {
            "image": "registry.acme.example/widget-grasp@sha256:" + "b" * 64,
            "serve_command": ["python", "-m", "acme_policy.serve", "--port", "8600"],
            "port": 8600,
            "handshake_kind": "http_json_v1",
            "credential_files": ["acme_license_token"],
            "gpu_required": True,
        },
        "observation_schema": {
            "cameras": [
                {"name": "exterior_image_1_left", "width": 320, "height": 180},
                {"name": "wrist_image_left", "width": 320, "height": 180},
            ],
            "state_keys": ["joint_position", "gripper_position"],
        },
        "action_schema": {
            "action_space_id": "acme_joint_velocity_v1",
            "chunk_rows": 15,
            "channels": [
                *(
                    {
                        "name": f"joint_velocity_{index}",
                        "kind": "bounded_continuous",
                        "command_interval": [-1.0, 1.0],
                        "raw_accepted_bounds": [-1.0, 1.0],
                        "executed_semantics": (
                            "normalized velocity mapped to bounded position delta"
                        ),
                    }
                    for index in range(7)
                ),
                {
                    "name": "gripper",
                    "kind": "threshold_scalar",
                    "command_interval": [0.0, 1.0],
                    "raw_accepted_bounds": [-0.25, 1.25],
                    "executed_semantics": (
                        "clip_to_command_interval_then_threshold_at_0.5"
                    ),
                },
            ],
        },
    }


def _refuses(contract: dict[str, Any], prefix: str) -> None:
    with pytest.raises(CompanyPolicyContractError) as excinfo:
        validate_company_policy_container_contract(contract)
    assert any(error.startswith(prefix) for error in excinfo.value.errors), (
        excinfo.value.errors
    )


def test_golden_contract_normalizes_with_injected_loopback_and_digest() -> None:
    normalized = validate_company_policy_container_contract(_golden_contract())

    # The endpoint is injected, never declared: loopback on the declared port.
    assert normalized["endpoint"] == {"host": "127.0.0.1", "port": 8600}
    assert normalized["claim_ceiling"] == "development_only"
    assert normalized["policy_id"] == "acme_widget_grasp_v3"
    assert len(normalized["action_schema"]["channels"]) == 8
    assert normalized["contract_digest"] == canonical_digest(
        normalized, digest_field="contract_digest"
    )
    assert normalized["contract_digest"].startswith("sha256:")

    # Deterministic: the same declared bytes always seal to the same digest.
    again = validate_company_policy_container_contract(_golden_contract())
    assert again["contract_digest"] == normalized["contract_digest"]


def test_the_normalized_output_revalidates_as_a_fixed_point() -> None:
    """Downstream seams re-validate rather than trust; the output must pass."""

    normalized = validate_company_policy_container_contract(_golden_contract())
    revalidated = validate_company_policy_container_contract(normalized)
    assert revalidated == normalized

    # But a tampered digest on an otherwise valid contract refuses.
    tampered = copy.deepcopy(normalized)
    tampered["contract_digest"] = "sha256:" + "0" * 64
    _refuses(tampered, f"{BLOCKER_CONTRACT_INVALID}:contract_digest_mismatch")


@pytest.mark.parametrize(
    "field",
    [
        "license",
        "rights_provenance",
        "provider_use_status",
        "redistribution_status",
        "rights_ready",
    ],
)
def test_any_missing_rights_field_refuses(field: str) -> None:
    """Unrecorded rights fail closed -- the program rule, not a preference."""

    contract = _golden_contract()
    del contract["rights"][field]
    _refuses(contract, f"{BLOCKER_CONTRACT_INVALID}:rights_{field}")


def test_rights_ready_must_be_exactly_true() -> None:
    contract = _golden_contract()
    contract["rights"]["rights_ready"] = "yes"
    _refuses(contract, f"{BLOCKER_CONTRACT_INVALID}:rights_rights_ready")


def test_claim_ceiling_other_than_development_only_refuses() -> None:
    for ceiling in ("production", "evaluation", "", None):
        contract = _golden_contract()
        contract["claim_ceiling"] = ceiling
        _refuses(contract, f"{BLOCKER_CONTRACT_INVALID}:claim_ceiling")


@pytest.mark.parametrize(
    "image",
    [
        "registry.acme.example/widget-grasp:latest",  # tag-only
        "registry.acme.example/widget-grasp",  # bare
        "registry.acme.example/widget-grasp@sha256:beef",  # short digest
        "registry.acme.example/widget-grasp@md5:" + "b" * 64,  # wrong algorithm
        "",
    ],
)
def test_unpinned_container_images_refuse(image: str) -> None:
    """A tag is a moving branch with a registry attached."""

    contract = _golden_contract()
    contract["container"]["image"] = image
    _refuses(contract, BLOCKER_IMAGE_NOT_DIGEST_PINNED)


def test_a_tag_plus_digest_reference_is_still_digest_pinned() -> None:
    contract = _golden_contract()
    contract["container"]["image"] = (
        "registry.acme.example/widget-grasp:v3@sha256:" + "b" * 64
    )
    normalized = validate_company_policy_container_contract(contract)
    assert normalized["container"]["image"].endswith("@sha256:" + "b" * 64)


def test_declaring_a_host_anywhere_refuses() -> None:
    """Loopback serving is doctrine, not default: no input may name a host."""

    remote = _golden_contract()
    remote["endpoint"] = {"host": "policy.acme.example", "port": 443}
    _refuses(remote, BLOCKER_REMOTE_ENDPOINT_FORBIDDEN)

    container_host = _golden_contract()
    container_host["container"]["host"] = "0.0.0.0"
    _refuses(container_host, f"{BLOCKER_REMOTE_ENDPOINT_FORBIDDEN}:container.host")

    top_level_host = _golden_contract()
    top_level_host["host"] = "127.0.0.1"
    _refuses(top_level_host, f"{BLOCKER_REMOTE_ENDPOINT_FORBIDDEN}:host")

    # Even the loopback host on the wrong port is a declaration attempt.
    wrong_port = _golden_contract()
    wrong_port["endpoint"] = {"host": "127.0.0.1", "port": 9999}
    _refuses(wrong_port, BLOCKER_REMOTE_ENDPOINT_FORBIDDEN)


@pytest.mark.parametrize("port", [80, 1023, 0, -1, 65536, "8600", True, None])
def test_invalid_ports_refuse(port: Any) -> None:
    contract = _golden_contract()
    contract["container"]["port"] = port
    _refuses(contract, f"{BLOCKER_CONTRACT_INVALID}:container_port")


@pytest.mark.parametrize(
    "filename",
    [
        "/etc/passwd",
        "../render_api_key",
        "nested/secret",
        "windows\\secret",
        "host:corrupting",
        "..",
        "",
    ],
)
def test_credential_files_must_be_bare_filenames(filename: str) -> None:
    """These resolve against the canonical secrets dir and nowhere else."""

    contract = _golden_contract()
    contract["container"]["credential_files"] = [filename]
    _refuses(contract, f"{BLOCKER_CONTRACT_INVALID}:container_credential_file")


def test_an_empty_credential_file_list_is_admissible() -> None:
    contract = _golden_contract()
    contract["container"]["credential_files"] = []
    normalized = validate_company_policy_container_contract(contract)
    assert normalized["container"]["credential_files"] == []


def test_empty_channels_refuse() -> None:
    contract = _golden_contract()
    contract["action_schema"]["channels"] = []
    _refuses(contract, f"{BLOCKER_CONTRACT_INVALID}:action_channels_empty")


def test_raw_bounds_narrower_than_command_interval_refuse() -> None:
    """A runtime executing values its validator refused is self-contradictory."""

    contract = _golden_contract()
    contract["action_schema"]["channels"][7]["raw_accepted_bounds"] = [0.1, 0.9]
    _refuses(contract, f"{BLOCKER_RAW_BOUNDS_NARROWER}:gripper")

    # Narrower on one side only still refuses.
    one_side = _golden_contract()
    one_side["action_schema"]["channels"][7]["raw_accepted_bounds"] = [0.1, 1.25]
    _refuses(one_side, f"{BLOCKER_RAW_BOUNDS_NARROWER}:gripper")

    # Exactly equal bounds are the degenerate superset and admissible.
    equal = _golden_contract()
    equal["action_schema"]["channels"][7]["raw_accepted_bounds"] = [0.0, 1.0]
    validate_company_policy_container_contract(equal)


@pytest.mark.parametrize("chunk_rows", [0, -3, 1.5, True, None, "15"])
def test_non_positive_chunk_rows_refuse(chunk_rows: Any) -> None:
    contract = _golden_contract()
    contract["action_schema"]["chunk_rows"] = chunk_rows
    _refuses(contract, f"{BLOCKER_CONTRACT_INVALID}:action_chunk_rows")


def test_unknown_handshake_kinds_refuse() -> None:
    """A new handshake needs a new episode client -- code, not declared data."""

    contract = _golden_contract()
    contract["container"]["handshake_kind"] = "grpc_proto3"
    _refuses(contract, f"{BLOCKER_CONTRACT_INVALID}:container_handshake_kind")


def test_frozen_candidate_id_collisions_refuse_via_the_provisioning_guard() -> None:
    """A company contract must never impersonate a frozen ADP candidate."""

    from blueprint_pipeline.adp009d_policy_candidate_admission import (
        EXPECTED_CANDIDATES,
    )
    from blueprint_pipeline.adp009d_policy_provisioning import (
        BLOCKER_FROZEN_CANDIDATE_COLLISION,
        PolicyProvisioningError,
        assert_not_adp_frozen_candidate,
        company_policy_container_commands,
    )

    frozen_ids = (
        "pi05_droid",
        "groot_n17_droid",
        "groot_n16_droid",
        "cosmos3_edge_policy_droid",
    )
    # The guard reads the live registry; these named ids must all be in it.
    assert set(frozen_ids) <= set(EXPECTED_CANDIDATES)
    for policy_id in frozen_ids:
        with pytest.raises(PolicyProvisioningError) as excinfo:
            assert_not_adp_frozen_candidate(policy_id)
        assert any(
            error.startswith(f"{BLOCKER_FROZEN_CANDIDATE_COLLISION}:{policy_id}")
            for error in excinfo.value.errors
        )
        # And through the full seam: a valid contract with a colliding id
        # still refuses before any command is emitted.
        contract = _golden_contract()
        contract["policy_id"] = policy_id
        with pytest.raises(PolicyProvisioningError):
            company_policy_container_commands(contract)

    assert_not_adp_frozen_candidate("acme_widget_grasp_v3")


def test_unknown_fields_refuse_rather_than_silently_drop() -> None:
    """Anything the digest would not cover must not appear to be admitted."""

    contract = _golden_contract()
    contract["environment_variables"] = {"CUDA_VISIBLE_DEVICES": "0"}
    _refuses(contract, f"{BLOCKER_CONTRACT_INVALID}:unknown_field:environment_variables")


def test_channel_contracts_helper_returns_the_validated_channels() -> None:
    channels = company_policy_channel_contracts(_golden_contract())
    assert len(channels) == 8
    assert channels[7]["name"] == "gripper"
    assert channels[7]["command_interval"] == [0.0, 1.0]
    assert channels[7]["raw_accepted_bounds"] == [-0.25, 1.25]
    # And it validates rather than trusts: a broken contract refuses here too.
    broken = _golden_contract()
    broken["action_schema"]["channels"] = []
    with pytest.raises(CompanyPolicyContractError):
        company_policy_channel_contracts(broken)


def test_nonfinite_numbers_never_enter_the_contract() -> None:
    contract = _golden_contract()
    contract["action_schema"]["channels"][0]["command_interval"] = [
        float("-inf"),
        1.0,
    ]
    _refuses(contract, BLOCKER_CONTRACT_INVALID)
