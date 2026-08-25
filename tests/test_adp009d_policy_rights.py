from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from blueprint_pipeline import adp009d_policy_rights as rights_module
from blueprint_pipeline.adp009d_groot_worker_identity import (
    expected_checkpoint_content_binding,
)
from blueprint_pipeline.adp009d_policy_rights import (
    CandidatePolicyRightsError,
    build_candidate_policy_rights,
    validate_candidate_policy_rights,
)
from blueprint_pipeline.adp009d_scene_policy_readiness import (
    load_scene_policy_readiness,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_policy_bundle import (
    ADP009D_POLICY_READINESS_PATH,
    ADP009D_SCENARIO_SUITE_PATH,
    _candidate_runtime_binding,
)


def _readiness() -> dict:
    return load_scene_policy_readiness(
        ADP009D_POLICY_READINESS_PATH,
        scenario_suite_path=ADP009D_SCENARIO_SUITE_PATH,
    )


@pytest.mark.parametrize("candidate_id", ["pi05_droid", "groot_n17_droid"])
def test_candidate_rights_binding_is_deterministic_and_checkpoint_exact(
    candidate_id: str,
) -> None:
    policy, _, _ = _candidate_runtime_binding(candidate_id)
    spec = asdict(policy)
    first = build_candidate_policy_rights(
        _readiness(),
        candidate_id=candidate_id,
        policy_spec=spec,
        runtime_robot_id="franka_panda",
        scene_plan_digest="sha256:" + "1" * 64,
    )
    second = build_candidate_policy_rights(
        _readiness(),
        candidate_id=candidate_id,
        policy_spec=spec,
        runtime_robot_id="franka_panda",
        scene_plan_digest="sha256:" + "1" * 64,
    )

    assert first == second
    assert first["candidate_id"] == candidate_id
    assert first["rights"]["rights_ready"] is True
    assert first["interface_identity"]["action_adapter"]
    assert first["rights_receipt_digest"] == canonical_digest(
        first, digest_field="rights_receipt_digest"
    )
    if candidate_id == "pi05_droid":
        assert first["checkpoint_identity"]["repository"] == spec["checkpoint_uri"]
        assert first["checkpoint_identity"]["inventory_digest"] == (
            "sha256:" + spec["checkpoint_inventory_sha256"]
        )
        assert first["rights"]["checkpoint_specific_terms_bound"] is True
    else:
        assert first["checkpoint_identity"]["revision"] == spec[
            "checkpoint_revision"
        ]
        assert first["rights"]["gated_backbone"]["access_probe_status"] == (
            "authorized"
        )
        assert first["checkpoint_identity"]["content_manifest_digest"] == (
            expected_checkpoint_content_binding()["file_manifest_digest"]
        )


@pytest.mark.parametrize(
    ("candidate_id", "mutate", "message"),
    [
        (
            "pi05_droid",
            lambda value: value["rights"].update(rights_ready=False),
            "candidate_policy_rights_not_ready",
        ),
        (
            "pi05_droid",
            lambda value: value["rights"]["rights_provenance"].pop(
                "gemma_terms"
            ),
            "candidate_policy_rights_pi05_terms_invalid",
        ),
        (
            "groot_n17_droid",
            lambda value: value["rights"]["gated_backbone"].update(
                access_probe_status="missing"
            ),
            "candidate_policy_rights_groot_gated_access_invalid",
        ),
    ],
)
def test_candidate_rights_binding_fails_closed_after_semantic_tampering(
    candidate_id: str, mutate, message: str
) -> None:
    policy, _, _ = _candidate_runtime_binding(candidate_id)
    spec = asdict(policy)
    value = build_candidate_policy_rights(
        _readiness(),
        candidate_id=candidate_id,
        policy_spec=spec,
        runtime_robot_id="franka_panda",
        scene_plan_digest="sha256:" + "1" * 64,
    )
    mutate(value)
    value["rights_receipt_digest"] = canonical_digest(
        value, digest_field="rights_receipt_digest"
    )

    with pytest.raises(CandidatePolicyRightsError, match=message):
        validate_candidate_policy_rights(
            value, candidate_id=candidate_id, policy_spec=spec
        )


def test_candidate_rights_binding_rejects_cross_candidate_use() -> None:
    pi_policy, _, _ = _candidate_runtime_binding("pi05_droid")
    groot_policy, _, _ = _candidate_runtime_binding("groot_n17_droid")
    value = build_candidate_policy_rights(
        _readiness(),
        candidate_id="pi05_droid",
        policy_spec=asdict(pi_policy),
        runtime_robot_id="franka_panda",
        scene_plan_digest="sha256:" + "1" * 64,
    )

    with pytest.raises(CandidatePolicyRightsError, match="candidate_invalid"):
        validate_candidate_policy_rights(
            value,
            candidate_id="groot_n17_droid",
            policy_spec=asdict(groot_policy),
        )


def test_groot_rights_reject_same_size_content_identity_tamper() -> None:
    policy, _, _ = _candidate_runtime_binding("groot_n17_droid")
    spec = asdict(policy)
    value = build_candidate_policy_rights(
        _readiness(),
        candidate_id="groot_n17_droid",
        policy_spec=spec,
        runtime_robot_id="franka_panda",
        scene_plan_digest="sha256:" + "1" * 64,
    )
    value["checkpoint_identity"]["content_manifest"][0]["digest"] = "0" * 40
    value["checkpoint_identity"]["content_manifest_digest"] = canonical_digest(
        {"files": value["checkpoint_identity"]["content_manifest"]}
    )
    value["rights_receipt_digest"] = canonical_digest(
        value, digest_field="rights_receipt_digest"
    )

    with pytest.raises(
        CandidatePolicyRightsError,
        match="candidate_policy_rights_groot_identity_mismatch",
    ):
        validate_candidate_policy_rights(
            value, candidate_id="groot_n17_droid", policy_spec=spec
        )


def test_groot_rights_refuse_readiness_weight_digest_drift() -> None:
    policy, _, _ = _candidate_runtime_binding("groot_n17_droid")
    readiness = _readiness()
    candidate = next(
        row
        for row in readiness["candidates"]
        if row["candidate_id"] == "groot_n17_droid"
    )
    candidate["checkpoint"]["weight_sha256"][0] = "sha256:" + "0" * 64

    with pytest.raises(
        CandidatePolicyRightsError,
        match="candidate_policy_rights_groot_source_weight_identity_invalid",
    ):
        build_candidate_policy_rights(
            readiness,
            candidate_id="groot_n17_droid",
            policy_spec=asdict(policy),
            runtime_robot_id="franka_panda",
            scene_plan_digest="sha256:" + "1" * 64,
        )


def test_groot_content_binding_matches_authoritative_candidate_inventory() -> None:
    inventory_path = (
        ADP009D_POLICY_READINESS_PATH.parent
        / "adp009d_policy_candidate_inventory.v1.json"
    )
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    candidate = next(
        row
        for row in inventory["candidates"]
        if row["candidate_id"] == "groot_n17_droid"
    )
    checkpoint = candidate["checkpoint"]
    binding = expected_checkpoint_content_binding()
    by_path = {row["path"]: row for row in binding["file_manifest"]}

    assert binding["inventory_digest"] == checkpoint["snapshot_inventory_digest"]
    assert binding["file_count"] == checkpoint["file_count"]
    assert binding["total_bytes"] == checkpoint["total_bytes"]
    for weight in checkpoint["weight_files"]:
        assert by_path[weight["path"]] == {
            "path": weight["path"],
            "size_bytes": weight["size_bytes"],
            "digest_algorithm": "sha256",
            "digest": weight["sha256"].removeprefix("sha256:"),
        }
    for path, digest in checkpoint["metadata_blob_identities"].items():
        assert by_path[path]["digest_algorithm"] == "git_blob_sha1"
        assert by_path[path]["digest"] == digest


def test_candidate_rights_binding_requires_validated_readiness_bytes(
    tmp_path: Path,
) -> None:
    readiness = json.loads(ADP009D_POLICY_READINESS_PATH.read_text(encoding="utf-8"))
    readiness["candidates"][0]["rights_ready"] = False
    readiness["readiness_digest"] = canonical_digest(
        readiness, digest_field="readiness_digest"
    )
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps(readiness), encoding="utf-8")

    with pytest.raises(ValueError, match="rights_ready_invalid"):
        load_scene_policy_readiness(
            path, scenario_suite_path=ADP009D_SCENARIO_SUITE_PATH
        )


def test_candidate_rights_materializer_has_a_provider_free_cli(tmp_path: Path) -> None:
    policy, _, _ = _candidate_runtime_binding("groot_n17_droid")
    policy_spec = tmp_path / "policy-spec.json"
    policy_spec.write_text(json.dumps(asdict(policy)), encoding="utf-8")
    output = tmp_path / "candidate-rights.json"

    exit_code = rights_module.main(
        [
            "--readiness-path",
            str(ADP009D_POLICY_READINESS_PATH),
            "--scenario-suite-path",
            str(ADP009D_SCENARIO_SUITE_PATH),
            "--candidate-id",
            "groot_n17_droid",
            "--policy-spec",
            str(policy_spec),
            "--runtime-robot-id",
            "franka_panda",
            "--scene-plan-digest",
            "sha256:" + "1" * 64,
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert validate_candidate_policy_rights(
        json.loads(output.read_text(encoding="utf-8")),
        candidate_id="groot_n17_droid",
        policy_spec=asdict(policy),
    )["candidate_id"] == "groot_n17_droid"
