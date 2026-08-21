from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.dual_task_rehearsal_contract import validate_task_freeze
from blueprint_pipeline.freeze_amendment_carry_forward import CARRIES_FORWARD
from blueprint_pipeline.native_task_policy_cadence_amendment import (
    NativeTaskPolicyCadenceAmendmentError,
    materialize_native_task_policy_cadence_amendment,
    materialize_native_task_policy_cadence_amendment_request,
    validate_native_task_policy_cadence_amendment,
)
from blueprint_pipeline.replacement_construction_bindings import (
    seal_replacement_construction_bindings,
)
from tests.test_native_task_arena_packet import (
    _construction_row_with_evidence,
    _materialized_construction,
    _request,
)


TASK_FREEZE = (
    Path(__file__).resolve().parents[1]
    / "docs/arm_decision_proof_v1/manifests/third_scene_840920_task_a_freeze.v1.json"
)
AUTHORIZED_AT = "2026-08-20T22:00:00Z"


def _source_pair(tmp_path: Path) -> tuple[Path, Path, Path]:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    freeze = json.loads(TASK_FREEZE.read_text(encoding="utf-8"))
    request = _request(evidence, articulated=True)
    task_asset = request["assets"][-1]
    asset_id = freeze["removal_plan"]["replacement_asset_id"]
    task_asset.update(
        {
            "semantic_role": "replacement",
            "asset_id": asset_id,
            "object_type": "ARTICULATION",
            "reset_state": {
                "joint_positions": {"door_hinge": 0.0, "locked_hinge": 0.0}
            },
        }
    )
    request["task_id"] = freeze["task_id"]
    request["task_freeze_digest"] = freeze["task_freeze_digest"]
    request["task_spec"]["prompt"] = freeze["prompt"]
    request["task_spec"]["control_frequency_hz"] = 20
    request["task_spec"]["subject_asset_id"] = asset_id
    request["construction_bindings"] = _materialized_construction(
        seal_replacement_construction_bindings(
            scene_freeze_digest=freeze["scene_freeze_digest"],
            task_freeze_join_digest="sha256:" + "4" * 64,
            bindings=[
                _construction_row_with_evidence(
                    {
                        "task_id": freeze["task_id"],
                        "asset_id": asset_id,
                        "task_freeze_digest": freeze["task_freeze_digest"],
                        "source_object_instance_id": freeze["source_object"][
                            "instance_id"
                        ],
                        "removal_id": freeze["removal_plan"]["removal_id"],
                        "mask_set_id": freeze["removal_plan"]["mask_set_id"],
                        "mask_set_receipt_digest": "sha256:" + "5" * 64,
                        "source_removal_receipt_digest": "sha256:" + "6" * 64,
                        "source_removal_qualified": True,
                        "collider_deletion_id": freeze["removal_plan"][
                            "collider_deletion_id"
                        ],
                        "source_collider_prim_path": freeze["removal_plan"][
                            "source_collider_prim_path"
                        ],
                        "collider_deletion_receipt_digest": "sha256:" + "7" * 64,
                        "collider_deletion_qualified": True,
                        "replacement_qualification_id": freeze["removal_plan"][
                            "replacement_qualification_id"
                        ],
                        "replacement_qualification_receipt_digest": "sha256:"
                        + "8" * 64,
                        "replacement_asset_sha256": task_asset["source"]["sha256"],
                        "replacement_simulator_import_qualified": True,
                    }
                )
            ],
        )
    )
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    request_path = tmp_path / "source_packet_request.json"
    request_path.write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return TASK_FREEZE, request_path, evidence


def _authority(tmp_path: Path, freeze_path: Path, request_path: Path) -> Path:
    path = tmp_path / "cadence_amendment_request.json"
    materialize_native_task_policy_cadence_amendment_request(
        task_freeze_path=freeze_path,
        packet_request_path=request_path,
        authorized_by="operator",
        authorized_at=AUTHORIZED_AT,
        output_path=path,
    )
    return path


def test_materializer_emits_exact_15_hz_packet_and_preserves_prompt(
    tmp_path: Path,
) -> None:
    freeze_path, request_path, evidence = _source_pair(tmp_path)
    authority = _authority(tmp_path, freeze_path, request_path)
    output = tmp_path / "amended"

    receipt = materialize_native_task_policy_cadence_amendment(
        amendment_request_path=authority,
        task_freeze_path=freeze_path,
        packet_request_path=request_path,
        evidence_root=evidence,
        output_dir=output,
    )

    source_freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    source_request = json.loads(request_path.read_text(encoding="utf-8"))
    amended_freeze = json.loads(
        (output / "native_task_policy_15hz_freeze.v1.json").read_text()
    )
    amended_request = json.loads(
        (
            output / "native_task_arena_packet_request.policy_15hz.v1.json"
        ).read_text()
    )
    scene_plan = json.loads(
        (
            output
            / "native_task_arena_packet/native_task_arena_scene_plan.v1.json"
        ).read_text()
    )

    validate_task_freeze(amended_freeze)
    assert amended_freeze["task_freeze_digest"] != source_freeze["task_freeze_digest"]
    assert amended_freeze["execution_contract"]["control_frequency_hz"] == 15
    assert amended_request["task_freeze_digest"] == amended_freeze["task_freeze_digest"]
    assert amended_request["task_spec"]["control_frequency_hz"] == 15
    assert amended_request["task_spec"]["prompt"] == source_freeze["prompt"]
    assert scene_plan["task_spec"]["prompt"] == source_freeze["prompt"]
    assert scene_plan["cadence"] == {
        "control_decimation": 8,
        "control_frequency_hz": 15.0,
        "episode_length_seconds": 1.6666666666666667,
        "maximum_action_steps": 20,
        "physics_dt_seconds": 1.0 / 120.0,
        "physics_frequency_hz": 120.0,
        "settle_window_samples": 4,
    }
    assert receipt["freeze_changed_paths"] == [
        "execution_contract.control_frequency_hz",
        "freeze_amendments",
    ]
    assert {
        "task_freeze_digest",
        "task_spec.control_frequency_hz",
        "construction_bindings.construction_digest",
    }.issubset(receipt["packet_request_changed_paths"])
    assert any(
        path.endswith(".evidence_receipts.task_freeze.sha256")
        for path in receipt["packet_request_changed_paths"]
    )
    assert receipt["scenario_suite_carry_forward"]["status"] == CARRIES_FORWARD
    assert receipt["provider_mutation_performed"] is False
    assert receipt["spend_incurred_usd"] == 0.0
    assert (
        validate_native_task_policy_cadence_amendment(
            receipt,
            source_freeze=source_freeze,
            source_packet_request=source_request,
            amended_freeze=amended_freeze,
            amended_packet_request=amended_request,
        )
        == receipt
    )


def test_hand_edited_amendment_request_is_refused_without_output(
    tmp_path: Path,
) -> None:
    freeze_path, request_path, evidence = _source_pair(tmp_path)
    authority = _authority(tmp_path, freeze_path, request_path)
    value = json.loads(authority.read_text())
    value["target_control_frequency_hz"] = 20
    authority.write_text(json.dumps(value), encoding="utf-8")
    output = tmp_path / "amended"

    with pytest.raises(
        NativeTaskPolicyCadenceAmendmentError,
        match="policy_cadence_amendment_request_field_invalid",
    ):
        materialize_native_task_policy_cadence_amendment(
            amendment_request_path=authority,
            task_freeze_path=freeze_path,
            packet_request_path=request_path,
            evidence_root=evidence,
            output_dir=output,
        )

    assert not output.exists()


def test_stale_source_binding_is_refused_without_output(tmp_path: Path) -> None:
    freeze_path, request_path, evidence = _source_pair(tmp_path)
    authority = _authority(tmp_path, freeze_path, request_path)
    request = json.loads(request_path.read_text())
    request["task_spec"]["maximum_action_steps"] += 1
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    request_path.write_text(json.dumps(request), encoding="utf-8")
    output = tmp_path / "amended"

    with pytest.raises(
        NativeTaskPolicyCadenceAmendmentError,
        match="source_packet_request_(digest|sha256)",
    ):
        materialize_native_task_policy_cadence_amendment(
            amendment_request_path=authority,
            task_freeze_path=freeze_path,
            packet_request_path=request_path,
            evidence_root=evidence,
            output_dir=output,
        )

    assert not output.exists()


def test_prompt_drift_is_refused_before_authority_materialization(
    tmp_path: Path,
) -> None:
    freeze_path, request_path, _evidence = _source_pair(tmp_path)
    request = json.loads(request_path.read_text())
    request["task_spec"]["prompt"] = "Open something else."
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(
        NativeTaskPolicyCadenceAmendmentError,
        match="policy_cadence_prompt_mismatch",
    ):
        _authority(tmp_path, freeze_path, request_path)


def test_stale_scenario_carry_forward_is_refused_even_if_redigested(
    tmp_path: Path,
) -> None:
    freeze_path, request_path, evidence = _source_pair(tmp_path)
    authority = _authority(tmp_path, freeze_path, request_path)
    output = tmp_path / "amended"
    receipt = materialize_native_task_policy_cadence_amendment(
        amendment_request_path=authority,
        task_freeze_path=freeze_path,
        packet_request_path=request_path,
        evidence_root=evidence,
        output_dir=output,
    )
    source_freeze = json.loads(freeze_path.read_text())
    source_request = json.loads(request_path.read_text())
    amended_freeze = json.loads(
        (output / "native_task_policy_15hz_freeze.v1.json").read_text()
    )
    amended_request = json.loads(
        (
            output / "native_task_arena_packet_request.policy_15hz.v1.json"
        ).read_text()
    )
    stale = receipt["scenario_suite_carry_forward"]
    stale["superseded_task_freeze_digest"] = "sha256:" + "0" * 64
    stale["carry_forward_digest"] = canonical_digest(
        stale, digest_field="carry_forward_digest"
    )
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )

    with pytest.raises(
        NativeTaskPolicyCadenceAmendmentError,
        match="policy_cadence_scenario_carry_forward_stale",
    ):
        validate_native_task_policy_cadence_amendment(
            receipt,
            source_freeze=source_freeze,
            source_packet_request=source_request,
            amended_freeze=amended_freeze,
            amended_packet_request=amended_request,
        )


@pytest.mark.parametrize(
    ("freeze_hz", "packet_hz", "physics_hz", "blocker"),
    [
        (15, 20, 120, "policy_cadence_source_freeze_frequency_invalid"),
        (20, 15, 120, "policy_cadence_source_packet_frequency_invalid"),
        (20, 20, 100, "policy_cadence_physics_frequency_invalid"),
    ],
)
def test_only_exact_20_to_15_at_120_hz_is_admitted(
    tmp_path: Path,
    freeze_hz: int,
    packet_hz: int,
    physics_hz: int,
    blocker: str,
) -> None:
    freeze_path, request_path, _evidence = _source_pair(tmp_path)
    freeze = json.loads(freeze_path.read_text())
    freeze["execution_contract"]["control_frequency_hz"] = freeze_hz
    freeze["task_freeze_digest"] = canonical_digest(
        freeze, digest_field="task_freeze_digest"
    )
    local_freeze = tmp_path / "source_freeze.json"
    local_freeze.write_text(json.dumps(freeze), encoding="utf-8")
    request = json.loads(request_path.read_text())
    request["task_freeze_digest"] = freeze["task_freeze_digest"]
    request["task_spec"]["control_frequency_hz"] = packet_hz
    request["physics_frequency_hz"] = physics_hz
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(NativeTaskPolicyCadenceAmendmentError, match=blocker):
        _authority(tmp_path, local_freeze, request_path)
