"""A freeze amendment costs what it touches, not what it hashes.

Scene 840920's washer door was hinged the wrong way and jammed at 6.01 degrees.
Correcting one joint-axis component changed the freeze file, and every sealed
CAD receipt refused -- because those receipts pin the freeze by whole-file
sha256, and the CAD agent reads two fields out of that file, neither of which
moved.

These tests pin the distinction: an amendment that misses every consumed field
carries receipts forward, an amendment that hits one does not, and a proof for
one amendment cannot be reused for another.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.freeze_amendment_carry_forward import (
    CARRIES_FORWARD,
    REQUIRES_REDERIVATION,
    FreezeAmendmentCarryForwardError,
    evaluate_freeze_amendment_carry_forward,
    main,
    validate_freeze_amendment_carry_forward,
)


CAD_REQUEST = "simready_cad_agent_request.v1"
SUPERSEDED_FILE_SHA = "sha256:" + "a" * 64
AMENDED_FILE_SHA = "sha256:" + "b" * 64


def _freeze(*, axis_z: int = 1, task_id: str = "task_a") -> dict:
    payload = {
        "schema_version": "dual_task_task_freeze.v1",
        "task_id": task_id,
        "task_kind": "articulated_interaction",
        "removal_plan": {
            "replacement_asset_id": "washer_replacement_v1",
            "source_collider_prim_path": "/Root/source_165",
        },
        "articulation_graph": {
            "joints": [
                {"joint_id": "door_hinge", "axis": [0, 0, axis_z], "role": "target"}
            ]
        },
        "task_freeze_digest": "",
    }
    payload["task_freeze_digest"] = canonical_digest(
        payload, digest_field="task_freeze_digest"
    )
    return payload


def _amended(**kwargs) -> dict:
    """The real amendment: flip the axis and record why it was safe."""

    payload = _freeze(**kwargs)
    payload["freeze_amendments"] = [
        {"amended_at": "2026-08-17", "reason": "door opened into its own cabinet"}
    ]
    payload["task_freeze_digest"] = ""
    payload["task_freeze_digest"] = canonical_digest(
        payload, digest_field="task_freeze_digest"
    )
    return payload


def _evaluate(superseded: dict, amended: dict, schema: str = CAD_REQUEST) -> dict:
    return evaluate_freeze_amendment_carry_forward(
        superseded_freeze=superseded,
        amended_freeze=amended,
        sealed_schema=schema,
        superseded_file_sha256=SUPERSEDED_FILE_SHA,
        amended_file_sha256=AMENDED_FILE_SHA,
    )


def test_axis_correction_carries_cad_receipts_forward() -> None:
    """The actual 2026-08-17 amendment: one leaf, none of it consumed."""

    report = _evaluate(_freeze(axis_z=1), _amended(axis_z=-1))
    assert report["status"] == CARRIES_FORWARD
    assert report["changed_freeze_paths"] == ["articulation_graph.joints[0].axis[2]"]
    assert report["colliding_freeze_paths"] == []
    assert report["task_semantics_changed"] is False
    assert report["spend_incurred_usd"] == 0.0


def test_changing_a_consumed_field_forces_rederivation() -> None:
    """The replacement asset id IS read by the CAD agent, so it cannot carry."""

    amended = _amended(axis_z=-1)
    amended["removal_plan"]["replacement_asset_id"] = "washer_replacement_v2"
    amended["task_freeze_digest"] = ""
    amended["task_freeze_digest"] = canonical_digest(
        amended, digest_field="task_freeze_digest"
    )
    report = _evaluate(_freeze(axis_z=1), amended)
    assert report["status"] == REQUIRES_REDERIVATION
    assert report["colliding_freeze_paths"] == ["removal_plan.replacement_asset_id"]
    assert report["task_semantics_changed"] is True


def test_a_sibling_of_a_consumed_field_does_not_collide() -> None:
    """`removal_plan.source_collider_prim_path` is not the consumed leaf."""

    amended = _amended(axis_z=1)
    amended["removal_plan"]["source_collider_prim_path"] = "/Root/source_999"
    amended["task_freeze_digest"] = ""
    amended["task_freeze_digest"] = canonical_digest(
        amended, digest_field="task_freeze_digest"
    )
    report = _evaluate(_freeze(axis_z=1), amended)
    assert report["status"] == CARRIES_FORWARD
    assert report["colliding_freeze_paths"] == []


def test_unknown_sealed_schema_carries_nothing() -> None:
    """Silence about what a schema reads is never read as 'reads nothing'."""

    report = _evaluate(_freeze(), _amended(axis_z=-1), schema="some_other_receipt.v1")
    assert report["status"] == REQUIRES_REDERIVATION
    assert report["reason"] == "sealed_schema_declares_no_consumed_freeze_fields"


def test_an_identical_freeze_cannot_manufacture_a_proof() -> None:
    """No amendment means there is nothing to rule on."""

    with pytest.raises(FreezeAmendmentCarryForwardError) as excinfo:
        _evaluate(_freeze(), _freeze())
    assert "freeze_amendment_absent" in excinfo.value.errors


def test_a_proof_is_bound_to_one_amendment_and_one_schema() -> None:
    """Otherwise one cheap ruling would launder every sealed receipt."""

    proof = _evaluate(_freeze(axis_z=1), _amended(axis_z=-1))
    assert (
        validate_freeze_amendment_carry_forward(
            proof,
            sealed_schema=CAD_REQUEST,
            superseded_file_sha256=SUPERSEDED_FILE_SHA,
            amended_file_sha256=AMENDED_FILE_SHA,
        )
        == proof
    )

    for kwargs, expected in (
        (
            {"sealed_schema": "simready_cad_agent_output.v1"},
            "freeze_carry_forward_schema_mismatch",
        ),
        (
            {"superseded_file_sha256": "sha256:" + "c" * 64},
            "freeze_carry_forward_superseded_mismatch",
        ),
        (
            {"amended_file_sha256": "sha256:" + "d" * 64},
            "freeze_carry_forward_amended_mismatch",
        ),
    ):
        arguments = {
            "sealed_schema": CAD_REQUEST,
            "superseded_file_sha256": SUPERSEDED_FILE_SHA,
            "amended_file_sha256": AMENDED_FILE_SHA,
            **kwargs,
        }
        with pytest.raises(FreezeAmendmentCarryForwardError) as excinfo:
            validate_freeze_amendment_carry_forward(proof, **arguments)
        assert expected in excinfo.value.errors


def test_a_rederivation_verdict_is_not_a_usable_proof() -> None:
    amended = _amended(axis_z=-1)
    amended["removal_plan"]["replacement_asset_id"] = "other"
    amended["task_freeze_digest"] = ""
    amended["task_freeze_digest"] = canonical_digest(
        amended, digest_field="task_freeze_digest"
    )
    verdict = _evaluate(_freeze(axis_z=1), amended)
    with pytest.raises(FreezeAmendmentCarryForwardError) as excinfo:
        validate_freeze_amendment_carry_forward(
            verdict,
            sealed_schema=CAD_REQUEST,
            superseded_file_sha256=SUPERSEDED_FILE_SHA,
            amended_file_sha256=AMENDED_FILE_SHA,
        )
    assert "freeze_carry_forward_status_invalid" in excinfo.value.errors


def test_a_tampered_proof_is_refused() -> None:
    proof = _evaluate(_freeze(axis_z=1), _amended(axis_z=-1))
    proof["colliding_freeze_paths"] = ["removal_plan.replacement_asset_id"]
    with pytest.raises(FreezeAmendmentCarryForwardError) as excinfo:
        validate_freeze_amendment_carry_forward(
            proof,
            sealed_schema=CAD_REQUEST,
            superseded_file_sha256=SUPERSEDED_FILE_SHA,
            amended_file_sha256=AMENDED_FILE_SHA,
        )
    assert "freeze_carry_forward_digest_invalid" in excinfo.value.errors


def test_cli_exits_nonzero_when_any_schema_needs_rederivation(tmp_path: Path) -> None:
    """A mixed verdict must never read as a green light."""

    superseded = tmp_path / "old.json"
    superseded.write_text(json.dumps(_freeze(axis_z=1)), encoding="utf-8")
    amended = tmp_path / "new.json"
    amended.write_text(json.dumps(_amended(axis_z=-1)), encoding="utf-8")

    argv = [
        "--superseded-freeze",
        str(superseded),
        "--amended-freeze",
        str(amended),
        "--sealed-schema",
        CAD_REQUEST,
        "--output-dir",
        str(tmp_path / "out"),
    ]
    assert main(argv) == 0
    written = json.loads(
        (tmp_path / "out" / "carry_forward_simready_cad_agent_request_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert written["status"] == CARRIES_FORWARD
    # The proof must name the real file bytes, not a placeholder.
    assert written["superseded_freeze_file_sha256"].startswith("sha256:")
    assert (
        written["superseded_freeze_file_sha256"]
        != written["amended_freeze_file_sha256"]
    )

    assert main([*argv, "--sealed-schema", "unknown_receipt.v1"]) == 3
