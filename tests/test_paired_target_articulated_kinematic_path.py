from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.paired_target_articulated_kinematic_path import (
    PairedTargetArticulatedKinematicPathError,
    materialize_paired_target_articulated_kinematic_path,
)
from blueprint_pipeline.paired_target_interaction_affordance_candidate import (
    materialize_paired_target_interaction_affordance_candidate,
)
from tests.test_paired_target_interaction_affordance_candidate import (
    _freeze,
    _registered,
    _usd,
)


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    freeze = _freeze(tmp_path / "freeze.json", task_kind="articulated_interaction")
    usd = _usd(tmp_path / "asset.usda", articulated=True)
    registered = _registered(
        tmp_path / "registered.json", freeze, usd, task_id="task_a"
    )
    affordance = tmp_path / "affordance.json"
    materialize_paired_target_interaction_affordance_candidate(
        task_freeze_path=freeze,
        registered_asset_receipt_path=registered,
        robot_base_position_world_m=[0.0, -1.0, 0.0],
        output_path=affordance,
    )
    return freeze, affordance


def test_derives_reset_to_target_path_from_usd_joint_frames(tmp_path: Path) -> None:
    freeze, affordance = _inputs(tmp_path)

    result = materialize_paired_target_articulated_kinematic_path(
        task_freeze_path=freeze,
        interaction_affordance_path=affordance,
        output_path=tmp_path / "path.json",
        waypoint_count=5,
    )

    rows = result["joint_contact_path"]
    assert len(rows) == 5
    assert rows[0]["joint_positions"] == {"hinge": 0.0}
    assert rows[-1]["joint_positions"]["hinge"] == pytest.approx(0.6)
    assert len(rows[0]["contact_pose_asset_root"]["position_m"]) == 3
    assert rows[0]["contact_pose_asset_root"]["position_m"] != rows[-1][
        "contact_pose_asset_root"
    ]["position_m"]
    assert result["native_ik_or_contact_executed"] is False


def test_reset_fk_mismatch_fails_closed(tmp_path: Path) -> None:
    freeze, affordance = _inputs(tmp_path)
    value = json.loads(affordance.read_text())
    usd = Path(value["registered_usd"]["path"])
    body = usd.read_text()
    usd.write_text(body.replace("(0, 0, 0.3)", "(0, 0, 0.2)", 1))
    value["registered_usd"]["size_bytes"] = usd.stat().st_size
    import hashlib

    value["registered_usd"]["sha256"] = (
        "sha256:" + hashlib.sha256(usd.read_bytes()).hexdigest()
    )
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    affordance.write_text(json.dumps(value))

    with pytest.raises(
        PairedTargetArticulatedKinematicPathError,
        match="reset_fk_mismatch",
    ):
        materialize_paired_target_articulated_kinematic_path(
            task_freeze_path=freeze,
            interaction_affordance_path=affordance,
            output_path=tmp_path / "path.json",
        )
