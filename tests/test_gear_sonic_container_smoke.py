from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from blueprint_pipeline import gear_sonic_container_smoke as smoke
from blueprint_pipeline import gear_sonic_joint_order_contract as contract

FIXTURE_MODEL = (
    Path(__file__).parent / "fixtures" / "gear_sonic_g1_min" / "g1_29dof_with_hand_min.xml"
)


def _install_tree(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "wbc"
    model = root / "gear_sonic_deploy" / "g1" / "g1_29dof_with_hand.xml"
    model.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURE_MODEL, model)
    return root, model


def test_container_smoke_fail_closes_when_wbc_tree_absent(tmp_path) -> None:
    with pytest.raises(
        RuntimeError, match="official_gear_sonic_container_smoke_wbc_tree_missing"
    ):
        smoke.run_container_smoke(root=tmp_path / "wbc")


def test_container_smoke_fail_closes_when_model_absent(tmp_path) -> None:
    root = tmp_path / "wbc"
    (root / "gear_sonic_deploy").mkdir(parents=True)
    with pytest.raises(
        RuntimeError, match="official_gear_sonic_container_smoke_robot_model_missing"
    ):
        smoke.run_container_smoke(root=root)


def test_container_smoke_hermetic_fixture_passes(tmp_path) -> None:
    pytest.importorskip("mujoco", reason="mujoco_not_installed_in_venv")
    root, model = _install_tree(tmp_path)
    report = smoke.run_container_smoke(root=root, model_path=model)
    assert report["schema_version"] == smoke.SMOKE_SCHEMA_VERSION
    assert report["status"] == "passed"
    assert report["joint_order_schema_version"] == contract.JOINT_ORDER_SCHEMA_VERSION
    assert report["mapping_digest"] == contract.PROTOCOL_V4_MAPPING_DIGEST
    assert report["robot_model_sha256"]
    evidence = report["perturbation_evidence"]
    assert evidence["left_perturbed_joint"] == "left_elbow_joint"
    assert evidence["right_perturbed_joint"] == "right_elbow_joint"
    assert evidence["fk_states_distinct"] is True


def test_container_smoke_hermetic_rejects_wrong_joint_set(tmp_path) -> None:
    pytest.importorskip("mujoco", reason="mujoco_not_installed_in_venv")
    root, model = _install_tree(tmp_path)
    model.write_text(
        model.read_text(encoding="utf-8").replace(
            "right_wrist_yaw_joint", "right_wrist_extra_joint"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mujoco_model_joint_names_unknown"):
        smoke.run_container_smoke(root=root, model_path=model)


@pytest.mark.slow
def test_container_smoke_against_pinned_wbc_tree_or_precise_blocker(monkeypatch) -> None:
    """Runs the real /opt/wbc smoke when present; otherwise the smoke must
    fail closed with the precise wbc-tree blocker rather than pass vacuously."""
    monkeypatch.delenv(smoke.ROOT_ENV, raising=False)
    monkeypatch.delenv(smoke.MODEL_ENV, raising=False)
    if Path(smoke.DEFAULT_ROOT).is_dir():
        report = smoke.run_container_smoke()
        assert report["status"] == "passed"
        assert report["mapping_digest"] == contract.PROTOCOL_V4_MAPPING_DIGEST
    else:
        with pytest.raises(
            RuntimeError, match="official_gear_sonic_container_smoke_wbc_tree_missing"
        ):
            smoke.run_container_smoke()
