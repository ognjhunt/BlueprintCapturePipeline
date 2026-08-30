from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_configured_controls_progression_worker as worker
from blueprint_pipeline.task_evaluation_configured_controls_plan import (
    TaskEvaluationConfiguredControlsPlanError,
    materialize_configured_controls_plan,
)


SOURCE_LAUNCH_ID = "scene-839873-qualifying"
SOURCE_COMMIT = "a" * 40
TARGET_COMMIT = "b" * 40
CONFIGURATION_RUN_ID = "scene-839873-configuration"


def _write(path: Path, value: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o440)
    return path.resolve()


def _bindings(tmp_path: Path) -> dict[str, object]:
    inputs = tmp_path / "inputs"
    top = {
        name: str(
            _write(
                inputs / f"{name}.json",
                {
                    "schema_version": f"test.{name}.v1",
                    "source_commit": (
                        SOURCE_COMMIT
                        if name
                        in {
                            "robot_mount_interface_path",
                            "scene_camera_calibration_path",
                        }
                        else TARGET_COMMIT
                    ),
                },
            )
        )
        for name in (
            "robot_mount_interface_path",
            "scene_camera_calibration_path",
            "base_pose_candidate_path",
            "cameras_path",
            "runtime_binding_path",
        )
    }
    phases: dict[str, dict[str, object]] = {}
    for phase in ("construction", "controls"):
        row: dict[str, object] = {
            name: str(
                _write(
                    inputs / phase / f"{name}.json",
                    {
                        "schema_version": f"test.{phase}.{name}.v1",
                        "expected_production_commit": TARGET_COMMIT,
                    },
                )
            )
            for name in (
                "release_window_template_path",
                "authorization_path",
                "launch_authority_path",
            )
        }
        if phase == "construction":
            row["lineage_path"] = str(
                _write(
                    inputs / phase / "lineage_path.json",
                    {
                        "schema_version": "test.construction.lineage_path.v1",
                        "source_commit": TARGET_COMMIT,
                    },
                )
            )
        phases[phase] = row
    return {**top, "phases": phases}


def _qualifying_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        worker,
        "_validate_source",
        lambda _root: (
            {"run_id": CONFIGURATION_RUN_ID},
            {
                "launch_id": SOURCE_LAUNCH_ID,
                "source_commit": SOURCE_COMMIT,
                "receipt_digest": "sha256:" + "1" * 64,
            },
            {"provider_zero_verified": True},
        ),
    )


def test_materializes_exact_present_inputs_and_binds_future_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _qualifying_source(monkeypatch)
    bindings = _bindings(tmp_path)
    kwargs = {
        "source_launch_id": SOURCE_LAUNCH_ID,
        "launch_state_root": tmp_path / "launch-runs",
        "expected_production_commit": TARGET_COMMIT,
        "submitted_by": "configured-controls-materializer",
        "bindings": bindings,
        "plan_root": tmp_path / "plans",
        "profile_dir": tmp_path / "profiles",
    }
    (tmp_path / "profiles").mkdir()
    first = materialize_configured_controls_plan(**kwargs)
    second = materialize_configured_controls_plan(**kwargs)

    assert first["status"] == "materialized"
    assert second["status"] == "replayed"
    plan_path = Path(first["plan_path"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert stat.S_IMODE(plan_path.stat().st_mode) == 0o440
    assert plan["source_launch_receipt_digest"] == "sha256:" + "1" * 64
    assert plan["source_configuration_commit"] == SOURCE_COMMIT
    assert plan["expected_production_commit"] == TARGET_COMMIT
    assert plan["future_outputs"]["construction"]["expected_activation_id"].endswith(
        "-episode-construction"
    )
    assert plan["profile_dir"] == str(tmp_path / "profiles")
    assert "lineage_path" not in plan["phases"]["controls"]
    assert "lineage_artifact_paths" not in plan["phases"]["controls"]
    assert plan["provider_mutation_performed"] is False


def test_plan_destination_is_scoped_to_the_expected_production_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A redeploy authors a plan whose bytes differ only by commit. Keying the
    file on the launch id alone turns that into a permanent
    configured_controls_plan_immutable_conflict, so the filename must carry the
    commit and give each production commit its own immutable plan."""

    _qualifying_source(monkeypatch)
    kwargs = {
        "source_launch_id": SOURCE_LAUNCH_ID,
        "launch_state_root": tmp_path / "launch-runs",
        "expected_production_commit": TARGET_COMMIT,
        "submitted_by": "configured-controls-materializer",
        "bindings": _bindings(tmp_path),
        "plan_root": tmp_path / "plans",
        "profile_dir": tmp_path / "profiles",
    }
    (tmp_path / "profiles").mkdir()
    first = materialize_configured_controls_plan(**kwargs)

    assert Path(first["plan_path"]).name == (
        f"{SOURCE_LAUNCH_ID}-{TARGET_COMMIT[:12]}.json"
    )
    # Idempotency within one commit is unchanged.
    assert materialize_configured_controls_plan(**kwargs)["status"] == "replayed"
    # A successor commit therefore claims a distinct destination rather than
    # colliding with the plan already sealed for its predecessor.
    successor = "c" * 40
    assert (
        f"{SOURCE_LAUNCH_ID}-{successor[:12]}.json"
        != Path(first["plan_path"]).name
    )


def test_nonqualifying_source_refuses_before_plan_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        worker,
        "_validate_source",
        lambda _root: (_ for _ in ()).throw(
            worker.TaskEvaluationConfiguredControlsProgressionWorkerError(
                "configured_controls_worker_qualifying_terminal_missing"
            )
        ),
    )
    with pytest.raises(
        worker.TaskEvaluationConfiguredControlsProgressionWorkerError,
        match="configured_controls_worker_qualifying_terminal_missing",
    ):
        materialize_configured_controls_plan(
            source_launch_id=SOURCE_LAUNCH_ID,
            launch_state_root=tmp_path / "launch-runs",
            expected_production_commit=TARGET_COMMIT,
            submitted_by="configured-controls-materializer",
            bindings=_bindings(tmp_path),
            plan_root=tmp_path / "plans",
            profile_dir=tmp_path / "profiles",
        )
    assert not (tmp_path / "plans").exists()


def test_worker_refuses_present_input_changed_after_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _qualifying_source(monkeypatch)
    bindings = _bindings(tmp_path)
    (tmp_path / "profiles").mkdir()
    result = materialize_configured_controls_plan(
        source_launch_id=SOURCE_LAUNCH_ID,
        launch_state_root=tmp_path / "launch-runs",
        expected_production_commit=TARGET_COMMIT,
        submitted_by="configured-controls-materializer",
        bindings=bindings,
        plan_root=tmp_path / "plans",
        profile_dir=tmp_path / "profiles",
    )
    changed = Path(str(bindings["runtime_binding_path"]))
    changed.chmod(0o640)
    changed.write_text('{"schema_version":"changed"}\n', encoding="utf-8")
    with pytest.raises(
        worker.TaskEvaluationConfiguredControlsProgressionWorkerError,
        match="configured_controls_worker_plan_inventory_invalid",
    ):
        worker._plan(Path(result["plan_path"]))


def test_profile_directory_must_exist_before_plan_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _qualifying_source(monkeypatch)
    with pytest.raises(
        TaskEvaluationConfiguredControlsPlanError,
        match="configured_controls_plan_profile_dir_invalid",
    ):
        materialize_configured_controls_plan(
            source_launch_id=SOURCE_LAUNCH_ID,
            launch_state_root=tmp_path / "launch-runs",
            expected_production_commit=TARGET_COMMIT,
            submitted_by="configured-controls-materializer",
            bindings=_bindings(tmp_path),
            plan_root=tmp_path / "plans",
            profile_dir=tmp_path / "missing-profiles",
        )


def test_configuration_artifact_cannot_claim_the_target_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _qualifying_source(monkeypatch)
    bindings = _bindings(tmp_path)
    mount = Path(str(bindings["robot_mount_interface_path"]))
    mount.chmod(0o640)
    mount.write_text(
        json.dumps(
            {
                "schema_version": "test.robot_mount_interface_path.v1",
                "source_commit": TARGET_COMMIT,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "profiles").mkdir()

    with pytest.raises(
        TaskEvaluationConfiguredControlsPlanError,
        match="configured_controls_plan_artifact_commit_mismatch",
    ):
        materialize_configured_controls_plan(
            source_launch_id=SOURCE_LAUNCH_ID,
            launch_state_root=tmp_path / "launch-runs",
            expected_production_commit=TARGET_COMMIT,
            submitted_by="configured-controls-materializer",
            bindings=bindings,
            plan_root=tmp_path / "plans",
            profile_dir=tmp_path / "profiles",
        )
