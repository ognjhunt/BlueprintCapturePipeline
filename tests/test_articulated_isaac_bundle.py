from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline.articulated_isaac_bundle import (
    ARTICULATED_ISAAC_BUNDLE_SCHEMA_VERSION,
    ArticulatedIsaacBundleError,
    build_articulated_isaac_bundle,
)


def _probe_root(tmp_path: Path) -> Path:
    """A frozen native probe as materialize_articulated_native_probe writes it."""

    from blueprint_pipeline.articulated_native_probe import (
        materialize_articulated_native_probe,
    )
    from tests.test_articulated_native_probe import _candidate

    root = tmp_path / "probe"
    materialize_articulated_native_probe(
        candidate_usd_path=_candidate(tmp_path / "candidate.usda"),
        destination=root,
        task_joint_prim_path="/Asset/joints/upper_door_hinge",
        locked_joint_prim_paths=["/Asset/joints/lower_door_hinge"],
        commanded_sweep_degrees=[0.0, 25.0, 45.0, 55.0],
        reset_joint_positions_rad={
            "/Asset/joints/upper_door_hinge": 0.0,
            "/Asset/joints/lower_door_hinge": 0.0,
        },
        locked_joint_motion_tolerance_rad=0.001,
        settle_samples=40,
        control_frequency_hz=15.0,
    )
    return root


def _build(tmp_path: Path, **overrides):
    # only freeze a fresh probe when the caller did not supply one, so a test
    # that tampers with the probe is not silently handed a regenerated copy
    arguments = {
        "probe_root": overrides.pop("probe_root", None) or _probe_root(tmp_path),
        "job_dir": tmp_path / "job",
        "worker_source": Path(__file__).resolve().parents[1]
        / "scripts/run_adp009d_articulated_isaac_worker.py",
        "source_commit_sha": "a" * 40,
        "generated_at": "2026-08-09T00:00:00+00:00",
    }
    arguments.update(overrides)
    return build_articulated_isaac_bundle(**arguments)


def test_bundle_ships_the_frozen_probe_and_the_articulated_worker(
    tmp_path: Path,
) -> None:
    receipt = _build(tmp_path)

    assert receipt["schema_version"] == ARTICULATED_ISAAC_BUNDLE_SCHEMA_VERSION
    assert receipt["status"] == "ready"
    assert receipt["retry_cap"] == 0
    assert receipt["blockers"] == []
    assert receipt["probe_spec_sha256"].startswith("sha256:")
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = set(archive.namelist())
    assert "provider_runtime/run_articulated_isaac_runtime.sh" in names
    assert "provider_runtime/articulated_isaac_worker.py" in names
    assert "provider_runtime/native/articulated_native_probe_spec.json" in names
    assert "provider_runtime/native/blank_physics_stage.usda" in names
    assert "provider_runtime/native/articulation_stage.usda" in names


def test_bundle_carries_no_rigid_stimulus_stages(tmp_path: Path) -> None:
    """The rigid lane's drop/slide/tip probes are meaningless for a door."""

    receipt = _build(tmp_path)

    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = "\n".join(archive.namelist())
    for stale in ("drop_stage", "isaac_slide_stage", "isaac_tip_stage", "gripper_stage"):
        assert stale not in names


def test_entrypoint_writes_a_typed_result_when_isaac_dies(tmp_path: Path) -> None:
    """A silent process exit must not read as 'no result, therefore fine'."""

    receipt = _build(tmp_path)

    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        entrypoint = archive.read(
            "provider_runtime/run_articulated_isaac_runtime.sh"
        ).decode("utf-8")
    assert "isaac_runner_process_exited_without_runtime_result" in entrypoint
    assert "articulated_isaac_worker.py" in entrypoint
    assert "/isaac-sim/python.sh" in entrypoint


def test_bundle_binds_the_exact_probe_and_candidate_digests(tmp_path: Path) -> None:
    root = _probe_root(tmp_path)
    spec = json.loads((root / "articulated_native_probe_spec.json").read_text())

    receipt = _build(tmp_path, probe_root=root)

    assert receipt["candidate_usd_sha256"] == spec["candidate_usd_sha256"]
    assert receipt["expected"]["assembly_joint_count"] == 2
    assert receipt["expected"]["maximum_commanded_degrees"] == 55.0
    assert receipt["required_readbacks"] == spec["required_readbacks"]


def test_a_probe_that_was_already_executed_is_refused(tmp_path: Path) -> None:
    root = _probe_root(tmp_path)
    spec_path = root / "articulated_native_probe_spec.json"
    spec = json.loads(spec_path.read_text())
    spec["status"] = "executed"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ArticulatedIsaacBundleError) as excinfo:
        _build(tmp_path, probe_root=root)

    assert any("probe_not_frozen" in error for error in excinfo.value.errors)


def test_a_tampered_probe_spec_is_refused(tmp_path: Path) -> None:
    root = _probe_root(tmp_path)
    stage = root / "blank_physics_stage.usda"
    stage.write_text(stage.read_text() + "\n# tampered\n", encoding="utf-8")

    with pytest.raises(ArticulatedIsaacBundleError) as excinfo:
        _build(tmp_path, probe_root=root)

    assert any("stage_digest_mismatch" in error for error in excinfo.value.errors)


def test_a_short_commit_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ArticulatedIsaacBundleError) as excinfo:
        _build(tmp_path, source_commit_sha="abc123")

    assert any("source_commit_invalid" in error for error in excinfo.value.errors)


def test_bundle_pins_the_same_image_the_allocator_checks(tmp_path: Path) -> None:
    """A separate image constant here would fail admission on every run."""

    from blueprint_pipeline.public_scene_simready_isaac_bundle import (
        DEFAULT_IMAGE as LANE_IMAGE,
    )

    receipt = _build(tmp_path)

    assert receipt["container_image"] == LANE_IMAGE
    assert "@sha256:" in receipt["container_image"]


def test_bundle_is_deterministic(tmp_path: Path) -> None:
    root = _probe_root(tmp_path)

    first = _build(tmp_path, probe_root=root, job_dir=tmp_path / "a")
    second = _build(tmp_path, probe_root=root, job_dir=tmp_path / "b")

    assert first["bundle_sha256"] == second["bundle_sha256"]
