from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.articulated_native_probe import LOCKED_HINGE_RIGID_MODE
from blueprint_pipeline.articulated_isaac_bundle import build_articulated_isaac_bundle
from blueprint_pipeline.paired_native_simready_transition import (
    PairedNativeSimReadyTransitionError,
    bind_paired_native_simready_predecessor,
    materialize_paired_native_simready_probe,
)


ROOT = Path(__file__).resolve().parents[1]
CLI_SCRIPTS = (
    ROOT / "scripts/materialize_paired_native_simready_probe.py",
    ROOT / "scripts/build_scene_bound_articulated_simready_bundle.py",
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict, *, digest_field: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(value)
    if digest_field is not None:
        payload[digest_field] = canonical_digest(payload, digest_field=digest_field)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _predecessor(
    tmp_path: Path,
    *,
    scene_id: str = "840920",
    candidate_source: Path | None = None,
    task_id: str = "task_a_washer_door_open",
    asset_id: str | None = None,
) -> dict[str, Path | str]:
    candidate = tmp_path / "replacement.usda"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    if candidate_source is None:
        candidate.write_text("#usda 1.0\ndef Xform \"Asset\" {}\n", encoding="utf-8")
    else:
        candidate.write_bytes(candidate_source.read_bytes())
    candidate_sha = _sha256(candidate)
    asset_id = asset_id or f"{scene_id}_simready_washer_candidate"
    replacement = {
        "index": 0,
        "task_id": task_id,
        "asset_id": asset_id,
        "relative_path": "assets/replacement.usda",
        "size_bytes": candidate.stat().st_size,
        "sha256": candidate_sha,
        "asset_frame_registration_digest": "sha256:" + "1" * 64,
        "registered_static_qualification_digest": "sha256:" + "2" * 64,
    }
    request = _write(
        tmp_path / "paired_target_native_import_request.v1.json",
        {
            "schema_version": "paired_target_native_import_request.v1",
            "status": "frozen_pending_native_isaac_import",
            "scene_id": scene_id,
            "source_native_render_request": {
                "receipt_digest": "sha256:" + "3" * 64
            },
            "replacement_count": 1,
            "replacements": [replacement],
            "request_digest": "",
        },
        digest_field="request_digest",
    )
    request_value = json.loads(request.read_text())
    receipt = _write(
        tmp_path / "paired_target_native_import_bundle_receipt.v1.json",
        {
            "schema_version": "paired_target_native_import_provider_bundle.v1",
            "status": "ready",
            "request_digest": request_value["request_digest"],
            "source_request_digest": "sha256:" + "3" * 64,
            "bundle_sha256": "sha256:" + "4" * 64,
            "replacements": [replacement],
            "input_files": [
                {
                    "relative_path": "paired_target_native_import_request.v1.json",
                    "size_bytes": request.stat().st_size,
                    "sha256": _sha256(request),
                }
            ],
            "receipt_digest": "",
        },
        digest_field="receipt_digest",
    )
    probe = _write(
        tmp_path / "execution/probes/candidate.json",
        {
            "schema_version": "simready_replacement_native_import_probe_result.v1",
            "status": "completed",
            "blockers": [],
            "asset_id": asset_id,
            "native_isaac_executed": True,
            "native_simulator_import_qualified": True,
            "physical_equivalence_claimed": False,
            "replacement_asset_sha256": candidate_sha,
            "native_readback": {
                "asset_frame_registration_digest": "sha256:" + "1" * 64
            },
            "registered_static_qualification_digest": "sha256:" + "2" * 64,
            "result_digest": "",
        },
        digest_field="result_digest",
    )
    probe_value = json.loads(probe.read_text())
    runtime = _write(
        tmp_path / "execution/paired_target_native_import_runtime_result.v1.json",
        {
            "schema_version": "paired_target_native_import_runtime_result.v1",
            "status": "completed",
            "blockers": [],
            "scene_id": scene_id,
            "request_digest": request_value["request_digest"],
            "all_replacements_import_qualified": True,
            "replacement_count": 1,
            "replacements": [
                {
                    "task_id": task_id,
                    "asset_id": asset_id,
                    "blockers": [],
                    "native_simulator_import_qualified": True,
                    "probe_result_path": "probes/candidate.json",
                    "probe_result_sha256": _sha256(probe),
                    "probe_result_digest": probe_value["result_digest"],
                }
            ],
            "result_digest": "",
        },
        digest_field="result_digest",
    )
    terminal = _write(
        tmp_path / "paired_target_native_import_vast_result.v1.json",
        {
            "schema_version": "paired_target_native_import_vast_run.v1",
            "status": "completed",
            "blockers": [],
            "bundle_sha256": "sha256:" + "4" * 64,
            "request_digest": request_value["request_digest"],
            "replacement_count": 1,
            "native_result_path": str(runtime.resolve()),
            "continuing_spend_from_this_run": False,
            "teardown_manifest_path": "/retained/teardown.json",
            "artifact_manifest_path": "/retained/artifact.json",
        },
    )
    return {
        "scene_id": scene_id,
        "task_id": task_id,
        "asset_id": asset_id,
        "candidate": candidate,
        "receipt": receipt,
        "request": request,
        "terminal": terminal,
        "runtime": runtime,
        "probe": probe,
    }


def _notebook_candidate(path: Path) -> Path:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(asset.GetPrim())
    for name, center in (("base", (0.0, 0.0, 0.02)), ("display", (0.0, 0.1, 0.16))):
        link = UsdGeom.Xform.Define(stage, f"/Asset/links/{name}")
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        UsdPhysics.MassAPI.Apply(link.GetPrim()).CreateMassAttr().Set(1.0)
        cube = UsdGeom.Cube.Define(stage, f"/Asset/links/{name}/visual")
        cube.CreateSizeAttr(0.2)
        UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(*center))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/Asset/joints/display_hinge")
    hinge.CreateBody0Rel().SetTargets([Sdf.Path("/Asset/links/base")])
    hinge.CreateBody1Rel().SetTargets([Sdf.Path("/Asset/links/display")])
    hinge.CreateAxisAttr().Set("X")
    hinge.CreateLowerLimitAttr().Set(0.0)
    hinge.CreateUpperLimitAttr().Set(126.05071)
    drive = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(), "angular")
    drive.CreateTargetPositionAttr().Set(100.0)
    drive.CreateStiffnessAttr().Set(200.0)
    drive.CreateDampingAttr().Set(20.0)
    stage.GetRootLayer().Save()
    return path


def _bind(paths: dict[str, Path | str], **overrides):
    return bind_paired_native_simready_predecessor(
        scene_id=overrides.pop("scene_id", paths["scene_id"]),
        task_id=paths["task_id"],
        asset_id=paths["asset_id"],
        candidate_usd_path=paths["candidate"],
        paired_bundle_receipt_path=paths["receipt"],
        paired_request_path=paths["request"],
        paired_terminal_result_path=paths["terminal"],
        paired_runtime_result_path=paths["runtime"],
        paired_candidate_probe_path=overrides.pop(
            "paired_candidate_probe_path", paths["probe"]
        ),
        **overrides,
    )


@pytest.mark.parametrize("script", CLI_SCRIPTS, ids=lambda path: path.stem)
def test_no_spend_cli_is_invocable(script: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    assert completed.returncode == 0, completed.stderr
    assert "provider" not in completed.stderr.lower()


def test_transition_binds_scene_candidate_and_every_predecessor_byte(tmp_path: Path) -> None:
    paths = _predecessor(tmp_path)

    binding = _bind(paths)

    assert binding["scene_id"] == "840920"
    assert binding["candidate_usd_sha256"] == _sha256(paths["candidate"])
    assert binding["paired_request_digest"].startswith("sha256:")
    for role in (
        "bundle_receipt",
        "request",
        "terminal_result",
        "runtime_result",
        "candidate_probe",
    ):
        assert binding[role]["sha256"] == _sha256(Path(binding[role]["path"]))
    assert binding["binding_digest"] == canonical_digest(
        binding, digest_field="binding_digest"
    )


def test_transition_refuses_a_terminal_result_from_another_scene(tmp_path: Path) -> None:
    paths = _predecessor(tmp_path, scene_id="840313")

    with pytest.raises(PairedNativeSimReadyTransitionError) as excinfo:
        _bind(paths, scene_id="840920")

    assert "request_binding_invalid" in str(excinfo.value)
    assert "result_binding_invalid" in str(excinfo.value)


def test_transition_refuses_candidate_bytes_not_in_the_terminal_probe(tmp_path: Path) -> None:
    paths = _predecessor(tmp_path)
    Path(paths["candidate"]).write_bytes(Path(paths["candidate"]).read_bytes() + b"# changed\n")

    with pytest.raises(PairedNativeSimReadyTransitionError) as excinfo:
        _bind(paths)

    assert "candidate_digest_mismatch" in str(excinfo.value)


def test_transition_refuses_a_swapped_candidate_probe(tmp_path: Path) -> None:
    paths = _predecessor(tmp_path)
    other = _predecessor(tmp_path / "other")

    with pytest.raises(PairedNativeSimReadyTransitionError) as excinfo:
        _bind(paths, paired_candidate_probe_path=other["probe"])

    assert "result_probe_mismatch" in str(excinfo.value)


def test_materialized_articulated_probe_carries_the_predecessor_binding(
    tmp_path: Path,
) -> None:
    from tests.test_articulated_native_probe import _candidate

    candidate = _candidate(tmp_path / "candidate.usda")
    paths = _predecessor(
        tmp_path / "predecessor", candidate_source=candidate
    )

    receipt = materialize_paired_native_simready_probe(
        scene_id="840920",
        task_id="task_a_washer_door_open",
        asset_id="840920_simready_washer_candidate",
        candidate_usd_path=paths["candidate"],
        paired_bundle_receipt_path=paths["receipt"],
        paired_request_path=paths["request"],
        paired_terminal_result_path=paths["terminal"],
        paired_runtime_result_path=paths["runtime"],
        paired_candidate_probe_path=paths["probe"],
        destination=tmp_path / "probe-root",
        task_joint_prim_path="/Asset/joints/upper_door_hinge",
        locked_joint_prim_paths=["/Asset/joints/lower_door_hinge"],
        commanded_sweep_degrees=[0.0, 25.0, 55.0],
        reset_joint_positions_rad={
            "/Asset/joints/upper_door_hinge": 0.0,
            "/Asset/joints/lower_door_hinge": 0.0,
        },
        locked_joint_motion_tolerance_rad=0.001,
        settle_samples=40,
        control_frequency_hz=15.0,
    )

    assert receipt["scene_id"] == "840920"
    assert receipt["paired_native_predecessor"]["binding_digest"].startswith(
        "sha256:"
    )


def test_task_b_locked_hinge_builds_without_any_joint_command(tmp_path: Path) -> None:
    import importlib.util

    candidate = _notebook_candidate(tmp_path / "notebook.usda")
    paths = _predecessor(
        tmp_path / "predecessor",
        candidate_source=candidate,
        task_id="task_b_notebook_relocation",
        asset_id="840920_simready_notebook_candidate",
    )
    probe = materialize_paired_native_simready_probe(
        scene_id="840920",
        task_id="task_b_notebook_relocation",
        asset_id="840920_simready_notebook_candidate",
        candidate_usd_path=paths["candidate"],
        paired_bundle_receipt_path=paths["receipt"],
        paired_request_path=paths["request"],
        paired_terminal_result_path=paths["terminal"],
        paired_runtime_result_path=paths["runtime"],
        paired_candidate_probe_path=paths["probe"],
        destination=tmp_path / "probe-root",
        task_joint_prim_path="",
        locked_joint_prim_paths=["/Asset/joints/display_hinge"],
        commanded_sweep_degrees=[],
        reset_joint_positions_rad={
            "/Asset/joints/display_hinge": 1.745329252
        },
        locked_joint_motion_tolerance_rad=0.001,
        settle_samples=40,
        control_frequency_hz=15.0,
        validation_mode=LOCKED_HINGE_RIGID_MODE,
    )

    assert probe["validation_mode"] == LOCKED_HINGE_RIGID_MODE
    assert probe["expected"]["task_joint_prim_path"] is None
    assert probe["expected"]["commanded_sweep_degrees"] == []
    assert probe["probe_drive"] is None
    assert "no_joint_command_issued" in probe["required_readbacks"]
    assert "commanded_sweep_reaches_maximum" not in probe["required_readbacks"]
    assert probe["claim_boundary"]["task_joint_commanded"] is False
    assert probe["claim_boundary"]["locked_hinge_only"] is True

    worker_path = ROOT / "scripts/run_adp009d_articulated_isaac_worker.py"
    spec = importlib.util.spec_from_file_location("locked_hinge_worker", worker_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module._command_schedule(probe) == []

    bundle = build_articulated_isaac_bundle(
        probe_root=tmp_path / "probe-root",
        job_dir=tmp_path / "bundle",
        worker_source=worker_path,
        source_commit_sha="a" * 40,
    )
    assert bundle["scene_id"] == "840920"
    assert bundle["candidate_usd_sha256"] == _sha256(paths["candidate"])
    assert bundle["probe_names"] == sorted(probe["required_readbacks"])


@pytest.mark.parametrize(
    "task_joint,sweep,drive",
    [
        ("/Asset/joints/display_hinge", [], 0.0),
        ("", [0.0, 10.0], 0.0),
        ("", [], 100.0),
    ],
)
def test_task_b_locked_hinge_refuses_actuation(
    tmp_path: Path, task_joint: str, sweep: list[float], drive: float
) -> None:
    from blueprint_pipeline.articulated_native_probe import (
        ArticulatedNativeProbeError,
        materialize_articulated_native_probe,
    )

    candidate = _notebook_candidate(tmp_path / "notebook.usda")
    predecessor = {
        "schema_version": "paired_native_simready_predecessor_binding.v1",
        "scene_id": "840920",
        "candidate_usd_sha256": _sha256(candidate),
        "binding_digest": "",
    }
    predecessor["binding_digest"] = canonical_digest(
        predecessor, digest_field="binding_digest"
    )

    with pytest.raises(ArticulatedNativeProbeError) as excinfo:
        materialize_articulated_native_probe(
            candidate_usd_path=candidate,
            destination=tmp_path / "probe",
            task_joint_prim_path=task_joint,
            locked_joint_prim_paths=["/Asset/joints/display_hinge"],
            commanded_sweep_degrees=sweep,
            reset_joint_positions_rad={
                "/Asset/joints/display_hinge": 1.745329252
            },
            locked_joint_motion_tolerance_rad=0.001,
            settle_samples=40,
            control_frequency_hz=15.0,
            probe_drive_stiffness=drive,
            validation_mode=LOCKED_HINGE_RIGID_MODE,
            scene_id="840920",
            paired_native_predecessor=predecessor,
        )

    assert "locked_mode" in str(excinfo.value)
