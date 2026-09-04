from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from pxr import Usd, UsdGeom

from blueprint_pipeline import public_scene_host_input_intake as intake
from blueprint_pipeline import public_scene_source_preparation as preparation
from tests.test_public_scene_raw_input_intake import _request as raw_request
from tests.test_sage_collision_identity import _box, _mesh


def _installed_scene(root: Path, *, shared_collider: bool = False, external_reference: bool = False) -> Path:
    request_path = raw_request(root)
    request = json.loads(request_path.read_text())
    by_role = {row["role"]: Path(row["path"]) for row in request["files"]}
    subject = (0.05, 0.1, 0.3, 0.25, 0.5, 0.325)
    support = subject if shared_collider else (0, 0, 0, 0.4, 0.8, 0.3)
    by_role["semantic_metadata"].write_text(json.dumps([
        {"ins_id": "101", "label": "rigid subject", "bounding_box": _box(*subject)},
        {"ins_id": "202", "label": "cabinet", "bounding_box": _box(*support)},
    ]))
    by_role["scene_structure"].write_text(json.dumps({
        "rooms": [{"profile": [[-1, -1], [2, -1], [2, 2], [-1, 2]]}],
        "walls": [], "holes": [],
    }))
    stage = Usd.Stage.CreateNew(str(by_role["collision_usd"]))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    _mesh(stage, "/Root/Subject", subject[:3], subject[3:])
    if not shared_collider:
        _mesh(stage, "/Root/Support", support[:3], support[3:])
    stage.GetRootLayer().Save()
    stage = None
    if external_reference:
        by_role["collision_usd"].write_text('#usda 1.0\n( subLayers = [@../outside.usda@] )\n')
    for row in request["files"]:
        row["sha256"] = intake._sha256_file(Path(row["path"]))
    rights_path = Path(request["rights_receipts"][0]["path"])
    rights = json.loads(rights_path.read_text())
    rights["authorized_source_sha256"] = [row["sha256"] for row in request["files"]]
    rights_path.write_text(json.dumps(rights))
    request["rights_receipts"][0]["sha256"] = intake._sha256_file(rights_path)
    request_path.write_text(json.dumps(request))
    stream = io.BytesIO()
    intake.build_packet_archive(request_path, stream)
    stream.seek(0)
    receipt = intake.install_packet_archive(
        stream, destination_root=root / "installed",
        allowed_roots=(root,), service_account=None,
    )
    return Path(receipt["destination_root"]) / "public_scene_host_input_installation_receipt.v1.json"


def _objects(*, explicit_support: bool = False) -> list[dict]:
    result = [{"role": "movable_subject", "source_instance_id": "101"}]
    if explicit_support:
        result.append({"role": "source_support", "source_instance_id": "202"})
    result.append({"role": "supplemental_destination", "description": "matte document container"})
    return result


def _prepare(root: Path, receipt: Path, objects: list[dict]) -> dict:
    return preparation.materialize_public_scene_source_preparation(
        installation_receipt_path=receipt, task_objects=objects,
        expected_source_commit=intake._verified_checkout_head(), approved_roots=(root,),
        output_root=root / "prepared",
    )


@pytest.mark.parametrize("explicit_support", [False, True])
def test_scene_and_objects_derive_identity_and_frame_without_supplied_receipts(
    tmp_path: Path, explicit_support: bool,
) -> None:
    installed = _installed_scene(tmp_path)
    objects = _objects(explicit_support=explicit_support)
    result = _prepare(tmp_path, installed, objects)
    assert result["status"] == "source_context_prepared_pending_calibrated_views"
    assert result["shared_frame_receipt_digest"].startswith("sha256:")
    assert {row["source_instance_id"] for row in result["source_identities"]} == {"101", "202"}
    assert result["task_objects"] == objects
    assert result["claim_boundary"]["method_input"] is False
    assert result["claim_boundary"]["evaluation_authorized"] is False
    assert result["candidate_policy_queried"] is False
    assert result["paid_resource_used"] is False
    assert len(result["artifacts"]) == 4
    assert _prepare(tmp_path, installed, objects) == result


def test_shared_furniture_collider_is_retained_as_a_failure(tmp_path: Path) -> None:
    installed = _installed_scene(tmp_path, shared_collider=True)
    result = _prepare(tmp_path, installed, _objects(explicit_support=True))
    assert result["status"] == "blocked"
    assert result["blockers"] == ["source_preparation_source_colliders_not_distinct"]
    assert result["shared_frame_receipt_digest"] is None
    assert len(result["source_identities"]) == 2
    assert (tmp_path / "prepared/public_scene_source_preparation.v1.json").is_file()


def test_supplemental_object_cannot_acquire_fake_source_identity(tmp_path: Path) -> None:
    installed = _installed_scene(tmp_path)
    objects = _objects()
    objects[-1]["source_instance_id"] = "202"
    with pytest.raises(preparation.PublicSceneSourcePreparationError, match="fake_destination_source"):
        _prepare(tmp_path, installed, objects)


def test_cached_context_reopens_artifacts_and_refuses_changed_evidence(tmp_path: Path) -> None:
    installed = _installed_scene(tmp_path)
    result = _prepare(tmp_path, installed, _objects())
    artifact = tmp_path / "prepared" / result["artifacts"][0]["relative_path"]
    artifact.chmod(0o640)
    artifact.write_text("{}")
    with pytest.raises(preparation.PublicSceneSourcePreparationError, match="output_conflict"):
        _prepare(tmp_path, installed, _objects())


def test_changed_installed_source_is_rejected_before_measurements(tmp_path: Path) -> None:
    installed = _installed_scene(tmp_path)
    receipt = json.loads(installed.read_text())
    source = installed.parent / next(row["relative_path"] for row in receipt["files"]
                                    if row.get("role") == "collision_usd")
    source.chmod(0o640)
    source.write_text(source.read_text() + "\n# changed")
    with pytest.raises(preparation.PublicSceneSourcePreparationError, match="input_bytes_changed"):
        _prepare(tmp_path, installed, _objects())


def test_production_intake_command_routes_to_real_source_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    installed = _installed_scene(tmp_path)
    objects = tmp_path / "task-objects.json"
    objects.write_text(json.dumps(_objects()))
    real = preparation.materialize_public_scene_source_preparation

    def fixture_roots(**kwargs):
        return real(**kwargs, approved_roots=(tmp_path,))

    monkeypatch.setattr(preparation, "materialize_public_scene_source_preparation", fixture_roots)
    assert intake.main([
        "prepare", "--installation-receipt", str(installed),
        "--task-objects", str(objects), "--output-root", str(tmp_path / "prepared"),
    ]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == (
        "source_context_prepared_pending_calibrated_views"
    )


def test_unadmitted_usd_reference_is_rejected_before_stage_composition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = _installed_scene(tmp_path, external_reference=True)

    def forbidden_measurement(**_kwargs):
        raise AssertionError("composed an unadmitted source layer")

    monkeypatch.setattr(preparation, "inspect_sage_collision_identity", forbidden_measurement)
    with pytest.raises(preparation.PublicSceneSourcePreparationError, match="external_usd_dependency"):
        _prepare(tmp_path, installed, _objects(explicit_support=True))
