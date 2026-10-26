"""Retained SDK bytes use real static checks, never a fabricated CLI terminal."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_object_artisan_candidate import (
    package_validated_sdk_candidate, qualify_artisan_asset_static, verify_artisan_candidate,
)
from blueprint_pipeline.task_object_artisan_exports import validated_sdk_exports
from blueprint_pipeline.task_object_authoring_contracts import AuthoringContractError, file_record
from .test_task_object_artisan_exports import annotations, construction_request, export_fixture, usd_fixture
from .test_task_evaluation_scene_configuration_artisan_driver import stage_fixture  # noqa: F401


def sdk_fixture(root, *, articulated=False):
    terminal, manifest = export_fixture(root)
    source = usd_fixture(root / "geometry", articulated=articulated)
    usd = root / "exports/object.usd"
    usd.write_bytes(source.read_bytes())
    record = file_record(usd)
    manifest["artifacts"]["object.usd"].update(sha256=record["sha256"][7:], size_bytes=record["size_bytes"])
    fields = {"build_id": "fixture-build", "validation_algorithm_version": "fixture-physics",
              "validation_policy_sha256": "a" * 64, "simulator_collision_digest": "b" * 64,
              "accepted_fcl_pose_sha256": "c" * 64}
    manifest.update(fields)
    report = {"candidate_id": manifest["candidate_id"], "model_sha256": manifest["model_sha256"],
              "build_id": fields["build_id"], "status": "pass", "blocking_findings": [],
              "algorithm_version": fields["validation_algorithm_version"],
              **{key: fields[key] for key in ("validation_policy_sha256", "simulator_collision_digest",
                                             "accepted_fcl_pose_sha256")}}
    physics = root / "exports/physics.json"
    physics.write_text(json.dumps(report))
    manifest["physics_report_sha256"] = file_record(physics)["sha256"][7:]
    path = root / "exports/export_manifest.json"
    path.write_text(json.dumps(manifest))
    return {"manifest_path": path, "model_path": root / "model.py", "blueprint_path": root / "blueprint.json"}


def request_fixture(root, *, articulated=False):
    value = construction_request(articulated=articulated)
    source = root / "retained-source.glb"
    source.write_bytes(b"synthetic retained source geometry fixture")
    record = file_record(source)
    value.update(replacement_identity={"id": "fixture-assembly", "version": "v1"}, source_frames=[],
                 source_candidate={**record, "digest": record["sha256"]})
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    path = root / "request.json"
    path.write_text(json.dumps(value))
    return path, value


@pytest.mark.parametrize("articulated", [False, True])
def test_real_usd_sdk_candidate_enters_existing_static_boundary_without_registry_claim(tmp_path, articulated):
    kwargs = sdk_fixture(tmp_path, articulated=articulated)
    request_path, request = request_fixture(tmp_path, articulated=articulated)
    params = tmp_path / "parameters.json"
    params.write_text('{"fixture":true}')
    packaged = package_validated_sdk_candidate(**kwargs, request_path=request_path,
        output_root=tmp_path / "stage3/candidates/0000", sdk_source_commit="d" * 40,
        controller="independent SDK controller fixture", program_dependencies=[params],
        annotations_path=annotations(tmp_path, request) if articulated else None)
    receipt = json.loads(Path(packaged["receipt"]["path"]).read_text())
    graph = json.loads(Path(packaged["graph"]["path"]).read_text())
    assert "upstream_terminal" not in receipt
    assert receipt["upstream_runtime"]["packaging_provider_calls"] == 0
    static = qualify_artisan_asset_static(asset_path=packaged["asset"]["path"], graph_spec=graph,
        authoring_receipt=receipt, replacement_identity=request["replacement_identity"], output_path=tmp_path / "static.json")
    assert static["authored_structure_statically_qualified"] is True
    assert static["authoring_execution"] == "retained_validated_sdk_export"
    assert static["claim_boundary"]["native_simulator_import_qualified"] is False
    if articulated:
        assert static["task_contact"]["contact_link_id"] == static["task_joint"]["child_link_id"]
        bounds = static["task_contact"]["handle_bounds_link_frame_m"]
        assert [bounds["maximum"][i] - bounds["minimum"][i] for i in range(3)] == pytest.approx([.05, .3, .05])
    params.write_text('{"changed":true}')
    with pytest.raises(AuthoringContractError, match="source_bytes_changed"):
        verify_artisan_candidate(receipt=receipt, graph=graph, asset=Path(packaged["asset"]["path"]))


@pytest.mark.parametrize("change", ["rejected", "blocked_physics", "wrong_collision", "changed_export", "escape"])
def test_sdk_adoption_refuses_invalid_or_changed_validation_before_packaging(tmp_path, change):
    kwargs = sdk_fixture(tmp_path)
    manifest = json.loads(kwargs["manifest_path"].read_text())
    physics = tmp_path / "exports/physics.json"
    report = json.loads(physics.read_text())
    if change == "rejected":
        manifest["status"] = "rejected"
    elif change == "blocked_physics":
        report["blocking_findings"] = ["collision"]
    elif change == "wrong_collision":
        report["simulator_collision_digest"] = "e" * 64
    elif change == "changed_export":
        (tmp_path / "exports/object.usd").write_text("changed bytes")
    else:
        manifest["artifacts"]["object.usd"]["path"] = "../model.py"
    physics.write_text(json.dumps(report))
    manifest["physics_report_sha256"] = file_record(physics)["sha256"][7:]
    kwargs["manifest_path"].write_text(json.dumps(manifest))
    with pytest.raises(AuthoringContractError):
        validated_sdk_exports(**kwargs)


def test_sdk_receipt_cannot_be_relabelled_as_cli_done(tmp_path):
    kwargs = sdk_fixture(tmp_path)
    request_path, _ = request_fixture(tmp_path)
    packaged = package_validated_sdk_candidate(**kwargs, request_path=request_path,
        output_root=tmp_path / "candidate", sdk_source_commit="d" * 40, controller="fixture")
    receipt = json.loads(Path(packaged["receipt"]["path"]).read_text())
    graph = json.loads(Path(packaged["graph"]["path"]).read_text())
    changed = deepcopy(receipt)
    changed["upstream_terminal"] = {"phase": "DONE", "registered": True}
    changed["result_digest"] = canonical_digest(changed, digest_field="result_digest")
    with pytest.raises(AuthoringContractError, match="controller_identity_required"):
        verify_artisan_candidate(receipt=changed, graph=graph, asset=Path(packaged["asset"]["path"]))


def test_production_stage_adopts_sdk_candidate_before_credentials_runtime_or_cost_gate(request, tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_configuration_artisan_driver as driver
    from blueprint_pipeline.task_object_authoring_contracts import build_construction_request

    fixture = request.getfixturevalue("stage_fixture")
    stage = json.loads(Path(fixture.environment[driver.INPUT]).read_text())
    deps = json.loads(Path(fixture.environment[driver.DEPENDENCIES]).read_text())
    request = build_construction_request(stage, deps, fixture.frames, {"role": "synthetic source handoff fixture"})
    request_path = tmp_path / "sdk-request.json"
    request_path.write_text(json.dumps(request))
    source = tmp_path / "sdk"
    source.mkdir()
    package_validated_sdk_candidate(**sdk_fixture(source), request_path=request_path,
        output_root=fixture.output / "candidates/0000", sdk_source_commit="d" * 40, controller="fixture")

    def forbidden(*_args, **_kwargs):
        pytest.fail("SDK candidate adoption must not generate, load credentials, or allocate spend")

    monkeypatch.setattr(driver, "validate_artisan_runtime", forbidden)
    monkeypatch.setattr(driver, "scene_configuration_openai_stage_scope", forbidden)
    result = driver.execute_artisan_component(environment=fixture.environment, runner=forbidden, cost_gate_factory=forbidden)
    assert result["status"] == "completed"
    assert not fixture.calls and not fixture.events
