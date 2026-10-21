"""Actual packet/compiler/bundle joins; no native construction is claimed."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from blueprint_pipeline.adp_task_scoring import seal_rigid_task_success_contract
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.exact_workcell_native_packet import (
    PREPARATION_FILENAME,
    SOURCE_PLAN_FILENAME,
    replay_native_packet_cell,
)
from blueprint_pipeline.exact_workcell_native_runtime import native_runtime_embodiment_digest
from blueprint_pipeline.exact_workcell_variation_matrix import (
    ExactWorkcellVariationError,
    compile_variation_matrix,
)
from blueprint_pipeline.native_task_arena_construction_bundle import (
    build_native_task_arena_construction_bundle,
)
from blueprint_pipeline.native_task_arena_construction_worker import _load_and_verify_manifest
from blueprint_pipeline.native_task_arena_packet import materialize_native_task_arena_packet
from blueprint_pipeline.native_task_construction_plan import (
    materialize_native_task_construction_phase_plan,
)
from blueprint_pipeline.native_task_arena_feedback_bootstrap_runtime import (
    verified_construction_phase_plan_path,
)
from tests.test_exact_workcell_variation_matrix import _dimension, _request as matrix_request
from tests.test_adp_task_scoring import _rigid_v2_spec
from tests.test_native_task_construction_plan import _rigid_fixture
from tests.test_native_task_arena_bundle import _runtime_source_packet
from tests.test_native_task_arena_packet import _request as packet_request


def _read(path):
    return json.loads(path.read_text())


def _source(root, *, site="site-a", y=1.3):
    evidence = root / "evidence"
    evidence.mkdir()
    request = packet_request(evidence, articulated=False)
    request.update(scene_id=site, task_id=site + ".move")
    subject = next(row for row in request["assets"] if row["semantic_role"] == "task_object")
    subject["pose_world"]["position_world_m"][1] = y
    subject.update(
        asset_id="fixture_part",
        object_type="RIGID",
        reset_state={
            "root_pose_world": copy.deepcopy(subject["pose_world"]),
            "joint_positions": {},
        },
    )
    usd_sources = {
        "task_object": """#usda 1.0
(defaultPrim = "Asset"; metersPerUnit = 1; upAxis = "Z")
def Xform "Asset" {
    def Xform "body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
        def Cube "shape" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
            double size = 0.1
        }
    }
}
""",
        "scene_collision": """#usda 1.0
(defaultPrim = "Scene"; metersPerUnit = 1; upAxis = "Z")
def Xform "Scene" {
    def Cube "support" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        double size = 4
    }
}
""",
    }
    for row in request["assets"]:
        if row["semantic_role"] in usd_sources:
            path = evidence / row["source"]["relative_path"]
            path.write_text(usd_sources[row["semantic_role"]])
            row["source"].update(
                size_bytes=path.stat().st_size,
                sha256="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            )
    request["task_spec"].update(_rigid_v2_spec())
    request["task_spec"].update(_rigid_fixture(asset_id="fixture_part")["task_spec"])
    request["task_spec"].update(
        schema_version="adp_task_spec.v1",
        maximum_action_steps=240,
        start_pose_world=[1.0, y, 0.0, 0.0, 0.0, 0.0, 1.0],
        destination_position_world_m=[1.2, y, 0.0],
        support_plane_z_m=0.0,
    )
    request["task_spec"]["task_success_contract"] = seal_rigid_task_success_contract(
        task_spec=request["task_spec"],
        site_id=site,
        task_id=request["task_id"],
        author_source="task_owner",
        author_id="cpu-test-fixture",
        confirmation_status="confirmed",
        confirmed_by_team_id="fixture-team",
    )
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    source_root = root / "source"
    materialize_native_task_arena_packet(
        request=request, evidence_root=evidence, output_dir=source_root
    )
    return (
        request,
        evidence,
        source_root,
        _read(source_root / "native_task_arena_scene_plan.v1.json"),
    )


def _cell_input(source):
    request = matrix_request(task_id=source["task_id"], embodiment_id="franka_panda")
    subject = next(row for row in source["objects"] if row["task_subject"])
    camera = next(row for row in source["cameras"] if row["role"] == "external")
    request["scene_binding"].update(
        scene_id=source["scene_id"],
        canonical_object_asset_id=subject["asset_id"],
        canonical_object_asset_digest=subject["sha256"],
    )
    request["task_binding"].update(
        task_digest=canonical_digest(source["task_spec"]),
        success_contract_digest=source["task_spec"]["task_success_contract"]["contract_digest"],
    )
    request["embodiment_binding"]["embodiment_digest"] = native_runtime_embodiment_digest(source)
    y = subject["pose_world"]["position_world_m"][1]
    camera_x = camera["frame_from_camera_matrix"][3]
    request["variation_dimensions"] = []
    for name, family, nominal, delta, unit, target in (
        ("start_y", "placement_approach", y, 0.02, "m", "object_start_position_m.y"),
        ("yaw", "placement_approach", 0.0, 5.0, "degrees", "object_orientation.yaw"),
        ("light", "illumination", 1.0, 0.2, "ratio", "task_light.intensity_scale"),
        ("camera_x", "camera_sensor", camera_x, 0.02, "m", "external_camera.pose.position.x"),
    ):
        dimension = _dimension(
            name,
            family,
            nominal=nominal,
            minimum=nominal - delta,
            maximum=nominal + delta,
            unit=unit,
        )
        dimension["application_target"] = "EventManager.reset." + target
        request["variation_dimensions"].append(dimension)
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    matrix = compile_variation_matrix(request)
    return {
        "request": request,
        "matrix": matrix,
        "cell_id": matrix["cells"][-1]["cell_id"],
        "source_plan_digest": source["plan_digest"],
    }


@pytest.mark.slow
@pytest.mark.parametrize("site,y,use_cli", [("site-a", 1.3, False), ("site-b", 2.4, True)])
def test_actual_cell_packet_replays_and_existing_worker_manifest_binds_prepared_plan(
    tmp_path, site, y, use_cli
):
    request, evidence, original, source = _source(tmp_path, site=site, y=y)
    original_bytes = {
        path.relative_to(original): path.read_bytes()
        for path in original.rglob("*")
        if path.is_file()
    }
    cell_input = _cell_input(source)
    output = tmp_path / "cell"
    if use_cli:
        repo = Path(__file__).resolve().parents[1]
        request_path = tmp_path / "packet-request.json"
        request_path.write_text(json.dumps(request))
        cell_path = tmp_path / "cell-preparation.json"
        cell_path.write_text(json.dumps(cell_input))

        def run_cli(destination, *optional):
            result = subprocess.run(
                [
                    sys.executable,
                    str(repo / "scripts/materialize_paired_target_native_inputs.py"),
                    "arena-packet",
                    "--request", str(request_path),
                    "--evidence-root", str(evidence),
                    "--output-dir", str(destination),
                    *optional,
                ],
                cwd=repo,
                env={**os.environ, "PYTHONPATH": str(repo / "src")},
                capture_output=True,
                text=True,
                timeout=90,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            summary = json.loads(result.stdout)
            assert summary["status"] == "materialized"
            assert summary["provider_mutation_performed"] is False
            retained = _read(destination / "native_task_arena_packet_receipt.v1.json")
            assert summary["receipt_digest"] == retained["receipt_digest"]
            return retained

        legacy = tmp_path / "legacy-cli-packet"
        legacy_receipt = run_cli(legacy)
        assert _read(legacy / "native_task_arena_scene_plan.v1.json") == source
        assert "exact_workcell_cell_preparation_digest" not in legacy_receipt
        receipt = run_cli(output, "--exact-workcell-cell", str(cell_path))
    else:
        receipt = materialize_native_task_arena_packet(
            request=request,
            evidence_root=evidence,
            output_dir=output,
            exact_workcell_cell=cell_input,
        )
    preparation = _read(output / PREPARATION_FILENAME)
    retained_source = _read(output / SOURCE_PLAN_FILENAME)
    plan = _read(output / "native_task_arena_scene_plan.v1.json")
    assert retained_source == source
    assert preparation["cell_preparation"] == cell_input
    assert (
        replay_native_packet_cell(source_scene_plan=retained_source, preparation=preparation)
        == plan
    )
    assert receipt["request_digest"] == request["request_digest"]
    assert (
        plan["runtime_contract_digest"]
        == receipt["runtime_contract_digest"]
        == source["runtime_contract_digest"]
    )
    assert receipt["scenario_instance_digest"] == plan["scenario"]["instance_digest"]
    assert receipt["arena_scene_plan_digest"] == plan["plan_digest"]
    assert receipt["exact_workcell_cell_preparation_digest"] == preparation["preparation_digest"]
    subject = next(row for row in plan["objects"] if row["task_subject"])
    resolved = cell_input["matrix"]["cells"][-1]["resolved_values"]
    assert subject["pose_world"]["position_world_m"][1] == resolved["start_y"]
    assert plan["task_spec"]["start_pose_world"][1] == resolved["start_y"]
    assert any(
        row["unit"] == "degrees" and row["resolved_value"] != row["nominal_value"]
        for row in plan["scenario"]["parameter_bindings"]
    )
    for path in (output / "assets").rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (original / path.relative_to(output)).read_bytes()
    for path, content in original_bytes.items():
        assert (original / path).read_bytes() == content
    assert preparation["construction_result_created"] is False
    assert not (output / "native_task_arena_construction_result.v1.json").exists()
    bundle = build_native_task_arena_construction_bundle(
        job_dir=tmp_path / "job",
        packet_dir=output,
        runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
        implementation_commit="a" * 40,
        generated_at="fixed",
    )
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(bundle["bundle_path"]) as archive:
        archive.extractall(extracted)
    runtime = extracted / "provider_runtime"
    manifest = _load_and_verify_manifest(runtime)
    bundled = runtime / "native_task_packet"
    bundled_plan = _read(bundled / "native_task_arena_scene_plan.v1.json")
    assert bundled_plan == plan and manifest["arena_scene_plan_digest"] == plan["plan_digest"]
    assert manifest["packet_receipt_digest"] == receipt["receipt_digest"]
    assert (
        replay_native_packet_cell(
            source_scene_plan=_read(bundled / SOURCE_PLAN_FILENAME),
            preparation=_read(bundled / PREPARATION_FILENAME),
        )
        == bundled_plan
    )
    assert _read(
        verified_construction_phase_plan_path(runtime, manifest)
    ) == materialize_native_task_construction_phase_plan(bundled_plan)
    for artifact in receipt["artifacts"]:
        body = (bundled / artifact["relative_path"]).read_bytes()
        assert len(body) == artifact["size_bytes"]
        assert "sha256:" + hashlib.sha256(body).hexdigest() == artifact["sha256"]


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("source", "source_plan_digest_mismatch"),
        ("target", "target_unsupported"),
        ("shape", "preparation_invalid"),
    ],
)
def test_cell_packet_preparation_refuses_changed_source_or_unavailable_inputs(
    tmp_path, mutation, code
):
    request, evidence, _, source = _source(tmp_path)
    value = _cell_input(source)
    if mutation == "source":
        value["source_plan_digest"] = "sha256:" + "e" * 64
    elif mutation == "target":
        value["request"]["variation_dimensions"][0]["application_target"] = (
            "EventManager.reset.unsupported"
        )
        value["request"]["request_digest"] = canonical_digest(
            value["request"], digest_field="request_digest"
        )
        value["matrix"] = compile_variation_matrix(value["request"])
        value["cell_id"] = value["matrix"]["cells"][-1]["cell_id"]
    else:
        value["caller_qualified_construction"] = True
    output = tmp_path / "refused"
    with pytest.raises(ExactWorkcellVariationError, match=code):
        materialize_native_task_arena_packet(
            request=request,
            evidence_root=evidence,
            output_dir=output,
            exact_workcell_cell=value,
        )
    assert not output.exists()


def test_cell_packet_replay_refuses_mutated_preparation_receipt(tmp_path):
    from blueprint_pipeline.exact_workcell_native_packet import prepare_native_packet_cell

    request, _, _, source = _source(tmp_path)
    _, receipt = prepare_native_packet_cell(
        source_scene_plan=source,
        cell_preparation=_cell_input(source),
        source_packet_request_digest=request["request_digest"],
        source_runtime_contract_digest=source["runtime_contract_digest"],
    )
    changed = copy.deepcopy(receipt)
    changed["execution_authority_granted"] = True
    with pytest.raises(ExactWorkcellVariationError, match="preparation_digest_mismatch"):
        replay_native_packet_cell(source_scene_plan=source, preparation=changed)
