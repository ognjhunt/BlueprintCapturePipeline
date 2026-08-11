from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from blueprint_pipeline.agent_cad_graph_visual_composition import (
    AgentCadGraphVisualCompositionError,
    materialize_agent_cad_visual_composition,
    seal_agent_cad_visual_binding,
    validate_agent_cad_visual_composition,
    validate_agent_cad_visual_composition_set,
)
from blueprint_pipeline.cad_agent_mesh_projection import (
    PACKET_SCHEMA_VERSION,
    materialize_mesh_usd_projection,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.simready_cad_agent_contract import (
    INSPECTION_SCHEMA_VERSION,
    file_record,
    seal_cad_agent_execution_receipt,
    seal_cad_agent_output,
    seal_cad_agent_reference_manifest,
    seal_cad_agent_request,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE_A = (
    REPO_ROOT
    / "docs/arm_decision_proof_v1/manifests/third_scene_840920_task_a_freeze.v1.json"
)
TASK_ID = "task_a_washer_door_open"
ASSET_ID = "840920_simready_washer_candidate"


def _write(path: Path, payload: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(payload, encoding="utf-8")
    return path


def _record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _backend(tmp_path: Path) -> dict:
    archive = tmp_path / "earthtojake.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as source:
        source.comment = b"1" * 40
        source.writestr("LICENSE", "MIT\n")
    return {
        "backend_id": "earthtojake_text_to_cad",
        "execution_mode": "codex_skill_step_first",
        "agent_authored_geometry": True,
        "deterministic_geometry_generator_used": False,
        "graph_geometry_used_for_cad_authoring": False,
        "deterministic_format_conversion_only": True,
        "repository_url": "https://github.com/earthtojake/text-to-cad",
        "commit": "1" * 40,
        "tree": "2" * 40,
        "source_archive": _record(archive),
        "license": "MIT",
        "model_id": "codex_gpt_5_6_luna",
    }


def _mesh_projection(tmp_path: Path) -> tuple[Path, Path]:
    step = _write(tmp_path / "agent.step", b"exact-agent-authored-step")
    packet = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "geometry_authority": "exact_agent_authored_step",
        "deterministic_geometry_generator_used": False,
        "conversion_only": True,
        "step": _record(step),
        "linear_tolerance_mm": 0.2,
        "angular_tolerance_rad": 0.1,
        "mesh_count": 2,
        "meshes": [
            {
                "prim_path": "/Asset/links/body/geometry/shell",
                "link_id": "body",
                "solid_id": "shell",
                "assembly_transform_applied": True,
                "points_mm": [[0, 0, 0], [100, 0, 0], [0, 100, 0]],
                "triangles": [[0, 1, 2]],
            },
            {
                "prim_path": "/Asset/links/door/geometry/rim",
                "link_id": "door",
                "solid_id": "rim",
                "assembly_transform_applied": True,
                "points_mm": [[0, 0, 10], [50, 0, 10], [0, 50, 10]],
                "triangles": [[0, 1, 2]],
            },
        ],
        "claim_boundary": {
            "cad_authored_by_projection": False,
            "appearance_working_copy_only": True,
            "collision_authority": False,
            "physics_authority": False,
            "simready_qualified": False,
        },
        "packet_digest": "",
    }
    packet["packet_digest"] = canonical_digest(packet, digest_field="packet_digest")
    packet_path = _write(tmp_path / "mesh_packet.json", json.dumps(packet))
    projection_usd = tmp_path / "projection.usda"
    receipt = materialize_mesh_usd_projection(
        packet_path=packet_path, output_usd_path=projection_usd
    )
    receipt_path = _write(
        tmp_path / "projection.receipt.json", json.dumps(receipt, sort_keys=True)
    )
    return step, receipt_path


def _cad_output(tmp_path: Path, *, step: Path) -> Path:
    brief = _write(tmp_path / "brief.md", "Exact observed dimensions only")
    image = _write(tmp_path / "reference.png", b"reference")
    reference = seal_cad_agent_reference_manifest(
        scene_id="fixture_scene",
        objects=[
            {
                "replacement_slot": 1,
                "task_id": TASK_ID,
                "asset_id": ASSET_ID,
                "task_freeze_path": FREEZE_A,
                "reference_image_paths": [image],
            }
        ],
    )
    reference_path = _write(tmp_path / "references.json", json.dumps(reference))
    request = seal_cad_agent_request(
        request_id="fixture-agent-cad",
        scene_id="fixture_scene",
        task_id=TASK_ID,
        asset_id=ASSET_ID,
        replacement_slot=1,
        backend=_backend(tmp_path),
        task_freeze_path=FREEZE_A,
        cad_brief_path=brief,
        metric_envelope_mm=[600.112, 604.104004, 847.564026],
        reference_manifest_path=reference_path,
    )
    generator = _write(tmp_path / "agent.py", "# agent authored\n")
    snapshot = _write(tmp_path / "snapshot.png", b"snapshot")
    inspection = {
        "schema_version": INSPECTION_SCHEMA_VERSION,
        "status": "passed",
        "step": file_record(step),
        "measured_envelope_mm": [600.112, 604.104004, 847.564026],
        "measured_center_mm": [0.0, 0.0, 423.782013],
        "topology": {"face_count": 6, "edge_count": 12},
        "inspector": {
            "backend": "build123d_import_step",
            "build123d_version": "fixture",
            "ocp_version": "fixture",
            "python_version": "fixture",
            "module_source": file_record(
                REPO_ROOT / "src/blueprint_pipeline/simready_cad_agent_contract.py"
            ),
        },
        "claim_boundary": {
            "step_bytes_reopened": True,
            "metric_envelope_measured": True,
            "visual_quality_qualified": False,
            "simready_qualified": False,
            "physical_equivalence": False,
        },
        "receipt_digest": "",
    }
    inspection["receipt_digest"] = canonical_digest(
        inspection, digest_field="receipt_digest"
    )
    inspection_path = _write(tmp_path / "inspection.json", json.dumps(inspection))
    execution = seal_cad_agent_execution_receipt(
        request=request,
        generator_source_path=generator,
        cad_brief_path=brief,
        output_step_path=step,
        event_rows=[{"event": "agent_authored_step", "status": "passed"}],
    )
    execution_path = _write(tmp_path / "execution.json", json.dumps(execution))
    output = seal_cad_agent_output(
        request=request,
        generator_source_path=generator,
        step_path=step,
        inspection_receipt_path=inspection_path,
        snapshot_paths=[snapshot],
        execution_receipt_path=execution_path,
        measured_envelope_mm=[600.112, 604.104004, 847.564026],
        actual_cost_usd=0.0,
    )
    return _write(tmp_path / "cad_output.json", json.dumps(output))


def _graph_authoring_receipt(
    tmp_path: Path,
    *,
    collision_isolated: bool = True,
    extra_renderable: bool = False,
) -> Path:
    graph_usd = tmp_path / "graph.usda"
    stage = Usd.Stage.CreateNew(str(graph_usd))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    root.GetPrim().SetCustomDataByKey("blueprint:assetId", ASSET_ID)
    stage.SetDefaultPrim(root.GetPrim())
    body = UsdGeom.Xform.Define(stage, "/Asset/links/body")
    UsdGeom.Xform.Define(stage, "/Asset/links/door")
    proxy = UsdGeom.Cube.Define(stage, "/Asset/links/body/geometry/proxy")
    UsdPhysics.CollisionAPI.Apply(proxy.GetPrim())
    proxy.GetPrim().SetCustomDataByKey("blueprint:collisionGeometryOnly", True)
    if collision_isolated:
        proxy.CreatePurposeAttr(UsdGeom.Tokens.guide)
        proxy.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    if extra_renderable:
        UsdGeom.Cube.Define(stage, "/Asset/links/body/legacy_visual")
    stage.GetRootLayer().Save()
    freeze = json.loads(FREEZE_A.read_text(encoding="utf-8"))
    receipt = {
        "schema_version": "simready_graph_asset_receipt.v1",
        "status": "simready_candidate_authored",
        "asset_id": ASSET_ID,
        "task_id": TASK_ID,
        "task_freeze_digest": freeze["task_freeze_digest"],
        "spec_digest": "sha256:" + "a" * 64,
        "output_usd": {
            **_record(graph_usd),
            "default_prim": "/Asset",
            "meters_per_unit": 1.0,
            "up_axis": "Z",
        },
        "link_paths": {"body": str(body.GetPath()), "door": "/Asset/links/door"},
        "joint_paths": {},
        "claim_boundary": {"native_simulator_import_qualified": False},
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return _write(tmp_path / "graph.receipt.json", json.dumps(receipt))


def _identity() -> list[list[float]]:
    return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def _prepared(tmp_path: Path) -> tuple[Path, Path]:
    step, projection = _mesh_projection(tmp_path / "projection")
    cad_output = _cad_output(tmp_path / "cad", step=step)
    graph = _graph_authoring_receipt(tmp_path / "graph")
    binding = seal_agent_cad_visual_binding(
        graph_authoring_receipt_path=graph,
        cad_agent_output_receipt_path=cad_output,
        mesh_projection_receipt_path=projection,
        link_bindings=[
            {
                "agent_link_id": "body",
                "graph_link_id": "body",
                "T_graph_link_from_agent_asset": _identity(),
            },
            {
                "agent_link_id": "door",
                "graph_link_id": "door",
                "T_graph_link_from_agent_asset": _identity(),
            },
        ],
        unmapped_graph_link_reasons={},
        output_path=tmp_path / "binding.json",
    )
    assert binding["task_id"] == TASK_ID
    return tmp_path / "binding.json", tmp_path / "composed.usda"


def test_agent_cad_visuals_are_composed_without_promoting_collision_meshes(
    tmp_path: Path,
) -> None:
    binding_path, destination = _prepared(tmp_path)
    receipt = materialize_agent_cad_visual_composition(
        binding_path=binding_path, destination_usd_path=destination
    )

    assert receipt["visual_mesh_count"] == 2
    assert receipt["collision_visual_isolation_verified"] is True
    assert receipt["claim_boundary"]["native_simulator_import_qualified"] is False
    assert validate_agent_cad_visual_composition(receipt) == receipt
    stage = Usd.Stage.Open(str(destination))
    visual = stage.GetPrimAtPath("/Asset/links/body/visuals/body__shell")
    assert visual.IsA(UsdGeom.Mesh)
    assert list(UsdGeom.Mesh(visual).GetPointsAttr().Get())[1][0] == pytest.approx(0.1)
    assert not visual.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdGeom.Imageable(visual).ComputePurpose() == UsdGeom.Tokens.default_
    proxy = stage.GetPrimAtPath("/Asset/links/body/geometry/proxy")
    assert UsdGeom.Imageable(proxy).ComputePurpose() == UsdGeom.Tokens.guide
    assert UsdGeom.Imageable(proxy).ComputeVisibility() == UsdGeom.Tokens.invisible


def test_binding_rejects_nonrigid_agent_to_graph_transform(tmp_path: Path) -> None:
    step, projection = _mesh_projection(tmp_path / "projection")
    cad_output = _cad_output(tmp_path / "cad", step=step)
    graph = _graph_authoring_receipt(tmp_path / "graph")
    scaled = _identity()
    scaled[0][0] = 2.0
    with pytest.raises(AgentCadGraphVisualCompositionError) as caught:
        seal_agent_cad_visual_binding(
            graph_authoring_receipt_path=graph,
            cad_agent_output_receipt_path=cad_output,
            mesh_projection_receipt_path=projection,
            link_bindings=[
                {
                    "agent_link_id": "body",
                    "graph_link_id": "body",
                    "T_graph_link_from_agent_asset": scaled,
                },
                {
                    "agent_link_id": "door",
                    "graph_link_id": "door",
                    "T_graph_link_from_agent_asset": _identity(),
                },
            ],
            unmapped_graph_link_reasons={},
            output_path=tmp_path / "binding.json",
        )
    assert "agent_cad_visual_link_bindings_invalid" in caught.value.codes


def test_binding_rejects_render_visible_graph_collision(tmp_path: Path) -> None:
    step, projection = _mesh_projection(tmp_path / "projection")
    cad_output = _cad_output(tmp_path / "cad", step=step)
    graph = _graph_authoring_receipt(tmp_path / "graph", collision_isolated=False)
    with pytest.raises(AgentCadGraphVisualCompositionError) as caught:
        seal_agent_cad_visual_binding(
            graph_authoring_receipt_path=graph,
            cad_agent_output_receipt_path=cad_output,
            mesh_projection_receipt_path=projection,
            link_bindings=[
                {
                    "agent_link_id": "body",
                    "graph_link_id": "body",
                    "T_graph_link_from_agent_asset": _identity(),
                },
                {
                    "agent_link_id": "door",
                    "graph_link_id": "door",
                    "T_graph_link_from_agent_asset": _identity(),
                },
            ],
            unmapped_graph_link_reasons={},
            output_path=tmp_path / "binding.json",
        )
    assert "agent_cad_visual_collision_not_isolated" in caught.value.codes


def test_binding_rejects_renderable_noncollision_graph_geometry(tmp_path: Path) -> None:
    step, projection = _mesh_projection(tmp_path / "projection")
    cad_output = _cad_output(tmp_path / "cad", step=step)
    graph = _graph_authoring_receipt(tmp_path / "graph", extra_renderable=True)
    with pytest.raises(AgentCadGraphVisualCompositionError) as caught:
        seal_agent_cad_visual_binding(
            graph_authoring_receipt_path=graph,
            cad_agent_output_receipt_path=cad_output,
            mesh_projection_receipt_path=projection,
            link_bindings=[
                {
                    "agent_link_id": "body",
                    "graph_link_id": "body",
                    "T_graph_link_from_agent_asset": _identity(),
                },
                {
                    "agent_link_id": "door",
                    "graph_link_id": "door",
                    "T_graph_link_from_agent_asset": _identity(),
                },
            ],
            unmapped_graph_link_reasons={},
            output_path=tmp_path / "binding.json",
        )
    assert "agent_cad_visual_graph_renderable_geometry_present" in caught.value.codes


def test_composition_set_caps_replacements_at_five(tmp_path: Path) -> None:
    binding_path, destination = _prepared(tmp_path)
    materialize_agent_cad_visual_composition(
        binding_path=binding_path, destination_usd_path=destination
    )
    receipt_path = destination.with_suffix(".receipt.json")
    valid_set = {
        "schema_version": "scene_agent_cad_visual_composition_set.v1",
        "scene_id": "fixture_scene",
        "maximum_replacement_objects": 5,
        "compositions": [_record(receipt_path)],
        "set_digest": "",
    }
    valid_set["set_digest"] = canonical_digest(valid_set, digest_field="set_digest")
    assert validate_agent_cad_visual_composition_set(valid_set) == valid_set
    oversized = dict(valid_set)
    oversized["compositions"] = [_record(receipt_path)] * 6
    oversized["set_digest"] = canonical_digest(oversized, digest_field="set_digest")
    with pytest.raises(AgentCadGraphVisualCompositionError) as caught:
        validate_agent_cad_visual_composition_set(oversized)
    assert "agent_cad_visual_composition_set_count_invalid" in caught.value.codes
