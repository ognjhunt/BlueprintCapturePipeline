from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from PIL import Image
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from blueprint_pipeline.agent_cad_graph_visual_composition import (
    validate_agent_cad_visual_binding,
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
from blueprint_pipeline.cad_agent_review_media import (
    RECEIPT_FILENAME,
    materialize_cad_agent_visual_comparison,
    seal_cad_agent_visual_reference_review,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.freeze_amendment_carry_forward import (
    evaluate_freeze_amendment_carry_forward,
)
from blueprint_pipeline.replacement_asset_frame_registration import (
    seal_replacement_asset_frame_registration,
)
from blueprint_pipeline.simready_cad_agent_contract import (
    INSPECTION_SCHEMA_VERSION,
    file_record,
    seal_cad_agent_execution_receipt,
    seal_cad_agent_matrix,
    seal_cad_agent_output,
    seal_cad_agent_reference_manifest,
    seal_cad_agent_request,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_INPUTS_CLI = REPO_ROOT / "scripts/materialize_paired_target_native_inputs.py"
FREEZE_A = (
    REPO_ROOT / "docs/arm_decision_proof_v1/manifests/third_scene_840920_task_a_freeze.v1.json"
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


def _write_png(path: Path, color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 30), color).save(path)
    return path


def _run_native_inputs_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    return subprocess.run(
        [sys.executable, str(NATIVE_INPUTS_CLI), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def _backend(tmp_path: Path, backend_id: str = "earthtojake_text_to_cad") -> dict:
    archive = tmp_path / f"{backend_id}.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as source:
        source.comment = b"1" * 40
        source.writestr("LICENSE", "MIT\n")
    return {
        "backend_id": backend_id,
        "execution_mode": (
            "codex_skill_step_first"
            if backend_id == "earthtojake_text_to_cad"
            else "codex_agent_direct_repo_route"
        ),
        "agent_authored_geometry": True,
        "deterministic_geometry_generator_used": False,
        "graph_geometry_used_for_cad_authoring": False,
        "deterministic_format_conversion_only": True,
        "repository_url": (
            "https://github.com/earthtojake/text-to-cad"
            if backend_id == "earthtojake_text_to_cad"
            else "https://github.com/Pan-Chera/Multi-Agent-CAD"
        ),
        "commit": "1" * 40,
        "tree": "2" * 40,
        "source_archive": _record(archive),
        "license": "MIT",
        "model_id": (
            "codex_gpt_5_6_luna" if backend_id == "earthtojake_text_to_cad" else "gpt-5.6"
        ),
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
                "agent_authored_display_color_rgba": [0.8, 0.8, 0.8, 1.0],
            },
            {
                "prim_path": "/Asset/links/door/geometry/rim",
                "link_id": "door",
                "solid_id": "rim",
                "assembly_transform_applied": True,
                "points_mm": [[0, 0, 10], [50, 0, 10], [0, 50, 10]],
                "triangles": [[0, 1, 2]],
                "agent_authored_display_color_rgba": [0.1, 0.1, 0.1, 1.0],
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
    receipt_path = _write(tmp_path / "projection.receipt.json", json.dumps(receipt, sort_keys=True))
    return step, receipt_path


def _cad_output(
    tmp_path: Path,
    *,
    step: Path,
    backend_id: str = "earthtojake_text_to_cad",
    reference_manifest_path: Path | None = None,
) -> Path:
    brief = _write(tmp_path / "brief.md", "Exact observed dimensions only")
    if reference_manifest_path is None:
        image = _write_png(tmp_path / "reference.png", (120, 10, 10))
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
    else:
        reference_path = reference_manifest_path
    request = seal_cad_agent_request(
        request_id="fixture-agent-cad",
        scene_id="fixture_scene",
        task_id=TASK_ID,
        asset_id=ASSET_ID,
        replacement_slot=1,
        backend=_backend(tmp_path, backend_id),
        task_freeze_path=FREEZE_A,
        cad_brief_path=brief,
        metric_envelope_mm=[600.112, 604.104004, 847.564026],
        reference_manifest_path=reference_path,
    )
    generator = _write(tmp_path / "agent.py", "# agent authored\n")
    snapshot = _write_png(tmp_path / "snapshot_iso.png", (10, 110, 180))
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
    inspection["receipt_digest"] = canonical_digest(inspection, digest_field="receipt_digest")
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
        actual_cost_usd=0.0 if backend_id == "earthtojake_text_to_cad" else 0.75,
    )
    return _write(tmp_path / "cad_output.json", json.dumps(output))


def _visual_review_for_matrix(tmp_path: Path, matrix_path: Path) -> Path:
    media = materialize_cad_agent_visual_comparison(
        matrix_path=matrix_path,
        output_dir=tmp_path / "review_media",
    )
    decisions: list[dict] = []
    for row in media["rows"]:
        references = [reference["sha256"] for reference in row["reference_images"]]
        for candidate in row["candidates"]:
            decisions.append(
                {
                    "replacement_slot": row["replacement_slot"],
                    "task_id": row["task_id"],
                    "asset_id": row["asset_id"],
                    "backend_id": candidate["backend_id"],
                    "cad_agent_output_receipt_digest": candidate["output_receipt_digest"],
                    "reference_signature": row["reference_signature"],
                    "reviewed_reference_image_digests": references,
                    "review_status": "conditionally_admitted_for_construction",
                    "observed_feature_findings": [
                        {
                            "feature_id": f"reference_{index}",
                            "status": "mismatch",
                            "evidence_reference_image_digests": [digest],
                        }
                        for index, digest in enumerate(references)
                    ],
                    "visible_mismatch_codes": ["fixture_visual_mismatch"],
                    "generated_candidate_content_labels": [
                        "unseen_geometry_is_generated_candidate"
                    ],
                }
            )
    review_path = tmp_path / "review_media" / "visual_review.v1.json"
    seal_cad_agent_visual_reference_review(
        review_media_receipt_path=tmp_path / "review_media" / RECEIPT_FILENAME,
        reviewer={
            "reviewer_kind": "codex_visual_review",
            "reviewer_id": "fixture-codex",
            "visual_input_mode": ("all_manifest_bound_reference_frames_and_candidate_snapshots"),
        },
        candidate_decisions=decisions,
        output_path=review_path,
    )
    return review_path


def _visual_review_for_candidate(tmp_path: Path, *, candidate_path: Path, step: Path) -> Path:
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    backend_id = candidate["request"]["backend"]["backend_id"]
    alternate_backend = (
        "pan_chera_multi_agent_cad"
        if backend_id == "earthtojake_text_to_cad"
        else "earthtojake_text_to_cad"
    )
    alternate = json.loads(
        _cad_output(
            tmp_path / "alternate",
            step=step,
            backend_id=alternate_backend,
            reference_manifest_path=Path(
                candidate["request"]["inputs"]["reference_manifest"]["path"]
            ),
        ).read_text(encoding="utf-8")
    )
    matrix = seal_cad_agent_matrix(
        objects=[
            {
                "replacement_slot": 1,
                "task_id": TASK_ID,
                "asset_id": ASSET_ID,
                "candidates": [candidate, alternate],
            }
        ]
    )
    matrix_path = _write(tmp_path / "matrix.v1.json", json.dumps(matrix))
    return _visual_review_for_matrix(tmp_path, matrix_path)


def _graph_authoring_receipt(
    tmp_path: Path,
    *,
    collision_isolated: bool = True,
    extra_renderable: bool = False,
    door_translation: tuple[float, float, float] | None = None,
    task_freeze_digest: str | None = None,
) -> Path:
    graph_usd = tmp_path / "graph.usda"
    stage = Usd.Stage.CreateNew(str(graph_usd))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    root.GetPrim().SetCustomDataByKey("blueprint:assetId", ASSET_ID)
    stage.SetDefaultPrim(root.GetPrim())
    body = UsdGeom.Xform.Define(stage, "/Asset/links/body")
    door = UsdGeom.Xform.Define(stage, "/Asset/links/door")
    if door_translation is not None:
        door.AddTranslateOp().Set(Gf.Vec3d(*door_translation))
        door.AddOrientOp().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
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
        "task_freeze_digest": task_freeze_digest or freeze["task_freeze_digest"],
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
    visual_review = _visual_review_for_candidate(
        tmp_path / "visual_review", candidate_path=cad_output, step=step
    )
    graph = _graph_authoring_receipt(tmp_path / "graph")
    binding = seal_agent_cad_visual_binding(
        graph_authoring_receipt_path=graph,
        cad_agent_output_receipt_path=cad_output,
        cad_agent_visual_review_path=visual_review,
        mesh_projection_receipt_path=projection,
        link_bindings=[
            {
                "agent_link_id": "body",
                "graph_link_id": "body",
                "transform_mode": "explicit_rigid_transform",
                "T_graph_link_from_agent_asset": _identity(),
            },
            {
                "agent_link_id": "door",
                "graph_link_id": "door",
                "transform_mode": "explicit_rigid_transform",
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


def test_native_inputs_cli_static_qualification_reopens_registered_asset_bytes(
    tmp_path: Path,
) -> None:
    """Registered mode follows the real graph->visual->registration chain."""

    source_asset = _write(tmp_path / "source.usda", "#usda 1.0\n")
    source_receipt = {
        "schema_version": "articulated_source_asset.v1",
        "status": "materialized",
        "source_collision_prim_path": "/Root/ZFAVSKZVAJTGUPTUKM888888",
        "target": {"interiorgs_instance_id": "165"},
        "output_asset": {
            "relative_path": source_asset.name,
            "size_bytes": source_asset.stat().st_size,
            "sha256": _record(source_asset)["sha256"],
        },
        "receipt_digest": "",
    }
    source_receipt["receipt_digest"] = canonical_digest(
        source_receipt, digest_field="receipt_digest"
    )
    source_receipt_path = _write(
        tmp_path / "source.receipt.json",
        json.dumps(source_receipt, indent=2, sort_keys=True) + "\n",
    )
    graph_spec = json.loads(
        (
            REPO_ROOT
            / "docs/arm_decision_proof_v1/manifests/"
            "third_scene_840920_task_a_simready_graph_asset_spec.v1.json"
        ).read_text()
    )
    graph_spec["source_asset_receipt_digest"] = source_receipt["receipt_digest"]
    graph_spec["spec_digest"] = canonical_digest(
        graph_spec, digest_field="spec_digest"
    )
    graph_spec_path = _write(
        tmp_path / "graph-spec.json",
        json.dumps(graph_spec, indent=2, sort_keys=True) + "\n",
    )
    graph_usd = tmp_path / "graph.usda"
    graph_receipt = tmp_path / "graph.receipt.json"
    graph_process = _run_native_inputs_cli(
        "simready-graph-asset",
        "--spec",
        str(graph_spec_path),
        "--task-freeze",
        str(FREEZE_A),
        "--source-asset-receipt",
        str(source_receipt_path),
        "--output-usd",
        str(graph_usd),
        "--output-receipt",
        str(graph_receipt),
    )
    assert graph_process.returncode == 0, graph_process.stderr + graph_process.stdout
    assert json.loads(graph_process.stdout)["provider_mutation_performed"] is False

    step, projection = _mesh_projection(tmp_path / "projection")
    cad_output = _cad_output(tmp_path / "cad", step=step)
    visual_review = _visual_review_for_candidate(
        tmp_path / "visual-review", candidate_path=cad_output, step=step
    )
    link_bindings = [
        {
            "agent_link_id": link_id,
            "graph_link_id": link_id,
            "transform_mode": "explicit_rigid_transform",
            "T_graph_link_from_agent_asset": _identity(),
        }
        for link_id in ("body", "door")
    ]
    unmapped = {
        link_id: "no selected agent-authored visual mesh for this graph link"
        for link_id in ("drawer", "drum", "latch", "selector")
    }
    binding_path = tmp_path / "binding.json"
    seal_agent_cad_visual_binding(
        graph_authoring_receipt_path=graph_receipt,
        cad_agent_output_receipt_path=cad_output,
        cad_agent_visual_review_path=visual_review,
        mesh_projection_receipt_path=projection,
        link_bindings=link_bindings,
        unmapped_graph_link_reasons=unmapped,
        output_path=binding_path,
    )
    composed_usd = tmp_path / "composed.usda"
    composition = materialize_agent_cad_visual_composition(
        binding_path=binding_path,
        destination_usd_path=composed_usd,
    )
    composition_receipt = composed_usd.with_suffix(".receipt.json")
    assert composition["visual_mesh_count"] == 2

    references = [
        _write_png(tmp_path / "registration" / f"view-{index}.png", color)
        for index, color in enumerate(((20, 30, 40), (50, 60, 70)))
    ]
    registration_path = tmp_path / "frame-registration.json"
    seal_replacement_asset_frame_registration(
        scene_id="fixture_scene",
        task_id=TASK_ID,
        asset_id=ASSET_ID,
        asset_local_forward_axis=[1.0, 0.0, 0.0],
        asset_local_up_axis=[0.0, 0.0, 1.0],
        observed_world_forward_axis=[1.0, 0.0, 0.0],
        observed_world_up_axis=[0.0, 0.0, 1.0],
        reference_image_paths=references,
        reviewed_by="fixture-independent-reviewer",
        output_path=registration_path,
    )
    registered_usd = tmp_path / "registered.usda"
    registered_receipt = tmp_path / "registered.receipt.json"
    registered_process = _run_native_inputs_cli(
        "registered-asset",
        "--visual-composition-receipt",
        str(composition_receipt),
        "--frame-registration",
        str(registration_path),
        "--output-usd",
        str(registered_usd),
        "--output-receipt",
        str(registered_receipt),
    )
    assert registered_process.returncode == 0, (
        registered_process.stderr + registered_process.stdout
    )
    assert json.loads(registered_process.stdout)["provider_mutation_performed"] is False

    qualification_path = tmp_path / "registered-static-qualification.json"
    qualification_process = _run_native_inputs_cli(
        "simready-static-qualification",
        "--spec",
        str(graph_spec_path),
        "--authoring-receipt",
        str(graph_receipt),
        "--registered-asset-receipt",
        str(registered_receipt),
        "--output",
        str(qualification_path),
    )
    assert qualification_process.returncode == 0, (
        qualification_process.stderr + qualification_process.stdout
    )
    qualification_summary = json.loads(qualification_process.stdout)
    assert qualification_summary["provider_mutation_performed"] is False
    qualification = json.loads(qualification_path.read_text())
    registered = json.loads(registered_receipt.read_text())
    assert qualification["status"] == "authored_structure_statically_qualified"
    assert qualification["structural_findings"] == []
    assert qualification["replacement_usd"] == registered["output_usd"]
    assert qualification["registered_replacement_asset"]["receipt_digest"] == registered[
        "receipt_digest"
    ]
    assert qualification["registered_visual_readback"] == {
        "render_visible_visual_mesh_count": 2,
        "bound_material_visual_mesh_count": 2,
        "agent_authored_color_visual_mesh_count": 2,
        "expected_visual_mesh_count": 2,
        "asset_frame_registration_digest": json.loads(
            registration_path.read_text()
        )["registration_digest"],
    }
    assert qualification["receipt_digest"] == canonical_digest(
        qualification, digest_field="receipt_digest"
    )


def test_binding_rejects_nonrigid_agent_to_graph_transform(tmp_path: Path) -> None:
    step, projection = _mesh_projection(tmp_path / "projection")
    cad_output = _cad_output(tmp_path / "cad", step=step)
    visual_review = _visual_review_for_candidate(
        tmp_path / "visual_review", candidate_path=cad_output, step=step
    )
    graph = _graph_authoring_receipt(tmp_path / "graph")
    scaled = _identity()
    scaled[0][0] = 2.0
    with pytest.raises(AgentCadGraphVisualCompositionError) as caught:
        seal_agent_cad_visual_binding(
            graph_authoring_receipt_path=graph,
            cad_agent_output_receipt_path=cad_output,
            cad_agent_visual_review_path=visual_review,
            mesh_projection_receipt_path=projection,
            link_bindings=[
                {
                    "agent_link_id": "body",
                    "graph_link_id": "body",
                    "transform_mode": "explicit_rigid_transform",
                    "T_graph_link_from_agent_asset": scaled,
                },
                {
                    "agent_link_id": "door",
                    "graph_link_id": "door",
                    "transform_mode": "explicit_rigid_transform",
                    "T_graph_link_from_agent_asset": _identity(),
                },
            ],
            unmapped_graph_link_reasons={},
            output_path=tmp_path / "binding.json",
        )
    assert "agent_cad_visual_link_bindings_invalid" in caught.value.codes


def test_binding_can_derive_link_local_transform_from_graph_reset_pose(
    tmp_path: Path,
) -> None:
    step, projection = _mesh_projection(tmp_path / "projection")
    cad_output = _cad_output(tmp_path / "cad", step=step)
    visual_review = _visual_review_for_candidate(
        tmp_path / "visual_review", candidate_path=cad_output, step=step
    )
    graph = _graph_authoring_receipt(tmp_path / "graph", door_translation=(0.0, 0.2, 0.3))
    binding_path = tmp_path / "binding.json"
    binding = seal_agent_cad_visual_binding(
        graph_authoring_receipt_path=graph,
        cad_agent_output_receipt_path=cad_output,
        cad_agent_visual_review_path=visual_review,
        mesh_projection_receipt_path=projection,
        link_bindings=[
            {
                "agent_link_id": "body",
                "graph_link_id": "body",
                "transform_mode": "explicit_rigid_transform",
                "T_graph_link_from_agent_asset": _identity(),
            },
            {
                "agent_link_id": "door",
                "graph_link_id": "door",
                "transform_mode": "graph_link_reset_inverse",
            },
        ],
        unmapped_graph_link_reasons={},
        output_path=binding_path,
    )
    door_binding = next(row for row in binding["link_bindings"] if row["agent_link_id"] == "door")
    assert door_binding["T_graph_link_from_agent_asset"] == [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -0.2],
        [0.0, 0.0, 1.0, -0.3],
        [0.0, 0.0, 0.0, 1.0],
    ]
    destination = tmp_path / "composed.usda"
    materialize_agent_cad_visual_composition(
        binding_path=binding_path, destination_usd_path=destination
    )
    stage = Usd.Stage.Open(str(destination))
    visual = UsdGeom.Mesh(stage.GetPrimAtPath("/Asset/links/door/visuals/door__rim"))
    assert list(visual.GetPointsAttr().Get())[0] == pytest.approx((0.0, -0.2, -0.29))


def test_binding_selects_one_exact_candidate_from_a_verified_cad_matrix(
    tmp_path: Path,
) -> None:
    step, projection = _mesh_projection(tmp_path / "projection")
    earth_path = _cad_output(tmp_path / "earth", step=step)
    earth = json.loads(earth_path.read_text(encoding="utf-8"))
    multi_path = _cad_output(
        tmp_path / "multi",
        step=step,
        backend_id="pan_chera_multi_agent_cad",
        reference_manifest_path=Path(earth["request"]["inputs"]["reference_manifest"]["path"]),
    )
    multi = json.loads(multi_path.read_text(encoding="utf-8"))
    matrix = seal_cad_agent_matrix(
        objects=[
            {
                "replacement_slot": 1,
                "task_id": TASK_ID,
                "asset_id": ASSET_ID,
                "candidates": [earth, multi],
            }
        ]
    )
    matrix_path = _write(tmp_path / "matrix.json", json.dumps(matrix))
    visual_review = _visual_review_for_matrix(tmp_path / "visual_review", matrix_path)
    graph = _graph_authoring_receipt(tmp_path / "graph")
    binding = seal_agent_cad_visual_binding(
        graph_authoring_receipt_path=graph,
        cad_agent_matrix_path=matrix_path,
        cad_agent_backend_id="earthtojake_text_to_cad",
        cad_agent_visual_review_path=visual_review,
        mesh_projection_receipt_path=projection,
        link_bindings=[
            {
                "agent_link_id": "body",
                "graph_link_id": "body",
                "transform_mode": "explicit_rigid_transform",
                "T_graph_link_from_agent_asset": _identity(),
            },
            {
                "agent_link_id": "door",
                "graph_link_id": "door",
                "transform_mode": "explicit_rigid_transform",
                "T_graph_link_from_agent_asset": _identity(),
            },
        ],
        unmapped_graph_link_reasons={},
        output_path=tmp_path / "binding.json",
    )
    assert binding["cad_agent_matrix"]["sha256"] == _record(matrix_path)["sha256"]
    assert binding["cad_agent_backend_id"] == "earthtojake_text_to_cad"
    assert binding["cad_agent_output_receipt_digest"] == earth["receipt_digest"]


def test_binding_rejects_render_visible_graph_collision(tmp_path: Path) -> None:
    step, projection = _mesh_projection(tmp_path / "projection")
    cad_output = _cad_output(tmp_path / "cad", step=step)
    visual_review = _visual_review_for_candidate(
        tmp_path / "visual_review", candidate_path=cad_output, step=step
    )
    graph = _graph_authoring_receipt(tmp_path / "graph", collision_isolated=False)
    with pytest.raises(AgentCadGraphVisualCompositionError) as caught:
        seal_agent_cad_visual_binding(
            graph_authoring_receipt_path=graph,
            cad_agent_output_receipt_path=cad_output,
            cad_agent_visual_review_path=visual_review,
            mesh_projection_receipt_path=projection,
            link_bindings=[
                {
                    "agent_link_id": "body",
                    "graph_link_id": "body",
                    "transform_mode": "explicit_rigid_transform",
                    "T_graph_link_from_agent_asset": _identity(),
                },
                {
                    "agent_link_id": "door",
                    "graph_link_id": "door",
                    "transform_mode": "explicit_rigid_transform",
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
    visual_review = _visual_review_for_candidate(
        tmp_path / "visual_review", candidate_path=cad_output, step=step
    )
    graph = _graph_authoring_receipt(tmp_path / "graph", extra_renderable=True)
    with pytest.raises(AgentCadGraphVisualCompositionError) as caught:
        seal_agent_cad_visual_binding(
            graph_authoring_receipt_path=graph,
            cad_agent_output_receipt_path=cad_output,
            cad_agent_visual_review_path=visual_review,
            mesh_projection_receipt_path=projection,
            link_bindings=[
                {
                    "agent_link_id": "body",
                    "graph_link_id": "body",
                    "transform_mode": "explicit_rigid_transform",
                    "T_graph_link_from_agent_asset": _identity(),
                },
                {
                    "agent_link_id": "door",
                    "graph_link_id": "door",
                    "transform_mode": "explicit_rigid_transform",
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


def _amended_freeze_document() -> dict:
    """FREEZE_A with one joint-axis component flipped and its digest resealed.

    The exact shape of Scene 840920's 2026-08-17 amendment: the washer door
    hinge was authored backwards, the fix flipped one leaf the CAD lane never
    reads, and the graph asset was re-authored against the amended freeze while
    the CAD receipts stayed sealed to the superseded one.
    """

    amended = json.loads(FREEZE_A.read_text(encoding="utf-8"))
    joints = amended["articulation_graph"]["joints"]
    joints[0]["axis"] = [-value for value in joints[0]["axis"]]
    amended["task_freeze_digest"] = ""
    amended["task_freeze_digest"] = canonical_digest(
        amended, digest_field="task_freeze_digest"
    )
    return amended


def _binding_carry_forward_proof(amended: dict) -> dict:
    return evaluate_freeze_amendment_carry_forward(
        superseded_freeze=json.loads(FREEZE_A.read_text(encoding="utf-8")),
        amended_freeze=amended,
        sealed_schema="simready_agent_cad_visual_binding.v2",
    )


def test_binding_across_the_freeze_amendment_needs_its_proof(tmp_path: Path) -> None:
    """CAD sealed to the superseded freeze, graph to the amended one.

    Without a proof that the amendment changed nothing the binding consumes,
    that divergence stays exactly as fatal as it was -- an unexplained split
    between the two sides is not a carry-forward.
    """

    amended = _amended_freeze_document()
    step, projection = _mesh_projection(tmp_path / "projection")
    cad_output = _cad_output(tmp_path / "cad", step=step)
    visual_review = _visual_review_for_candidate(
        tmp_path / "visual_review", candidate_path=cad_output, step=step
    )
    graph = _graph_authoring_receipt(
        tmp_path / "graph", task_freeze_digest=amended["task_freeze_digest"]
    )
    bindings = [
        {
            "agent_link_id": "body",
            "graph_link_id": "body",
            "transform_mode": "explicit_rigid_transform",
            "T_graph_link_from_agent_asset": _identity(),
        },
        {
            "agent_link_id": "door",
            "graph_link_id": "door",
            "transform_mode": "explicit_rigid_transform",
            "T_graph_link_from_agent_asset": _identity(),
        },
    ]

    with pytest.raises(AgentCadGraphVisualCompositionError) as caught:
        seal_agent_cad_visual_binding(
            graph_authoring_receipt_path=graph,
            cad_agent_output_receipt_path=cad_output,
            cad_agent_visual_review_path=visual_review,
            mesh_projection_receipt_path=projection,
            link_bindings=bindings,
            unmapped_graph_link_reasons={},
            output_path=tmp_path / "binding_unproven.json",
        )
    assert "agent_cad_visual_binding_identity_mismatch" in caught.value.codes

    binding = seal_agent_cad_visual_binding(
        graph_authoring_receipt_path=graph,
        cad_agent_output_receipt_path=cad_output,
        cad_agent_visual_review_path=visual_review,
        mesh_projection_receipt_path=projection,
        link_bindings=bindings,
        unmapped_graph_link_reasons={},
        output_path=tmp_path / "binding.json",
        freeze_carry_forward=_binding_carry_forward_proof(amended),
    )
    # The binding is sealed to the amended freeze, names the superseded one it
    # spans, and carries the proof -- so it re-validates with no side channel.
    assert binding["task_freeze_digest"] == amended["task_freeze_digest"]
    freeze_a = json.loads(FREEZE_A.read_text(encoding="utf-8"))
    assert binding["superseded_task_freeze_digest"] == freeze_a["task_freeze_digest"]
    assert validate_agent_cad_visual_binding(binding) == binding


def test_binding_refuses_a_proof_for_a_different_amendment(tmp_path: Path) -> None:
    """One cheap ruling must not launder every divergence in the tree."""

    amended = _amended_freeze_document()
    proof = _binding_carry_forward_proof(amended)
    tampered = dict(proof)
    tampered["superseded_task_freeze_digest"] = "sha256:" + "e" * 64
    tampered["carry_forward_digest"] = ""
    tampered["carry_forward_digest"] = canonical_digest(
        tampered, digest_field="carry_forward_digest"
    )

    step, projection = _mesh_projection(tmp_path / "projection")
    cad_output = _cad_output(tmp_path / "cad", step=step)
    visual_review = _visual_review_for_candidate(
        tmp_path / "visual_review", candidate_path=cad_output, step=step
    )
    graph = _graph_authoring_receipt(
        tmp_path / "graph", task_freeze_digest=amended["task_freeze_digest"]
    )
    with pytest.raises(AgentCadGraphVisualCompositionError) as caught:
        seal_agent_cad_visual_binding(
            graph_authoring_receipt_path=graph,
            cad_agent_output_receipt_path=cad_output,
            cad_agent_visual_review_path=visual_review,
            mesh_projection_receipt_path=projection,
            link_bindings=[
                {
                    "agent_link_id": "body",
                    "graph_link_id": "body",
                    "transform_mode": "explicit_rigid_transform",
                    "T_graph_link_from_agent_asset": _identity(),
                },
                {
                    "agent_link_id": "door",
                    "graph_link_id": "door",
                    "transform_mode": "explicit_rigid_transform",
                    "T_graph_link_from_agent_asset": _identity(),
                },
            ],
            unmapped_graph_link_reasons={},
            output_path=tmp_path / "binding.json",
            freeze_carry_forward=tampered,
        )
    assert "agent_cad_visual_binding_identity_mismatch" in caught.value.codes
