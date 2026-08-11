from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import zipfile
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from blueprint_pipeline import adp_content_agents_vast as content_agents
from blueprint_pipeline import adp_content_agents_bundle_preflight as bundle_preflight
from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionGrant
from blueprint_pipeline.simready_cad_agent_contract import (
    INSPECTION_SCHEMA_VERSION,
    file_record,
    seal_cad_agent_execution_receipt,
    seal_cad_agent_output,
    seal_cad_agent_reference_manifest,
    seal_cad_agent_request,
)
from blueprint_pipeline.vast_provider_adapter import (
    _blueprint_bundle_preflight,
    _probe_env,
    _probe_shell_script,
    _resolve_launch_mode,
)
from blueprint_pipeline.wam_provider_output import inspect_provider_runtime_output_zip


ROOT = Path(__file__).resolve().parents[1]
TASK_A_FREEZE = (
    ROOT
    / "docs/arm_decision_proof_v1/manifests/third_scene_840920_task_a_freeze.v1.json"
)
AGENT_CAD_CONTENT_BUNDLE_MATRIX = (
    ROOT
    / "docs/arm_decision_proof_v1/manifests/"
    "third_scene_840920_dual_task_agent_cad_content_agents_bundles.v1.json"
)


def _provider_runner_module():
    path = ROOT / "scripts/adp_content_agents_provider_runner.py"
    spec = importlib.util.spec_from_file_location(
        "test_adp_content_agents_provider_runner", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = tmp_path / "content-agents"
    source.mkdir()
    material_library = (
        source
        / "apps/material_agent/data/materials/material_libs_default/materials.yaml"
    )
    material_library.parent.mkdir(parents=True)
    material_library.write_text("materials: []\n", encoding="utf-8")
    vision = source / "world_understanding/functions/models/vision_language_models.py"
    backend = source / "world_understanding/functions/models/backends/public/openai.py"
    vision.parent.mkdir(parents=True)
    backend.parent.mkdir(parents=True)
    vision.write_text(
        "from world_understanding.telemetry import traced_vlm\n\n"
        "class OpenAIVLM(object):\n"
        + "    def call(self, temperature=None):\n"
        + "        if temperature is not None:\n"
        + "            pass\n" * 1
        + "    def acall(self, temperature=None):\n"
        + "        if temperature is not None:\n"
        + "            pass\n"
        + "    def paired(self, temperature=None):\n"
        + "        if temperature is not None:\n"
        + "            pass\n\n"
        + "class AnthropicVLM(object):\n"
        + "    pass\n",
        encoding="utf-8",
    )
    backend.write_text(
        "from world_understanding.functions.models.backends.registry import (\n"
        "    register_chat_backend,\n"
        ")\n"
        "_DEFAULT_OPENAI_MODEL = \"gpt-5.4\"\n"
        "def create_openai_chat(model=None, temperature=None):\n"
        "    chat_kwargs = {}\n"
        "    if temperature is not None:\n"
        "        chat_kwargs[\"temperature\"] = temperature\n"
        "    return chat_kwargs\n",
        encoding="utf-8",
    )

    def fake_git(_repo: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return content_agents.SOURCE_COMMIT
        if args == ("rev-parse", "HEAD^{tree}"):
            return content_agents.SOURCE_TREE
        if args == ("status", "--porcelain"):
            return ""
        raise AssertionError(args)

    original_run = content_agents.subprocess.run

    def fake_run(command, **kwargs):
        if command[:4] == ["git", "-C", str(source), "archive"]:
            output = next(item.removeprefix("--output=") for item in command if item.startswith("--output="))
            with zipfile.ZipFile(output, "w") as archive:
                for path in sorted(item for item in source.rglob("*") if item.is_file()):
                    info = zipfile.ZipInfo(
                        path.relative_to(source).as_posix(),
                        date_time=(1980, 1, 1, 0, 0, 0),
                    )
                    archive.writestr(info, path.read_bytes())
            return None
        return original_run(command, **kwargs)

    monkeypatch.setattr(content_agents, "_git", fake_git)
    monkeypatch.setattr(content_agents.subprocess, "run", fake_run)
    return source


def _reference_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "reference.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nblueprint-owned-test")
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(content_agents, "REFERENCE_IMAGE_SHA256", digest)
    return path


def _write_receipt(path: Path, payload: dict) -> dict:
    payload = dict(payload)
    payload["receipt_digest"] = canonical_digest(payload, digest_field="receipt_digest")
    write_json(path, payload)
    return payload


def _cad_backend(
    tmp_path: Path, backend_id: str = "earthtojake_text_to_cad"
) -> dict:
    archive = tmp_path / f"{backend_id}.zip"
    with zipfile.ZipFile(archive, "w") as source:
        source.comment = b"1" * 40
        source.writestr("LICENSE", "MIT\n")
    if backend_id == "earthtojake_text_to_cad":
        execution_mode = "codex_skill_step_first"
        repository_url = "https://github.com/earthtojake/text-to-cad"
        model_id = "codex_fixture_agent"
    elif backend_id == "pan_chera_multi_agent_cad":
        execution_mode = "codex_agent_direct_repo_route"
        repository_url = "https://github.com/Pan-Chera/Multi-Agent-CAD"
        model_id = "codex_fixture_agent"
    else:
        raise AssertionError(backend_id)
    return {
        "backend_id": backend_id,
        "execution_mode": execution_mode,
        "agent_authored_geometry": True,
        "deterministic_geometry_generator_used": False,
        "graph_geometry_used_for_cad_authoring": False,
        "deterministic_format_conversion_only": True,
        "repository_url": repository_url,
        "commit": "1" * 40,
        "tree": "2" * 40,
        "source_archive": file_record(archive),
        "license": "MIT",
        "model_id": model_id,
    }


def _agent_cad_evidence(
    tmp_path: Path,
    backend_id: str = "earthtojake_text_to_cad",
    *,
    replacement_slot: int = 1,
    task_id: str = "task_a_washer_door_open",
    asset_id: str = "840920_simready_washer_candidate",
) -> tuple[Path, Path, Path, Path]:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    root = tmp_path / "agent-cad"
    root.mkdir(parents=True)
    brief = root / "brief.md"
    brief.write_text("agent CAD brief\n", encoding="utf-8")
    reference = root / "reference.png"
    reference.write_bytes(b"\x89PNG\r\n\x1a\nagent-cad-reference")
    reference_manifest = seal_cad_agent_reference_manifest(
        scene_id="840920",
        objects=[
            {
                "replacement_slot": replacement_slot,
                "task_id": task_id,
                "asset_id": asset_id,
                "task_freeze_path": TASK_A_FREEZE,
                "reference_image_paths": [reference],
            }
        ],
    )
    reference_manifest_path = root / "reference_manifest.json"
    write_json(reference_manifest_path, reference_manifest)
    request = seal_cad_agent_request(
        request_id=f"agent-cad-fixture-request-{backend_id}",
        scene_id="840920",
        task_id=task_id,
        asset_id=asset_id,
        replacement_slot=replacement_slot,
        backend=_cad_backend(root, backend_id),
        task_freeze_path=TASK_A_FREEZE,
        cad_brief_path=brief,
        metric_envelope_mm=[600.112, 604.104004, 847.564026],
        reference_manifest_path=reference_manifest_path,
    )
    generator = root / "agent_source.py"
    generator.write_text("def build(): return 'agent-authored'\n", encoding="utf-8")
    step = root / "candidate.step"
    step.write_bytes(f"agent-authored-step-{backend_id}".encode("utf-8"))
    inspection_path = root / "inspection.json"
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
                ROOT / "src/blueprint_pipeline/simready_cad_agent_contract.py"
            ),
        },
        "claim_boundary": {
            "step_bytes_reopened": True,
            "metric_envelope_measured": True,
            "visual_quality_qualified": False,
            "simready_qualified": False,
            "physical_equivalence": False,
        },
    }
    inspection["receipt_digest"] = canonical_digest(
        inspection, digest_field="receipt_digest"
    )
    write_json(inspection_path, inspection)
    execution_path = root / "execution.json"
    execution = seal_cad_agent_execution_receipt(
        request=request,
        generator_source_path=generator,
        cad_brief_path=brief,
        output_step_path=step,
        event_rows=[{"event": "agent_authored_step", "status": "passed"}],
    )
    write_json(execution_path, execution)
    snapshot = root / "snapshot.png"
    snapshot.write_bytes(b"\x89PNG\r\n\x1a\nsnapshot")
    output = seal_cad_agent_output(
        request=request,
        generator_source_path=generator,
        step_path=step,
        inspection_receipt_path=inspection_path,
        snapshot_paths=[snapshot],
        execution_receipt_path=execution_path,
        measured_envelope_mm=[600.112, 604.104004, 847.564026],
        actual_cost_usd=0.0 if backend_id == "earthtojake_text_to_cad" else 1.0,
    )
    output_path = root / "cad_agent_output.json"
    write_json(output_path, output)

    usd_path = root / "agent_input.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    material = UsdShade.Material.Define(stage, "/Asset/materials/agent_input_neutral")
    shader = UsdShade.Shader.Define(
        stage, "/Asset/materials/agent_input_neutral/PreviewSurface"
    )
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.5, 0.5, 0.5)
    )
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    for prim_path in (
        "/Asset/links/body/geometry/shell",
        "/Asset/links/door/geometry/panel",
    ):
        mesh = UsdGeom.Mesh.Define(stage, prim_path)
        mesh.CreatePointsAttr(
            [Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)]
        )
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreatePurposeAttr(UsdGeom.Tokens.default_)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    projection = {
        "schema_version": "cad_agent_mesh_usd_projection.v1",
        "status": "mesh_working_copy_authored",
        "packet": {"path": str((root / "packet.json").resolve()), "size_bytes": 1, "sha256": "sha256:" + "a" * 64},
        "packet_digest": "sha256:" + "b" * 64,
        "step": file_record(step),
        "output_usd": file_record(usd_path),
        "mesh_prim_paths": [
            "/Asset/links/body/geometry/shell",
            "/Asset/links/door/geometry/panel",
        ],
        "mesh_count": 2,
        "point_count": 6,
        "triangle_count": 2,
        "default_material_path": "/Asset/materials/agent_input_neutral",
        "content_agents_input_eligible": True,
        "canonical_simulator_asset": False,
        "claim_boundary": {
            "deterministic_format_conversion_only": True,
            "cad_authored_by_projection": False,
            "collision_authority": False,
            "physics_authority": False,
            "native_simulator_import_qualified": False,
            "physical_equivalence": False,
        },
    }
    projection["receipt_digest"] = canonical_digest(
        projection, digest_field="receipt_digest"
    )
    projection_path = root / "projection_receipt.json"
    write_json(projection_path, projection)
    return output_path, projection_path, reference, usd_path


def _qualified_model_access() -> dict:
    llm = {
        "schema_version": bundle_preflight.LLM_PREFLIGHT_SCHEMA_VERSION,
        "status": "qualified",
        "backend": "openai",
        "model": content_agents.CONTENT_LLM_MODEL,
        "reasoning_effort": content_agents.CONTENT_LLM_REASONING_EFFORT,
        "probe_profile": bundle_preflight.LLM_PROBE_PROFILE,
        "verified_capabilities": list(bundle_preflight.LLM_REQUIRED_CAPABILITIES),
        "blockers": [],
        "receipt_digest": "",
    }
    llm["receipt_digest"] = canonical_digest(llm, digest_field="receipt_digest")
    image = {
        "schema_version": bundle_preflight.IMAGE_PREFLIGHT_SCHEMA_VERSION,
        "status": "qualified",
        "provider": "openai",
        "model": content_agents.CONTENT_IMAGE_MODEL,
        "output": {
            "width": 1024,
            "height": 1024,
            "bytes_retained": False,
        },
        "uploaded_scene_bytes": False,
        "blockers": [],
        "receipt_digest": "",
    }
    image["receipt_digest"] = canonical_digest(image, digest_field="receipt_digest")
    return {
        "provider": "openai",
        "models": {
            content_agents.CONTENT_LLM_MODEL: llm,
            content_agents.CONTENT_IMAGE_MODEL: image,
        },
        "paid_inference_performed": True,
        "uploaded_scene_bytes": False,
    }


def _match_v2_evidence(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    evidence = tmp_path / "evidence"
    usd = repo / "docs/arm_decision_proof_v1/assets/match-v2.usda"
    snapshot = evidence / "simready/cad_match_v2/snapshot.png"
    usd.parent.mkdir(parents=True)
    snapshot.parent.mkdir(parents=True)
    usd.write_text("#usda 1.0\n", encoding="utf-8")
    snapshot.write_bytes(b"\x89PNG\r\n\x1a\nmatch-v2")
    control = _write_receipt(
        repo / content_agents.MATCH_V2_RECEIPT_RELATIVE_PATH,
        {
            "control_id": "adp009a-840313-canned-beverage-multiview-match-v2",
            "status": "prepared_for_independent_validation",
            "checks": {
                "cad_inspection_passed": True,
                "target_dimensions_derived_not_caller_asserted": True,
            },
            "visual_match_evidence": {"projected_scale_and_pose_gate_passed": True},
            "usd": {
                "relative_path": usd.relative_to(repo).as_posix(),
                "sha256": "sha256:" + hashlib.sha256(usd.read_bytes()).hexdigest(),
            },
            "cad_evidence": {
                "snapshot": {
                    "relative_path": snapshot.relative_to(evidence).as_posix(),
                    "sha256": "sha256:"
                    + hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                }
            },
        },
    )
    replacement = _write_receipt(
        repo / content_agents.MATCH_V2_REPLACEMENT_RECEIPT_RELATIVE_PATH,
        {
            "status": "composed_static_candidate",
            "bindings": {"simready_control_receipt_digest": control["receipt_digest"]},
        },
    )
    _write_receipt(
        repo / content_agents.MATCH_V2_HUMAN_REVIEW_RELATIVE_PATH,
        {
            "status": "human_accepted_for_native_validation",
            "technical_admission": False,
            "artifact_chain": {
                "replacement_receipt_digest": replacement["receipt_digest"]
            },
        },
    )
    return repo, evidence, snapshot


def test_match_v2_variant_binds_approved_receipt_chain(tmp_path: Path) -> None:
    repo, evidence, snapshot = _match_v2_evidence(tmp_path)

    resolved = content_agents._resolve_input_variant(
        repo=repo,
        evidence_root=evidence,
        reference_source=snapshot,
        variant="match_v2",
    )

    assert resolved["variant"] == "match_v2"
    assert resolved["control_receipt_digest"].startswith("sha256:")
    assert resolved["replacement_receipt_digest"].startswith("sha256:")
    assert resolved["human_review_receipt_digest"].startswith("sha256:")
    assert "interiorgs_dataset_bytes" in resolved["reference_image_authority"]


def test_match_v2_variant_rejects_changed_approved_snapshot(tmp_path: Path) -> None:
    repo, evidence, snapshot = _match_v2_evidence(tmp_path)
    snapshot.write_bytes(snapshot.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="source_identity_mismatch"):
        content_agents._resolve_input_variant(
            repo=repo,
            evidence_root=evidence,
            reference_source=snapshot,
            variant="match_v2",
        )


def test_match_v2_configs_remove_unproven_aluminum_assertion(tmp_path: Path) -> None:
    assets = ROOT / "docs" / "arm_decision_proof_v1" / "assets"
    sources = {
        "material_agent.yaml": assets / "adp009a_content_agents_material.vast.yaml",
        "texture_agent.yaml": assets / "adp009a_content_agents_texture.vast.yaml",
        "physics_agent.yaml": assets / "adp009a_content_agents_physics.vast.yaml",
    }
    destination = tmp_path / "configs"
    destination.mkdir()

    hashes = content_agents._materialize_remote_configs(
        config_sources=sources, destination=destination, variant="match_v2"
    )

    material = (destination / "material_agent.yaml").read_text(encoding="utf-8")
    texture = (destination / "texture_agent.yaml").read_text(encoding="utf-8")
    assert set(hashes) == set(sources)
    assert "Do not assume aluminum" in material
    assert "bright green aluminum" not in texture
    assert "pale mint green" in texture


def test_legacy_content_agents_input_is_not_selectable_for_new_authoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    reference = _reference_image(tmp_path, monkeypatch)
    with pytest.raises(
        ValueError,
        match="deterministic_cad_authoring_removed_use_agent_backend",
    ):
        content_agents.build_content_agents_vast_bundle(
            repo_root=ROOT,
            content_agents_root=source,
            reference_image_path=reference,
            job_dir=tmp_path / "rejected",
        )


def test_agent_cad_variant_binds_agent_output_and_mesh_projection(
    tmp_path: Path,
) -> None:
    output_path, projection_path, reference, usd = _agent_cad_evidence(tmp_path)

    resolved = content_agents._resolve_input_variant(
        repo=ROOT,
        evidence_root=None,
        reference_source=reference,
        variant="agent_cad_v1",
        agent_cad_output_manifest_path=output_path,
        agent_mesh_projection_receipt_path=projection_path,
    )

    assert resolved["variant"] == "agent_cad_v1"
    assert resolved["usd_source"] == usd
    assert resolved["cad_agent_backend_id"] == "earthtojake_text_to_cad"
    assert resolved["task_id"] == "task_a_washer_door_open"
    assert resolved["replacement_slot"] == 1
    assert resolved["mesh_prim_paths"] == [
        "/Asset/links/body/geometry/shell",
        "/Asset/links/door/geometry/panel",
    ]
    assert resolved["candidate_step_sha256"].startswith("sha256:")


def test_agent_cad_bundle_uses_mesh_only_input_without_historical_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    output_path, projection_path, _reference, _usd = _agent_cad_evidence(tmp_path)

    receipt = content_agents.build_content_agents_vast_bundle(
        repo_root=ROOT,
        content_agents_root=source,
        job_dir=tmp_path / "agent-cad-bundle",
        input_variant="agent_cad_v1",
        agent_cad_output_manifest_path=output_path,
        agent_mesh_projection_receipt_path=projection_path,
        generated_at="fixed",
    )

    assert receipt["status"] == "ready"
    assert receipt["input_variant"] == "agent_cad_v1"
    assert receipt["input_usd_normalization"]["agent_cad_mesh_working_copy"] is True
    assert receipt["input_usd_normalization"]["mesh_count"] == 2
    assert receipt["joint_agent_plan"]["reason"] == (
        "agent_cad_mesh_working_copy_has_no_articulation_task"
    )
    assert (
        receipt["joint_agent_plan"]["joint_agent_inapplicable_single_rigid_body"]
        is False
    )
    assert receipt["agent_output_is_simready_authority"] is False
    assert receipt["canonical_simready_construction_unresolved"] is True
    assert receipt["deterministic_usd_construction_remains_primary"] is False
    runtime_configs = Path(receipt["bundle_path"]).with_name(
        "provider_runtime"
    ) / "configs"
    texture = yaml.safe_load(
        (runtime_configs / "texture_agent.yaml").read_text(encoding="utf-8")
    )
    assert texture["texture"]["uv_target_prim_paths"] == [
        "/Asset/links/body/geometry/shell",
        "/Asset/links/door/geometry/panel",
    ]
    assert texture["material_textures"]["agent_cad_visible_surfaces"][
        "material_path"
    ] == "/Asset/materials/agent_input_neutral"


def _single_agent_cad_content_bundle_matrix(
    *,
    bundle_receipt_path: Path,
    bundle_receipt: Mapping[str, object],
    cad_output_path: Path,
    projection_path: Path,
) -> dict[str, object]:
    cad_output = json.loads(cad_output_path.read_text(encoding="utf-8"))
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    item = {
        "replacement_slot": cad_output["request"]["replacement_slot"],
        "task_id": cad_output["request"]["task_id"],
        "asset_id": cad_output["request"]["asset_id"],
        "cad_agent_backend_id": cad_output["request"]["backend"]["backend_id"],
        "cad_agent_output_receipt_digest": cad_output["receipt_digest"],
        "mesh_projection_receipt_digest": projection["receipt_digest"],
        "mesh_packet_digest": projection["packet_digest"],
        "candidate_step_sha256": cad_output["artifacts"]["step"]["sha256"],
        "bundle": {
            "path": bundle_receipt["bundle_path"],
            "sha256": bundle_receipt["bundle_sha256"],
            "size_bytes": bundle_receipt["bundle_size_bytes"],
        },
        "bundle_receipt": {
            "path": str(bundle_receipt_path),
            "sha256": "sha256:"
            + hashlib.sha256(bundle_receipt_path.read_bytes()).hexdigest(),
            "size_bytes": bundle_receipt_path.stat().st_size,
        },
        "blockers": [],
        "exact_bundle_entrypoint_rehearsal_status": "passed",
        "agent_output_is_simready_authority": False,
        "canonical_simready_construction_unresolved": True,
    }
    second_item = json.loads(json.dumps(item))
    second_item["cad_agent_backend_id"] = "pan_chera_multi_agent_cad"
    matrix = {
        "schema_version": "third_scene_agent_cad_content_agents_bundle_matrix.v1",
        "status": "local_bundles_ready_for_paid_resource_preflight",
        "input_variant": "agent_cad_v1",
        "replacement_object_capacity": {
            "minimum": 1,
            "maximum": 5,
            "sealed_slots": 1,
        },
        "candidate_count": 2,
        "items": [item, second_item],
        "claim_boundary": {
            "content_agents_bundles_built": True,
            "exact_entrypoint_rehearsed": True,
            "content_agents_executed": False,
            "simready_qualified": False,
            "native_simulator_import_qualified": False,
            "physical_equivalence": False,
        },
        "receipt_digest": "",
    }
    matrix["receipt_digest"] = canonical_digest(matrix, digest_field="receipt_digest")
    return matrix


def _agent_cad_bundle_item(
    *,
    tmp_path: Path,
    source: Path,
    backend_id: str,
    replacement_slot: int = 1,
    job_name: str | None = None,
) -> dict[str, object]:
    output_path, projection_path, _reference, _usd = _agent_cad_evidence(
        tmp_path / backend_id / f"slot-{replacement_slot}",
        backend_id,
        replacement_slot=replacement_slot,
    )
    bundle_receipt = content_agents.build_content_agents_vast_bundle(
        repo_root=ROOT,
        content_agents_root=source,
        job_dir=tmp_path / (job_name or f"bundle-{replacement_slot}-{backend_id}"),
        input_variant="agent_cad_v1",
        agent_cad_output_manifest_path=output_path,
        agent_mesh_projection_receipt_path=projection_path,
        generated_at="fixed",
    )
    bundle_receipt_path = Path(bundle_receipt["bundle_path"]).with_name(
        "adp_content_agents_bundle_receipt.json"
    )
    matrix = _single_agent_cad_content_bundle_matrix(
        bundle_receipt_path=bundle_receipt_path,
        bundle_receipt=bundle_receipt,
        cad_output_path=output_path,
        projection_path=projection_path,
    )
    return dict(matrix["items"][0])


def _agent_cad_content_bundle_matrix(
    *,
    tmp_path: Path,
    source: Path,
    replacement_slots: int = 1,
) -> dict[str, object]:
    items = [
        _agent_cad_bundle_item(
            tmp_path=tmp_path,
            source=source,
            backend_id=backend_id,
            replacement_slot=slot,
        )
        for slot in range(1, replacement_slots + 1)
        for backend_id in ("earthtojake_text_to_cad", "pan_chera_multi_agent_cad")
    ]
    matrix = {
        "schema_version": "third_scene_agent_cad_content_agents_bundle_matrix.v1",
        "status": "local_bundles_ready_for_paid_resource_preflight",
        "input_variant": "agent_cad_v1",
        "replacement_object_capacity": {
            "minimum": 1,
            "maximum": 5,
            "sealed_slots": replacement_slots,
        },
        "candidate_count": len(items),
        "items": items,
        "claim_boundary": {
            "content_agents_bundles_built": True,
            "exact_entrypoint_rehearsed": True,
            "content_agents_executed": False,
            "simready_qualified": False,
            "native_simulator_import_qualified": False,
            "physical_equivalence": False,
        },
        "receipt_digest": "",
    }
    matrix["receipt_digest"] = canonical_digest(matrix, digest_field="receipt_digest")
    return matrix


def _bundle_receipt_from_item(item: Mapping[str, object]) -> tuple[Path, dict]:
    record = item["bundle_receipt"]
    assert isinstance(record, Mapping)
    path = Path(str(record["path"]))
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_content_agents_execution_readiness_records_missing_authority_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    matrix = _agent_cad_content_bundle_matrix(tmp_path=tmp_path, source=source)

    readiness = content_agents.materialize_content_agents_execution_readiness(
        content_agents_bundle_matrix=matrix,
        output_path=tmp_path / "readiness.json",
        generated_at="fixed",
    )

    assert readiness["schema_version"] == content_agents.EXECUTION_READINESS_SCHEMA
    assert readiness["status"] == "blocked_before_paid_execution"
    assert readiness["provider_mutations_performed"] == 0
    assert readiness["paid_resource_allocated"] is False
    assert readiness["items"][0]["local_bundle_ready"] is True
    assert readiness["items"][0]["execute_admitted"] is False
    assert readiness["items"][0]["blockers"] == [
        "content_agents_local_docker_config_preflight_missing",
        "content_agents_paid_attempt_authority_missing",
        "content_agents_paid_model_access_preflight_missing",
        "content_agents_static_config_preflight_missing",
    ]
    assert readiness["receipt_digest"] == canonical_digest(
        readiness, digest_field="receipt_digest"
    )


def test_content_agents_execution_readiness_accepts_local_no_paid_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    matrix = _agent_cad_content_bundle_matrix(tmp_path=tmp_path, source=source)
    bundle_receipt_path, bundle_receipt = _bundle_receipt_from_item(
        matrix["items"][0]
    )
    local_preflight = _passing_local_config_preflight(
        tmp_path / "local-preflight",
        bundle_receipt_path=bundle_receipt_path,
        bundle_receipt=bundle_receipt,
    )
    key = "task_a_washer_door_open|1|earthtojake_text_to_cad"

    readiness = content_agents.materialize_content_agents_execution_readiness(
        content_agents_bundle_matrix=matrix,
        output_path=tmp_path / "readiness.json",
        config_preflight_receipts={key: local_preflight},
        generated_at="fixed",
    )

    row = readiness["items"][0]
    assert row["config_preflight"] is None
    assert row["local_config_preflight"]["receipt_digest"] == json.loads(
        local_preflight.read_text(encoding="utf-8")
    )["receipt_digest"]
    assert row["blockers"] == [
        "content_agents_paid_attempt_authority_missing",
        "content_agents_paid_model_access_preflight_missing",
    ]


def test_content_agents_execution_readiness_accepts_static_no_docker_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    matrix = _agent_cad_content_bundle_matrix(tmp_path=tmp_path, source=source)
    bundle_receipt_path, bundle_receipt = _bundle_receipt_from_item(
        matrix["items"][0]
    )
    static_preflight = _passing_static_config_preflight(
        tmp_path / "static-preflight",
        bundle_receipt_path=bundle_receipt_path,
        bundle_receipt=bundle_receipt,
    )
    key = "task_a_washer_door_open|1|earthtojake_text_to_cad"

    readiness = content_agents.materialize_content_agents_execution_readiness(
        content_agents_bundle_matrix=matrix,
        output_path=tmp_path / "readiness.json",
        config_preflight_receipts={key: static_preflight},
        generated_at="fixed",
    )

    row = readiness["items"][0]
    assert row["config_preflight"] is None
    assert row["local_config_preflight"] is None
    assert row["static_config_preflight"]["receipt_digest"] == json.loads(
        static_preflight.read_text(encoding="utf-8")
    )["receipt_digest"]
    assert row["blockers"] == [
        "content_agents_local_docker_config_preflight_missing",
        "content_agents_paid_attempt_authority_missing",
        "content_agents_paid_model_access_preflight_missing",
    ]


def test_content_agents_execution_readiness_retains_static_and_local_preflights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    matrix = _agent_cad_content_bundle_matrix(tmp_path=tmp_path, source=source)
    bundle_receipt_path, bundle_receipt = _bundle_receipt_from_item(
        matrix["items"][0]
    )
    static_preflight = _passing_static_config_preflight(
        tmp_path / "static-preflight",
        bundle_receipt_path=bundle_receipt_path,
        bundle_receipt=bundle_receipt,
    )
    local_preflight = _passing_local_config_preflight(
        tmp_path / "local-preflight",
        bundle_receipt_path=bundle_receipt_path,
        bundle_receipt=bundle_receipt,
    )
    key = "task_a_washer_door_open|1|earthtojake_text_to_cad"

    readiness = content_agents.materialize_content_agents_execution_readiness(
        content_agents_bundle_matrix=matrix,
        output_path=tmp_path / "readiness.json",
        config_preflight_receipts={key: [static_preflight, local_preflight]},
        generated_at="fixed",
    )

    row = readiness["items"][0]
    assert row["config_preflight"] is None
    assert row["static_config_preflight"]["receipt_digest"] == json.loads(
        static_preflight.read_text(encoding="utf-8")
    )["receipt_digest"]
    assert row["local_config_preflight"]["receipt_digest"] == json.loads(
        local_preflight.read_text(encoding="utf-8")
    )["receipt_digest"]
    assert row["blockers"] == [
        "content_agents_paid_attempt_authority_missing",
        "content_agents_paid_model_access_preflight_missing",
    ]


def test_content_agents_execution_readiness_accepts_five_replacement_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    matrix = _agent_cad_content_bundle_matrix(
        tmp_path=tmp_path,
        source=source,
        replacement_slots=5,
    )

    readiness = content_agents.materialize_content_agents_execution_readiness(
        content_agents_bundle_matrix=matrix,
        output_path=tmp_path / "readiness.json",
        generated_at="fixed",
    )

    assert readiness["candidate_count"] == 10
    assert readiness["replacement_object_capacity"] == {
        "minimum": 1,
        "maximum": 5,
        "sealed_slots": 5,
    }
    assert {
        (row["replacement_slot"], row["cad_agent_backend_id"])
        for row in readiness["items"]
    } == {
        (slot, backend_id)
        for slot in range(1, 6)
        for backend_id in ("earthtojake_text_to_cad", "pan_chera_multi_agent_cad")
    }


def test_content_agents_execution_readiness_rejects_duplicate_or_partial_backend_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    output_path, projection_path, _reference, _usd = _agent_cad_evidence(tmp_path)
    bundle_receipt = content_agents.build_content_agents_vast_bundle(
        repo_root=ROOT,
        content_agents_root=source,
        job_dir=tmp_path / "agent-cad-bundle",
        input_variant="agent_cad_v1",
        agent_cad_output_manifest_path=output_path,
        agent_mesh_projection_receipt_path=projection_path,
        generated_at="fixed",
    )
    bundle_receipt_path = Path(bundle_receipt["bundle_path"]).with_name(
        "adp_content_agents_bundle_receipt.json"
    )
    matrix = _single_agent_cad_content_bundle_matrix(
        bundle_receipt_path=bundle_receipt_path,
        bundle_receipt=bundle_receipt,
        cad_output_path=output_path,
        projection_path=projection_path,
    )
    first = json.loads(json.dumps(matrix["items"][0]))
    second = json.loads(json.dumps(matrix["items"][0]))
    matrix["items"] = [first, second]
    matrix["candidate_count"] = 2
    matrix["receipt_digest"] = canonical_digest(matrix, digest_field="receipt_digest")

    with pytest.raises(
        ValueError, match="adp_content_agents_readiness_matrix_item_identity_invalid"
    ):
        content_agents.materialize_content_agents_execution_readiness(
            content_agents_bundle_matrix=matrix,
            output_path=tmp_path / "readiness.json",
            generated_at="fixed",
        )

    second["cad_agent_backend_id"] = "pan_chera_multi_agent_cad"
    second["asset_id"] = "different_asset_same_slot"
    matrix["items"] = [first, second]
    matrix["receipt_digest"] = canonical_digest(matrix, digest_field="receipt_digest")

    with pytest.raises(
        ValueError, match="adp_content_agents_readiness_matrix_item_identity_invalid"
    ):
        content_agents.materialize_content_agents_execution_readiness(
            content_agents_bundle_matrix=matrix,
            output_path=tmp_path / "readiness.json",
            generated_at="fixed",
        )


def test_content_agents_execution_readiness_rejects_relabelled_bundle_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    output_path, projection_path, _reference, _usd = _agent_cad_evidence(tmp_path)
    bundle_receipt = content_agents.build_content_agents_vast_bundle(
        repo_root=ROOT,
        content_agents_root=source,
        job_dir=tmp_path / "agent-cad-bundle",
        input_variant="agent_cad_v1",
        agent_cad_output_manifest_path=output_path,
        agent_mesh_projection_receipt_path=projection_path,
        generated_at="fixed",
    )
    bundle_receipt_path = Path(bundle_receipt["bundle_path"]).with_name(
        "adp_content_agents_bundle_receipt.json"
    )
    matrix = _single_agent_cad_content_bundle_matrix(
        bundle_receipt_path=bundle_receipt_path,
        bundle_receipt=bundle_receipt,
        cad_output_path=output_path,
        projection_path=projection_path,
    )

    with pytest.raises(
        ValueError, match="adp_content_agents_readiness_bundle_binding_mismatch"
    ):
        content_agents.materialize_content_agents_execution_readiness(
            content_agents_bundle_matrix=matrix,
            output_path=tmp_path / "readiness.json",
            generated_at="fixed",
        )


def test_content_agents_execution_readiness_rejects_stale_bound_manifest_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    matrix = _agent_cad_content_bundle_matrix(tmp_path=tmp_path, source=source)
    first_receipt_path, _first_receipt = _bundle_receipt_from_item(matrix["items"][0])
    first_receipt = json.loads(first_receipt_path.read_text(encoding="utf-8"))
    bound_output = Path(
        first_receipt["input_variant_bindings"]["cad_agent_output_manifest"]["path"]
    )
    bound_output.write_text(
        bound_output.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="adp_content_agents_readiness_bundle_binding_file_mismatch",
    ):
        content_agents.materialize_content_agents_execution_readiness(
            content_agents_bundle_matrix=matrix,
            output_path=tmp_path / "readiness.json",
            generated_at="fixed",
        )


def test_content_agents_execution_readiness_rejects_tampered_bundle_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    output_path, projection_path, _reference, _usd = _agent_cad_evidence(tmp_path)
    bundle_receipt = content_agents.build_content_agents_vast_bundle(
        repo_root=ROOT,
        content_agents_root=source,
        job_dir=tmp_path / "agent-cad-bundle",
        input_variant="agent_cad_v1",
        agent_cad_output_manifest_path=output_path,
        agent_mesh_projection_receipt_path=projection_path,
        generated_at="fixed",
    )
    bundle_receipt_path = Path(bundle_receipt["bundle_path"]).with_name(
        "adp_content_agents_bundle_receipt.json"
    )
    matrix = _single_agent_cad_content_bundle_matrix(
        bundle_receipt_path=bundle_receipt_path,
        bundle_receipt=bundle_receipt,
        cad_output_path=output_path,
        projection_path=projection_path,
    )
    matrix["items"][0]["bundle"]["sha256"] = "sha256:" + "0" * 64
    matrix["receipt_digest"] = canonical_digest(matrix, digest_field="receipt_digest")

    with pytest.raises(ValueError, match="readiness_bundle_bytes_invalid"):
        content_agents.materialize_content_agents_execution_readiness(
            content_agents_bundle_matrix=matrix,
            output_path=tmp_path / "readiness.json",
            generated_at="fixed",
        )


def test_agent_cad_reference_is_derived_from_manifest_not_operator_guess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    output_path, projection_path, _reference, _usd = _agent_cad_evidence(tmp_path)

    receipt = content_agents.build_content_agents_vast_bundle(
        repo_root=ROOT,
        content_agents_root=source,
        job_dir=tmp_path / "agent-cad-derived-reference",
        input_variant="agent_cad_v1",
        agent_cad_output_manifest_path=output_path,
        agent_mesh_projection_receipt_path=projection_path,
        generated_at="fixed",
    )

    output = json.loads(output_path.read_text(encoding="utf-8"))
    first_reference = output["request"]["inputs"]["reference_images"][0]
    assert receipt["reference_image_sha256"] == first_reference["sha256"]


def test_agent_cad_bundle_rejects_manual_reference_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    output_path, projection_path, reference, _usd = _agent_cad_evidence(tmp_path)

    with pytest.raises(
        ValueError,
        match="adp_content_agents_agent_cad_reference_must_come_from_manifest",
    ):
        content_agents.build_content_agents_vast_bundle(
            repo_root=ROOT,
            content_agents_root=source,
            job_dir=tmp_path / "agent-cad-manual-reference-rejected",
            input_variant="agent_cad_v1",
            reference_image_path=reference,
            agent_cad_output_manifest_path=output_path,
            agent_mesh_projection_receipt_path=projection_path,
            generated_at="fixed",
        )


def test_agent_cad_variant_rejects_changed_projection_usd(
    tmp_path: Path,
) -> None:
    output_path, projection_path, reference, usd = _agent_cad_evidence(tmp_path)
    usd.write_text("#usda 1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source_identity_mismatch"):
        content_agents._resolve_input_variant(
            repo=ROOT,
            evidence_root=None,
            reference_source=reference,
            variant="agent_cad_v1",
            agent_cad_output_manifest_path=output_path,
            agent_mesh_projection_receipt_path=projection_path,
        )


def test_checked_in_agent_cad_content_agents_bundle_matrix_is_claim_bounded() -> None:
    manifest = json.loads(AGENT_CAD_CONTENT_BUNDLE_MATRIX.read_text(encoding="utf-8"))

    assert manifest["receipt_digest"] == canonical_digest(
        manifest, digest_field="receipt_digest"
    )
    assert manifest["schema_version"] == (
        "third_scene_agent_cad_content_agents_bundle_matrix.v1"
    )
    assert manifest["replacement_object_capacity"] == {
        "minimum": 1,
        "maximum": 5,
        "sealed_slots": 2,
    }
    assert manifest["candidate_count"] == 4
    assert {
        (row["replacement_slot"], row["cad_agent_backend_id"])
        for row in manifest["items"]
    } == {
        (1, "earthtojake_text_to_cad"),
        (1, "pan_chera_multi_agent_cad"),
        (2, "earthtojake_text_to_cad"),
        (2, "pan_chera_multi_agent_cad"),
    }
    assert all(
        row["exact_bundle_entrypoint_rehearsal_status"] == "passed"
        and row["blockers"] == []
        and row["agent_output_is_simready_authority"] is False
        and row["canonical_simready_construction_unresolved"] is True
        for row in manifest["items"]
    )
    assert manifest["claim_boundary"] == {
        "content_agents_bundles_built": True,
        "exact_entrypoint_rehearsed": True,
        "content_agents_executed": False,
        "simready_qualified": False,
        "native_simulator_import_qualified": False,
        "physical_equivalence": False,
    }


def test_bundle_is_deterministic_and_provider_preflight_accepts_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    reference = _reference_image(tmp_path, monkeypatch)
    first = content_agents.build_content_agents_vast_bundle(
        repo_root=ROOT,
        content_agents_root=source,
        reference_image_path=reference,
        job_dir=tmp_path / "first",
        generated_at="fixed",
        historical_replay_only=True,
    )
    second = content_agents.build_content_agents_vast_bundle(
        repo_root=ROOT,
        content_agents_root=source,
        reference_image_path=reference,
        job_dir=tmp_path / "second",
        generated_at="fixed",
        historical_replay_only=True,
    )

    assert first["blockers"] == []
    assert first["status"] == "ready"
    assert first["bundle_sha256"] == second["bundle_sha256"]
    assert first["container_image"] == content_agents.DEFAULT_IMAGE
    assert first["input_usd_normalization"]["default_purpose_bbox_nonempty"] is True
    assert (
        first["input_usd_normalization"]["source_input_usd_sha256"]
        != first["input_usd_sha256"]
    )
    assert first["remote_config_contract_validated"] is True
    assert first["retry_cap"] == 0
    assert first["execution_role"] == "optional_construction_enrichment"
    assert first["failure_blocks_native_simulator_qualification"] is False
    assert first["agent_output_is_simready_authority"] is False
    assert first["runtime_input_binding"]["relative_path"] == (
        "input/source_asset.usda"
    )
    assert first["joint_agent_plan"] == {
        "planned": False,
        "executed_by_content_agents_bundle": False,
        "reason": "single_rigid_body_has_no_articulation_task",
        "input_variant": "control_v1",
        "input_joint_count": 0,
        "input_rigid_body_count": 1,
        "input_articulation_root_count": 0,
        "joint_agent_inapplicable_single_rigid_body": True,
    }
    preflight = _blueprint_bundle_preflight(
        job_dir=tmp_path / "preflight",
        generated_at="fixed",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp_content_agents",
        bundle_path=Path(first["bundle_path"]),
        provider_bundle_url="https://example.com/bundle.zip?signature=redacted",
        provider_output_put_url="https://example.com/output.zip?signature=redacted",
    )
    assert preflight["status"] == "passed"
    assert preflight["blockers"] == []


def test_bundle_rejects_changed_reference_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    reference = _reference_image(tmp_path, monkeypatch)
    reference.write_bytes(reference.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="reference_image_identity_mismatch"):
        content_agents.build_content_agents_vast_bundle(
            repo_root=ROOT,
            content_agents_root=source,
            reference_image_path=reference,
            job_dir=tmp_path / "job",
            historical_replay_only=True,
        )


@pytest.mark.parametrize(
    ("filename", "before", "after"),
    [
        ("material_agent.yaml", "on_failure: warn", "on_failure: fail"),
        ("texture_agent.yaml", "model: gpt-image-2", "model: unavailable-image"),
        (
            "physics_agent.yaml",
            "    enabled: false\n    vlm:\n      backend: openai",
            "    enabled: true\n    vlm:\n      backend: openai",
        ),
    ],
)
def test_remote_config_contract_rejects_known_paid_runtime_failure_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    before: str,
    after: str,
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    assets = ROOT / "docs" / "arm_decision_proof_v1" / "assets"
    config_sources = {}
    for name in ("material_agent.yaml", "texture_agent.yaml", "physics_agent.yaml"):
        source_name = "adp009a_content_agents_" + name.removesuffix("_agent.yaml") + ".vast.yaml"
        destination = tmp_path / name
        value = (assets / source_name).read_text(encoding="utf-8")
        if name == filename:
            assert before in value
            value = value.replace(before, after, 1)
        destination.write_text(value, encoding="utf-8")
        config_sources[name] = destination

    with pytest.raises(ValueError, match="remote_config_contract_invalid"):
        content_agents._validate_remote_configs(
            source=source,
            config_sources=config_sources,
        )


def test_remote_configs_freeze_successor_models_for_both_scene_fixtures() -> None:
    assets = ROOT / "docs" / "arm_decision_proof_v1" / "assets"
    for prefix in ("adp009a_content_agents_", "adp009d_content_agents_articulated_"):
        material = yaml.safe_load((assets / f"{prefix}material.vast.yaml").read_text())
        physics = yaml.safe_load((assets / f"{prefix}physics.vast.yaml").read_text())
        texture = yaml.safe_load((assets / f"{prefix}texture.vast.yaml").read_text())
        for config in (
            material["steps"]["predict"]["vlm"],
            material["steps"]["predict"]["llm"],
            physics["steps"]["identify_asset"]["vlm"],
            physics["steps"]["predict"]["vlm"],
        ):
            assert config["model"] == content_agents.CONTENT_LLM_MODEL
            assert (
                config["reasoning_effort"]
                == content_agents.CONTENT_LLM_REASONING_EFFORT
            )
        assert texture["texture"]["image_gen"] == {
            "backend": "openai",
            "model": content_agents.CONTENT_IMAGE_MODEL,
        }


def test_bundle_preflight_rejects_catalog_only_model_access() -> None:
    assert bundle_preflight._valid_model_access(
        {
            "provider": "openai",
            "models": {
                model: {"http_status": 200, "returned_id": model}
                for model in bundle_preflight.REQUIRED_MODELS
            },
            "paid_inference_performed": False,
        }
    ) is False


def test_vast_adapter_uses_gpu_rendering_and_bounded_bundle_path(tmp_path: Path) -> None:
    assert (
        _resolve_launch_mode(
            requested="auto",
            enable_isaac_smoke=False,
            enable_blueprint_bundle=True,
            provider_bundle_kind="adp_content_agents",
        )
        == "ssh_direct"
    )
    env = _probe_env(
        job_dir=tmp_path,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp_content_agents",
        forward_hf_token=False,
    )
    assert env["NVIDIA_DRIVER_CAPABILITIES"] == "all"
    assert "ACCEPT_EULA" not in env
    script = _probe_shell_script(
        "https://example.com",
        enable_isaac_smoke=False,
        enable_blueprint_bundle=True,
        provider_bundle_kind="adp_content_agents",
    )
    assert "run_adp_content_agents_provider_runtime.sh" in script
    assert "adp_content_agents_provider_runtime_output.zip" in script
    assert "apt-get install" in script


def test_provider_runtime_pins_native_dependency_closure_before_agent_execution() -> None:
    runtime = (ROOT / "scripts/run_adp_content_agents_provider_runtime.sh").read_text()
    runner = (ROOT / "scripts/adp_content_agents_provider_runner.py").read_text()

    assert 'NATIVE_OVRTX_ENV="${SOURCE_DIR}/.ovrtx_native_venv"' in runtime
    assert '"ovrtx==0.4.0.346409"' in runtime
    assert '"ovstage==0.1.0.346039"' in runtime
    assert 'm.version("ovrtx") == "0.4.0.346409"' in runtime
    assert 'm.version("ovstage") == "0.1.0.346039"' in runtime
    assert "ovrtx.__version__" not in runtime
    assert "content_agents_native_ovrtx_dependency_closure_failed" in runtime
    assert 'content_agents_source/.ovrtx_native_venv/bin/python' in runner
    assert runner.index("native, native_blockers = _native_probes(") < runner.index(
        'for name in ("material", "texture", "physics"):'
    )
    assert "skipped_after_native_probe_failure" in runner


def test_provider_output_inspector_recognizes_content_agents_result(tmp_path: Path) -> None:
    output = tmp_path / "output.zip"
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(
            "adp_content_agents_vast_result.json",
            json.dumps({"status": "blocked", "blockers": ["typed_runtime_failure"]}),
        )

    inspection = inspect_provider_runtime_output_zip(output)

    assert inspection["runtime_result_present"] is True
    assert inspection["runtime_result_status"] == "blocked"
    assert inspection["runtime_result_blockers"] == ["typed_runtime_failure"]


def _passing_config_preflight(tmp_path: Path, bundle_receipt: dict) -> Path:
    executions = {}
    for name in ("material", "texture", "physics"):
        log_path = tmp_path / f"{name}.log"
        log_path.write_text(bundle_preflight.DRY_RUN_MARKERS[name], encoding="utf-8")
        executions[name] = {
            "entrypoint": bundle_preflight.ENTRYPOINTS[name],
            "arguments": [
                "run",
                "/bundle/" + bundle_preflight.CONFIG_MEMBERS[name],
                "--dry-run",
            ],
            "secret_environment_names_passed_by_name": list(
                bundle_preflight.SECRET_ENV_NAMES
            ),
            "returncode": 0,
            "required_marker": bundle_preflight.DRY_RUN_MARKERS[name],
            "log_path": str(log_path.resolve()),
            "log_size_bytes": log_path.stat().st_size,
            "log_sha256": "sha256:"
            + hashlib.sha256(log_path.read_bytes()).hexdigest(),
        }
    validation_log = tmp_path / "material-validate.log"
    validation_log.write_text(
        bundle_preflight.MATERIAL_VALIDATE_MARKER, encoding="utf-8"
    )
    executions["material_validate_input"] = {
        "entrypoint": bundle_preflight.ENTRYPOINTS["material"],
        "arguments": [
            "run",
            "/bundle/" + bundle_preflight.CONFIG_MEMBERS["material"],
            "--only",
            "validate_input",
            "--clean",
        ],
        "secret_environment_names_passed_by_name": ["OPENAI_API_KEY"],
        "returncode": 0,
        "required_marker": bundle_preflight.MATERIAL_VALIDATE_MARKER,
        "log_path": str(validation_log.resolve()),
        "log_size_bytes": validation_log.stat().st_size,
        "log_sha256": "sha256:"
        + hashlib.sha256(validation_log.read_bytes()).hexdigest(),
    }
    bbox_log = tmp_path / "bbox.log"
    bbox_log.write_text(bundle_preflight.USD_BBOX_MARKER, encoding="utf-8")
    executions["usd_default_purpose_bbox"] = {
        "entrypoint": "python",
        "arguments": [
            "-c",
            bundle_preflight.usd_bbox_script(
                "adp009a_840313_canned_beverage_control.usda"
            ),
        ],
        "secret_environment_names_passed_by_name": [],
        "returncode": 0,
        "required_marker": bundle_preflight.USD_BBOX_MARKER,
        "log_path": str(bbox_log.resolve()),
        "log_size_bytes": bbox_log.stat().st_size,
        "log_sha256": "sha256:" + hashlib.sha256(bbox_log.read_bytes()).hexdigest(),
    }
    bundle_receipt_path = tmp_path / "receipt.json"
    receipt = {
        "schema_version": bundle_preflight.SCHEMA_VERSION,
        "generated_at": "fixed",
        "generated_by": "blueprint_pipeline.adp_content_agents_bundle_preflight",
        "orchestrator_source_identity": {
            "commit": "a" * 40,
            "tree": "b" * 40,
            "checkout_clean": True,
        },
        "status": "passed",
        "bundle_receipt_path": str(bundle_receipt_path),
        "bundle_receipt_sha256": "sha256:"
        + hashlib.sha256(bundle_receipt_path.read_bytes()).hexdigest(),
        "bundle_path": bundle_receipt["bundle_path"],
        "bundle_sha256": bundle_receipt["bundle_sha256"],
        "content_agents_source_commit": content_agents.SOURCE_COMMIT,
        "content_agents_source_tree": content_agents.SOURCE_TREE,
        "local_container_image": {
            "reference": bundle_preflight.LOCAL_IMAGE,
            "id": next(iter(bundle_preflight.LOCAL_IMAGE_ADMITTED_IDS)),
            "recipe": dict(bundle_preflight.LOCAL_IMAGE_RECIPE),
            "platform": bundle_preflight.LOCAL_IMAGE_PLATFORM,
        },
        "model_access": _qualified_model_access(),
        "configs": bundle_preflight._bundle_config_records(
            Path(bundle_receipt["bundle_path"])
        ),
        "executions": executions,
        "all_required_dry_runs_executed": True,
        "provider_mutations_performed": 0,
        "paid_resource_allocated": False,
        "raw_secret_values_recorded": False,
        "blockers": [],
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    path = tmp_path / "config-preflight.json"
    write_json(path, receipt)
    return path


def _passing_local_config_preflight(
    tmp_path: Path, *, bundle_receipt_path: Path, bundle_receipt: dict
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "receipt.json").write_bytes(bundle_receipt_path.read_bytes())
    path = _passing_config_preflight(tmp_path, bundle_receipt)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["schema_version"] = bundle_preflight.LOCAL_SCHEMA_VERSION
    receipt["status"] = "local_passed_paid_model_access_not_checked"
    receipt["bundle_receipt_path"] = str(bundle_receipt_path.resolve())
    receipt["bundle_receipt_sha256"] = "sha256:" + hashlib.sha256(
        bundle_receipt_path.read_bytes()
    ).hexdigest()
    receipt["model_access"] = {
        "provider": "openai",
        "status": "not_executed",
        "models": {
            content_agents.CONTENT_LLM_MODEL: {"status": "not_executed"},
            content_agents.CONTENT_IMAGE_MODEL: {"status": "not_executed"},
        },
        "paid_inference_performed": False,
        "uploaded_scene_bytes": False,
        "blockers": ["content_agents_paid_model_access_preflight_missing"],
    }
    receipt["docker_network_disabled"] = True
    receipt["paid_model_access_required"] = False
    receipt["blockers"] = ["content_agents_paid_model_access_preflight_missing"]
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    write_json(path, receipt)
    return path


def _passing_static_config_preflight(
    tmp_path: Path, *, bundle_receipt_path: Path, bundle_receipt: dict
) -> Path:
    path = _passing_local_config_preflight(
        tmp_path,
        bundle_receipt_path=bundle_receipt_path,
        bundle_receipt=bundle_receipt,
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["schema_version"] = bundle_preflight.STATIC_SCHEMA_VERSION
    receipt["status"] = "static_passed_docker_and_paid_model_access_not_checked"
    receipt["docker_executed"] = False
    receipt["docker_network_disabled"] = None
    receipt["blockers"] = [
        "content_agents_local_docker_config_preflight_missing",
        "content_agents_paid_model_access_preflight_missing",
    ]
    receipt["input_records"] = {
        "source_archive": {
            "member": "provider_runtime/content_agents_source.zip",
            "size_bytes": 1,
            "sha256": "sha256:" + "1" * 64,
        },
        "reference_image": {
            "member": "provider_runtime/input/reference.png",
            "size_bytes": 1,
            "sha256": "sha256:" + "2" * 64,
        },
        "input_usd": {
            "member": "provider_runtime/input/source_asset.usda",
            "size_bytes": 1,
            "sha256": "sha256:" + "3" * 64,
        },
    }
    receipt["input_usd"] = {
        "default_prim_path": "/Asset",
        "mesh_count": 1,
        "default_purpose_mesh_count": 1,
        "declared_target_mesh_count": 0,
        "bbox_min": [0.0, 0.0, 0.0],
        "bbox_max": [1.0, 1.0, 0.0],
        "source_archive_extractable": True,
        "input_usd_opened": True,
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    write_json(path, receipt)
    return path


def _allocator_bundle(
    tmp_path: Path, *, content: bytes = b"config", native_probe: bool = False
) -> tuple[Path, dict]:
    bundle = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for member in bundle_preflight.CONFIG_MEMBERS.values():
            archive.writestr(member, content)
    receipt_value = {
        "status": "ready",
        "source_commit": content_agents.SOURCE_COMMIT,
        "source_tree": content_agents.SOURCE_TREE,
        "container_image": content_agents.DEFAULT_IMAGE,
        "retry_cap": 0,
        "blockers": [],
        "bundle_path": str(bundle),
        "bundle_sha256": "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest(),
    }
    if native_probe:
        receipt_value["native_probe"] = {
            "sage_collision_sha256": "sha256:" + "c" * 64
        }
    receipt = tmp_path / "receipt.json"
    write_json(receipt, receipt_value)
    return receipt, receipt_value


def _allocator_args(
    tmp_path: Path,
    receipt: Path,
    preflight: Path,
    *,
    execute: bool,
    attempt_authority: Path | None = None,
) -> list[str]:
    args = [
        "gpu-canary",
        "--probe-kind",
        content_agents.PROBE_KIND,
        "--provider",
        "vast",
        "--provider-launch-request",
        str(tmp_path / "unused-request.json"),
        "--release-evidence",
        str(tmp_path / "unused-release.json"),
        "--model-cache-evidence",
        str(tmp_path / "unused-model.json"),
        "--preflight-bundle",
        str(tmp_path / "unused-preflight.json"),
        "--admission-out",
        str(tmp_path / "admission.json"),
        "--bound-request-out",
        str(tmp_path / "unused-bound.json"),
        "--adapter-output",
        str(tmp_path / "adapter.json"),
        "--pod-name",
        "adp-content-agents",
        "--adp-content-agents-bundle-receipt",
        str(receipt),
        "--adp-content-agents-config-preflight-receipt",
        str(preflight),
        "--adp-job-dir",
        str(tmp_path / "job"),
        "--adp-max-hourly-rate-usd",
        "0.75",
        "--adp-max-spend-usd",
        "2.00",
        "--adp-hard-ttl-seconds",
        "7200",
    ]
    if attempt_authority is not None:
        args.extend(["--adp-content-agents-attempt-authority", str(attempt_authority)])
    if execute:
        args.append("--execute")
    return args


def _content_agents_attempt_authority(
    *,
    receipt: Path,
    receipt_value: Mapping[str, object],
    preflight: Path,
    preflight_value: Mapping[str, object],
    max_hourly_rate_usd: float = 0.75,
    hard_cap_usd: float = 2.0,
    hard_ttl_seconds: int = 7200,
) -> dict[str, object]:
    authority: dict[str, object] = {
        "schema_version": content_agents.PAID_ATTEMPT_AUTHORITY_SCHEMA,
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authority_reference": "fixture-explicit-content-agents-authority",
        "authorized_by": "fixture-user",
        "authorized_on": "2026-08-11",
        "purpose": "nvidia_content_agents_advisory_enrichment",
        "provider": "vast",
        "paid_compute_authorized": True,
        "bundle_sha256": receipt_value["bundle_sha256"],
        "bundle_receipt_sha256": "sha256:"
        + hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "config_preflight_receipt_sha256": "sha256:"
        + hashlib.sha256(preflight.read_bytes()).hexdigest(),
        "config_preflight_receipt_digest": preflight_value["receipt_digest"],
        "content_agents_source_commit": content_agents.SOURCE_COMMIT,
        "content_agents_source_tree": content_agents.SOURCE_TREE,
        "container_image": content_agents.DEFAULT_IMAGE,
        "maximum_paid_attempts": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "zero_retry": True,
        "hard_attempt_spend_cap_usd": hard_cap_usd,
        "maximum_hourly_rate_usd": max_hourly_rate_usd,
        "maximum_single_resource_ttl_seconds": hard_ttl_seconds,
        "agent_output_is_simready_authority": False,
        "native_simulator_import_qualified": False,
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    return authority


@pytest.mark.parametrize("execute", [False, True])
def test_canonical_allocator_issues_grant_only_for_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execute: bool
) -> None:
    receipt, receipt_value = _allocator_bundle(tmp_path)
    preflight = _passing_config_preflight(tmp_path, receipt_value)
    preflight_value = json.loads(preflight.read_text(encoding="utf-8"))
    authority_path = None
    if execute:
        authority = _content_agents_attempt_authority(
            receipt=receipt,
            receipt_value=receipt_value,
            preflight=preflight,
            preflight_value=preflight_value,
        )
        authority_path = tmp_path / "content-agents-attempt-authority.json"
        write_json(authority_path, authority)
    observed: dict = {}
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "completed" if kwargs["execute"] else "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_content_agents_vast", fake_run)
    monkeypatch.setattr(
        content_agents, "AUTHORIZATION_CONSUMPTION_ROOT", tmp_path / "consumed"
    )
    assert (
        allocator.main(
            _allocator_args(
                tmp_path,
                receipt,
                preflight,
                execute=execute,
                attempt_authority=authority_path,
            )
        )
        == 0
    )
    assert observed["execute"] is execute
    assert isinstance(observed["paid_resource_admission_grant"], PaidResourceAdmissionGrant) is (
        execute
    )
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["retry_cap"] == 0
    assert admission["private_or_licensed_dataset_bytes_uploaded"] is False
    assert admission["hard_cap_usd"] == 2.0
    assert admission["allocation_binding"]["config_preflight_receipt_sha256"]
    assert admission["paid_attempt_authority_required_for_execute"] is True
    if execute:
        assert (
            admission["authority"]
            == "explicit_content_agents_paid_attempt_authority_bound"
        )
        assert admission["allocation_binding"]["paid_attempt_authority_digest"]
        assert list((tmp_path / "consumed").glob("content-agents-*.json"))
    else:
        assert admission["authority"] == "dry_run_no_paid_authority_required"


def test_canonical_allocator_discloses_public_sage_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, receipt_value = _allocator_bundle(tmp_path, native_probe=True)
    preflight = _passing_config_preflight(tmp_path, receipt_value)
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )
    monkeypatch.setattr(
        allocator,
        "run_content_agents_vast",
        lambda **_kwargs: {"status": "dry_run_ready"},
    )

    assert allocator.main(_allocator_args(tmp_path, receipt, preflight, execute=False)) == 0
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["authority"] == "dry_run_no_paid_authority_required"
    assert admission["private_or_licensed_dataset_bytes_uploaded"] is True
    assert admission["private_or_gated_dataset_bytes_uploaded"] is False
    assert admission["public_licensed_sage_collision_bytes_uploaded"] is True
    assert admission["public_licensed_dataset_identity"]["license"] == "CC-BY-NC-4.0"
    assert admission["input_is_blueprint_owned_parametric_control"] is False
    assert admission["input_contains_blueprint_owned_parametric_control"] is True


def test_content_agents_paid_attempt_authority_is_single_use(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receipt, receipt_value = _allocator_bundle(tmp_path)
    preflight = _passing_config_preflight(tmp_path, receipt_value)
    preflight_value = json.loads(preflight.read_text(encoding="utf-8"))
    authority = _content_agents_attempt_authority(
        receipt=receipt,
        receipt_value=receipt_value,
        preflight=preflight,
        preflight_value=preflight_value,
    )
    validated = content_agents.validate_content_agents_paid_attempt_authority(
        authority,
        prepared_bundle=receipt_value,
        bundle_receipt_sha256="sha256:" + hashlib.sha256(receipt.read_bytes()).hexdigest(),
        config_preflight=preflight_value,
        config_preflight_receipt_sha256="sha256:"
        + hashlib.sha256(preflight.read_bytes()).hexdigest(),
        max_hourly_rate_usd=0.75,
        hard_cap_usd=2.0,
        hard_ttl_seconds=7200,
    )
    monkeypatch.setattr(
        content_agents, "AUTHORIZATION_CONSUMPTION_ROOT", tmp_path / "consumed"
    )

    first = content_agents.consume_content_agents_paid_attempt_authority_once(
        validated, blueprint_commit="a" * 40
    )
    second = content_agents.consume_content_agents_paid_attempt_authority_once(
        validated, blueprint_commit="a" * 40
    )

    assert first["status"] == "consumed"
    assert second == {
        "status": "blocked",
        "blockers": ["adp_content_agents_paid_attempt_authority_consumed"],
    }


def test_content_agents_execute_requires_paid_attempt_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, receipt_value = _allocator_bundle(tmp_path)
    preflight = _passing_config_preflight(tmp_path, receipt_value)
    observed = {"called": False}
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )
    monkeypatch.setattr(
        allocator,
        "run_content_agents_vast",
        lambda **_kwargs: observed.__setitem__("called", True) or {},
    )

    assert allocator.main(_allocator_args(tmp_path, receipt, preflight, execute=True)) == 2
    result = json.loads((tmp_path / "adapter.json").read_text())
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert result["blockers"] == [
        "paid_resource_admission_has_blockers",
        "paid_resource_admission_not_admitted",
    ]
    assert (
        "adp_content_agents_paid_attempt_authority_missing"
        in admission["blockers"]
    )
    assert admission["authority"] == "missing_content_agents_paid_attempt_authority"
    assert admission["paid_attempt_authority_required_for_execute"] is True
    assert observed["called"] is False


def test_content_agents_consumed_authority_blocks_before_provider_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, receipt_value = _allocator_bundle(tmp_path)
    preflight = _passing_config_preflight(tmp_path, receipt_value)
    preflight_value = json.loads(preflight.read_text(encoding="utf-8"))
    authority = _content_agents_attempt_authority(
        receipt=receipt,
        receipt_value=receipt_value,
        preflight=preflight,
        preflight_value=preflight_value,
    )
    authority_path = tmp_path / "content-agents-attempt-authority.json"
    write_json(authority_path, authority)
    consumed_root = tmp_path / "consumed"
    monkeypatch.setattr(content_agents, "AUTHORIZATION_CONSUMPTION_ROOT", consumed_root)
    content_agents.consume_content_agents_paid_attempt_authority_once(
        authority, blueprint_commit="a" * 40
    )
    observed = {"called": False}
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )
    monkeypatch.setattr(
        allocator,
        "run_content_agents_vast",
        lambda **_kwargs: observed.__setitem__("called", True) or {},
    )

    assert (
        allocator.main(
            _allocator_args(
                tmp_path,
                receipt,
                preflight,
                execute=True,
                attempt_authority=authority_path,
            )
        )
        == 2
    )
    result = json.loads((tmp_path / "adapter.json").read_text())
    assert result["blockers"] == ["adp_content_agents_paid_attempt_authority_consumed"]
    assert result["provider_mutations_performed"] == 0
    assert observed["called"] is False


def test_canonical_allocator_rejects_mutated_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, receipt_value = _allocator_bundle(tmp_path, content=b"before")
    preflight = _passing_config_preflight(tmp_path, receipt_value)
    bundle = Path(receipt_value["bundle_path"])
    bundle.write_bytes(b"after")
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )
    assert allocator.main(_allocator_args(tmp_path, receipt, preflight, execute=False)) == 2
    result = json.loads((tmp_path / "adapter.json").read_text())
    assert "adp_content_agents_bundle_binding_invalid" in result["blockers"]


def test_canonical_allocator_requires_exact_bundle_config_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, receipt_value = _allocator_bundle(tmp_path)
    preflight = _passing_config_preflight(tmp_path, receipt_value)
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )
    arguments = _allocator_args(tmp_path, receipt, preflight, execute=False)
    flag = arguments.index("--adp-content-agents-config-preflight-receipt")
    del arguments[flag : flag + 2]

    assert allocator.main(arguments) == 2
    result = json.loads((tmp_path / "adapter.json").read_text())
    assert "adp_content_agents_config_preflight_receipt" in result["blockers"]


def test_canonical_allocator_rejects_changed_preflight_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, receipt_value = _allocator_bundle(tmp_path)
    preflight = _passing_config_preflight(tmp_path, receipt_value)
    (tmp_path / "texture.log").write_text("changed after preflight", encoding="utf-8")
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )

    assert allocator.main(_allocator_args(tmp_path, receipt, preflight, execute=False)) == 2
    result = json.loads((tmp_path / "adapter.json").read_text())
    assert (
        "adp_content_agents_config_preflight_execution_invalid:texture"
        in result["blockers"]
    )


def _executable_preflight_fixture(tmp_path: Path) -> tuple[Path, str]:
    secret = "unit-test-secret-that-must-not-be-recorded"
    source_buffer = io.BytesIO()
    with zipfile.ZipFile(source_buffer, "w") as archive:
        archive.writestr(
            "apps/material_agent/data/materials/material_libs_default/materials.yaml",
            "materials: []\n",
        )
    bundle = tmp_path / "exact-bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr(
            "provider_runtime/content_agents_source.zip", source_buffer.getvalue()
        )
        archive.writestr("provider_runtime/adp_content_agents_provider_runner.py", "pass\n")
        archive.writestr(
            "provider_runtime/run_adp_content_agents_provider_runtime.sh",
            "#!/bin/sh\n",
        )
        archive.writestr("provider_runtime/input/reference.png", b"PNG")
        for member in bundle_preflight.CONFIG_MEMBERS.values():
            archive.writestr(
                member,
                "project:\n"
                "  name: exact-bundle-test\n"
                "input:\n"
                "  usd_path: ../input/source_asset.usda\n"
                "  reference_images:\n"
                "    - ../input/reference.png\n",
            )
        archive.writestr(
            "provider_runtime/input/source_asset.usda",
            """#usda 1.0
(
    defaultPrim = "Asset"
)
def Xform "Asset"
{
    def Mesh "mesh"
    {
        uniform token subdivisionScheme = "none"
        token purpose = "default"
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
    }
}
""",
        )
    bundle_receipt = tmp_path / "exact-bundle-receipt.json"
    write_json(
        bundle_receipt,
        {
            "status": "ready",
            "source_commit": content_agents.SOURCE_COMMIT,
            "source_tree": content_agents.SOURCE_TREE,
            "bundle_path": str(bundle),
            "bundle_sha256": "sha256:"
            + hashlib.sha256(bundle.read_bytes()).hexdigest(),
        },
    )
    return bundle_receipt, secret


def test_exact_bundle_preflight_executes_all_clis_and_never_records_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_receipt, secret = _executable_preflight_fixture(tmp_path)
    observed_commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        command = list(command)
        observed_commands.append(command)
        if command[0] == "git":
            if command[-2:] == ["rev-parse", "HEAD"]:
                stdout = "c" * 40 + "\n"
            elif command[-2:] == ["rev-parse", "HEAD^{tree}"]:
                stdout = "d" * 40 + "\n"
            elif command[-2:] == ["status", "--porcelain"]:
                stdout = ""
            else:
                raise AssertionError(command)
            return subprocess.CompletedProcess(command, 0, stdout, "")
        if command[1:3] == ["image", "inspect"]:
            stdout = json.dumps(
                [
                    {
                        "Id": next(iter(bundle_preflight.LOCAL_IMAGE_ADMITTED_IDS)),
                        "Os": "linux",
                        "Architecture": "arm64",
                    }
                ]
            )
            return subprocess.CompletedProcess(command, 0, stdout, "")
        assert secret not in " ".join(command)
        entrypoint = command[command.index("--entrypoint") + 1]
        if entrypoint == "python":
            return subprocess.CompletedProcess(
                command, 0, bundle_preflight.USD_BBOX_MARKER, ""
            )
        assert kwargs["env"]["OPENAI_API_KEY"] == secret
        if "--only" in command:
            return subprocess.CompletedProcess(
                command, 0, bundle_preflight.MATERIAL_VALIDATE_MARKER, ""
            )
        name = next(
            name
            for name, expected in bundle_preflight.ENTRYPOINTS.items()
            if expected == entrypoint
        )
        return subprocess.CompletedProcess(
            command, 0, bundle_preflight.DRY_RUN_MARKERS[name], ""
        )

    monkeypatch.setattr(bundle_preflight.subprocess, "run", fake_run)
    monkeypatch.setattr(bundle_preflight, "_secret", lambda: secret)
    monkeypatch.setattr(
        bundle_preflight,
        "_probe_model_access",
        lambda _secret, _output: _qualified_model_access(),
    )
    evidence = tmp_path / "evidence"

    receipt = bundle_preflight.materialize_bundle_config_preflight(
        bundle_receipt_path=bundle_receipt,
        evidence_dir=evidence,
        generated_at="fixed",
    )

    assert len(observed_commands) == 9
    assert receipt["status"] == "passed"
    assert receipt["all_required_dry_runs_executed"] is True
    assert bundle_preflight.validate_bundle_config_preflight(
        preflight=receipt,
        prepared_bundle=json.loads(bundle_receipt.read_text()),
        preflight_receipt_path=(
            evidence / "adp_content_agents_bundle_config_preflight.json"
        ),
        expected_orchestrator_source_commit="c" * 40,
    ) == []
    assert secret not in json.dumps(receipt)
    assert secret not in "".join(path.read_text() for path in evidence.glob("*.log"))
    assert not list(tmp_path.glob("adp-content-agents-preflight-*"))


def test_local_bundle_preflight_executes_offline_without_paid_model_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_receipt, _secret = _executable_preflight_fixture(tmp_path)
    observed_commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        command = list(command)
        observed_commands.append(command)
        if command[0] == "git":
            if command[-2:] == ["rev-parse", "HEAD"]:
                stdout = "c" * 40 + "\n"
            elif command[-2:] == ["rev-parse", "HEAD^{tree}"]:
                stdout = "d" * 40 + "\n"
            elif command[-2:] == ["status", "--porcelain"]:
                stdout = ""
            else:
                raise AssertionError(command)
            return subprocess.CompletedProcess(command, 0, stdout, "")
        if command[1:3] == ["image", "inspect"]:
            stdout = json.dumps(
                [
                    {
                        "Id": next(iter(bundle_preflight.LOCAL_IMAGE_ADMITTED_IDS)),
                        "Os": "linux",
                        "Architecture": "arm64",
                    }
                ]
            )
            return subprocess.CompletedProcess(command, 0, stdout, "")
        assert "--network" in command
        assert command[command.index("--network") + 1] == "none"
        entrypoint = command[command.index("--entrypoint") + 1]
        if entrypoint == "python":
            return subprocess.CompletedProcess(
                command, 0, bundle_preflight.USD_BBOX_MARKER, ""
            )
        assert (
            kwargs["env"]["OPENAI_API_KEY"]
            == bundle_preflight.LOCAL_NO_PAID_SECRET
        )
        if "--only" in command:
            return subprocess.CompletedProcess(
                command, 0, bundle_preflight.MATERIAL_VALIDATE_MARKER, ""
            )
        name = next(
            name
            for name, expected in bundle_preflight.ENTRYPOINTS.items()
            if expected == entrypoint
        )
        return subprocess.CompletedProcess(
            command, 0, bundle_preflight.DRY_RUN_MARKERS[name], ""
        )

    monkeypatch.setattr(bundle_preflight.subprocess, "run", fake_run)
    monkeypatch.setattr(
        bundle_preflight,
        "_secret",
        lambda: (_ for _ in ()).throw(AssertionError("secret not allowed")),
    )
    monkeypatch.setattr(
        bundle_preflight,
        "_probe_model_access",
        lambda _secret, _output: (_ for _ in ()).throw(
            AssertionError("paid model probe not allowed")
        ),
    )
    evidence = tmp_path / "local-evidence"

    receipt = bundle_preflight.materialize_local_bundle_config_preflight(
        bundle_receipt_path=bundle_receipt,
        evidence_dir=evidence,
        generated_at="fixed",
    )

    assert receipt["schema_version"] == bundle_preflight.LOCAL_SCHEMA_VERSION
    assert receipt["status"] == "local_passed_paid_model_access_not_checked"
    assert receipt["docker_network_disabled"] is True
    assert receipt["paid_model_access_required"] is False
    assert receipt["model_access"]["paid_inference_performed"] is False
    assert receipt["blockers"] == [
        "content_agents_paid_model_access_preflight_missing"
    ]
    assert bundle_preflight.validate_local_bundle_config_preflight(
        preflight=receipt,
        prepared_bundle=json.loads(bundle_receipt.read_text()),
        preflight_receipt_path=(
            evidence / "adp_content_agents_bundle_config_preflight.json"
        ),
        expected_orchestrator_source_commit="c" * 40,
    ) == []
    assert "adp_content_agents_config_preflight_binding_invalid" in (
        bundle_preflight.validate_bundle_config_preflight(
            preflight=receipt,
            prepared_bundle=json.loads(bundle_receipt.read_text()),
            preflight_receipt_path=(
                evidence / "adp_content_agents_bundle_config_preflight.json"
            ),
            expected_orchestrator_source_commit="c" * 40,
        )
    )
    assert not list(tmp_path.glob("adp-content-agents-preflight-*"))


def test_static_bundle_preflight_inspects_bundle_without_docker_or_paid_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_receipt, _secret = _executable_preflight_fixture(tmp_path)
    observed_commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        command = list(command)
        observed_commands.append(command)
        assert command[0] == "git"
        if command[-2:] == ["rev-parse", "HEAD"]:
            stdout = "c" * 40 + "\n"
        elif command[-2:] == ["rev-parse", "HEAD^{tree}"]:
            stdout = "d" * 40 + "\n"
        elif command[-2:] == ["status", "--porcelain"]:
            stdout = ""
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(bundle_preflight.subprocess, "run", fake_run)
    monkeypatch.setattr(
        bundle_preflight,
        "_secret",
        lambda: (_ for _ in ()).throw(AssertionError("secret not allowed")),
    )
    monkeypatch.setattr(
        bundle_preflight,
        "_probe_model_access",
        lambda _secret, _output: (_ for _ in ()).throw(
            AssertionError("paid model probe not allowed")
        ),
    )
    evidence = tmp_path / "static-evidence"

    receipt = bundle_preflight.materialize_static_bundle_config_preflight(
        bundle_receipt_path=bundle_receipt,
        evidence_dir=evidence,
        generated_at="fixed",
    )

    assert observed_commands == [
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        ["git", "-C", str(ROOT), "rev-parse", "HEAD^{tree}"],
        ["git", "-C", str(ROOT), "status", "--porcelain"],
    ]
    assert receipt["schema_version"] == bundle_preflight.STATIC_SCHEMA_VERSION
    assert receipt["status"] == "static_passed_docker_and_paid_model_access_not_checked"
    assert receipt["docker_executed"] is False
    assert receipt["input_usd"]["mesh_count"] == 1
    assert receipt["blockers"] == [
        "content_agents_local_docker_config_preflight_missing",
        "content_agents_paid_model_access_preflight_missing",
    ]
    assert bundle_preflight.validate_static_bundle_config_preflight(
        preflight=receipt,
        prepared_bundle=json.loads(bundle_receipt.read_text()),
        preflight_receipt_path=(
            evidence / "adp_content_agents_static_bundle_config_preflight.json"
        ),
        expected_orchestrator_source_commit="c" * 40,
    ) == []
    assert "adp_content_agents_local_config_preflight_binding_invalid" in (
        bundle_preflight.validate_local_bundle_config_preflight(
            preflight=receipt,
            prepared_bundle=json.loads(bundle_receipt.read_text()),
            preflight_receipt_path=(
                evidence / "adp_content_agents_static_bundle_config_preflight.json"
            ),
            expected_orchestrator_source_commit="c" * 40,
        )
    )


def test_exact_bundle_preflight_fails_before_receipt_when_any_cli_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_receipt, secret = _executable_preflight_fixture(tmp_path)

    def fake_run(command, **kwargs):
        command = list(command)
        if command[0] == "git":
            if command[-2:] == ["rev-parse", "HEAD"]:
                stdout = "c" * 40 + "\n"
            elif command[-2:] == ["rev-parse", "HEAD^{tree}"]:
                stdout = "d" * 40 + "\n"
            elif command[-2:] == ["status", "--porcelain"]:
                stdout = ""
            else:
                raise AssertionError(command)
            return subprocess.CompletedProcess(command, 0, stdout, "")
        if command[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    [
                        {
                            "Id": next(iter(bundle_preflight.LOCAL_IMAGE_ADMITTED_IDS)),
                            "Os": "linux",
                            "Architecture": "arm64",
                        }
                    ]
                ),
                "",
            )
        entrypoint = command[command.index("--entrypoint") + 1]
        return subprocess.CompletedProcess(
            command,
            1 if entrypoint == "texture-agent" else 0,
            "failed" if entrypoint == "texture-agent" else "Dry run complete",
            "",
        )

    monkeypatch.setattr(bundle_preflight.subprocess, "run", fake_run)
    monkeypatch.setattr(bundle_preflight, "_secret", lambda: secret)
    monkeypatch.setattr(
        bundle_preflight,
        "_probe_model_access",
        lambda _secret, _output: _qualified_model_access(),
    )
    evidence = tmp_path / "failed-evidence"

    with pytest.raises(
        bundle_preflight.ContentAgentsBundlePreflightError,
        match="dry_run_failed:texture",
    ):
        bundle_preflight.materialize_bundle_config_preflight(
            bundle_receipt_path=bundle_receipt,
            evidence_dir=evidence,
        )

    assert not (evidence / "adp_content_agents_bundle_config_preflight.json").exists()


ARTICULATED_MANIFEST_RELATIVE_PATH = (
    "docs/arm_decision_proof_v1/manifests/"
    "second_scene_840796_deterministic_simready_candidate.v1.json"
)


def _articulated_evidence(tmp_path: Path):
    repo = tmp_path / "repo"
    evidence = tmp_path / "evidence"
    (repo / "docs/arm_decision_proof_v1/manifests").mkdir(parents=True)
    candidate_dir = evidence / "simready_candidate/840796_deterministic_v1"
    candidate_dir.mkdir(parents=True)
    candidate = candidate_dir / "simready_candidate_deterministic.usda"
    candidate.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    snapshot = candidate_dir / "candidate_reference.png"
    snapshot.write_bytes(b"\x89PNG\r\n\x1a\n" + b"articulated-reference")
    manifest = {
        "schema_version": "second_scene_deterministic_simready_candidate.v1",
        "publisher_scene_id": "840796",
        "target_instance_id": "123",
        "construction_path": "deterministic_parametric_from_frozen_observations",
        "authoring": {
            "status": "simready_candidate_statically_admitted",
            "output_usd_sha256": content_agents._sha256(candidate),
            "task_joint_prim_path": "/Asset/joints/upper_door_hinge",
        },
        "evidence_relative_paths": {
            "candidate_usd": "simready_candidate/840796_deterministic_v1/"
            "simready_candidate_deterministic.usda",
            "reference_image": "simready_candidate/840796_deterministic_v1/"
            "candidate_reference.png",
        },
        "reference_image_sha256": content_agents._sha256(snapshot),
        "topology_validation_receipt_digest": "sha256:" + "1" * 64,
        "physics_validation_receipt_digest": "sha256:" + "2" * 64,
        "receipt_digest": "",
    }
    manifest["receipt_digest"] = canonical_digest(manifest, digest_field="receipt_digest")
    write_json(repo / ARTICULATED_MANIFEST_RELATIVE_PATH, manifest)
    return repo, evidence, snapshot, candidate


def test_articulated_variant_binds_statically_admitted_candidate(tmp_path: Path) -> None:
    repo, evidence, snapshot, candidate = _articulated_evidence(tmp_path)

    resolved = content_agents._resolve_input_variant(
        repo=repo,
        evidence_root=evidence,
        reference_source=snapshot,
        variant="articulated_v1",
    )

    assert resolved["variant"] == "articulated_v1"
    assert resolved["usd_source"] == candidate
    assert set(resolved["config_sources"]) == {
        "material_agent.yaml",
        "texture_agent.yaml",
        "physics_agent.yaml",
    }
    assert resolved["candidate_receipt_digest"].startswith("sha256:")
    assert resolved["task_joint_prim_path"] == "/Asset/joints/upper_door_hinge"
    assert "interiorgs_dataset_bytes" in resolved["reference_image_authority"]


def test_articulated_variant_rejects_changed_candidate_bytes(tmp_path: Path) -> None:
    repo, evidence, snapshot, candidate = _articulated_evidence(tmp_path)
    candidate.write_text('#usda 1.0\ndef Xform "Tampered" {}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="source_identity_mismatch"):
        content_agents._resolve_input_variant(
            repo=repo,
            evidence_root=evidence,
            reference_source=snapshot,
            variant="articulated_v1",
        )


def test_articulated_variant_rejects_unadmitted_candidate(tmp_path: Path) -> None:
    repo, evidence, snapshot, _candidate = _articulated_evidence(tmp_path)
    manifest = json.loads(
        (repo / ARTICULATED_MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    manifest["authoring"]["status"] = "blocked"
    manifest["receipt_digest"] = ""
    manifest["receipt_digest"] = canonical_digest(
        manifest, digest_field="receipt_digest"
    )
    write_json(repo / ARTICULATED_MANIFEST_RELATIVE_PATH, manifest)

    with pytest.raises(ValueError, match="candidate_receipt_not_eligible"):
        content_agents._resolve_input_variant(
            repo=repo,
            evidence_root=evidence,
            reference_source=snapshot,
            variant="articulated_v1",
        )


def test_articulated_configs_preserve_articulation() -> None:
    """No SimReady pass may re-derive geometry or rebuild the articulation."""

    assets = ROOT / "docs" / "arm_decision_proof_v1" / "assets"
    for name in ("material", "texture", "physics"):
        text = (
            assets / f"adp009d_content_agents_articulated_{name}.vast.yaml"
        ).read_text(encoding="utf-8")
        assert "840313" not in text
        assert "840796" in text
        assert "apply_joint_rigger" not in text
        if "optimize_usd:" in text:
            block = text.split("optimize_usd:", 1)[1][:120]
            assert "enabled: false" in block
    physics = (
        assets / "adp009d_content_agents_articulated_physics.vast.yaml"
    ).read_text(encoding="utf-8")
    assert "mass_scale_policy: skip_mass" in physics


def test_articulated_bundle_normalizes_scene_neutral_runtime_input_names(
    tmp_path: Path,
) -> None:
    """Scene-specific config inputs are normalized at the reusable seam."""

    assets = ROOT / "docs" / "arm_decision_proof_v1" / "assets"
    sources = {
        f"{agent}_agent.yaml": assets
        / f"adp009d_content_agents_articulated_{agent}.vast.yaml"
        for agent in ("material", "texture", "physics")
    }
    destination = tmp_path / "configs"
    destination.mkdir()
    content_agents._materialize_remote_configs(
        config_sources=sources,
        destination=destination,
        variant="articulated_v1",
    )
    for name in sources:
        payload = yaml.safe_load((destination / name).read_text(encoding="utf-8"))
        assert payload["input"]["usd_path"] == "../input/source_asset.usda"
        if "reference_images" in payload["input"]:
            assert payload["input"]["reference_images"] == ["../input/reference.png"]


def test_articulated_configs_preserve_agent_policy_while_normalizing_inputs(
    tmp_path: Path,
) -> None:
    assets = ROOT / "docs" / "arm_decision_proof_v1" / "assets"
    sources = {
        f"{agent}_agent.yaml": assets
        / f"adp009d_content_agents_articulated_{agent}.vast.yaml"
        for agent in ("material", "texture", "physics")
    }
    destination = tmp_path / "configs"
    destination.mkdir()

    hashes = content_agents._materialize_remote_configs(
        config_sources=sources, destination=destination, variant="articulated_v1"
    )

    for name in sources:
        assert hashes[name] == content_agents._sha256(destination / name)
        assert "apply_joint_rigger" not in (destination / name).read_text()


def _articulated_runtime_configs(tmp_path: Path) -> dict:
    assets = ROOT / "docs" / "arm_decision_proof_v1" / "assets"
    destination = tmp_path / "configs"
    destination.mkdir()
    sources = {
        f"{agent}_agent.yaml": assets
        / f"adp009d_content_agents_articulated_{agent}.vast.yaml"
        for agent in ("material", "texture", "physics")
    }
    content_agents._materialize_remote_configs(
        config_sources=sources, destination=destination, variant="articulated_v1"
    )
    return {name: destination / name for name in sources}


def test_remote_config_contract_accepts_the_articulated_target_prims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The contract guards runtime failure modes, not one scene's prim paths."""

    content_agents._validate_remote_configs(
        source=_fake_source(tmp_path, monkeypatch),
        config_sources=_articulated_runtime_configs(tmp_path),
    )


def test_remote_config_contract_rejects_inconsistent_texture_targets(
    tmp_path: Path,
) -> None:
    configs = _articulated_runtime_configs(tmp_path)
    import yaml as _yaml

    payload = _yaml.safe_load(configs["texture_agent.yaml"].read_text())
    payload["target_prims"] = ["/Asset/lower_door/component_002"]
    configs["texture_agent.yaml"].write_text(_yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match="remote_config_contract_invalid"):
        content_agents._validate_remote_configs(
            source=Path("/private/tmp/usd-content-agents-0.5.2-20260808"),
            config_sources=configs,
        )


def test_articulated_input_normalization_preserves_joints_and_purposes(
    tmp_path: Path,
) -> None:
    """The articulated candidate needs its own NVIDIA-compat normalization."""

    from pxr import Usd, UsdGeom, UsdPhysics

    source = tmp_path / "candidate.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(asset.GetPrim())
    for link in ("cabinet", "upper_door"):
        link_prim = UsdGeom.Xform.Define(stage, f"/Asset/{link}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(link_prim)
        mesh = UsdGeom.Mesh.Define(stage, f"/Asset/{link}/geom")
        mesh.CreatePointsAttr(
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        )
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.GetPurposeAttr().Set(UsdGeom.Tokens.render)
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Asset/joints/upper_door_hinge")
    joint.CreateBody0Rel().SetTargets(["/Asset/cabinet"])
    joint.CreateBody1Rel().SetTargets(["/Asset/upper_door"])
    joint.CreateAxisAttr().Set("Z")
    stage.GetRootLayer().Save()

    result = content_agents._materialize_content_agents_input(
        source, tmp_path / "runtime_input.usda", variant="articulated_v1"
    )

    assert result["source_input_usd_sha256"] == content_agents._sha256(source)
    assert result["default_purpose_bbox_nonempty"] is True
    assert result["articulation_preserved"] is True
    assert result["joint_count"] == 1
    assert result["rigid_body_count"] == 2
    reopened = Usd.Stage.Open(str(tmp_path / "runtime_input.usda"))
    assert str(reopened.GetDefaultPrim().GetPath()) == "/Asset"
    assert (
        len([p for p in reopened.Traverse() if p.IsA(UsdPhysics.Joint)]) == 1
    )
    for prim in reopened.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            assert UsdGeom.Mesh(prim).ComputePurpose() == UsdGeom.Tokens.default_


def test_joint_agent_plan_is_derived_from_topology_for_both_fixtures() -> None:
    rigid = content_agents._derive_joint_agent_plan(
        input_variant="control_v1",
        input_normalization={
            "joint_count": 0,
            "rigid_body_count": 1,
            "articulation_root_count": 0,
        },
    )
    articulated = content_agents._derive_joint_agent_plan(
        input_variant="articulated_v1",
        input_normalization={
            "joint_count": 2,
            "rigid_body_count": 3,
            "articulation_root_count": 1,
        },
    )

    assert rigid["joint_agent_inapplicable_single_rigid_body"] is True
    assert rigid["reason"] == "single_rigid_body_has_no_articulation_task"
    assert articulated["joint_agent_inapplicable_single_rigid_body"] is False
    assert articulated["reason"] == (
        "preexisting_articulation_preserved_by_enrichment_pass"
    )


def test_provider_runner_resolves_manifest_bound_articulated_input(
    tmp_path: Path,
) -> None:
    runner = _provider_runner_module()
    runtime = tmp_path / "provider_runtime"
    input_path = runtime / "input/refrigerator.usda"
    input_path.parent.mkdir(parents=True)
    input_path.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(input_path.read_bytes()).hexdigest()
    plan = content_agents._derive_joint_agent_plan(
        input_variant="articulated_v1",
        input_normalization={
            "joint_count": 2,
            "rigid_body_count": 3,
            "articulation_root_count": 1,
        },
    )
    write_json(
        runtime / "adp_content_agents_provider_manifest.json",
        {
            "runtime_input_binding": {
                "relative_path": "input/refrigerator.usda",
                "sha256": digest,
            },
            "joint_agent_plan": plan,
        },
    )

    resolved, resolved_plan = runner._runtime_input_plan(runtime)

    assert resolved == input_path.resolve()
    assert resolved_plan == plan


def test_provider_runner_rejects_changed_bound_input(tmp_path: Path) -> None:
    runner = _provider_runner_module()
    runtime = tmp_path / "provider_runtime"
    input_path = runtime / "input/object.usda"
    input_path.parent.mkdir(parents=True)
    input_path.write_text('#usda 1.0\n', encoding="utf-8")
    write_json(
        runtime / "adp_content_agents_provider_manifest.json",
        {
            "runtime_input_binding": {
                "relative_path": "input/object.usda",
                "sha256": "sha256:" + "0" * 64,
            },
            "joint_agent_plan": {
                "planned": False,
                "executed_by_content_agents_bundle": False,
                "reason": "single_rigid_body_has_no_articulation_task",
                "input_joint_count": 0,
                "input_rigid_body_count": 1,
            },
        },
    )

    with pytest.raises(ValueError, match="runtime_input_binding_invalid"):
        runner._runtime_input_plan(runtime)


def test_local_preflight_image_admission_is_recipe_bound(monkeypatch) -> None:
    """A rebuild of the reviewed recipe is admissible; a stray image is not."""

    import subprocess as _subprocess

    admitted_id = next(iter(bundle_preflight.LOCAL_IMAGE_ADMITTED_IDS))

    def _inspect(command, **kwargs):
        image_id = admitted_id if "admitted" in command[-1] else "sha256:" + "f" * 64
        return _subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                [{"Id": image_id, "Os": "linux", "Architecture": "arm64"}]
            ),
            stderr="",
        )

    monkeypatch.setattr(bundle_preflight.subprocess, "run", _inspect)

    record = bundle_preflight._inspect_image(docker="docker", image="admitted")
    assert record["id"] == admitted_id
    assert record["recipe"]["dockerfile_sha256"].startswith("sha256:")
    assert record["recipe"]["base_image"].startswith("ghcr.io/astral-sh/uv:")

    with pytest.raises(
        bundle_preflight.ContentAgentsBundlePreflightError,
        match="local_preflight_image_identity_mismatch",
    ):
        bundle_preflight._inspect_image(docker="docker", image="stray")


def test_local_preflight_image_recipe_matches_checked_in_dockerfile() -> None:
    dockerfile = (
        ROOT
        / "docs/arm_decision_proof_v1/assets"
        / "adp009a_usd_content_agents_linux_arm64.Dockerfile"
    )
    observed = "sha256:" + hashlib.sha256(dockerfile.read_bytes()).hexdigest()
    for recipe in bundle_preflight.LOCAL_IMAGE_ADMITTED_IDS.values():
        assert recipe["dockerfile_sha256"] == observed
        assert recipe["content_agents_source_tree"] == content_agents.SOURCE_TREE


def test_usd_bbox_probe_script_is_derived_from_the_bundle_input() -> None:
    script = bundle_preflight.usd_bbox_script(
        "adp009d_840796_articulated_refrigerator_candidate.usda"
    )

    assert "adp009d_840796_articulated_refrigerator_candidate.usda" in script
    assert "canned_beverage" not in script
    assert "GetDefaultPrim" in script
    assert "ComputePurpose()==UsdGeom.Tokens.default_" in script


def test_bundle_preflight_accepts_any_admitted_variant_input_names(
    tmp_path: Path,
) -> None:
    """Required entries must not hardcode one variant's input filenames."""

    bundle = tmp_path / "articulated-bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for member in (
            "provider_runtime/run_adp_content_agents_provider_runtime.sh",
            "provider_runtime/adp_content_agents_provider_runner.py",
            "provider_runtime/adp_content_agents_provider_manifest.json",
            "provider_runtime/content_agents_source.zip",
            "provider_runtime/configs/material_agent.yaml",
            "provider_runtime/configs/texture_agent.yaml",
            "provider_runtime/configs/physics_agent.yaml",
        ):
            archive.writestr(member, "{}")
        archive.writestr(
            "provider_runtime/input/"
            "adp009d_840796_articulated_refrigerator_candidate.usda",
            '#usda 1.0\n',
        )
        archive.writestr(
            "provider_runtime/input/"
            "adp009d_840796_articulated_refrigerator_candidate_reference.png",
            b"\x89PNG\r\n\x1a\n",
        )

    preflight = _blueprint_bundle_preflight(
        job_dir=tmp_path / "preflight",
        generated_at="fixed",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp_content_agents",
        bundle_path=bundle,
        provider_bundle_url="https://example.com/bundle.zip?signature=redacted",
        provider_output_put_url="https://example.com/output.zip?signature=redacted",
    )

    assert "provider_runtime_bundle_required_entries_missing" not in (
        preflight.get("blockers") or []
    )


def test_bundle_preflight_still_requires_one_input_usd_and_reference(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "no-input-bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for member in (
            "provider_runtime/run_adp_content_agents_provider_runtime.sh",
            "provider_runtime/adp_content_agents_provider_runner.py",
            "provider_runtime/adp_content_agents_provider_manifest.json",
            "provider_runtime/content_agents_source.zip",
            "provider_runtime/configs/material_agent.yaml",
            "provider_runtime/configs/texture_agent.yaml",
            "provider_runtime/configs/physics_agent.yaml",
        ):
            archive.writestr(member, "{}")

    preflight = _blueprint_bundle_preflight(
        job_dir=tmp_path / "preflight",
        generated_at="fixed",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp_content_agents",
        bundle_path=bundle,
        provider_bundle_url="https://example.com/bundle.zip?signature=redacted",
        provider_output_put_url="https://example.com/output.zip?signature=redacted",
    )

    assert "provider_runtime_bundle_required_entries_missing" in (
        preflight.get("blockers") or []
    )


def test_articulated_agent_configs_describe_the_articulated_object() -> None:
    """A config carried over from another scene silently mis-classifies.

    The first 840796 material pass returned Car_Paint_Green for a pale
    off-white refrigerator because the prompts, derived from the 840313 packet,
    still asked the model to classify a bright green beverage can. The model was
    answering the question it was given.
    """

    assets = ROOT / "docs/arm_decision_proof_v1/assets"
    foreign = ("beverage", "canned", "green can", "soda")
    for name in ("material", "texture", "physics"):
        text = (
            assets / f"adp009d_content_agents_articulated_{name}.vast.yaml"
        ).read_text(encoding="utf-8")
        lowered = text.lower()
        assert "refrigerator" in lowered, name
        for token in foreign:
            assert token not in lowered, f"{name} config still mentions {token!r}"


def test_articulated_texture_config_targets_a_real_render_material() -> None:
    """The texture stage needs a material that is actually bound in the asset."""

    text = (
        ROOT
        / "docs/arm_decision_proof_v1/assets"
        / "adp009d_content_agents_articulated_texture.vast.yaml"
    ).read_text(encoding="utf-8")

    assert "material_path: /Asset/Looks/Render/" in text
    assert "physics_materials" not in text
