"""CAD transport is hermetic: fake kernel outputs never become native evidence."""
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import trimesh

from blueprint_pipeline import website_synthetic_cad as cad
from blueprint_pipeline.website_synthetic_subject import write_candidate
from tests.test_website_task_preparation import SYNTHETIC_BALL


@pytest.fixture
def fake_cad_kernel(tmp_path, monkeypatch):
    """Replace the existing CAD subprocess seam, not the writer or its validators."""
    from blueprint_pipeline import asset_authoring_sandbox, production_cad_skill_sources, task_object_agent_cad
    calls = []
    original_version = cad.importlib.metadata.version
    monkeypatch.setattr(cad.importlib.metadata, "version",
        lambda name: "0.11.1" if name == "build123d" else original_version(name))
    root = tmp_path / "sealed-cad-sources"
    root.mkdir()
    for spec in production_cad_skill_sources.SOURCE_SPECS:
        (root / f"{spec['id']}-{spec['commit'][:8]}").mkdir()
    monkeypatch.setattr(production_cad_skill_sources, "DEFAULT_ROOT", str(root))
    class Sandbox:
        def __init__(self, **kwargs):
            calls.append(("sandbox", kwargs))
        def preflight(self): calls.append(("preflight",))
        def __call__(self, argv, **kwargs):
            calls.append(("snapshot", argv, kwargs))
            return SimpleNamespace(returncode=1, stdout="", stderr="Hermetic fixture: no browser")
    monkeypatch.setattr(asset_authoring_sandbox, "SandboxedAssetRunner", Sandbox)
    def execute(**kwargs):
        calls.append(("compile", kwargs))
        kwargs["sandbox"].preflight()
        output = kwargs["output_root"]
        output.mkdir()
        d = np.asarray(kwargs["request"].dimensions_m)*1000
        program = kwargs["program"]
        mesh = (trimesh.creation.icosphere(subdivisions=3, radius=d[0]/2) if "candidate = Sphere" in program else
                trimesh.creation.cylinder(radius=d[0]/2, height=d[2], sections=64) if "candidate = Cylinder" in program else
                trimesh.creation.box(extents=d))
        (output / "asset.py").write_text(program)
        (output / "candidate.step").write_text("Hermetic STEP byte fixture, not a kernel result")
        mesh.export(output / "candidate.stl")
        sources = {name: {"commit": production_cad_skill_sources.SOURCE_SPECS[index]["commit"]}
                   for name, index in [("cad", 0), ("mac", 1)]}
        (output / "source-receipt.json").write_text(json.dumps(sources))
        readback = {"passed": True, "build123d": "0.11.1", "solid_count": 1,
                    "measured_dimensions_mm": d.tolist(), "valid": True, "volume_mm3": float(mesh.volume)}
        (output / "step-readback.json").write_text(json.dumps(readback))
        result = {"passed": True, "readback": readback, "step_path": str(output / "candidate.step"),
                  "stl_path": str(output / "candidate.stl"), "source_unchanged_after": True}
        (output / "candidate-receipt.json").write_text(json.dumps(result))
        return result
    monkeypatch.setattr(task_object_agent_cad, "execute_cad_program", execute)
    return calls


@pytest.mark.parametrize("shape,dimensions", [("sphere", [.08]*3), ("box", [.08, .06, .04]),
                                             ("cylinder", [.08, .08, .04])])
def test_pinned_compiler_artifacts_and_independent_mesh_readback_travel_together(tmp_path, fake_cad_kernel, shape, dimensions):
    design = {**SYNTHETIC_BALL, "shape": shape, "dimensions_m": dimensions}
    path = tmp_path / "candidate.usda"
    result = write_candidate(design, center_xyz_m=[.5, -.1, .3], path=path,
                             authoring=cad.SyntheticAuthoring().model_dump())
    evidence = result["cad_authoring"]
    archive = path.parent / evidence["evidence"]["path"]
    manifest = cad.verify_evidence(archive, design=design, expected_manifest_digest=evidence["manifest_digest"])
    assert manifest["claim_ceiling"] == "development_only" and not manifest["native_qualified"]
    assert manifest["stl_readback"]["passed"] and manifest["step_readback"]["passed"]
    assert {"asset.py", "candidate.step", "candidate.stl", "parameters.json", "source-receipt.json"} <= set(manifest["files"])
    assert manifest["provider_called"] is False and any(c[0] == "snapshot" for c in fake_cad_kernel)
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(path))
    vertices = np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath("/Root/SourceObjectCandidate/Designed")).GetPointsAttr().Get())
    assert vertices.max(axis=0)-vertices.min(axis=0) == pytest.approx(dimensions, abs=1e-6)
    assert vertices.mean(axis=0) == pytest.approx([.5, -.1, .3], abs=.001)
    with zipfile.ZipFile(archive) as original:
        members = {n: original.read(n) for n in original.namelist()}
    members["candidate.step"] = b"Changed retained export"
    with zipfile.ZipFile(archive, "w") as changed:
        for n, data in members.items():
            changed.writestr(n, data)
    with pytest.raises(ValueError, match="evidence_changed"):
        cad.verify_evidence(archive, design=design)


def test_missing_kernel_sandbox_is_a_named_hold_without_legacy_fallback(tmp_path, fake_cad_kernel, monkeypatch):
    from blueprint_pipeline import task_object_agent_cad
    from blueprint_pipeline.asset_authoring_sandbox import AssetSandboxError
    def blocked(**kwargs): raise AssetSandboxError("asset_sandbox_unavailable")
    monkeypatch.setattr(task_object_agent_cad, "execute_cad_program", blocked)
    with pytest.raises(ValueError, match="website_synthetic_cad_sandbox_unavailable"):
        write_candidate(SYNTHETIC_BALL, center_xyz_m=[0, 0, .0325], path=tmp_path / "candidate.usda",
                        authoring=cad.SyntheticAuthoring().model_dump())
    assert not (tmp_path / "candidate.usda").exists()


def test_legacy_geometry_stays_legacy_and_new_preparations_choose_cad(tmp_path, monkeypatch):
    from blueprint_pipeline.website_task_preparation import compile_website_scene_preparation
    from tests.test_website_task_preparation import synthetic_arguments, _arguments
    def no_cad(*args, **kwargs): raise AssertionError("Legacy source must not invoke CAD")
    monkeypatch.setattr(cad, "author_candidate", no_cad)
    result = write_candidate(SYNTHETIC_BALL, center_xyz_m=[0, 0, .0325], path=tmp_path / "legacy.usda")
    assert "cad_authoring" not in result and result["sphere_subdivisions"] == 4
    (tmp_path / "prepare").mkdir()
    args = synthetic_arguments(_arguments(tmp_path / "prepare"))
    preparation = compile_website_scene_preparation(**args)
    assert preparation["authoring_inputs"]["configuration"]["synthetic_authoring"] == cad.SyntheticAuthoring().model_dump()
    with pytest.raises(ValueError):
        compile_website_scene_preparation(**args, synthetic_authoring_backend="arbitrary_executable")


def test_cad_zip_is_published_and_reopened_by_real_native_input_validator(tmp_path, monkeypatch, fake_cad_kernel):
    from tests.test_website_native_submission import setup
    from blueprint_pipeline.website_native_submission import materialize_website_submission
    from blueprint_pipeline.task_evaluation_scene_configuration_submission_publication import _validated_inventory
    from tests.test_task_evaluation_scene_configuration_submission import SHA
    (tmp_path / "publication").mkdir()
    kwargs, _ = setup(tmp_path / "publication", monkeypatch, synthetic=True, synthetic_cad=True)
    materialize_website_submission(**kwargs)
    request = json.loads((kwargs["staging_root"] / "scene_configuration_preparation_request.v1.json").read_text())
    assert request["scene"]["website_native_inputs"]["frames"] == []
    assert request["scene"]["website_native_inputs"]["cad_evidence"]["size_bytes"] > 0
    _validated_inventory(kwargs["staging_root"], SHA)


def test_worker_constructs_from_transport_after_original_cad_sources_are_unavailable(tmp_path, monkeypatch, fake_cad_kernel):
    import tests.test_website_native_background as fixture
    from tests.test_website_task_preparation import synthetic_arguments
    from blueprint_pipeline.task_evaluation_completed_scene_adapters import execute_provided_mesh_authoring
    real = fixture._arguments
    monkeypatch.setattr(fixture, "_arguments", lambda root: synthetic_arguments(real(root)))
    runtime, inputs, envelope, config_path = fixture.staged(tmp_path)
    second = fixture.execute(tmp_path, inputs, envelope, config_path)
    assert any(row["role"] == "source_cad_authoring_evidence" for row in second["output_artifacts"])
    third = runtime["object_authoring"]["configuration"]
    third_path = tmp_path / "stage-3.json"
    third_path.write_text(json.dumps(third))
    output = tmp_path / "stage-3"
    output.mkdir()
    result = execute_provided_mesh_authoring(envelope={},
        stage={"stage_id": "stage-3", "capability": "rigid_replacement_authoring", "execution_class": "no_spend",
               "adapter": {"id": "provided_mesh_rigid_authoring", "version": "v1"}},
        configuration=third, configuration_path=third_path, dependency_results=(second,), output_root=output)
    receipt = json.loads((output / "replacement_authoring_receipt.v1.json").read_text())
    assert result["status"] == "completed" and receipt["model_called"] is False
    assert receipt["source_candidate_claim"] == "synthetic_design_candidate_geometry"
    assert receipt["cad_authoring_manifest_digest"].startswith("sha256:")
    assert receipt["candidate_physics_completion"]["metric_envelope_validation"]["status"] == "within_preregistered_metric_envelope"
    cad_zip = next(row for row in second["output_artifacts"] if row["role"] == "source_cad_authoring_evidence")
    Path(cad_zip["path"]).write_bytes(b"changed archive")
    with pytest.raises(ValueError, match="artifact_changed"):
        execute_provided_mesh_authoring(envelope={},
            stage={"execution_class": "no_spend", "adapter": {"id": "provided_mesh_rigid_authoring"}},
            configuration=third, configuration_path=third_path, dependency_results=(second,), output_root=tmp_path / "bad")


def test_completed_identical_cad_is_adopted_without_compiling_again(tmp_path, fake_cad_kernel, monkeypatch):
    from blueprint_pipeline import task_object_agent_cad
    first = tmp_path / "first.usda"
    selected = cad.SyntheticAuthoring().model_dump()
    original = write_candidate(SYNTHETIC_BALL, center_xyz_m=[0, 0, .0325], path=first, authoring=selected)
    def forbidden(**kwargs): raise AssertionError("Completed CAD must be adopted")
    monkeypatch.setattr(task_object_agent_cad, "execute_cad_program", forbidden)
    adopted = write_candidate(SYNTHETIC_BALL, center_xyz_m=[0, 0, .0325], path=tmp_path / "second.usda", authoring=selected)
    assert adopted == original
    with pytest.raises(ValueError, match="evidence_invalid"):
        write_candidate({**SYNTHETIC_BALL, "mass_kg": .1}, center_xyz_m=[0, 0, .0325],
                        path=tmp_path / "wrong-design.usda", authoring=selected)
