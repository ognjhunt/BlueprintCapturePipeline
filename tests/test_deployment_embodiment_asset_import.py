"""Development-only importer contract: fake official SDK, real USD structure/receipts.

No Isaac app, GPU, source network request or native episode is executed here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import deployment_embodiment_asset_import as importer
from blueprint_pipeline import deployment_embodiment_assets as assets
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest


def _source(tmp_path, monkeypatch, format_name="urdf", *, content=None, extra=None, arguments=None):
    entry = "robot." + ("xml" if format_name == "mjcf" else format_name)
    recipe = {"embodiment_id": "fixture_dual_option", "arm_count": 2,
              "status": "source_available", "source": {
                  "repository_url": "https://github.com/fixture/source", "revision": "a" * 40,
                  "entry_path": entry, "format": format_name, "license_id": "MIT", "license_path": "LICENSE",
                  "license_note": "Development fixture only", "include_paths": [entry, "LICENSE"],
                  "import_arguments": {}, "package_roots": {"fixture": "."}, "source_arm_count": 1},
              "composition_required": ["exact_dual_mount_transform", "actual_end_effectors_and_cameras"],
              "missing": ["controller_reset_and_runtime_receipt"]}
    recipe["source"]["include_paths"] += list(extra or {})
    recipe["source"]["import_arguments"] = arguments or {}
    monkeypatch.setattr(assets, "recipe_for", lambda identity: recipe)
    monkeypatch.setattr(importer, "recipe_for", lambda identity: recipe)
    files = {entry: content or b'<robot name="fixture"><link name="base"/></robot>',
             "LICENSE": b"MIT fixture notice\n", **(extra or {})}
    rows = [{"path": name, "type": "blob", "mode": "100644", "size": len(value),
             "sha": hashlib.sha1(b"blob " + str(len(value)).encode() + b"\0" + value).hexdigest()}
            for name, value in files.items()]

    def get(url, *, maximum):
        return (json.dumps({"truncated": False, "tree": rows}).encode() if "/git/trees/" in url
                else files[url.split("/" + "a" * 40 + "/", 1)[1]])

    source = tmp_path / "sources"
    assets.materialize_sources(recipe["embodiment_id"], source, get=get)
    return source


def _sdk(monkeypatch, asset=None):
    calls = []

    class Config:
        def __init__(self, **kwargs):
            self.values = kwargs

    class Converter:
        def __init__(self, config):
            from pxr import Usd, UsdPhysics

            calls.append(config.values)
            path = Path(config.values["usd_dir"]) / "fixture.usda"
            path.parent.mkdir(parents=True, exist_ok=True)
            stage = Usd.Stage.CreateNew(str(path))
            stage.SetDefaultPrim(stage.DefinePrim("/robot", "Xform"))
            for name in ("base", "tool"):
                UsdPhysics.RigidBodyAPI.Apply(stage.DefinePrim("/robot/" + name, "Xform"))
            UsdPhysics.RevoluteJoint.Define(stage, "/robot/joint")
            if config.values["fix_base"]:
                UsdPhysics.FixedJoint.Define(stage, "/robot/anchor").CreateBody1Rel().SetTargets(["/robot/base"])
            if asset is not None:
                from pxr import Sdf
                stage.GetPrimAtPath("/robot/tool").CreateAttribute("texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(asset))
            stage.GetRootLayer().Save()
            self.usd_path = str(path)

    sdk = SimpleNamespace(UrdfConverterCfg=Config, UrdfConverter=Converter,
                          MjcfConverterCfg=Config, MjcfConverter=Converter)
    real_import = importer.importlib.import_module
    monkeypatch.setattr(importer.importlib, "import_module",
                        lambda name, *args: sdk if name == "isaaclab.sim.converters" else real_import(name, *args))
    return calls


@pytest.mark.parametrize("format_name", ["urdf", "mjcf"])
def test_official_converter_options_preserve_source_and_produce_unqualified_digest_bound_usd(tmp_path, monkeypatch, format_name):
    source = _source(tmp_path, monkeypatch, format_name)
    before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    calls = _sdk(monkeypatch)
    destination = tmp_path / "prepared"
    receipt = importer.import_source_asset(source, destination, fix_base=True)
    settings = calls[0]
    assert settings["fix_base"] is True and settings["link_density"] == 0.0
    assert settings["collision_from_visuals"] is False and settings["merge_mesh"] is False
    if format_name == "urdf":
        assert settings["merge_fixed_joints"] is False and settings["joint_drive"] is None
        assert settings["ros_package_paths"] == [{"name": "fixture", "path": str(source / "source")}]
    else:
        assert settings["override_gain_type"] is None and settings["override_gain_prm"] is None
    assert receipt["source_arm_count"] == 1  # A dual option never assembles a second arm automatically.
    assert receipt["execution_authorized"] is False and receipt["claim_ceiling"] == "source_asset_not_qualified"
    assert receipt["structure"]["joint_names"] == ["joint"]
    assert receipt["structure"]["world_anchor_links"] == ["base"]
    assert receipt["composition_required"] == ["exact_dual_mount_transform", "actual_end_effectors_and_cameras"]
    assert receipt["model"]["sha256"] == "sha256:" + hashlib.sha256((destination / "robot.usda").read_bytes()).hexdigest()
    assert receipt["import_digest"] == cross_runtime_canonical_digest(receipt, digest_field="import_digest")
    assert before == {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    assert importer.import_source_asset(source, destination, fix_base=True) == receipt
    assert len(calls) == 1  # A retry verifies retained output without re-importing.
    (destination / "robot.usda").write_text("changed")
    with pytest.raises(assets.AssetPreparationError, match="output_changed"):
        importer.import_source_asset(source, destination, fix_base=True)


@pytest.mark.parametrize("format_name,missing", [("xacro", "xacro_runtime_unavailable"),
                                                  ("urdf", "initialized_isaac_runtime_unavailable")])
def test_missing_isaac_or_xacro_reports_unavailable_without_partial_destination(tmp_path, monkeypatch, format_name, missing):
    source = _source(tmp_path, monkeypatch, format_name)
    real_import = importer.importlib.import_module

    def unavailable(name, *args):
        if name in {"xacro", "isaaclab.sim.converters"}:
            raise ModuleNotFoundError(name)
        return real_import(name, *args)

    monkeypatch.setattr(importer.importlib, "import_module", unavailable)
    with pytest.raises(assets.AssetPreparationError, match=missing):
        importer.import_source_asset(source, tmp_path / "prepared", fix_base=True)
    assert not (tmp_path / "prepared").exists()


def test_changed_source_and_missing_explicit_base_mode_cannot_be_imported(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch)
    _sdk(monkeypatch)
    with pytest.raises(assets.AssetPreparationError, match="base_mode_required"):
        importer.import_source_asset(source, tmp_path / "prepared", fix_base=None)
    (source / "source/robot.urdf").write_bytes(b"changed")
    with pytest.raises(assets.AssetPreparationError, match="file_changed"):
        importer.import_source_asset(source, tmp_path / "prepared", fix_base=True)
    assert not (tmp_path / "prepared").exists()


def test_importer_internalizes_source_texture_and_rejects_unbound_paths(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch, extra={"textures/tool.png": b"fixture texture bytes"})
    _sdk(monkeypatch, asset=str(source / "source/textures/tool.png"))
    result = importer.import_source_asset(source, tmp_path / "prepared", fix_base=True)
    assert {row["location"] for row in result["dependencies"]} == {"prepared"}
    assert {row["path"] for row in result["dependencies"]} == {"robot.usda", "retained_source/textures/tool.png"}
    assert result["source_bundle_required_for_dependency_resolution"] is False
    assert result["prepared_directory_self_contained"] is True
    assert result["single_file_self_contained"] is False
    assert str(source) not in (tmp_path / "prepared/robot.usda").read_text()
    # The deliberately fake texture stands in only for an asset-path binding;
    # no renderer or simulator interprets it.
    _sdk(monkeypatch, asset="https://unapproved.example/robot.usd")
    with pytest.raises(assets.AssetPreparationError, match="unbound_dependency"):
        importer.import_source_asset(source, tmp_path / "unbound", fix_base=True)
    assert not (tmp_path / "unbound").exists()


def test_published_textured_model_loads_after_source_and_preparation_move(tmp_path, monkeypatch):
    """Exercise a real USD asset value after both original roots disappear."""
    from pxr import Sdf, Usd

    source = _source(tmp_path, monkeypatch, extra={"textures/tool.png": b"fixture texture bytes"})
    _sdk(monkeypatch, asset=str(source / "source/textures/tool.png"))
    prepared = tmp_path / "prepared"
    result = importer.import_source_asset(source, prepared, fix_base=True)
    source.chmod(0o755)  # The test owns these immutable publication directories.
    source.rename(tmp_path / "source-moved")
    moved = tmp_path / "moved" / "robot-preparation"
    moved.parent.mkdir()
    prepared.chmod(0o755)
    prepared.rename(moved)
    stage = Usd.Stage.Open(str(moved / "robot.usda"))
    value = stage.GetPrimAtPath("/robot/tool").GetAttribute("texture").Get()
    assert value.path == "retained_source/textures/tool.png"
    resolved = Path(Sdf.ComputeAssetPathRelativeToLayer(stage.GetRootLayer(), value.path))
    assert resolved.read_bytes() == b"fixture texture bytes"
    observed = importer._dependencies(moved / "robot.usda", staged=moved)
    assert importer._closure(observed) == result["dependency_closure"]
    assert importer.verify_prepared_asset_directory(moved) == result
    assert result["dependency_closure"]["closure_digest"] == cross_runtime_canonical_digest(
        result["dependency_closure"], digest_field="closure_digest")


def test_runtime_content_sidecars_are_inventoried_and_changed_texture_is_refused(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch, extra={
        "mesh/tool.obj": b"mtllib tool.mtl\nv 0 0 0\n",
        "mesh/tool.mtl": b"map_Kd ../textures/tool.png\n",
        "textures/tool.png": b"fixture texture bytes"})
    _sdk(monkeypatch, asset=str(source / "source/mesh/tool.obj"))
    prepared = tmp_path / "prepared"
    result = importer.import_source_asset(source, prepared, fix_base=True)
    assert {row["path"] for row in result["dependency_closure"]["files"]} == {
        "robot.usda", "retained_source/mesh/tool.obj", "retained_source/mesh/tool.mtl",
        "retained_source/textures/tool.png"}
    (prepared / "retained_source/textures/tool.png").write_bytes(b"changed")
    with pytest.raises(assets.AssetPreparationError, match="output_changed"):
        importer.import_source_asset(source, prepared, fix_base=True)


def test_converter_sublayers_are_flattened_and_internal_texture_path_survives_publish(tmp_path, monkeypatch):
    from pxr import Sdf, Usd, UsdPhysics

    source = _source(tmp_path, monkeypatch)

    def convert(format_name, entry, output, **kwargs):
        output.mkdir()
        content = output / "content"
        content.mkdir()
        (content / "texture.png").write_bytes(b"fixture converter texture")
        child = Usd.Stage.CreateNew(str(content / "child.usda"))
        child.SetDefaultPrim(child.DefinePrim("/robot", "Xform"))
        UsdPhysics.RigidBodyAPI.Apply(child.DefinePrim("/robot/base", "Xform"))
        UsdPhysics.RevoluteJoint.Define(child, "/robot/joint")
        child.GetPrimAtPath("/robot/base").CreateAttribute("texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath("texture.png"))
        child.GetRootLayer().Save()
        stage = Usd.Stage.CreateNew(str(output / "robot.usda"))
        stage.GetRootLayer().subLayerPaths = ["content/child.usda"]
        stage.SetDefaultPrim(stage.GetPrimAtPath("/robot"))
        stage.GetRootLayer().Save()
        return output / "robot.usda", {}, {"class": "fixture only"}

    monkeypatch.setattr(importer, "_converter", convert)
    prepared = tmp_path / "prepared"
    result = importer.import_source_asset(source, prepared, fix_base=True)
    assert not Sdf.Layer.FindOrOpen(str(prepared / "robot.usda")).subLayerPaths
    assert result["structure"]["joint_names"] == ["joint"]
    assert {row["path"] for row in result["dependency_closure"]["files"]} == {
        "robot.usda", "converted/content/texture.png"}
    assert ".blueprint-asset-import-" not in (prepared / "robot.usda").read_text()


def test_redigested_receipt_cannot_claim_external_or_different_content_closure(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch)
    _sdk(monkeypatch)
    prepared = tmp_path / "prepared"
    result = importer.import_source_asset(source, prepared, fix_base=True)
    result["dependency_closure"]["files"] = []
    result["dependency_closure"]["closure_digest"] = cross_runtime_canonical_digest(
        result["dependency_closure"], digest_field="closure_digest")
    result["import_digest"] = cross_runtime_canonical_digest(result, digest_field="import_digest")
    (prepared / importer.RECEIPT).write_text(json.dumps(result))
    with pytest.raises(assets.AssetPreparationError, match="dependency_changed"):
        importer.import_source_asset(source, prepared, fix_base=True)


@pytest.mark.parametrize("defect", ["escape", "absolute", "duplicate", "oversized", "total_bytes", "model_binding", "symlink"])
def test_public_prepared_verifier_refuses_unsafe_or_unbounded_inventory(tmp_path, monkeypatch, defect):
    source = _source(tmp_path, monkeypatch)
    _sdk(monkeypatch)
    prepared = tmp_path / "prepared"
    result = importer.import_source_asset(source, prepared, fix_base=True)
    row = result["files"][0]
    if defect == "escape":
        row["path"] = "../outside.usda"
    elif defect == "absolute":
        row["path"] = str(tmp_path / "outside.usda")
    elif defect == "duplicate":
        result["files"].append(dict(row))
    elif defect == "oversized":
        row["size_bytes"] = importer.MAX_PREPARED_FILE_BYTES + 1
    elif defect == "total_bytes":
        monkeypatch.setattr(importer, "MAX_PREPARED_BYTES", 1)
    elif defect == "model_binding":
        result["model"]["sha256"] = "sha256:" + "0" * 64
    else:
        prepared.chmod(0o755)
        (prepared / "extra-link").symlink_to(tmp_path / "outside")
    result["import_digest"] = cross_runtime_canonical_digest(result, digest_field="import_digest")
    (prepared / importer.RECEIPT).write_text(json.dumps(result))
    with pytest.raises(assets.AssetPreparationError, match="asset_import_"):
        importer.verify_prepared_asset_directory(prepared)


def test_public_verifier_reads_current_usd_reference_bytes_without_network_or_layer_cache(tmp_path, monkeypatch):
    from pxr import Sdf, Usd

    source = _source(tmp_path, monkeypatch)
    _sdk(monkeypatch)
    prepared = tmp_path / "prepared"
    result = importer.import_source_asset(source, prepared, fix_base=True)
    prepared.chmod(0o755)
    stage = Usd.Stage.Open(str(prepared / "robot.usda"))
    stage.GetPrimAtPath("/robot/tool").CreateAttribute("texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("https://must-not-resolve.example/texture.png"))
    stage.GetRootLayer().Save()
    model_bytes = (prepared / "robot.usda").read_bytes()
    updated = {"path": "robot.usda", "size_bytes": len(model_bytes),
               "sha256": "sha256:" + hashlib.sha256(model_bytes).hexdigest()}
    result["files"] = [updated if row["path"] == "robot.usda" else row for row in result["files"]]
    result["model"] = dict(updated)
    result["import_digest"] = cross_runtime_canonical_digest(result, digest_field="import_digest")
    (prepared / importer.RECEIPT).write_text(json.dumps(result))
    with pytest.raises(assets.AssetPreparationError, match="unbound_dependency"):
        importer.verify_prepared_asset_directory(prepared)


def test_explicit_station_is_retained_reproducible_and_uses_the_same_official_converter(tmp_path, monkeypatch):
    import importlib.util
    import xml.etree.ElementTree as ET
    from blueprint_pipeline import deployment_embodiment_station as station

    source = _source(tmp_path, monkeypatch, content=b'''<robot name="fixture"><link name="base"/><link name="tool"/>
      <joint name="joint" type="revolute"><parent link="base"/><child link="tool"/><axis xyz="0 0 1"/>
      <limit lower="-2" upper="2" effort="1" velocity="1"/></joint></robot>''')
    monkeypatch.setattr(station, "recipe_for", importer.recipe_for)
    calls = _sdk(monkeypatch)
    params = {"schema_version": station.PARAMETERS_SCHEMA, "embodiment_id": "fixture_dual_option",
              "composition_mode": "two_instances", "station_root_link": "station", "arms": [
                  {"name": name, "prefix": prefix, "source_base_link": "base", "source_mount_joint": None,
                   "station_from_base": {"translation_m": xyz, "quaternion_wxyz": [1, 0, 0, 0]}}
                  for name, prefix, xyz in (("left", "L_", [0, .3, 0]), ("right", "R_", [0, -.3, 0]))]}
    prepared = tmp_path / "prepared"
    result = importer.import_source_asset(source, prepared, fix_base=True, station_parameters=params)
    assert calls[0]["asset_path"].endswith("/station/robot.urdf")
    assert result["composed_arm_count"] == 2 and result["source_arm_count"] == 1
    assert result["station_composition"]["composition"]["parameters"] == params
    model = ET.parse(prepared / "station/robot.urdf")
    assert {node.get("name") for node in model.findall("joint")} == {
        "L_joint", "R_joint", "left_station_mount", "right_station_mount"}
    spec = importlib.util.spec_from_file_location("retained_station_generator", prepared / "station/station.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    assert generator.gen_urdf() == (prepared / "station/robot.urdf").read_text()
    assert importer.import_source_asset(source, prepared, fix_base=True, station_parameters=params) == result
    changed = json.loads(json.dumps(params))
    changed["arms"][1]["station_from_base"]["translation_m"][0] = .1
    with pytest.raises(assets.AssetPreparationError, match="station_parameters_changed"):
        importer.import_source_asset(source, prepared, fix_base=True, station_parameters=changed)


def _prepared_operator_case(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch, extra={"textures/tool.png": b"real retained fixture content"})
    _sdk(monkeypatch, asset=str(source / "source/textures/tool.png"))
    prepared = tmp_path / "prepared"
    receipt = importer.import_source_asset(source, prepared, fix_base=True)
    url = "https://assets.blueprint.example/operator/team-a/robot.usd"
    manifest = {"model": {"url": url, "sha256": receipt["model"]["sha256"]}}
    profile = {"prepared_robot_models": [{
        "team_id": "team-a", "checkpoint_id": "checkpoint-a", "model_url": url,
        "sha256": receipt["model"]["sha256"], "directory": str(prepared),
        "closure_digest": receipt["dependency_closure"]["closure_digest"]}]}
    return prepared, receipt, profile, manifest


def test_real_import_receipt_survives_operator_owner_mapping_and_sealed_bundle_copy(tmp_path, monkeypatch):
    from pxr import Sdf, Usd
    from blueprint_pipeline.team_robot_model_delivery import copy_prepared_closure, stage_prepared_robot_model

    prepared, receipt, profile, manifest = _prepared_operator_case(tmp_path, monkeypatch)
    staged, bundle = tmp_path / "operator-staged", tmp_path / "worker-bundle"
    delivery = stage_prepared_robot_model(profile=profile, manifest=manifest, team_id="team-a",
                                         checkpoint_id="checkpoint-a", target_directory=staged)
    assert delivery["source_import_digest"] == receipt["import_digest"]
    assert delivery["source_import_provenance"] == receipt
    assert delivery["sha256"] == receipt["model"]["sha256"]
    assert delivery["dependency_closure"] == receipt["dependency_closure"]
    assert delivery["team_id"] == "team-a" and delivery["checkpoint_id"] == "checkpoint-a"
    copy_prepared_closure(source=staged, destination=bundle, closure=delivery["dependency_closure"])
    # The actual copied USD reads the retained texture through its relative path.
    stage = Usd.Stage.Open(str(bundle / "robot.usda"))
    value = stage.GetPrimAtPath("/robot/tool").GetAttribute("texture").Get()
    path = Path(Sdf.ComputeAssetPathRelativeToLayer(stage.GetRootLayer(), value.path))
    assert path.is_relative_to(bundle) and path.read_bytes() == b"real retained fixture content"
    assert {r["path"] for r in receipt["dependency_closure"]["files"]} == {
        "robot.usda", "retained_source/textures/tool.png"}
    assert importer.verify_prepared_asset_directory(prepared) == receipt


@pytest.mark.parametrize("mutation", ["texture", "owner", "checkpoint"])
def test_operator_mapping_refuses_changed_sidecar_or_wrong_candidate_before_staging(tmp_path, monkeypatch, mutation):
    from blueprint_pipeline.team_robot_model_delivery import RobotModelDeliveryError, stage_prepared_robot_model

    prepared, _, profile, manifest = _prepared_operator_case(tmp_path, monkeypatch)
    if mutation == "texture":
        (prepared / "retained_source/textures/tool.png").write_bytes(b"changed")
    with pytest.raises(RobotModelDeliveryError, match="prepared_asset_invalid|owner_binding_missing"):
        stage_prepared_robot_model(profile=profile, manifest=manifest,
            team_id="other-team" if mutation == "owner" else "team-a",
            checkpoint_id="other-checkpoint" if mutation == "checkpoint" else "checkpoint-a",
            target_directory=tmp_path / "must-not-stage")
    assert not (tmp_path / "must-not-stage").exists()


@pytest.mark.parametrize("initialize,failure", [(False, False), (True, False), (True, True)])
def test_cli_only_initializes_isaac_explicitly_and_closes_app(tmp_path, monkeypatch, capsys, initialize, failure):
    import sys

    calls = []
    app = SimpleNamespace(close=lambda: calls.append("close"))
    monkeypatch.setattr(importer, "verify_source_bundle", lambda *args, **kwargs: calls.append("verify"))

    class Launcher:
        def __init__(self, **kwargs):
            assert kwargs == {"headless": True}
            calls.append("start")
            self.app = app

    monkeypatch.setitem(sys.modules, "isaaclab.app", SimpleNamespace(AppLauncher=Launcher))

    def run(*args, **kwargs):
        if failure:
            raise assets.AssetPreparationError("fixture_import_failure")
        return {"execution_authorized": False}

    monkeypatch.setattr(importer, "import_source_asset", run)
    argv = ["--source-bundle", str(tmp_path), "--output", str(tmp_path / "out"), "--fixed-base"]
    if initialize:
        argv.append("--initialize-isaac")
    if failure:
        with pytest.raises(SystemExit) as caught:
            importer.main(argv)
        assert caught.value.code == 2
    else:
        assert importer.main(argv) == 0
        assert json.loads(capsys.readouterr().out)["execution_authorized"] is False
    assert calls == (["verify", "start", "close"] if initialize else ["verify"])


@pytest.mark.parametrize("format_name,content,extra", [
    ("urdf", b'<robot><mesh filename="/private/forbidden.stl"/></robot>', {}),
    ("urdf", b'<robot><mesh filename="file:///private/forbidden.stl"/></robot>', {}),
    ("urdf", b'<robot><mesh filename="https://forbidden.example/mesh.stl"/></robot>', {}),
    ("urdf", b'<robot><mesh filename="package://unknown/mesh.stl"/></robot>', {}),
    ("mjcf", b'<mujoco><include file="../../forbidden.xml"/></mujoco>', {}),
    ("mjcf", b'<mujoco><compiler meshdir="../../private"/><asset><mesh file="mesh.stl"/></asset></mujoco>', {}),
    ("mjcf", b'<mujoco><include file="child.xml"/></mujoco>',
     {"child.xml": b'<mujocoinclude><mesh file="/private/forbidden.stl"/></mujocoinclude>'}),
    ("urdf", b'<robot><mesh filename="bad.obj"/></robot>',
     {"bad.obj": b'mtllib\t/private/forbidden.mtl\n'}),
    ("urdf", b'<robot><mesh filename="bad.dae"/></robot>',
     {"bad.dae": b'<COLLADA><image><init_from>file:///private/forbidden.png</init_from></image></COLLADA>'}),
])
def test_source_reference_escape_is_refused_before_converter_reads(tmp_path, monkeypatch, format_name, content, extra):
    source = _source(tmp_path, monkeypatch, format_name, content=content, extra=extra)
    calls = []
    monkeypatch.setattr(importer, "_converter", lambda *args, **kwargs: calls.append("converter"))
    with pytest.raises(assets.AssetPreparationError, match="asset_import_input_"):
        importer.import_source_asset(source, tmp_path / "prepared", fix_base=True)
    assert calls == [] and not (tmp_path / "prepared").exists()


def _xacro_module(monkeypatch, action):
    from types import ModuleType
    import xml.dom.minidom

    module = ModuleType("xacro")
    module.action = action
    module.make_document = lambda: xml.dom.minidom.parseString('<robot name="fixture"><link name="base"/></robot>')
    # The real xacro read boundary uses these functions' module globals.
    exec("""
def parse(inp, filename=None):
    with open(filename) as stream:
        return stream.read()
def load_yaml(filename):
    with open(filename) as stream:
        return stream.read()
def eval_extension(value):
    raise AssertionError('unexpected original extension evaluation')
def abs_filename_spec(value):
    return value
def eval_text(value, symbols):
    return value
def get_include_files(value, symbols):
    raise AssertionError('unexpected original include resolver')
def process_file(filename, mappings):
    parse(None, filename)
    action(filename, mappings)
    return make_document()
""", module.__dict__)
    original = object()
    module.open = original
    real_import = importer.importlib.import_module
    monkeypatch.setattr(importer.importlib, "import_module",
                        lambda name, *args: module if name == "xacro" else real_import(name, *args))
    return module, original


def test_dynamic_xacro_yaml_escape_is_refused_before_read_and_restores_opener(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch, "xacro", content=b'<robot xmlns:xacro="http://wiki.ros.org/xacro">'
                     b'<xacro:property name="data" value="${xacro.load_yaml(params)}"/></robot>')
    module, original = _xacro_module(monkeypatch, lambda filename, args: module.load_yaml("/private/forbidden.yaml"))
    calls = []
    monkeypatch.setattr(importer, "_converter", lambda *args, **kwargs: calls.append("converter"))
    with pytest.raises(assets.AssetPreparationError, match="input_reference_unbound"):
        importer.import_source_asset(source, tmp_path / "prepared", fix_base=True)
    assert calls == [] and module.open is original


def test_dynamic_xacro_yaml_within_inventory_preserves_arguments_and_package_alias(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch, "xacro", extra={"config/fixture.yaml": b"mass: 1.0\n"},
                     arguments={"robot_type": "fixture"})
    calls = _sdk(monkeypatch)
    observed = []

    def action(filename, arguments):
        alias = module.eval_extension("$(find fixture)")
        observed.append(module.load_yaml(str(Path(alias) / "config" / (arguments["robot_type"] + ".yaml"))))

    module, original = _xacro_module(monkeypatch, action)
    receipt = importer.import_source_asset(source, tmp_path / "prepared", fix_base=True)
    assert observed == ["mass: 1.0\n"] and module.open is original
    assert len(calls) == 1 and receipt["execution_authorized"] is False


def test_xacro_environment_eval_is_refused_and_module_hooks_restore(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch, "xacro")
    module, original = _xacro_module(monkeypatch, lambda filename, args: module.eval_extension("$(eval env('SECRET'))"))
    calls = []
    monkeypatch.setattr(importer, "_converter", lambda *args, **kwargs: calls.append("converter"))
    original_extension = module.eval_extension
    with pytest.raises(assets.AssetPreparationError, match="environment_or_eval_substitution_refused"):
        importer.import_source_asset(source, tmp_path / "prepared", fix_base=True)
    assert calls == [] and module.open is original and module.eval_extension is original_extension


def test_xacro_dynamic_include_escape_is_refused_and_hooks_restore(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch, "xacro")
    module, original = _xacro_module(monkeypatch, lambda filename, args: list(module.get_include_files("/private/forbidden.xml", {})))
    include = module.get_include_files
    calls = []
    monkeypatch.setattr(importer, "_converter", lambda *args, **kwargs: calls.append("converter"))
    with pytest.raises(assets.AssetPreparationError, match="input_reference_unbound"):
        importer.import_source_asset(source, tmp_path / "prepared", fix_base=True)
    assert calls == [] and module.open is original and module.get_include_files is include


def test_xacro_module_read_scope_is_serialized_between_preparers(tmp_path, monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor

    source = _source(tmp_path, monkeypatch, "xacro")
    _sdk(monkeypatch)
    active = []
    overlap = []

    def action(filename, arguments):
        active.append(filename)
        overlap.append(len(active))
        time.sleep(0.02)
        active.remove(filename)

    module, original = _xacro_module(monkeypatch, action)
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(lambda i: importer.import_source_asset(source, tmp_path / f"prepared-{i}", fix_base=True), [1, 2]))
    assert max(overlap) == 1 and module.open is original
    assert all(receipt["execution_authorized"] is False for receipt in receipts)


def test_invalid_source_bundle_cannot_start_isaac_even_with_explicit_flag(tmp_path, monkeypatch):
    import sys

    calls = []
    monkeypatch.setitem(sys.modules, "isaaclab.app", SimpleNamespace(AppLauncher=lambda **kwargs: calls.append("start")))
    with pytest.raises(SystemExit) as failure:
        importer.main(["--source-bundle", str(tmp_path / "missing"), "--output", str(tmp_path / "out"),
                       "--fixed-base", "--initialize-isaac"])
    assert failure.value.code == 2 and calls == []


@pytest.mark.parametrize("binary", [False, True])
def test_gltf_input_closure_binds_external_buffers_and_embedded_images(tmp_path, monkeypatch, binary):
    import struct

    model = {"asset": {"version": "2.0"}, "buffers": [{"uri": "buffer.bin"}],
             "images": [{"uri": "data:image/png;base64,AA=="}]}
    data = json.dumps(model).encode()
    data += b" " * (-len(data) % 4)
    if binary:
        data = struct.pack("<4sIII4s", b"glTF", 2, 20 + len(data), len(data), b"JSON") + data
    filename = "mesh.glb" if binary else "mesh.gltf"
    source = _source(tmp_path, monkeypatch, content=f'<robot><mesh filename="{filename}"/></robot>'.encode(),
                     extra={filename: data, "buffer.bin": b"development fixture buffer"})
    receipt = assets.verify_source_bundle(source)
    importer._precheck_input_closure(source / "source/robot.urdf",
        allowed={(source / "source" / row["path"]).resolve() for row in receipt["files"]},
        packages={"fixture": source / "source"})


def test_gltf_external_url_is_refused_before_converter(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch, content=b'<robot><mesh filename="mesh.gltf"/></robot>',
                     extra={"mesh.gltf": b'{"buffers":[{"uri":"file:///private/forbidden.bin"}]}'})
    calls = []
    monkeypatch.setattr(importer, "_converter", lambda *args, **kwargs: calls.append("converter"))
    with pytest.raises(assets.AssetPreparationError, match="external_reference_refused"):
        importer.import_source_asset(source, tmp_path / "prepared", fix_base=True)
    assert calls == []
