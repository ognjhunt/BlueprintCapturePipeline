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


def _source(tmp_path, monkeypatch, format_name="urdf"):
    entry = "robot." + ("xml" if format_name == "mjcf" else format_name)
    recipe = {"embodiment_id": "fixture_dual_option", "arm_count": 2,
              "status": "source_available", "source": {
                  "repository_url": "https://github.com/fixture/source", "revision": "a" * 40,
                  "entry_path": entry, "format": format_name, "license_id": "MIT", "license_path": "LICENSE",
                  "license_note": "Development fixture only", "include_paths": [entry, "LICENSE"],
                  "import_arguments": {}, "package_roots": {"fixture": "."}, "source_arm_count": 1},
              "composition_required": ["exact_dual_mount_transform", "actual_end_effectors_and_cameras"],
              "missing": ["controller_reset_and_runtime_receipt"]}
    monkeypatch.setattr(assets, "recipe_for", lambda identity: recipe)
    monkeypatch.setattr(importer, "recipe_for", lambda identity: recipe)
    files = {entry: b'<robot name="fixture"><link name="base"/></robot>', "LICENSE": b"MIT fixture notice\n"}
    rows = [{"path": name, "type": "blob", "mode": "100644", "size": len(value),
             "sha": hashlib.sha1(b"blob " + str(len(value)).encode() + b"\0" + value).hexdigest()}
            for name, value in files.items()]

    def get(url, *, maximum):
        return json.dumps({"truncated": False, "tree": rows}).encode() if "/git/trees/" in url else files[url.rsplit("/", 1)[-1]]

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


def test_importer_records_source_texture_dependency_and_rejects_unbound_paths(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch)
    _sdk(monkeypatch, asset=str(source / "source/LICENSE"))
    result = importer.import_source_asset(source, tmp_path / "prepared", fix_base=True)
    dependencies = [row for row in result["dependencies"] if row["location"] == "source_bundle"]
    assert [row["path"] for row in dependencies] == ["LICENSE"]
    assert result["source_bundle_required_for_dependency_resolution"] is True
    # The deliberately fake texture stands in only for an asset-path binding;
    # no renderer or simulator interprets it.
    _sdk(monkeypatch, asset="https://unapproved.example/robot.usd")
    with pytest.raises(assets.AssetPreparationError, match="unbound_dependency"):
        importer.import_source_asset(source, tmp_path / "unbound", fix_base=True)
    assert not (tmp_path / "unbound").exists()


@pytest.mark.parametrize("initialize,failure", [(False, False), (True, False), (True, True)])
def test_cli_only_initializes_isaac_explicitly_and_closes_app(tmp_path, monkeypatch, capsys, initialize, failure):
    import sys

    calls = []
    app = SimpleNamespace(close=lambda: calls.append("close"))

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
    assert calls == (["start", "close"] if initialize else [])
