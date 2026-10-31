"""Development-only importer contract: fake official SDK, real USD structure/receipts.

No Isaac app, GPU, source network request or native episode is executed here.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import deployment_embodiment_asset_import as importer
from blueprint_pipeline import deployment_embodiment_assets as assets
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest


def _source(tmp_path, monkeypatch, format_name="urdf", *, content=None, extra=None, arguments=None, entry=None):
    entry = entry or "robot." + ("xml" if format_name == "mjcf" else format_name)
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


# --- Isaac Sim 6.0.1 importer layout -------------------------------------------------------
# A faithful development fake of what batch06 ran (nvcr.io/nvidia/isaac-sim:6.0.1, Isaac Lab
# ffff603e = v3.0.0-beta2.patch1, urdf-usd-converter 0.1.3 pinned by deps/pip_urdf_usd.toml).
# It reproduces the output structure, not the simulator: geometry is a placeholder cube.
from blueprint_pipeline.team_embodiment import usd_prim_name as _usdex_prim_name  # noqa: E402


def _urdf_usd_converter(urdf, asset_dir, output_dir):
    """urdf_usd_converter.Converter(layer_structure=False).convert (v0.1.3 convert.py:39-164).

    A layered asset (interface -> payload Contents, materials referenced from a
    library whose material prims are instanceable, material.py:168) is flattened
    by _flatten.export_flattened into ``output_dir/<URDF robot name>.usdc``. Link and
    joint prims use usdex names and keep a different source name as displayName
    (link.py:45-48, 339-340). A massless, geometry-free ("ghost") root link or ghost
    behind a fixed joint gets no rigid body (link.py:68-80); joints from a ghost root
    use the asset's default prim as body0, and a non-ghost root gets ``root_joint``
    from the default prim (link.py:284-312).
    """
    from defusedxml import ElementTree as SafeXML
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    root = SafeXML.fromstring(urdf.read_bytes())
    robot = _usdex_prim_name(root.get("name"))
    links = {link.get("name"): link for link in root.findall("link")}
    joints = root.findall("joint")
    incoming = {joint.find("child").get("link"): joint for joint in joints}
    children = {}
    for joint in joints:
        children.setdefault(joint.find("parent").get("link"), []).append(joint)
    root_link = next(name for name in links if name not in incoming)

    def ghost(name):
        link = links[name]
        mass = link.find("inertial/mass")
        return ((mass is None or float(mass.get("value", "0")) == 0)
                and link.find("visual") is None and link.find("collision") is None)

    def rigid(name):
        return not (ghost(name) and (name == root_link or incoming[name].get("type") == "fixed"))

    colors = {material.get("name"): material for material in root.findall("material")}
    payload = asset_dir / "Payload"
    payload.mkdir(parents=True)
    library = Usd.Stage.CreateNew(str(payload / "Materials.usdc"))
    library.SetDefaultPrim(library.DefinePrim("/Materials", "Scope"))
    content = Usd.Stage.CreateNew(str(payload / "Contents.usda"))
    content.SetDefaultPrim(content.DefinePrim("/" + robot, "Xform"))
    materials = {}

    def material(node):
        name = node.get("name")
        if name not in materials:
            declared = node if node.find("color") is not None or node.find("texture") is not None else colors[name]
            safe = _usdex_prim_name(name)
            shader = UsdShade.Shader.Define(library, f"/Materials/{safe}/PreviewSurface")
            shader.CreateIdAttr("UsdPreviewSurface")
            rgba = [float(v) for v in (declared.find("color").get("rgba") if declared.find("color") is not None
                                       else "1 1 1 1").split()]
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgba[:3]))
            if declared.find("texture") is not None:
                texture = urdf.parent / declared.find("texture").get("filename")
                (payload / "Textures").mkdir(exist_ok=True)
                shutil.copyfile(texture, payload / "Textures" / texture.name)
                shader.CreateInput("diffuseTexture", Sdf.ValueTypeNames.Asset).Set(
                    Sdf.AssetPath("./Textures/" + texture.name))
            UsdShade.Material.Define(library, f"/Materials/{safe}").CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface")
            library.GetPrimAtPath(f"/Materials/{safe}").SetInstanceable(True)
            local = content.DefinePrim(f"/{robot}/Materials/{safe}", "Material")
            local.GetReferences().AddReference("./Materials.usdc", f"/Materials/{safe}")
            materials[name] = local
        return materials[name]

    def convert(parent, name, articulated):
        prim = content.DefinePrim(parent.AppendChild(_usdex_prim_name(name)), "Xform")
        if _usdex_prim_name(name) != name:
            prim.SetDisplayName(name)
        if rigid(name):
            UsdPhysics.RigidBodyAPI.Apply(prim)
            if not (name == root_link and ghost(name)) and not articulated:
                UsdPhysics.ArticulationRootAPI.Apply(prim)
                articulated = True
            mass = links[name].find("inertial/mass")
            if mass is not None:
                UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass.get("value")))
        for kind in ("visual", "collision"):
            for index, node in enumerate(links[name].findall(kind)):
                mesh = node.find("geometry/mesh")
                if mesh is not None and Path(mesh.get("filename")).suffix.lower() not in {".stl", ".obj", ".dae"}:
                    continue  # mesh.py:57-67: "Unsupported mesh format" (TRLC's GLB visuals)
                geometry = UsdGeom.Cube.Define(content, prim.GetPath().AppendChild(f"{kind}_{index}"))
                if kind == "collision":
                    UsdPhysics.CollisionAPI.Apply(geometry.GetPrim())
                elif node.find("material") is not None:
                    UsdShade.MaterialBindingAPI.Apply(geometry.GetPrim()).Bind(
                        UsdShade.Material(material(node.find("material"))))
        for joint in children.get(name, []):
            convert(prim.GetPath(), joint.find("child").get("link"), articulated)
        return prim

    default = content.GetDefaultPrim()
    convert(default.GetPath().AppendChild("Geometry"), root_link, False)
    content.DefinePrim(f"/{robot}/Geometry", "Scope")
    scope = content.DefinePrim(f"/{robot}/Physics", "Scope").GetPath()

    def link_path(name):
        path, chain = None, [name]
        while chain[-1] in incoming:
            chain.append(incoming[chain[-1]].find("parent").get("link"))
        path = default.GetPath().AppendChild("Geometry")
        for link in reversed(chain):
            path = path.AppendChild(_usdex_prim_name(link))
        return path

    kinds = {"fixed": UsdPhysics.FixedJoint, "revolute": UsdPhysics.RevoluteJoint,
             "continuous": UsdPhysics.RevoluteJoint, "prismatic": UsdPhysics.PrismaticJoint}
    authored = [] if ghost(root_link) else [("root_joint", "fixed", None, root_link)]
    authored += [(joint.get("name"), joint.get("type"), joint.find("parent").get("link"), joint.find("child").get("link"))
                 for joint in joints]
    for name, kind, parent, child in authored:
        if not rigid(child):
            continue
        prim = kinds[kind].Define(content, scope.AppendChild(_usdex_prim_name(name)))
        if _usdex_prim_name(name) != name:
            prim.GetPrim().SetDisplayName(name)
        body0 = default.GetPath() if parent is None or (parent == root_link and ghost(root_link)) else link_path(parent)
        prim.CreateBody0Rel().SetTargets([body0])
        prim.CreateBody1Rel().SetTargets([link_path(child)])
    library.GetRootLayer().Save()
    content.GetRootLayer().Save()
    interface = Usd.Stage.CreateNew(str(asset_dir / (root.get("name") + ".usda")))
    interface.SetDefaultPrim(interface.DefinePrim("/" + robot, "Xform"))
    interface.GetDefaultPrim().GetPayloads().AddPayload("./Payload/Contents.usda")
    interface.GetRootLayer().Save()
    output_dir.mkdir(parents=True)
    flattened = output_dir / (root.get("name") + ".usdc")
    interface.Flatten().Export(str(flattened))
    # export_flattened rewrites scratch-anchored material inputs to ./Textures/... and
    # copies the textures beside the flattened file (_flatten.py:20-40).
    stage = Usd.Stage.Open(str(flattened))
    for prim in stage.Traverse():
        for attribute in prim.GetAttributes():
            value = attribute.Get() if attribute.GetTypeName() == Sdf.ValueTypeNames.Asset else None
            if value is not None and Path(value.path).is_relative_to(asset_dir):
                attribute.Set(Sdf.AssetPath("./Textures/" + Path(value.path).name))
    stage.Save()
    if (payload / "Textures").is_dir():
        shutil.copytree(payload / "Textures", output_dir / "Textures")
    return flattened


def _isaac_apply_fix_base(stage):
    """isaacsim.asset.importer.utils asset_utils.apply_fix_base (v6.0.1 asset_utils.py:63-303)."""
    from pxr import UsdPhysics

    def body(joint, index):
        relation = UsdPhysics.Joint(joint).GetBody0Rel() if index == 0 else UsdPhysics.Joint(joint).GetBody1Rel()
        targets = relation.GetTargets() if relation else []
        return str(targets[0]) if targets else None

    bodies = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    paths = {str(prim.GetPath()) for prim in bodies}
    joints = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]
    moved = {body(joint, 1) for joint in joints if not joint.IsA(UsdPhysics.FixedJoint)
             and body(joint, 0) in paths and body(joint, 1) in paths}
    root = next((prim for prim in bodies if str(prim.GetPath()) not in moved), None)
    if root is None:
        return  # "Cannot apply fix_base - no rigid body link found."
    fixed = False
    for joint in joints:
        ends = (body(joint, 0), body(joint, 1))
        if joint.IsA(UsdPhysics.FixedJoint) and str(root.GetPath()) in ends:
            other = ends[0] if ends[1] == str(root.GetPath()) else ends[1]
            target = stage.GetPrimAtPath(other) if other else None
            fixed |= other is None or not target or not target.HasAPI(UsdPhysics.RigidBodyAPI)
    if not fixed:
        UsdPhysics.FixedJoint.Define(stage, stage.GetDefaultPrim().GetPath().AppendChild("fix_base_joint")
                                     ).CreateBody1Rel().SetTargets([root.GetPath()])
    for prim in [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI) and p.HasAPI(UsdPhysics.RigidBodyAPI)]:
        if not prim.GetParent().HasAPI(UsdPhysics.ArticulationRootAPI):
            UsdPhysics.ArticulationRootAPI.Apply(prim.GetParent())
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)


def _transformer_package(stage, output_dir, robot_name):
    """The run_asset_transformer=True package (isaacsim_structure.json, interface.py:121-300).

    ``<robot_name>.usda`` references ``./payloads/base.usda`` and selects a Physics
    variant whose payload is ``./payloads/Physics/physx.usda``; that layer sublayers
    ``./physics.usda`` next to itself. The base layer's texture sits in
    ``payloads/Textures``. Every relative value resolves against its own layer.
    """
    from pxr import Sdf

    for prim in stage.Traverse():
        if prim.IsInstance():
            prim.SetInstanceable(False)  # geometries.py _make_references_non_instanceable
    payloads = output_dir / "payloads"
    (payloads / "Physics").mkdir(parents=True)
    (payloads / "Textures").mkdir()
    (payloads / "Textures" / "grey.png").write_bytes(b"development texture bytes")
    base = stage.Flatten()
    name = stage.GetDefaultPrim().GetName()
    shader = next(path for path in _spec_paths(base)
                  if path.name == "PreviewSurface" and path.HasPrefix(Sdf.Path("/" + name)))
    texture = Sdf.AttributeSpec(base.GetPrimAtPath(shader), "inputs:diffuseTexture", Sdf.ValueTypeNames.Asset)
    texture.default = Sdf.AssetPath("./Textures/grey.png")
    base.Export(str(payloads / "base.usda"))
    physics = Sdf.Layer.CreateNew(str(payloads / "Physics" / "physics.usda"))
    physics.defaultPrim = name
    over = Sdf.CreatePrimInLayer(physics, "/" + name)
    over.specifier = Sdf.SpecifierOver
    Sdf.AttributeSpec(over, "physics:fixtureNeutralPhysics", Sdf.ValueTypeNames.Bool).default = True
    physics.Save()
    physx = Sdf.Layer.CreateNew(str(payloads / "Physics" / "physx.usda"))
    physx.subLayerPaths = ["./physics.usda"]
    physx.defaultPrim = name
    over = Sdf.CreatePrimInLayer(physx, "/" + name)
    over.specifier = Sdf.SpecifierOver
    Sdf.AttributeSpec(over, "physxArticulation:fixtureSolverIterations", Sdf.ValueTypeNames.Int).default = 32
    physx.Save()
    interface = Sdf.Layer.CreateNew(str(output_dir / f"{robot_name}.usda"))
    prim = Sdf.CreatePrimInLayer(interface, "/" + name)
    prim.specifier, prim.typeName = Sdf.SpecifierDef, "Xform"
    interface.defaultPrim = name
    prim.referenceList.Prepend(Sdf.Reference("./payloads/base.usda"))
    variants = Sdf.VariantSetSpec(prim, "Physics")
    Sdf.VariantSpec(variants, "None")
    for variant in ("physics", "physx"):
        Sdf.VariantSpec(variants, variant).primSpec.payloadList.Prepend(
            Sdf.Payload(f"./payloads/Physics/{variant}.usda"))
    prim.variantSetNameList.Append("Physics")
    prim.variantSelections["Physics"] = "physx"
    interface.Save()


def _spec_paths(layer):
    paths = []
    layer.Traverse("/", paths.append)
    return paths


def _isaac6_sdk(monkeypatch, scratch, *, package=False):
    """Isaac Lab UrdfConverter -> Isaac Sim 6.0.1 URDFImporter.import_urdf, end to end.

    Output is ``<usd_dir>/<robot_name>/<robot_name>.usda`` where robot_name is the
    input file's stem (urdf_converter.py:64-67, importer_utils.parse_robot_name,
    converter.py:220-223). With run_asset_transformer=False (our setting) that file
    is ``stage.Export`` of the converter stage (converter.py:239-241), i.e. USD
    flattening: instanceable materials become ``/Flattened_Prototype_N`` plus internal
    references, and relative texture paths are anchored into the importer's scratch
    directory, which is deleted afterwards (converter.py:245-247).
    """
    from pxr import Usd

    calls = []
    scratch.mkdir(exist_ok=True)

    class Config:
        def __init__(self, **kwargs):
            self.values = kwargs

    class Converter:
        def __init__(self, config):
            values = config.values
            calls.append(values)
            source = Path(values["asset_path"])
            robot_name = source.stem.lstrip(".")
            usd_dir = Path(values["usd_dir"])
            usd_dir.mkdir(parents=True, exist_ok=True)
            (usd_dir / ".asset_hash").write_text("development md5 of config and asset")
            work = Path(tempfile.mkdtemp(prefix=f"urdf_import_{robot_name}_", dir=scratch))
            try:
                usdex = _urdf_usd_converter(source, work / "asset", work / f"usdex_{robot_name}")
                stage = Usd.Stage.Open(str(usdex))
                if values["fix_base"]:
                    _isaac_apply_fix_base(stage)
                output = usd_dir / robot_name
                output.mkdir()
                if package:
                    _transformer_package(stage, output, robot_name)
                else:
                    stage.Export(str(output / f"{robot_name}.usda"))
            finally:
                shutil.rmtree(work)
            (usd_dir / "config.yaml").write_text(f"asset_path: {source}\nusd_dir: {usd_dir}\n")
            self.usd_path = str(output / f"{robot_name}.usda")

    sdk = SimpleNamespace(UrdfConverterCfg=Config, UrdfConverter=Converter)
    real_import = importer.importlib.import_module
    monkeypatch.setattr(importer.importlib, "import_module",
                        lambda name, *args: sdk if name == "isaaclab.sim.converters" else real_import(name, *args))
    return calls


_ARM = b'''<robot name="development_arm">
  <material name="grey"><color rgba="0.5 0.5 0.5 1"/></material>
  <link name="world"/><link name="base_link"/>
  <link name="base_link_inertia"><inertial><mass value="4"/></inertial>
    <visual><geometry><box size="0.1 0.1 0.1"/></geometry><material name="grey"/></visual></link>
  <link name="shoulder_link"><inertial><mass value="2"/></inertial>
    <visual><geometry><box size="0.1 0.1 0.1"/></geometry>
      <material name="blue"><color rgba="0 0 1 1"/></material></visual></link>
  <link name="tool-flange"><inertial><mass value="1"/></inertial>
    <collision><geometry><box size="0.05 0.05 0.05"/></geometry></collision></link>
  <link name="tcp"/>
  <link name="finger-a"><inertial><mass value="0.1"/></inertial></link>
  <link name="finger-b"><inertial><mass value="0.1"/></inertial></link>
  <joint name="world_joint" type="fixed"><parent link="world"/><child link="base_link"/></joint>
  <joint name="base_link-base_link_inertia" type="fixed"><parent link="base_link"/>
    <child link="base_link_inertia"/></joint>
  <joint name="shoulder_pan_joint" type="revolute"><parent link="base_link_inertia"/><child link="shoulder_link"/>
    <axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="1"/></joint>
  <joint name="wrist_joint" type="revolute"><parent link="shoulder_link"/><child link="tool-flange"/>
    <origin xyz="0 0 0.4"/><axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="10" velocity="1"/></joint>
  <joint name="tcp_mount" type="fixed"><parent link="tool-flange"/><child link="tcp"/><origin xyz="0 0 0.1"/></joint>
  <joint name="finger" type="prismatic"><parent link="tool-flange"/><child link="finger-a"/>
    <axis xyz="0 1 0"/><limit lower="0" upper="0.04" velocity="0.1" effort="10"/></joint>
  <joint name="slave" type="prismatic"><parent link="tool-flange"/><child link="finger-b"/>
    <axis xyz="0 1 0"/><limit lower="-0.04" upper="0" velocity="0.1" effort="10"/>
    <mimic joint="finger" multiplier="-1" offset="0"/></joint>
</robot>'''


def _two_arm_station(identity="fixture_dual_option", base="world"):
    from blueprint_pipeline import deployment_embodiment_station as station

    return {"schema_version": station.PARAMETERS_SCHEMA, "embodiment_id": identity,
            "composition_mode": "two_instances", "station_root_link": "blueprint_station", "arms": [
                {"name": side, "prefix": side + "_", "source_base_link": base, "source_mount_joint": None,
                 "station_from_base": {"translation_m": [0.0, y, 0.0], "quaternion_wxyz": [1, 0, 0, 0]}}
                for side, y in (("left", .35), ("right", -.35))]}


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


def test_isaac_builtin_mdl_is_recorded_as_runtime_dependency_and_reverified(tmp_path, monkeypatch):
    """Isaac binds URDF colours to Kit's OmniPBR.mdl; batch05 refused all such imports."""
    source = _source(tmp_path, monkeypatch)
    _sdk(monkeypatch, asset="OmniPBR.mdl")
    prepared = tmp_path / "prepared"
    result = importer.import_source_asset(source, prepared, fix_base=True)
    assert result["runtime_builtin_assets"] == ["OmniPBR.mdl"]
    assert [row["path"] for row in result["dependency_closure"]["files"]] == ["robot.usda"]
    assert "@OmniPBR.mdl@" in (prepared / "robot.usda").read_text()
    assert importer.verify_prepared_asset_directory(prepared) == result
    # A recorded runtime dependency cannot be dropped by re-digesting the receipt.
    receipt = json.loads((prepared / importer.RECEIPT).read_text())
    receipt["runtime_builtin_assets"] = []
    receipt["import_digest"] = cross_runtime_canonical_digest(receipt, digest_field="import_digest")
    (prepared / importer.RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n")
    with pytest.raises(assets.AssetPreparationError, match="dependency_changed"):
        importer.verify_prepared_asset_directory(prepared)


@pytest.mark.parametrize("asset", ["OmniUnlisted.mdl", "materials/OmniPBR.mdl"])
def test_unlisted_or_pathed_mdl_remains_unbound_and_names_the_file(tmp_path, monkeypatch, asset):
    source = _source(tmp_path, monkeypatch)
    _sdk(monkeypatch, asset=asset)
    expected = "unbound_dependency:" + Path(asset).name.replace(".", r"\.")
    with pytest.raises(assets.AssetPreparationError, match=expected):
        importer.import_source_asset(source, tmp_path / "prepared", fix_base=True)


def test_builtin_name_with_a_sibling_file_is_checked_as_that_file_not_the_runtime(tmp_path):
    from pxr import Sdf, Usd

    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "OmniPBR.mdl").write_bytes(b"mdl 1.4;\n")
    stage = Usd.Stage.CreateNew(str(staged / "robot.usda"))
    stage.DefinePrim("/robot").CreateAttribute("material", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("OmniPBR.mdl"))
    stage.GetRootLayer().Save()
    builtins = set()
    # The local file shadows Kit's module, so the ordinary file closure applies
    # (and an MDL file closure remains unsupported).
    with pytest.raises(assets.AssetPreparationError, match="input_format_dependency_unsupported"):
        importer._dependencies(staged / "robot.usda", staged=staged, builtins=builtins)
    assert builtins == set()


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


def test_urdf_mujoco_extension_does_not_change_urdf_mesh_resolution(tmp_path):
    entry = tmp_path / "g1.urdf"
    mesh = tmp_path / "meshes" / "pelvis.STL"
    mesh.parent.mkdir()
    mesh.write_bytes(b"fixture mesh")
    entry.write_text('<robot name="g1"><mujoco><compiler meshdir="meshes"/></mujoco>'
                     '<link name="pelvis"><visual><geometry>'
                     '<mesh filename="meshes/pelvis.STL"/>'
                     '</geometry></visual></link></robot>')
    allowed = {entry.resolve(), mesh.resolve()}
    assert importer._precheck_input_closure(entry, allowed=allowed, packages={}) == allowed
    with pytest.raises(assets.AssetPreparationError, match="input_reference_unbound"):
        importer._precheck_input_closure(entry, allowed={entry.resolve()}, packages={})


def test_mjcf_meshdir_still_resolves_assets_relative_to_compiler_directory(tmp_path):
    entry = tmp_path / "robot.xml"
    mesh = tmp_path / "meshes" / "pelvis.stl"
    mesh.parent.mkdir()
    mesh.write_bytes(b"fixture mesh")
    entry.write_text('<mujoco><compiler meshdir="meshes"/>'
                     '<asset><mesh name="pelvis" file="pelvis.stl"/></asset></mujoco>')
    allowed = {entry.resolve(), mesh.resolve()}
    assert importer._precheck_input_closure(entry, allowed=allowed, packages={}) == allowed


def test_isaac_urdf_closure_does_not_treat_gazebo_plugin_as_mesh(tmp_path):
    entry = tmp_path / "xarm.urdf"
    mesh = tmp_path / "gripper.stl"
    mesh.write_bytes(b"fixture mesh")
    entry.write_text('<robot name="xarm"><link name="gripper"><visual><geometry>'
                     '<mesh filename="gripper.stl"/></geometry></visual></link>'
                     '<gazebo><plugin filename="libgazebo_mimic_joint_plugin.so" '
                     'name="mimic"><joint>drive_joint</joint></plugin></gazebo></robot>')
    allowed = {entry.resolve(), mesh.resolve()}
    assert importer._precheck_input_closure(entry, allowed=allowed, packages={}) == allowed
    with pytest.raises(assets.AssetPreparationError, match="input_reference_unbound"):
        importer._precheck_input_closure(entry, allowed={entry.resolve()}, packages={})


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


@pytest.mark.parametrize("active", [False, True])
def test_xacro_only_active_package_references_require_aliases(tmp_path, monkeypatch, active):
    reference = '<xacro:include filename="$(find obsolete_visualization)/old.xacro"/>'
    content = ('<robot xmlns:xacro="http://wiki.ros.org/xacro">'
               + (reference if active else '<!-- ' + reference + ' -->') + '</robot>').encode()
    source = _source(tmp_path, monkeypatch, "xacro", content=content)
    calls = _sdk(monkeypatch)
    _xacro_module(monkeypatch, lambda filename, args: None)
    if active:
        with pytest.raises(assets.AssetPreparationError, match="xacro_package_alias_missing"):
            importer.import_source_asset(source, tmp_path / "prepared", fix_base=True)
        assert not calls
    else:
        importer.import_source_asset(source, tmp_path / "prepared", fix_base=True)
        assert len(calls) == 1
        assert any(path.read_bytes() == content for path in (source / "source").glob("*.xacro"))


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


def _moved(tmp_path, prepared):
    moved = tmp_path / "moved" / "robot-preparation"
    moved.parent.mkdir()
    prepared.chmod(0o755)  # The test owns this immutable publication directory.
    prepared.rename(moved)
    return moved


@pytest.mark.parametrize("entry,format_name,station,robot_name", [
    ("robot.urdf", "urdf", True, "robot"),  # universal_robots_ur5e_bimanual: station/robot.urdf
    ("axol_standalone.urdf", "urdf", False, "axol_standalone"),
    ("yam_station_linear_4310_d405.urdf", "urdf", False, "yam_station_linear_4310_d405"),
    ("assembly.urdf.xacro", "xacro", False, "assembly.urdf"),  # expanded to assembly.urdf.urdf
    # GoFa/KUKA/Yaskawa simulation models import staged simulation-model.urdf (gofa_model.py MODELED_URDF).
    ("simulation-model.urdf", "urdf", False, "simulation-model"),
])
def test_isaac6_flattened_import_with_instanced_materials_is_walked_retained_and_relocatable(
        tmp_path, monkeypatch, entry, format_name, station, robot_name):
    """batch06: 13 of 14 bodies stopped at asset_import_unbound_dependency:<robot_name>."""
    from pxr import Sdf, Usd, UsdShade

    source = _source(tmp_path, monkeypatch, format_name, entry=entry,
                     content=b'<robot name="development_arm"/>' if format_name == "xacro" else _ARM)
    if format_name == "xacro":
        import xml.dom.minidom

        module, _ = _xacro_module(monkeypatch, lambda filename, args: None)
        module.make_document = lambda: xml.dom.minidom.parseString(_ARM)
    if station:
        from blueprint_pipeline import deployment_embodiment_station as station_module
        monkeypatch.setattr(station_module, "recipe_for", importer.recipe_for)
    calls = _isaac6_sdk(monkeypatch, tmp_path / "isaac-scratch")
    prepared = tmp_path / "prepared"
    receipt = importer.import_source_asset(source, prepared, fix_base=True,
                                            station_parameters=_two_arm_station() if station else None)
    assert calls[0]["run_asset_transformer"] is False
    # The importer's own package: <usd_dir>/<robot_name>/<robot_name>.usda, retained as staged content.
    converted = prepared / "converted" / robot_name / f"{robot_name}.usda"
    assert converted.is_file() and (prepared / "converted/config.yaml").is_file()
    # What the old walker read: an internal arc's empty asset path, which it joined
    # to the layer's directory and refused under that directory's name.
    raw = Sdf.Layer.OpenAsAnonymous(str(converted))
    assert "" in raw.GetExternalReferences() and "Flattened_Prototype_1" in converted.read_text()
    assert (converted.parent / Path("")).is_dir() and (converted.parent / Path("")).name == robot_name
    published = (prepared / "robot.usda").read_text()
    assert "Flattened_Prototype_1" in published and str(tmp_path) not in published
    assert [row["path"] for row in receipt["dependency_closure"]["files"]] == ["robot.usda"]
    assert receipt["runtime_builtin_assets"] == [] and receipt["single_file_self_contained"] is True
    sides = ("left_", "right_") if station else ("",)
    assert receipt["structure"]["world_anchor_links"] == [side + "base_link_inertia" for side in sides]
    prefix = "{ENV_REGEX_NS}/Robot/Geometry/" + ("blueprint_station/" if station else "")
    assert receipt["structure"]["body_source_names"] == {
        f"{prefix}{side}world/{side}base_link/{side}base_link_inertia/{side}shoulder_link/"
        f"{_usdex_prim_name(side + 'tool-flange')}{suffix}": side + name for side in sides
        for name, suffix in (("tool-flange", ""), ("finger-a", "/" + _usdex_prim_name(side + "finger-a")),
                             ("finger-b", "/" + _usdex_prim_name(side + "finger-b")))}
    moved = _moved(tmp_path, prepared)
    assert importer.verify_prepared_asset_directory(moved) == receipt
    stage = Usd.Stage.Open(str(moved / "robot.usda"))
    bound = [UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0] for prim in stage.Traverse()
             if prim.HasAPI(UsdShade.MaterialBindingAPI)]
    assert bound and all(material.GetPrim().IsInstance() for material in bound)


def test_isaac6_transformer_package_resolves_each_layer_against_itself_and_retains_textures(tmp_path, monkeypatch):
    """run_asset_transformer=True layout: interface -> payloads/base.usda, Physics variant -> physx.usda -> physics.usda."""
    from pxr import Sdf, Usd

    source = _source(tmp_path, monkeypatch, content=_ARM, entry="axol_standalone.urdf")
    _isaac6_sdk(monkeypatch, tmp_path / "isaac-scratch", package=True)
    prepared = tmp_path / "prepared"
    receipt = importer.import_source_asset(source, prepared, fix_base=True)
    package = prepared / "converted/axol_standalone"
    assert Sdf.Layer.FindOrOpen(str(package / "payloads/Physics/physx.usda")).subLayerPaths == ["./physics.usda"]
    assert {row["path"] for row in receipt["dependency_closure"]["files"]} == {
        "robot.usda", "converted/axol_standalone/payloads/Textures/grey.png"}
    moved = _moved(tmp_path, prepared)
    stage = Usd.Stage.Open(str(moved / "robot.usda"))
    root = stage.GetDefaultPrim()
    # The flattened model composed the selected physx variant and the sublayer it carries.
    assert root.GetAttribute("physxArticulation:fixtureSolverIterations").Get() == 32
    assert root.GetAttribute("physics:fixtureNeutralPhysics").Get() is True
    texture = next(attr for prim in stage.Traverse() for attr in prim.GetAttributes()
                   if attr.GetName() == "inputs:diffuseTexture").Get()
    assert texture.path == "converted/axol_standalone/payloads/Textures/grey.png"
    assert Path(Sdf.ComputeAssetPathRelativeToLayer(stage.GetRootLayer(), texture.path)).read_bytes() == (
        b"development texture bytes")
    assert importer.verify_prepared_asset_directory(moved) == receipt


def test_isaac6_texture_left_in_deleted_importer_scratch_is_refused_with_a_named_diagnostic(tmp_path, monkeypatch):
    """Flattening anchors ./Textures/x into Isaac's scratch directory, which import_urdf then deletes."""
    content = _ARM.replace(b'<color rgba="0.5 0.5 0.5 1"/>',
                           b'<color rgba="0.5 0.5 0.5 1"/><texture filename="textures/grey.png"/>')
    source = _source(tmp_path, monkeypatch, content=content, extra={"textures/grey.png": b"png fixture"})
    _isaac6_sdk(monkeypatch, tmp_path / "isaac-scratch")
    with pytest.raises(assets.AssetPreparationError) as refused:
        importer.import_source_asset(source, tmp_path / "prepared", fix_base=True)
    assert str(refused.value) == (
        "asset_import_unbound_dependency:grey.png;kind=asset_attribute;layer=robot.usda;reason=missing")
    assert not (tmp_path / "prepared").exists()


def test_authored_dependencies_classify_every_arc_and_skip_only_internal_arcs(tmp_path):
    from pxr import Sdf

    layer = Sdf.Layer.CreateNew(str(tmp_path / "robot.usda"))
    layer.subLayerPaths = ["./physics.usda"]
    layer.customLayerData = {"source": Sdf.AssetPath("./notes.txt")}
    prim = Sdf.CreatePrimInLayer(layer, "/robot")
    prim.specifier = Sdf.SpecifierDef
    prim.referenceList.Prepend(Sdf.Reference("./payloads/base.usda"))
    prim.referenceList.Prepend(Sdf.Reference("", "/Flattened_Prototype_1"))  # internal: no file
    prim.payloadList.deletedItems = [Sdf.Payload("./retired.usda")]
    variant = Sdf.VariantSpec(Sdf.VariantSetSpec(prim, "Physics"), "physx")
    variant.primSpec.payloadList.Prepend(Sdf.Payload("./payloads/Physics/physx.usda"))
    prim.assetInfo = {"identifier": Sdf.AssetPath("/host/robot.usd")}
    texture = Sdf.AttributeSpec(prim, "texture", Sdf.ValueTypeNames.Asset)
    texture.default = Sdf.AssetPath("./a.png")
    texture.customData = {"origin": Sdf.AssetPath("./origin.txt")}
    layer.SetTimeSample(texture.path, 1.0, Sdf.AssetPath("./b.png"))
    frames = Sdf.AttributeSpec(prim, "frames", Sdf.ValueTypeNames.AssetArray)
    frames.default = Sdf.AssetPathArray([Sdf.AssetPath("./c.png"), Sdf.AssetPath("")])
    assert sorted(importer._authored_dependencies(layer)) == sorted([
        ("sublayer", "./physics.usda"), ("metadata", "./notes.txt"), ("reference", "./payloads/base.usda"),
        ("payload", "./retired.usda"), ("payload", "./payloads/Physics/physx.usda"),
        ("metadata", "/host/robot.usd"), ("asset_attribute", "./a.png"), ("asset_attribute", "./b.png"),
        ("metadata", "./origin.txt"), ("asset_attribute", "./c.png")])


def test_published_model_metadata_host_path_is_refused_with_its_kind(tmp_path, monkeypatch):
    from pxr import Sdf, Usd

    source = _source(tmp_path, monkeypatch)
    _sdk(monkeypatch)
    prepared = tmp_path / "prepared"
    result = importer.import_source_asset(source, prepared, fix_base=True)
    prepared.chmod(0o755)
    stage = Usd.Stage.Open(str(prepared / "robot.usda"))
    stage.GetPrimAtPath("/robot").SetAssetInfoByKey("identifier", Sdf.AssetPath(str(tmp_path / "host.usd")))
    stage.GetRootLayer().Save()
    model_bytes = (prepared / "robot.usda").read_bytes()
    updated = {"path": "robot.usda", "size_bytes": len(model_bytes),
               "sha256": "sha256:" + hashlib.sha256(model_bytes).hexdigest()}
    result["files"] = [updated if row["path"] == "robot.usda" else row for row in result["files"]]
    result["model"] = dict(updated)
    result["import_digest"] = cross_runtime_canonical_digest(result, digest_field="import_digest")
    (prepared / importer.RECEIPT).write_text(json.dumps(result))
    with pytest.raises(assets.AssetPreparationError) as refused:
        importer.verify_prepared_asset_directory(prepared)
    assert str(refused.value) == ("asset_import_dependency_not_relocatable:host.usd;kind=metadata;"
                                  "layer=robot.usda;reason=absolute_path")
