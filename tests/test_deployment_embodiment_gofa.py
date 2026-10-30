"""Pinned public XML + synthetic STL/SDK fixtures: software tests, not robot proof."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as XML

import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from blueprint_pipeline import deployment_embodiment_assets as assets
from blueprint_pipeline import deployment_embodiment_asset_import as importer
from blueprint_pipeline import deployment_embodiment_gofa_model as gofa
from blueprint_pipeline import deployment_embodiment_runtime_configuration as runtime
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest as digest
from blueprint_pipeline.deployment_embodiment_runtime_compatibility import (
    B0_GENERATOR,
    B0_PROFILES,
    PREDECESSOR_GENERATOR,
)
from tests.test_deployment_embodiment_fr3_composition import fr3_composed  # noqa: F401
from tests.test_deployment_embodiment_tool_composition import composed  # noqa: F401


def _stl():
    # Explicit development-only tetrahedron; no source geometry fidelity claim.
    vertices = [(0, 0, 0), (0.1, 0, 0), (0, 0.2, 0), (0, 0, 0.3)]
    faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
    return (
        b"development-only".ljust(80, b" ")
        + struct.pack("<I", 4)
        + b"".join(
            struct.pack("<12fH", 0, 0, 0, *(v for n in face for v in vertices[n]), 0)
            for face in faces
        )
    )


@pytest.fixture
def source_case(tmp_path):
    def prepare(identity, *, omit=None):
        recipe = assets.recipe_for(identity)
        root = Path(__file__).parent / "fixtures/gofa"
        files = {
            p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()
        }
        for variant in ["5_95", "12_127"]:
            for kind in ["collision", "visual"]:
                for link in ["base_link", *[f"link_{n}" for n in range(1, 7)]]:
                    # Match actual upstream sparse 12/1.27 tree, not invented substitutes.
                    if variant == "12_127" and link in ["link_3", "link_5", "link_6"]:
                        continue
                    files[f"abb_crb15000_support/meshes/crb15000_{variant}/{kind}/{link}.stl"] = (
                        _stl()
                    )
        if omit:
            del files[omit]

        def get(url, *, maximum):
            if "/git/trees/" in url:
                return json.dumps(
                    {
                        "truncated": False,
                        "tree": [
                            {
                                "path": p,
                                "type": "blob",
                                "mode": "100644",
                                "size": len(data),
                                "sha": assets._git_blob_sha(data),
                            }
                            for p, data in files.items()
                        ],
                    }
                ).encode()
            return files[url.split("/" + recipe["source"]["revision"] + "/", 1)[1]]

        folder = tmp_path / identity
        assets.materialize_sources(identity, folder, get=get)
        return folder

    return prepare


@pytest.mark.parametrize("identity", list(gofa.PAYLOAD_KG))
def test_exact_source_limits_tcp_no_tool_and_declared_simulation_dynamics(
    source_case, tmp_path, identity
):
    bundle = source_case(identity)
    before = {p: p.read_bytes() for p in (bundle / "source").rglob("*") if p.is_file()}
    output = tmp_path / "configured"
    result = runtime.prepare_runtime_configuration(bundle, output)
    assert runtime.verify_runtime_configuration(output) == result
    fields = result["manifest_fields"]
    assert [j["name"] for j in fields["joints"]] == [f"joint_{n}" for n in range(1, 7)]
    assert fields["grippers"] == [] and "arms" not in fields
    assert fields["reset_joint_positions"] == {f"joint_{n}": 0.0 for n in range(1, 7)}
    assert [c["name"] for c in fields["cameras"]] == ["overhead", "arm_wrist"]
    original, modeled = (
        XML.parse(output / gofa.SOURCE_MODEL).getroot(),
        XML.parse(output / "robot.urdf").getroot(),
    )
    assert all(original.find(f"link[@name='link_{n}']/inertial") is None for n in range(1, 7))
    assert all(
        float(j.find("limit").get("effort")) == 0
        for j in original.findall("joint")
        if j.get("type") == "revolute"
    )
    assert (
        original.find("joint[@name='flange-tool0']/origin").get("rpy") == "0 1.5707963267948966 0"
    )
    assert len(modeled.findall("link/inertial")) == 7
    assert result["source_dynamics"]["moving_links_without_positive_authored_mass_and_inertia"] == [
        f"link_{n}" for n in range(1, 7)
    ]
    assert result["source_dynamics"]["estimated_inertias_added"] is True
    assert result["source_dynamics"]["simulator_dynamics_validated"] is False
    report = result["simulation_model"]
    assert (
        report["manufacturer_dynamics"] is False
        and result["policy_compatibility_established"] is False
    )
    assert report["modeled_robot_mass_total_kg"] == pytest.approx(gofa.NOMINAL_ROBOT_KG[identity])
    assert report["moving_link_mass_total_kg"] < report["modeled_robot_mass_total_kg"]
    assert all(j["effort_limit_nm"] > 0 for j in report["joints"])
    assert fields["joints"][0]["stiffness"] == report["joints"][0]["stiffness_nm_per_rad"]
    assert all(p.read_bytes() == data for p, data in before.items())
    limit = original.find("joint[@name='joint_1']/limit")
    assert float(limit.get("upper")) == pytest.approx(
        3.141592653589793 if identity.endswith("5_95") else 4.71238898038469
    )
    assert assets.recipe_for("abb_gofa_crb15000")["source"] is None


def test_cross_variant_mesh_and_both_shared_licenses_are_required(source_case, tmp_path):
    path = "abb_crb15000_support/meshes/crb15000_5_95/collision/link_3.stl"
    source = source_case("abb_gofa_crb15000_12_127", omit=path)
    with pytest.raises(assets.AssetPreparationError):
        runtime.prepare_runtime_configuration(source, tmp_path / "incomplete")
    for identity in gofa.PAYLOAD_KG:
        includes = assets.recipe_for(identity)["source"]["include_paths"]
        assert "abb_resources/urdf" in includes and "abb_resources/LICENSE" in includes
        assert "abb_crb15000_support/LICENSE" in includes


def test_mesh_scale_origin_and_uniform_box_inertia(source_case, tmp_path):
    from blueprint_pipeline.deployment_embodiment_station import _origin_matrix

    path = tmp_path / "test.stl"
    path.write_bytes(_stl())
    link = XML.fromstring(
        '<link><collision><origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/><geometry><mesh filename="test.stl" scale="2 3 4"/></geometry></collision></link>'
    )
    dims, center, evidence = gofa._bounds(link, tmp_path)
    assert dims == pytest.approx([0.6, 0.2, 1.2])
    assert center == pytest.approx([0.7, 2.1, 3.6])
    assert evidence[0]["link_from_collision"] == _origin_matrix(link.find("collision"))


def _sdk(monkeypatch):
    from tests.test_deployment_embodiment_runtime_configuration import _fake_sdk_from_urdf

    _fake_sdk_from_urdf(monkeypatch)
    converter = importer._converter

    def convert(*args, **kwargs):
        path, options, identity = converter(*args, **kwargs)
        stage = Usd.Stage.Open(str(path))
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        stage.GetRootLayer().Save()
        return path, options, identity

    monkeypatch.setattr(importer, "_converter", convert)


@pytest.mark.parametrize("identity", list(gofa.PAYLOAD_KG))
def test_actual_saved_usd_estimates_readback_and_binding_with_fake_sdk(
    source_case, tmp_path, monkeypatch, identity
):
    bundle = source_case(identity)
    _sdk(monkeypatch)
    config, prepared = tmp_path / "configured", tmp_path / "imported"
    fields = runtime.prepare_runtime_configuration(bundle, config)
    receipt = importer.import_source_asset(config, prepared, fix_base=True)
    assert receipt["simulation_model"] == fields["simulation_model"]
    assert importer.verify_prepared_asset_directory(prepared) == receipt
    binding = runtime.bind_prepared_configuration(
        config, prepared, "https://example.com/development-only.usda"
    )
    assert (
        len(binding["robot_embodiment"]["joints"]) == 6
        and binding["robot_embodiment"]["grippers"] == []
    )
    assert binding["execution_authorized"] is False
    observations = gofa.usd_values(prepared / "robot.usda", fields["simulation_model"])
    assert len(observations["links"]) == 7 and len(observations["joints"]) == 6
    assert all(j["effort_limit_nm"] > 0 for j in observations["joints"])
    # Reject independently changed authoring even if a caller tries to reuse the old report.
    prepared.chmod(0o755)
    (prepared / "robot.usda").chmod(0o644)
    stage = Usd.Stage.Open(str(prepared / "robot.usda"))
    UsdPhysics.DriveAPI(stage.GetPrimAtPath("/robot/joint_1"), "angular").GetMaxForceAttr().Set(0)
    stage.GetRootLayer().Save()
    with pytest.raises(assets.AssetPreparationError, match="usd_effort_changed"):
        gofa.usd_values(prepared / "robot.usda", fields["simulation_model"])


def _legacy_b0():
    path = Path(__file__).parent / "fixtures/runtime_configuration_b0.py"
    assert "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() == B0_GENERATOR
    spec = importlib.util.spec_from_file_location("blueprint_pipeline._b0_runtime_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_existing_nineteenth_fr3_and_original_dual_keep_exact_b0_receipts(
    fr3_composed,  # noqa: F811
    tmp_path,
    monkeypatch,  # noqa: F811
):  # noqa: F811
    fr3, _, ur, _ = fr3_composed
    assert len(B0_PROFILES) == 19 and not set(gofa.PAYLOAD_KG) & B0_PROFILES
    for source, name in [(fr3, "fr3"), (ur, "dual")]:
        folder = tmp_path / name
        receipt = _legacy_b0().prepare_runtime_configuration(source, folder)
        before = (folder / runtime.RECEIPT).read_bytes()
        assert runtime.verify_runtime_configuration(folder) == receipt
        assert runtime.prepare_runtime_configuration(source, folder) == receipt
        assert (folder / runtime.RECEIPT).read_bytes() == before


@pytest.mark.parametrize("generator", [B0_GENERATOR, PREDECESSOR_GENERATOR, "sha256:" + "a" * 64])
def test_new_variants_cannot_borrow_old_or_unknown_generator(source_case, tmp_path, generator):
    folder = tmp_path / "configured"
    result = runtime.prepare_runtime_configuration(source_case("abb_gofa_crb15000_5_95"), folder)
    result["generator_module_sha256"] = generator
    result["configuration_digest"] = digest(result, digest_field="configuration_digest")
    (folder / runtime.RECEIPT).write_text(json.dumps(result))
    with pytest.raises(assets.AssetPreparationError, match="receipt_changed"):
        runtime.verify_runtime_configuration(folder)


@pytest.mark.parametrize("field", ["mass", "gain", "warning"])
def test_resealed_configuration_cannot_hide_estimate_changes(source_case, tmp_path, field):
    folder = tmp_path / "configured"
    result = deepcopy(
        runtime.prepare_runtime_configuration(source_case("abb_gofa_crb15000_5_95"), folder)
    )
    if field == "mass":
        result["simulation_model"]["links"][0]["mass_kg"] += 1
    elif field == "gain":
        result["manifest_fields"]["joints"][0]["stiffness"] += 1
    else:
        result["source_limitations"] = []
    result["configuration_digest"] = digest(result, digest_field="configuration_digest")
    (folder / runtime.RECEIPT).write_text(json.dumps(result))
    with pytest.raises(assets.AssetPreparationError, match="gofa_simulation_"):
        runtime.verify_runtime_configuration(folder)


@pytest.mark.parametrize("defect", ["mass", "com", "inertia", "axes", "units", "mass_units", "drive_type"])
def test_saved_usd_modeling_semantics_cannot_be_changed(source_case, tmp_path, monkeypatch, defect):
    from pxr import Gf

    bundle = source_case("abb_gofa_crb15000_5_95")
    _sdk(monkeypatch)
    prepared = tmp_path / "prepared"
    receipt = importer.import_source_asset(bundle, prepared, fix_base=True)
    prepared.chmod(0o755)
    model = prepared / "robot.usda"
    model.chmod(0o644)
    stage = Usd.Stage.Open(str(model))
    mass = UsdPhysics.MassAPI(stage.GetPrimAtPath("/robot/link_1"))
    if defect == "mass":
        mass.GetMassAttr().Set(0)
    elif defect == "com":
        mass.GetCenterOfMassAttr().Set(Gf.Vec3f(9, 9, 9))
    elif defect == "inertia":
        mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(1, 1, 1))
    elif defect == "axes":
        mass.GetPrincipalAxesAttr().Set(Gf.Quatf(0, 1, 0, 0))
    elif defect == "units":
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    elif defect == "mass_units":
        UsdPhysics.SetStageKilogramsPerUnit(stage, .001)
    else:
        UsdPhysics.DriveAPI(stage.GetPrimAtPath("/robot/joint_1"), "angular").GetTypeAttr().Set(
            "acceleration"
        )
    stage.GetRootLayer().Save()
    with pytest.raises(assets.AssetPreparationError, match="gofa_simulation_usd_"):
        gofa.usd_values(model, receipt["simulation_model"])


def test_b0_identity_still_reconstructs_every_contract_field(fr3_composed, tmp_path):  # noqa: F811
    folder = tmp_path / "previous-fr3"
    record = _legacy_b0().prepare_runtime_configuration(fr3_composed[0], folder)
    record["manifest_fields"]["joints"][0]["max_velocity"] = 0.5
    record["configuration_digest"] = digest(record, digest_field="configuration_digest")
    (folder / runtime.RECEIPT).write_text(json.dumps(record))
    with pytest.raises(assets.AssetPreparationError, match="predecessor_metadata_changed"):
        runtime.verify_runtime_configuration(folder)


def test_model_report_cannot_change_exact_variant(source_case, tmp_path):
    folder = tmp_path / "configured"
    record = runtime.prepare_runtime_configuration(source_case("abb_gofa_crb15000_5_95"), folder)
    report = deepcopy(record["simulation_model"])
    report["embodiment_id"] = "abb_gofa_crb15000_12_127"
    with pytest.raises(assets.AssetPreparationError, match="model_identity_changed"):
        gofa.verify_report(folder, report, identity="abb_gofa_crb15000_5_95")
