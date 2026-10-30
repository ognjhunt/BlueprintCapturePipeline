"""Pinned public XML, synthetic test meshes and USD importer doubles; no native proof."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as XML

import numpy as np
import pytest

from blueprint_pipeline import deployment_embodiment_assets as assets
from blueprint_pipeline import deployment_embodiment_asset_import as importer
from blueprint_pipeline import deployment_embodiment_runtime_configuration as runtime
from blueprint_pipeline import deployment_embodiment_runtime_compatibility as compatibility
from blueprint_pipeline import deployment_embodiment_yaskawa_composition as composition
from blueprint_pipeline import deployment_embodiment_yaskawa_model as model
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_deployment_embodiment_gofa import _stl, _sdk
from tests.test_deployment_embodiment_industrial import _poses
from tests.test_deployment_embodiment_industrial import public_sources as industrial_sources  # noqa: F401

FIXTURE = Path(__file__).parent / "fixtures/yaskawa_robotiq"


@pytest.fixture
def yaskawa_sources(tmp_path):
    industrial = FIXTURE.parent / "industrial_robotiq"
    inventory = {
        **json.loads((FIXTURE / "inventory.json").read_text()),
        "tool": json.loads((industrial / "inventory.json").read_text())["tool"],
    }
    sets = {}
    for name, component in inventory.items():
        files = {}
        for row in component["files"]:
            if row["synthetic_geometry_in_tests"]:
                payload = b"<COLLADA/>" if row["path"].lower().endswith(".dae") else _stl()
            else:
                payload = (
                    (industrial if name == "tool" else FIXTURE) / name / row["path"]
                ).read_bytes()
                assert len(payload) == row["size_bytes"]
                assert "sha256:" + hashlib.sha256(payload).hexdigest() == row["sha256"]
            files[row["path"]] = payload
        sets.setdefault((component["repository_url"], component["revision"]), {}).update(files)

    def get(url, *, maximum):
        for (repo, revision), files in sets.items():
            if revision in url and repo.removeprefix("https://github.com/") in url:
                if "/git/trees/" in url:
                    return json.dumps(
                        {
                            "truncated": False,
                            "tree": [
                                {
                                    "path": name,
                                    "type": "blob",
                                    "mode": "100644",
                                    "size": len(data),
                                    "sha": assets._git_blob_sha(data),
                                }
                                for name, data in files.items()
                            ],
                        }
                    ).encode()
                return files[url.split("/" + revision + "/", 1)[1]]
        pytest.fail("Unexpected URL: " + url)

    def prepare(identity):
        folder = tmp_path / (identity + "-source")
        assets.materialize_sources(identity, folder, get=get)
        return folder

    return prepare


@pytest.mark.parametrize("identity", composition.VARIANTS)
def test_source_fk_limits_mass_separation_import_and_named_tool_binding(
    yaskawa_sources, tmp_path, monkeypatch, identity
):
    bare, configured = tmp_path / "bare", tmp_path / "configuration"
    base = composition.VARIANTS[identity]
    runtime.prepare_runtime_configuration(yaskawa_sources(base), bare)
    bundle = yaskawa_sources(identity)
    source_before = {p: p.read_bytes() for p in (bundle / "source").rglob("*") if p.is_file()}
    record = runtime.prepare_runtime_configuration(bundle, configured)
    assert runtime.verify_runtime_configuration(configured) == record
    original, result = (
        XML.parse(bare / "robot.urdf").getroot(),
        XML.parse(configured / "robot.urdf").getroot(),
    )
    source_model = XML.parse(configured / model.SOURCE_MODEL).getroot()
    assert len(original.findall("link/inertial")) == 0
    assert len(source_model.findall("link/inertial")) == 9
    assert len(result.findall("link/inertial")) == 16
    names = [j.get("name") for j in original.findall("joint") if j.get("type") == "revolute"]
    for joint in original.findall("joint"):
        other = result.find(f"joint[@name='{joint.get('name')}']")
        assert other is not None
        assert {e.tag: e.attrib for e in joint} == {
            e.tag: e.attrib for e in other if e.tag != "drive"
        }
    for scalar in (0.0, 0.07, -0.12):
        angles = {name: scalar * (i + 1) for i, name in enumerate(names)}
        ap, bp = _poses(original, angles), _poses(result, angles)
        for link in ap:
            assert bp[link] == pytest.approx(ap[link], abs=1e-12)
        assert np.linalg.inv(bp["tool0"]) @ bp["robotiq_85_base_link"] == pytest.approx(
            np.eye(4), abs=1e-12
        )
        assert np.linalg.inv(bp["robotiq_85_base_link"]) @ bp["grasp_link"] == pytest.approx(
            np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0.134], [0, 0, 0, 1]]), abs=1e-12
        )
    report = record["simulation_model"]
    arm = report["arm_model"]
    assert arm["modeled_robot_mass_total_kg"] == pytest.approx(model.NOMINAL_ROBOT_KG[base])
    assert report["tool_source_dynamics"]["mass_total_kg"] == pytest.approx(0.92086427304)
    assert report["combined_mass_total_kg"] == pytest.approx(
        model.NOMINAL_ROBOT_KG[base] + 0.92086427304
    )
    # Independently recompute tensors and normalization from source geometry,
    # rather than comparing the implementation to mirrored output constants.
    from blueprint_pipeline.deployment_embodiment_gofa_model import _bounds

    volumes = []
    for row in arm["links"]:
        size, center, _ = _bounds(source_model.find(f"link[@name='{row['link']}']"), configured)
        volumes.append(np.prod(size))
        assert row["center_of_mass_m"] == pytest.approx(center)
        expected = row["mass_kg"] / 12 * (np.dot(size, size) - np.array(size) ** 2)
        assert row["diagonal_inertia_kg_m2"] == pytest.approx(expected)
        assert max(expected) <= sum(expected) / 2 + 1e-12
    assert [r["mass_kg"] for r in arm["links"]] == pytest.approx(
        model.NOMINAL_ROBOT_KG[base] * np.array(volumes) / sum(volumes)
    )
    fields = record["manifest_fields"]
    assert len(fields["joints"]) == 6 and len(fields["grippers"][0]["joints"]) == 6
    assert [c["name"] for c in fields["cameras"]] == ["overhead", "arm_wrist"]
    for declaration, drive in zip(fields["joints"], arm["joints"], strict=True):
        assert declaration["stiffness"] == drive["stiffness_nm_per_rad"]
        assert declaration["damping"] == drive["damping_nm_s_per_rad"]
    if base == "yaskawa_gp12":
        assert fields["joints"][2]["upper"] == 2.6179
        assert arm["mass_reference"]["exact_source_variant_established"] is False
        assert any("155_degrees" in x for x in record["simulation_assumptions"])
    else:
        assert arm["mass_reference"]["exact_source_variant_established"] is True
    assert record["source_dynamics"]["simulator_dynamics_validated"] is False
    _sdk(monkeypatch)
    prepared = tmp_path / "imported"
    imported = importer.import_source_asset(configured, prepared, fix_base=True)
    assert importer.verify_prepared_asset_directory(prepared) == imported
    assert imported["simulation_model"] == report
    observations = model.usd_values(prepared / "robot.usda", report)
    assert (
        len(observations["arm_model"]["links"]) == 7
        and len(observations["tool_source_dynamics"]["links"]) == 9
    )
    assert [r["effort_limit_nm"] for r in observations["arm_model"]["joints"]] == pytest.approx(
        [float(original.find(f"joint[@name='{n}']/limit").get("effort")) for n in names]
    )
    binding = runtime.bind_prepared_configuration(
        configured, prepared, "https://models.example/development-only.usda"
    )
    assert binding["execution_authorized"] is False
    assert binding["robot_embodiment"]["embodiment_id"] == identity
    assert all(p.read_bytes() == payload for p, payload in source_before.items())


@pytest.mark.parametrize("fault", ["mass", "tensor", "force", "unit", "tool_mass", "cross_variant"])
def test_model_and_actual_usd_tampering_is_rejected(yaskawa_sources, tmp_path, monkeypatch, fault):
    from pxr import Gf, Usd, UsdPhysics

    _sdk(monkeypatch)
    folder = tmp_path / "imported"
    receipt = importer.import_source_asset(
        yaskawa_sources("yaskawa_gp12_robotiq_2f85_sim"), folder, fix_base=True
    )
    path = folder / "robot.usda"
    folder.chmod(0o755)
    path.chmod(0o644)
    report = receipt["simulation_model"]
    if fault == "cross_variant":
        changed = deepcopy(report)
        changed["embodiment_id"] = "yaskawa_gp20_robotiq_2f85_sim"
        changed["model_digest"] = canonical_digest(changed, digest_field="model_digest")
        with pytest.raises(assets.AssetPreparationError, match="model_identity_changed"):
            model.verify_report(folder, changed, identity=receipt["embodiment_id"])
        return
    stage = Usd.Stage.Open(str(path))
    prim = stage.GetPrimAtPath("/robot/link_2_l")
    if fault == "mass":
        UsdPhysics.MassAPI(prim).GetMassAttr().Set(0.1)
    elif fault == "tensor":
        UsdPhysics.MassAPI(prim).GetPrincipalAxesAttr().Set(Gf.Quatf(0, 1, 0, 0))
    elif fault == "force":
        UsdPhysics.DriveAPI(stage.GetPrimAtPath("/robot/joint_2_l"), "angular").GetTypeAttr().Set(
            "acceleration"
        )
    elif fault == "unit":
        UsdPhysics.SetStageKilogramsPerUnit(stage, 0.001)
    else:
        UsdPhysics.MassAPI(stage.GetPrimAtPath("/robot/robotiq_85_base_link")).GetMassAttr().Set(
            1.0
        )
    stage.GetRootLayer().Save()
    with pytest.raises(assets.AssetPreparationError, match="usd_"):
        model.usd_values(path, report)


def previous_runtime():
    path = FIXTURE / "runtime_configuration_4dde.py"
    assert (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        == compatibility.CURRENT_28_GENERATOR
    )
    spec = importlib.util.spec_from_file_location("blueprint_pipeline._previous28", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module._PROFILES) == compatibility.CURRENT_28_PROFILES
    assert all(runtime._PROFILES[key] == value for key, value in module._PROFILES.items())
    return module


def test_original28_profiles_34_recipes_109_rows_18_policy_identities_unchanged():
    previous = json.loads((FIXTURE / "predecessor_identities.json").read_text())
    package = Path(assets.__file__).parent
    root = package.parents[1]
    recipes = {r["embodiment_id"]: r for r in assets.asset_recipes()["recipes"]}
    rows = {
        r["candidate_id"]: r
        for r in json.loads((root / "config/beta_policy_candidates.v1.json").read_text())[
            "candidates"
        ]
    }
    assert (len(recipes), len(rows), len(runtime._PROFILES)) == (49, 120, 42)
    for name, expected in previous["recipes"].items():
        assert canonical_digest(recipes[name]) == expected
    for name, expected in previous["catalog"].items():
        assert canonical_digest(rows[name]) == expected
    for name, expected in previous["modules"].items():
        assert "sha256:" + hashlib.sha256((package / name).read_bytes()).hexdigest() == expected
    current = json.loads(
        (
            root / "config/policy_runtimes/scripted_robot_baseline_registration_profile.json"
        ).read_text()
    )
    for key, expected in previous["registration_profile"].items():
        assert (
            {b: current[key][b] for b in expected} if key == "bodies" else current[key]
        ) == expected
    assert len(current["bodies"]) == 24
    previous_runtime()


@pytest.mark.parametrize(
    "identity", ["fanuc_crx_10ia_robotiq_2f85_sim", "abb_gofa_crb15000_5_95_robotiq_2f85_sim"]
)
def test_previous28_reconstruction_preserves_estimated_composite_dynamics(
    request, tmp_path, monkeypatch, identity
):
    from tests.test_deployment_embodiment_runtime_compatibility import _reseal

    old_sources = request.getfixturevalue("industrial_sources")
    old = previous_runtime()
    folder = tmp_path / "old"
    record = old.prepare_runtime_configuration(old_sources(identity), folder)
    before = (folder / runtime.RECEIPT).read_bytes()
    monkeypatch.setattr(
        runtime, "_expand_xacro", lambda *a, **kw: pytest.fail("No old re-expansion")
    )
    assert runtime.verify_runtime_configuration(folder) == record
    assert (folder / runtime.RECEIPT).read_bytes() == before
    changed = deepcopy(record)
    changed["manifest_fields"]["joints"][0]["upper"] += 0.01
    _reseal(folder, changed)
    with pytest.raises(assets.AssetPreparationError, match="predecessor_metadata_changed"):
        runtime.verify_runtime_configuration(folder)


def test_predecessor_identity_never_accepts_new_yaskawa_configuration(yaskawa_sources, tmp_path):
    from tests.test_deployment_embodiment_runtime_compatibility import _reseal

    folder = tmp_path / "new"
    record = runtime.prepare_runtime_configuration(
        yaskawa_sources(next(iter(composition.VARIANTS))), folder
    )
    record["generator_module_sha256"] = compatibility.CURRENT_28_GENERATOR
    _reseal(folder, record)
    with pytest.raises(assets.AssetPreparationError):
        runtime.verify_runtime_configuration(folder)


@pytest.mark.parametrize("fault", ["pin", "mount", "claim", "tool"])
def test_fixed_source_composition_refuses_unreviewed_changes(fault):
    recipe = deepcopy(assets.recipe_for(next(iter(composition.VARIANTS))))
    if fault == "pin":
        recipe["source"]["revision"] = "a" * 40
    elif fault == "mount":
        recipe["composition"]["mount"]["translation_m"][2] = 0.011
    elif fault == "claim":
        recipe["composition"]["publisher_hardware_match_established"] = True
    else:
        recipe["composition"]["tool_source"]["revision"] = "a" * 40
    with pytest.raises(assets.AssetPreparationError):
        composition.component_recipe(recipe)


@pytest.mark.parametrize("fault", ["assembly", "unrecorded"])
def test_generated_composition_and_complete_source_closure_are_verified(yaskawa_sources, fault):
    source = yaskawa_sources(next(iter(composition.VARIANTS)))
    folder = source / "source"
    folder.chmod(0o755)
    target = folder / (composition.ENTRY if fault == "assembly" else "unrecorded.xml")
    if target.exists():
        target.chmod(0o644)
    target.write_text("<robot name='unreviewed'/>")
    with pytest.raises(assets.AssetPreparationError):
        assets.verify_source_bundle(source)


def test_resealed_model_estimate_does_not_bypass_source_reconstruction(yaskawa_sources, tmp_path):
    from tests.test_deployment_embodiment_runtime_compatibility import _reseal

    folder = tmp_path / "configured"
    record = runtime.prepare_runtime_configuration(
        yaskawa_sources(next(iter(composition.VARIANTS))), folder
    )
    record["simulation_model"]["arm_model"]["links"][0]["mass_kg"] *= 1.1
    record["simulation_model"]["model_digest"] = canonical_digest(
        record["simulation_model"], digest_field="model_digest"
    )
    _reseal(folder, record)
    with pytest.raises(assets.AssetPreparationError, match="model_report_changed"):
        runtime.verify_runtime_configuration(folder)
