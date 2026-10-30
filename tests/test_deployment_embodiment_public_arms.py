"""Pinned public XML with synthetic meshes/SDK doubles: software contracts, not native proof."""

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
from blueprint_pipeline import deployment_embodiment_hc20_model as hc20_model
from blueprint_pipeline import deployment_embodiment_public_arm_composition as composition
from blueprint_pipeline import deployment_embodiment_runtime_compatibility as compatibility
from blueprint_pipeline import deployment_embodiment_runtime_configuration as runtime
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.deployment_embodiment_station import _origin_matrix
from tests.test_deployment_embodiment_gofa import _sdk, _stl
from tests.test_deployment_embodiment_industrial import _poses
from tests.test_deployment_embodiment_kuka import public_sources as kuka_sources  # noqa: F401
from tests.test_deployment_embodiment_yaskawa import yaskawa_sources  # noqa: F401

FIXTURE = Path(__file__).parent / "fixtures/public_fixed_arms"
TOOL_MASS_KG = 0.92086427304
KUKA_IONTEC = ("kuka_kr20_r3100", "kuka_kr30_r2100", "kuka_kr50_r2100", "kuka_kr50_r2500", "kuka_kr70_r2100")
BARE = {"fanuc_m10id_12": 6, "yaskawa_hc20": 6, "agile_robots_diana_7": 7, **dict.fromkeys(KUKA_IONTEC, 6)}
NEW_RECIPES = {*BARE, "kinova_jaco2_j2n6s300", *composition.VARIANTS}


@pytest.fixture
def public_sources(tmp_path):
    industrial = FIXTURE.parent / "industrial_robotiq"
    inventory = {**json.loads((FIXTURE / "inventory.json").read_text()),
                 "tool": json.loads((industrial / "inventory.json").read_text())["tool"]}
    sets = {}
    for name, component in inventory.items():
        files = {}
        for row in component["files"]:
            if row["synthetic_geometry_in_tests"]:
                payload = b"<COLLADA/>" if row["path"].lower().endswith(".dae") else _stl()
            else:
                payload = ((industrial if name == "tool" else FIXTURE) / name / row["path"]).read_bytes()
                assert len(payload) == row["size_bytes"]
                assert "sha256:" + hashlib.sha256(payload).hexdigest() == row["sha256"]
                assert assets._git_blob_sha(payload) == row["git_blob_sha"]
            files[row["path"]] = payload
        sets.setdefault((component["repository_url"], component["revision"]), {}).update(files)

    def get(url, *, maximum):
        for (repo, revision), files in sets.items():
            if revision in url and repo.removeprefix("https://github.com/") in url:
                if "/git/trees/" in url:
                    return json.dumps({"truncated": False, "tree": [
                        {"path": name, "type": "blob", "mode": "100644", "size": len(data),
                         "sha": assets._git_blob_sha(data)} for name, data in files.items()]}).encode()
                return files[url.split("/" + revision + "/", 1)[1]]
        pytest.fail("Unexpected source URL: " + url)

    def prepare(identity):
        folder = tmp_path / (identity + "-source")
        if not folder.exists():
            assets.materialize_sources(identity, folder, get=get)
        return folder

    return prepare


def test_new_recipes_record_exact_public_sources_and_explicit_gaps():
    rows = {row["embodiment_id"]: row for row in assets.asset_recipes()["recipes"]}
    assert NEW_RECIPES <= set(rows)
    for identity in NEW_RECIPES:
        row = rows[identity]
        assert row["status"] == "source_available" and row["arm_count"] == 1
        assert row["source"]["license_id"] in {"Apache-2.0", "BSD-3-Clause"}
        assert row["missing"] and row["reason"]
    for identity in KUKA_IONTEC:
        assert {"deployed_kr_iontec_variant_unconfirmed", "suction_end_effector_not_modeled"} <= set(rows[identity]["missing"])
        assert "not Contoro's confirmed robot" in rows[identity]["reason"]
        assert identity + "_robotiq_2f85_sim" not in rows
    assert "inertial_parameters_not_present_in_hc20_macro" in rows["yaskawa_hc20"]["missing"]
    assert "link0_inertial_values_appear_to_be_placeholders" in rows["agile_robots_diana_7"]["missing"]
    assert "publisher_status_of_new_agilerobots_apac_repository_unconfirmed" in rows["agile_robots_diana_7"]["missing"]
    assert "runtime_profile_not_registered_for_three_finger_hand" in rows["kinova_jaco2_j2n6s300"]["missing"]
    for identity in composition.VARIANTS:
        assert "robotiq_2f85_is_simulation_stand_in_not_deployed_tool" in rows[identity]["missing"]
        assert "stand-in" in rows[identity]["reason"]
        assert rows[identity]["composition"]["tool_role"] == composition.TOOL_ROLE
    assert "link_dynamics_are_declared_simulation_estimates" in rows["yaskawa_hc20_robotiq_2f85_sim"]["missing"]
    # Public sources that were reviewed and are not reusable stay out of the catalog.
    assert not {"universal_robots_ur20", "galaxea_r1", "fetch"} & set(rows)


@pytest.mark.parametrize("identity", sorted(BARE))
def test_bare_public_arm_configures_from_exact_source_without_a_tool_claim(public_sources, tmp_path, identity):
    bundle = public_sources(identity)
    before = {p: p.read_bytes() for p in (bundle / "source").rglob("*") if p.is_file()}
    config = tmp_path / "configuration"
    record = runtime.prepare_runtime_configuration(bundle, config)
    assert runtime.verify_runtime_configuration(config) == record
    fields = record["manifest_fields"]
    assert len(fields["joints"]) == BARE[identity] and fields["grippers"] == []
    assert "bare_arm_without_application_tool_or_grasp_capability" in record["simulation_assumptions"]
    dynamics = record["source_dynamics"]
    assert dynamics["moving_link_count"] == BARE[identity] and dynamics["estimated_inertias_added"] is False
    if identity == "yaskawa_hc20":
        assert len(dynamics["moving_links_without_positive_authored_mass_and_inertia"]) == 6
        assert any(row.startswith("missing_positive_authored_moving_link_inertias:")
                   for row in record["source_limitations"])
    else:
        assert dynamics["moving_links_without_positive_authored_mass_and_inertia"] == []
    assert record["execution_authorized"] is False and record["policy_compatibility_established"] is False
    assert all(p.read_bytes() == data for p, data in before.items())


def test_kr_iontec_entry_expands_without_gazebo_or_other_packages(public_sources, tmp_path):
    root = XML.parse(_expanded(public_sources("kuka_kr50_r2100"), tmp_path)).getroot()
    assert [j.get("name") for j in root.findall("joint") if j.get("type") == "revolute"] == [
        "joint_" + str(i) for i in range(1, 7)]
    assert len(root.findall("link/inertial")) == 7
    assert root.find("gazebo") is None


def _expanded(bundle, tmp_path):
    receipt = assets.verify_source_bundle(bundle)
    description = assets.source_description(assets.recipe_for(receipt["embodiment_id"]), receipt)
    source = bundle / "source"
    files = {(source / row["path"]).resolve() for row in receipt["files"]}
    return importer._expand_xacro(source / description["entry_path"], source=source, work=tmp_path / "expanded",
                                  arguments=description["import_arguments"],
                                  roots=importer._package_roots(source, description["package_roots"]),
                                  source_files=files)


def test_jaco2_three_finger_hand_is_recorded_without_an_invented_gripper_binding(public_sources, tmp_path):
    bundle = public_sources("kinova_jaco2_j2n6s300")
    root = XML.parse(_expanded(bundle, tmp_path)).getroot()
    moving = {j.get("name"): j for j in root.findall("joint") if j.get("type") != "fixed"}
    arm = [name for name in moving if "finger" not in name]
    fingers = sorted(name for name in moving if "finger" in name)
    assert len(arm) == 6 and sum(moving[name].get("type") == "continuous" for name in arm) == 4
    assert fingers == sorted([f"j2n6s300_joint_finger_{i}" for i in range(1, 4)]
                             + [f"j2n6s300_joint_finger_tip_{i}" for i in range(1, 4)])
    assert all(moving[name].find("mimic") is None for name in fingers)
    assert {(moving[n].find("limit").get("lower"), moving[n].find("limit").get("upper")) for n in fingers} == {
        ("0", "1.51"), ("0", "2")}
    assert "kinova_jaco2_j2n6s300" not in runtime._PROFILES
    with pytest.raises(assets.AssetPreparationError, match="profile_unavailable:kinova_jaco2_j2n6s300"):
        runtime.prepare_runtime_configuration(bundle, tmp_path / "configuration")


@pytest.mark.parametrize("identity", sorted(composition.VARIANTS))
def test_stand_in_assembly_preserves_source_chain_mount_and_fake_sdk_binding(public_sources, tmp_path, monkeypatch,
                                                                              identity):
    base = composition.VARIANTS[identity]
    bare = tmp_path / "bare"
    runtime.prepare_runtime_configuration(public_sources(base), bare)
    bundle = public_sources(identity)
    before = {p: p.read_bytes() for p in (bundle / "source").rglob("*") if p.is_file()}
    source = assets.verify_source_bundle(bundle)
    assert len(source["licenses"]) == 2 and source["policy_compatibility_established"] is False
    assert source["generated_assembly"]["composition"]["tool_role"] == composition.TOOL_ROLE
    config = tmp_path / "configuration"
    record = runtime.prepare_runtime_configuration(bundle, config)
    assert runtime.verify_runtime_configuration(config) == record
    fields = record["manifest_fields"]
    assert len(fields["joints"]) == BARE[base] and len(fields["grippers"]) == 1
    assert len(fields["grippers"][0]["joints"]) == 6 and [a["name"] for a in fields["arms"]] == ["arm"]
    assert [c["name"] for c in fields["cameras"]] == ["overhead", "arm_wrist"]
    assert fields["arms"][0]["tool_from_tcp"]["position_m"] == pytest.approx([0, 0, .134])
    original = XML.parse(bare / "robot.urdf").getroot()
    result = XML.parse(config / "robot.urdf").getroot()
    parent = composition.mount_descriptor(identity)["parent_link"]
    mount = next(j for j in result.findall("joint") if j.find("child").get("link") == "robotiq_85_base_link")
    assert mount.find("parent").get("link") == parent
    assert np.array(_origin_matrix(mount)) == pytest.approx(np.eye(4))
    names = [j.get("name") for j in original.findall("joint") if j.get("type") in {"revolute", "continuous"}]
    assert len(names) == BARE[base]
    for scalar in (0.0, 0.05, -0.1):
        angles = {name: scalar * (index + 1) for index, name in enumerate(names)}
        ap, bp = _poses(original, angles), _poses(result, angles)
        base_link = runtime._PROFILES[base][0]["base_link"]
        for link in ap:
            if link in {"world", "ee_link", "base"}:
                continue
            assert np.linalg.inv(ap[base_link]) @ ap[link] == pytest.approx(
                np.linalg.inv(bp[base_link]) @ bp[link], abs=1e-12)
        assert np.linalg.inv(bp[parent]) @ bp["robotiq_85_base_link"] == pytest.approx(np.eye(4), abs=1e-12)
        assert np.linalg.inv(bp["robotiq_85_base_link"]) @ bp["grasp_link"] == pytest.approx(
            np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, .134], [0, 0, 0, 1]]), abs=1e-12)
    for joint in original.findall("joint"):
        other = result.find(f"joint[@name='{joint.get('name')}']")
        assert other is not None
        assert {e.tag: e.attrib for e in joint} == {e.tag: e.attrib for e in other if e.tag != "drive"}
    if identity in hc20_model.VARIANTS:
        report = record["simulation_model"]
        assert report["claim_ceiling"] == "unvalidated_simulation_estimate" and report["manufacturer_dynamics"] is False
        arm = report["arm_model"]
        assert arm["modeled_robot_mass_total_kg"] == pytest.approx(140)
        assert arm["mass_reference"]["exact_source_variant_established"] is False
        assert report["tool_source_dynamics"]["mass_total_kg"] == pytest.approx(TOOL_MASS_KG)
        assert report["combined_mass_total_kg"] == pytest.approx(140 + TOOL_MASS_KG)
        assert record["source_dynamics"]["estimated_inertias_added"] is True
        assert "robotiq_2f85_is_simulation_stand_in_not_deployed_tool" in record["simulation_assumptions"]
        assert "hc20_geometry_based_dynamics_and_PD_estimates" in record["simulation_assumptions"]
        from blueprint_pipeline.deployment_embodiment_gofa_model import _bounds
        source_model = XML.parse(config / hc20_model.SOURCE_MODEL).getroot()
        volumes = []
        for row in arm["links"]:
            size, center, _ = _bounds(source_model.find(f"link[@name='{row['link']}']"), config)
            volumes.append(np.prod(size))
            assert row["center_of_mass_m"] == pytest.approx(center)
            assert row["diagonal_inertia_kg_m2"] == pytest.approx(
                row["mass_kg"] / 12 * (np.dot(size, size) - np.array(size) ** 2))
        assert [r["mass_kg"] for r in arm["links"]] == pytest.approx(140 * np.array(volumes) / sum(volumes))
        for declaration, drive in zip(fields["joints"], arm["joints"], strict=True):
            assert declaration["stiffness"] == drive["stiffness_nm_per_rad"]
            assert declaration["damping"] == drive["damping_nm_s_per_rad"]
        assert fields["joints"][0]["lower"] == pytest.approx(-np.pi)
    else:
        assert "simulation_model" not in record
        assert record["source_dynamics"]["moving_links_without_positive_authored_mass_and_inertia"] == []
    _sdk(monkeypatch)
    prepared = tmp_path / "imported"
    imported = importer.import_source_asset(config, prepared, fix_base=True)
    assert importer.verify_prepared_asset_directory(prepared) == imported
    if identity in hc20_model.VARIANTS:
        assert imported["simulation_model"] == record["simulation_model"]
        observations = hc20_model.usd_values(prepared / "robot.usda", record["simulation_model"])
        assert len(observations["arm_model"]["links"]) == 7
        assert len(observations["tool_source_dynamics"]["links"]) == 9
    binding = runtime.bind_prepared_configuration(config, prepared, "https://models.example/development-only.usda")
    assert binding["robot_embodiment"]["embodiment_id"] == identity and binding["execution_authorized"] is False
    assert all(p.read_bytes() == data for p, data in before.items())


@pytest.mark.parametrize("fault", ["mass", "force", "tool_mass", "cross_variant"])
def test_hc20_estimated_model_and_usd_tampering_are_rejected(public_sources, tmp_path, monkeypatch, fault):
    from pxr import Usd, UsdPhysics

    _sdk(monkeypatch)
    folder = tmp_path / "imported"
    receipt = importer.import_source_asset(public_sources("yaskawa_hc20_robotiq_2f85_sim"), folder, fix_base=True)
    report = receipt["simulation_model"]
    if fault == "cross_variant":
        changed = deepcopy(report)
        changed["embodiment_id"] = "yaskawa_gp12_robotiq_2f85_sim"
        changed["model_digest"] = canonical_digest(changed, digest_field="model_digest")
        with pytest.raises(assets.AssetPreparationError, match="model_identity_changed"):
            hc20_model.verify_report(folder, changed, identity=receipt["embodiment_id"])
        return
    path = folder / "robot.usda"
    folder.chmod(0o755)
    path.chmod(0o644)
    stage = Usd.Stage.Open(str(path))
    if fault == "mass":
        UsdPhysics.MassAPI(stage.GetPrimAtPath("/robot/link_2_l")).GetMassAttr().Set(0.1)
    elif fault == "force":
        UsdPhysics.DriveAPI(stage.GetPrimAtPath("/robot/joint_2_l"), "angular").GetTypeAttr().Set("acceleration")
    else:
        UsdPhysics.MassAPI(stage.GetPrimAtPath("/robot/robotiq_85_base_link")).GetMassAttr().Set(1.0)
    stage.GetRootLayer().Save()
    with pytest.raises(assets.AssetPreparationError, match="usd_"):
        hc20_model.usd_values(path, report)


def test_resealed_hc20_estimate_does_not_bypass_source_reconstruction(public_sources, tmp_path):
    from tests.test_deployment_embodiment_runtime_compatibility import _reseal

    folder = tmp_path / "configured"
    record = runtime.prepare_runtime_configuration(public_sources("yaskawa_hc20_robotiq_2f85_sim"), folder)
    record["simulation_model"]["arm_model"]["links"][0]["mass_kg"] *= 1.1
    record["simulation_model"]["model_digest"] = canonical_digest(record["simulation_model"], digest_field="model_digest")
    _reseal(folder, record)
    with pytest.raises(assets.AssetPreparationError, match="model_report_changed"):
        runtime.verify_runtime_configuration(folder)


@pytest.mark.parametrize("fault", ["pin", "mount", "claim", "tool", "tool_role"])
def test_fixed_stand_in_composition_refuses_unreviewed_changes(fault):
    for identity in composition.VARIANTS:
        recipe = deepcopy(assets.recipe_for(identity))
        if fault == "pin":
            recipe["source"]["revision"] = "a" * 40
        elif fault == "mount":
            recipe["composition"]["mount"]["translation_m"][2] = 0.011
        elif fault == "claim":
            recipe["composition"]["publisher_hardware_match_established"] = True
        elif fault == "tool":
            recipe["composition"]["tool_source"]["revision"] = "a" * 40
        else:
            recipe["composition"]["tool_role"] = "deployed_tool"
        with pytest.raises(assets.AssetPreparationError):
            composition.component_recipe(recipe)


@pytest.mark.parametrize("fault", ["assembly", "unrecorded"])
def test_generated_stand_in_assembly_and_complete_closure_are_verified(public_sources, fault):
    source = public_sources("agile_robots_diana_7_robotiq_2f85_sim")
    folder = source / "source"
    folder.chmod(0o755)
    target = folder / (composition.ENTRY if fault == "assembly" else "unrecorded.xml")
    if target.exists():
        target.chmod(0o644)
    target.write_text("<robot name='unreviewed'/>")
    with pytest.raises(assets.AssetPreparationError):
        assets.verify_source_bundle(source)


def previous_runtime():
    path = FIXTURE / "runtime_configuration_867c.py"
    assert "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() == compatibility.CURRENT_31_GENERATOR
    spec = importlib.util.spec_from_file_location("blueprint_pipeline._previous31", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module._PROFILES) == compatibility.CURRENT_31_PROFILES
    assert all(runtime._PROFILES[key] == value for key, value in module._PROFILES.items())
    return module


def test_original31_profiles_37_recipes_112_rows_21_policy_identities_unchanged():
    previous = json.loads((FIXTURE / "predecessor_identities.json").read_text())
    package = Path(assets.__file__).parent
    root = package.parents[1]
    recipes = {r["embodiment_id"]: r for r in assets.asset_recipes()["recipes"]}
    rows = {r["candidate_id"]: r
            for r in json.loads((root / "config/beta_policy_candidates.v1.json").read_text())["candidates"]}
    assert (len(previous["recipes"]), len(previous["catalog"]), len(previous["modules"])) == (37, 112, 20)
    assert (len(recipes), len(rows), len(runtime._PROFILES)) == (49, 120, 42)
    assert set(recipes) == set(previous["recipes"]) | NEW_RECIPES
    for name, expected in previous["recipes"].items():
        assert canonical_digest(recipes[name]) == expected
    for name, expected in previous["catalog"].items():
        assert canonical_digest(rows[name]) == expected
    for name, expected in previous["modules"].items():
        assert "sha256:" + hashlib.sha256((package / name).read_bytes()).hexdigest() == expected
    current = json.loads((root / "config/policy_runtimes/scripted_robot_baseline_registration_profile.json").read_text())
    for key, expected in previous["registration_profile"].items():
        assert ({b: current[key][b] for b in expected} if key == "bodies" else current[key]) == expected
    assert len(previous["registration_profile"]["bodies"]) == 21 and len(current["bodies"]) == 24
    previous_runtime()


@pytest.mark.parametrize("identity,fixture", [
    ("kuka_lbr_iiwa_14_robotiq_2f85_sim", "kuka_sources"),
    ("yaskawa_gp12_robotiq_2f85_sim", "yaskawa_sources"),
])
def test_previous31_receipt_reuse_preserves_complete_model_metadata(request, tmp_path, monkeypatch, identity,
                                                                     fixture):
    from tests.test_deployment_embodiment_runtime_compatibility import _reseal

    sources = request.getfixturevalue(fixture)
    old = previous_runtime()
    folder = tmp_path / "old"
    record = old.prepare_runtime_configuration(sources(identity), folder)
    before = (folder / runtime.RECEIPT).read_bytes()
    monkeypatch.setattr(runtime, "_expand_xacro", lambda *a, **k: pytest.fail("No historical re-expansion"))
    assert runtime.verify_runtime_configuration(folder) == record
    assert (folder / runtime.RECEIPT).read_bytes() == before
    changed = deepcopy(record)
    changed["manifest_fields"]["cameras"][0]["position_m"][0] += 0.01
    _reseal(folder, changed)
    with pytest.raises(assets.AssetPreparationError, match="predecessor_metadata_changed"):
        runtime.verify_runtime_configuration(folder)


@pytest.mark.parametrize("identity", ["fanuc_m10id_12", "yaskawa_hc20_robotiq_2f85_sim"])
def test_predecessor_generator_never_accepts_a_new_configuration(public_sources, tmp_path, identity):
    from tests.test_deployment_embodiment_runtime_compatibility import _reseal

    folder = tmp_path / "new"
    record = runtime.prepare_runtime_configuration(public_sources(identity), folder)
    record["generator_module_sha256"] = compatibility.CURRENT_31_GENERATOR
    _reseal(folder, record)
    with pytest.raises(assets.AssetPreparationError):
        runtime.verify_runtime_configuration(folder)


def previous42_runtime():
    path = FIXTURE / "runtime_configuration_d06a.py"
    assert "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() == compatibility.CURRENT_42_GENERATOR
    spec = importlib.util.spec_from_file_location("blueprint_pipeline._previous42", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module._PROFILES) == compatibility.CURRENT_42_PROFILES and module._PROFILES == runtime._PROFILES
    return module


@pytest.mark.parametrize("identity", ["fanuc_m10id_12", "yaskawa_hc20_robotiq_2f85_sim"])
def test_previous42_receipt_reuse_after_isaac6_name_binding_preserves_complete_metadata(
        public_sources, tmp_path, monkeypatch, identity):
    from tests.test_deployment_embodiment_runtime_compatibility import _reseal

    old = previous42_runtime()
    folder = tmp_path / "old"
    record = old.prepare_runtime_configuration(public_sources(identity), folder)
    assert record["generator_module_sha256"] == compatibility.CURRENT_42_GENERATOR
    monkeypatch.setattr(runtime, "_expand_xacro", lambda *a, **k: pytest.fail("No historical re-expansion"))
    assert runtime.verify_runtime_configuration(folder) == record
    changed = deepcopy(record)
    changed["manifest_fields"]["cameras"][0]["position_m"][0] += 0.01
    _reseal(folder, changed)
    with pytest.raises(assets.AssetPreparationError, match="predecessor_metadata_changed"):
        runtime.verify_runtime_configuration(folder)
