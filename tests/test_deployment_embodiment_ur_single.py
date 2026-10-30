"""Synthetic source/SDK tests: importer wiring, not native or hardware proof."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as XML

import pytest

from blueprint_pipeline import deployment_embodiment_assets as assets
from blueprint_pipeline import deployment_embodiment_runtime_configuration as runtime
from blueprint_pipeline import deployment_embodiment_runtime_compatibility as compatibility
from blueprint_pipeline import deployment_embodiment_ur_single_composition as composition
from tests.test_deployment_embodiment_tool_composition import composed  # noqa: F401
from tests.test_deployment_embodiment_fr3_composition import fr3_composed  # noqa: F401
from tests.test_deployment_embodiment_gofa import source_case as gofa_source  # noqa: F401


@pytest.mark.parametrize("identity", composition.VARIANTS)
def test_single_source_composition_preserves_mount_tool_limits_and_import_binding(composed, tmp_path, monkeypatch, identity):  # noqa: F811 - imported pytest fixture
    from blueprint_pipeline import deployment_embodiment_asset_import as importer
    from tests.test_deployment_embodiment_runtime_configuration import _fake_sdk_from_urdf
    _, _, get = composed
    bundle = tmp_path / "single-source"
    receipt = assets.materialize_sources(identity, bundle, get=get)
    assert receipt["source_arm_count"] == 1
    assert [row["name"] for row in receipt["components"]] == ["arm", "tool"]
    assert len(receipt["licenses"]) == 2
    assert assets.verify_source_bundle(bundle) == receipt
    assert assets.materialize_sources(identity, bundle, get=lambda *a, **k: pytest.fail("retry fetch")) == receipt
    configured = tmp_path / "configured"
    record = runtime.prepare_runtime_configuration(bundle, configured)
    fields = record["manifest_fields"]
    assert len(fields["joints"]) == 6 and len(fields["grippers"]) == 1
    assert [arm["name"] for arm in fields["arms"]] == ["arm"]
    assert record["station_parameters"] is None
    assert fields["arms"][0]["tool_from_tcp"]["position_m"] == pytest.approx([0, 0, .134])
    assert fields["arms"][0]["tool_from_tcp"]["orientation_xyzw"] == pytest.approx([0, 0, 1, 0], abs=1e-14)
    assert fields["grippers"][0]["joints"] == [
        {"name": "robotiq_85_left_knuckle_joint", "open": 0., "closed": .7929},
        {"name": "right_finger", "open": 0., "closed": -.7929}]
    expanded = XML.parse(configured / "robot.urdf").getroot()
    assert expanded.find("joint[@name='adapter']/origin").get("xyz") == "0 0 .011"
    _fake_sdk_from_urdf(monkeypatch)
    prepared = tmp_path / "prepared"
    importer.import_source_asset(configured, prepared, fix_base=True)
    bound = runtime.bind_prepared_configuration(configured, prepared, "https://models.example/development-only.usda")
    assert bound["robot_embodiment"]["embodiment_id"] == identity
    assert bound["robot_embodiment"]["form"] == "fixed_base_arm"
    assert not bound["execution_authorized"] and not receipt["policy_compatibility_established"]


@pytest.mark.parametrize("fault", ["revision", "arm", "tool", "argument", "claim"])
def test_unreviewed_mount_source_variant_or_claim_cannot_reuse_fixed_recipe(fault):
    recipe = deepcopy(assets.recipe_for(next(iter(composition.VARIANTS))))
    if fault == "revision":
        recipe["source"]["revision"] = "a"*40
    elif fault == "arm":
        recipe["composition"]["arm_embodiment_id"] = "universal_robots_ur10e"
    elif fault == "tool":
        recipe["composition"]["tool_source"]["revision"] = "b"*40
    elif fault == "argument":
        recipe["source"]["import_arguments"]["ur_type"] = "ur10e"
    else:
        recipe["composition"]["publisher_hardware_match_established"] = True
    with pytest.raises(assets.AssetPreparationError):
        composition.component_recipe(recipe)


def current21():
    path = Path(__file__).parent / "fixtures/runtime_configuration_2c2977.py"
    assert "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() == compatibility.CURRENT_21_GENERATOR
    spec = importlib.util.spec_from_file_location("blueprint_pipeline._current21_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module._PROFILES) == compatibility.CURRENT_21_PROFILES
    assert all(runtime._PROFILES[key] == value for key, value in module._PROFILES.items())
    return module


@pytest.mark.parametrize("kind", ["dual_ur", "single_fr3", "gofa5", "gofa12"])
def test_exact_current21_generator_reuses_original_contract_without_rewrite(composed, fr3_composed, gofa_source, tmp_path, monkeypatch, kind):  # noqa: F811 - imported pytest fixtures
    from tests.test_deployment_embodiment_runtime_compatibility import _reseal
    source = {"dual_ur": composed[0], "single_fr3": fr3_composed[0]}.get(kind)
    if source is None:
        source = gofa_source("abb_gofa_crb15000_5_95" if kind == "gofa5" else "abb_gofa_crb15000_12_127")
    old = current21()
    output = tmp_path / "previous-config"
    original = old.prepare_runtime_configuration(source, output)
    before = (output / runtime.RECEIPT).read_bytes()
    monkeypatch.setattr(runtime, "_expand_xacro", lambda *a, **k: pytest.fail("retained source reexpansion"))
    assert runtime.verify_runtime_configuration(output) == original
    assert runtime.prepare_runtime_configuration(source, output) == original
    assert (output / runtime.RECEIPT).read_bytes() == before
    changed = deepcopy(original)
    changed["manifest_fields"]["cameras"][0]["position_m"][0] += .01
    _reseal(output, changed)
    with pytest.raises(assets.AssetPreparationError, match="predecessor_metadata_changed"):
        runtime.verify_runtime_configuration(output)
