"""Exact old-generator software fixtures; no historical-provider or task claim."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from blueprint_pipeline import deployment_embodiment_assets as assets
from blueprint_pipeline import deployment_embodiment_runtime_configuration as runtime
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest as digest
from blueprint_pipeline.deployment_embodiment_runtime_compatibility import PREDECESSOR_GENERATOR, PREDECESSOR_PROFILES
from tests.test_deployment_embodiment_tool_composition import composed  # noqa: F401
from tests.test_deployment_embodiment_fr3_composition import fr3_composed  # noqa: F401
from tests.test_deployment_embodiment_mjcf_configuration import source_case  # noqa: F401


def _legacy():
    path = Path(__file__).parent / "fixtures" / "runtime_configuration_9bd.py"
    assert "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() == PREDECESSOR_GENERATOR
    spec = importlib.util.spec_from_file_location("blueprint_pipeline._predecessor_runtime_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def previous(composed, tmp_path):  # noqa: F811 - imported pytest fixture
    source, _, _ = composed
    legacy = _legacy()
    output = tmp_path / "legacy-configured"
    record = legacy.prepare_runtime_configuration(source, output)
    assert record["generator_module_sha256"] == PREDECESSOR_GENERATOR
    assert legacy.verify_runtime_configuration(output) == record
    return source, output, record


def _reseal(output, record):
    path = output / runtime.RECEIPT
    path.chmod(0o644)
    record["configuration_digest"] = digest(record, digest_field="configuration_digest")
    path.write_text(json.dumps(record, indent=2) + "\n")


def test_exact_parent_generated_configuration_is_reused_read_only_without_xacro(previous, monkeypatch):
    source, output, record = previous
    before = (output / runtime.RECEIPT).read_bytes()
    monkeypatch.setattr(runtime, "_expand_xacro", lambda *a, **k: pytest.fail("re-expansion on retained read"))
    assert runtime.verify_runtime_configuration(output) == record
    assert runtime.prepare_runtime_configuration(source, output) == record
    assert (output / runtime.RECEIPT).read_bytes() == before
    assert len(PREDECESSOR_PROFILES) == 18
    assert "franka_research_3_robotiq_2f85_sim" not in PREDECESSOR_PROFILES


@pytest.mark.parametrize("generator", ["sha256:" + "f"*64,
    "sha256:6a70882d5aac2b06780ca67829c7f51c237235ac9611201de888dc1db102b61e"])
def test_unknown_and_pre_trlc_axol_correction_generators_are_refused(previous, generator):
    _, output, record = previous
    _reseal(output, {**record, "generator_module_sha256": generator})
    with pytest.raises(assets.AssetPreparationError, match="receipt_changed"):
        runtime.verify_runtime_configuration(output)


@pytest.mark.parametrize("field", ["closure", "tcp", "camera", "reset", "gain", "station", "warning", "extra"])
def test_resealed_changed_contract_cannot_borrow_predecessor_identity(previous, field):
    _, output, original = previous
    record = deepcopy(original)
    manifest = record["manifest_fields"]
    if field == "closure":
        joint = manifest["grippers"][0]["joints"][0]
        joint["open"], joint["closed"] = joint["closed"], joint["open"]
    elif field == "tcp":
        manifest["arms"][0]["tool_from_tcp"]["position_m"][2] += .01
    elif field == "camera":
        manifest["cameras"][0]["position_m"][0] += .01
    elif field == "reset":
        manifest["reset_joint_positions"][manifest["joints"][0]["name"]] += .01
    elif field == "gain":
        manifest["actuator"]["stiffness"] += 1
    elif field == "station":
        record["station_parameters"]["arms"][0]["station_from_base"]["translation_m"][1] += .01
    elif field == "warning":
        record["source_limitations"] = []
    else:
        record["unverified_override"] = True
    _reseal(output, record)
    with pytest.raises(assets.AssetPreparationError, match="predecessor_metadata_changed"):
        runtime.verify_runtime_configuration(output)


def test_existing_profile_semantics_cannot_change_under_predecessor_acceptance(previous, monkeypatch):
    _, output, record = previous
    profiles = deepcopy(runtime._PROFILES)
    profiles[record["embodiment_id"]][0]["max_width_m"] += .01
    monkeypatch.setattr(runtime, "_PROFILES", profiles)
    with pytest.raises(assets.AssetPreparationError, match="predecessor_metadata_changed"):
        runtime.verify_runtime_configuration(output)


def test_source_or_generated_bytes_still_must_match_before_legacy_metadata(previous):
    _, output, _ = previous
    path = output / "robot.urdf"
    path.chmod(0o644)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(assets.AssetPreparationError, match="receipt_changed"):
        runtime.verify_runtime_configuration(output)


def test_new_fr3_cannot_claim_predecessor_generator(fr3_composed, tmp_path):  # noqa: F811
    source, _, _, _ = fr3_composed
    output = tmp_path / "fr3-configured"
    record = runtime.prepare_runtime_configuration(source, output)
    _reseal(output, {**record, "generator_module_sha256": PREDECESSOR_GENERATOR})
    with pytest.raises(assets.AssetPreparationError, match="receipt_changed"):
        runtime.verify_runtime_configuration(output)


def test_exact_parent_mjcf_reconstructs_native_source_metadata(source_case, tmp_path):  # noqa: F811
    source = source_case()
    output = tmp_path / "legacy-kuka"
    record = _legacy().prepare_runtime_configuration(source, output)
    assert runtime.verify_runtime_configuration(output) == record
    record["source_kinematics"]["reset"]["qpos_rad"][0] += .01
    _reseal(output, record)
    with pytest.raises(assets.AssetPreparationError, match="predecessor_metadata_changed"):
        runtime.verify_runtime_configuration(output)
