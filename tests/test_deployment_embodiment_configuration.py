"""Actual USD inspection, signed transport and no-allocation boundary.

Robot bytes are synthetic OpenUSD fixtures; these tests establish software
behavior, never OEM compatibility, simulator execution or physical performance.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline import deployment_embodiment_configuration as configuration
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from tests.test_deployment_embodiment_support import _client, _native_inputs
from tests.test_team_embodiment import bimanual_inputs


def _fetch(source):
    def retain(url, destination):
        assert url.startswith("https://")
        destination.write_bytes(source.read_bytes())
    return retain


def test_real_usd_identity_and_structure_are_checked_without_scene_or_run(tmp_path):
    path, manifest, _ = _native_inputs(tmp_path)
    request = {"embodiment_id": manifest["embodiment_id"], "robot_embodiment": manifest}
    result = configuration.inspect_configuration(request, fetch=_fetch(path))
    assert result["status"] == "configured_not_executed"
    assert result["blockers"] == []
    assert result["claim_ceiling"] == "configuration_only"
    assert result["model"]["sha256"] == manifest["model"]["sha256"]
    assert result["request_digest"] == cross_runtime_canonical_digest(request)
    assert result["model"]["size_bytes"] == path.stat().st_size
    assert result["structure"]["joint_count"] == 7
    assert not result["execution_authorized"] and not result["provider_mutation_performed"]
    assert result["preflight_digest"] == cross_runtime_canonical_digest(result, digest_field="preflight_digest")
    assert "scene_plan" not in result and "run_id" not in result
    assert str(tmp_path) not in json.dumps(result)
    assert configuration.inspect_configuration(request, fetch=_fetch(path)) == result
    fixture = json.loads((Path(__file__).parent / "fixtures/deployment_embodiment/configuration-preflight.v1.json").read_text())
    assert fixture["fixture_scope"] == "development_only_synthetic_usd_software_contract"
    assert fixture["request"] == request and fixture["response"] == result


def test_bimanual_manifest_uses_two_real_declared_tool_channels(tmp_path):
    path, manifest = bimanual_inputs(tmp_path, embodiment_id="i2rt_yam_bimanual", arm_dof=6)
    result = configuration.inspect_configuration({"embodiment_id": "i2rt_yam_bimanual",
                                                "robot_embodiment": manifest}, fetch=_fetch(path))
    assert result["status"] == "configured_not_executed" and not result["blockers"]
    wrong = copy.deepcopy(manifest)
    wrong["embodiment_id"] = "universal_robots_ur5e_bimanual"
    with pytest.raises(ValueError, match="identity_mismatch"):
        configuration.inspect_configuration({"embodiment_id": "i2rt_yam_bimanual",
                                             "robot_embodiment": wrong}, fetch=_fetch(path))


def test_digest_and_actual_joint_structure_fail_truthfully(tmp_path):
    path, manifest, _ = _native_inputs(tmp_path)
    wrong = copy.deepcopy(manifest)
    wrong["joints"][0]["name"] = "missing_joint"
    wrong["reset_joint_positions"]["missing_joint"] = wrong["reset_joint_positions"].pop("test_joint_0")
    result = configuration.inspect_configuration({"embodiment_id": wrong["embodiment_id"],
                                                "robot_embodiment": wrong}, fetch=_fetch(path))
    assert result["status"] == "requires_setup"
    assert "team_embodiment_model_joint_missing:missing_joint" in result["blockers"]
    path.write_bytes(path.read_bytes() + b"\n# changed")
    with pytest.raises(ValueError, match="model_digest_mismatch"):
        configuration.inspect_configuration({"embodiment_id": manifest["embodiment_id"],
                                             "robot_embodiment": manifest}, fetch=_fetch(path))


@pytest.mark.parametrize("content", [
    b'#usda 1.0\n( subLayers = [@https://example.invalid/private.usda@] )\n',
    b'#usda 1.0\ndef Xform "robot" {\n asset hidden = @/etc/passwd@\n}\n',
    b'#usda 1.0\ndef Xform "robot" {\n asset hidden.timeSamples = {0: @https://example.invalid/texture@}\n}\n',
])
def test_unbound_usd_external_asset_is_not_opened(tmp_path, monkeypatch, content):
    import hashlib

    path, manifest, _ = _native_inputs(tmp_path)
    path.write_bytes(content)
    manifest["model"]["sha256"] = "sha256:" + hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(configuration, "read_model_structure", lambda _: pytest.fail("external layer opened"))
    with pytest.raises(ValueError, match="self_contained_usd_required"):
        configuration.inspect_configuration({"embodiment_id": manifest["embodiment_id"],
                                             "robot_embodiment": manifest}, fetch=_fetch(path))


def test_flattened_instance_prototypes_are_self_contained_but_an_external_arc_is_not(tmp_path):
    """USD flattening writes an internal reference (empty asset path) for each instance."""
    from pxr import Sdf

    text = ('#usda 1.0\n(\n defaultPrim = "robot"\n)\ndef Xform "robot" {\n'
            ' def Material "grey" (instanceable = true\n  prepend references = </Flattened_Prototype_1>\n ) {}\n}\n'
            'def Material "Flattened_Prototype_1" {}\n')
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.ImportFromString(text)
    assert list(layer.GetExternalReferences()) == [""]
    assert configuration._has_external_assets(layer, Sdf) is False
    layer.ImportFromString(text.replace("</Flattened_Prototype_1>", "@./materials.usda@</Flattened_Prototype_1>"))
    assert configuration._has_external_assets(layer, Sdf) is True


def test_signed_route_requires_trusted_issuer_and_bounds_body(tmp_path, monkeypatch):
    path, manifest, _ = _native_inputs(tmp_path)
    monkeypatch.setattr(configuration, "_fetch_model", _fetch(path))
    client = _client()
    headers = {"x-blueprint-pipeline-signature": "verified-by-admission-dependency"}
    route = "/api/live-pipeline/deployment-embodiment-native-preflight"
    payload = {"embodiment_id": manifest["embodiment_id"], "robot_embodiment": manifest}
    assert client.post(route, json=payload).status_code == 403
    assert _client(issuer="other").post(route, headers=headers, json=payload).status_code == 403
    response = client.post(route, headers=headers, json=payload)
    assert response.status_code == 200 and response.json()["status"] == "configured_not_executed"
    assert response.headers["cache-control"] == "private, no-store"
    assert client.post(route, headers=headers, content=b"x" * (128 * 1024 + 1)).status_code == 413
    assert client.post(route, headers=headers, json={**payload, "scene_plan": {}}).status_code == 422
    recipes = client.get("/api/live-pipeline/deployment-embodiment-asset-recipes", headers=headers)
    assert recipes.status_code == 200 and recipes.json()["claim_ceiling"] == "source_asset_not_qualified"
