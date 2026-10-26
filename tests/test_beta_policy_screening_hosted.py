"""Arbitrary compatible hosted identities use private delivery and inclusive caps."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import zipfile

import pytest

from blueprint_pipeline import beta_policy_screening_launch as launch
from blueprint_pipeline.beta_policy_candidate_catalog import load_default_catalog, seal_catalog
from blueprint_pipeline.beta_policy_screening_capabilities import build_planning_inputs
from blueprint_pipeline.beta_policy_screening_hosted import request_limit, validate_hosted_delivery
from blueprint_pipeline.beta_policy_screening_plan import build_screening_plan
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.team_candidate_quick10_registry import profile_for_capture
from tests.test_beta_policy_screening_launch import _plan


TOKEN = "hosted-private-token-never-in-bundle"


def hosted_plan(tmp_path, monkeypatch):
    base_plan, facts = _plan(tmp_path, monkeypatch, scene_half="team_scene")
    profile = profile_for_capture(str(tmp_path / "capture"))
    template = next(row for row in load_default_catalog()["candidates"] if row["candidate_id"] == "pi05_droid")
    native_template = launch._read(Path(facts["capabilities"]["candidate_bindings"]["pi05_droid"]
                                      ["execution_spec"]["policy_execution_spec"]["path"]))
    private = tmp_path / "operator" / "hosted-bearer"
    private.write_text(TOKEN)
    private.chmod(0o600)
    rows, settings, specs = [], {}, {}
    for key in ("hosted_alpha", "hosted_beta"):
        row = deepcopy(template)
        row.update(candidate_id=key, display_name=key)
        row["interface"]["policy_interface_id"] = "openpi_droid_https.v1"
        row["policy"].update(kind="hosted", source="https://policy.example.com", revision="release-1",
                             digest_kind="provider_metadata")
        row["eligibility"]["requires_credentials"] = ["HOSTED_DROID_CREDENTIAL"]
        rows.append(row)
        spec = deepcopy(native_template)
        spec.update(candidate_id=key, policy_interface_id="openpi_droid_https.v1",
                    policy_endpoint={"credential_env": "BLUEPRINT_BETA_POLICY_TOKEN_FILE"})
        spec["policy_spec"] = {"policy_id": key, "endpoint_url": "https://policy.example.com/infer",
            "metadata_url": "https://policy.example.com/metadata", "allowed_origin": "https://policy.example.com",
            "model_identity_digest": spec["checkpoint_digest"], "action_chunk_rows": 15, "open_loop_horizon": 8}
        spec["runtime_identity"] = {"policy_interface_id": spec["policy_interface_id"], "deployment": "release-1",
                                    "model_identity_digest": spec["checkpoint_digest"], "policy_id": key}
        spec["runtime_identity_digest"] = canonical_digest(spec["runtime_identity"])
        spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
        path = launch._write(tmp_path / (key + ".json"), spec)
        specs[key] = spec
        settings[key] = {"policy_execution_spec": launch._record(path), "checkpoint_inventory": None,
            "credential_file": str(private), "credential_origin": "https://policy.example.com",
            "provider_hard_cap_usd": "4.00", "hosted_inference_maximum_usd": "1.00",
            "hosted_inference_payment": "blueprint", "hosted_max_cost_per_request_usd": "0.00001"}
    catalog = seal_catalog({**load_default_catalog(), "candidates": rows})
    profile["screening_candidates"] = settings
    planning = build_planning_inputs(catalog=catalog, profile=profile, capture_root=tmp_path / "capture",
                                    source=base_plan["source"], inputs_dir=tmp_path / "hosted-inputs")
    plan = build_screening_plan(catalog, planning["scene"], capabilities=planning["capabilities"],
                               cost_quotes=planning["cost_quotes"], campaign_id="hosted-campaign")
    return plan, specs, settings


def test_two_arbitrary_hosted_candidates_have_inclusive_caps_and_private_sealed_bundles(tmp_path, monkeypatch):
    from blueprint_pipeline.policy_canary_session_bundle_identity import validate_policy_canary_provider_manifest
    from blueprint_pipeline.vast_provider_adapter import _blueprint_bundle_preflight
    plan, specs, settings = hosted_plan(tmp_path, monkeypatch)
    assert {row["status"] for row in plan["candidates"]} == {"selected"}
    assert plan["maximum_planned_cost_usd"] == "10.00"
    for candidate in plan["candidates"]:
        descriptor = candidate["execution_spec"]
        assert descriptor["hosted_http_request_limit"] == request_limit(specs[candidate["candidate_id"]], 10)
        assert candidate["max_cost_usd"] == "5.00" and descriptor["provider_hard_cap_usd"] == "4.00"
    candidate = plan["candidates"][0]
    prepared = launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=tmp_path / "attempt")
    authority = launch._read(Path(prepared["authority"]["path"]))
    assert authority["hard_cap_usd"] == 4.0 and authority["candidate_ids"] == [candidate["candidate_id"]]
    assert authority["run_kind"] == "blueprint_beta_screening" and "team_candidate_admission" not in authority
    assert "--beta-screening-credential-file" in prepared["argv"]
    assert "--team-candidate-credential-file" not in prepared["argv"]
    receipt = launch._read(Path(prepared["bundle_receipt"]["path"]))
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        manifest = validate_policy_canary_provider_manifest(json.loads(archive.read(
            "provider_runtime/adp_arena_provider_manifest.json")))
        assert manifest["team_credential_env"] == "BLUEPRINT_BETA_POLICY_TOKEN_FILE"
        assert manifest["secret_material_bundled"] is False
        assert not any("policy_provisioning" in name for name in archive.namelist())
        assert all(TOKEN.encode() not in archive.read(name) for name in archive.namelist())
    preflight = _blueprint_bundle_preflight(job_dir=tmp_path / "transport-preflight", generated_at="fixed",
        enable_blueprint_bundle=True, enable_isaac_smoke=True,
        provider_bundle_kind="native_task_arena_policy_canary_session", bundle_path=Path(receipt["bundle_path"]),
        provider_bundle_url="https://storage.example.com/bundle.zip",
        provider_output_put_url="https://storage.example.com/output.zip")
    assert preflight["status"] == "passed", preflight.get("blockers")
    # Exercise the actual canonical allocator dispatch and private transport.
    # Only provider allocation is replaced; no alternate launcher is accepted.
    from blueprint_pipeline import paid_resource_allocator as allocator
    from blueprint_pipeline import policy_canary_allocator_lane as lane
    from blueprint_pipeline import team_candidate_quick10_vast as transport
    observed = []
    monkeypatch.setattr(allocator, "_control_plane_checkout_blockers",
                        lambda: ([], {"orchestrator_source_commit": candidate["execution_spec"]["implementation_commit"]}))
    monkeypatch.setattr(lane, "_launch_environment_blockers", lambda *args: [])
    monkeypatch.delenv(lane.REQUIRE_INTEGRATION_CANARY_ENV, raising=False)
    monkeypatch.setattr(lane, "run_native_task_arena_policy_canary_session_vast",
                        lambda **kwargs: pytest.fail("private policy must use private transport"))
    monkeypatch.setattr(transport, "run_arena_native_control_vast",
                        lambda **kwargs: observed.append(kwargs) or {"status": "dry_run_ready",
                                                                     "provider_mutations_performed": 0})
    assert allocator.main([arg for arg in prepared["argv"] if arg != "--execute"]) == 0
    assert observed[0]["execute"] is False and observed[0]["hard_cap_usd"] == 4.0
    assert observed[0]["runtime_secret_file_paths"] == {
        "BLUEPRINT_BETA_POLICY_TOKEN_FILE": Path(candidate["execution_spec"]["credential_file"])}
    assert observed[0]["forward_hf_token"] is False
    admission = launch._read(tmp_path / "attempt" / "paid_admission.json")
    assert admission["allocation_binding"]["run_kind"] == "blueprint_beta_screening"
    assert admission["allocation_binding"]["candidate_ids"] == [candidate["candidate_id"]]
    assert TOKEN not in json.dumps(admission)
    assert not (tmp_path / "attempt" / "allocator" / "policy_canary_session_consumption.json").exists()


@pytest.mark.parametrize("change", ["origin", "public_secret", "low_cost", "request_bound", "unknown_price", "url_secret"])
def test_hosted_delivery_refuses_unbound_secret_or_api_cost_before_spend(tmp_path, monkeypatch, change):
    plan, specs, _ = hosted_plan(tmp_path, monkeypatch)
    candidate = plan["candidates"][0]
    descriptor = dict(candidate["execution_spec"])
    if change == "origin":
        descriptor["credential_origin"] = "https://different.example.com"
    elif change == "public_secret":
        Path(descriptor["credential_file"]).chmod(0o644)
    elif change == "low_cost":
        descriptor["hosted_inference_maximum_usd"] = "0.00"
    elif change == "request_bound":
        descriptor["hosted_http_request_limit"] -= 1
    elif change == "unknown_price":
        descriptor["hosted_max_cost_per_request_usd"] = "0"
    else:
        specs[candidate["candidate_id"]]["policy_spec"]["endpoint_url"] += "?api_key=must-not-retain"
    with pytest.raises(ValueError, match="beta_screening_hosted"):
        validate_hosted_delivery(specs[candidate["candidate_id"]], descriptor, episodes=10)


def test_code_release_mismatch_refuses_before_build(tmp_path, monkeypatch):
    plan, _, _ = hosted_plan(tmp_path, monkeypatch)
    candidate = plan["candidates"][0]
    candidate["execution_spec"]["implementation_commit"] = "f" * 40
    candidate["admission_digest"] = canonical_digest(candidate["execution_spec"])
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    with pytest.raises(ValueError, match="implementation_commit_changed"):
        launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=tmp_path / "attempt")
    assert not (tmp_path / "attempt").exists()


def test_partial_json_write_never_publishes_a_launch_intent(tmp_path, monkeypatch):
    def interrupted(_value, stream, **kwargs):
        stream.write('{"incomplete":')
        raise OSError("simulated power loss while retaining metadata")
    monkeypatch.setattr(launch.json, "dump", interrupted)
    path = tmp_path / "allocator_launch_intent.json"
    with pytest.raises(OSError):
        launch._write(path, {"prepared_digest": "sha256:" + "a" * 64})
    assert not path.exists() and list(tmp_path.iterdir()) == []
