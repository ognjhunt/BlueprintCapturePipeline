"""A policy we run ourselves enters screening as one registry entry and runs over the standard contract.

Plan 2.1 (``docs/launch-readiness/2026-11-11-evaluation-service-plan.md``). Hermetic: the real
planner, capabilities, launch preparation, bundle, allocator dispatch (provider replaced), coordinator,
settlement and publication; no GPU, no provider, no spend.
"""
from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import zipfile

import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import beta_policy_screening_launch as launch
from blueprint_pipeline.beta_policy_effective_catalog import catalog_for_profile
from blueprint_pipeline.beta_policy_candidate_catalog import load_default_catalog
from blueprint_pipeline.beta_policy_screening_capabilities import build_planning_inputs
from blueprint_pipeline.beta_policy_screening_plan import build_screening_plan
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.session_served_policy import ENTRY_SCHEMA, TOKEN_ENV
from blueprint_pipeline.session_served_policy_catalog import catalog_for_profile as served_catalog
from blueprint_pipeline.team_candidate_quick10_registry import profile_for_capture
from tests.test_beta_policy_screening_coordinator import Backend, native_result, publication
from tests.test_beta_policy_screening_launch import _plan

CANDIDATE = "pi05_droid_contract"
ENTRY = {"schema_version": ENTRY_SCHEMA, "server": "openpi_droid_contract_server.v1",
         "native_candidate_id": "pi05_droid", "display_name": "π0.5 DROID (standard connection)"}


def served_plan(tmp_path, monkeypatch, entry=ENTRY):
    base_plan, _facts = _plan(tmp_path, monkeypatch, scene_half="team_scene")
    profile = profile_for_capture(str(tmp_path / "capture"))
    profile["screening_candidates"] = {CANDIDATE: {"session_served_policy": dict(entry)}}
    catalog = catalog_for_profile(profile)
    planning = build_planning_inputs(catalog=catalog, profile=profile, capture_root=tmp_path / "capture",
                                     source=base_plan["source"], inputs_dir=tmp_path / "served-inputs")
    plan = build_screening_plan(catalog, planning["scene"], capabilities=planning["capabilities"],
                                cost_quotes=planning["cost_quotes"], campaign_id="served-campaign",
                                candidate_ids=[CANDIDATE])
    return plan, planning, catalog


def test_registry_entry_is_a_catalog_row_with_the_native_identity_over_the_contract():
    base = load_default_catalog()
    catalog = served_catalog({"screening_candidates": {CANDIDATE: {"session_served_policy": ENTRY}}}, base)
    row = next(item for item in catalog["candidates"] if item["candidate_id"] == CANDIDATE)
    native = next(item for item in base["candidates"] if item["candidate_id"] == "pi05_droid")
    assert row["policy"] == native["policy"] and row["eligibility"]["license_id"] == native["eligibility"]["license_id"]
    assert row["interface"] == {**native["interface"], "policy_interface_id": "openpi_droid_https.v1"}
    assert row["runtime"] == native["runtime"] and row["eligibility"]["requires_credentials"] == []
    # Every packaged row is unchanged; only the registry entry is new.
    assert [item for item in catalog["candidates"] if item["candidate_id"] != CANDIDATE] == base["candidates"]
    assert served_catalog({}, base) == base


@pytest.mark.parametrize("change", ["native_id", "unknown_server", "port", "takes_packaged_id"])
def test_an_entry_that_cannot_be_derived_is_listed_with_its_reason_and_never_enabled(change):
    base = load_default_catalog()
    entry, key = dict(ENTRY), CANDIDATE
    if change == "native_id":
        entry["native_candidate_id"] = "groot_n17_droid"
    elif change == "unknown_server":
        entry["server"] = "some_other_server.v9"
    elif change == "port":
        entry["port"] = 8000
    else:
        key = "molmoact2_droid"
    catalog = served_catalog({"screening_candidates": {key: {"session_served_policy": entry}}}, base)
    rows = {item["candidate_id"]: item for item in catalog["candidates"]}
    if change == "takes_packaged_id":
        assert rows[key] == next(item for item in base["candidates"] if item["candidate_id"] == key)
    else:
        assert rows[key]["eligibility"]["enabled"] is False and rows[key]["eligibility"]["blockers"]


def test_served_candidate_is_planned_with_the_gpu_cap_only_and_bundled_without_operator_secret(tmp_path, monkeypatch):
    from blueprint_pipeline.policy_canary_session_bundle_identity import (
        run_kind_entrypoint_blockers, run_kind_required_entries, validate_policy_canary_provider_manifest,
    )
    from blueprint_pipeline.vast_provider_adapter import _blueprint_bundle_preflight

    plan, planning, _ = served_plan(tmp_path, monkeypatch)
    candidate = next(row for row in plan["candidates"] if row["candidate_id"] == CANDIDATE)
    assert candidate["status"] == "selected", candidate.get("blockers")
    descriptor = candidate["execution_spec"]
    # No hosted fields, no credential: a native candidate's money and nothing else.
    assert not {"credential_file", "hosted_inference_maximum_usd", "provider_hard_cap_usd"} & set(descriptor)
    assert candidate["max_cost_usd"] == "4.00"
    assert planning["capabilities"]["credentials_available"] == []
    spec = launch._read(Path(descriptor["policy_execution_spec"]["path"]))
    native = launch._read(Path(planning["capabilities"]["candidate_bindings"]["pi05_droid"]["execution_spec"]
                               ["policy_execution_spec"]["path"]))
    assert spec["session_policy_server"]["native_policy_spec"] == native["policy_spec"]
    assert spec["checkpoint_digest"] == native["checkpoint_digest"]
    assert spec["policy_spec"]["endpoint_url"] == "http://127.0.0.1:8765/v1/infer"
    assert descriptor["checkpoint_inventory"] == planning["capabilities"]["candidate_bindings"]["pi05_droid"][
        "execution_spec"]["checkpoint_inventory"]
    prepared = launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=tmp_path / "attempt")
    assert "--beta-screening-credential-file" not in prepared["argv"]
    assert "--adp-max-spend-usd" in prepared["argv"]
    assert prepared["argv"][prepared["argv"].index("--adp-max-spend-usd") + 1] == "4.00"
    receipt = launch._read(Path(prepared["bundle_receipt"]["path"]))
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = set(archive.namelist())
        manifest = validate_policy_canary_provider_manifest(json.loads(archive.read(
            "provider_runtime/adp_arena_provider_manifest.json")))
        assert manifest["team_policy_interface_id"] == "openpi_droid_https.v1"
        assert "team_credential_env" not in manifest and manifest["secret_material_bundled"] is False
        assert manifest["session_policy_server"]["token_origin"] == "generated_on_session"
        assert run_kind_required_entries(manifest) <= names
        entrypoint = archive.read("provider_runtime/run_adp_arena_provider_runtime.sh").decode()
        assert run_kind_entrypoint_blockers(manifest, entrypoint) == []
        assert f'export {TOKEN_ENV}="$SESSION_POLICY_DIR/token"' in entrypoint
        assert f"adp009d_policy_provisioning.{CANDIDATE}.sh" in entrypoint
        assert "--terminate-ready-server" in entrypoint
        script = archive.read(f"provider_runtime/adp009d_policy_provisioning.{CANDIDATE}.sh").decode()
        # The native venv, source and checkpoint; the contract server instead of the websocket one.
        assert "/opt/adp009d-policy-venv/pi05_droid" in script and "/opt/adp009d-checkpoints/pi05_droid" in script
        assert "blueprint_pipeline.session_policy_server_start" in script
        assert "adp009d_policy_server_worker.py" not in script
    preflight = _blueprint_bundle_preflight(job_dir=tmp_path / "transport-preflight", generated_at="fixed",
        enable_blueprint_bundle=True, enable_isaac_smoke=True,
        provider_bundle_kind="native_task_arena_policy_canary_session", bundle_path=Path(receipt["bundle_path"]),
        provider_bundle_url="https://storage.example.com/bundle.zip",
        provider_output_put_url="https://storage.example.com/output.zip")
    assert preflight["status"] == "passed", preflight.get("blockers")
    # The canonical allocator dispatches it on the ordinary lane, with no private transport.
    from blueprint_pipeline import paid_resource_allocator as allocator
    from blueprint_pipeline import policy_canary_allocator_lane as lane
    from blueprint_pipeline import team_candidate_quick10_vast as transport
    observed = []
    monkeypatch.setattr(allocator, "_control_plane_checkout_blockers",
                        lambda: ([], {"orchestrator_source_commit": descriptor["implementation_commit"]}))
    monkeypatch.setattr(lane, "_launch_environment_blockers", lambda *args: [])
    monkeypatch.delenv(lane.REQUIRE_INTEGRATION_CANARY_ENV, raising=False)
    monkeypatch.setattr(transport, "run_arena_native_control_vast",
                        lambda **kwargs: pytest.fail("a session-served policy has no private credential transport"))
    monkeypatch.setattr(lane, "run_native_task_arena_policy_canary_session_vast",
                        lambda **kwargs: observed.append(kwargs) or {"status": "dry_run_ready",
                                                                     "provider_mutations_performed": 0})
    assert allocator.main([arg for arg in prepared["argv"] if arg != "--execute"]) == 0
    assert observed and observed[0]["execute"] is False


def test_served_candidate_settles_and_publishes_through_the_real_coordinator(tmp_path, monkeypatch):
    plan, _, catalog = served_plan(tmp_path, monkeypatch)
    plan["candidates"] = [row for row in plan["candidates"] if row["candidate_id"] == CANDIDATE]
    plan["maximum_planned_cost_usd"] = plan["candidates"][0]["max_cost_usd"]
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    queue = tmp_path / "queue"
    coordinator.enqueue_screening_campaign(plan, queue_root=queue)
    result = native_result(plan, plan["candidates"][0])
    backend = Backend({"status": "completed", "provider_allocation_performed": True,
                       "provider_zero_verified": True, "result": result})
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=queue, executor=backend)
    published = publication(queue, plan["campaign_id"])
    row = published["candidates"][0]
    assert backend.launches == 1 and published["status"] == "completed"
    assert row["candidate_id"] == CANDIDATE and row["episodes_scored"] == 10 and row["episodes_succeeded"] == 7
    # Settled at the GPU cap, exactly like a native candidate: no API fee is added.
    assert Decimal(published["budget"]["spent_upper_bound_usd"]) == Decimal("4")


def test_existing_native_plan_and_publication_are_unchanged_by_the_new_path(tmp_path, monkeypatch):
    """The packaged plan fixture is byte-identical with no registry entry; adding one changes only its own row."""
    plan, facts = _plan(tmp_path / "plain", monkeypatch, scene_half="team_scene")
    plain = {row["candidate_id"]: row for row in plan["candidates"]}
    served, _, _ = served_plan(tmp_path / "served", monkeypatch)
    # Every native row's identity (configuration, admission, runtime) is what it was without the entry.
    for key in ("pi05_droid", "groot_n17_droid", "molmoact2_droid"):
        before = plain[key]
        assert before["configuration_digest"] == next(row for row in load_default_catalog()["candidates"]
                                                      if row["candidate_id"] == key)["configuration_digest"]
        assert before["execution_spec"]["runtime_digest"] == facts["capabilities"]["candidate_bindings"][key]["runtime_digest"]
    assert plan["catalog_digest"] == load_default_catalog()["catalog_digest"]
    assert {row["candidate_id"] for row in served["candidates"]} == {CANDIDATE}
