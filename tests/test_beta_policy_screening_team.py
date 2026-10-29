"""Reusable robots reach real sealed beta bundles; each candidate keeps its outcome."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import zipfile

import pytest

from blueprint_pipeline import beta_policy_screening_launch as launch
from blueprint_pipeline.beta_policy_candidate_catalog import CATALOG_ENV, load_default_catalog, seal_catalog
from blueprint_pipeline.beta_policy_screening_capabilities import build_planning_inputs
from blueprint_pipeline.beta_policy_screening_plan import build_screening_plan
from blueprint_pipeline.beta_policy_screening_team import build_registered_team_spec, team_policy_interface
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.team_candidate_quick10_registry import profile_for_capture, scene_facts
from blueprint_pipeline.team_embodiment import validate_embodiment_manifest
from tests.test_beta_policy_screening_launch import _plan
from tests.test_team_embodiment import manifest, write_robot_usd
from tests.test_deployment_embodiment_runtime_configuration import configured as configured


def team_plan(tmp_path, monkeypatch, *, count=1, invalid=False):
    base, facts = _plan(tmp_path, monkeypatch, scene_half="team_scene")
    profile = profile_for_capture(str(tmp_path / "capture"))
    model = write_robot_usd(tmp_path / "robot.usda")
    declared = validate_embodiment_manifest(manifest(model, control_hz=60))
    manifest_path = launch._write(tmp_path / "robot.manifest.json", declared)
    credential = tmp_path / "operator" / "team-bearer"
    credential.write_text("private-team-token-never-in-bundle")
    credential.chmod(0o600)
    registrations, rows = {}, []
    for index in range(count):
        key = f"team_policy_{index:02d}"
        rights = {"schema_version": "blueprint.team_policy_candidate_rights.v1", "status": "admitted",
            "team_id": "test-team", "checkpoint_id": key, "candidate_id": key, "license_to_run_granted": True,
            "policy_self_grading_permitted": False, "official_ranking_permitted": False,
            "secret_material_recorded": False, "adp_qualified_evidence_permitted": False}
        rights["rights_digest"] = canonical_digest(rights, digest_field="rights_digest")
        registration = {"manifest": launch._record(manifest_path), "model": launch._record(model),
            "team_id": "test-team", "checkpoint_id": key, "rights_binding": rights,
            "policy_spec": {"policy_id": key, "policy_interface_id": "blueprint_policy_https.v1",
                "endpoint_url": "https://policy.example.com/infer", "metadata_url": "https://policy.example.com/metadata",
                "allowed_origin": "https://policy.example.com", "model_identity_digest": canonical_digest({"policy": key}),
                "embodiment_id": declared["embodiment_id"], "action_chunk_rows": 4, "open_loop_horizon": 4,
                "camera_names": [camera["name"] for camera in declared["cameras"]]}}
        spec, _ = build_registered_team_spec(candidate_id=key, registration=registration,
            scene_plan=facts["scene"]["native_scene_plan"], contract=facts["scene"]["task_spec"]["task_success_contract"],
            prompt=scene_facts(profile)["prompt"], scene_revision_digest=facts["scene"]["scene_revision_digest"])
        row = deepcopy(load_default_catalog()["candidates"][0])
        row.update(candidate_id=key, display_name=key, interface=team_policy_interface(spec))
        row["runtime"]["minimum_gpu_memory_gb"] = 48
        row["policy"].update(kind="hosted", source="https://policy.example.com", revision="test-release-1",
                             digest=spec["checkpoint_digest"], digest_kind="provider_metadata")
        row["eligibility"].update(commercial_use="allowed", requires_credentials=["TEAM_POLICY_CREDENTIAL"])
        rows.append(row)
        registrations[key] = {"robot_policy": registration, "credential_file": str(credential),
            "credential_origin": "https://policy.example.com", "provider_hard_cap_usd": "1.00",
            "hosted_inference_maximum_usd": "0.00", "hosted_inference_payment": "caller_paid",
            "hosted_max_cost_per_request_usd": "0"}
    if invalid:
        registrations["team_policy_00"]["robot_policy"]["model"]["sha256"] = "sha256:" + "e" * 64
    catalog = seal_catalog({**load_default_catalog(), "candidates": rows})
    catalog_path = launch._write(tmp_path / "team-catalog.json", catalog)
    monkeypatch.setenv(CATALOG_ENV, str(catalog_path))
    profile.update(screening_candidates=registrations, screening_hard_ttl_seconds=1800,
                   screening_candidate_cap_usd="1.00")
    planning = build_planning_inputs(catalog=catalog, profile=profile, capture_root=tmp_path / "capture",
        source=base["source"], inputs_dir=tmp_path / "team-inputs")
    plan = build_screening_plan(catalog, planning["scene"], capabilities=planning["capabilities"],
        cost_quotes=planning["cost_quotes"], campaign_id="team-screening")
    return plan, planning, registrations


def test_sixteen_compatible_configurations_select_fifteen_and_keep_budget_reason(tmp_path, monkeypatch):
    plan, _, _ = team_plan(tmp_path, monkeypatch, count=16)
    assert len(plan["candidates"]) == 16
    assert sum(row["status"] == "selected" for row in plan["candidates"]) == 15
    assert plan["maximum_planned_cost_usd"] == "15.00"
    assert plan["candidates"][-1]["blockers"] == ["site_screening_budget_insufficient"]
    assert all(row["scenario_cells"] == plan["scenario_cells"] for row in plan["candidates"][:-1])


def test_one_changed_robot_preserves_other_candidates_and_reports_its_reason(tmp_path, monkeypatch):
    plan, _, _ = team_plan(tmp_path, monkeypatch, count=3, invalid=True)
    assert [row["status"] for row in plan["candidates"]] == ["unavailable", "selected", "selected"]
    assert "beta_screening_input_bytes_changed" in plan["candidates"][0]["blockers"]


def test_registered_robot_is_copied_into_sealed_bundle_without_secret(tmp_path, monkeypatch):
    from blueprint_pipeline.policy_canary_session_bundle_identity import validate_policy_canary_provider_manifest

    plan, _, registrations = team_plan(tmp_path, monkeypatch)
    candidate = plan["candidates"][0]
    assert candidate["status"] == "selected", candidate
    prepared = launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=tmp_path / "attempt")
    receipt = launch._read(Path(prepared["bundle_receipt"]["path"]))
    original = Path(registrations[candidate["candidate_id"]]["robot_policy"]["model"]["path"])
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        sealed = validate_policy_canary_provider_manifest(json.loads(archive.read(
            "provider_runtime/adp_arena_provider_manifest.json")))
        assert sealed["team_policy_interface_id"] == "blueprint_policy_https.v1"
        assert sealed["minimum_gpu_memory_gb"] == 48
        assert archive.read("provider_runtime/native_task_packet/team_embodiment/robot.usd") == original.read_bytes()
        assert all(b"private-team-token-never-in-bundle" not in archive.read(name) for name in archive.namelist())
    assert not (tmp_path / "attempt/allocator_launch_intent.json").exists()


def test_candidate_memory_floor_preserves_existing_floor_and_excludes_small_gpu():
    from blueprint_pipeline.policy_canary_session_bundle_identity import minimum_gpu_ram_mb
    from blueprint_pipeline.vast_provider_adapter import _select_offer

    assert minimum_gpu_ram_mb({}) == 46_000
    assert minimum_gpu_ram_mb({"minimum_gpu_memory_gb": 12}) == 46_000
    assert minimum_gpu_ram_mb({"minimum_gpu_memory_gb": 48}) == 48_000
    small = {"id": 1, "gpu_name": "RTX 4090", "gpu_ram": 24_000, "dph_total": 0.4, "machine_id": 11}
    large = {"id": 2, "gpu_name": "RTX A6000", "gpu_ram": 48_000, "dph_total": 0.8, "machine_id": 12}
    floor = minimum_gpu_ram_mb({"minimum_gpu_memory_gb": 48})
    assert _select_offer([small], max_hourly_rate=1, min_gpu_ram_mb=floor) is None
    assert _select_offer([small, large], max_hourly_rate=1, min_gpu_ram_mb=floor)["ask_contract_id"] == 2
    for value in (True, 0, -1, float("nan"), float("inf"), "48"):
        with pytest.raises(ValueError, match="gpu_memory_bound_invalid"):
            minimum_gpu_ram_mb({"minimum_gpu_memory_gb": value})


def test_registered_catalog_updates_when_credential_delivery_is_completed(tmp_path, monkeypatch):
    from blueprint_pipeline.beta_policy_screening_team import catalog_for_profile
    _, _, registrations = team_plan(tmp_path, monkeypatch)
    base = load_default_catalog()
    profile = {"screening_candidates": registrations}
    ready = catalog_for_profile(profile, base)
    row = next(row for row in ready["candidates"] if row["candidate_id"] == "team_policy_00")
    assert row["eligibility"]["enabled"]
    credential = Path(registrations["team_policy_00"]["credential_file"])
    credential.chmod(0o644)
    unavailable = catalog_for_profile(profile, base)
    assert unavailable["catalog_digest"] != ready["catalog_digest"]
    assert unavailable["candidates"][0]["eligibility"]["blockers"] == ["beta_screening_hosted_credential_not_private"]
    credential.chmod(0o600)
    assert catalog_for_profile(profile, base) == ready
    replacement = credential.with_name("rotated-private-delivery")
    replacement.write_text("replacement-private-token")
    replacement.chmod(0o600)
    registrations["team_policy_00"]["credential_file"] = str(replacement)
    rotated = catalog_for_profile(profile, base)
    assert rotated["catalog_digest"] != ready["catalog_digest"]
    assert rotated["candidates"][0]["eligibility"]["enabled"]
    assert "rotated-private-delivery" not in json.dumps(rotated)
    assert "private-team-token-never-in-bundle" not in json.dumps(ready)


@pytest.mark.parametrize("change", ["candidate", "checkpoint", "digest", "permission"])
def test_unrelated_or_unsealed_rights_cannot_enable_registered_policy(tmp_path, monkeypatch, change):
    from blueprint_pipeline.beta_policy_screening_team import catalog_for_profile
    _, _, registrations = team_plan(tmp_path, monkeypatch)
    registration = registrations["team_policy_00"]["robot_policy"]
    rights = registration["rights_binding"]
    if change == "candidate":
        rights["candidate_id"] = "another-policy"
    elif change == "checkpoint":
        rights["checkpoint_id"] = "another-checkpoint"
    elif change == "permission":
        rights["license_to_run_granted"] = False
    else:
        rights["rights_digest"] = "sha256:" + "e" * 64
    if change != "digest":
        rights["rights_digest"] = canonical_digest(rights, digest_field="rights_digest")
    effective = catalog_for_profile({"screening_candidates": registrations}, load_default_catalog())
    assert effective["candidates"][0]["eligibility"]["enabled"] is False
    assert effective["candidates"][0]["eligibility"]["blockers"][0].startswith("team_candidate_rights_")


@pytest.mark.parametrize("field,value", [("minimum_gpu_memory_gb", "48"), ("display_name", ""), ("policy_revision", [])])
def test_invalid_registration_metadata_does_not_discard_valid_candidates(tmp_path, monkeypatch, field, value):
    from blueprint_pipeline.beta_policy_screening_team import catalog_for_profile
    _, _, registrations = team_plan(tmp_path, monkeypatch, count=3)
    registrations["team_policy_00"]["robot_policy"][field] = value
    # An empty optional revision uses the served identity; use a nonempty bad value.
    if field == "policy_revision":
        registrations["team_policy_00"]["robot_policy"][field] = ["invalid"]
    effective = catalog_for_profile({"screening_candidates": registrations}, load_default_catalog())
    assert [row["eligibility"]["enabled"] for row in effective["candidates"]] == [False, True, True]
    assert effective["candidates"][0]["eligibility"]["blockers"] == ["beta_catalog_schema_invalid"]


def test_dream_registration_uses_real_scene_revision_and_reaches_sealed_bundle(tmp_path, monkeypatch):
    from dataclasses import asdict
    from blueprint_pipeline.beta_policy_screening_team import catalog_for_profile
    from blueprint_pipeline.dreamlabs_policy_client import (
        DreamLabsPolicySpec, ENDPOINT_URL, METADATA_URL, GATEWAY_ORIGIN, provider_metadata_digest,
    )
    from tests.test_dreamlabs_policy_client import MODEL
    from tests.test_dreamlabs_team_candidate_admission import dream_world

    world = dream_world(tmp_path, monkeypatch)
    payload = world["request"]["policy_package"]["policy_api_endpoint"]
    manifest_value = validate_embodiment_manifest(payload["robot_embodiment"])
    model = next(tmp_path.glob("*.usda"))
    manifest_path = launch._write(tmp_path / "dream-manifest.json", manifest_value)
    rights = launch._read(world["operator"] / "rights.json")
    key = rights["candidate_id"]
    policy = DreamLabsPolicySpec(policy_id="fixture-dream", endpoint_url=ENDPOINT_URL, metadata_url=METADATA_URL,
        allowed_origin=GATEWAY_ORIGIN, model_identity_digest=provider_metadata_digest(MODEL),
        embodiment_id=manifest_value["embodiment_id"], action_chunk_rows=48, open_loop_horizon=30,
        camera_names=[row["name"] for row in manifest_value["cameras"]], **payload["policy_config"])
    registration = {"manifest": launch._record(manifest_path), "model": launch._record(model),
        "team_id": rights["team_id"], "checkpoint_id": rights["checkpoint_id"], "rights_binding": rights,
        "policy_spec": asdict(policy), "provider_model_metadata": MODEL,
        "provider_disclosure": rights["provider_disclosure"]}
    credential = world["operator"] / "dream-bearer"
    credential.write_text("fixture-dream-private-bearer")
    credential.chmod(0o600)
    registry_path = world["operator"] / "registry.json"
    registry = launch._read(registry_path)
    registry["profiles"][0].update(screening_candidates={key: {"robot_policy": registration,
        "credential_file": str(credential), "credential_origin": GATEWAY_ORIGIN, "provider_hard_cap_usd": "1.00",
        "hosted_inference_maximum_usd": "0.00", "hosted_inference_payment": "caller_paid",
        "hosted_max_cost_per_request_usd": "0"}}, screening_hard_ttl_seconds=1800, screening_candidate_cap_usd="1.00")
    registry["registry_digest"] = canonical_digest(registry, digest_field="registry_digest")
    registry_path.write_text(json.dumps(registry))
    profile = profile_for_capture(world["capture"])
    facts = scene_facts(profile)
    source = {"capture_id": "capture1", "task_id": profile["task_id"], "scene_intent_id": "intent1",
              "scene_intent_digest": "sha256:" + "a" * 64, "task_context_digest": "sha256:" + "b" * 64,
              "scene_revision_digest": facts["inputs"]["scene_revision_digest"]}
    catalog = catalog_for_profile(profile)
    planning = build_planning_inputs(catalog=catalog, profile=profile, capture_root=world["capture"],
        source=source, inputs_dir=tmp_path / "dream-planning")
    plan = build_screening_plan(catalog, planning["scene"], campaign_id="dream-screening",
        capabilities=planning["capabilities"], cost_quotes=planning["cost_quotes"])
    candidate = next(row for row in plan["candidates"] if row["candidate_id"] == key)
    assert candidate["status"] == "selected", candidate
    prepared = launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=tmp_path / "dream-attempt")
    receipt = launch._read(Path(prepared["bundle_receipt"]["path"]))
    assert receipt["team_policy_interface_id"] == "dreamlabs_dream1.v1"
    spec = launch._read(Path(candidate["execution_spec"]["policy_execution_spec"]["path"]))
    assert spec["provider_disclosure"]["scene_revision_digest"] == source["scene_revision_digest"]
    assert spec["checkpoint_identity_basis"] == "provider_metadata" and spec["weights_identity_verified"] is False
    assert world["dream_service"].calls == [] and world["allocator_calls"] == []


def test_source_import_output_registers_without_reconstructing_manifest_by_hand(configured, tmp_path, monkeypatch):
    from blueprint_pipeline import deployment_embodiment_asset_import as importer
    from blueprint_pipeline import deployment_embodiment_runtime_configuration as runtime
    from blueprint_pipeline.beta_policy_screening_team import registration_from_prepared_inputs
    from tests.test_deployment_embodiment_runtime_configuration import _fake_sdk_from_urdf

    case = configured()
    _fake_sdk_from_urdf(monkeypatch)  # Real USD bytes; this test never claims actual Isaac import.
    imported = tmp_path / "imported"
    receipt = importer.import_source_asset(case.output, imported, fix_base=True,
                                          station_parameters=case.record["station_parameters"])
    inputs = runtime.bind_prepared_configuration(case.output, imported, "https://assets.example/fixture.usda")
    rights = {"schema_version": "blueprint.team_policy_candidate_rights.v1", "status": "admitted",
        "team_id": "test-team", "checkpoint_id": "test-checkpoint", "candidate_id": "test-policy",
        "license_to_run_granted": True, "policy_self_grading_permitted": False,
        "official_ranking_permitted": False, "secret_material_recorded": False, "adp_qualified_evidence_permitted": False}
    rights["rights_digest"] = canonical_digest(rights, digest_field="rights_digest")
    registration = registration_from_prepared_inputs(prepared=inputs, model_path=imported / receipt["model"]["path"],
        destination=tmp_path / "registered", candidate_id="test-policy", team_id="test-team",
        checkpoint_id="test-checkpoint", policy_spec={}, rights=rights, display_name="Fixture only")
    assert launch._read(launch._bound_path(registration["manifest"])) == inputs["robot_embodiment"]
    assert registration["source_import_provenance"] == receipt
    assert registration["dependency_closure"] == receipt["dependency_closure"]
    assert registration["model"]["sha256"] == receipt["model"]["sha256"]
