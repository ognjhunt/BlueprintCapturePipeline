"""Real beta bundles and crash-safe canonical allocator starts, without spend."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import zipfile

import pytest

from blueprint_pipeline import beta_policy_screening_launch as launch
from blueprint_pipeline.beta_policy_candidate_catalog import load_default_catalog
from blueprint_pipeline.beta_policy_screening_capabilities import build_planning_inputs
from blueprint_pipeline.beta_policy_screening_plan import build_screening_plan
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_policy_canary_session import (
    BETA_RUN_KIND, PolicyCanarySessionError, validate_runtime_input_manifest, validate_session_authority,
)
from blueprint_pipeline.policy_canary_session_bundle_identity import validate_policy_canary_provider_manifest
from blueprint_pipeline.team_candidate_quick10_registry import profile_for_capture
from tests.test_team_candidate_quick10_launch import _world


def _plan(tmp_path, monkeypatch, *, catalog_override=None, scene_half="frozen_pair"):
    world = _world(tmp_path, monkeypatch, derived=scene_half == "team_scene", scene_half=scene_half)
    profile = profile_for_capture(str(world["capture"]))
    source = {"capture_id": "capture1", "task_id": profile["task_id"], "scene_intent_id": "intent1",
              "scene_intent_digest": "sha256:" + "a" * 64, "task_context_digest": "sha256:" + "b" * 64,
              "scene_revision_digest": world["probe_receipt"]["configured_scene_revision_digest"]}
    catalog = catalog_override or load_default_catalog()
    facts = build_planning_inputs(catalog=catalog, profile=profile, capture_root=world["capture"],
                                 source=source, inputs_dir=tmp_path / "screening-inputs")
    plan = build_screening_plan(catalog, facts["scene"], capabilities=facts["capabilities"],
                                cost_quotes=facts["cost_quotes"], campaign_id="campaign1")
    return plan, facts


@pytest.mark.parametrize("scene_half", ["frozen_pair", "team_scene"])
def test_real_single_candidate_robolab_bundle_passes_both_pre_create_gates(tmp_path, monkeypatch, scene_half):
    from blueprint_pipeline.vast_provider_adapter import _blueprint_bundle_preflight

    plan, facts = _plan(tmp_path, monkeypatch, scene_half=scene_half)
    selected = [row for row in plan["candidates"] if row["status"] == "selected"]
    assert {row["candidate_id"] for row in selected} == {"pi05_droid", "groot_n17_droid", "molmoact2_droid"}
    assert plan["maximum_planned_cost_usd"] == "12.00"
    candidate = next(row for row in selected if row["candidate_id"] == "pi05_droid")
    prepared = launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=tmp_path / "attempt")
    authority = validate_session_authority(launch._read(Path(prepared["authority"]["path"])))
    assert authority["run_kind"] == BETA_RUN_KIND
    assert authority["candidate_ids"] == ["pi05_droid"] and authority["learned_policy_rollout_count"] == 10
    assert authority["beta_screening_admission"]["plan_digest"] == plan["plan_digest"]
    receipt = launch._read(Path(prepared["bundle_receipt"]["path"]))
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        manifest = validate_policy_canary_provider_manifest(json.loads(archive.read(
            "provider_runtime/adp_arena_provider_manifest.json")))
        assert manifest["evaluation_backend"] == "robolab.v1"
        assert "provider_runtime/robolab/eval/episode.py" in archive.namelist()
        assert "provider_runtime/adp009d_policy_provisioning.groot_n17_droid.sh" not in archive.namelist()
    preflight = _blueprint_bundle_preflight(job_dir=tmp_path / "transport-preflight", generated_at="fixed",
        enable_blueprint_bundle=True, enable_isaac_smoke=True,
        provider_bundle_kind="native_task_arena_policy_canary_session", bundle_path=Path(receipt["bundle_path"]),
        provider_bundle_url="https://storage.example.com/bundle.zip",
        provider_output_put_url="https://storage.example.com/output.zip")
    assert preflight["status"] == "passed", preflight.get("blockers")
    assert launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=tmp_path / "attempt") == prepared
    assert not (tmp_path / "attempt" / "allocator_launch_intent.json").exists()


def test_launch_intent_crash_is_not_retried_or_released(tmp_path, monkeypatch):
    from blueprint_pipeline import team_candidate_quick10_launch as native
    from blueprint_pipeline import beta_policy_screening_worker as service

    root = tmp_path / "attempt"
    root.mkdir()
    monkeypatch.setattr(launch, "_validate_prepared", lambda *args: None)
    monkeypatch.setattr(launch, "_check_quote", lambda *args: None)
    monkeypatch.setattr(service, "require_current_source", lambda *args, **kwargs: {})
    calls = []
    def crash(argv, **kwargs):
        calls.append(argv)
        raise RuntimeError("process may have started before parent crashed")
    monkeypatch.setattr(native, "_start_allocator", crash)
    prepared = {"prepared_digest": "sha256:" + "a" * 64, "argv": ["gpu-canary"]}
    kwargs = dict(plan={"source": {}, "site_id": "site-request1"}, candidate={}, attempt_dir=root, prepared=prepared)
    with pytest.raises(RuntimeError):
        launch.launch_candidate(**kwargs)
    result = launch.launch_candidate(**kwargs)
    assert result["status"] == "reconciliation_required" and len(calls) == 1
    assert result["no_allocation_proven"] is False
    assert "provider_allocation_performed" not in result


def test_closed_allocator_refusal_is_the_only_free_release(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "_validate_prepared", lambda *args: None)
    launch._write(tmp_path / "allocator_result.json", {"status": "blocked", "provider_mutations_performed": 0,
                                                       "provider_create_attempted": False})
    result = launch.reconcile_candidate(plan={}, candidate={}, attempt_dir=tmp_path, prepared={})
    assert result["no_allocation_proven"] and result["provider_allocation_performed"] is False


def test_current_source_refusal_never_spawns_and_stays_proven_unallocated(tmp_path, monkeypatch):
    from blueprint_pipeline import team_candidate_quick10_launch as native
    from blueprint_pipeline import beta_policy_screening_worker as service

    monkeypatch.setattr(launch, "_validate_prepared", lambda *args: None)
    monkeypatch.setattr(launch, "_check_quote", lambda *args: None)
    def refuse(*args, **kwargs):
        raise ValueError("current consent withdrawn")
    monkeypatch.setattr(service, "require_current_source", refuse)
    monkeypatch.setattr(native, "_start_allocator", lambda *args, **kwargs: pytest.fail("must not spawn"))
    kwargs = dict(plan={"source": {}, "site_id": "site-request1"}, candidate={}, attempt_dir=tmp_path,
                  prepared={"prepared_digest": "sha256:" + "a" * 64})
    first = launch.launch_candidate(**kwargs)
    assert first["status"] == "blocked" and first["no_allocation_proven"]
    assert first["provider_allocation_performed"] is False
    assert launch.launch_candidate(**kwargs) == first
    assert launch.reconcile_candidate(**kwargs) == first


@pytest.mark.parametrize("field,value", [("control_hz", 30), ("image_height", 225),
    ("robot_profile_id", "another_robot"), ("camera_roles", ["external"]),
    ("normalization_digest", "sha256:" + "d" * 64)])
def test_changed_catalog_interface_does_not_admit_the_same_runtime(tmp_path, monkeypatch, field, value):
    from blueprint_pipeline.beta_policy_candidate_catalog import CATALOG_ENV, seal_catalog
    catalog = load_default_catalog()
    next(row for row in catalog["candidates"] if row["candidate_id"] == "pi05_droid")["interface"][field] = value
    catalog = seal_catalog(catalog)
    override = tmp_path / "operator-catalog.json"
    override.write_text(json.dumps(catalog))
    monkeypatch.setenv(CATALOG_ENV, str(override))
    plan, facts = _plan(tmp_path, monkeypatch, catalog_override=catalog)
    assert "pi05_droid" not in facts["capabilities"]["candidate_bindings"]
    candidate = next(row for row in plan["candidates"] if row["candidate_id"] == "pi05_droid")
    assert candidate["status"] == "unavailable"


def test_derived_profiles_accept_operator_screening_config_but_never_scene_authored_limits(tmp_path):
    from blueprint_pipeline import team_candidate_quick10_registry as registry
    defaults = {"allowed_founder_approvers": [], "max_hourly_rate_usd": 0.8, "open_loop_horizon": 8,
                "screening_candidates": {"custom_droid": {"operator_spec": "sealed.json"}},
                "screening_hard_ttl_seconds": 7200, "screening_candidate_cap_usd": "2.00"}
    settings = {"derived_profiles": {"root": str(tmp_path), "defaults": defaults}}
    assert registry._derived_settings(settings)["defaults"] == defaults
    assert not set(defaults).intersection(registry.DERIVED_FIELDS)
    settings["derived_profiles"]["defaults"]["scene_revision_digest"] = "sha256:" + "a" * 64
    with pytest.raises(registry.TeamCandidateRegistryError, match="derived_profile_settings_invalid"):
        registry._derived_settings(settings)


@pytest.mark.parametrize("interface,candidate_id", [("openpi_droid.v1", "local_alias"), ("groot_n17_droid.v1", "local_alias")])
def test_interface_without_its_actual_delivery_path_is_not_a_capability(interface, candidate_id):
    from blueprint_pipeline.beta_policy_screening_capabilities import _interface_matches
    spec = {"policy_interface_id": interface, "candidate_id": candidate_id}
    assert _interface_matches({}, spec) is False


def test_beta_runtime_requires_its_admission_and_keeps_adp_pair_rule(tmp_path):
    from tests.test_native_task_arena_policy_canary_session import _authority
    authority, inputs = _authority(tmp_path)
    single = deepcopy(inputs)
    single["candidate_ids"] = ["pi05_droid"]
    single["runtime_inputs_digest"] = canonical_digest(single, digest_field="runtime_inputs_digest")
    with pytest.raises(PolicyCanarySessionError, match="candidate_pair_invalid"):
        validate_runtime_input_manifest(single)
    single["run_kind"] = BETA_RUN_KIND
    single["runtime_inputs_digest"] = canonical_digest(single, digest_field="runtime_inputs_digest")
    with pytest.raises(PolicyCanarySessionError, match="beta_admission_binding_invalid"):
        validate_runtime_input_manifest(single)


def test_preparation_refuses_changed_packet_before_allocator(tmp_path, monkeypatch):
    plan, _ = _plan(tmp_path, monkeypatch)
    candidate = next(row for row in plan["candidates"] if row["status"] == "selected")
    scene = launch._read(Path(plan["scene_inputs"]["path"]))
    Path(scene["base_native_packet"]["path"]).write_text("changed\n")
    with pytest.raises(ValueError, match="input_bytes_changed"):
        launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=tmp_path / "attempt")
    assert not (tmp_path / "attempt" / "allocator_launch_intent.json").exists()
