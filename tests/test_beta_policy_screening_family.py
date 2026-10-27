"""Published capture family facts reach bounded native candidate preparation."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import pytest

from blueprint_pipeline import beta_policy_screening_launch as launch
from blueprint_pipeline.beta_policy_candidate_catalog import CATALOG_ENV, load_default_catalog, seal_catalog
from blueprint_pipeline.beta_policy_screening_capabilities import build_planning_inputs
from blueprint_pipeline.beta_policy_screening_plan import build_screening_plan
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_family_native_runtime import GYMNASIUM_FACTORY
from blueprint_pipeline.task_family_policy_execution import family_policy_execution_binding
from blueprint_pipeline.team_candidate_quick10_registry import profile_for_capture, scene_facts
from tests.test_native_task_arena_bundle import _runtime_source_packet
from tests.test_task_evaluation_task_family_native_adapter import _runtime_contract
from tests.test_task_family_native_runtime import _seal_catalog, _sha
from tests.test_task_family_policy_execution import _spec
from tests.test_website_task_family_scene_inputs import source_fixture


def configure_family_candidate(profile, scene_plan, *, root, monkeypatch, registration):
    """Configure one real operator catalog/profile override on existing facts."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    spec = _spec(scene_plan)
    camera = scene_plan["native_runtime_contract"]["cameras"][0]
    spec["policy_spec"].update(camera_id=camera["camera_id"],
        image_width=camera["resolution"][0], image_height=camera["resolution"][1])
    spec.update(execution_authority="internal_policy_canary_unqualified", claim_ceiling="diagnostic_policy_execution",
        ranking_permitted=False, qualification_permitted=False, scene_promotion_permitted=False)
    spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
    spec_path = launch._write(root / "family-execution-spec.json", spec)
    candidate = deepcopy(load_default_catalog()["candidates"][0])
    policy = spec["policy_spec"]
    candidate.update(candidate_id=spec["candidate_id"], display_name="Qualified native button policy",
        supported_task_kinds=["button_press"])
    candidate["interface"] = {"policy_interface_id": "task_family_policy_https.v1",
        "robot_profile_id": policy["embodiment_id"], "controller": policy["controller"],
        "observation_profile_id": policy["observation_schema_id"], "action_schema_id": policy["action_schema_id"],
        "action_bounds": policy["action_bounds"], "control_hz": spec["control_frequency_hz"],
        "action_horizon": policy.get("action_chunk_rows", 1), "execution_horizon": spec["open_loop_horizon"],
        "camera_roles": [policy["camera_id"]], "image_height": policy["image_height"], "image_width": policy["image_width"],
        "image_encoding": "rgb_uint8", "image_preprocessing": "identity", "history_frames": 1,
        "normalization_digest": None}
    candidate["runtime"] = {"backend": "task_family_native.v1", "runtime_id": registration["runtime_id"],
                            "minimum_gpu_memory_gb": 1}
    candidate["policy"].update(kind="hosted", source=policy["allowed_origin"], revision="operator-release-1",
                              digest=spec["checkpoint_digest"], digest_kind="provider_metadata")
    candidate["eligibility"].update(enabled=True, commercial_use="allowed", blockers=[],
                                   requires_credentials=["NATIVE_FAMILY_POLICY_CREDENTIAL"])
    catalog = seal_catalog({**load_default_catalog(), "candidates": [candidate]})
    catalog_path = root / "candidate-catalog.json"
    write_json(catalog_path, catalog)
    catalog_path.chmod(0o600)
    monkeypatch.setenv(CATALOG_ENV, str(catalog_path))
    credential = root / "credential"
    credential.write_text("hermetic-native-private-token")
    credential.chmod(0o600)
    profile = {**profile, "screening_candidates": {spec["candidate_id"]: {
        "policy_execution_spec": launch._record(spec_path), "credential_file": str(credential),
        "credential_origin": policy["allowed_origin"], "provider_hard_cap_usd": "4.00",
        "hosted_inference_maximum_usd": "1.00", "hosted_inference_payment": "blueprint",
        "hosted_max_cost_per_request_usd": "0.00001"}}}
    return profile, catalog, spec


def family_planning_fixture(tmp_path, monkeypatch):
    operator = tmp_path / "operator"
    operator.mkdir()
    source = operator / "qualified_family_control_plane_unavailable.py"
    source.write_text("raise RuntimeError('SDK must not load on the control plane')\n")
    native = _runtime_contract()
    registration = {"runtime_id": "family-native-test", "factory_id": GYMNASIUM_FACTORY,
        "runtime_identity_digest": native["runtime_identity_digest"], "engine_id": "engine1",
        "task_families": ["button_press"], "robot_interfaces": [native["robot_configuration"]],
        "policy_interfaces": ["task_family_policy_https.v1"], "gymnasium_environment_id": "NativeFamily-v1",
        "registration_module": source.stem, "entry_point": source.stem + ":Environment",
        "runtime_sources": [{"module": source.stem, "sha256": _sha(source)}]}
    args = {"runtime_catalog_path": operator / "runtime-catalog.json"}
    _seal_catalog(args, registration)
    monkeypatch.setenv("BLUEPRINT_TASK_FAMILY_RUNTIME_CATALOG_PATH", str(args["runtime_catalog_path"]))
    runtime_source = _runtime_source_packet(tmp_path / "sources")
    capture = tmp_path / "capture"
    capture.mkdir()
    written, _registry, _preparation, _runtime = source_fixture(capture, monkeypatch,
        runtime_source_receipt=launch._record(runtime_source), runtime_selection=args["qualified_runtime_selection"])
    profile = profile_for_capture(capture)
    facts = scene_facts(profile)
    scene_plan = launch._read(facts["packet_dir"] / "native_task_arena_scene_plan.v1.json")
    profile, catalog, spec = configure_family_candidate(profile, scene_plan, root=operator / "candidate",
        monkeypatch=monkeypatch, registration=registration)
    planning = build_planning_inputs(catalog=catalog, profile=profile, capture_root=capture,
        source=facts["inputs"]["source"], inputs_dir=tmp_path / "planning")
    return planning, catalog, profile, facts, spec, written


def test_source_backed_family_capability_uses_native_interface_and_bounded_quote(tmp_path, monkeypatch):
    planning, catalog, _profile, _facts, spec, _written = family_planning_fixture(tmp_path, monkeypatch)
    assert "qualified_family_control_plane_unavailable" not in sys.modules
    assert planning["scene"]["pre_policy_site_qualification_required"] is True
    assert "robot_free_probe_digest" not in planning["scene"]
    binding = planning["capabilities"]["candidate_bindings"][spec["candidate_id"]]
    assert binding["backend"] == "task_family_native.v1"
    assert binding["execution_spec"]["hosted_http_request_limit"] == 80
    plan = build_screening_plan(catalog, planning["scene"], campaign_id="family-screening",
        capabilities=planning["capabilities"], cost_quotes=planning["cost_quotes"])
    assert plan["candidates"][0]["status"] == "selected"
    assert plan["maximum_planned_cost_usd"] == "5.00"
    assert all(cell["resolved_scenario"]["parameters"] == {} for cell in plan["scenario_cells"])


@pytest.mark.parametrize("controls_paused", [True, False])
def test_family_prepare_builds_real_sealed_provider_bundle(tmp_path, monkeypatch, controls_paused):
    planning, catalog, _profile, _facts, _spec_value, _written = family_planning_fixture(tmp_path, monkeypatch)
    plan = build_screening_plan(catalog, planning["scene"], campaign_id="family-screening",
        capabilities=planning["capabilities"], cost_quotes=planning["cost_quotes"])
    from blueprint_pipeline import task_evaluation_control_stage_policy
    assert task_evaluation_control_stage_policy.CONTROLS_PAUSED is True
    monkeypatch.setattr(task_evaluation_control_stage_policy, "CONTROLS_PAUSED", controls_paused)
    prepared = launch.prepare_candidate(plan=plan, candidate=plan["candidates"][0], attempt_dir=tmp_path / "attempt")
    authority = json.loads(Path(prepared["authority"]["path"]).read_text())
    assert authority["hard_cap_usd"] == 4 and authority["pre_policy_site_qualification_required"] is True
    runtime_inputs = launch._read(tmp_path / "attempt/runtime_inputs/policy_canary_runtime_inputs.json")
    assert all(cell["control_diagnostic"] == {
        "mode": "native_family_prepolicy_required", "typed_gap": "native_site_qualification_pending",
        "policy_execution_blocked": True} for cell in runtime_inputs["cells"])
    assert "qualified_family_control_plane_unavailable" not in sys.modules
    assert prepared["argv"][0] == "gpu-canary" and prepared["argv"][-1] != "--execute"
    assert "--beta-screening-credential-file" in prepared["argv"]
    assert not (tmp_path / "attempt/allocator_launch_intent.json").exists()


@pytest.mark.parametrize("field,value", [("controller", "other-controller"), ("action_bounds", [[-2, 2]]),
    ("observation_profile_id", "other-observation"), ("control_hz", 2), ("image_width", 32)])
def test_unmatched_native_interface_never_creates_capability(tmp_path, monkeypatch, field, value):
    _planning, catalog, profile, facts, spec, _written = family_planning_fixture(tmp_path, monkeypatch)
    changed = deepcopy(catalog)
    changed["candidates"][0]["interface"][field] = value
    changed = seal_catalog(changed)
    planning = build_planning_inputs(catalog=changed, profile=profile, capture_root=Path(profile["capture_root"]),
        source=facts["inputs"]["source"], inputs_dir=tmp_path / "unmatched-planning")
    assert spec["candidate_id"] not in planning["capabilities"]["candidate_bindings"]
    assert not planning["cost_quotes"]


def test_explicit_source_policy_selection_limits_the_plan_to_current_catalog_configuration(tmp_path, monkeypatch):
    _planning, catalog, profile, facts, spec, _written = family_planning_fixture(tmp_path, monkeypatch)
    another = deepcopy(catalog["candidates"][0])
    another.update(candidate_id="other-policy", display_name="Other qualified policy")
    catalog = seal_catalog({**catalog, "candidates": [catalog["candidates"][0], another]})
    write_json(Path(os.environ[CATALOG_ENV]), catalog)
    selected = {"candidate_id": spec["candidate_id"], "catalog_digest": catalog["catalog_digest"],
                "configuration_digest": catalog["candidates"][0]["configuration_digest"]}
    capture = tmp_path / "selected-capture"
    capture.mkdir()
    source_fixture(capture, monkeypatch, policy_selection=selected,
        runtime_selection=facts["scene"]["task_family_runtime_selection"],
        runtime_source_receipt=facts["inputs"]["runtime_source"])
    selected_profile = profile_for_capture(capture)
    selected_facts = scene_facts(selected_profile)
    scene_plan = launch._read(selected_facts["packet_dir"] / "native_task_arena_scene_plan.v1.json")
    selected_spec = {**spec, **family_policy_execution_binding(scene_plan),
        "task_success_contract": selected_facts["inputs"]["task_success_contract"],
        "task_success_contract_digest": selected_facts["inputs"]["task_success_contract_digest"]}
    selected_spec["execution_spec_digest"] = canonical_digest(selected_spec, digest_field="execution_spec_digest")
    spec_path = launch._write(tmp_path / "selected-execution.json", selected_spec)
    settings = deepcopy(profile["screening_candidates"][spec["candidate_id"]])
    settings["policy_execution_spec"] = launch._record(spec_path)
    selected_profile["screening_candidates"] = {spec["candidate_id"]: settings}
    planning = build_planning_inputs(catalog=catalog, profile=selected_profile, capture_root=capture,
        source=selected_facts["inputs"]["source"], inputs_dir=tmp_path / "selected-planning")
    assert planning["candidate_ids"] == [spec["candidate_id"]]
    plan = build_screening_plan(catalog, planning["scene"], candidate_ids=planning["candidate_ids"],
        campaign_id="selected-family", capabilities=planning["capabilities"], cost_quotes=planning["cost_quotes"])
    assert [row["candidate_id"] for row in plan["candidates"]] == [spec["candidate_id"]]
    changed = seal_catalog({**catalog, "revision": "changed operator catalog"})
    with pytest.raises(ValueError, match="policy_selection_changed"):
        build_planning_inputs(catalog=changed, profile=selected_profile, capture_root=capture,
            source=selected_facts["inputs"]["source"], inputs_dir=tmp_path / "changed-planning")


@pytest.mark.parametrize("source_state", ["current", "revoked", "expired"])
def test_paused_family_launch_keeps_source_gate_before_allocator(tmp_path, monkeypatch, source_state):
    from blueprint_pipeline import beta_policy_screening_worker as worker
    from blueprint_pipeline import task_evaluation_control_stage_policy as control_policy
    from blueprint_pipeline import team_candidate_quick10_launch as native

    assert control_policy.CONTROLS_PAUSED is True
    planning, catalog, _profile, _facts, _spec_value, _written = family_planning_fixture(tmp_path, monkeypatch)
    plan = build_screening_plan(catalog, planning["scene"], campaign_id="family-screening",
        capabilities=planning["capabilities"], cost_quotes=planning["cost_quotes"])
    candidate = plan["candidates"][0]
    attempt = tmp_path / "attempt"
    prepared = launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=attempt)
    calls = []

    def current_source(source, *, site_id):
        calls.append("current_source")
        assert source == plan["source"] and site_id == plan["site_id"]
        if source_state != "current":
            # The current-source service can refuse either ended authority.
            # This checks propagation through the real prelaunch sequence.
            raise ValueError("source_authority_" + source_state)
        return {}

    def start_allocator(argv, **_):
        assert calls == ["current_source"]
        assert argv == prepared["argv"]
        calls.append("allocator_transport")
        return 12345

    monkeypatch.setattr(worker, "require_current_source", current_source)
    monkeypatch.setattr(native, "_start_allocator", start_allocator)
    monkeypatch.setattr(native, "_process_start_ticks", lambda _pid: "1")
    result = launch.launch_candidate(plan=plan, candidate=candidate, attempt_dir=attempt, prepared=prepared)
    assert "qualified_family_control_plane_unavailable" not in sys.modules
    if source_state == "current":
        assert result["status"] == "running"
        assert calls == ["current_source", "allocator_transport"]
    else:
        assert result["status"] == "blocked" and result["no_allocation_proven"] is True
        assert result["provider_allocation_performed"] is False
        assert calls == ["current_source"]
        assert launch.launch_candidate(plan=plan, candidate=candidate, attempt_dir=attempt, prepared=prepared) == result
        assert calls == ["current_source"]


def test_paused_family_launch_keeps_expired_quote_before_allocator(tmp_path, monkeypatch):
    from datetime import datetime, timedelta
    from types import SimpleNamespace
    from blueprint_pipeline import task_evaluation_control_stage_policy as control_policy
    from blueprint_pipeline import team_candidate_quick10_launch as native

    assert control_policy.CONTROLS_PAUSED is True
    planning, catalog, _profile, _facts, _spec_value, _written = family_planning_fixture(tmp_path, monkeypatch)
    plan = build_screening_plan(catalog, planning["scene"], campaign_id="family-screening",
        capabilities=planning["capabilities"], cost_quotes=planning["cost_quotes"])
    candidate = plan["candidates"][0]
    attempt = tmp_path / "attempt"
    prepared = launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=attempt)
    expires = datetime.fromisoformat(candidate["cost_quote"]["valid_until"].replace("Z", "+00:00"))
    monkeypatch.setattr(launch, "datetime", SimpleNamespace(fromisoformat=datetime.fromisoformat,
        now=lambda _timezone: expires + timedelta(seconds=1)))
    monkeypatch.setattr(native, "_start_allocator", lambda *_, **__: pytest.fail("expired quote reached allocator"))
    result = launch.launch_candidate(plan=plan, candidate=candidate, attempt_dir=attempt, prepared=prepared)
    assert result["status"] == "blocked" and result["no_allocation_proven"] is True
    assert result["provider_allocation_performed"] is False


@pytest.mark.parametrize("ended_authority", ["revoked", "expired"])
@pytest.mark.parametrize("when", ["after_plan", "after_prepared"])
def test_paused_family_reopens_late_original_intent_before_preparation_or_launch(
    tmp_path, monkeypatch, ended_authority, when,
):
    from types import SimpleNamespace
    from blueprint_pipeline import task_evaluation_scene_owner_authority as authority
    from blueprint_pipeline import task_evaluation_control_stage_policy as control_policy
    from blueprint_pipeline import task_family_native_runtime as runtime
    from blueprint_pipeline import team_candidate_quick10_launch as native
    from blueprint_pipeline.task_family_policy_https_client import TaskFamilyPolicyHttpsClient

    assert control_policy.CONTROLS_PAUSED is True
    planning, catalog, _profile, facts, _spec_value, _written = family_planning_fixture(tmp_path, monkeypatch)
    plan = build_screening_plan(catalog, planning["scene"], campaign_id="family-screening",
        capabilities=planning["capabilities"], cost_quotes=planning["cost_quotes"])
    candidate = plan["candidates"][0]
    attempt = tmp_path / "attempt"
    prepared = (launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=attempt)
                if when == "after_prepared" else None)
    reference = facts["inputs"]["references"]["scene_intent"]
    intent_path = Path(reference["path"])
    if ended_authority == "revoked":
        write_json(intent_path.parent / "revoked.json", {"revoked": True})
    else:
        original = json.loads(intent_path.read_text())
        expired_now = original["request"]["execution"]["expires_at_epoch"] + 1
        monkeypatch.setattr(authority, "time", SimpleNamespace(time=lambda: expired_now))
    monkeypatch.setattr(native, "_start_allocator", lambda *_, **__: pytest.fail("ended authority reached allocator"))
    monkeypatch.setattr(runtime, "build_task_family_native_runtime",
        lambda *_, **__: pytest.fail("ended authority constructed native runtime"))
    monkeypatch.setattr(TaskFamilyPolicyHttpsClient, "__init__",
        lambda *_, **__: pytest.fail("ended authority constructed policy client"))
    if when == "after_plan":
        with pytest.raises(ValueError, match="scene_owner_authority_" + ended_authority):
            launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=attempt)
        assert not attempt.exists()
    else:
        result = launch.launch_candidate(plan=plan, candidate=candidate, attempt_dir=attempt, prepared=prepared)
        assert result["status"] == "blocked" and result["no_allocation_proven"] is True
        assert result["provider_allocation_performed"] is False
        assert not (attempt / "allocator_process.json").exists()
        assert launch.launch_candidate(plan=plan, candidate=candidate, attempt_dir=attempt, prepared=prepared) == result
    assert "qualified_family_control_plane_unavailable" not in sys.modules
