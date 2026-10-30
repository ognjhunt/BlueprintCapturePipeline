"""Development-only prepared candidate through real sealing; no GPU/provider calls."""
import json
import pytest
from pathlib import Path
import zipfile


from blueprint_pipeline import beta_policy_screening_launch as launch
from blueprint_pipeline.beta_policy_screening_capabilities import build_planning_inputs
from blueprint_pipeline.beta_policy_screening_plan import build_screening_plan
from blueprint_pipeline.beta_policy_screening_team import catalog_for_profile
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.scripted_robot_baseline import EVIDENCE_KIND
from blueprint_pipeline.scripted_robot_baseline_runtime import candidate_id
from blueprint_pipeline.scripted_robot_runtime_dispatch import build_policy_spec
from blueprint_pipeline.team_candidate_quick10_registry import profile_for_capture
from tests.test_scripted_robot_grasp_frames import development_manifest


def registered_world(tmp_path, monkeypatch, identity="agilex_piper_bimanual"):
    from pxr import Usd, UsdPhysics
    from tests import test_native_task_arena_policy_canary_lifecycle_rehearsal as lifecycle
    from tests.test_native_task_construction_plan import _rigid_fixture
    from tests.test_beta_policy_screening_launch import _plan

    original = lifecycle._scene_plan
    def authored_fixture():
        plan = original()
        authored = _rigid_fixture(asset_id="development-only-grasp-subject")
        plan["task_spec"].update(authored["task_spec"])
        subject = next(row for row in plan["objects"] if row.get("task_subject"))
        subject.update(authored["objects"][0])
        plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
        return plan
    monkeypatch.setattr(lifecycle, "_scene_plan", authored_fixture)
    base, _ = _plan(tmp_path, monkeypatch, scene_half="team_scene")
    profile = profile_for_capture(str(tmp_path / "capture"))
    declared = development_manifest(identity)
    model = tmp_path / "development-only-robot.usda"
    stage = Usd.Stage.CreateNew(str(model))
    stage.SetDefaultPrim(stage.DefinePrim("/robot", "Xform"))
    links = {arm[key] for arm in declared["arms"] for key in ("base_link", "tool_link")}
    links.update(link for gripper in declared["grippers"] for link in gripper["contact_links"])
    for name in sorted(links):
        UsdPhysics.RigidBodyAPI.Apply(stage.DefinePrim("/robot/" + name, "Xform"))
    for joint in declared["joints"]:
        UsdPhysics.RevoluteJoint.Define(stage, "/robot/" + joint["name"])
    for channel in declared["grippers"]:
        for joint in channel["joints"]:
            UsdPhysics.PrismaticJoint.Define(stage, "/robot/" + joint["name"])
    for arm in declared["arms"]:
        UsdPhysics.FixedJoint.Define(stage, "/robot/" + arm["name"] + "_anchor").CreateBody1Rel().SetTargets(
            ["/robot/" + arm["base_link"]])
    stage.GetRootLayer().Save()
    declared = development_manifest(identity, model)
    manifest_path = launch._write(tmp_path / "development-only.manifest.json", declared)
    key = candidate_id(declared["embodiment_id"])
    rights = {"schema_version": "blueprint.team_policy_candidate_rights.v1", "status": "admitted",
        "team_id": "development-only", "checkpoint_id": key, "candidate_id": key, "license_to_run_granted": True,
        "policy_self_grading_permitted": False, "official_ranking_permitted": False,
        "secret_material_recorded": False, "adp_qualified_evidence_permitted": False}
    rights["rights_digest"] = canonical_digest(rights, digest_field="rights_digest")
    registration = {"manifest": launch._record(manifest_path), "model": launch._record(model),
        "team_id": rights["team_id"], "checkpoint_id": key, "rights_binding": rights,
        "policy_spec": build_policy_spec(declared), "minimum_gpu_memory_gb": 46,
        "deployment": {"configuration_id": declared["embodiment_id"], "policy_route_id": "blueprint_scripted_baseline",
            "companies": [], "relationship": "public_controller_integration", "evidence_url": "development-only"}}
    profile.update(screening_candidates={key: {"robot_policy": registration}},
                   screening_hard_ttl_seconds=1800, screening_candidate_cap_usd="1.00")
    path = Path(__import__("os").environ["BLUEPRINT_TEAM_CANDIDATE_QUICK10_REGISTRY"])
    registry = launch._read(path)
    registry["profiles"] = [profile]
    registry["registry_digest"] = canonical_digest(registry, digest_field="registry_digest")
    path.write_text(json.dumps(registry))
    return base, profile, registration, key


@pytest.mark.parametrize("identity", [
    "yaskawa_gp12_robotiq_2f85_sim", "yaskawa_gp20_robotiq_2f85_sim",
    "kuka_lbr_iiwa_14_robotiq_2f85_sim",
    "fanuc_crx_10ia_robotiq_2f85_sim", "fanuc_crx_10ia_l_robotiq_2f85_sim",
    "fanuc_m710ic_45m_robotiq_2f85_sim", "abb_gofa_crb15000_5_95_robotiq_2f85_sim",
    "abb_gofa_crb15000_12_127_robotiq_2f85_sim", "agilex_piper_bimanual", "universal_robots_ur5e_robotiq_2f85_sim",
    "universal_robots_ur10e_robotiq_2f85_sim", "franka_research_3_robotiq_2f85_sim"])
def test_scripted_candidate_seals_without_checkpoint_server_credentials_or_learned_count(tmp_path, monkeypatch, identity):
    from blueprint_pipeline.policy_canary_session_bundle_identity import validate_policy_canary_provider_manifest
    from blueprint_pipeline.team_candidate_policy_client import canary_policy_client
    from blueprint_pipeline.vast_provider_adapter import _blueprint_bundle_preflight

    base, profile, _, key = registered_world(tmp_path, monkeypatch, identity)
    catalog = catalog_for_profile(profile)
    configured = next(row for row in catalog["candidates"] if row["candidate_id"] == key)
    assert configured["policy"]["kind"] == "scripted"
    assert configured["policy"]["digest_kind"] == "source_code_inventory"
    assert configured["eligibility"]["enabled"] and configured["eligibility"]["requires_credentials"] == []
    planning = build_planning_inputs(catalog=catalog, profile=profile, capture_root=tmp_path / "capture",
        source=base["source"], inputs_dir=tmp_path / "scripted-plan")
    plan = build_screening_plan(catalog, planning["scene"], capabilities=planning["capabilities"],
        cost_quotes=planning["cost_quotes"], campaign_id="development-only-scripted")
    candidate = next(row for row in plan["candidates"] if row["candidate_id"] == key)
    assert candidate["status"] == "selected", candidate["blockers"]
    assert candidate["execution_evidence_kind"] == EVIDENCE_KIND
    spec = launch._read(launch._bound_path(candidate["execution_spec"]["policy_execution_spec"]))
    client = canary_policy_client(spec)
    assert client.execution_evidence_kind == EVIDENCE_KIND and not client.scripted_controller_invoked
    assert spec["policy_endpoint"] == {"transport": "in_process"}
    assert spec["weights_identity_verified"] is False
    prepared = launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=tmp_path / "attempt")
    authority = launch._read(Path(prepared["authority"]["path"]))
    assert authority["learned_policy_rollout_count"] == 0 and authority["scripted_rollout_count"] == 10
    bundle = launch._read(Path(prepared["bundle_receipt"]["path"]))
    with zipfile.ZipFile(bundle["bundle_path"]) as archive:
        manifest = validate_policy_canary_provider_manifest(json.loads(archive.read("provider_runtime/adp_arena_provider_manifest.json")))
        assert manifest["model_delivery"] == "scripted_in_process"
        assert manifest["learned_policy_rollout_count"] == 0 and manifest["scripted_rollout_count"] == 10
        assert manifest["candidate_execution_evidence_kinds"] == {key: EVIDENCE_KIND}
        assert "team_credential_env" not in manifest
        assert not any("adp009d_policy_provisioning." in name for name in archive.namelist())
        assert "provider_runtime/blueprint_pipeline/scripted_robot_grasp_frames.py" in archive.namelist()
    transport = _blueprint_bundle_preflight(job_dir=tmp_path / "transport-preflight", generated_at="development-only",
        enable_blueprint_bundle=True, enable_isaac_smoke=True, provider_bundle_kind="native_task_arena_policy_canary_session",
        bundle_path=Path(bundle["bundle_path"]), provider_bundle_url="https://storage.example/bundle.zip",
        provider_output_put_url="https://storage.example/output.zip")
    assert transport["status"] == "passed", transport.get("blockers")
    assert not (tmp_path / "attempt/allocator_launch_intent.json").exists()


def test_changed_source_identity_keeps_scripted_candidate_with_individual_reason(tmp_path, monkeypatch):
    _, profile, registration, key = registered_world(tmp_path, monkeypatch)
    original = catalog_for_profile(profile)
    registration["policy_spec"]["model_identity_digest"] = "sha256:" + "0" * 64
    changed = catalog_for_profile(profile)
    scripted = next(row for row in changed["candidates"] if row["candidate_id"] == key)
    assert scripted["policy"]["kind"] == "scripted" and not scripted["eligibility"]["enabled"]
    assert scripted["eligibility"]["blockers"] == ["scripted_baseline_robot_or_source_binding_changed"]
    assert [row for row in changed["candidates"] if row["candidate_id"] != key] == [row for row in original["candidates"] if row["candidate_id"] != key]
