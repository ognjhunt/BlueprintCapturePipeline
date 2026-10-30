"""Development-only full beta binding/packaging; never a provider or GPU claim."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from blueprint_pipeline import beta_policy_screening_launch as launch
from blueprint_pipeline.beta_policy_candidate_catalog import load_default_catalog
from blueprint_pipeline.beta_policy_screening_capabilities import build_planning_inputs
from blueprint_pipeline.beta_policy_screening_plan import build_screening_plan
from blueprint_pipeline.beta_policy_screening_team import catalog_for_profile
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.molmoact2_yam_runtime import (
    CANDIDATE_ID, EMBODIMENT_ID, MINIMUM_GPU_MEMORY_GB, build_policy_spec, policy_binding, configure_prepared_inputs,
)
from blueprint_pipeline.team_candidate_quick10_registry import profile_for_capture
from blueprint_pipeline.team_embodiment import validate_embodiment_manifest
from tests.test_beta_policy_screening_launch import _plan
from tests.test_team_arm_kinematics import named_inputs


def prepared_registration(tmp_path, monkeypatch):
    base, _ = _plan(tmp_path, monkeypatch, scene_half="team_scene")
    profile = profile_for_capture(str(tmp_path / "capture"))
    model, manifest = named_inputs(tmp_path)
    # Exact joint/channel names in a development-only USD, not native evidence.
    raw = json.dumps(manifest)
    usd = model.read_text()
    for side in ("left", "right"):
        for index in range(6):
            before, after = f"{side}_joint_{index}", f"{side}_joint{index + 1}"
            raw, usd = raw.replace(before, after), usd.replace(before, after)
    model.write_text(usd)
    manifest = json.loads(raw)
    manifest["model"]["sha256"] = "sha256:" + hashlib.sha256(model.read_bytes()).hexdigest()
    for arm, gripper in zip(manifest["arms"], manifest["grippers"], strict=True):
        arm["gripper_name"] += "_gripper"
        gripper["name"] += "_gripper"
    manifest.update(embodiment_id=EMBODIMENT_ID, action_chunk_rows=30, control_hz=60)
    for camera, name in zip(manifest["cameras"], ("overhead", "left_wrist", "right_wrist"), strict=True):
        camera["name"] = name
    manifest = configure_prepared_inputs({"schema_version": "embodiment_runtime_binding_inputs.v1",
                                          "robot_embodiment": manifest})["robot_embodiment"]
    manifest_path = launch._write(tmp_path / "piper-manifest.json", manifest)
    rights = {"schema_version": "blueprint.team_policy_candidate_rights.v1", "status": "admitted",
              "team_id": "development-only", "checkpoint_id": CANDIDATE_ID, "candidate_id": CANDIDATE_ID,
              "license_to_run_granted": True, "policy_self_grading_permitted": False,
              "official_ranking_permitted": False, "secret_material_recorded": False,
              "adp_qualified_evidence_permitted": False}
    rights["rights_digest"] = canonical_digest(rights, digest_field="rights_digest")
    registration = {"manifest": launch._record(manifest_path), "model": launch._record(model),
        "team_id": rights["team_id"], "checkpoint_id": CANDIDATE_ID, "rights_binding": rights,
        "policy_spec": build_policy_spec(manifest), "minimum_gpu_memory_gb": MINIMUM_GPU_MEMORY_GB,
        "deployment": {"configuration_id": EMBODIMENT_ID, "policy_route_id": "customer_controller",
            "companies": [], "relationship": "public_controller_integration",
            "evidence_url": "https://huggingface.co/allenai/MolmoAct2-BimanualYAM"}}
    profile.update(screening_candidates={CANDIDATE_ID: {"robot_policy": registration}},
                   screening_hard_ttl_seconds=1800, screening_candidate_cap_usd="1.00")
    registry_path = Path(__import__("os").environ["BLUEPRINT_TEAM_CANDIDATE_QUICK10_REGISTRY"])
    registry = launch._read(registry_path)
    registry["profiles"] = [profile]
    registry["registry_digest"] = canonical_digest(registry, digest_field="registry_digest")
    registry_path.write_text(json.dumps(registry))
    return base, profile, registration


def test_registered_local_checkpoint_retains_all_routes_and_builds_real_sealed_bundle(tmp_path, monkeypatch):
    from blueprint_pipeline.policy_canary_session_bundle_identity import validate_policy_canary_provider_manifest
    from blueprint_pipeline.molmoact2_yam_bundle import transfer_byte_budget
    from blueprint_pipeline.team_candidate_policy_client import canary_policy_client
    from blueprint_pipeline import native_task_arena_vast as transport
    from blueprint_pipeline.vast_provider_adapter import _blueprint_bundle_preflight

    base, profile, _ = prepared_registration(tmp_path, monkeypatch)
    catalog = catalog_for_profile(profile)
    assert len(catalog["candidates"]) == len(load_default_catalog()["candidates"])
    row = next(row for row in catalog["candidates"] if row["candidate_id"] == CANDIDATE_ID)
    assert row["eligibility"]["enabled"] and row["eligibility"]["requires_credentials"] == []
    assert row["deployment"]["policy_route_id"] == "customer_controller"
    planning = build_planning_inputs(catalog=catalog, profile=profile, capture_root=tmp_path / "capture",
                                    source=base["source"], inputs_dir=tmp_path / "molmoact2-plan")
    plan = build_screening_plan(catalog, planning["scene"], capabilities=planning["capabilities"],
                               cost_quotes=planning["cost_quotes"], campaign_id="molmoact2-fixture")
    candidate = next(row for row in plan["candidates"] if row["candidate_id"] == CANDIDATE_ID)
    assert candidate["status"] == "selected", candidate["blockers"]
    spec = launch._read(launch._bound_path(candidate["execution_spec"]["policy_execution_spec"]))
    client = canary_policy_client(spec)
    assert client.require_named_arm_state and client.action_chunk_rows == 30
    assert client.binding.execution_horizon == 25
    assert client.evidence_summary()["execution_binding_digest"] == spec["execution_spec_digest"]
    result = launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=tmp_path / "attempt")
    bundle = launch._read(Path(result["bundle_receipt"]["path"]))
    with zipfile.ZipFile(bundle["bundle_path"]) as archive:
        manifest = validate_policy_canary_provider_manifest(json.loads(archive.read("provider_runtime/adp_arena_provider_manifest.json")))
        assert manifest["team_policy_interface_id"] == "molmoact2_yam_joint14.v1"
        assert "team_credential_env" not in manifest and manifest["minimum_gpu_memory_gb"] == 48
        assert "provider_runtime/native_task_packet/team_embodiment/robot.usd" in archive.namelist()
        assert "provider_runtime/blueprint_pipeline/_catalog_data/molmoact2_yam_release.v1.json" in archive.namelist()
        assert b"molmoact2_yam_provisioning" in archive.read(f"provider_runtime/adp009d_policy_provisioning.{CANDIDATE_ID}.sh")
    download, upload = transfer_byte_budget(bundle)
    assert download > 21_768_955_912 + 64 * 1024**3 and upload == 8_000_000_000
    preflight = _blueprint_bundle_preflight(job_dir=tmp_path / "transport-preflight", generated_at="fixture",
        enable_blueprint_bundle=True, enable_isaac_smoke=True,
        provider_bundle_kind="native_task_arena_policy_canary_session", bundle_path=Path(bundle["bundle_path"]),
        provider_bundle_url="https://storage.example.com/bundle.zip", provider_output_put_url="https://storage.example.com/output.zip")
    assert preflight["status"] == "passed", preflight.get("blockers")
    observed = []
    monkeypatch.setenv("BLUEPRINT_POLICY_CANARY_OUTPUT_DELIVERY", "download")
    monkeypatch.setattr(transport, "run_arena_native_control_vast", lambda **kwargs: observed.append(kwargs) or {"status": "dry_run_ready"})
    authority = launch._read(Path(result["authority"]["path"]))
    transport.run_native_task_arena_policy_canary_session_vast(job_dir=tmp_path / "transport",
        prepared_bundle=bundle, session_authority=authority, paid_resource_admission_grant=None, execute=False,
        hard_cap_usd=float(authority["hard_cap_usd"]), hard_ttl_seconds=authority["hard_ttl_seconds"])
    assert len(observed) == 1 and observed[0]["expected_provider_download_bytes"] == download
    assert observed[0]["expected_provider_upload_bytes"] == upload
    assert observed[0]["min_gpu_ram_mb"] == 48_000
    assert observed[0]["forward_hf_token"] is False and observed[0]["stale_offer_create_retry_limit"] == 0
    assert observed[0]["require_independent_watchdog"] is True
    assert not (tmp_path / "attempt/allocator_launch_intent.json").exists()


@pytest.mark.parametrize("fault", ["body", "cadence", "frames", "checkpoint", "chunk", "rights", "memory"])
def test_invalid_local_binding_keeps_individual_reason_and_frozen_candidates(tmp_path, monkeypatch, fault):
    _, profile, registration = prepared_registration(tmp_path, monkeypatch)
    policy = registration["policy_spec"]
    if fault == "body":
        policy["embodiment_id"] = "unrelated_body"
    elif fault == "cadence":
        policy["action_hz"] = 15
    elif fault == "frames":
        policy["binding"]["camera_names"] = ["unknown"]
    elif fault == "checkpoint":
        policy["model_identity_digest"] = "sha256:" + "e" * 64
    elif fault == "chunk":
        policy["action_chunk_rows"] = 32
    elif fault == "memory":
        registration["minimum_gpu_memory_gb"] = True
    else:
        registration["rights_binding"]["license_to_run_granted"] = False
    catalog = catalog_for_profile(profile)
    row = next(row for row in catalog["candidates"] if row["candidate_id"] == CANDIDATE_ID)
    assert not row["eligibility"]["enabled"] and row["eligibility"]["blockers"]
    for original in load_default_catalog()["candidates"]:
        if original["candidate_id"] == CANDIDATE_ID:
            assert row["eligibility"]["blockers"] != original["eligibility"]["blockers"]
            continue
        assert next(row for row in catalog["candidates"] if row["candidate_id"] == original["candidate_id"]) == original



def test_rig_override_preserves_prepared_mounts_and_explicit_assumptions(tmp_path, monkeypatch):
    _, _, registration = prepared_registration(tmp_path, monkeypatch)
    manifest = launch._read(launch._bound_path(registration["manifest"]))
    binding = policy_binding(registration["policy_spec"])
    assert "not_publisher_or_site_calibration" in binding.camera_binding_reference
    assert "D405_CAD" in binding.camera_binding_reference
    assert [(c["width"], c["height"], c["horizontal_fov_deg"]) for c in manifest["cameras"]] == [
        (640, 360, 69.4), (640, 360, 87), (640, 360, 87)]
    changed = deepcopy(manifest)
    changed.pop("embodiment_digest")
    changed["cameras"][0]["height"] = 176
    with pytest.raises(ValueError, match="robot_binding_invalid"):
        build_policy_spec(validate_embodiment_manifest(changed))


def test_actual_native_loop_holds_yam_joint_rows_twice_and_requeries_after_25(tmp_path, monkeypatch):
    """Real client and native episode loop; fake physics is not GPU evidence."""
    import numpy as np
    from types import SimpleNamespace
    from blueprint_pipeline import adp_task_scoring, episode_visual_evidence, native_task_episode_environment
    from blueprint_pipeline.molmoact2_yam_policy import MolmoAct2YAMPolicyClient
    from blueprint_pipeline.molmoact2_yam_policy_server import MolmoAct2YAMServer
    from blueprint_pipeline.team_embodiment import (
        apply_team_embodiment, read_model_structure, team_embodiment_binding, verify_model_file,
    )
    from blueprint_pipeline.team_embodiment_runtime import run_team_episode
    from tests.test_molmoact2_yam_policy import Backend, Connection
    from tests.test_team_arm_kinematics import CartesianRobot
    from tests.test_team_embodiment import _Array, _Camera, _Scoring, station_plan

    _, _, registration = prepared_registration(tmp_path, monkeypatch)
    declared = launch._read(launch._bound_path(registration["manifest"]))
    model = launch._bound_path(registration["model"])
    bound = team_embodiment_binding(declared, model_record=verify_model_file(declared, model),
                                    structure=read_model_structure(model), scene_plan=station_plan())
    plan = {**apply_team_embodiment(station_plan(), bound), "scenario": {"seed": 7}}
    plan["cadence"].update(maximum_action_steps=52, settle_window_samples=0)
    row = plan["robot"]
    class Env:
        def __init__(self):
            self.robot, self.steps = CartesianRobot(row), []
            cameras = {}
            for name in ("team_camera_0", "team_camera_1", "team_camera_2", "external_camera_2"):
                camera = _Camera()
                camera.data.output["rgb"] = _Array(np.full((1, 360, 640, 3), 90, dtype=np.uint8))
                cameras[name] = camera
            self.unwrapped = SimpleNamespace(scene={"robot": self.robot, **cameras}, device="cpu")
        def reset(self, seed):
            self.robot.q = np.array([row["joint_reset_positions_rad"][name] for name in self.robot.joint_names])
        def step(self, action):
            order = [*row["joint_names"], *(joint["name"] for joint in row["gripper_joints"])]
            self.steps.append(list(action[0]))
            self.robot.q = np.array([action[0][order.index(name)] for name in self.robot.joint_names])
    env, backend, requests = Env(), Backend(), []
    backend.action[:, :6] = .01
    backend.action[:, 7:13] = -.01
    backend.action[:, 6], backend.action[:, 13] = 0., 1.
    binding = policy_binding(registration["policy_spec"])
    server = MolmoAct2YAMServer(backend=backend, binding=binding, checkpoint_receipt={"checkpoint_bytes_verified": True})
    client = MolmoAct2YAMPolicyClient(binding=binding, connection_factory=lambda *_, **__: Connection(server, requests))
    built = SimpleNamespace(env=env, camera_scene_names={"team_camera_0": "team_camera_0",
        "team_camera_1": "team_camera_1", "team_camera_2": "team_camera_2", "overview": "external_camera_2"})
    monkeypatch.setattr(native_task_episode_environment, "NativeRigidScoringEnvironment", _Scoring)
    monkeypatch.setattr(episode_visual_evidence, "finalize_multicamera_visual_evidence", lambda **_: ({"status": "complete"}, []))
    monkeypatch.setattr(adp_task_scoring, "score_task_episode_from_spec", lambda **_: {"task_succeeded": False})
    result = run_team_episode(built=built, task_readback=None, policy=client, plan=plan, prompt="Move the object",
        max_policy_queries=2, open_loop_horizon=25, media_root=tmp_path / "episodes", episode_id="development-only-yam",
        progress={}, to_action=lambda targets: [targets])
    assert len(backend.calls) == 2 and len(env.steps) == 52
    assert backend.calls[1]["state"][:6] == pytest.approx([.01] * 6)
    assert backend.calls[1]["state"][7:13] == pytest.approx([-.01] * 6)
    assert env.steps[-1][-2:] == pytest.approx([0., .03])
    assert result["policy_request_evidence_complete"] is True
    assert result["policy_cadence"]["native_hold_ticks_per_action"] == 2
    assert result["score"]["task_succeeded"] is False
    assert client.evidence_summary()["native_execution_validated"] is False
