"""Development-only full beta binding/packaging; never a provider or GPU claim."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from blueprint_pipeline import beta_policy_screening_launch as launch
from blueprint_pipeline.beta_policy_candidate_catalog import load_default_catalog
from blueprint_pipeline.beta_policy_screening_capabilities import build_planning_inputs
from blueprint_pipeline.beta_policy_screening_plan import build_screening_plan
from blueprint_pipeline.beta_policy_screening_team import catalog_for_profile
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.trlc_pi05_runtime import (
    CANDIDATE_ID,
    EMBODIMENT_ID,
    MINIMUM_GPU_MEMORY_GB,
    build_policy_spec,
    policy_binding,
    configure_prepared_inputs,
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
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(model))
    for gripper in manifest["grippers"]:
        first = gripper["joints"][0]
        first.update(open=0.0, closed=-0.045)
        second = {**first, "name": first["name"] + "_second"}
        gripper["joints"].append(second)
        manifest["reset_joint_positions"][first["name"]] = 0.0
        manifest["reset_joint_positions"][second["name"]] = 0.0
        UsdPhysics.PrismaticJoint.Define(stage, "/robot/" + second["name"])
    stage.GetRootLayer().Save()
    manifest["model"]["sha256"] = "sha256:" + hashlib.sha256(model.read_bytes()).hexdigest()
    manifest.update(embodiment_id=EMBODIMENT_ID, action_chunk_rows=50, control_hz=60)
    for camera, name in zip(
        manifest["cameras"], ("overhead", "left_wrist", "right_wrist"), strict=True
    ):
        camera["name"] = name
    manifest = configure_prepared_inputs(
        {"schema_version": "embodiment_runtime_binding_inputs.v1", "robot_embodiment": manifest}
    )["robot_embodiment"]
    manifest_path = launch._write(tmp_path / "piper-manifest.json", manifest)
    rights = {
        "schema_version": "blueprint.team_policy_candidate_rights.v1",
        "status": "admitted",
        "team_id": "development-only",
        "checkpoint_id": CANDIDATE_ID,
        "candidate_id": CANDIDATE_ID,
        "license_to_run_granted": True,
        "policy_self_grading_permitted": False,
        "official_ranking_permitted": False,
        "secret_material_recorded": False,
        "adp_qualified_evidence_permitted": False,
    }
    rights["rights_digest"] = canonical_digest(rights, digest_field="rights_digest")
    registration = {
        "manifest": launch._record(manifest_path),
        "model": launch._record(model),
        "team_id": rights["team_id"],
        "checkpoint_id": CANDIDATE_ID,
        "rights_binding": rights,
        "policy_spec": build_policy_spec(manifest),
        "minimum_gpu_memory_gb": MINIMUM_GPU_MEMORY_GB,
        "deployment": {
            "configuration_id": EMBODIMENT_ID,
            "policy_route_id": "pi05_custom_checkpoint",
            "companies": [],
            "relationship": "public_controller_integration",
            "evidence_url": "https://huggingface.co/djkesu/pi05_trlc_rabc_3cam_alpha025_20260319_v2_step10000_merged",
        },
    }
    profile.update(
        screening_candidates={CANDIDATE_ID: {"robot_policy": registration}},
        screening_hard_ttl_seconds=1800,
        screening_candidate_cap_usd="1.00",
    )
    registry_path = Path(__import__("os").environ["BLUEPRINT_TEAM_CANDIDATE_QUICK10_REGISTRY"])
    registry = launch._read(registry_path)
    registry["profiles"] = [profile]
    registry["registry_digest"] = canonical_digest(registry, digest_field="registry_digest")
    registry_path.write_text(json.dumps(registry))
    return base, profile, registration


def test_registered_local_checkpoint_retains_all_routes_and_builds_real_sealed_bundle(
    tmp_path, monkeypatch
):
    from blueprint_pipeline.policy_canary_session_bundle_identity import (
        validate_policy_canary_provider_manifest,
    )
    from blueprint_pipeline.trlc_pi05_bundle import transfer_byte_budget
    from blueprint_pipeline.team_candidate_policy_client import canary_policy_client
    from blueprint_pipeline import native_task_arena_vast as transport
    from blueprint_pipeline.vast_provider_adapter import _blueprint_bundle_preflight

    base, profile, _ = prepared_registration(tmp_path, monkeypatch)
    catalog = catalog_for_profile(profile)
    assert len(catalog["candidates"]) == len(load_default_catalog()["candidates"])
    row = next(row for row in catalog["candidates"] if row["candidate_id"] == CANDIDATE_ID)
    assert row["eligibility"]["enabled"] and row["eligibility"]["requires_credentials"] == []
    assert row["deployment"]["policy_route_id"] == "pi05_custom_checkpoint"
    planning = build_planning_inputs(
        catalog=catalog,
        profile=profile,
        capture_root=tmp_path / "capture",
        source=base["source"],
        inputs_dir=tmp_path / "trlc_pi05-plan",
    )
    plan = build_screening_plan(
        catalog,
        planning["scene"],
        capabilities=planning["capabilities"],
        cost_quotes=planning["cost_quotes"],
        campaign_id="trlc_pi05-fixture",
    )
    candidate = next(row for row in plan["candidates"] if row["candidate_id"] == CANDIDATE_ID)
    assert candidate["status"] == "selected", candidate["blockers"]
    spec = launch._read(launch._bound_path(candidate["execution_spec"]["policy_execution_spec"]))
    client = canary_policy_client(spec)
    assert client.require_named_arm_state and client.action_chunk_rows == 50
    assert client.binding.execution_horizon == 50
    assert client.evidence_summary()["execution_binding_digest"] == spec["execution_spec_digest"]
    from blueprint_pipeline import trlc_pi05_bundle as bundle_module

    # Source/tokenizer bytes have separate real verifier tests. This tiny
    # native source fixture deliberately contains no4GB wheel closure.
    monkeypatch.setattr(bundle_module, "validate_native_inputs", lambda **_: {})
    original_stage = bundle_module.stage_runtime

    def stage(runtime):
        from tests.test_trlc_pi05_policy import fake_tokenizer

        directory = fake_tokenizer(tmp_path / "tokenizer", monkeypatch)
        monkeypatch.setenv("BLUEPRINT_G1_PI_TOKENIZER_DIR", str(directory))
        return original_stage(runtime)

    monkeypatch.setattr(bundle_module, "stage_runtime", stage)
    result = launch.prepare_candidate(
        plan=plan, candidate=candidate, attempt_dir=tmp_path / "attempt"
    )
    bundle = launch._read(Path(result["bundle_receipt"]["path"]))
    with zipfile.ZipFile(bundle["bundle_path"]) as archive:
        manifest = validate_policy_canary_provider_manifest(
            json.loads(archive.read("provider_runtime/adp_arena_provider_manifest.json"))
        )
        assert manifest["team_policy_interface_id"] == "pi05_trlc_joint14.v1"
        assert "team_credential_env" not in manifest and manifest["minimum_gpu_memory_gb"] == 48
        assert "provider_runtime/native_task_packet/team_embodiment/robot.usd" in archive.namelist()
        assert (
            "provider_runtime/blueprint_pipeline/_catalog_data/trlc_pi05_release.v1.json"
            in archive.namelist()
        )
        assert b"trlc_pi05_provisioning" in archive.read(
            f"provider_runtime/adp009d_policy_provisioning.{CANDIDATE_ID}.sh"
        )
        archive.extractall(tmp_path / "extracted")
    # Exercise the extracted package, not imports from this controller checkout.
    # Only tokenizer bytes are fixture substitutions; its production inventory,
    # notices, source and dependency manifests remain the actual sealed files.
    from blueprint_pipeline.trlc_pi05_assets import tokenizer_inventory

    fixture_inventory = tmp_path / "fixture-tokenizer.json"
    fixture_inventory.write_text(json.dumps(tokenizer_inventory()))
    script = """
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);sys.path.insert(0,str(root))
from blueprint_pipeline import trlc_pi05_assets as assets,trlc_pi05_bundle as bundle,trlc_pi05_source as source
from blueprint_pipeline.native_g1_beta_dependency_profile import artifact_manifest,verify_wheelhouse
assert pathlib.Path(bundle.__file__).is_relative_to(root)
contract=bundle.runtime_contract()
assert all((root/name).is_file() for name in contract['required_entries'])
assert len(artifact_manifest()['wheels'])==96
try: verify_wheelhouse({})
except ValueError as exc: assert str(exc)=='g1_beta_dependency_wheel_set_mismatch'
else: raise AssertionError('missing wheel closure accepted')
assert assets.tokenizer_inventory()['source_revision']=='35e4f46485b4d07967e7e9935bc3786aad50687c'
for row in bundle.notices_inventory()['files']:
    raw=(root/'trlc_notices'/row['path']).read_bytes()
    assert len(raw)==row['size_bytes'] and 'sha256:'+hashlib.sha256(raw).hexdigest()==row['sha256']
assets.tokenizer_inventory=lambda:json.loads(pathlib.Path(sys.argv[2]).read_text())
assert assets.verify_tokenizer_directory(root/'trlc_tokenizer')['tokenizer_bytes_verified']
source.extract_source(root/source.SOURCE_ARCHIVE,root/source.SOURCE_MANIFEST,root/'verified-publisher')
(root/'trlc_tokenizer/added_tokens.json').write_text('{}')
try: assets.verify_tokenizer_directory(root/'trlc_tokenizer')
except ValueError: pass
else: raise AssertionError('unverified tokenizer sidecar accepted')
"""
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            script,
            str(tmp_path / "extracted/provider_runtime"),
            str(fixture_inventory),
        ],
        cwd=tmp_path,
        check=True,
        timeout=30,
    )
    download, upload = transfer_byte_budget(bundle)
    assert download > 7_473_120_997 + 64 * 1024**3 and upload == 8_000_000_000
    preflight = _blueprint_bundle_preflight(
        job_dir=tmp_path / "transport-preflight",
        generated_at="fixture",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="native_task_arena_policy_canary_session",
        bundle_path=Path(bundle["bundle_path"]),
        provider_bundle_url="https://storage.example.com/bundle.zip",
        provider_output_put_url="https://storage.example.com/output.zip",
    )
    assert preflight["status"] == "passed", preflight.get("blockers")
    observed = []
    monkeypatch.setenv("BLUEPRINT_POLICY_CANARY_OUTPUT_DELIVERY", "download")
    monkeypatch.setattr(
        transport,
        "run_arena_native_control_vast",
        lambda **kwargs: observed.append(kwargs) or {"status": "dry_run_ready"},
    )
    authority = launch._read(Path(result["authority"]["path"]))
    transport.run_native_task_arena_policy_canary_session_vast(
        job_dir=tmp_path / "transport",
        prepared_bundle=bundle,
        session_authority=authority,
        paid_resource_admission_grant=None,
        execute=False,
        hard_cap_usd=float(authority["hard_cap_usd"]),
        hard_ttl_seconds=authority["hard_ttl_seconds"],
    )
    assert len(observed) == 1 and observed[0]["expected_provider_download_bytes"] == download
    assert observed[0]["expected_provider_upload_bytes"] == upload
    assert observed[0]["min_gpu_ram_mb"] == 48_000
    assert (
        observed[0]["forward_hf_token"] is False
        and observed[0]["stale_offer_create_retry_limit"] == 0
    )
    assert observed[0]["require_independent_watchdog"] is True
    assert not (tmp_path / "attempt/allocator_launch_intent.json").exists()


@pytest.mark.parametrize(
    "fault", ["body", "cadence", "frames", "checkpoint", "chunk", "rights", "memory"]
)
def test_invalid_local_binding_keeps_individual_reason_and_frozen_candidates(
    tmp_path, monkeypatch, fault
):
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
        assert (
            next(
                row
                for row in catalog["candidates"]
                if row["candidate_id"] == original["candidate_id"]
            )
            == original
        )


def test_rig_override_preserves_prepared_mounts_and_explicit_assumptions(tmp_path, monkeypatch):
    _, _, registration = prepared_registration(tmp_path, monkeypatch)
    manifest = launch._read(launch._bound_path(registration["manifest"]))
    binding = policy_binding(registration["policy_spec"])
    assert "not_publisher_or_site_calibration" in binding.camera_binding_reference
    assert "out_of_distribution" in binding.camera_binding_reference
    assert [(c["width"], c["height"]) for c in manifest["cameras"]] == [
        (800, 600),
        (960, 540),
        (960, 540),
    ]
    changed = deepcopy(manifest)
    changed.pop("embodiment_digest")
    changed["cameras"][0]["height"] = 176
    with pytest.raises(ValueError, match="robot_binding_invalid"):
        build_policy_spec(validate_embodiment_manifest(changed))


def test_actual_native_loop_holds_trlc_joint_rows_twice_and_requeries_after_50(
    tmp_path, monkeypatch
):
    """Real client and native episode loop; fake physics is not GPU evidence."""
    import numpy as np
    from types import SimpleNamespace
    from blueprint_pipeline import (
        adp_task_scoring,
        episode_visual_evidence,
        native_task_episode_environment,
    )
    from blueprint_pipeline.trlc_pi05_policy import TRLCPi05PolicyClient
    from blueprint_pipeline.trlc_pi05_policy_server import TRLCPi05Server
    from blueprint_pipeline.team_embodiment import (
        apply_team_embodiment,
        read_model_structure,
        team_embodiment_binding,
        verify_model_file,
    )
    from blueprint_pipeline.team_embodiment_runtime import run_team_episode
    from tests.test_trlc_pi05_policy import Backend, Connection
    from tests.test_team_arm_kinematics import CartesianRobot
    from tests.test_team_embodiment import _Array, _Camera, _Scoring, station_plan

    _, _, registration = prepared_registration(tmp_path, monkeypatch)
    declared = launch._read(launch._bound_path(registration["manifest"]))
    model = launch._bound_path(registration["model"])
    bound = team_embodiment_binding(
        declared,
        model_record=verify_model_file(declared, model),
        structure=read_model_structure(model),
        scene_plan=station_plan(),
    )
    plan = {**apply_team_embodiment(station_plan(), bound), "scenario": {"seed": 7}}
    plan["cadence"].update(maximum_action_steps=102, settle_window_samples=0)
    row = plan["robot"]

    class Env:
        def __init__(self):
            self.robot, self.steps = CartesianRobot(row), []
            cameras = {}
            for i, name in enumerate(
                ("team_camera_0", "team_camera_1", "team_camera_2", "external_camera_2")
            ):
                camera = _Camera()
                camera.data.output["rgb"] = _Array(
                    np.full(
                        (1, 600 if i == 0 else 540, 800 if i == 0 else 960, 3), 90, dtype=np.uint8
                    )
                )
                cameras[name] = camera
            self.unwrapped = SimpleNamespace(scene={"robot": self.robot, **cameras}, device="cpu")

        def reset(self, seed):
            self.robot.q = np.array(
                [row["joint_reset_positions_rad"][name] for name in self.robot.joint_names]
            )

        def step(self, action):
            order = [*row["joint_names"], *(joint["name"] for joint in row["gripper_joints"])]
            self.steps.append(list(action[0]))
            self.robot.q = np.array(
                [action[0][order.index(name)] for name in self.robot.joint_names]
            )

    env, backend, requests = Env(), Backend(), []
    backend.action[:, :6] = 0.01
    backend.action[:, 7:13] = -0.01
    backend.action[:, 6], backend.action[:, 13] = 0.0, 1.0
    binding = policy_binding(registration["policy_spec"])
    server = TRLCPi05Server(
        backend=backend,
        binding=binding,
        checkpoint_receipt={"checkpoint_bytes_verified": True, "strict_checkpoint_loaded": True},
    )
    client = TRLCPi05PolicyClient(
        binding=binding, connection_factory=lambda *_, **__: Connection(server, requests)
    )
    built = SimpleNamespace(
        env=env,
        camera_scene_names={
            "team_camera_0": "team_camera_0",
            "team_camera_1": "team_camera_1",
            "team_camera_2": "team_camera_2",
            "overview": "external_camera_2",
        },
    )
    monkeypatch.setattr(native_task_episode_environment, "NativeRigidScoringEnvironment", _Scoring)
    monkeypatch.setattr(
        episode_visual_evidence,
        "finalize_multicamera_visual_evidence",
        lambda **_: ({"status": "complete"}, []),
    )
    monkeypatch.setattr(
        adp_task_scoring, "score_task_episode_from_spec", lambda **_: {"task_succeeded": False}
    )
    result = run_team_episode(
        built=built,
        task_readback=None,
        policy=client,
        plan=plan,
        prompt="Move the object",
        max_policy_queries=2,
        open_loop_horizon=50,
        media_root=tmp_path / "episodes",
        episode_id="development-only-trlc",
        progress={},
        to_action=lambda targets: [targets],
    )
    assert len(backend.calls) == 2 and len(env.steps) == 102
    assert backend.calls[1]["state"][:6] == pytest.approx([0.01] * 6)
    assert backend.calls[1]["state"][7:13] == pytest.approx([-0.01] * 6)
    assert env.steps[-1][-4:] == pytest.approx([0.0, 0.0, -0.045, -0.045])
    assert result["policy_request_evidence_complete"] is True
    assert result["policy_cadence"]["native_hold_ticks_per_action"] == 2
    assert result["score"]["task_succeeded"] is False
    assert client.evidence_summary()["native_execution_validated"] is False
