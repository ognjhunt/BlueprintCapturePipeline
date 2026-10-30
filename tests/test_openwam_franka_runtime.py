"""Real routing/bundle software with source-named development USD; no native proof."""

from copy import deepcopy
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
from blueprint_pipeline.openwam_franka_runtime import (
    CANDIDATE_ID,
    EMBODIMENT_ID,
    MINIMUM_GPU_MEMORY_GB,
    build_policy_spec,
    configure_prepared_inputs,
    policy_binding,
)
from blueprint_pipeline.team_candidate_quick10_registry import profile_for_capture
from blueprint_pipeline.team_embodiment import validate_embodiment_manifest
from tests.test_beta_policy_screening_launch import _plan


def development_model(tmp_path):
    from pxr import Usd, UsdPhysics

    fields = json.loads(
        (Path(__file__).parent / "fixtures/openwam_franka/manifest-fields.json").read_text()
    )["manifest_fields"]
    model = tmp_path / "development-only-fr3-robotiq.usda"
    stage = Usd.Stage.CreateNew(str(model))
    stage.SetDefaultPrim(stage.DefinePrim("/robot", "Xform"))
    names = {arm[key] for arm in fields["arms"] for key in ("base_link", "tool_link")}
    names.update(link for grip in fields["grippers"] for link in grip["contact_links"])
    names.update(camera["mount_link"] for camera in fields["cameras"] if camera.get("mount_link"))
    for name in sorted(names):
        UsdPhysics.RigidBodyAPI.Apply(stage.DefinePrim("/robot/" + name, "Xform"))
    for joint in fields["joints"]:
        UsdPhysics.RevoluteJoint.Define(stage, "/robot/" + joint["name"])
    for channel in fields["grippers"]:
        for joint in channel["joints"]:
            UsdPhysics.RevoluteJoint.Define(stage, "/robot/" + joint["name"])
    UsdPhysics.FixedJoint.Define(stage, "/robot/anchor").CreateBody1Rel().SetTargets(
        ["/robot/fr3_link0"]
    )
    stage.GetRootLayer().Save()
    manifest = validate_embodiment_manifest(
        {
            **fields,
            "model": {
                "format": "usd",
                "url": "https://assets.example.invalid/development-only.usda",
                "sha256": launch._record(model)["sha256"],
            },
        }
    )
    return model, manifest


def prepared_registration(tmp_path, monkeypatch):
    base, _ = _plan(tmp_path, monkeypatch, scene_half="team_scene")
    profile = profile_for_capture(str(tmp_path / "capture"))
    model, original = development_model(tmp_path)
    manifest = configure_prepared_inputs({"robot_embodiment": original})["robot_embodiment"]
    manifest_path = launch._write(tmp_path / "development-only-fr3-manifest.json", manifest)
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
            "policy_route_id": "customer_controller",
            "companies": [],
            "relationship": "public_controller_integration",
            "evidence_url": "https://huggingface.co/OpenWAM/OpenWAM-Alpha-Real-Single-Arm-Franka",
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
    from blueprint_pipeline.beta_robolab_policy_execution import (
        validate_robolab_policy_execution_spec,
    )
    from blueprint_pipeline.policy_canary_session_bundle_identity import (
        validate_policy_canary_provider_manifest,
    )
    from blueprint_pipeline.openwam_franka_bundle import transfer_byte_budget
    from blueprint_pipeline.team_candidate_policy_client import canary_policy_client
    from blueprint_pipeline import openwam_piper_client_dependencies as client_dependencies
    from blueprint_pipeline import native_task_arena_vast as transport
    from blueprint_pipeline.vast_provider_adapter import _blueprint_bundle_preflight

    base, profile, _ = prepared_registration(tmp_path, monkeypatch)
    catalog = catalog_for_profile(profile)
    assert len(catalog["candidates"]) == len(load_default_catalog()["candidates"])
    row = next(row for row in catalog["candidates"] if row["candidate_id"] == CANDIDATE_ID)
    assert row["eligibility"]["enabled"] and row["eligibility"]["requires_credentials"] == []
    assert row["deployment"]["policy_route_id"] == "customer_controller"
    planning = build_planning_inputs(
        catalog=catalog,
        profile=profile,
        capture_root=tmp_path / "capture",
        source=base["source"],
        inputs_dir=tmp_path / "openwam-plan",
    )
    plan = build_screening_plan(
        catalog,
        planning["scene"],
        capabilities=planning["capabilities"],
        cost_quotes=planning["cost_quotes"],
        campaign_id="openwam-fixture",
    )
    candidate = next(row for row in plan["candidates"] if row["candidate_id"] == CANDIDATE_ID)
    assert candidate["status"] == "selected", candidate["blockers"]
    spec = launch._read(launch._bound_path(candidate["execution_spec"]["policy_execution_spec"]))
    assert validate_robolab_policy_execution_spec(spec, candidate_id=CANDIDATE_ID) == spec
    altered = deepcopy(spec)
    altered["policy_spec"]["source_revision"] = "0" * 40
    altered["execution_spec_digest"] = canonical_digest(
        altered, digest_field="execution_spec_digest"
    )
    with pytest.raises(ValueError, match="openwam_runtime_policy_identity_invalid"):
        validate_robolab_policy_execution_spec(altered, candidate_id=CANDIDATE_ID)
    dependency_activations = []
    monkeypatch.setattr(
        client_dependencies,
        "activate_client_dependencies",
        lambda: dependency_activations.append(True),
    )
    client = canary_policy_client(spec)
    assert dependency_activations == [True]
    assert client.require_named_arm_state and client.action_chunk_rows == 1
    assert client.binding.execution_horizon == 8
    assert client.evidence_summary()["execution_binding_digest"] == spec["execution_spec_digest"]
    result = launch.prepare_candidate(
        plan=plan, candidate=candidate, attempt_dir=tmp_path / "attempt"
    )
    bundle = launch._read(Path(result["bundle_receipt"]["path"]))
    with zipfile.ZipFile(bundle["bundle_path"]) as archive:
        manifest = validate_policy_canary_provider_manifest(
            json.loads(archive.read("provider_runtime/adp_arena_provider_manifest.json"))
        )
        assert manifest["team_policy_interface_id"] == "openwam_franka_eef10.v1"
        assert "team_credential_env" not in manifest and manifest["minimum_gpu_memory_gb"] == 80
        assert "provider_runtime/native_task_packet/team_embodiment/robot.usd" in archive.namelist()
        assert "provider_runtime/openwam-piper-source.v1.json" in archive.namelist()
        assert b"openwam_franka_provisioning" in archive.read(
            f"provider_runtime/adp009d_policy_provisioning.{CANDIDATE_ID}.sh"
        )
    download, upload = transfer_byte_budget(bundle)
    assert download > 24_835_000_000 + 64 * 1024**3 and upload == 8_000_000_000
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
    assert observed[0]["min_gpu_ram_mb"] == 80_000
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
        policy["action_hz"] = 30
    elif fault == "frames":
        policy["binding"]["native_tcp_from_policy_tcp"] = {}
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
            continue
        assert (
            next(
                row
                for row in catalog["candidates"]
                if row["candidate_id"] == original["candidate_id"]
            )
            == original
        )


def test_configuration_changes_only_native_rate_rows_and_real_view_dimensions(tmp_path):
    _, original = development_model(tmp_path)
    prepared = {
        "robot_embodiment": original,
        "simulation_assumptions": ["development_only"],
        "source_import_provenance": {"unchanged": True},
    }
    configured = configure_prepared_inputs(prepared)
    changed = configured["robot_embodiment"]
    assert changed["control_hz"] == 60 and changed["action_chunk_rows"] == 1
    for before, after in zip(original["cameras"], changed["cameras"], strict=True):
        assert (after["width"], after["height"]) == (640, 480)
        assert {k: v for k, v in before.items() if k not in {"width", "height"}} == {
            k: v for k, v in after.items() if k not in {"width", "height"}
        }
    restored = deepcopy(changed)
    for key in ("control_hz", "action_chunk_rows", "cameras", "embodiment_digest"):
        restored[key] = original[key]
    assert restored == original and prepared["robot_embodiment"] == original
    assert configured["source_import_provenance"] == prepared["source_import_provenance"]
    spec = build_policy_spec(changed)
    assert spec["action_hz"] == 15 and spec["binding"]["execution_horizon"] == 8
    assert (
        "publisher_SKU_mount_TCP_calibration_unknown" in spec["binding"]["frame_binding_reference"]
    )


def test_actual_native_named_arm_loop_holds_15hz_eef_target_four_ticks(tmp_path, monkeypatch):
    """Real protocol and native loop, explicitly fake Cartesian physics and false scorer."""
    import numpy as np
    from types import SimpleNamespace
    from blueprint_pipeline import (
        adp_task_scoring,
        episode_visual_evidence,
        native_task_episode_environment,
    )
    from blueprint_pipeline.openwam_franka_policy import OpenWAMFrankaPolicyClient
    from blueprint_pipeline.openwam_franka_policy_server import OpenWAMFrankaServer
    from blueprint_pipeline.team_embodiment import (
        apply_team_embodiment,
        read_model_structure,
        team_embodiment_binding,
        verify_model_file,
    )
    from blueprint_pipeline.team_embodiment_runtime import run_team_episode
    from tests.test_openwam_piper_policy import Backend, Socket
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
    plan["cadence"].update(maximum_action_steps=12, settle_window_samples=0)
    row = plan["robot"]

    class Robot(CartesianRobot):
        def __init__(self):
            super().__init__(row)
            self.body_names = ["fr3_link0", "robotiq_85_base_link"]

        def get_jacobians(self):
            value = np.zeros((1, 1, 6, len(self.joint_names)))
            for axis, joint in enumerate(row["arms"][0]["joint_names"][:6]):
                value[0, 0, axis, self.joint_names.index(joint)] = 1.0
            return value

    class Env:
        def __init__(self):
            self.robot, self.steps = Robot(), []
            cameras = {}
            for name in ("team_camera_0", "team_camera_1", "external_camera_2"):
                camera = _Camera()
                camera.data.output["rgb"] = _Array(np.full((1, 480, 640, 3), 77, dtype=np.uint8))
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

    env = Env()
    backend = Backend(action=[0.01, 0.02, 0.03, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0])
    binding = policy_binding(registration["policy_spec"])
    server = OpenWAMFrankaServer(
        backend=backend, binding=binding, checkpoint_receipt={"checkpoint_bytes_verified": True}
    )
    client = OpenWAMFrankaPolicyClient(
        binding=binding, socket_factory=lambda *_a, **_k: Socket(server)
    )
    built = SimpleNamespace(
        env=env,
        camera_scene_names={
            "team_camera_0": "team_camera_0",
            "team_camera_1": "team_camera_1",
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
        max_policy_queries=3,
        open_loop_horizon=1,
        media_root=tmp_path / "episodes",
        episode_id="development-only-franka",
        progress={},
        to_action=lambda targets: [targets],
    )
    assert len(backend.calls) == 3 and len(env.steps) == 12
    assert all(
        len(call["state"]) == 10 and call["prompt"] == "Move the object" for call in backend.calls
    )
    assert result["policy_cadence"]["native_hold_ticks_per_action"] == 4
    assert result["policy_request_evidence_complete"] is True
    assert result["score"]["task_succeeded"] is False
    assert client.evidence_summary()["native_execution_validated"] is False
    expected = [joint["closed"] for joint in row["gripper_joints"]]
    assert env.steps[-1][7:] == pytest.approx(expected)


def test_provisioning_uses_only_fixed_franka_files_shared_dependency_budget_and_deadline(
    tmp_path, monkeypatch
):
    import time
    from blueprint_pipeline import openwam_franka_provisioning as provisioner
    from blueprint_pipeline import openwam_piper_provisioning as shared
    from blueprint_pipeline import openwam_piper_dependencies as dependencies
    from blueprint_pipeline import openwam_piper_client_dependencies as client_deps
    from blueprint_pipeline.openwam_franka_assets import EXPECTED_FILES
    from blueprint_pipeline.openwam_franka_policy import MODEL_ID, MODEL_REVISION

    _, _, registration = prepared_registration(tmp_path, monkeypatch)
    # The routing test seals a fully validated execution spec. This test isolates
    # bootstrap transfer identity and performs no downloader or subprocess call.
    spec = {"policy_spec": registration["policy_spec"], "candidate_id": CANDIDATE_ID}
    monkeypatch.setattr(provisioner, "validate_execution_spec", lambda value: value)
    dependency = {
        "interpreter": {
            "filename": "development-only-python.tar",
            "python_relative_path": "bin/python",
            "version": "3.12.12",
            "maximum_expanded_bytes": 1000,
            "size_bytes": 2,
            "sha256": "development-only",
            "url": "https://example.invalid/python",
        },
        "server_wheels": [],
        "thin_client_wheels": [],
        "total_download_bytes": 2,
    }
    monkeypatch.setattr(dependencies, "verified_dependency_manifest", lambda _: dependency)
    monkeypatch.setattr(
        shared, "extract_source", lambda archive, manifest, destination: destination
    )
    monkeypatch.setattr(shared, "_extract_interpreter", lambda *_a, **_k: None)
    commands = []
    monkeypatch.setattr(shared, "_command", lambda argv, **kwargs: commands.append(argv))
    monkeypatch.setattr(shared, "_install_wheels", lambda *_a, **_k: None)
    monkeypatch.setattr(client_deps, "activate_client_dependencies", lambda *_: None)
    transfers = []

    def fetch(row, destination, *, deadline, budget):
        transfers.append((row, destination, deadline, budget.maximum_bytes))
        return {"development_only": True}

    monkeypatch.setattr(shared, "fetch_exact", fetch)
    verified = []
    monkeypatch.setattr(
        provisioner,
        "verify_checkpoint_directory",
        lambda path, **kwargs: verified.append((path, kwargs)),
    )
    started = []

    def start(**kwargs):
        started.append(kwargs)
        return {"status": "ready", "development_only": True}

    monkeypatch.setattr(shared, "_start_server", start)
    deadline = time.time() + 60
    monkeypatch.setenv(shared.DEADLINE_ENV, str(deadline))
    root = tmp_path / "bootstrap"
    (tmp_path / "runtime").mkdir()
    result = provisioner.provision(tmp_path / "runtime", tmp_path / "out", spec, root=root)
    assert len(transfers) == 1 + len(EXPECTED_FILES)
    expected_budget = 2 + sum(size for size, _ in EXPECTED_FILES.values())
    for row, destination, bound, budget in transfers[1:]:
        name = destination.relative_to(root / "checkpoint").as_posix()
        size, digest = EXPECTED_FILES[name]
        assert row == {
            "url": f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}?download=true",
            "size_bytes": size,
            "sha256": "sha256:" + digest,
        }
        assert bound == deadline and budget == expected_budget
    assert verified == [(root / "checkpoint", {"deadline_epoch": deadline})]
    assert started[0]["_profile"].server_module == "blueprint_pipeline.openwam_franka_policy_server"
    assert (
        result["maximum_download_bytes"] == expected_budget
        and result["peak_gpu_memory_measured"] is False
    )
    assert len(commands) == 1
