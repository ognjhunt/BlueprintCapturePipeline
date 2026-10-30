"""Real stdlib/CAS boundaries on synthetic development-only G1 source records."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

from tests.test_operator_door_public_policy_registration import (
    ROOT, PROFILE_ROOT, write, DoorConfig, CommandResult, enqueue, process_spool,
    validate_request, required_scope, RequestRefused,
)
from operator_door import native_g1_registration_contract as contract
from operator_door import native_g1_registration as installer
from blueprint_pipeline import native_g1_registration_install as builder
from blueprint_pipeline import team_candidate_quick10_registry as registry_module

SHA = "sha256:" + "a" * 64
ASSET_DATA = b"# synthetic development collision source\n"
DP, PI, DP_NAV, PI_NAV = contract.CANDIDATES


def seal(value, field):
    value[field] = contract.common.digest(value, field)
    return value


def ref(path):
    return {key: value for key, value in contract.file_record(path).items() if key != "path"}


def request(**changes):
    value = {"schema_version": contract.SCHEMA, "installation_id": "development-g1", "release_commit": "a" * 40,
        "site_id": "site-development", "source": {"capture_id": "development-capture", "task_id": "development-task",
            "scene_intent_id": "development-intent", "scene_intent_digest": SHA, "task_context_digest": SHA,
            "scene_revision_digest": SHA},
        "publisher_source": {"sha256": SHA, "size_bytes": 1}, "runtime_source": {"sha256": SHA, "size_bytes": 1},
        "shared_registry_digest": SHA, "expected_registry_digest": SHA,
        "candidates": [{"candidate_id": key, "scene_packet": None, "navigation_authority": None} for key in contract.CANDIDATES],
        "authorization_reference": "development_only user requested feasible internal simulator testing"}
    value.update(changes)
    return seal(value, "request_digest")


def make_packet(root, *, scene, request_value, source_bindings=None, robot_binding=None):
    root.mkdir(parents=True)
    for row in source_bindings or []:
        (root / row["staged_relative_path"]).write_bytes(ASSET_DATA)
    write(root / "native_task_arena_scene_plan.v1.json", seal(scene, "plan_digest"))
    write(root / "native_task_arena_packet_request.v1.json", seal(request_value, "request_digest"))
    write(root / "native_task_runtime_contract.v1.json", {"schema_version": "development_only"})
    artifacts = [{"relative_path": name, **ref(root / name)} for name in (
        "native_task_arena_scene_plan.v1.json", "native_task_arena_packet_request.v1.json", "native_task_runtime_contract.v1.json")]
    receipt = seal({"schema_version": "native_task_arena_packet_receipt.v1",
        "arena_scene_plan_digest": scene["plan_digest"], "request_digest": request_value["request_digest"], "artifacts": artifacts, "source_bindings": source_bindings or []}, "receipt_digest")
    if robot_binding:
        path = root / robot_binding["staged_relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(ASSET_DATA)
        receipt["robot_asset_binding"] = robot_binding
        seal(receipt, "receipt_digest")
    path = root / "native_task_arena_packet_receipt.v1.json"
    write(path, receipt)
    return path, receipt


def fixture(tmp_path):
    source = request()["source"]
    capture = tmp_path / "capture"
    context = seal({"schema_version": "website_site_task_context.v1", "scene_id": "site-development",
        "capture_id": source["capture_id"], "capture_rights": {"consent_status": "granted", "consent_revoked": False,
            "consent_scope": ["robot_evaluation"], "derived_scene_generation_allowed": True}}, "context_digest")
    source["task_context_digest"] = context["context_digest"]
    write(capture / "pipeline/website_task_context.json", context)
    site = tmp_path / "halves/site-development"
    task = {"control_frequency_hz": 50, "maximum_action_steps": 100, "maximum_episode_seconds": 2,
            "task_success_contract": {"contract_digest": SHA}, "task_success_contract_digest": SHA,
            "subject_asset_id": "development-subject", "start_pose_world": [0, 0, 0.8, 0, 0, 0, 1],
            "subject_collision_bounds_scoring_frame_m": {"minimum": [-0.1, -0.1, -0.1], "maximum": [0.1, 0.1, 0.1]}}
    scene = {"schema_version": "native_task_arena_scene_plan.v1", "scene_id": "site-development",
        "task_id": source["task_id"], "task_kind": "rigid_pick_place", "task_spec": task}
    base_request = {"schema_version": "native_task_arena_packet_request.v1", "scene_id": "site-development",
        "task_id": source["task_id"], "assets": [], "task_spec": task,
        "cameras": [{"role": "overview", "pose_frame": "world", "policy_input": False}]}
    bindings = [{"semantic_role": "scene_collision", "staged_relative_path": "collision.usda",
        "staged_sha256": "sha256:" + hashlib.sha256(ASSET_DATA).hexdigest(), "staged_size_bytes": len(ASSET_DATA)}]
    base_path, base_receipt = make_packet(site / "base", scene=deepcopy(scene), request_value=deepcopy(base_request), source_bindings=bindings)
    inputs = seal({"schema_version": "blueprint.team_scene_inputs.v1", "scene_revision_digest": SHA,
        "base_native_packet": contract.file_record(base_path), "task_success_contract": task["task_success_contract"],
        "task_success_contract_digest": SHA}, "inputs_digest")
    write(site / "team_scene_inputs.v1.json", inputs)
    profile = json.loads((PROFILE_ROOT / "native_g1_registration_profile.json").read_text())
    profile["runtime_wheels"] = []
    profile["robot_usd_sha256"] = "sha256:" + hashlib.sha256(ASSET_DATA).hexdigest()
    profile["robot_usd_size_bytes"] = len(ASSET_DATA)
    profile["robot_template"]["usd_sha256"] = profile["robot_usd_sha256"]
    publisher_root = tmp_path / "shared/publisher"
    source_file = publisher_root / "source/development.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"# synthetic development-only source\n")
    data = source_file.read_bytes()
    profile["publisher_files"] = [{"path": "development.py", "size_bytes": len(data), "git_blob_sha1":
        hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()}]
    profile_root = tmp_path / "installed"
    write(profile_root / "native_g1_registration_profile.json", profile)
    pose = {"position_world_m": [0.1, 0.2, 0.79], "orientation_xyzw": [0, 0, 0, 1]}
    robot = {**deepcopy(profile["robot_template"]), "base_pose_world": pose, "usd_path": "assets/robot_unitree_g1.usd",
        "usd_size_bytes": profile["robot_usd_size_bytes"], "usd_dependency_bindings": [], "usd_runtime_asset_dependencies": ["OmniPBR.mdl"]}
    scene.update(robot=robot, cameras=[profile["head_camera"]])
    stance = seal({"schema_version": "blueprint.team_free_base_stance.v1", "floor_z_m": 0,
        **pose, "yaw_rad": 0, "footprint_half_extent_m": [0.12, 0.23], "clear_stances": 1,
        "search_radius_m": 1.6, "grid_step_m": 0.05, "scene_collision_sha256": bindings[0]["staged_sha256"],
        "validator": "scene_placement.validation.validate_stand_pose"}, "stance_digest")
    assumptions = seal({"recipe_id": "native_g1_beta_retained_scene.v1", "current_task_source": source, "site_id": "site-development",
        "source_packet_receipt_digest": base_receipt["receipt_digest"], "source_scene_plan_digest": base_receipt["arena_scene_plan_digest"],
        "robot_usd_sha256": profile["robot_usd_sha256"], "task_success_contract_digest": SHA, "stance": stance,
        "subject_bounds_world_m": {"minimum": [-0.1, -0.1, 0.8 - 0.1], "maximum": [0.1, 0.1, 0.8 + 0.1]},
        "subject_bounds_basis": "retained_unrotated_task_scoring_bounds",
        "stance_method": "existing_scene_collision_floor_and_clear_footprint_search", "stance_is_geometric_estimate": True, "large_collision_bounds_retained": True,
        "measured_robot_calibration_claimed": False, "footprint_half_extent_m": [0.12, 0.23],
        "root_height_above_inferred_floor_m": 0.79, "head_camera": profile["head_camera"],
        "head_camera_calibration_measured": False, "head_camera_basis": "existing_g1_development_mount_simulation_assumption",
        "overview_basis": "unchanged_current_task_review_camera", "task_hand": "right", "task_hand_is_simulation_assumption": True,
        "control_frequency_hz": 50.0, "physics_frequency_hz": 200.0, "source_control_frequency_hz": 50,
        "source_maximum_action_steps": 100, "source_maximum_episode_seconds": 2,
        "maximum_action_steps": 100, "maximum_episode_seconds": 2,
        "navigation_goal_added": False, "source_geometry_changed": False, "policy_queried": False,
        "simulator_executed": False, "physical_outcome_claimed": False, "new_human_review_claimed": False,
        "current_consent_revalidated": False}, "assumptions_digest")
    g1_request = {**base_request, "physics_frequency_hz": 200, "robot_base_pose_world": pose,
        "cameras": [profile["head_camera"], base_request["cameras"][0]],
        "g1_scene_derivation": {"schema_version": "native_g1_beta_scene_packet_derivation.v1", "claim_ceiling": "development_only",
            "source_packet_receipt_digest": base_receipt["receipt_digest"], "source_scene_plan_digest": base_receipt["arena_scene_plan_digest"],
            "assumptions": assumptions}}
    g1_path, _ = make_packet(site / "native-g1" / base_receipt["receipt_digest"][7:] / contract.common.digest(source)[7:],
        scene=scene, request_value=g1_request, source_bindings=bindings,
        robot_binding={"staged_relative_path": "assets/robot_unitree_g1.usd", "staged_sha256": profile["robot_usd_sha256"],
                       "staged_size_bytes": profile["robot_usd_size_bytes"]})
    publisher = seal({"schema_version": "native_g1_publisher_source_stage.v1", "status": "ready",
        "source_repository": "https://github.com/William-wAng618/HumanoidArena", "source_revision": profile["source_revision"],
        "source_root": str(source_file.parent), "inventory_file_sha256": profile["inventory_sha256"],
        "model_weights_included": False, "gpu_allocated": False}, "receipt_digest")
    publisher_path = publisher_root / "native_g1_publisher_source_stage.v1.json"
    write(publisher_path, publisher)
    runtime_root = tmp_path / "shared/runtime"
    runtime_root.mkdir(parents=True)
    repositories = []
    entries = {}
    for name, url, commit, tree in (
        ("isaaclab", "https://github.com/isaac-sim/IsaacLab.git", "ffff603eafc6b74264a5261cc0183d6a65390d78", "2f82f1afb2cfaf6816b328e03c7b3ddc12069658"),
        ("arena", "https://github.com/isaac-sim/IsaacLab-Arena.git", "8b82dca224f2b5af08f339f987613c59ce9cdbaa", "a52514015a8573ac03b6448688bfa61f9cea18a9")):
        name += "/development.py"
        entries[name] = b"# synthetic source\n"
        repositories.append({"repository": url, "commit": commit, "tree": tree,
            "files": [{"archive_path": name, "size_bytes": len(entries[name]), "sha256": "sha256:" + hashlib.sha256(entries[name]).hexdigest()}]})
    manifest = seal({"schema_version": "native_task_runtime_source_manifest.v1", "runtime_profile": "unitree_g1",
        "redistribution_permitted": True, "policy_bytes_included": False, "repositories": repositories, "runtime_dependency_wheels": []}, "manifest_digest")
    packet = runtime_root / "native_task_runtime_sources.zip"
    with zipfile.ZipFile(packet, "w") as archive:
        archive.writestr("native_task_runtime_source_manifest.v1.json", json.dumps(manifest))
        for name, data in entries.items():
            archive.writestr(name, data)
    record = contract.file_record(packet)
    runtime = seal({"schema_version": "native_task_runtime_source_packet.v1", "runtime_profile": "unitree_g1",
        "redistribution_permitted": True, "policy_bytes_included": False, "manifest_digest": manifest["manifest_digest"],
        "packet_path": str(packet), "packet_sha256": record["sha256"], "packet_size_bytes": record["size_bytes"],
        "runtime_dependency_wheels": []}, "receipt_digest")
    runtime_path = runtime_root / "native_task_runtime_source_packet.v1.json"
    write(runtime_path, runtime)
    shared = seal({"schema_version": "native_g1_team_campaign_registry.v1", "bindings": [{
        "scene_id": "interiorgs-841757", "publisher_source_dir": str(publisher_root),
        "runtime_source_receipt_path": str(runtime_path), "manipulation_packet_dir": "/old/task/never-used"}]}, "registry_digest")
    shared_path = tmp_path / "etc/shared.json"
    write(shared_path, shared)
    derived = seal({"schema_version": "blueprint.team_candidate_derived_profile.v1", "capture_root": str(capture),
        "scene_id": "site-development", "task_id": source["task_id"], "scene_revision_digest": SHA,
        "team_scene_inputs_path": str(site / "team_scene_inputs.v1.json")}, "profile_digest")
    write(tmp_path / "derived/site.json", derived)
    registry = seal({"schema_version": "blueprint.team_candidate_quick10_registry.v1", "profiles": [],
        "unrelated": {"keep": True}, "derived_profiles": {"root": str(tmp_path / "derived"),
            "packet_inputs": {"output_root": str(tmp_path / "halves")}, "defaults": {
                "allowed_founder_approvers": ["development-owner"], "max_hourly_rate_usd": 0.8, "open_loop_horizon": 8,
                "screening_candidates": {"existing": {"keep": True}}}}}, "registry_digest")
    registry_path = tmp_path / "etc/registry.json"
    write(registry_path, registry)
    queue = tmp_path / "queue"
    write(queue / "processed/current.json", seal({"schema_version": "blueprint.team_scene_packet_request.v1",
        "intent_id": source["scene_intent_id"], "intent_digest": source["scene_intent_digest"],
        "capture_id": source["capture_id"], "task_id": source["task_id"], "scene_revision_digest": SHA}, "request_digest"))
    req = request(source=source, publisher_source=ref(publisher_path), runtime_source=ref(runtime_path),
        shared_registry_digest=shared["registry_digest"], expected_registry_digest=registry["registry_digest"])
    options = {"registry_path": registry_path, "state_root": tmp_path / "state", "profile_root": profile_root,
        "audit_root": tmp_path / "audit", "shared_registry_path": shared_path, "source_queue": queue}
    builder_options = {k: v for k, v in options.items() if k != "audit_root"}
    prepared = builder.prepare(req, release_commit=req["release_commit"], **builder_options)
    return SimpleNamespace(request=req, options=options, builder_options=builder_options, registry=registry,
        registry_path=registry_path, derived=derived, g1_path=g1_path, prepared=prepared, publisher_file=source_file,
        proposal=tmp_path / "state/development-g1/proposal.json")


def test_root_isolated_entrypoint_imports_without_pipeline_or_site_packages():
    result = subprocess.run([sys.executable, "-I", "-S", str(ROOT /
        "deploy/operator-door/operator_door/native_g1_registration.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--request-json" in result.stdout


def test_request_has_only_fixed_identities_and_no_path_command_rights_inputs():
    req = request()
    assert validate_request({"kind": contract.OPERATION, "request": req})["request"] == req
    assert required_scope(contract.OPERATION) == "operate"
    for change in ({"path": "/tmp/arbitrary"}, {"rights": {"approved": True}}, {"command": "id"},
                   {"candidates": [{"candidate_id": "new-policy", "scene_packet": None, "navigation_authority": None}]}):
        with pytest.raises(RequestRefused):
            validate_request({"kind": contract.OPERATION, "request": request(**change)})


def test_root_reconstructs_site_registration_cas_idempotence_and_per_candidate_reasons(tmp_path):
    retained = fixture(tmp_path)
    assert retained.prepared["registered_count"] == 2
    receipt = installer.apply(retained.request, **retained.options)
    assert receipt["registered_count"] == 2
    assert receipt["execution_authority_granted"] is False and receipt["candidate_policy_queried"] is False
    assert receipt["live_task_api_queried"] is False and receipt["provider_allocations_performed"] == 0
    assert [row["status"] for row in receipt["results"]] == ["registered", "registered", "unavailable", "unavailable"]
    assert installer.apply(retained.request, **retained.options) == receipt
    after = contract.common.read(retained.registry_path)
    scoped = after.pop("site_screening_candidates")["site-development"]
    assert scoped["source"] == retained.request["source"]
    assert set(scoped["screening_candidates"]) == {DP, PI}
    after["registry_digest"] = retained.registry["registry_digest"]
    assert after == retained.registry


@pytest.mark.parametrize("fault", ["proposal_rights", "proposal_path", "source_bytes", "scene_bytes", "registry_cas", "request_identity", "symlink"])
def test_root_rejects_helper_tamper_before_any_registry_mutation(tmp_path, fault):
    retained = fixture(tmp_path)
    if fault.startswith("proposal_"):
        proposal = contract.common.read(retained.proposal)
        if fault == "proposal_rights":
            proposal["execution_authority_granted"] = True
        else:
            proposal["registrations"][DP]["native_g1"]["publisher_source"]["path"] = "/etc/passwd"
        write(retained.proposal, proposal)
    elif fault == "source_bytes":
        retained.publisher_file.write_text("changed")
    elif fault == "scene_bytes":
        path = retained.g1_path.parent / "native_task_arena_scene_plan.v1.json"
        value = contract.common.read(path)
        value["scene_id"] = "site-other"
        write(path, seal(value, "plan_digest"))
    elif fault == "registry_cas":
        updated = {**retained.registry, "other_change": True}
        write(retained.registry_path, seal(updated, "registry_digest"))
    elif fault == "request_identity":
        retained.request["authorization_reference"] = "different instruction"
        seal(retained.request, "request_digest")
    else:
        original = retained.publisher_file.with_suffix(".retained")
        retained.publisher_file.rename(original)
        retained.publisher_file.symlink_to(original)
    before = retained.registry_path.read_bytes()
    with pytest.raises(ValueError):
        installer.apply(retained.request, **retained.options)
    assert retained.registry_path.read_bytes() == before


def test_recovery_after_atomic_replace_before_receipt(tmp_path, monkeypatch):
    retained = fixture(tmp_path)
    original = installer.atomic._write_once
    def interrupted(path, value):
        if path.name == "receipt.json":
            raise OSError("development_only simulated process interruption")
        return original(path, value)
    monkeypatch.setattr(installer.atomic, "_write_once", interrupted)
    with pytest.raises(OSError):
        installer.apply(retained.request, **retained.options)
    after = retained.registry_path.read_bytes()
    monkeypatch.setattr(installer.atomic, "_write_once", original)
    result = installer.apply(retained.request, **retained.options)
    assert result["registered_count"] == 2 and retained.registry_path.read_bytes() == after


def test_site_overlay_preserves_derived_refresh_and_filters_wrong_task(tmp_path):
    retained = fixture(tmp_path)
    installer.apply(retained.request, **retained.options)
    registry = contract.common.read(retained.registry_path)
    profile = {**retained.derived, **registry["derived_profiles"]["defaults"]}
    actual = registry_module._site_screening_profile(registry, profile)
    assert actual["screening_candidates"]["existing"] == {"keep": True}
    assert actual["native_g1_registration_source"] == retained.request["source"]
    assert {DP, PI} <= set(actual["screening_candidates"])
    changed = {**profile, "scene_revision_digest": "sha256:" + "b" * 64}
    assert registry_module._site_screening_profile(registry, changed) == changed
    assert retained.derived == contract.common.read(tmp_path / "derived/site.json")


def test_spool_launch_stays_networkless_and_never_releases_founder_hold(tmp_path):
    config = DoorConfig(state_root=str(tmp_path / "door"))
    for sub in ("pending", "processing", "completed", "results"):
        (Path(config.spool_root) / sub).mkdir(parents=True)
    enqueue(config, {"kind": contract.OPERATION, "request": request()}, requested_by="development-only", operation_key="development-only")
    calls = []
    process_spool(config, runner=SimpleNamespace(run=lambda argv, timeout: calls.append(argv) or CommandResult(0, "", "")))
    launch, = calls
    assert "--property=PrivateNetwork=yes" in launch and "--property=RestrictAddressFamilies=AF_UNIX" in launch
    assert launch[-1].endswith("/door-install-public-piper-policy.sh")
    assert not any("TOKEN=" in row or "KEY=" in row or "dispatcher" in row for row in launch)


def test_service_owned_derived_record_is_valid_for_root_but_shared_registry_is_not(tmp_path, monkeypatch):
    path = tmp_path / "derived.json"
    write(path, {"development_only": True})
    service_uid = path.stat().st_uid
    if service_uid == 0:
        pytest.skip("requires an unprivileged fixture owner")
    monkeypatch.setattr(contract.os, "geteuid", lambda: 0)
    monkeypatch.setattr(contract.pwd, "getpwnam", lambda name: SimpleNamespace(pw_uid=service_uid))
    assert contract.protected(path, service_owned=True) == path
    with pytest.raises(ValueError, match="protected_record_invalid"):
        contract.protected(path)


def test_install_retry_after_registry_replacement_does_not_require_old_cas_again(tmp_path):
    retained = fixture(tmp_path)
    receipt = installer.apply(retained.request, **retained.options)
    result = builder.prepare(retained.request, release_commit=retained.request["release_commit"], **retained.builder_options)
    assert result == retained.prepared
    assert installer.apply(retained.request, **retained.options) == receipt


def test_old_ignored_bytecode_is_never_treated_as_publisher_source(tmp_path):
    retained = fixture(tmp_path)
    cache = retained.publisher_file.parent / "__pycache__/development.cpython-312.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"untrusted historical bytecode is not shipped")
    receipt = installer.apply(retained.request, **retained.options)
    assert receipt["registered_count"] == 2
    extra = retained.publisher_file.parent / "untracked.py"
    extra.write_text("raise RuntimeError('must never execute')")
    records = contract.common.read(retained.proposal)["registrations"][DP]["native_g1"]
    profile = contract.common.read(retained.options["profile_root"] / "native_g1_registration_profile.json")
    with pytest.raises(ValueError, match="publisher_source_inventory_changed"):
        contract.publisher(records["publisher_source"], profile)


def test_installed_profile_pins_actual_scientific_source_robot_and_runtime_closure():
    from blueprint_pipeline.native_g1_beta_authority import CANDIDATE_DIGESTS, SOURCE_REVISION, ROBOT_USD_SHA256
    from blueprint_pipeline.native_task_runtime_source_packet import runtime_dependency_contracts
    profile = contract.common.read(PROFILE_ROOT / "native_g1_registration_profile.json")
    assert profile["candidate_digests"] == CANDIDATE_DIGESTS
    assert profile["source_revision"] == SOURCE_REVISION and profile["robot_usd_sha256"] == ROBOT_USD_SHA256
    assert profile["runtime_wheels"] == list(runtime_dependency_contracts("unitree_g1"))
    assert len(profile["publisher_files"]) == 1347
    assert len({row["path"] for row in profile["publisher_files"]}) == len(profile["publisher_files"])
    assert {"lerobot/scripts/serve_lerobot_vla_http.py", "isaaclab_twist2_g1/action_provider/action_provider_sonic.py"} <= {
        row["path"] for row in profile["publisher_files"]}


def test_real_isolated_root_process_reconstructs_from_records_without_pipeline_import(tmp_path):
    retained = fixture(tmp_path)
    payload = {"request": retained.request, "options": {key: str(value) for key, value in retained.options.items()}}
    path = tmp_path / "isolated-input.json"
    write(path, payload)
    code = """import importlib.util,json,pathlib,sys
spec=importlib.util.spec_from_file_location('installed',sys.argv[1]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
value=json.loads(pathlib.Path(sys.argv[2]).read_text())
result=module.apply(value['request'],**{k:pathlib.Path(v) for k,v in value['options'].items()})
assert not any(k.startswith('blueprint_pipeline') for k in sys.modules)
print(json.dumps(result))
"""
    result = subprocess.run([sys.executable, "-I", "-S", "-c", code,
        str(ROOT / "deploy/operator-door/operator_door/native_g1_registration.py"), str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["registered_count"] == 2


from tests.test_beta_policy_screening_g1_prepare import prepared_scene  # noqa: E402,F401


def test_service_prepares_real_cpu_packet_and_root_independently_accepts_it(prepared_scene, tmp_path, monkeypatch):  # noqa: F811
    """Real USD/stance/materialization with declared synthetic robot identity."""
    from blueprint_pipeline.native_g1_arena_robot_configuration import build_pinned_g1_arena_robot_configuration
    args, _original, inputs = prepared_scene
    base_path = Path(args["base_packet_record"]["path"])
    base_receipt, base_request, base_scene = contract.packet(base_path)
    profile = contract.common.read(PROFILE_ROOT / "native_g1_registration_profile.json")
    asset = Path(args["robot_asset_record"]["path"])
    template = build_pinned_g1_arena_robot_configuration(asset=asset, evidence_root=asset.parent,
        base_pose_world={"position_world_m": [0, 0, 0.79], "orientation_xyzw": [0, 0, 0, 1]}, task_hand="right")
    for key in ("usd_path", "base_pose_world", "asset_source"):
        template.pop(key)
    profile.update(robot_usd_sha256=args["robot_asset_record"]["sha256"],
        robot_usd_size_bytes=args["robot_asset_record"]["size_bytes"], robot_template=template,
        robot_runtime_asset_dependencies=[])
    profile_root = tmp_path / "installed-synthetic"
    write(profile_root / "native_g1_registration_profile.json", profile)
    historical = tmp_path / "historical-model-locator"
    write(historical / "native_task_arena_scene_plan.v1.json", {"robot": {"usd_path": str(asset)},
        "task": "historical task must not be copied", "approval": "historical approval must not be copied"})
    publisher = tmp_path / "shared-source/native_g1_publisher_source_stage.v1.json"
    runtime = tmp_path / "runtime-source/native_task_runtime_source_packet.v1.json"
    write(publisher, {"synthetic_source_reference_only": True})
    write(runtime, {"synthetic_runtime_reference_only": True})
    shared = seal({"schema_version": "native_g1_team_campaign_registry.v1", "bindings": [{
        "publisher_source_dir": str(publisher.parent), "runtime_source_receipt_path": str(runtime),
        "manipulation_packet_dir": str(historical)}]}, "registry_digest")
    shared_path = tmp_path / "shared-registry.json"
    write(shared_path, shared)
    req = request(site_id=args["site_id"], source=args["source"], publisher_source=ref(publisher),
        runtime_source=ref(runtime), shared_registry_digest=shared["registry_digest"])
    # Site/queue ownership is separately verified by the complete installer
    # fixture. Here retain the actual compiled USD packet and exercise the CPU seam.
    monkeypatch.setattr(builder.contract, "site_inputs", lambda *a, **k:
        (args["output_root"], inputs, base_receipt, base_request, base_scene))
    builder._prepare_current_packet(req, {}, profile_root=profile_root,
        shared_registry_path=shared_path, source_queue=tmp_path / "unused")
    packet_path = args["output_root"] / "native-g1" / base_receipt["receipt_digest"][7:] / contract.common.digest(args["source"])[7:] / "native_task_arena_packet_receipt.v1.json"
    assert packet_path.is_file()
    generated_request = (packet_path.parent / "native_task_arena_packet_request.v1.json").read_text()
    assert "historical approval" not in generated_request and "historical task" not in generated_request
    contract.verify_scene(contract.file_record(packet_path), candidate_id=DP, profile=profile,
        source=args["source"], site_id=args["site_id"], inputs=inputs, base_receipt=base_receipt,
        base_request=base_request, base_scene=base_scene)
    before = packet_path.read_bytes()
    builder._prepare_current_packet(req, {}, profile_root=profile_root,
        shared_registry_path=shared_path, source_queue=tmp_path / "unused")
    assert packet_path.read_bytes() == before
