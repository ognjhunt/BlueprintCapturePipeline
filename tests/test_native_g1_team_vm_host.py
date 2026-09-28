"""VM host lifecycle tests; fake Docker never proves GPU or VM qualification."""

from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest

from blueprint_pipeline import native_g1_team_vm_host as host
from blueprint_pipeline.native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest


@pytest.fixture
def rehearsal(tmp_path, monkeypatch, request):
    """Fake Docker/Isaac; actual sockets, JSONL, native leases, score and media."""
    from blueprint_pipeline import native_g1_team_vm_output as output
    from blueprint_pipeline import native_g1_joint_episode_environment as environment
    from blueprint_pipeline import native_task_arena_readback
    from blueprint_pipeline.native_g1_team_policy_jsonl_client import NativeG1TeamPolicyJsonlClient
    from blueprint_pipeline.native_g1_team_policy_conformance import run_g1_team_policy_synthetic_conformance
    from blueprint_pipeline.native_g1_team_runtime_session import NativeG1TeamRuntimeSession, CONFORMANCE_FILENAME
    from blueprint_pipeline.native_g1_team_container_runtime import NativeG1TeamContainerLease
    from blueprint_pipeline.native_g1_team_artifact_runtime import NativeG1TeamArtifactLease
    from blueprint_pipeline.native_g1_team_supervised_episode import run_g1_team_supervised_episode
    from blueprint_pipeline.native_g1_team_paid_output import verify_g1_team_paid_output
    from blueprint_pipeline import native_g1_team_policy_worker as worker
    from blueprint_pipeline import native_g1_team_worker_supervisor as supervisor
    from blueprint_pipeline.native_g1_team_policy_execution_packet import SCHEMA as PACKET_SCHEMA
    from tests.test_native_g1_team_scored_scene_episode import _inputs
    from tests.test_native_g1_team_policy_jsonl_client import _process
    from tests.test_native_g1_team_runtime_session import _action
    from tests.test_native_g1_team_policy_approval import _approval
    from tests.test_native_g1_shared_scene_episode import _Bridge
    from tests.test_native_task_episode_environment import _RigidNativeReadback
    from tests.test_adp_task_scoring import _rigid_v2_spec
    from tests.test_team_policy_delivery_profile import OWNER, _profile
    from blueprint_pipeline import native_g1_team_archive_gpu as archive_gpu
    import hashlib
    import subprocess
    import time

    setup, profile, scene = _inputs(tmp_path, monkeypatch)
    mode = getattr(request, "param", "container")
    if mode == "noncontainer_artifact":
        profile = _profile(setup, {"mode": mode, "artifact_uri": "https://files.example.org/policy.tar.gz",
                                  "artifact_sha256": "sha256:" + hashlib.sha256(b"fixture archive bytes").hexdigest(), "entrypoint": "policy/run.py",
                                  "protocol": "jsonl_observation_action_v1"})
    scene.plan["task_spec"] = {**_rigid_v2_spec(), **scene.plan["task_spec"],
                               "start_pose_world": [1.1, 2.1, 0.8, 0.0, 0.0, 0.0, 1.0]}
    scene.plan["plan_digest"] = canonical_digest(scene.plan, digest_field="plan_digest")
    binding = ({"mode": mode, "profile_digest": profile["profile_digest"], "image_ref": profile["delivery"]["image_ref"],
                "gpu_device": 0} if mode == "container" else
               {"mode": mode, "profile_digest": profile["profile_digest"], "artifact_sha256": profile["delivery"]["artifact_sha256"],
                "staged_artifact_path": str(tmp_path / "operator-staged.tar")})
    approval = _approval(setup, profile, binding)
    approval["expires_at_epoch"] = time.time() + 3600
    approval["approval_digest"] = canonical_digest(approval, digest_field="approval_digest")
    packet = {"schema_version": PACKET_SCHEMA, "request": {"owner": OWNER, "policy_profile": profile,
              "objective_id": "task_success"}, "trusted_setup": setup, "operator_approval": approval}
    packet["packet_digest"] = cross_runtime_canonical_digest(packet, digest_field="packet_digest")
    runtime = tmp_path / "vm/provider_runtime"
    runtime.mkdir(parents=True)
    if mode == "noncontainer_artifact":
        archived = runtime / "inputs/team-policy/policy.tar"
        archived.parent.mkdir(parents=True)
        archived.write_bytes(b"fixture archive bytes")
    root = runtime.parent / "runtime_output"
    manifest = {"scene_plan_digest": scene.plan["plan_digest"], "scene_packet_receipt_digest": "sha256:" + "f" * 64}
    inputs = {"packet": packet, "manifest": manifest}
    monkeypatch.setattr(host, "verify_g1_team_sealed_inputs", lambda path: inputs)
    gpu_binding = None
    if mode == "noncontainer_artifact":
        with monkeypatch.context() as scoped:
            scoped.setattr(archive_gpu, "_gpu_identity", lambda: {"gpu_index": 0, "gpu_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef", "driver_version": "580.65.06"})
            scoped.setattr(archive_gpu, "_uvm_major", lambda: 239)
            scoped.setattr(archive_gpu, "_device_record", lambda path, major, minor: {"path": path, "major": major, "minor": minor, "uid": 0, "mode": 0o666, "inode": minor + 100, "filesystem_device": 5})
            gpu_binding = archive_gpu.observe_archive_gpu_binding(profile_digest=profile["profile_digest"], execution_packet_digest=packet["packet_digest"])
    monkeypatch.setattr(host, "preflight_g1_vm_host", lambda packet: {
        "schema_version": host.PREFLIGHT_SCHEMA, "status": "host_capabilities_observed",
        "execution_packet_digest": packet["packet_digest"], "delivery_mode": mode,
        "simulator_image_ref": NATIVE_TASK_ARENA_IMAGE, "simulator_local_image_id": "sha256:" + "c" * 64,
        "policy_local_image_id": "sha256:" + "d" * 64 if mode == "container" else None,
        "python_abi": "cp312", "numpy_version": "2.3.1", "rfc8785_version": "0.1.4",
        "claim_ceiling": "development_only", "provider_teardown_verified": False,
        "guest_gpu_inference_verified": False, "archive_gpu_device_exposure_verified": False,
        "archive_gpu_device_binding": gpu_binding,
        "archive_sandbox_required_features": list(host.BWRAP_REQUIRED_OPTIONS) if mode == "noncontainer_artifact" else None,
    })
    calls, processes, sessions, config_paths = [], [], [], []
    original_popen = subprocess.Popen
    original_run = subprocess.run
    def docker(command, **kwargs):
        if command[0] != "docker":
            return original_run(command, **kwargs)
        calls.append(command)
        if command[1:3] == ["rm", "--force"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        assert command[1:3] == ["container", "inspect"]
        return SimpleNamespace(returncode=1, stdout="", stderr="Error: No such object: owned")
    monkeypatch.setattr(host.subprocess, "run", docker)
    def policy(**kwargs):
        directory = kwargs["output_dir"]
        directory.mkdir(mode=0o700)
        with monkeypatch.context() as scoped:
            scoped.setattr(subprocess, "Popen", lambda *a, **kw: original_popen(*a, **{**kw, "start_new_session": True}))
            process = _process("{'ok': True, 'action_chunk': " + repr([_action()]) + "}")
        processes.append(process)
        client = NativeG1TeamPolicyJsonlClient(process, timeout_seconds=2, profile_digest=profile["profile_digest"])
        conformance = run_g1_team_policy_synthetic_conformance(profile=profile, trusted_setup=setup,
                                                              authenticated_owner=OWNER, policy_client=client)
        host._write(directory / CONFORMANCE_FILENAME, conformance)
        stderr = (directory / "fake_policy_private_stderr.log").open("wb")
        common = {"process": process, "client": client, "profile_digest": profile["profile_digest"],
                  "output_dir": directory, "stderr_file": stderr}
        if mode == "noncontainer_artifact":
            probe = host._write(directory / archive_gpu.PROBE_FILENAME, {
                "schema_version": archive_gpu.PROBE_SCHEMA, "status": "cuda_device_memory_access_observed",
                "binding_digest": gpu_binding["receipt_digest"],
                "probe_source_sha256": "sha256:" + hashlib.sha256(archive_gpu.CUDA_PROBE.encode()).hexdigest(),
                "guest_gpu_inference_verified": False, "claim_ceiling": "development_only",
                "observed": {"visible_gpu_count": 1, "gpu_uuid_hex": "0123456789abcdef0123456789abcdef", "cuda_driver_api_version": 13000,
                             "memory_roundtrip_verified": True, "memory_freed": True, "context_destroyed": True}})
            common.update(gpu_binding_digest=gpu_binding["receipt_digest"], gpu_probe_digest=probe["receipt_digest"])
        lease = (NativeG1TeamContainerLease(**common, container_name="blueprint-team-policy-" + "f" * 32,
                 image_ref=profile["delivery"]["image_ref"], image_id="sha256:" + "d" * 64) if mode == "container" else
                 NativeG1TeamArtifactLease(**common, artifact_sha256=profile["delivery"]["artifact_sha256"]))
        session = NativeG1TeamRuntimeSession(client=client, conformance=conformance,
            profile_digest=profile["profile_digest"], delivery_mode=mode, output_dir=directory, lease=lease)
        sessions.append(session)
        return session
    monkeypatch.setattr(host, "open_g1_team_runtime_session", policy)
    monkeypatch.setattr(environment, "NativeG1JointEpisodeEnvironment", lambda **kw: scene)
    monkeypatch.setattr(native_task_arena_readback, "NativeRigidTaskArenaReadback", lambda built: _RigidNativeReadback(
        finger_separation_m=0.08, grasp_frame_position_world_m=[1.1, 2.1, 0.9],
        destination_scene_forbidden_contact_peak_force_n=0.0))
    real_runner = supervisor.run_g1_team_worker_process
    def simulator(**kwargs):
        command = kwargs["command"]
        private = next(value.split("=", 1)[1] for value in command if value.startswith(host.POLICY_RELAY_FILE_ENV + "="))
        config_paths.append(Path(private))
        worker_root = root / "selected-worker/worker"
        worker_root.mkdir(parents=True, mode=0o700)
        episode = run_g1_team_supervised_episode(built=type("Built", (), {"plan": scene.plan})(),
            profile=profile, trusted_setup=setup, authenticated_owner=OWNER, operator_approval=approval,
            sonic_bridge=_Bridge(), objective_id="task_success", max_steps=2, output_dir=worker_root / "episode",
            to_tensor=lambda value: value, make_action_tensor=lambda value, **kw: value,
            policy_relay_config_path=Path(private), execution_packet_digest=packet["packet_digest"])
        assert episode["status"] == "completed_development_only", episode
        native_exit = real_runner(command=[sys.executable, "-c", "pass"],
            diagnostics_dir=root / "selected-worker/private_diagnostics", timeout_seconds=5)
        core = {"claim_ceiling": "development_only", "execution_packet_digest": packet["packet_digest"],
                "operator_approval_digest": approval["approval_digest"],
                "scene_packet_receipt_digest": manifest["scene_packet_receipt_digest"],
                "profile_digest": profile["profile_digest"], "objective_id": "task_success", "delivery_mode": mode,
                "supervised_episode_result_digest": episode["result_digest"], "policy_query_count": episode["policy_query_count"],
                "blocker_type": None, "ranking_eligible": False, "provider_teardown_verified": False,
                "official_billing_reconciled": False, "public_redistribution_authorized": False}
        host._write(worker_root / worker.PRECLOSE_FILENAME, {**core, "schema_version": worker.PRECLOSE_SCHEMA,
                    "status": "awaiting_simulator_close", "teardown": {"environment": "closed", "simulator": "close_requested"}},
                    field="preclose_digest")
        host._write(worker_root / worker.FILENAME, {**core, "schema_version": worker.SCHEMA,
                    "status": "completed_development_only", "teardown": {"environment": "closed", "simulator": "closed"}},
                    field="result_digest")
        verified = verify_g1_team_paid_output(output_dir=worker_root, execution_packet=packet,
                                              scene_plan_digest=manifest["scene_plan_digest"],
                                              scene_packet_receipt_digest=manifest["scene_packet_receipt_digest"])
        supervised = host._write(root / "selected-worker" / supervisor.RESULT_FILENAME, {
            "schema_version": supervisor.RESULT_SCHEMA, "status": "completed_development_only",
            "execution_packet_digest": packet["packet_digest"], "verified_output": verified,
            "child_exit_receipt_digest": native_exit["receipt_digest"]}, field="result_digest")
        host._write(root / host.RESULT_FILENAME, {"schema_version": "native_g1_team_provider_result.v1",
                    "status": "completed_development_only", "execution_packet_digest": packet["packet_digest"],
                    "worker_output_relative_path": "selected-worker/worker", "candidate_policy_queried": True,
                    "verified_output": verified, "supervised_result_digest": supervised["result_digest"],
                    "provider_teardown_verified": False, "official_billing_reconciled": False,
                    "public_redistribution_authorized": False}, field="result_digest")
        return real_runner(**{**kwargs, "command": [sys.executable, "-c", "pass"]})
    monkeypatch.setattr(host, "run_g1_team_worker_process", simulator)
    args = {"runtime_root": runtime, "output_dir": root, "timeout_seconds": 5}
    verify_args = {"output_dir": root, "execution_packet": packet, **manifest}
    yield args, verify_args, processes, sessions, calls, config_paths, output
    for session in sessions:
        session.close()


@pytest.mark.slow
@pytest.mark.parametrize("rehearsal", ["container", "noncontainer_artifact"], indirect=True)
def test_real_policy_lifecycle_score_and_media_under_host_supervision(rehearsal):
    args, verify_args, processes, sessions, calls, configs, output = rehearsal
    result = host.run_g1_team_vm_host(**args)
    assert result["status"] == "completed_development_only", (result["stage_reached"], result["blocker_type"], result)
    verified = output.verify_g1_team_vm_host_output(**verify_args)
    assert verified["verified_output"]["score"]["outcome"] == "never_moved"
    assert verified["verified_output"]["policy_query_count"] == 2
    assert set(verified["verified_output"]["media"]["review_videos"]) == {"head", "overview"}
    assert all(process.poll() is not None for process in processes)
    assert all(not config.exists() for config in configs)
    assert any(command[1:3] == ["rm", "--force"] and "blueprint-g1-simulator" in command[-1] for command in calls)
    assert result["provider_teardown_verified"] is False
    assert result["official_billing_reconciled"] is False


@pytest.mark.slow
@pytest.mark.parametrize("rehearsal", ["noncontainer_artifact"], indirect=True)
@pytest.mark.parametrize("fault", ["probe_uuid", "probe_missing", "child_binding", "sandbox_features"])
def test_archive_gpu_proof_is_required_and_cross_bound(rehearsal, fault):
    from blueprint_pipeline.native_g1_team_archive_gpu import PROBE_FILENAME
    args, verify_args, _, _, _, _, output = rehearsal
    assert host.run_g1_team_vm_host(**args)["status"] == "completed_development_only"
    runtime = args["output_dir"] / "policy-host/runtime"
    path = runtime / ("native_g1_team_artifact_teardown.v1.json" if fault == "child_binding" else PROBE_FILENAME)
    if fault == "sandbox_features":
        path = args["output_dir"] / host.PREFLIGHT_FILENAME
    if fault == "probe_missing":
        path.unlink()
    else:
        receipt = json.loads(path.read_text())
        if fault == "probe_uuid":
            receipt["observed"]["gpu_uuid_hex"] = "f" * 32
        elif fault == "sandbox_features":
            receipt.pop("archive_sandbox_required_features")
        else:
            receipt["gpu_binding_digest"] = "sha256:" + "f" * 64
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        output.verify_g1_team_vm_evidence(**verify_args)
    with pytest.raises(ValueError):
        output.verify_g1_team_vm_host_output(**verify_args)


@pytest.mark.slow
@pytest.mark.parametrize("fault", ["nonzero", "timeout", "guest_missing"])
def test_simulator_failure_still_reaps_policy_and_forces_container_removal(rehearsal, monkeypatch, fault):
    args, _, processes, sessions, calls, _, _ = rehearsal
    from blueprint_pipeline.native_g1_team_worker_supervisor import run_g1_team_worker_process
    def failed(**kwargs):
        script = "import sys; sys.exit(7)" if fault == "nonzero" else (
                 "import time; time.sleep(60)" if fault == "timeout" else "pass")
        return run_g1_team_worker_process(**{**kwargs, "command": [sys.executable, "-c", script],
                                             "timeout_seconds": 0.05 if fault == "timeout" else 5})
    monkeypatch.setattr(host, "run_g1_team_worker_process", failed)
    result = host.run_g1_team_vm_host(**args)
    assert result["status"] == "blocked"
    assert all(process.poll() is not None for process in processes)
    assert result["policy_session_closed"] is True
    assert any(command[1:3] == ["rm", "--force"] and "blueprint-g1-simulator" in command[-1] for command in calls)
    assert result["verified_output"] is None


@pytest.mark.slow
@pytest.mark.parametrize("fault", ["child_missing", "child_digest", "simulator_absence", "relay_queries", "guest_session", "worker_exit", "preflight_claim", "preflight_abi"])
def test_changed_host_or_child_evidence_blocks_retained_result(rehearsal, fault):
    args, verify_args, _, _, _, _, output = rehearsal
    result = host.run_g1_team_vm_host(**args)
    assert result["status"] == "completed_development_only", result
    root = args["output_dir"]
    paths = {"child_missing": root / "policy-host/runtime/native_g1_team_container_teardown.v1.json",
             "child_digest": root / "policy-host/runtime/native_g1_team_container_teardown.v1.json",
             "simulator_absence": root / "vm-simulator" / host.SIMULATOR_FILENAME,
             "relay_queries": root / "policy-host" / host.RELAY_FILENAME,
             "guest_session": root / "selected-worker/worker/episode/runtime/native_g1_team_runtime_session.v1.json",
             "worker_exit": root / "selected-worker/private_diagnostics/worker.exit.json",
             "preflight_claim": root / host.PREFLIGHT_FILENAME,
             "preflight_abi": root / host.PREFLIGHT_FILENAME}
    path = paths[fault]
    if fault == "child_missing":
        path.unlink()
    else:
        value = json.loads(path.read_text())
        if fault == "child_digest":
            value["receipt_digest"] = "sha256:" + "a" * 64
        else:
            field, changed = {"simulator_absence": ("container_absent_verified", False),
                              "relay_queries": ("inference_query_count", 1),
                              "guest_session": ("linked_scored_episode_result_digest", "sha256:" + "a" * 64),
                              "worker_exit": ("returncode", 7),
                              "preflight_claim": ("guest_gpu_inference_verified", True),
                              "preflight_abi": ("python_abi", "cp310")}[fault]
            value[field] = changed
            value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        output.verify_g1_team_vm_evidence(**verify_args)
    with pytest.raises(ValueError):
        output.verify_g1_team_vm_host_output(**verify_args)


@pytest.mark.parametrize("fault", [None, "platform", "python", "numpy", "nvidia_runtime", "gpu_driver", "simulator_image", "policy_image", "archive_sandbox"])
def test_host_preflight_observes_exact_abi_driver_and_local_images(monkeypatch, fault):
    monkeypatch.setattr(host.platform, "system", lambda: "Darwin" if fault == "platform" else "Linux")
    monkeypatch.setattr(host.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(host.os, "geteuid", lambda: 0)
    monkeypatch.setattr(host.sys, "version_info", (3, 10) if fault == "python" else (3, 12))
    monkeypatch.setattr(host, "version", lambda name: (
        "1.26.4" if fault == "numpy" and name == "numpy" else {"numpy": "2.3.1", "rfc8785": "0.1.4"}[name]))
    images = []
    def image(reference):
        images.append(reference)
        if ((fault == "simulator_image" and reference == NATIVE_TASK_ARENA_IMAGE)
                or (fault == "policy_image" and reference != NATIVE_TASK_ARENA_IMAGE)):
            raise ValueError("g1_local_image_not_qualified")
        return "sha256:" + "b" * 64
    monkeypatch.setattr(host, "_verified_local_image", image)
    commands = []
    def invoke(command, **kwargs):
        commands.append(command)
        if command[:2] == ["docker", "info"]:
            return SimpleNamespace(returncode=0, stdout='{}' if fault == "nvidia_runtime" else '{"nvidia": {}}')
        if command[0] == "nvidia-smi":
            return SimpleNamespace(returncode=0, stdout="0, 550.54.14" if fault == "gpu_driver" else "0, 580.65.06")
        assert command == ["bwrap", "--version"]
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(host.subprocess, "run", invoke)
    packet = {"packet_digest": "sha256:" + "a" * 64, "request": {"policy_profile": {"delivery": {
        "mode": "noncontainer_artifact" if fault == "archive_sandbox" else "container",
        "image_ref": "example.invalid/team@sha256:" + "c" * 64}}}}
    if fault is not None:
        with pytest.raises(ValueError):
            host.preflight_g1_vm_host(packet)
        if fault in {"platform", "python", "numpy"}:
            assert commands == [] and images == []
    else:
        result = host.preflight_g1_vm_host(packet)
        assert result["python_abi"] == "cp312"
        assert result["guest_gpu_inference_verified"] is False
        assert result["archive_gpu_device_exposure_verified"] is False
        assert result["simulator_image_ref"] == NATIVE_TASK_ARENA_IMAGE


def test_simulator_command_has_exact_image_and_narrow_mounts(tmp_path):
    runtime = tmp_path / "bundle/provider_runtime"
    runtime.mkdir(parents=True)
    output = runtime.parent / "runtime_output"
    output.mkdir()
    relay = tmp_path / "private-relay"
    relay.mkdir(mode=0o700)
    command = host.simulator_container_command(
        runtime_root=runtime, output_dir=output, relay_directory=relay,
        container_name="blueprint-g1-simulator-" + "a" * 32,
    )
    assert command[0:2] == ["docker", "run"]
    assert command[-2:] == [NATIVE_TASK_ARENA_IMAGE, str(runtime / "run_adp_arena_provider_runtime.sh")]
    assert command[command.index("--network") + 1] == "bridge"
    assert "--privileged" not in command and "--ipc=host" not in command
    assert not any("docker.sock" in arg for arg in command)
    mounts = [command[index + 1] for index, value in enumerate(command) if value == "--mount"]
    assert len(mounts) == 4
    assert f"type=bind,source={runtime.parent},target={runtime.parent},readonly" in mounts
    assert f"type=bind,source={relay},target={relay},readonly" in mounts
    assert f"type=bind,source={output},target={output}" in mounts
    assert "--policy-command" not in command


@pytest.mark.parametrize("absent", [True, False])
def test_simulator_cleanup_requires_observed_container_absence(tmp_path, monkeypatch, absent):
    calls = []
    def invoke(command, **kwargs):
        calls.append(command)
        if command[1:3] == ["rm", "--force"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=1 if absent else 0, stdout="[]",
                               stderr="Error: No such object: owned" if absent else "")
    monkeypatch.setattr(host.subprocess, "run", invoke)
    result = host.close_simulator_container(
        container_name="blueprint-g1-simulator-" + "a" * 32,
        image_id="sha256:" + "b" * 64, output_dir=tmp_path,
    )
    assert result["status"] == ("container_removed" if absent else "teardown_blocked")
    assert result["container_absent_verified"] is absent
    assert calls[0][1:3] == ["rm", "--force"]
    assert calls[1][1:3] == ["container", "inspect"]
    assert result["provider_teardown_verified"] is False
