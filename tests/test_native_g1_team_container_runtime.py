"""A team image must be pinned, isolated, probed, and removed after use."""

import io
import json
import subprocess

import pytest

from blueprint_pipeline import native_g1_team_container_runtime as runtime
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_native_g1_team_policy_jsonl_client import _process
from tests.test_team_policy_delivery_profile import OWNER, _profile, _setup


IMAGE = "registry.example.org/team/g1@sha256:" + "a" * 64
IMAGE_ID = "sha256:" + "b" * 64
NAME = "blueprint-team-policy-" + "c" * 32


def _bound_profile():
    setup = _setup()
    profile = _profile(
        setup,
        {"mode": "container", "image_ref": IMAGE, "protocol": "jsonl_observation_action_v1"},
    )
    return setup, profile


def test_container_command_has_fixed_isolation_and_rejects_unpinned_image():
    command = runtime.isolated_container_command(
        image_ref=IMAGE, container_name=NAME, gpu_device=0
    )
    assert command[-1] == IMAGE
    assert command[:3] == ["docker", "run", "--pull"]
    for flag, value in (
        ("--network", "none"), ("--read-only", "--cap-drop"),
        ("--cap-drop", "ALL"), ("--user", "65534:65534"),
        ("--gpus", "device=0"),
    ):
        assert command[command.index(flag) + 1] == value
    assert "--mount" not in command
    assert "--volume" not in command
    assert "--privileged" not in command
    with pytest.raises(ValueError, match="configuration_invalid"):
        runtime.isolated_container_command(
            image_ref="registry.example.org/team/g1:latest", container_name=NAME,
            gpu_device=None,
        )
    with pytest.raises(ValueError, match="configuration_invalid"):
        runtime.isolated_container_command(image_ref=IMAGE, container_name=NAME, gpu_device=True)


def test_local_image_inspection_requires_exact_repository_digest(monkeypatch):
    outputs = iter(
        [
            subprocess.CompletedProcess([], 0, json.dumps([IMAGE]), ""),
            subprocess.CompletedProcess([], 0, IMAGE_ID + "\n", ""),
        ]
    )
    monkeypatch.setattr(runtime.subprocess, "run", lambda *_args, **_kwargs: next(outputs))
    assert runtime._verified_local_image(IMAGE) == IMAGE_ID

    outputs = iter(
        [
            subprocess.CompletedProcess([], 0, "[]", ""),
            subprocess.CompletedProcess([], 0, IMAGE_ID + "\n", ""),
        ]
    )
    with pytest.raises(ValueError, match="local_image_identity_invalid"):
        runtime._verified_local_image(IMAGE)


def test_approved_image_uses_real_jsonl_wire_and_teardown(monkeypatch, tmp_path):
    setup, profile = _bound_profile()
    action = [0.0] * 40
    action[3:9] = [1, 0, 0, 1, 0, 0]
    process = _process("{'ok': True, 'action_chunk': " + repr([action]) + "}")
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[1] == "rm":
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 1, "", "Error: No such object: " + command[-1])

    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setattr(runtime, "_verified_local_image", lambda _image: IMAGE_ID)
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda command, **_kwargs: calls.append(command) or process)
    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    lease, receipt = runtime.launch_g1_team_container_synthetic_probe(
        profile=profile, trusted_setup=setup, authenticated_owner=OWNER,
        operator_approved_profile_digest=profile["profile_digest"],
        operator_approved_image_ref=IMAGE, output_dir=tmp_path / "probe", gpu_device=None,
    )
    assert receipt["synthetic_policy_query_count"] == 1
    assert receipt["site_policy_query_count"] == 0
    assert receipt["task_scored"] is False
    assert calls[0][1] == "run"
    teardown = lease.close()
    assert teardown["status"] == "container_removed"
    assert teardown["container_absent_verified"] is True
    assert teardown["receipt_digest"] == canonical_digest(teardown, digest_field="receipt_digest")
    assert [call[1] for call in calls[1:]] == ["rm", "container"]
    assert lease.close() == teardown
    assert (tmp_path / "probe" / runtime.CONFORMANCE_FILENAME).is_file()


def test_close_attempts_force_removal_even_if_process_terminate_fails(monkeypatch, tmp_path):
    class BrokenProcess:
        def poll(self):
            return None

        def terminate(self):
            raise OSError("local Docker client failed")

    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[1] == "rm":
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 1, "", "No such container: " + NAME)

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    lease = runtime.NativeG1TeamContainerLease(
        process=BrokenProcess(), client=None, container_name=NAME,
        image_ref=IMAGE, image_id=IMAGE_ID, profile_digest="sha256:" + "d" * 64,
        output_dir=tmp_path, stderr_file=io.BytesIO(),
    )
    teardown = lease.close()
    assert teardown["status"] == "container_removed"
    assert teardown["process_close_error"] == "OSError"
    assert [call[1] for call in calls] == ["rm", "container"]


def test_operator_binding_mismatch_blocks_before_image_probe(monkeypatch, tmp_path):
    setup, profile = _bound_profile()
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setattr(
        runtime, "_verified_local_image", lambda _image: pytest.fail("image probed early")
    )
    with pytest.raises(ValueError, match="admission_invalid"):
        runtime.launch_g1_team_container_synthetic_probe(
            profile=profile, trusted_setup=setup, authenticated_owner=OWNER,
            operator_approved_profile_digest="sha256:" + "f" * 64,
            operator_approved_image_ref=IMAGE, output_dir=tmp_path / "blocked",
            gpu_device=None,
        )
    assert not (tmp_path / "blocked").exists()
