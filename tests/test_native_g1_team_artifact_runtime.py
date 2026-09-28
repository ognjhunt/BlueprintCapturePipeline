"""A team archive must be digest-bound and namespace-isolated before probing."""

import hashlib
import io
import os
import stat
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import native_g1_team_artifact_runtime as runtime
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_team_policy_delivery_profile import OWNER, _profile, _setup


def _archive(path: Path, members: list[tuple[str, bytes | None]]) -> str:
    with tarfile.open(path, "w:gz") as stream:
        for name, content in members:
            info = tarfile.TarInfo(name)
            if content is None:
                info.type = tarfile.SYMTYPE
                info.linkname = "/etc/passwd"
                stream.addfile(info)
            else:
                info.size = len(content)
                stream.addfile(info, io.BytesIO(content))
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_profile(digest: str):
    setup = _setup()
    profile = _profile(
        setup,
        {
            "mode": "noncontainer_artifact",
            "artifact_uri": "https://files.example.org/team-g1.tar.gz",
            "artifact_sha256": digest,
            "entrypoint": "policy/run.py",
            "protocol": "jsonl_observation_action_v1",
        },
    )
    return setup, profile


def _sandbox_command(artifact: Path, entrypoint: str, **kwargs):
    descriptor = os.open(artifact, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        return runtime.isolated_artifact_command(
            artifact_root=artifact, entrypoint=entrypoint,
            artifact_directory_fd=descriptor, **kwargs,
        )
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("uid,gid,mode,allowed", [
    (11, 12, 0o700, True), (10, 12, 0o710, True), (10, 13, 0o701, True),
    (10, 13, 0o750, False), (11, 12, 0o600, False), (10, 13, 0o700, False),
    (11, 12, 0o001, False), (10, 12, 0o001, False),
])
def test_namespace_ancestor_access_uses_mapped_ids_without_host_dac_override(uid, gid, mode, allowed):
    observed = SimpleNamespace(st_uid=uid, st_gid=gid, st_mode=stat.S_IFDIR | mode)
    assert runtime._mapped_directory_access(observed, uid=11, gid=12, read=False) is allowed


def test_namespace_artifact_requires_read_and_execute():
    observed = SimpleNamespace(st_uid=11, st_gid=12, st_mode=stat.S_IFDIR | 0o300)
    assert runtime._mapped_directory_access(observed, uid=11, gid=12, read=False) is True
    assert runtime._mapped_directory_access(observed, uid=11, gid=12, read=True) is False


def test_foreign_private_ancestor_is_typed_refusal(monkeypatch, tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    monkeypatch.setattr(runtime, "_mapped_directory_access", lambda *a, **kw: False)
    with pytest.raises(ValueError, match="namespace_path_inaccessible"):
        runtime._require_namespace_artifact_access(artifact)


def test_bwrap_command_exposes_readonly_artifact_without_network(monkeypatch, tmp_path):
    artifact = tmp_path / "artifact"
    (artifact / "policy").mkdir(parents=True)
    (artifact / "policy" / "run.py").write_text("#!/usr/bin/env python3\n")
    monkeypatch.setattr(runtime.shutil, "which", lambda _name: "/usr/bin/bwrap")
    command = _sandbox_command(artifact, "policy/run.py")
    assert command[0:4] == ["/usr/bin/bwrap", "--unshare-all", "--die-with-parent", "--new-session"]
    assert command[command.index("--ro-bind-fd") + 2] == "/work"
    assert str(artifact) not in command
    assert "--bind" not in command
    assert command[command.index("--uid") + 1] == "65534"
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert command[-1] == "/work/policy/run.py"
    assert "--clearenv" in command
    with pytest.raises(ValueError, match="entrypoint_invalid"):
        _sandbox_command(artifact, "../etc/passwd")


def test_gpu_mounts_are_only_bound_nodes_and_cpu_has_no_gpu_devices(monkeypatch, tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "run").write_text("#!/bin/sh\n")
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/usr/bin/bwrap")
    observed = []
    monkeypatch.setattr(runtime, "recheck_archive_gpu_binding", lambda binding: observed.append(binding))
    binding = {"devices": [{"path": path} for path in ("/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm")]}
    command = _sandbox_command(artifact, "run", gpu_binding=binding)
    devices = [command[i + 1:i + 3] for i, item in enumerate(command) if item == "--dev-bind"]
    assert devices == [[row["path"], row["path"]] for row in binding["devices"]]
    assert observed == [binding]
    assert "--bind" not in command and "--share-net" not in command
    assert "--dev-bind" not in _sandbox_command(artifact, "run")


@pytest.mark.parametrize("fault", ["foreign", "file", "closed", "boolean"])
def test_private_archive_binding_requires_its_own_directory_descriptor(monkeypatch, tmp_path, fault):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "run").write_text("#!/bin/sh\n")
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    path = artifact / "run" if fault == "file" else foreign if fault == "foreign" else artifact
    descriptor = os.open(path, os.O_RDONLY)
    if fault == "closed":
        os.close(descriptor)
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/usr/bin/bwrap")
    try:
        with pytest.raises(ValueError, match="artifact_directory_fd_invalid"):
            runtime.isolated_artifact_command(
                artifact_root=artifact, entrypoint="run",
                artifact_directory_fd=True if fault == "boolean" else descriptor,
            )
    finally:
        if fault != "closed":
            os.close(descriptor)


@pytest.mark.parametrize("name,content", [("../escape", b"bad"), ("policy/run.py", None)])
def test_archive_rejects_traversal_and_links(tmp_path, name, content):
    source = tmp_path / "team.tar.gz"
    _archive(source, [(name, content)])
    destination = tmp_path / "output"
    destination.mkdir()
    with pytest.raises(ValueError, match="member_unsafe"):
        runtime._extract_regular_archive(source, destination)
    assert not (tmp_path / "escape").exists()


def test_archive_preserves_host_disk_floor(monkeypatch, tmp_path):
    source = tmp_path / "team.tar.gz"
    _archive(source, [("policy/run.py", b"#!/usr/bin/env python3\n")])
    destination = tmp_path / "output"
    destination.mkdir()
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda _path: SimpleNamespace(free=1))
    with pytest.raises(ValueError, match="capacity_insufficient"):
        runtime._extract_regular_archive(source, destination)
    assert list(destination.iterdir()) == []


def test_digest_mismatch_blocks_before_extraction(monkeypatch, tmp_path):
    source = tmp_path / "team.tar.gz"
    _archive(source, [("policy/run.py", b"#!/usr/bin/env python3\n")])
    setup, profile = _bound_profile("sha256:" + "f" * 64)
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setattr(runtime.os, "geteuid", lambda: 0)
    with pytest.raises(ValueError, match="digest_mismatch"):
        runtime.launch_g1_team_artifact_synthetic_probe(
            profile=profile, trusted_setup=setup, authenticated_owner=OWNER,
            operator_approved_profile_digest=profile["profile_digest"],
            operator_approved_artifact_sha256=profile["delivery"]["artifact_sha256"],
            staged_artifact_path=source, output_dir=tmp_path / "probe",
        )
    assert not (tmp_path / "probe").exists()


def test_verified_archive_uses_real_jsonl_wire_and_teardown(monkeypatch, tmp_path):
    source = tmp_path / "team.tar.gz"
    digest = _archive(source, [("policy/run.py", b"#!/usr/bin/env python3\n")])
    setup, profile = _bound_profile(digest)
    action = [0.0] * 40
    action[3:9] = [1, 0, 0, 1, 0, 0]
    script = "\n".join(
        [
            "import json,sys",
            "for line in sys.stdin:",
            "    request=json.loads(line)",
            f"    response={{'ok':True,'action_chunk':{action!r},'protocol':request['protocol'],'request_id':request['request_id']}}",
            "    if 'profile_digest' in request: response['profile_digest']=request['profile_digest']",
            "    response['action_chunk']=[response['action_chunk']]",
            "    print(json.dumps(response),flush=True)",
        ]
    )
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setattr(runtime.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        runtime.shutil, "disk_usage",
        lambda _path: SimpleNamespace(free=runtime._MIN_FREE_AFTER_EXTRACT + 1024**3),
    )
    monkeypatch.setattr(
        runtime, "isolated_artifact_command",
        lambda **_kwargs: [sys.executable, "-u", "-c", script],
    )
    lease, receipt = runtime.launch_g1_team_artifact_synthetic_probe(
        profile=profile, trusted_setup=setup, authenticated_owner=OWNER,
        operator_approved_profile_digest=profile["profile_digest"],
        operator_approved_artifact_sha256=digest, staged_artifact_path=source,
        output_dir=tmp_path / "probe",
    )
    assert receipt["status"] == "synthetic_wire_compatible"
    assert receipt["site_policy_query_count"] == 0
    assert (tmp_path / "probe" / "artifact" / "policy" / "run.py").is_file()
    close = lease.close()
    assert close["status"] == "process_exited"
    assert close["receipt_digest"] == canonical_digest(close, digest_field="receipt_digest")
    assert lease.close() == close


def test_profile_mismatch_blocks_before_archive_read(monkeypatch, tmp_path):
    source = tmp_path / "team.tar.gz"
    digest = _archive(source, [("policy/run.py", b"x")])
    setup, profile = _bound_profile(digest)
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setattr(runtime, "_sha256", lambda _path: pytest.fail("archive read early"))
    with pytest.raises(ValueError, match="admission_invalid"):
        runtime.launch_g1_team_artifact_synthetic_probe(
            profile=profile, trusted_setup=setup, authenticated_owner=OWNER,
            operator_approved_profile_digest="sha256:" + "f" * 64,
            operator_approved_artifact_sha256=digest,
            staged_artifact_path=source, output_dir=tmp_path / "probe",
        )


def test_client_initialization_failure_kills_sandbox_process(monkeypatch, tmp_path):
    source = tmp_path / "team.tar.gz"
    digest = _archive(source, [("policy/run.py", b"#!/usr/bin/env python3\n")])
    setup, profile = _bound_profile(digest)
    kills = []
    launch_options = []

    class StartedProcess:
        pid = 12345

        def wait(self, timeout):
            assert timeout == 5
            return -9

    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setattr(runtime.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        runtime.shutil, "disk_usage",
        lambda _path: SimpleNamespace(free=runtime._MIN_FREE_AFTER_EXTRACT + 1024**3),
    )
    monkeypatch.setattr(runtime, "isolated_artifact_command", lambda **_kwargs: ["sandbox"])
    monkeypatch.setattr(
        runtime.subprocess, "Popen",
        lambda *_args, **kwargs: launch_options.append(kwargs) or StartedProcess(),
    )
    monkeypatch.setattr(
        runtime, "NativeG1TeamPolicyJsonlClient",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("client failed")),
    )
    monkeypatch.setattr(runtime.os, "killpg", lambda pid, sig: kills.append((pid, sig)))
    with pytest.raises(ValueError, match="client failed"):
        runtime.launch_g1_team_artifact_synthetic_probe(
            profile=profile, trusted_setup=setup, authenticated_owner=OWNER,
            operator_approved_profile_digest=profile["profile_digest"],
            operator_approved_artifact_sha256=digest,
            staged_artifact_path=source, output_dir=tmp_path / "probe",
        )
    assert kills == [(12345, runtime.signal.SIGKILL)]
    assert set(launch_options[0]["env"]) == {"PATH", "HOME", "XDG_CACHE_HOME", "LANG"}
    assert len(launch_options[0]["pass_fds"]) == 1
    with pytest.raises(OSError):
        os.fstat(launch_options[0]["pass_fds"][0])


def test_gpu_probe_failure_closes_artifact_descriptor_without_starting_policy(monkeypatch, tmp_path):
    source = tmp_path / "team.tar.gz"
    digest = _archive(source, [("policy/run.py", b"#!/usr/bin/env python3\n")])
    setup, profile = _bound_profile(digest)
    descriptors = []
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setattr(runtime.os, "geteuid", lambda: 0)
    # Hardware caller is fake root on macOS; path permission semantics have
    # separate tests and the real Linux immutable-source rehearsal.
    monkeypatch.setattr(runtime, "_require_namespace_artifact_access", lambda _: None)
    monkeypatch.setattr(runtime, "recheck_archive_gpu_binding", lambda _: None)
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/usr/bin/bwrap")
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda _: SimpleNamespace(free=runtime._MIN_FREE_AFTER_EXTRACT + 1024**3))

    def refuse(**kwargs):
        command = kwargs["command"]
        descriptor = int(command[command.index("--ro-bind-fd") + 1])
        os.fstat(descriptor)
        descriptors.append(descriptor)
        raise ValueError("probe refused")

    monkeypatch.setattr(runtime, "probe_archive_gpu_namespace", refuse)
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *a, **kw: pytest.fail("policy started"))
    with pytest.raises(ValueError, match="probe refused"):
        runtime.launch_g1_team_artifact_synthetic_probe(
            profile=profile, trusted_setup=setup, authenticated_owner=OWNER,
            operator_approved_profile_digest=profile["profile_digest"],
            operator_approved_artifact_sha256=digest, staged_artifact_path=source,
            output_dir=tmp_path / "probe",
            gpu_binding={"profile_digest": profile["profile_digest"], "devices": []},
        )
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
