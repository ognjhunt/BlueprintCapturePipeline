"""A team archive must be digest-bound and namespace-isolated before probing."""

import hashlib
import io
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


def test_bwrap_command_exposes_readonly_artifact_without_network(monkeypatch, tmp_path):
    artifact = tmp_path / "artifact"
    (artifact / "policy").mkdir(parents=True)
    (artifact / "policy" / "run.py").write_text("#!/usr/bin/env python3\n")
    monkeypatch.setattr(runtime.shutil, "which", lambda _name: "/usr/bin/bwrap")
    command = runtime.isolated_artifact_command(artifact_root=artifact, entrypoint="policy/run.py")
    assert command[0:4] == ["/usr/bin/bwrap", "--unshare-all", "--die-with-parent", "--new-session"]
    assert command[command.index("--ro-bind") + 1 : command.index("--ro-bind") + 3] == [
        str(artifact), "/work"
    ]
    assert "--bind" not in command
    assert command[command.index("--uid") + 1] == "65534"
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert command[-1] == "/work/policy/run.py"
    with pytest.raises(ValueError, match="entrypoint_invalid"):
        runtime.isolated_artifact_command(artifact_root=artifact, entrypoint="../etc/passwd")


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
