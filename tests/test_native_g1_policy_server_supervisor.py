from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from blueprint_pipeline import native_g1_policy_server_supervisor as supervisor


def test_source_revision_requires_clean_pinned_checkout(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "upstream"
    (root / "scripts").mkdir(parents=True)
    (root / "src").mkdir()
    source = root / "scripts/server.py"
    source.write_text("source")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run([
        "git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.com",
        "commit", "-qm", "source",
    ], check=True)
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    monkeypatch.setattr(supervisor, "PINNED_SOURCE_REVISION", revision)
    assert supervisor._source_revision(source) == revision
    source.write_text("changed")
    with pytest.raises(ValueError, match="revision_or_tree_mismatch"):
        supervisor._source_revision(source)


def test_source_revision_accepts_official_monorepo_layout(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "HumanoidArena"
    server = root / "lerobot/scripts/serve_lerobot_vla_http.py"
    sonic = root / "isaaclab_twist2_g1/action_provider/action_provider_sonic.py"
    server.parent.mkdir(parents=True)
    sonic.parent.mkdir(parents=True)
    (root / "lerobot/src").mkdir()
    server.write_text("server")
    sonic.write_text("sonic")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run([
        "git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.com",
        "commit", "-qm", "source",
    ], check=True)
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    monkeypatch.setattr(supervisor, "PINNED_SOURCE_REVISION", revision)
    assert supervisor._source_revision(server) == revision
    assert supervisor._source_revision(sonic, expected_parent="action_provider") == revision


def test_linux_listener_requires_owning_process(tmp_path: Path) -> None:
    net = tmp_path / "net"
    net.mkdir()
    (net / "tcp").write_text(
        "sl local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode\n"
        "0: 0100007F:20FB 00000000:0000 0A 0 0 0 0 0 12345\n"
    )
    fd = tmp_path / "417" / "fd"
    fd.mkdir(parents=True)
    (fd / "3").symlink_to("socket:[12345]")
    assert supervisor._linux_listener_owner_pids(8443, proc_root=tmp_path) == {417}
    (fd / "3").unlink()
    with pytest.raises(ValueError, match="owner_unresolved"):
        supervisor._linux_listener_owner_pids(8443, proc_root=tmp_path)


def test_external_base_model_reference_is_not_admitted(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({
        "source_revision": supervisor.PINNED_SOURCE_REVISION,
        "candidates": [{"candidate_id": "pi", "subdirectory": "pi/model"}],
    }))
    config = tmp_path / "checkpoints/pi/model/config.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"pretrained_path": "/publisher/private/base"}))
    with pytest.raises(ValueError, match="external_base_model_unverified"):
        supervisor._candidate_policy_dir(inventory, "pi", tmp_path / "checkpoints")


@pytest.mark.parametrize("candidate_id,base_hint", list(supervisor.PUBLISHER_PI05_BASE_HINTS.items()))
def test_exact_publisher_pi05_training_hint_does_not_block_local_inference(
    tmp_path: Path, candidate_id: str, base_hint: str
) -> None:
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({
        "source_revision": supervisor.PINNED_SOURCE_REVISION,
        "candidates": [{"candidate_id": candidate_id, "subdirectory": "pi/model"}],
    }))
    config = tmp_path / "checkpoints/pi/model/config.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({
        "type": "pi05", "use_peft": False, "pretrained_path": base_hint,
    }))
    assert supervisor._candidate_policy_dir(
        inventory, candidate_id, tmp_path / "checkpoints"
    ) == config.parent
    config.write_text(json.dumps({"type": "pi05", "pretrained_path": "/other/base"}))
    with pytest.raises(ValueError, match="external_base_model_unverified"):
        supervisor._candidate_policy_dir(inventory, candidate_id, tmp_path / "checkpoints")
    config.write_text(json.dumps({
        "type": "pi05", "use_peft": True, "pretrained_path": base_hint,
    }))
    with pytest.raises(ValueError, match="external_base_model_unverified"):
        supervisor._candidate_policy_dir(inventory, candidate_id, tmp_path / "checkpoints")


class _Process:
    pid = 417

    def __init__(self) -> None:
        self.exit_code = None
        self.terminated = False

    def poll(self):
        return self.exit_code

    def terminate(self):
        self.terminated = True
        self.exit_code = -15

    def wait(self, *, timeout):
        assert timeout == 10
        return self.exit_code


class _Client:
    def __init__(self, *, base_url: str) -> None:
        assert base_url == "http://127.0.0.1:8443"
        self.resets = []

    def reset(self, *, seed: int) -> None:
        self.resets.append(seed)


def _launch_inputs(tmp_path: Path, monkeypatch) -> dict:
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({
        "source_revision": supervisor.PINNED_SOURCE_REVISION,
        "candidates": [{"candidate_id": "dp", "subdirectory": "small/model"}],
    }))
    config = tmp_path / "checkpoints/small/model/config.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"type": "diffusion"}))
    source = tmp_path / "source/scripts/server.py"
    source.parent.mkdir(parents=True)
    source.write_text("source")
    monkeypatch.setattr(supervisor, "_source_revision", lambda path: supervisor.PINNED_SOURCE_REVISION)
    monkeypatch.setattr(supervisor, "preflight_g1_shared_scene_run", lambda **kwargs: {
        "status": "staged_inputs_verified",
        "scene_plan_digest": "sha256:" + "a" * 64,
        "candidate_id": "dp",
        "candidate_inventory_digest": "sha256:" + "b" * 64,
    })
    return {
        "preflight_inputs": {
            "policy_server_source": source,
            "inventory_path": inventory,
            "checkpoint_root": tmp_path / "checkpoints",
        },
        "python_executable": Path(sys.executable),
        "port": 8443,
        "device": "cuda:0",
        "log_path": tmp_path / "server.log",
        "client_factory": _Client,
    }


def test_supervisor_binds_child_listener_and_closes_only_owned_process(tmp_path, monkeypatch) -> None:
    args = _launch_inputs(tmp_path, monkeypatch)
    process = _Process()
    launched = []

    def popen(argv, **kwargs):
        launched.append(argv)
        assert kwargs["start_new_session"] is True
        return process

    lease = supervisor.start_g1_policy_server(
        **args, owner_reader=lambda port: {process.pid}, popen_factory=popen,
    )
    assert launched[0][-6:] == ["--device", "cuda:0", "--host", "127.0.0.1", "--port", "8443"]
    assert lease.receipt["status"] == "server_ready_process_bound"
    assert lease.receipt["reset_ack_verified"] is True
    assert lease.receipt["inference_observed"] is False
    assert lease.client.resets == [0]
    assert lease.close()["status"] == "child_exited"
    assert process.terminated is True


def test_unowned_listener_fails_and_reaps_child(tmp_path, monkeypatch) -> None:
    args = _launch_inputs(tmp_path, monkeypatch)
    process = _Process()
    with pytest.raises(RuntimeError, match="listener_owner_mismatch"):
        supervisor.start_g1_policy_server(
            **args, owner_reader=lambda port: {999},
            popen_factory=lambda argv, **kwargs: process,
        )
    assert process.terminated is True
