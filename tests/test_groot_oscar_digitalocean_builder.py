import base64
import json
from pathlib import Path

import pytest

from blueprint_pipeline.groot_oscar_digitalocean_builder import (
    BUILDER_TAG,
    TEARDOWN_TAG,
    build_cloud_init,
    build_droplet_payload,
    known_hosts_line,
    launch_detached_builder,
    live_machine_probe_command,
    parse_live_machine_probe,
    run_builder,
)


def test_cloud_init_binds_host_key_and_known_builder_packages() -> None:
    private = base64.b64encode(b"private-host-key").decode()
    public = base64.b64encode(b"ssh-ed25519 AAAAhostkey builder\n").decode()
    text = build_cloud_init(
        host_private_b64=private,
        host_public_b64=public,
        shutdown_minutes=120,
    )
    assert private in text
    assert public in text
    assert "ssh_deletekeys: false" in text
    assert "bootcmd:" in text
    assert "systemctl restart ssh" not in text.split("package_update:", 1)[0]
    assert "systemctl restart ssh" in text
    assert "docker.io" in text
    assert "docker-buildx" in text
    assert "docker info" in text
    assert "shutdown -h +120" in text
    assert "docker_pat" not in text
    assert "docker_username" not in text


def test_cloud_init_refuses_ttl_above_two_hours() -> None:
    with pytest.raises(ValueError, match="shutdown_minutes"):
        build_cloud_init(
            host_private_b64="a", host_public_b64="b", shutdown_minutes=121
        )


def test_droplet_payload_uses_only_verified_profile() -> None:
    payload = build_droplet_payload(
        name="builder", region="sfo3", ssh_key_id=123, user_data="#cloud-config"
    )
    assert payload["size"] == "s-8vcpu-16gb-amd"
    assert payload["image"] == "ubuntu-24-04-x64"
    assert payload["ssh_keys"] == [123]
    assert payload["tags"] == [BUILDER_TAG, TEARDOWN_TAG]


def test_known_hosts_line_uses_exact_launch_bound_ed25519_key() -> None:
    assert known_hosts_line(
        ip="203.0.113.5", public_key_text="ssh-ed25519 AAAAhostkey comment"
    ) == "203.0.113.5 ssh-ed25519 AAAAhostkey\n"
    with pytest.raises(ValueError, match="public_host_key_invalid"):
        known_hosts_line(ip="203.0.113.5", public_key_text="ssh-rsa AAAA")


def test_detached_launch_uses_new_session_and_records_only_nonsecret_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    observed = {}

    class Process:
        pid = 4321

    def fake_popen(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return Process()

    monkeypatch.setattr(
        "blueprint_pipeline.groot_oscar_digitalocean_builder.subprocess.Popen",
        fake_popen,
    )
    result = launch_detached_builder(
        output_dir=tmp_path / "run",
        run_arguments=["--output-dir", str(tmp_path / "run"), "--allow-paid"],
    )
    assert observed["start_new_session"] is True
    assert observed["stdin"] is not None
    assert observed["command"][-1] == "--allow-paid"
    assert result["pid"] == 4321
    assert result["raw_secret_values_recorded"] is False


def test_live_machine_probe_is_validated_as_direct_machine_evidence() -> None:
    command = live_machine_probe_command(mount_path="/")
    assert 'os.path.isfile("/root/blueprint-builder-ready")' in command
    assert "os.statvfs" in command
    assert 'ok(["docker", "info"])' in command
    evidence = parse_live_machine_probe(
        "boot banner\n"
        + json.dumps(
            {
                "observation_source": "live_machine_probe",
                "system": "Linux",
                "architecture": "x86_64",
                "mount_path": "/",
                "free_bytes": 130 * 1024**3,
                "docker_cli_present": True,
                "docker_daemon_responding": True,
                "docker_buildx_available": True,
                "builder_ready_marker": True,
            }
        )
    )
    assert evidence["status"] == "verified"
    assert evidence["free_bytes"] == 130 * 1024**3


def test_live_machine_probe_rejects_catalog_or_requested_configuration() -> None:
    evidence = parse_live_machine_probe(
        json.dumps(
            {
                "observation_source": "provider_catalog",
                "system": "Linux",
                "architecture": "x86_64",
                "mount_path": "/",
                "free_bytes": 320 * 1024**3,
                "docker_cli_present": True,
                "docker_daemon_responding": True,
                "docker_buildx_available": True,
            }
        )
    )
    assert evidence["status"] == "blocked"
    assert "live_machine_observation_source_invalid" in evidence["blockers"]


def test_run_builder_is_dry_and_does_not_read_secrets_without_paid_gate(
    tmp_path: Path,
) -> None:
    packet = {
        "status": "ready",
        "source_commit": "a" * 40,
        "source_worktree_dirty": False,
        "provider_launch_performed_by_packet": False,
    }
    builder = {
        "provider": "digitalocean",
        "purpose": "image_build",
        "platform": "linux/amd64",
        "docker_daemon_verified": True,
        "docker_buildx_verified": True,
        "free_disk_bytes": 120 * 1024**3,
        "registry_push_auth_file_verified": True,
        "independent_teardown_watchdog": True,
        "ssh_host_key_sha256": "SHA256:" + "d" * 43,
        "ssh_host_key_independently_verified": True,
        "ssh_host_key_verification_method": "launch_bound_generated_host_key",
        "expected_source_commit": "a" * 40,
    }
    spend = {
        "paid_mutation_authorized": False,
        "max_spend_usd": 0.35,
        "hard_ttl_seconds": 7200,
        "one_resource_limit": True,
        "independent_teardown_watchdog": True,
    }
    paths = []
    for name, payload in (
        ("packet.json", packet),
        ("builder.json", builder),
        ("spend.json", spend),
    ):
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    missing = tmp_path / "must-not-be-read"
    result = run_builder(
        output_dir=tmp_path / "out",
        packet_manifest_path=paths[0],
        builder_evidence_path=paths[1],
        spend_path=paths[2],
        token_file=missing,
        docker_username_file=missing,
        docker_password_file=missing,
        login_private_key=missing,
        host_private_key=missing,
        ssh_key_id=1,
        region="sfo3",
        allow_paid=False,
    )
    assert result["status"] == "blocked_pre_allocation"
    assert result["provider_mutation_performed"] is False
    assert "builder_paid_mutation_not_authorized" in result["blockers"]
    assert "digitalocean_builder_allow_paid_flag_missing" in result["blockers"]
