import base64
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from blueprint_pipeline.groot_oscar_digitalocean_builder import (
    BUILDER_TAG,
    REMOTE_BUILD_REQUIRED_RESULTS,
    TEARDOWN_TAG,
    _delete_with_fail_closed_evidence,
    _list_droplets_by_tag,
    _reconcile_ambiguous_create,
    _ssh_options,
    build_cloud_init,
    build_droplet_payload,
    known_hosts_line,
    launch_detached_builder,
    live_machine_probe_command,
    observe_local_machine,
    parse_live_machine_probe,
    run_builder,
    validate_remote_carrier_result,
    validate_remote_build_results,
    verify_packet_tarball,
)


def test_builder_ssh_options_keep_silent_remote_builds_alive(tmp_path: Path) -> None:
    options = _ssh_options(
        private_key=tmp_path / "login-key",
        known_hosts=tmp_path / "known-hosts",
    )

    assert "ServerAliveInterval=30" in options
    assert "ServerAliveCountMax=20" in options
    assert "TCPKeepAlive=yes" in options


def test_builder_inventory_follows_every_tag_filtered_page(monkeypatch) -> None:
    observed_paths: list[str] = []
    first_page = [{"id": index, "tags": [BUILDER_TAG]} for index in range(200)]

    def fake_request(*, token, method, path, payload=None):
        del token, method, payload
        observed_paths.append(path)
        if "page=1" in path:
            return 200, {"droplets": first_page}
        return 200, {"droplets": [{"id": 999, "tags": [BUILDER_TAG]}]}

    monkeypatch.setattr(
        "blueprint_pipeline.groot_oscar_digitalocean_builder._request",
        fake_request,
    )
    http_status, rows = _list_droplets_by_tag(token="secret", tag=BUILDER_TAG)
    assert http_status == 200
    assert len(rows) == 201
    assert any("page=2" in path for path in observed_paths)


def test_teardown_transport_error_is_persistable_fail_closed(monkeypatch) -> None:
    def fail_delete(**_kwargs):
        raise TimeoutError("lost delete response")

    monkeypatch.setattr(
        "blueprint_pipeline.groot_oscar_digitalocean_builder._delete_and_verify",
        fail_delete,
    )
    result = _delete_with_fail_closed_evidence(token="secret", droplet_id="123")
    assert result["provider_absence_confirmed"] is False
    assert result["teardown_error_type"] == "TimeoutError"
    assert "lost delete response" not in json.dumps(result)


def test_digitalocean_watchdog_persists_teardown_transport_error(
    tmp_path: Path, monkeypatch
) -> None:
    from blueprint_pipeline.groot_oscar_digitalocean_builder import watchdog

    state = tmp_path / "allocation_state.json"
    state.write_text(json.dumps({"droplet_id": "123", "deadline_epoch": 0}), encoding="utf-8")
    token_file = tmp_path / "token"
    token_file.write_text("secret", encoding="utf-8")
    monkeypatch.setattr(
        "blueprint_pipeline.groot_oscar_digitalocean_builder._delete_and_verify",
        lambda **_kwargs: (_ for _ in ()).throw(TimeoutError("secret response")),
    )
    assert watchdog(state_path=state, token_file=token_file) == 2
    persisted = json.loads((tmp_path / "watchdog_result.json").read_text())
    armed = json.loads((tmp_path / "watchdog_armed.json").read_text())
    assert armed["status"] == "armed"
    assert armed["droplet_id"] == "123"
    assert persisted["status"] == "teardown_unverified"
    assert persisted["teardown_error_type"] == "TimeoutError"
    assert "secret response" not in json.dumps(persisted)


def test_digitalocean_watchdog_reconciles_precreate_identity(tmp_path: Path, monkeypatch) -> None:
    from blueprint_pipeline.groot_oscar_digitalocean_builder import watchdog

    state = tmp_path / "allocation_state.json"
    state.write_text(
        json.dumps(
            {
                "droplet_id": None,
                "name": "blueprint-groot-oscar-thin-aaaaaaaa",
                "region": "sfo3",
                "deadline_epoch": 0,
                "watchdog_nonce": "nonce-for-test",
            }
        ),
        encoding="utf-8",
    )
    token_file = tmp_path / "token"
    token_file.write_text("secret", encoding="utf-8")
    observed: dict[str, str] = {}

    def reconcile(**kwargs):
        observed.update(kwargs)
        return {"provider_absence_confirmed": True, "status": "provider_terminal"}

    monkeypatch.setattr(
        "blueprint_pipeline.groot_oscar_digitalocean_builder._reconcile_ambiguous_create",
        reconcile,
    )
    assert watchdog(state_path=state, token_file=token_file) == 0
    assert observed == {
        "token": "secret",
        "name": "blueprint-groot-oscar-thin-aaaaaaaa",
        "region": "sfo3",
    }
    armed = json.loads((tmp_path / "watchdog_armed.json").read_text())
    assert armed["watchdog_nonce"] == "nonce-for-test"


def test_builder_arms_watchdog_before_provider_create() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/blueprint_pipeline/groot_oscar_digitalocean_builder.py"
    ).read_text(encoding="utf-8")
    run_source = source[
        source.index("def run_builder(") : source.index("def launch_detached_builder(")
    ]
    assert run_source.index("watchdog_process = subprocess.Popen(") < run_source.index(
        'method="POST", path="/droplets"'
    )
    assert run_source.index(
        'watchdog_armed_path = output / "watchdog_armed.json"'
    ) < run_source.index('method="POST", path="/droplets"')


def test_ambiguous_create_reconciliation_deletes_exact_name_tag_match(
    monkeypatch,
) -> None:
    calls: list[tuple[str, str]] = []
    inventories = [
        {
            "droplets": [
                {
                    "id": 123,
                    "name": "blueprint-groot-oscar-thin-aaaaaaaa",
                    "region": {"slug": "sfo3"},
                    "tags": [BUILDER_TAG, TEARDOWN_TAG],
                },
                {
                    "id": 999,
                    "name": "unrelated",
                    "region": {"slug": "sfo3"},
                    "tags": [BUILDER_TAG, TEARDOWN_TAG],
                },
            ]
        },
        {"droplets": []},
    ]

    def fake_request(*, token, method, path, payload=None):
        del token, payload
        calls.append((method, path))
        return 200, inventories.pop(0)

    monkeypatch.setattr(
        "blueprint_pipeline.groot_oscar_digitalocean_builder._request",
        fake_request,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.groot_oscar_digitalocean_builder._delete_and_verify",
        lambda **_kwargs: {"provider_absence_confirmed": True},
    )
    result = _reconcile_ambiguous_create(
        token="secret",
        name="blueprint-groot-oscar-thin-aaaaaaaa",
        region="sfo3",
        attempts=2,
        sleeper=lambda _seconds: None,
    )
    assert result["status"] == "provider_terminal"
    assert result["provider_absence_confirmed"] is True
    assert result["reconciled_droplet_ids"] == ["123"]
    assert all(method == "GET" for method, _path in calls)


def test_ambiguous_create_reconciliation_fails_closed_when_inventory_unverified(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "blueprint_pipeline.groot_oscar_digitalocean_builder._request",
        lambda **_kwargs: (503, {}),
    )
    result = _reconcile_ambiguous_create(
        token="secret",
        name="blueprint-groot-oscar-thin-aaaaaaaa",
        region="sfo3",
        attempts=2,
        sleeper=lambda _seconds: None,
    )
    assert result["status"] == "teardown_unverified"
    assert result["provider_absence_confirmed"] is False


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
    assert "/etc/apt/apt.conf.d/80blueprint-transport" in text
    assert 'Acquire::Retries "10";' in text
    assert 'Acquire::http::Timeout "30";' in text
    assert 'Acquire::https::Timeout "30";' in text
    assert 'Acquire::http::Pipeline-Depth "0";' in text
    boot_commands = text.split("package_update:", 1)[0]
    assert boot_commands.count("https://mirrors.digitalocean.com") == 1
    assert boot_commands.count("https://security.ubuntu.com") == 1
    assert boot_commands.count("https://archive.ubuntu.com") == 1
    assert text.splitlines().count("  - docker.io") == 1
    assert "docker-buildx" in text
    assert "docker info" in text
    assert "shutdown -h +120" in text
    assert "docker_pat" not in text
    assert "docker_username" not in text


def test_carrier_cloud_init_uses_the_governed_docker_build_plane() -> None:
    text = build_cloud_init(
        host_private_b64="private",
        host_public_b64="public",
        shutdown_minutes=30,
        packet_kind="carrier_image",
    )
    assert text.splitlines().count("  - docker.io") == 1
    assert "docker-buildx" in text
    assert "touch /root/blueprint-builder-ready" in text


def test_carrier_remote_result_binds_digest_base_source_and_dockerfile(
    tmp_path: Path,
) -> None:
    packet = {
        "carrier_image_ref": "docker.io/example/carrier:versioned",
        "carrier_base_image_ref": "docker.io/example/base@sha256:" + "b" * 64,
        "carrier_dockerfile_sha256": "c" * 64,
        "source_commit": "a" * 40,
    }
    payload = {
        "schema_version": "groot_oscar_carrier_remote_build_result.v1",
        "status": "completed",
        "blockers": [],
        "image_ref": packet["carrier_image_ref"],
        "resolved_digest_ref": "docker.io/example/carrier@sha256:" + "d" * 64,
        "base_image_ref": packet["carrier_base_image_ref"],
        "dockerfile_sha256": packet["carrier_dockerfile_sha256"],
        "source_commit": packet["source_commit"],
        "platform": "linux/amd64",
        "raw_secret_values_recorded": False,
    }
    (tmp_path / "groot_oscar_carrier_remote_build_result.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    verified = validate_remote_carrier_result(tmp_path, packet=packet)
    assert verified["status"] == "verified"
    assert verified["resolved_digest_ref"] == payload["resolved_digest_ref"]


def test_carrier_remote_result_rejects_wrong_digest_repository(tmp_path: Path) -> None:
    packet = {
        "carrier_image_ref": "docker.io/example/carrier:versioned",
        "carrier_base_image_ref": "docker.io/example/base@sha256:" + "b" * 64,
        "carrier_dockerfile_sha256": "c" * 64,
        "source_commit": "a" * 40,
    }
    payload = {
        "schema_version": "groot_oscar_carrier_remote_build_result.v1",
        "status": "completed",
        "blockers": [],
        "image_ref": packet["carrier_image_ref"],
        "resolved_digest_ref": "docker.io/example/other@sha256:" + "d" * 64,
        "base_image_ref": packet["carrier_base_image_ref"],
        "dockerfile_sha256": packet["carrier_dockerfile_sha256"],
        "source_commit": packet["source_commit"],
        "platform": "linux/amd64",
        "raw_secret_values_recorded": False,
    }
    (tmp_path / "groot_oscar_carrier_remote_build_result.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    blocked = validate_remote_carrier_result(tmp_path, packet=packet)

    assert blocked["status"] == "blocked"
    assert "carrier_remote_build_digest_repository_mismatch" in blocked["blockers"]


def test_model_cache_runtime_bundle_cloud_init_installs_docker() -> None:
    text = build_cloud_init(
        host_private_b64="private",
        host_public_b64="public",
        shutdown_minutes=120,
        packet_kind="model_cache_s3",
        runtime_bundle_requested=True,
    )
    assert text.splitlines().count("  - docker.io") == 1
    assert "docker-buildx" not in text
    assert "systemctl enable --now docker" in text
    assert "docker info" in text
    assert "python3 -m venv /root/blueprint-venv-probe" in text
    ready_command = next(
        line for line in text.splitlines() if "touch /root/blueprint-builder-ready" in line
    )
    assert "docker info" in ready_command
    assert "python3 -m venv" in ready_command
    assert text.splitlines().count("  - touch /root/blueprint-builder-ready") == 0


def test_plain_model_cache_cloud_init_does_not_install_docker() -> None:
    text = build_cloud_init(
        host_private_b64="private",
        host_public_b64="public",
        shutdown_minutes=120,
        packet_kind="model_cache_s3",
    )
    assert "docker.io" not in text
    assert "systemctl enable --now docker" not in text
    assert "python3 -m venv /root/blueprint-venv-probe" in text
    ready_command = next(
        line for line in text.splitlines() if "touch /root/blueprint-builder-ready" in line
    )
    assert "docker info" not in ready_command
    assert "python3 -m venv" in ready_command


def test_cloud_init_refuses_ttl_above_two_hours() -> None:
    with pytest.raises(ValueError, match="shutdown_minutes"):
        build_cloud_init(host_private_b64="a", host_public_b64="b", shutdown_minutes=121)


def test_droplet_payload_uses_only_verified_profile() -> None:
    payload = build_droplet_payload(
        name="builder", region="sfo3", ssh_key_id=123, user_data="#cloud-config"
    )
    assert payload["size"] == "s-8vcpu-16gb-amd"
    assert payload["image"] == "ubuntu-24-04-x64"
    assert payload["ssh_keys"] == [123]
    assert payload["tags"] == [BUILDER_TAG, TEARDOWN_TAG]


def test_known_hosts_line_uses_exact_launch_bound_ed25519_key() -> None:
    assert (
        known_hosts_line(ip="203.0.113.5", public_key_text="ssh-ed25519 AAAAhostkey comment")
        == "203.0.113.5 ssh-ed25519 AAAAhostkey\n"
    )
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
    assert (tmp_path / "run/supervisor.lock").is_file()
    with pytest.raises(ValueError, match="already_has_supervisor_lock"):
        launch_detached_builder(
            output_dir=tmp_path / "run",
            run_arguments=["--output-dir", str(tmp_path / "run"), "--allow-paid"],
        )


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


def test_thin_release_live_machine_probe_executes_and_emits_json() -> None:
    completed = subprocess.run(
        ["bash", "-c", live_machine_probe_command(mount_path="/")],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    evidence = parse_live_machine_probe(completed.stdout)
    assert evidence["observation_source"] == "live_machine_probe"
    assert evidence["mount_path"] == "/"
    assert evidence["s3_endpoint_host"] is None


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


def test_local_machine_probe_requires_observed_builder_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Completed:
        returncode = 0

    monkeypatch.setattr(
        "blueprint_pipeline.groot_oscar_digitalocean_builder.subprocess.run",
        lambda *_args, **_kwargs: Completed(),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.groot_oscar_digitalocean_builder.shutil.which",
        lambda _name: "/usr/bin/docker",
    )
    original_is_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda self: (
            False if self == Path("/root/blueprint-builder-ready") else original_is_file(self)
        ),
    )

    evidence = observe_local_machine(mount_path=tmp_path)

    assert evidence["builder_ready_marker"] is False
    assert evidence["status"] == "blocked"
    assert "live_machine_builder_initialization_incomplete" in evidence["blockers"]


def test_builder_verifies_transfer_archive_digest_before_allocation(
    tmp_path: Path,
) -> None:
    tarball = tmp_path / "packet.tar.gz"
    tarball.write_bytes(b"checksum-bound-packet")
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    packet = {"tarball_path": str(tarball), "tarball_sha256": digest}
    assert verify_packet_tarball(packet)["status"] == "verified"

    tarball.write_bytes(b"substituted-packet")
    result = verify_packet_tarball(packet)
    assert result["status"] == "blocked"
    assert "digitalocean_builder_packet_tarball_digest_mismatch" in result["blockers"]


def test_builder_source_copies_registry_files_to_fixed_remote_names() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/blueprint_pipeline/groot_oscar_digitalocean_builder.py"
    ).read_text(encoding="utf-8")
    assert '(docker_username_file.expanduser(), "/root/blueprint-build/docker_username")' in source
    assert '(docker_password_file.expanduser(), "/root/blueprint-build/docker_pat")' in source


def test_remote_build_results_must_be_complete_and_completed(tmp_path: Path) -> None:
    results = tmp_path / "remote_results"
    results.mkdir()
    blocked = validate_remote_build_results(results)
    assert blocked["status"] == "blocked"
    assert any(item.startswith("remote_build_result_missing:") for item in blocked["blockers"])

    for name in blocked["required_results"]:
        if name.startswith("groot_oscar_thin"):
            payload = {"status": "completed"}
        elif name in {
            "release_supply_chain_manifest.json",
            "release_supply_chain_disk_admission.json",
        }:
            payload = {"status": "passed"}
        else:
            payload = {}
        (results / name).write_text(json.dumps(payload), encoding="utf-8")
    assert validate_remote_build_results(results)["status"] == "verified"


def test_remote_build_results_accept_digest_pinned_serverless_foundation_reuse(
    tmp_path: Path,
) -> None:
    results = tmp_path / "remote_results"
    results.mkdir()
    foundation_ref = "registry.example/foundation@sha256:" + "a" * 64
    release_result = {
        "status": "completed",
        "foundation_image_ref": foundation_ref,
        "serverless_worker_contract": {
            "status": "passed",
            "worker_source_packaged": True,
            "worker_command_packaged": True,
            "runpod_sdk_exactly_pinned": True,
            "models_externalized": True,
        },
        "thin_release_contract": {
            "status": "passed",
            "foundation_image_ref": foundation_ref,
            "release_delta_budget_passed": True,
            "models_externalized": True,
        },
    }
    for name in REMOTE_BUILD_REQUIRED_RESULTS:
        if name == "foundation_buildx_metadata.json":
            continue
        if name == "groot_oscar_thin_remote_build_result.json":
            payload = release_result
        elif name == "foundation_registry_diagnostic.json":
            payload = {
                "status": "completed",
                "blockers": [],
                "image_ref": foundation_ref,
                "resolved_digest_ref": foundation_ref,
            }
        elif name in {
            "release_supply_chain_manifest.json",
            "release_supply_chain_disk_admission.json",
        }:
            payload = {"status": "passed"}
        else:
            payload = {}
        (results / name).write_text(json.dumps(payload), encoding="utf-8")

    verified = validate_remote_build_results(results)
    assert verified["status"] == "verified"
    assert verified["digest_pinned_foundation_reused"] is True
    assert "foundation_buildx_metadata.json" not in verified["required_results"]


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


def test_run_builder_persists_live_cost_cap_failure_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tarball = tmp_path / "packet.tar.gz"
    tarball.write_bytes(b"bound-build-packet")
    packet = {
        "status": "ready",
        "source_commit": "a" * 40,
        "source_worktree_dirty": False,
        "provider_launch_performed_by_packet": False,
        "tarball_path": str(tarball),
        "tarball_sha256": hashlib.sha256(tarball.read_bytes()).hexdigest(),
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
        "paid_mutation_authorized": True,
        "max_spend_usd": 0.01,
        "hard_ttl_seconds": 7200,
        "one_resource_limit": True,
        "independent_teardown_watchdog": True,
    }
    paths: dict[str, Path] = {}
    for name, payload in (
        ("packet", packet),
        ("builder", builder),
        ("spend", spend),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path
    token_file = tmp_path / "do-token"
    token_file.write_text("test-token", encoding="utf-8")
    monkeypatch.setattr(
        "blueprint_pipeline.groot_oscar_digitalocean_builder._live_profile",
        lambda **_kwargs: (
            {
                "status": "verified",
                "blockers": [],
                "observed": {"price_hourly_usd": 0.10},
            },
            [],
        ),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.groot_oscar_digitalocean_builder._host_key_material",
        lambda _path: ("private", "public", "SHA256:" + "d" * 43),
    )
    missing = tmp_path / "not-read-before-cap"
    output = tmp_path / "out"
    result = run_builder(
        output_dir=output,
        packet_manifest_path=paths["packet"],
        builder_evidence_path=paths["builder"],
        spend_path=paths["spend"],
        token_file=token_file,
        docker_username_file=missing,
        docker_password_file=missing,
        login_private_key=missing,
        host_private_key=missing,
        ssh_key_id=1,
        region="sfo3",
        allow_paid=True,
    )
    assert result["status"] == "blocked_pre_allocation"
    assert result["provider_mutation_performed"] is False
    assert result["required_maximum_compute_spend_usd"] == pytest.approx(0.20)
    assert json.loads((output / "builder_run_result.json").read_text()) == result

    spend["max_spend_usd"] = 1.0
    paths["spend"].write_text(json.dumps(spend), encoding="utf-8")
    credential_result = run_builder(
        output_dir=tmp_path / "credential-check",
        packet_manifest_path=paths["packet"],
        builder_evidence_path=paths["builder"],
        spend_path=paths["spend"],
        token_file=token_file,
        docker_username_file=missing,
        docker_password_file=missing,
        login_private_key=missing,
        host_private_key=missing,
        ssh_key_id=1,
        region="sfo3",
        allow_paid=True,
    )
    assert credential_result["status"] == "blocked_pre_allocation"
    assert credential_result["provider_mutation_performed"] is False
    assert credential_result["blockers"] == ["digitalocean_builder_local_credentials_unavailable"]
