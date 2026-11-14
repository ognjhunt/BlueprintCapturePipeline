import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEMD_DIR = REPO_ROOT / "deploy" / "systemd"
TERRAFORM_MAIN = REPO_ROOT / "deploy" / "terraform" / "main.tf"
TERRAFORM_TFVARS_EXAMPLE = REPO_ROOT / "deploy" / "terraform" / "terraform.tfvars.example"
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "install_live_pipeline_control_plane.sh"
PIPELINE_DEPLOY_SCRIPT = REPO_ROOT / "deploy" / "scripts" / "deploy.sh"


def _read(name: str) -> str:
    return (SYSTEMD_DIR / name).read_text(encoding="utf-8")


def _terraform_resource_body(text: str, resource_type: str, name: str) -> str:
    needle = f'resource "{resource_type}" "{name}" {{'
    start = text.index(needle)
    body_start = start + len(needle)
    depth = 1
    index = body_start
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[body_start:index]
        index += 1
    raise AssertionError(f"Unclosed Terraform resource block: {resource_type}.{name}")


def _terraform_variable_body(text: str, name: str) -> str:
    needle = f'variable "{name}" {{'
    start = text.index(needle)
    body_start = start + len(needle)
    depth = 1
    index = body_start
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[body_start:index]
        index += 1
    raise AssertionError(f"Unclosed Terraform variable block: {name}")


def test_production_systemd_units_set_fail_closed_runtime_posture():
    for unit in (
        "blueprint-pipeline-intake.service",
        "blueprint-pubsub-handoff-listener.service",
    ):
        text = _read(unit)

        assert re.search(r"/Users/[^/]+/", text) is None
        if unit == "blueprint-pubsub-handoff-listener.service":
            # The credential EnvironmentFile overrides Environment= and names an
            # archived checkout in BLUEPRINT_PIPELINE_REPO, so the release is
            # located through a unit-owned key and exported to the children.
            assert "BLUEPRINT_PUBSUB_HANDOFF_REPO=/opt/blueprint/task-evaluation-control-plane" in text
            assert (
                "BLUEPRINT_PIPELINE_PYTHON="
                "/opt/blueprint/BlueprintCapturePipeline/.venv/bin/python"
            ) in text
        else:
            assert (
                "BLUEPRINT_PIPELINE_REPO=/opt/blueprint/BlueprintCapturePipeline"
                in text
            )
        if unit == "blueprint-pubsub-handoff-listener.service":
            # Website handoffs register prepared sources in the same namespace
            # consumed by scene progression before enqueueing the signed intake.
            assert "EnvironmentFile=-/etc/blueprint/task-evaluation-scene-progression.env" in text
            assert text.count('exec env PYTHONPATH=src "$${BLUEPRINT_PIPELINE_PYTHON}" -m') == 2
            assert ".venv/bin/blueprint-pubsub-handoff-listener" not in text
            assert text.count('cd -P "$${BLUEPRINT_PUBSUB_HANDOFF_REPO}"') == 2
            assert "$${BLUEPRINT_PIPELINE_REPO}" not in text
            assert text.count('export BLUEPRINT_PIPELINE_REPO="$${PWD}"') == 2
        assert "BLUEPRINT_LAUNCH_PROOF_MODE=production" in text
        assert "PRIVACY_PIPELINE_ENABLED=true" in text
        assert "PRIVACY_FAIL_CLOSED=true" in text
        assert "PIPELINE_SYNC_REQUIRED=true" in text
        assert "RETRIEVAL_REQUIRE_PRIVACY_SAFE_VIDEO=true" in text
        assert "blueprint_pipeline.production_runtime_env_guard" in text


def test_production_systemd_units_run_nonroot_with_strict_resource_isolation() -> None:
    required_controls = (
        "User=blueprint",
        "Group=blueprint",
        "UMask=0077",
        "NoNewPrivileges=true",
        "PrivateTmp=true",
        "PrivateDevices=true",
        "ProtectSystem=strict",
        "ProtectHome=true",
        "ProtectKernelTunables=true",
        "ProtectKernelModules=true",
        "ProtectKernelLogs=true",
        "ProtectControlGroups=true",
        "RestrictSUIDSGID=true",
        "CapabilityBoundingSet=",
        "AmbientCapabilities=",
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
        "SystemCallFilter=@system-service",
        "ReadWritePaths=/var/lib/blueprint",
        "CPUQuota=200%",
    )
    for unit in SYSTEMD_DIR.glob("*.service"):
        if unit.name in {"blueprint-agent-execution.service", "blueprint-agent-stage-replay.service"}:
            text = unit.read_text(encoding="utf-8")
            narrower = ("RestrictAddressFamilies=", "ReadWritePaths=", "TasksMax=", "MemoryMax=", "CPUQuota=")
            for control in required_controls:
                if not control.startswith(narrower):
                    assert control in text, (unit.name, control)
            assert "TasksMax=128" in text and "CPUQuota=100%" in text
            assert "ReadWritePaths=/var/lib/blueprint/pipeline-control-plane/agent-execution\n" in text
            assert "ReadWritePaths=/var/lib/blueprint\n" not in text
            if unit.name == "blueprint-agent-stage-replay.service":
                assert "PrivateNetwork=true" in text and "RestrictAddressFamilies=AF_UNIX\n" in text
                assert "MemoryMax=4G" in text
                assert "InaccessiblePaths=-/etc/blueprint/provider-secrets" in text
            else:
                assert "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6\n" in text
                assert "MemoryMax=2G" in text
            continue
        if unit.name in {"blueprint-completed-replay-cache-gc.service", "blueprint-scene-project-spend-refresh.service"}:
            text = unit.read_text(encoding="utf-8")
            for control in ("NoNewPrivileges=true", "PrivateDevices=true", "PrivateNetwork=true",
                            "ProtectSystem=strict", "ProtectHome=true", "TasksMax=32", "MemoryMax=512M",
                            "SystemCallFilter=~ptrace process_vm_readv process_vm_writev"):
                assert control in text, (unit.name, control)
            assert "ReadWritePaths=/var/lib/blueprint " not in text
            if unit.name == "blueprint-completed-replay-cache-gc.service":
                assert "User=root" in text
                assert "CapabilityBoundingSet=CAP_DAC_OVERRIDE CAP_SYS_PTRACE" in text
                assert "ReadWritePaths=/var/lib/blueprint/task-evaluation-inputs/stage-replays " in text
            else:
                assert "User=blueprint" in text and "Group=blueprint" in text
                assert "CapabilityBoundingSet=\n" in text and "AmbientCapabilities=\n" in text
                assert "ReadWritePaths=/var/lib/blueprint/pipeline-control-plane/scene-project-spend" in text
                writable = next(line for line in text.splitlines() if line.startswith("ReadWritePaths=")).split("=", 1)[1].split()
                assert set(writable) == {
                    "/var/lib/blueprint/pipeline-control-plane/scene-project-spend",
                    "/var/lib/blueprint/pipeline-control-plane/task-evaluation-scene-intents/.lock",
                }
            continue
        if unit.name in {"blueprint-task-evaluation-episode-compilation-remote.service",
                         "blueprint-team-scene-packet-remote.service"}:
            # Plan 14 §1: the paid remote-compilation unit.  The no-spend compile unit's hardening, but narrow
            # writes instead of /var/lib/blueprint, the dispatcher key through LoadCredential= and a read-only
            # standing authority; it never compiles, so 2G bounds its readback and landing.
            text = unit.read_text(encoding="utf-8")
            for control in required_controls:
                if not control.startswith("ReadWritePaths="):
                    assert control in text, (unit.name, control)
            assert "ReadWritePaths=/var/lib/blueprint " not in text and "ReadWritePaths=/var/lib/blueprint\n" not in text
            writable = [path.removeprefix("-") for path in next(
                line for line in text.splitlines() if line.startswith("ReadWritePaths=")).split("=", 1)[1].split()]
            assert "/var/lib/blueprint" not in writable and all(path.startswith("/var/lib/blueprint/") for path in writable)
            assert "ReadOnlyPaths=/var/lib/blueprint/spend-authority/authorizations\n" in text
            assert "LoadCredential=remote-cpu-dispatcher:/etc/blueprint/credentials/remote-cpu-dispatcher.json" in text
            assert "TasksMax=512" in text and "MemoryMax=2G" in text
            continue
        if unit.name == NOTIFIER_UNIT:
            # The failure notifier: the service account, the full sandbox, and one writable directory
            # for its records; it posts on the capacity controller's webhook, so it keeps the network.
            text = unit.read_text(encoding="utf-8")
            for control in required_controls:
                if not control.startswith(("ReadWritePaths=", "CPUQuota=")):
                    assert control in text, (unit.name, control)
            assert "ReadWritePaths=/var/lib/blueprint/pipeline-control-plane/unit-failures\n" in text
            assert "ReadWritePaths=/var/lib/blueprint\n" not in text and "ReadWritePaths=/var/lib/blueprint " not in text
            assert "TasksMax=32" in text and "MemoryMax=256M" in text and "CPUQuota=100%" in text
            continue
        if unit.name == STATE_BACKUP_UNIT:
            # The state backup: root, because what it copies includes root-owned private records (scene
            # retirement journals and consents, the door's authority, /etc/blueprint), with only the read and
            # owner capabilities that needs, the full sandbox, and writes only to its report root and the site
            # journals where a withdrawal's backup purge receipt goes.
            text = unit.read_text(encoding="utf-8")
            for control in required_controls:
                if not control.startswith(("User=", "Group=", "UMask=", "CapabilityBoundingSet=", "AmbientCapabilities=",
                                           "ReadWritePaths=", "CPUQuota=")):
                    assert control in text, (unit.name, control)
            assert "User=root\n" in text and "UMask=0022\n" in text
            assert "CapabilityBoundingSet=CAP_DAC_READ_SEARCH CAP_DAC_OVERRIDE CAP_CHOWN\n" in text
            writable = next(line for line in text.splitlines() if line.startswith("ReadWritePaths=")).split("=", 1)[1].split()
            assert writable == ["/var/lib/blueprint/pipeline-control-plane/control-plane-backup",
                                "/var/lib/blueprint/pubsub-handoffs"]
            for control in ("TasksMax=32", "MemoryMax=512M", "CPUQuota=50%", "Nice=19", "IOSchedulingPriority=7",
                            "TimeoutStartSec=20min", "OnFailure=" + NOTIFIER_ON_FAILURE, "Slice=" + BATCH_SLICE):
                assert control in text, (unit.name, control)
            assert "-m blueprint_pipeline.control_plane_state_backup run'" in text
            # An unconfigured host skips the run rather than failing it every 15 minutes.
            for name in ("encryption_key", "b2_bucket"):
                condition = f"ConditionPathExists=/etc/blueprint/control-plane-backup/{name}\n"
                assert condition in text and f"_FILE=/etc/blueprint/control-plane-backup/{name}\n" in text
            continue
        if unit.name == WITHDRAWAL_DELETION_UNIT:
            # The withdrawal deletion pass: the service account (it writes beside files the intake and the workers
            # own), the full sandbox, writes only where a withdrawal deletion removes or records, low priority.
            text = unit.read_text(encoding="utf-8")
            for control in required_controls:
                if not control.startswith(("ReadWritePaths=", "CPUQuota=")):
                    assert control in text, (unit.name, control)
            writable = [path.removeprefix("-") for path in next(
                line for line in text.splitlines() if line.startswith("ReadWritePaths=")).split("=", 1)[1].split()]
            assert "/var/lib/blueprint" not in writable and "/etc/blueprint" not in " ".join(writable)
            assert writable[:2] == ["/var/lib/blueprint/pipeline-control-plane/website-withdrawal-deletion",
                                    "/var/lib/blueprint/pubsub-handoffs"]
            assert not any(part in path for path in writable for part in (
                "spend", "ledger", "standing-authorizations", "scene-intents", "website-source-bindings"))
            for control in ("TasksMax=32", "MemoryMax=1G", "CPUQuota=50%", "Nice=19", "IOSchedulingPriority=7",
                            "TimeoutStartSec=45min", "OnFailure=" + NOTIFIER_ON_FAILURE):
                assert control in text, (unit.name, control)
            assert "-m blueprint_pipeline.website_capture_withdrawal_automatic'" in text
            continue
        if unit.name == REAPER_HEARTBEAT_UNIT:
            # The off-host reaper's heartbeat: the service account, the full sandbox, one writable directory
            # for its status record; it publishes to Cloud Storage and pages on the webhook.
            text = unit.read_text(encoding="utf-8")
            for control in required_controls:
                if not control.startswith(("ReadWritePaths=", "CPUQuota=")):
                    assert control in text, (unit.name, control)
            assert "ReadWritePaths=/var/lib/blueprint/pipeline-control-plane/off-host-gpu-reaper\n" in text
            for control in ("TasksMax=32", "MemoryMax=256M", "CPUQuota=100%", "TimeoutStartSec=2min",
                            "OnFailure=" + NOTIFIER_ON_FAILURE):
                assert control in text, (unit.name, control)
            assert "-m blueprint_pipeline.off_host_gpu_reaper_heartbeat'" in text
            continue
        if unit.name == "blueprint-operator-door.service":
            # Cloud agents' HTTPS window onto this host (deploy/operator-door). It
            # reads as the service account, with journal access, in a read-only,
            # loopback-only sandbox that hides every known secret location.
            text = unit.read_text(encoding="utf-8")
            for control in ("User=blueprint", "Group=blueprint",
                            "SupplementaryGroups=systemd-journal blueprint-door\n",
                            "UMask=0077", "NoNewPrivileges=true", "PrivateTmp=true", "PrivateDevices=true",
                            "ProtectSystem=strict", "ProtectHome=true", "ProtectProc=invisible",
                            "ProtectKernelTunables=true", "ProtectKernelModules=true",
                            "ProtectKernelLogs=true", "ProtectControlGroups=true", "RestrictSUIDSGID=true",
                            "CapabilityBoundingSet=\n", "AmbientCapabilities=\n",
                            "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6", "IPAddressDeny=any",
                            "IPAddressAllow=localhost", "SystemCallFilter=@system-service",
                            "ReadWritePaths=/var/lib/blueprint-operator-door\n", "TasksMax=64",
                            "MemoryMax=1G", "CPUQuota=100%"):
                assert control in text, (unit.name, control)
            assert "InaccessiblePaths=-/etc/blueprint/provider-secrets" in text
            assert "ReadWritePaths=/var/lib/blueprint\n" not in text
            continue
        if unit.name == "blueprint-operator-door-runner.service":
            # Root oneshot behind the door's spool: revalidates each request and
            # only runs systemctl --no-block or starts a fixed transient unit. It
            # owns the directories it writes, so it holds no capability at all.
            text = unit.read_text(encoding="utf-8")
            assert "User=root" in text
            assert "CapabilityBoundingSet=\n" in text and "AmbientCapabilities=\n" in text
            assert "StartLimitIntervalSec=0" in text
            for control in ("NoNewPrivileges=true", "ProtectSystem=strict", "ProtectHome=true",
                            "PrivateNetwork=true", "RestrictAddressFamilies=AF_UNIX\n",
                            "SystemCallFilter=@system-service", "TasksMax=32", "MemoryMax=256M",
                            "ReadWritePaths=/var/lib/blueprint-operator-door/requests\n"):
                assert control in text, (unit.name, control)
            continue
        if unit.name == "blueprint-operator-door-hold-sweep.service":
            # Boot reconciliation must ask PID 1 to keep root-owned holds stopped
            # before timer/path targets start; it writes only the door spool.
            text = unit.read_text(encoding="utf-8")
            assert "User=root" in text
            assert "DefaultDependencies=no" in text
            assert "Before=timers.target paths.target" in text
            assert "CapabilityBoundingSet=\n" in text and "AmbientCapabilities=\n" in text
            for control in ("NoNewPrivileges=true", "ProtectSystem=strict", "ProtectHome=true",
                            "PrivateNetwork=true", "RestrictAddressFamilies=AF_UNIX\n",
                            "SystemCallFilter=@system-service", "TasksMax=32", "MemoryMax=256M",
                            "ReadWritePaths=/var/lib/blueprint-operator-door/requests\n"):
                assert control in text, (unit.name, control)
            continue
        if unit.name == "blueprint-control-plane-storage-gc.service":
            text = unit.read_text(encoding="utf-8")
            assert "User=root" in text
            assert "CapabilityBoundingSet=CAP_DAC_OVERRIDE" in text
            assert "AmbientCapabilities=CAP_DAC_OVERRIDE" in text
            assert "NoNewPrivileges=true" in text
            assert "ProtectSystem=strict" in text
            assert "ReadWritePaths=/var/lib/blueprint " not in text
            continue
        if unit.name == "blueprint-control-plane-release-lifecycle.service":
            # Root, because release worktrees are root-owned; writes only the managed trees and receipts.
            text = unit.read_text(encoding="utf-8")
            assert "User=root" in text
            assert "CapabilityBoundingSet=CAP_DAC_OVERRIDE CAP_CHOWN CAP_SYS_PTRACE" in text
            for control in ("NoNewPrivileges=true", "ProtectSystem=strict", "PrivateNetwork=true",
                            "RestrictAddressFamilies=AF_UNIX\n", "TasksMax=64", "MemoryMax=512M"):
                assert control in text, (unit.name, control)
            assert "ReadWritePaths=/var/lib/blueprint " not in text
            assert "ReadWritePaths=/var/lib/blueprint\n" not in text
            continue
        if unit.name in {"blueprint-owner-approved-deletion-plan.service",
                         "blueprint-owner-approved-deletion.service"}:
            # Root oneshots an operator starts by hand: reference checks read root-only pins and every
            # process's descriptors; only the apply unit may write the two allowlisted parents.
            text = unit.read_text(encoding="utf-8")
            assert "User=root" in text
            assert "CapabilityBoundingSet=CAP_DAC_OVERRIDE CAP_SYS_PTRACE\n" in text
            for control in ("NoNewPrivileges=true", "ProtectSystem=strict", "PrivateNetwork=true",
                            "RestrictAddressFamilies=AF_UNIX\n", "InaccessiblePaths=-/etc/blueprint/provider-secrets",
                            "SystemCallFilter=@system-service", "TasksMax=32", "MemoryMax=1G"):
                assert control in text, (unit.name, control)
            writable = next(line for line in text.splitlines()
                            if line.startswith("ReadWritePaths=")).split("=", 1)[1].split()
            expected = ["/var/lib/blueprint/pipeline-control-plane/storage-gc"]
            if unit.name == "blueprint-owner-approved-deletion.service":
                expected = ["/var/lib/blueprint/task-evaluation-inputs", "/var/lib/blueprint/pubsub-handoffs", *expected]
            assert writable == expected, unit.name
            continue
        if unit.name == "blueprint-control-plane-preflight.service":
            # Root housekeeping: replays each unit's sandbox through PID 1 and
            # writes one evidence root; blockers are content, so exit 2 is success.
            text = unit.read_text(encoding="utf-8")
            assert "User=root" in text
            assert "CapabilityBoundingSet=CAP_DAC_OVERRIDE" in text
            assert "AmbientCapabilities=CAP_DAC_OVERRIDE" in text
            assert "NoNewPrivileges=true" in text
            assert "ProtectSystem=strict" in text
            assert "SuccessExitStatus=2" in text
            assert "ReadWritePaths=/var/lib/blueprint/pipeline-control-plane/preflight" in text
            assert "ReadWritePaths=/var/lib/blueprint " not in text
            continue
        if unit.name == "blueprint-task-evaluation-configured-controls-progression.service":
            # Its worker exits 2 whenever a row is blocked; a failed oneshot never
            # re-arms the OnUnitInactiveSec timer, so exit 2 must count as success.
            assert "SuccessExitStatus=2" in unit.read_text(encoding="utf-8")
            timer = unit.with_suffix(".timer").read_text(encoding="utf-8")
            assert "OnUnitInactiveSec=" in timer
        if unit.name == "blueprint-control-plane-capacity.service":
            # Root housekeeping like the reaper: growing the work volume needs the
            # block device and the online-resize ioctl, nothing wider.  It writes
            # one evidence root and reads the reservation ledger and secrets.
            text = unit.read_text(encoding="utf-8")
            assert "User=root" in text
            assert "CapabilityBoundingSet=CAP_DAC_OVERRIDE CAP_SYS_RESOURCE CAP_SYS_ADMIN" in text
            assert "AmbientCapabilities=CAP_DAC_OVERRIDE CAP_SYS_RESOURCE CAP_SYS_ADMIN" in text
            assert "DeviceAllow=block-* rw" in text and "DevicePolicy=closed" in text
            assert "NoNewPrivileges=true" in text
            assert "ProtectSystem=strict" in text
            assert "ReadWritePaths=/var/lib/blueprint/pipeline-control-plane/capacity" in text
            assert "ReadWritePaths=/var/lib/blueprint " not in text
            assert "SystemCallFilter=@system-service" in text
            continue
        text = unit.read_text(encoding="utf-8")
        for control in required_controls:
            assert control in text, (unit.name, control)
        assert "TasksMax=512" in text, unit.name
        # Each unit's cap is in HOST_MEMORY (test_every_control_plane_unit_is_capped_below_the_host); a guard's
        # and a unit this host does not run keep the cap they had.
        memory_limit = (
            HOST_MEMORY[unit.name][2] if unit.name in HOST_MEMORY
            else HOST_MEMORY_KEPT[unit.name][1] if unit.name in HOST_MEMORY_KEPT
            else "8G"
        )
        assert f"MemoryMax={memory_limit}" in text, unit.name

    installer = INSTALL_SCRIPT.read_text(encoding="utf-8")
    assert 'SERVICE_USER="${SERVICE_USER:-blueprint}"' in installer
    assert 'SERVICE_GROUP="${SERVICE_GROUP:-blueprint}"' in installer
    assert 'CREDENTIALS_DIR="${CREDENTIALS_DIR:-${ENV_DIR}/credentials}"' in installer
    assert '"${HANDOFF_DIR}"' in installer
    assert '-m 0750 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}"' in installer
    assert '-m 0750 -o root -g "${SERVICE_GROUP}"' in installer
    assert '"${CREDENTIALS_DIR}"' in installer
    assert 'install -o root -g "${SERVICE_GROUP}" -m 0640' in installer
    assert 'install -d -m 0755 "${SYSTEMD_DIR}" "${ENV_DIR}" "${HANDOFF_DIR}"' not in installer


#: Money safety, and the door to the host (#3329): the OOM killer passes them over. Never weakened: no
#: throttle, no pressure kill, the memory cap each had before. G1 campaign settlement settles money too.
#: Three of them keep 8G, which is above the host and never acts; each file says so. Their peaks are not
#: measured on the host and a guessed cap must not kill a money-safety unit. They are outside BATCH_SLICE.
HOST_SAFETY_GUARD_UNITS = (
    "blueprint-gpu-spend-guard.service",
    "blueprint-operator-door.service",
    "blueprint-task-evaluation-launch-reconciler.service",
    "blueprint-task-evaluation-terminal-resource-release.service",
)
#: Host memory (8.3 GB, no swap). Units with a new cap that binds below the host, so a runaway is killed in
#: its own unit, never host-wide (the kernel's OOM killer took the whole host's view five times on
#: 2026-11-03): (OOMScoreAdjust, MemoryHigh, MemoryMax, highest recorded peak). The peak is the larger of the
#: host's journal since 10-27, docs/research/control-plane-host-profile-2026-10-31.md and a local run at
#: capture-447's scale, and MemoryMax is at least 1.5x it. MemoryHigh, 1G under MemoryMax from 2G up,
#: reclaims a unit's page cache before the host has to. +200: batch jobs, retried; -200: money ledgers and
#: the WebApp's intake, which go after them.
HOST_MEMORY = {
    "blueprint-team-scene-packet.service": ("200", "4G", "5G", "1.9G"),
    "blueprint-task-evaluation-scene-progression.service": ("200", "4G", "5G", "3.2G"),
    "blueprint-control-plane-storage-gc.service": ("200", "2G", "3G", "722M"),
    "blueprint-task-evaluation-launch-preparation.service": ("200", "1G", "2G", "32M"),
    "blueprint-task-evaluation-configured-controls-progression.service": ("200", "1G", "2G", "149M"),
    "blueprint-website-atlas-provider-poll.service": ("200", "1G", "2G", "62M"),
    "blueprint-agent-execution.service": ("200", "1G", "2G", "202M"),
    "blueprint-task-evaluation-episode-compilation.service": ("200", "3G", "4G", "60M"),
    "blueprint-agent-stage-replay.service": ("200", "3G", "4G", "116M"),
    "blueprint-provider-billing-reconciler.service": ("-200", "3G", "4G", "1.7G"),
    "blueprint-pipeline-intake.service": ("-200", "2G", "3G", "1.5G"),
    "blueprint-task-evaluation-episode-compilation-remote.service": ("-200", "1G", "2G", "424M"),
    # Not yet run on the host: the episode remote unit's host peak, on the same allocator, promotion and landing
    # code; landing a capture-447-scale half (1.07 GB) in 1 MiB blocks peaked at 69M locally.
    "blueprint-team-scene-packet-remote.service": ("-200", "1G", "2G", "424M"),
}
#: Units with no throttle of their own and an OOM order. For the batch units among them the heavy step at a
#: full room is not measured on the host (a website capture's preparation in the handoff listener, which
#: passed 4.8 GB locally; a full Atlas run; an allocator child's bundle reads), or a recorded peak times 1.5 does
#: not fit below the host (activation and embodiment preparation, 4.2 GB). Each of those was capped at 8G,
#: above the host, so the cap never acted; now BATCH_SLICE bounds them together and their own cap is the
#: slice's 5.5G. The rest have a cap that is already small. -500 for the door's own units, as the door.
HOST_MEMORY_KEPT = {
    "blueprint-pubsub-handoff-listener.service": ("200", "5.5G"),
    "blueprint-task-evaluation-launch-activation.service": ("200", "5.5G"),
    "blueprint-control-plane-preflight.service": ("200", "1G"),
    "blueprint-control-plane-capacity.service": ("200", "512M"),
    "blueprint-completed-replay-cache-gc.service": ("200", "512M"),
    "blueprint-control-plane-release-lifecycle.service": ("200", "512M"),
    "blueprint-owner-approved-deletion-plan.service": ("200", "1G"),
    "blueprint-owner-approved-deletion.service": ("200", "1G"),
    # Not measured on the host yet: one site's plan, the owner-approved deletion units' cap.
    "blueprint-website-withdrawal-deletion.service": ("200", "1G"),
    # Not measured on the host yet: one file of at most 16 MiB and one 8 MiB part held at a time.
    "blueprint-control-plane-state-backup.service": ("200", "512M"),
    # The off-host reaper's heartbeat (money safety); a run peaked at 122 MB locally.
    "blueprint-off-host-reaper-heartbeat.service": ("-200", "256M"),
    "blueprint-task-evaluation-launch-dispatcher.service": ("-200", "5.5G"),
    "blueprint-website-atlas-preview.service": ("-200", "5.5G"),
    "blueprint-team-evaluation-runner.service": ("-200", "5.5G"),
    "blueprint-beta-policy-screening.service": ("-200", "5.5G"),
    "blueprint-rigid-robot-free-probe-runner.service": ("-200", "5.5G"),
    "blueprint-task-evaluation-sam31-preparation-execution.service": ("-200", "5.5G"),
    "blueprint-agent-run-dispatcher.service": ("-200", "5.5G"),
    "blueprint-scene-project-spend-refresh.service": ("-200", "512M"),
    # The failure notifier carries a page; a run peaked at 27 MB locally.
    "blueprint-unit-failure@.service": ("-200", "256M"),
    "blueprint-operator-door-runner.service": ("-500", "256M"),
    "blueprint-operator-door-hold-sweep.service": ("-500", "256M"),
}


def _directives(unit: str) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for line in (SYSTEMD_DIR / unit).read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values.setdefault(key.strip(), []).append(value.strip())
    return values


def _gigabytes(value: str) -> float:
    return float(value[:-1]) / (1024 if value.endswith("M") else 1)


def _deployed_units() -> tuple[str, ...]:
    import ast

    tree = ast.parse((REPO_ROOT / "scripts" / "deploy_control_plane_commit.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(getattr(t, "id", "") == "DEFAULT_DEPLOYED_SYSTEMD_UNITS" for t in n.targets))
    return tuple(ast.literal_eval(node.value))


def _deployed_services() -> set[str]:
    return {unit for unit in _deployed_units() if unit.endswith(".service")}


def test_every_control_plane_unit_is_capped_below_the_host() -> None:
    services = {unit.name for unit in SYSTEMD_DIR.glob("*.service")}
    covered = set(HOST_MEMORY) | set(HOST_MEMORY_KEPT) | set(HOST_SAFETY_GUARD_UNITS)
    assert not set(HOST_MEMORY) & set(HOST_MEMORY_KEPT)
    assert services == covered
    assert _deployed_services() <= covered
    for unit, (adjust, high, maximum, peak) in HOST_MEMORY.items():
        values = _directives(unit)
        assert values["OOMScoreAdjust"] == [adjust], unit
        assert values["MemoryMax"] == [maximum], unit
        assert values.get("MemoryHigh") == (None if high is None else [high]), unit
        # Binds below the host's 8.3 GB with the always-on services' 2-3 GB left over, and never under 1.5x
        # the unit's highest recorded peak.
        assert 1.5 * _gigabytes(peak) <= _gigabytes(maximum) <= 5, unit
        assert high == (None if _gigabytes(maximum) < 2 else f"{int(_gigabytes(maximum)) - 1}G"), unit
        assert not {"ManagedOOMMemoryPressure", "ManagedOOMSwap"} & set(values), unit
    for unit, (adjust, maximum) in HOST_MEMORY_KEPT.items():
        values = _directives(unit)
        assert values["OOMScoreAdjust"] == [adjust], unit
        assert values["MemoryMax"] == [maximum], unit
        assert not {"MemoryHigh", "ManagedOOMMemoryPressure", "ManagedOOMSwap"} & set(values), unit
    for unit in HOST_SAFETY_GUARD_UNITS:
        values = _directives(unit)
        assert values["OOMScoreAdjust"] == ["-500"], unit
        assert not {"MemoryHigh", "ManagedOOMMemoryPressure", "ManagedOOMSwap"} & set(values), unit
        assert values["MemoryMax"] == (["1G"] if unit == "blueprint-operator-door.service"
                                       else ["8G"]), unit


#: The one memory slice for the batch units, and the failure notifier (docs/SYSTEMD_SECURITY_BASELINE.md).
BATCH_SLICE = "system-blueprintbatch.slice"
NOTIFIER_UNIT = "blueprint-unit-failure@.service"
NOTIFIER_ON_FAILURE = "blueprint-unit-failure@%n.service"
WITHDRAWAL_DELETION_UNIT = "blueprint-website-withdrawal-deletion.service"
STATE_BACKUP_UNIT = "blueprint-control-plane-state-backup.service"
REAPER_HEARTBEAT_UNIT = "blueprint-off-host-reaper-heartbeat.service"
#: Every worker that moves a job's payload: it decodes video, fuses or loads scenes, compiles episodes or
#: stages GPU bundles. BATCH_SLICE bounds them together.
BATCH_SLICE_UNITS = (
    "blueprint-pubsub-handoff-listener.service",
    "blueprint-website-atlas-preview.service",
    "blueprint-website-atlas-provider-poll.service",
    "blueprint-task-evaluation-scene-progression.service",
    "blueprint-task-evaluation-launch-preparation.service",
    "blueprint-task-evaluation-launch-activation.service",
    "blueprint-task-evaluation-launch-dispatcher.service",
    "blueprint-team-scene-packet.service",
    "blueprint-team-scene-packet-remote.service",
    "blueprint-rigid-robot-free-probe-runner.service",
    "blueprint-beta-policy-screening.service",
    "blueprint-task-evaluation-sam31-preparation-execution.service",
    "blueprint-team-evaluation-runner.service",
    "blueprint-agent-run-dispatcher.service",
    "blueprint-task-evaluation-episode-compilation.service",
    "blueprint-task-evaluation-episode-compilation-remote.service",
    "blueprint-task-evaluation-configured-controls-progression.service",
    "blueprint-agent-stage-replay.service",
    # Not a job mover, but bulk reads and uploads every 15 minutes: bounded with them so it never starves the host.
    "blueprint-control-plane-state-backup.service",
)
#: Every other service stays outside the slice, for the reason given: it must keep running, or keep money
#: safe, while the batch units sit at their bound.
OUTSIDE_BATCH_SLICE = {
    "blueprint-pipeline-intake.service": "always on: the WebApp's signed intake",
    "blueprint-agent-execution.service": "always on: a continuous service",
    "blueprint-operator-door.service": "always on: the door to the host",
    "blueprint-operator-door-runner.service": "the door's runner",
    "blueprint-operator-door-hold-sweep.service": "the door's hold sweep",
    "blueprint-gpu-spend-guard.service": "money safety: the spend guard and orphan reaper",
    "blueprint-provider-billing-reconciler.service": "money safety: the billing feed the guard reads",
    "blueprint-scene-project-spend-refresh.service": "money safety: the scene spend ledger's refresh",
    "blueprint-task-evaluation-launch-reconciler.service": "money safety: the provider-zero proof",
    "blueprint-task-evaluation-terminal-resource-release.service": "money safety: releases paid resources",
    "blueprint-control-plane-capacity.service": "the pager: capacity, stall and spend alerts",
    NOTIFIER_UNIT: "the failure notifier: it must run while the batch units are at their bound",
    "blueprint-control-plane-storage-gc.service": "housekeeping that frees the disk batch units wait for",
    "blueprint-completed-replay-cache-gc.service": "housekeeping, capped at 512M",
    "blueprint-control-plane-release-lifecycle.service": "housekeeping, capped at 512M",
    "blueprint-control-plane-preflight.service": "a diagnostic, capped at 1G",
    "blueprint-owner-approved-deletion-plan.service": "started by hand, capped at 1G",
    "blueprint-owner-approved-deletion.service": "started by hand, capped at 1G",
    "blueprint-website-withdrawal-deletion.service": "privacy housekeeping at low priority, capped at 1G",
    REAPER_HEARTBEAT_UNIT: "money safety: the off-host GPU reaper's heartbeat",
}
#: Units whose failure is abnormal and matters: the money guard, the intake, the pager and the units that
#: move a website job. Each starts the notifier when it fails.
UNIT_FAILURE_NOTIFIED_UNITS = (
    "blueprint-gpu-spend-guard.service",
    "blueprint-pipeline-intake.service",
    "blueprint-control-plane-capacity.service",
    "blueprint-pubsub-handoff-listener.service",
    "blueprint-website-atlas-preview.service",
    "blueprint-website-atlas-provider-poll.service",
    "blueprint-task-evaluation-scene-progression.service",
    "blueprint-task-evaluation-launch-preparation.service",
    "blueprint-task-evaluation-launch-activation.service",
    "blueprint-task-evaluation-launch-dispatcher.service",
    "blueprint-team-scene-packet.service",
    "blueprint-rigid-robot-free-probe-runner.service",
    "blueprint-beta-policy-screening.service",
    # A privacy promise: a withdrawn site's deletion must not stop silently.
    "blueprint-website-withdrawal-deletion.service",
    # An incomplete backup, or a withdrawal purge that could not be proven, must not stop silently.
    "blueprint-control-plane-state-backup.service",
    # Money safety: without it the off-host reaper reads the host as gone.
    "blueprint-off-host-reaper-heartbeat.service",
)
#: Units on the website job's path that are left without the notifier on purpose: there a non-zero exit is an
#: ordinary result, so the notifier would report routine work every hour.
UNIT_FAILURE_ROUTINE_EXIT_UNITS = {
    "blueprint-task-evaluation-launch-reconciler.service":
        "exits 2 while any teardown is still being proven (provider_zero_pending, recovery_pending)",
    "blueprint-provider-billing-reconciler.service":
        "exits 2 when a provider read is blocked; the spend guard then closes admission and reports it",
}
#: Units that keep an independent GPU watchdog child alive past their own stop or timeout.
KILL_MODE_PROCESS_UNITS = (
    "blueprint-agent-run-dispatcher.service",
    "blueprint-beta-policy-screening.service",
    "blueprint-pubsub-handoff-listener.service",
    "blueprint-rigid-robot-free-probe-runner.service",
    "blueprint-task-evaluation-launch-dispatcher.service",
    "blueprint-task-evaluation-sam31-preparation-execution.service",
    "blueprint-team-evaluation-runner.service",
)


def _timeout_seconds(value: str) -> int | None:
    """A bounded systemd time span in seconds; None for anything else (infinity, 0, a typo)."""
    match = re.fullmatch(r"(\d+)(s|m|min|h|d)?", value)
    if match is None:
        return None
    return int(match[1]) * {None: 1, "s": 1, "m": 60, "min": 60, "h": 3600, "d": 86400}[match[2]] or None


def test_a_hung_spend_guard_or_handoff_listener_run_ends_without_ending_real_work() -> None:
    guard = _directives("blueprint-gpu-spend-guard.service")
    guard_bound = _timeout_seconds(guard["TimeoutStartSec"][0])
    environment = dict(row.split("=", 1) for row in guard["Environment"])
    # Longer than the slowest run the refresh contract above models (600 s), and far shorter than the
    # age at which the guard itself reaps a booted orphan: a hang cannot outlast what it is there to end.
    assert 600 < guard_bound < int(environment["BLUEPRINT_GPU_SPEND_GUARD_MAX_BOOTED_ORPHAN_SECONDS"])
    assert guard_bound <= 30 * 60
    assert "KillMode" not in guard, "the guard starts no child that must outlive it"

    from blueprint_pipeline.website_source_video import _ENCODE_TIMEOUT_SECONDS

    listener = _directives("blueprint-pubsub-handoff-listener.service")
    listener_bound = _timeout_seconds(listener["TimeoutStartSec"][0])
    # Its longest capped step is the one-hour normalization encode; a tick may hold several hand-offs.
    assert 5 * _ENCODE_TIMEOUT_SECONDS < listener_bound <= 24 * 3600
    assert listener["KillMode"] == ["process"], "a timeout stops the worker, never its geometry watchdog"


def test_units_that_keep_a_watchdog_child_alive_still_do() -> None:
    services = {unit.name for unit in SYSTEMD_DIR.glob("*.service")}
    assert {unit for unit in services if _directives(unit).get("KillMode") == ["process"]} == set(
        KILL_MODE_PROCESS_UNITS
    )
    assert {unit for unit in services if "KillMode" in _directives(unit)} == set(KILL_MODE_PROCESS_UNITS)


def test_batch_units_are_in_the_slice_and_always_on_and_money_units_are_not() -> None:
    services = {unit.name for unit in SYSTEMD_DIR.glob("*.service")}
    assert len(set(BATCH_SLICE_UNITS)) == len(BATCH_SLICE_UNITS)
    assert not set(BATCH_SLICE_UNITS) & set(OUTSIDE_BATCH_SLICE)
    assert set(BATCH_SLICE_UNITS) | set(OUTSIDE_BATCH_SLICE) == services, "every service is placed"
    assert all(OUTSIDE_BATCH_SLICE.values()), "each unit outside the slice says why"
    slice_maximum = _gigabytes(_directives(BATCH_SLICE)["MemoryMax"][0])
    for unit in BATCH_SLICE_UNITS:
        values = _directives(unit)
        assert values.get("Slice") == [BATCH_SLICE], unit
        # A unit's own cap above the slice's would never act; 8G did not act at all.
        assert _gigabytes(values["MemoryMax"][0]) <= slice_maximum, unit
        # The slice bounds the sum; which unit goes first inside it is still each unit's own order.
        assert values["OOMScoreAdjust"] in (["200"], ["-200"]), unit
    for unit in OUTSIDE_BATCH_SLICE:
        assert "Slice" not in _directives(unit), unit
    for unit in ("blueprint-pipeline-intake.service", "blueprint-operator-door.service",
                 "blueprint-operator-door-runner.service", "blueprint-gpu-spend-guard.service",
                 "blueprint-provider-billing-reconciler.service", "blueprint-control-plane-capacity.service",
                 *HOST_SAFETY_GUARD_UNITS):
        assert unit in OUTSIDE_BATCH_SLICE, unit
    for unit in ("blueprint-pubsub-handoff-listener.service", "blueprint-website-atlas-preview.service",
                 "blueprint-website-atlas-provider-poll.service",
                 "blueprint-task-evaluation-scene-progression.service",
                 "blueprint-task-evaluation-launch-dispatcher.service",
                 "blueprint-task-evaluation-launch-activation.service",
                 "blueprint-task-evaluation-launch-preparation.service",
                 "blueprint-team-scene-packet.service", "blueprint-rigid-robot-free-probe-runner.service",
                 "blueprint-beta-policy-screening.service"):
        assert unit in BATCH_SLICE_UNITS, unit
    # The five money-safety units that keep 8G say that it never acts.
    capped_above_the_host = sorted(
        unit for unit in services if _directives(unit)["MemoryMax"] == ["8G"]
    )
    assert capped_above_the_host == sorted(
        unit for unit in HOST_SAFETY_GUARD_UNITS
        if unit != "blueprint-operator-door.service"
    )
    for unit in capped_above_the_host:
        assert "this cap never acts" in _read(unit), unit


def test_units_whose_failure_matters_start_the_failure_notifier() -> None:
    template = _read(NOTIFIER_UNIT)
    assert "OnFailure" not in _directives(NOTIFIER_UNIT), "the notifier never reports itself: no loop"
    assert '-m blueprint_pipeline.unit_failure_notifier --unit "%i"' in template
    # The capacity controller's alert route: the same environment file names the webhook for both.
    route = "EnvironmentFile=-/etc/blueprint/pipeline-control-plane.env"
    assert route in template and route in _read("blueprint-control-plane-capacity.service")
    assert _timeout_seconds(_directives(NOTIFIER_UNIT)["TimeoutStartSec"][0]) <= 5 * 60

    notified = set()
    for unit in sorted(path for suffix in ("service", "timer", "path") for path in SYSTEMD_DIR.glob(f"*.{suffix}")):
        values = _directives(unit.name)
        if "OnFailure" not in values:
            continue
        assert values["OnFailure"] == [NOTIFIER_ON_FAILURE], unit.name
        text = unit.read_text(encoding="utf-8")
        assert text.index("\nOnFailure=") < text.index("\n[Service]"), unit.name
        notified.add(unit.name)
    assert notified == set(UNIT_FAILURE_NOTIFIED_UNITS)
    for unit in UNIT_FAILURE_NOTIFIED_UNITS:
        # A non-zero exit is abnormal in each of them; none turns an exit status into success.
        assert "SuccessExitStatus" not in _directives(unit), unit
    assert not notified & set(UNIT_FAILURE_ROUTINE_EXIT_UNITS)
    assert all(UNIT_FAILURE_ROUTINE_EXIT_UNITS.values())
    assert all((SYSTEMD_DIR / unit).is_file() for unit in UNIT_FAILURE_ROUTINE_EXIT_UNITS)


def test_terraform_gpu_warm_pool_defaults_stay_scale_to_zero():
    # SCALE2-06: warm pools are a deliberate owner decision, not a default.
    # One warm GPU instance is ~$1,825/month at the $2.5/GPU-hr planning rate
    # (73% of the $2,500 spend-review threshold); current volume is >10x
    # below the ~20 invocations/hour breakeven. See
    # docs/GPU_WARM_POOL_ECONOMICS_2026-07-20.md.
    text = TERRAFORM_MAIN.read_text(encoding="utf-8")

    for variable in (
        "privacy_sam3_min_instances",
        "privacy_vip_min_instances",
        "privacy_deepprivacy2_min_instances",
        "video_to_world_min_instances",
    ):
        block = _terraform_resource_body_for_variable(text, variable)
        assert "default     = 0" in block, f"{variable} must default to scale-to-zero"
        assert "<= 2" in block, f"{variable} must stay capped at 2 warm instances"

    for key in ("sam3", "vip", "deepprivacy2", "video_to_world"):
        assert f"min_instance_count = local.privacy_runner_min_instances.{key}" in text
    # No GPU service may hardcode an always-warm floor.
    assert "min_instance_count = 1" not in text
    assert "min_instance_count = 2" not in text


def _terraform_resource_body_for_variable(text: str, name: str) -> str:
    marker = f'variable "{name}"'
    start = text.index(marker)
    end = text.index("\nvariable ", start + 1) if "\nvariable " in text[start + 1 :] else len(text)
    return text[start:end]


def test_terraform_privacy_runners_are_private_and_invoked_by_named_principals():
    text = TERRAFORM_MAIN.read_text(encoding="utf-8")

    assert "privacy_runner_public_invoker" not in text
    assert 'member   = "allUsers"' not in text
    assert 'member   = "allAuthenticatedUsers"' not in text
    assert 'resource "google_cloud_run_service_iam_member" "privacy_runner_invoker"' in text
    assert "serviceAccount:${google_service_account.pipeline_runner.email}" in text
    assert 'variable "additional_privacy_runner_invoker_members"' in text
    assert 'name  = "BLUEPRINT_CLOUD_RUN_IAM_AUTH_ENABLED"' in text
    assert 'value = "true"' in text


def test_privacy_runner_auth_tokens_fail_closed_before_deploy() -> None:
    terraform = TERRAFORM_MAIN.read_text(encoding="utf-8")
    deploy_script = PIPELINE_DEPLOY_SCRIPT.read_text(encoding="utf-8")
    privacy_secret = _terraform_variable_body(terraform, "privacy_runner_token_secret_name")
    video_secret = _terraform_variable_body(terraform, "video_to_world_runner_token_secret_name")

    assert 'var.deployment_scope == "remote_cpu" && var.privacy_runner_token_secret_name == ""' in privacy_secret
    assert "^[A-Za-z0-9_-]{1,255}$" in privacy_secret
    assert "video_to_world_runner_token_secret_name" in video_secret
    assert 'variable "privacy_runner_token"' not in terraform
    assert 'variable "pipeline_sync_token"' not in terraform
    assert "value = var.privacy_runner_token" not in terraform
    assert "value = var.pipeline_sync_token" not in terraform
    assert "value = local.video_to_world_runner_token" not in terraform
    assert 'name = "WORLDLABS_API_KEY"' in terraform
    assert "data.google_secret_manager_secret.worldlabs_api_key.secret_id" in terraform
    assert "validate_runtime_secret_references" in deploy_script
    assert "PRIVACY_RUNNER_TOKEN_SECRET_NAME" in deploy_script
    assert "VIDEO_TO_WORLD_RUNNER_TOKEN_SECRET_NAME" in deploy_script
    assert "PIPELINE_SYNC_TOKEN_SECRET_NAME" in deploy_script
    assert "WORLDLABS_API_KEY_SECRET_NAME" in deploy_script
    assert "TF_VAR_privacy_runner_token_secret_name" in deploy_script
    assert "TF_VAR_pipeline_sync_token_secret_name" in deploy_script
    assert "PRIVACY_RUNNER_TOKEN=" not in deploy_script
    assert "PIPELINE_SYNC_TOKEN=" not in deploy_script
    assert "WORLDLABS_API_KEY=" not in deploy_script
    assert "Creating terraform.tfvars" not in deploy_script


def test_us_only_beta_residency_is_enforced_as_code() -> None:
    terraform = TERRAFORM_MAIN.read_text(encoding="utf-8")
    tfvars_example = TERRAFORM_TFVARS_EXAMPLE.read_text(encoding="utf-8")
    deploy_script = PIPELINE_DEPLOY_SCRIPT.read_text(encoding="utf-8")
    primary = _terraform_variable_body(terraform, "primary_region")
    secondary = _terraform_variable_body(terraform, "secondary_regions")

    assert 'startswith(var.primary_region, "us-")' in primary
    assert 'startswith(region, "us-")' in secondary
    assert "europe-west1" not in secondary
    assert "europe-west1" not in tfvars_example
    assert "europe-west1" not in deploy_script
    assert "validate_beta_data_residency" in deploy_script
    assert '"$PRIMARY_REGION" != us-*' in deploy_script


def test_terraform_privacy_runners_have_per_service_gpu_caps_and_spend_alert():
    text = TERRAFORM_MAIN.read_text(encoding="utf-8")

    for variable in (
        "privacy_sam3_max_instances",
        "privacy_vip_max_instances",
        "privacy_deepprivacy2_max_instances",
        "video_to_world_max_instances",
        "gpu_runner_billable_instance_time_alert_threshold",
    ):
        assert f'variable "{variable}"' in text

    service_caps = {
        "privacy_sam3": "local.privacy_runner_max_instances.sam3",
        "privacy_vip": "local.privacy_runner_max_instances.vip",
        "privacy_deepprivacy2": "local.privacy_runner_max_instances.deepprivacy2",
        "video_to_world": "local.privacy_runner_max_instances.video_to_world",
    }
    for service, expected_cap in service_caps.items():
        body = _terraform_resource_body(text, "google_cloud_run_v2_service", service)
        assert f"max_instance_count = {expected_cap}" in body
        assert "max_instance_count = var.max_concurrent_jobs" not in body

    assert 'resource "google_monitoring_alert_policy" "gpu_runner_billable_instance_time"' in text
    assert "run.googleapis.com/container/billable_instance_time" in text
    assert "cloud_run_revision" in text
    assert "local.privacy_runner_monitoring_service_filter" in text
    assert "Sustained GPU runner billable instance time" in text


def test_terraform_declares_optional_gcp_billing_budget_for_gpu_fleet():
    text = TERRAFORM_MAIN.read_text(encoding="utf-8")
    body = _terraform_resource_body(text, "google_billing_budget", "gpu_fleet_beta")

    assert '"billingbudgets.googleapis.com"' in text
    assert 'variable "billing_account_id"' in text
    assert 'variable "gpu_fleet_billing_budget_usd"' in text
    assert 'variable "gpu_fleet_billing_budget_thresholds"' in text
    assert 'data "google_project" "current"' in text
    assert 'count = var.billing_account_id != "" ? 1 : 0' in body
    assert "billing_account = var.billing_account_id" in body
    assert 'display_name    = "Blueprint GPU Fleet Beta Budget"' in body
    assert 'projects = ["projects/${data.google_project.current.number}"]' in body
    assert 'currency_code = "USD"' in body
    assert "units         = tostring(var.gpu_fleet_billing_budget_usd)" in body
    assert "threshold_percent = threshold_rules.value" in body
    assert 'output "gpu_fleet_billing_budget"' in text


def test_systemd_spend_guard_enforces_5000_admission_lock_and_page_path():
    service = _read("blueprint-gpu-spend-guard.service")
    billing_service = _read("blueprint-provider-billing-reconciler.service")
    billing_timer = _read("blueprint-provider-billing-reconciler.timer")
    env_example = _read("pipeline-control-plane.env.example")
    installer = INSTALL_SCRIPT.read_text(encoding="utf-8")

    assert "BLUEPRINT_GPU_FLEET_MAX_TOTAL_SPEND_USD=5000.0" in service
    assert "BLUEPRINT_GPU_SPEND_GUARD_OUTPUT_ROOT=/var/lib/blueprint/pipeline-control-plane" in service
    assert '--output-root "$${BLUEPRINT_GPU_SPEND_GUARD_OUTPUT_ROOT}"' in service
    assert "BLUEPRINT_GPU_FLEET_MAX_TOTAL_SPEND_USD=500.0" not in service
    assert "--require-billing-reconciliation" in service
    assert "--admission-lock-report" in service
    assert "--admission-override" in service
    assert "ExecStopPost=" in service
    assert "blueprint_pipeline.live_pipeline_manifest_alert" in service
    assert "BLUEPRINT_GPU_SPEND_GUARD_PAGE_AUDIT" in service
    assert "BLUEPRINT_GPU_PROVIDER_SECRETS_DIR=/etc/blueprint/provider-secrets" in service
    assert "BLUEPRINT_REQUIRE_PAID_SPEND_ADMISSION_LOCK=true" in env_example
    assert "BLUEPRINT_GPU_BILLING_EXPORT=" in env_example
    assert "blueprint-gpu-spend-guard.service" in installer
    assert "blueprint-gpu-spend-guard.timer" in installer
    assert "systemctl enable --now blueprint-gpu-spend-guard.timer" in installer
    assert '"${STATE_DIR}/gpu_spend_guard"' in installer
    assert '"${PROVIDER_SECRETS_DIR}"' in installer
    assert "blueprint-provider-billing-reconciler.service" in service
    assert "blueprint_pipeline.provider_billing_reconciler" in billing_service
    assert "ReadOnlyPaths=/etc/blueprint/provider-secrets" in billing_service
    assert "Before=blueprint-gpu-spend-guard.service" in billing_service
    assert "OnUnitInactiveSec=10m" in billing_timer
    assert "blueprint-provider-billing-reconciler.service" in installer
    assert "blueprint-provider-billing-reconciler.timer" in installer
    assert "systemctl enable --now blueprint-provider-billing-reconciler.timer" in installer
    assert "BLUEPRINT_PROVIDER_BILLING_COHORT_START_AT=2026-01-01T00:00:00Z" in env_example
    for unit in (service, billing_service):
        assert "BLUEPRINT_PIPELINE_REPO" not in unit
        assert (
            "BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_REPO="
            "/opt/blueprint/task-evaluation-control-plane"
        ) in unit
        assert (
            "BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_PYTHON="
            "/opt/blueprint/BlueprintCapturePipeline/.venv/bin/python"
        ) in unit


def test_spend_guard_refresh_leaves_a_bounded_idle_window_after_slow_runs():
    # systemd's documented monotonic clocks use activation or deactivation as
    # their base. The initial boot trigger has already fired in these cases.
    settings = dict(
        line.split("=", 1)
        for line in _read("blueprint-gpu-spend-guard.timer").splitlines()
        if line and not line.startswith(("#", "["))
    )

    def seconds(value):
        match = re.fullmatch(r"(\d+)(s|min)", value)
        assert match is not None, "refresh interval must have a bounded time span"
        return int(match[1]) * {"s": 1, "min": 60}[match[2]]

    for runtime in (30, 208, 600):
        activated = 10_000
        completed = activated + runtime
        bases = {"OnUnitActiveSec": activated, "OnUnitInactiveSec": completed}
        deadlines = [base + seconds(settings[key]) for key, base in bases.items() if key in settings]
        assert deadlines, "the spend guard must retain its periodic refresh"
        idle_window = min(deadlines) - completed
        assert 0 < idle_window <= 120, (
            "a slow guard refresh must leave waiting workers an idle window, "
            "then refresh again within two minutes"
        )
    # Paid admission accepts a report at most five minutes old. A typical slow run
    # (about the observed 90th percentile) plus the wait must refresh within it,
    # and a deploy that restarts the timer must not wait for a full cycle.
    cycle = 208 + seconds(settings["OnUnitInactiveSec"]) + seconds(settings["RandomizedDelaySec"])
    assert cycle <= 300
    assert seconds(settings["OnActiveSec"]) <= 60


def test_website_capture_worker_loads_deployed_geometry_runtime(monkeypatch):
    service = _read("blueprint-pubsub-handoff-listener.service")
    assert service.index("EnvironmentFile=-/etc/blueprint/pipeline-control-plane.env") < service.index(
        "EnvironmentFile=-/etc/blueprint/task-evaluation-scene-configuration-release.env"
    )
    from blueprint_pipeline import vast_independent_watchdog_control as watchdog
    assert "KillMode=process" in service
    declaration = next(line.split("=", 2)[2] for line in service.splitlines()
                       if line.startswith(f"Environment={watchdog.CALLER_EXIT_SURVIVAL_ENV}="))
    monkeypatch.setenv("INVOCATION_ID", "website-service")
    monkeypatch.setenv(watchdog.CALLER_EXIT_SURVIVAL_ENV, declaration)
    assert watchdog._caller_exit_survival_proven(watchdog._caller_exit_survival_contract())


def test_paid_units_enable_the_provider_credit_guard_and_the_controller_can_read_credit():
    """The per-attempt credit guard and the hourly credit alert are on by default in code
    (BLUEPRINT_VAST_CREDIT_GUARD_ENABLED; only false/0 opts out). The unit files that
    reach the Vast adapter still set it explicitly, and the chain preflight refuses a
    paid unit without it, so an opt-out is always visible."""

    from blueprint_pipeline.task_evaluation_production_chain_preflight import PAID_ALLOCATION_UNITS

    flag = "Environment=BLUEPRINT_VAST_CREDIT_GUARD_ENABLED=true"
    # The two paid lanes website jobs use, which once ran with the guard off.
    assert {"blueprint-rigid-robot-free-probe-runner.service",
            "blueprint-beta-policy-screening.service"} <= set(PAID_ALLOCATION_UNITS)
    for name in PAID_ALLOCATION_UNITS:
        text = (SYSTEMD_DIR / name).read_text(encoding="utf-8")
        assert flag in text, name
        assert "BLUEPRINT_VAST_CREDIT_GUARD_ENABLED=false" not in text, name
    # A unit that can launch a Vast instance (it arms the caller-exit watchdog or holds
    # the instance SSH key) is a paid allocation unit, so the preflight checks its guard.
    launching = sorted(
        unit.name for unit in SYSTEMD_DIR.glob("blueprint-*.service")
        if re.search(r"^Environment=BLUEPRINT_VAST_(WATCHDOG_CALLER_EXIT_SURVIVAL|SSH_IDENTITY_FILE)=",
                     unit.read_text(encoding="utf-8"), re.M)
    )
    assert launching and sorted(set(launching) - set(PAID_ALLOCATION_UNITS)) == []
    capacity = (SYSTEMD_DIR / "blueprint-control-plane-capacity.service").read_text(encoding="utf-8")
    assert flag in capacity
    assert "Environment=VAST_API_KEY_FILE=/etc/blueprint/provider-secrets/vast_api_key" in capacity
    assert "ReadOnlyPaths=/etc/blueprint/provider-secrets" in capacity
    assert "EnvironmentFile=-/etc/blueprint/task-evaluation-scene-progression.env" in capacity


def test_probe_and_screening_attempts_stage_on_the_work_volume_and_stay_watched() -> None:
    """The root disk sits at its floor, so the probe's and screening's attempt bulk (bundles, the
    allocator job directory and its policy_canary_output hold) goes to the work volume. The queues
    stay on the root disk; the spend guard still finds the attempts' owner records; the deploy
    provisions both roots; the storage table classifies them."""

    from blueprint_pipeline.beta_policy_screening_coordinator import WORK_ROOT_ENV as SCREENING_WORK_ENV
    from blueprint_pipeline.control_plane_storage_roots import classify_path
    from blueprint_pipeline.rigid_robot_free_probe_runner import WORK_ENV as PROBE_WORK_ENV
    from tests.test_deploy_control_plane_commit import deploy

    guard = _read("blueprint-gpu-spend-guard.service")
    for unit, env, root, queue, storage_class in (
        ("blueprint-rigid-robot-free-probe-runner.service", PROBE_WORK_ENV,
         "/mnt/blueprint-work/rigid-robot-free-probe-work",
         "BLUEPRINT_RIGID_ROBOT_FREE_PROBE_QUEUE=/var/lib/blueprint/pipeline-control-plane/rigid-robot-free-probes",
         "work"),
        ("blueprint-beta-policy-screening.service", SCREENING_WORK_ENV,
         "/mnt/blueprint-work/beta-policy-screening-work",
         "BLUEPRINT_BETA_POLICY_SCREENING_QUEUE_ROOT=/var/lib/blueprint/pipeline-control-plane/beta-policy-screening",
         "evidence_hot"),
    ):
        text = _read(unit)
        assert f"Environment={env}={root}\n" in text, unit
        assert f"Environment={queue}\n" in text, unit
        rw = [token for line in text.splitlines() if line.startswith("ReadWritePaths=")
              for token in line.removeprefix("ReadWritePaths=").split()]
        assert root in rw, unit
        assert f"--output-root {root} " in guard and f"ReadOnlyPaths=-{root}\n" in guard
        assert root in deploy._WORK_VOLUME_PROVISIONABLE
        classified = classify_path(root + "/attempts/x")
        assert classified is not None and classified.path == root and classified.storage_class == storage_class


def test_launch_activation_unit_outlasts_its_preparation_command(tmp_path, monkeypatch):
    """2026-10-31: a scene-configuration bundle was still being written when the 1h unit timeout fired."""
    import json
    from blueprint_pipeline import task_evaluation_launch_activation_worker as worker

    seen = {}
    receipt = tmp_path / "receipt.json"

    def fake_run(command, **kwargs):
        seen.update(kwargs)
        receipt.write_text(json.dumps({"status": "prepared"}), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(worker.subprocess, "run", fake_run)
    worker.default_activation_preparer(
        lane="lane", context_path=tmp_path / "context.json", receipt_path=receipt, repository_root=tmp_path)
    text = _read("blueprint-task-evaluation-launch-activation.service")
    timeouts = [line.split("=", 1)[1] for line in text.splitlines() if line.startswith("TimeoutStartSec=")]
    assert seen["timeout"] == 4 * 3600
    assert timeouts == ["255min"] and seen["timeout"] < 255 * 60
