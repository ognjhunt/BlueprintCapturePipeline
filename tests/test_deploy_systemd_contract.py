import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEMD_DIR = REPO_ROOT / "deploy" / "systemd"
TERRAFORM_MAIN = REPO_ROOT / "deploy" / "terraform" / "main.tf"
TERRAFORM_TFVARS_EXAMPLE = REPO_ROOT / "deploy" / "terraform" / "terraform.tfvars.example"
ROOT_DOCKERFILE = REPO_ROOT / "Dockerfile"
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "install_live_pipeline_control_plane.sh"
PIPELINE_DEPLOY_SCRIPT = REPO_ROOT / "deploy" / "scripts" / "deploy.sh"
FULL_TEST_LANE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "full-test-lane.yml"
CI_REQUIRED_CHECKS_DOC = REPO_ROOT / "docs" / "CI_REQUIRED_CHECKS.md"
ACTIVE_LAUNCH_DOCS_AND_COMMAND_SOURCES = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "PAID_MARKETPLACE_BETA_LAUNCH_GATE.md",
    REPO_ROOT / "docs" / "FIRST_GPU_E2E_RUNBOOK.md",
    REPO_ROOT / "src" / "blueprint_pipeline" / "g1_controlled_proof_setup.py",
    REPO_ROOT / "src" / "blueprint_pipeline" / "wam_vla_policy_endpoint_setup.py",
)


def _read(name: str) -> str:
    return (SYSTEMD_DIR / name).read_text(encoding="utf-8")


def test_scene_progression_uses_pinned_checkout_and_config_gated_periodic_worker():
    unit = _read("blueprint-task-evaluation-scene-progression.service")
    assert "User=blueprint" in unit
    assert "ProtectSystem=strict" in unit and "NoNewPrivileges=true" in unit
    assert "ConditionPathExists=/etc/blueprint/task-evaluation-scene-progression.json" in unit
    assert "exec env PYTHONPATH=src" in unit
    assert "-m blueprint_pipeline.task_evaluation_scene_retirement_supervisor " in unit
    assert "--worker blueprint_pipeline.task_evaluation_scene_progression -- --config " in unit
    assert "paid_resource_allocator" not in unit
    queue = "Environment=BLUEPRINT_TASK_EVALUATION_SCENE_CONSTRUCTION_QUEUE_ROOT=/var/lib/blueprint/pipeline-control-plane/task-evaluation-scene-constructions"
    assert queue in unit
    assert queue in _read("blueprint-task-evaluation-launch-dispatcher.service")
    assert "Environment=VAST_API_KEY_FILE=/etc/blueprint/provider-secrets/vast_api_key" in unit
    assert "OnUnitInactiveSec=1min" in _read("blueprint-task-evaluation-scene-progression.timer")
    installer = INSTALL_SCRIPT.read_text()
    for suffix in ("service", "timer"):
        assert "blueprint-task-evaluation-scene-progression." + suffix in installer


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


def test_active_launch_docs_and_generated_commands_do_not_embed_laptop_paths():
    for path in ACTIVE_LAUNCH_DOCS_AND_COMMAND_SOURCES:
        text = path.read_text(encoding="utf-8")
        assert re.search(r"/Users/[^/]+/", text) is None, (
            f"personal absolute path leaked into {path.relative_to(REPO_ROOT)}"
        )


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
        task_limit = {"blueprint-existing-policy-canary-watchdog.service": 64,
                      "blueprint-existing-policy-canary-continuation.service": 128}.get(unit.name, 512)
        assert f"TasksMax={task_limit}" in text, unit.name
        # Each unit's cap is in HOST_MEMORY (test_every_control_plane_unit_is_capped_below_the_host); a guard's
        # and a unit this host does not run keep the cap they had.
        memory_limit = (
            HOST_MEMORY[unit.name][2] if unit.name in HOST_MEMORY
            else HOST_MEMORY_KEPT[unit.name][1] if unit.name in HOST_MEMORY_KEPT
            else "1G" if unit.name == "blueprint-existing-policy-canary-watchdog.service"
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

    baseline = (REPO_ROOT / "docs" / "SYSTEMD_SECURITY_BASELINE.md").read_text(encoding="utf-8")
    assert "systemd-analyze security" in baseline
    assert "threshold=40" in baseline


#: Money safety, and the door to the host (#3329): the OOM killer passes them over. Never weakened: no
#: throttle, no pressure kill, the memory cap each had before. G1 campaign settlement settles money too.
#: Five of them keep 8G, which is above the host and never acts; each file says so. Their peaks are not
#: measured on the host and a guessed cap must not kill a money-safety unit. They are outside BATCH_SLICE.
HOST_SAFETY_GUARD_UNITS = (
    "blueprint-gpu-spend-guard.service",
    "blueprint-operator-door.service",
    "blueprint-existing-policy-canary-watchdog.service",
    "blueprint-task-evaluation-launch-supervisor.service",
    "blueprint-task-evaluation-launch-reconciler.service",
    "blueprint-task-evaluation-terminal-resource-release.service",
    "blueprint-native-g1-team-campaign-settlement.service",
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
    "blueprint-embodiment-preparation.service": ("200", "5.5G"),
    "blueprint-scene-object-discovery.service": ("200", "5.5G"),
    "blueprint-control-plane-preflight.service": ("200", "1G"),
    "blueprint-control-plane-capacity.service": ("200", "512M"),
    "blueprint-completed-replay-cache-gc.service": ("200", "512M"),
    "blueprint-control-plane-release-lifecycle.service": ("200", "512M"),
    "blueprint-owner-approved-deletion-plan.service": ("200", "1G"),
    "blueprint-owner-approved-deletion.service": ("200", "1G"),
    "blueprint-task-evaluation-launch-dispatcher.service": ("-200", "5.5G"),
    "blueprint-website-atlas-preview.service": ("-200", "5.5G"),
    "blueprint-team-evaluation-runner.service": ("-200", "5.5G"),
    "blueprint-beta-policy-screening.service": ("-200", "5.5G"),
    "blueprint-rigid-robot-free-probe-runner.service": ("-200", "5.5G"),
    "blueprint-task-evaluation-policy-canary-dispatcher.service": ("-200", "5.5G"),
    "blueprint-native-g1-team-campaign-dispatcher.service": ("-200", "5.5G"),
    "blueprint-task-evaluation-sam31-preparation-execution.service": ("-200", "5.5G"),
    "blueprint-existing-policy-canary-continuation.service": ("-200", "5.5G"),
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
        assert values["MemoryMax"] == (["1G"] if unit in {"blueprint-operator-door.service",
                                                          "blueprint-existing-policy-canary-watchdog.service"}
                                       else ["8G"]), unit


#: The one memory slice for the batch units, and the failure notifier (docs/SYSTEMD_SECURITY_BASELINE.md).
BATCH_SLICE = "system-blueprintbatch.slice"
NOTIFIER_UNIT = "blueprint-unit-failure@.service"
NOTIFIER_ON_FAILURE = "blueprint-unit-failure@%n.service"
#: The live reading the slice is sized from (2026-11-10 15:10Z, nothing heavy running), in the decimal GB that
#: /proc/meminfo's kB make, and the readiness plan's ceiling for the heavy units' caps together.
HOST_MEMTOTAL_GB = 8.33
HOST_IDLE_AVAILABLE_GB = 6.19
PLAN_BATCH_CEILING_GB = 6.5
#: The largest peak recorded on the host for one batch unit (launch activation and embodiment preparation,
#: docs/research/control-plane-host-profile-2026-10-31.md), in systemd's binary units like the caps.
LARGEST_RECORDED_BATCH_PEAK = "4.2G"
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
    "blueprint-embodiment-preparation.service",
    "blueprint-scene-object-discovery.service",
    "blueprint-task-evaluation-sam31-preparation-execution.service",
    "blueprint-task-evaluation-policy-canary-dispatcher.service",
    "blueprint-existing-policy-canary-continuation.service",
    "blueprint-native-g1-team-campaign-dispatcher.service",
    "blueprint-team-evaluation-runner.service",
    "blueprint-agent-run-dispatcher.service",
    "blueprint-task-evaluation-episode-compilation.service",
    "blueprint-task-evaluation-episode-compilation-remote.service",
    "blueprint-task-evaluation-configured-controls-progression.service",
    "blueprint-agent-stage-replay.service",
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
    "blueprint-native-g1-team-campaign-settlement.service": "money safety: settles campaign spend",
    "blueprint-task-evaluation-launch-reconciler.service": "money safety: the provider-zero proof",
    "blueprint-task-evaluation-launch-supervisor.service": "money safety: launch supervision",
    "blueprint-task-evaluation-terminal-resource-release.service": "money safety: releases paid resources",
    "blueprint-existing-policy-canary-watchdog.service": "money safety: a paid canary's watchdog",
    "blueprint-control-plane-capacity.service": "the pager: capacity, stall and spend alerts",
    NOTIFIER_UNIT: "the failure notifier: it must run while the batch units are at their bound",
    "blueprint-control-plane-storage-gc.service": "housekeeping that frees the disk batch units wait for",
    "blueprint-completed-replay-cache-gc.service": "housekeeping, capped at 512M",
    "blueprint-control-plane-release-lifecycle.service": "housekeeping, capped at 512M",
    "blueprint-control-plane-preflight.service": "a diagnostic, capped at 1G",
    "blueprint-owner-approved-deletion-plan.service": "started by hand, capped at 1G",
    "blueprint-owner-approved-deletion.service": "started by hand, capped at 1G",
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
    "blueprint-embodiment-preparation.service",
    "blueprint-native-g1-team-campaign-dispatcher.service",
    "blueprint-pubsub-handoff-listener.service",
    "blueprint-rigid-robot-free-probe-runner.service",
    "blueprint-task-evaluation-launch-dispatcher.service",
    "blueprint-task-evaluation-policy-canary-dispatcher.service",
    "blueprint-task-evaluation-sam31-preparation-execution.service",
    "blueprint-team-evaluation-runner.service",
)
#: Timer- or path-driven oneshots allowed to run with no start timeout, each with its reason. None today.
ONESHOTS_WITHOUT_A_START_TIMEOUT: dict[str, str] = {}


def _binary_gigabytes(decimal_gigabytes: float) -> float:
    return decimal_gigabytes * 1e9 / 2**30


def _timeout_seconds(value: str) -> int | None:
    """A bounded systemd time span in seconds; None for anything else (infinity, 0, a typo)."""
    match = re.fullmatch(r"(\d+)(s|m|min|h|d)?", value)
    if match is None:
        return None
    return int(match[1]) * {None: 1, "s": 1, "m": 60, "min": 60, "h": 3600, "d": 86400}[match[2]] or None


def _triggered_services() -> dict[str, list[str]]:
    """Each service a timer or a path unit starts, with the triggers that start it."""
    started: dict[str, list[str]] = {}
    for trigger in sorted([*SYSTEMD_DIR.glob("*.timer"), *SYSTEMD_DIR.glob("*.path")]):
        target = (_directives(trigger.name).get("Unit") or [f"{trigger.stem}.service"])[0]
        started.setdefault(target, []).append(trigger.name)
    return started


def test_every_timer_or_path_driven_oneshot_has_a_start_timeout() -> None:
    """A oneshot has no start timeout by default, and a timer that waits for it to go inactive
    (OnUnitInactiveSec) never fires again once one run hangs: the spend guard and the handoff
    listener ran that way."""
    started = _triggered_services()
    assert len(started) >= 30 and all((SYSTEMD_DIR / service).is_file() for service in started)
    unbounded = []
    for service in started:
        values = _directives(service)
        if values.get("Type") != ["oneshot"]:
            continue
        spans = values.get("TimeoutStartSec") or []
        if len(spans) != 1 or _timeout_seconds(spans[0]) is None:
            unbounded.append(service)
    assert sorted(unbounded) == sorted(ONESHOTS_WITHOUT_A_START_TIMEOUT)
    assert all(ONESHOTS_WITHOUT_A_START_TIMEOUT.values()), "each exception says why"
    # The oneshots nothing triggers (started by hand or by Wants=) are bounded too.
    for unit in SYSTEMD_DIR.glob("*.service"):
        values = _directives(unit.name)
        if values.get("Type") == ["oneshot"]:
            assert _timeout_seconds((values.get("TimeoutStartSec") or [""])[0]) is not None, unit.name


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


def test_the_batch_slice_bounds_the_batch_units_below_the_host() -> None:
    values = _directives(BATCH_SLICE)
    assert len(values["MemoryHigh"]) == len(values["MemoryMax"]) == 1
    high, maximum = _gigabytes(values["MemoryHigh"][0]), _gigabytes(values["MemoryMax"][0])
    idle = _binary_gigabytes(HOST_IDLE_AVAILABLE_GB)
    # Below what the host has free with nothing heavy running, so the batch units at their bound leave
    # the always-on and money-safety units the difference; and within the plan's ceiling.
    assert maximum < idle < _binary_gigabytes(HOST_MEMTOTAL_GB)
    assert maximum <= _binary_gigabytes(PLAN_BATCH_CEILING_GB)
    # The throttle starts above every peak recorded on the host for a batch unit, below the kill line.
    recorded = [_gigabytes(LARGEST_RECORDED_BATCH_PEAK),
                *(_gigabytes(row[3]) for unit, row in HOST_MEMORY.items() if unit in BATCH_SLICE_UNITS)]
    assert max(recorded) < high < maximum
    # The largest overlap on record that survives today stays under the kill line: scene progression
    # with team scene packet. A limit that killed it would be a regression.
    overlap = sum(_gigabytes(HOST_MEMORY[unit][3]) for unit in (
        "blueprint-task-evaluation-scene-progression.service", "blueprint-team-scene-packet.service"))
    assert overlap < maximum
    assert not {"ManagedOOMMemoryPressure", "ManagedOOMSwap", "MemoryLow", "MemoryMin"} & set(values)


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
        if unit not in {"blueprint-operator-door.service", "blueprint-existing-policy-canary-watchdog.service"}
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


def test_the_slice_and_the_notifier_are_installed_before_the_units_that_name_them() -> None:
    deployed = _deployed_units()
    installer = INSTALL_SCRIPT.read_text(encoding="utf-8")
    naming = [unit for unit in deployed
              if unit.endswith(".service") and {"Slice", "OnFailure"} & set(_directives(unit))]
    assert set(BATCH_SLICE_UNITS) & set(deployed) <= set(naming)
    for name in (BATCH_SLICE, NOTIFIER_UNIT):
        assert (SYSTEMD_DIR / name).is_file()
        assert deployed.count(name) == 1
        assert deployed.index(name) < min(deployed.index(unit) for unit in naming)
        target = '"${SYSTEMD_DIR}/' + name + '"'
        assert installer.count(target) == 1
        first_naming_install = min(
            installer.index('"${SYSTEMD_DIR}/' + unit + '"') for unit in naming
            if '"${SYSTEMD_DIR}/' + unit + '"' in installer
        )
        assert installer.index(target) < first_naming_install < installer.index("systemctl daemon-reload")
    # The installer creates the notifier's one writable directory; the deploy tool derives it from the unit.
    assert '"${STATE_DIR}/unit-failures"' in installer


def test_pubsub_handoff_listener_has_repeated_deployed_runner():
    service = _read("blueprint-pubsub-handoff-listener.service")
    timer = _read("blueprint-pubsub-handoff-listener.timer")
    env_example = _read("pipeline-control-plane.env.example")
    installer = INSTALL_SCRIPT.read_text(encoding="utf-8")

    assert "blueprint_pipeline.pubsub_handoff_listener" in service
    assert "--subscription" in service
    assert "BLUEPRINT_PUBSUB_HANDOFF_STAGE_CONTROL_PLANE=true" in service
    assert "BLUEPRINT_PUBSUB_HANDOFF_SKIP_RUN_E2E=true" in service
    assert (
        "BLUEPRINT_PUBSUB_HANDOFF_SUBSCRIPTION=blueprint-pipeline-handoff-listener" in env_example
    )
    assert (
        "GOOGLE_APPLICATION_CREDENTIALS=/etc/blueprint/credentials/pipeline-handoff-listener.json"
    ) in env_example
    assert "GOOGLE_CLOUD_PROJECT=blueprint-8c1ca" in env_example
    assert "BLUEPRINT_PUBSUB_HANDOFF_STAGE_CONTROL_PLANE=true" in env_example
    assert "BLUEPRINT_PUBSUB_HANDOFF_SKIP_RUN_E2E=true" in env_example
    assert (
        "BLUEPRINT_ROBOT_EVAL_JOB_REQUEST_INBOX="
        "/var/lib/blueprint/pipeline-control-plane/robot-eval-job-requests"
    ) in env_example
    assert (
        "BLUEPRINT_LIVE_PIPELINE_INTAKE_WORK_DIR="
        "/var/lib/blueprint/pipeline-control-plane/incoming_webapp_job_requests"
    ) in env_example
    assert "OnActiveSec=30s" in timer
    assert "OnUnitInactiveSec=1min" in timer
    assert "OnBootSec=" not in timer
    assert "Unit=blueprint-pubsub-handoff-listener.service" in timer
    assert "blueprint-pubsub-handoff-listener.service" in installer
    assert "blueprint-pubsub-handoff-listener.timer" in installer
    assert "systemctl enable --now blueprint-pubsub-handoff-listener.timer" in installer
    assert 'chown -R --no-dereference "${SERVICE_USER}:${SERVICE_GROUP}"' in installer
    assert "${STATE_DIR}/robot-eval-job-requests" in installer
    assert "${STATE_DIR}/incoming_webapp_job_requests" in installer
    assert "${HANDOFF_DIR}" in installer


def test_capture_reconstruction_dispatcher_stays_retired():
    """The queue had no consumer after its dispatcher was retired (P-RT-01, H-03); nothing installs or feeds it."""
    env_example = _read("pipeline-control-plane.env.example")
    installer = INSTALL_SCRIPT.read_text(encoding="utf-8")

    for suffix in ("service", "path", "timer"):
        assert not (SYSTEMD_DIR / f"blueprint-capture-reconstruction-dispatcher.{suffix}").exists()
    assert "blueprint-capture-reconstruction-dispatcher" not in installer
    assert "capture-reconstruction-queue" not in installer
    for name in (
        "BLUEPRINT_CAPTURE_RECONSTRUCTION_POLICY_ROOT",
        "BLUEPRINT_CAPTURE_RECONSTRUCTION_QUEUE_ROOT",
        "BLUEPRINT_CAPTURE_RECONSTRUCTION_STATE_ROOT",
        "BLUEPRINT_CAPTURE_RECONSTRUCTION_DERIVED_ROOT",
    ):
        assert name not in env_example
    listener = (REPO_ROOT / "src/blueprint_pipeline/pubsub_handoff_listener.py").read_text(encoding="utf-8")
    assert "capture_reconstruction_launch_dispatcher" not in listener


def test_installer_binds_the_exact_canonical_worker_wheel_and_hash_pinned_runtime():
    env_example = _read("pipeline-control-plane.env.example")
    installer = INSTALL_SCRIPT.read_text(encoding="utf-8")

    assert "build_canonical_3dgs_worker_wheel.sh" in installer
    worker_builder = (
        REPO_ROOT / "scripts" / "build_canonical_3dgs_worker_wheel.sh"
    ).read_text(encoding="utf-8")
    assert '${TMPDIR:-/tmp}/blueprint-canonical-worker-build.XXXXXX' in worker_builder
    assert '${TMPDIR:-/private/tmp}' not in worker_builder
    assert '"$runtime_python" -m pip wheel' in worker_builder
    assert "--no-build-isolation" in worker_builder
    assert "--no-deps" in worker_builder
    assert 'SOURCE_DATE_EPOCH="$source_date_epoch"' in worker_builder
    assert "uv build" not in worker_builder
    assert 'WORKER_RELEASE_DIR="/opt/blueprint/releases/canonical-3dgs-worker/${SOURCE_COMMIT}"' in installer
    assert "BLUEPRINT_CANONICAL_3DGS_WORKER_WHEEL" in installer
    assert "BLUEPRINT_CANONICAL_3DGS_WORKER_WHEEL" not in env_example
    assert "REPLACED_BY_INSTALLER_EXACT_RELEASE_WHEEL" not in env_example
    assert "BLUEPRINT_AWS_" not in env_example
    runtime_requirements = _read("production-control-plane-requirements.txt")
    assert "production-control-plane-requirements.txt" in installer
    assert "--require-hashes" in installer
    assert 'm.version("rfc8785") == "0.1.4"' in installer
    assert "rfc8785==0.1.4" in runtime_requirements
    assert "sha256:520d690b448ecf0703691c76e1a34a24ddcd4fc5bc41d589cb7c58ec651bcd48" in runtime_requirements


def test_control_plane_postcheck_pages_or_fails_blocked_manifests():
    """Kept for the host's installed copy of the retired loop until an operator removes it (P-RT-06)."""
    postcheck = _read("blueprint-control-plane-postchecks.sh")
    env_example = _read("pipeline-control-plane.env.example")

    assert "blueprint_pipeline.live_pipeline_proof_audit" in postcheck
    assert "blueprint_pipeline.live_pipeline_manifest_alert" in postcheck
    assert "BLUEPRINT_OPERATOR_ALERT_REQUIRE_WEBHOOK=true" in env_example
    assert "BLUEPRINT_OPERATOR_ALERT_WEBHOOK_URL" in env_example


def test_terraform_alert_policies_require_notification_channels_or_explicit_waiver():
    text = TERRAFORM_MAIN.read_text(encoding="utf-8")

    assert 'variable "allow_empty_monitoring_notification_channels"' in text
    assert "notification_channels = var.monitoring_notification_channels" in text
    assert (
        text.count(
            "var.allow_empty_monitoring_notification_channels || length(var.monitoring_notification_channels) > 0"
        )
        >= 4
    )
    assert "Set allow_empty_monitoring_notification_channels=true only for dry-run plans." in text


def test_terraform_queue_depth_alert_uses_beta_backpressure_thresholds():
    text = TERRAFORM_MAIN.read_text(encoding="utf-8")
    body = _terraform_resource_body(text, "google_monitoring_alert_policy", "queue_depth")

    assert 'variable "pipeline_queue_depth_alert_threshold"' in text
    assert "default     = 50" in text
    assert 'variable "pipeline_queue_depth_alert_duration"' in text
    assert 'default     = "300s"' in text
    assert "duration        = var.pipeline_queue_depth_alert_duration" in body
    assert "threshold_value = var.pipeline_queue_depth_alert_threshold" in body
    assert "threshold_value = 100" not in body
    assert 'duration        = "600s"' not in body
    assert "per-user intake pressure" in body


def test_terraform_keeps_firestore_latency_alert_without_phantom_captures_indexes():
    # SCALE2-07: the four `captures_*` index resources targeted a literal
    # `captures` collection that no repo has ever written. The real
    # capture-record collection is creatorCaptures, whose composites (including
    # the createdAtShard scale-up companions) are owned by
    # Blueprint-WebApp/firestore.indexes.json. This repo keeps only the shared
    # database and the latency alerting.
    text = TERRAFORM_MAIN.read_text(encoding="utf-8")
    latency_alert = _terraform_resource_body(
        text,
        "google_monitoring_alert_policy",
        "firestore_request_latency",
    )

    assert 'resource "google_firestore_index" "captures_status"' not in text
    assert 'resource "google_firestore_index" "captures_user"' not in text
    assert 'resource "google_firestore_index" "captures_status_created_at_shard"' not in text
    assert 'resource "google_firestore_index" "captures_user_created_at_shard"' not in text
    assert 'collection = "captures"' not in text

    assert "serviceruntime.googleapis.com/api/request_latencies" in latency_alert
    assert "resource.service == 'firestore.googleapis.com'" in latency_alert
    assert "| condition val() > 0.25 's'" in latency_alert
    assert "creatorCaptures created_at composite indexes" in latency_alert
    assert "Blueprint-WebApp firestore.indexes.json" in latency_alert


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


def test_terraform_state_is_remote_locked_retained_and_cmek_validated() -> None:
    terraform = TERRAFORM_MAIN.read_text(encoding="utf-8")
    deploy_script = PIPELINE_DEPLOY_SCRIPT.read_text(encoding="utf-8")
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert 'backend "gcs" {}' in terraform
    assert "validate_terraform_state_backend.py" in deploy_script
    assert "TERRAFORM_STATE_BUCKET" in deploy_script
    assert "TERRAFORM_STATE_PREFIX" in deploy_script
    assert "TERRAFORM_STATE_KMS_KEY" in deploy_script
    assert "-backend-config=kms_encryption_key=" in deploy_script
    assert "-reconfigure" in deploy_script
    assert "*.tfstate" in gitignore
    assert "*.tfstate.*" in gitignore
    assert "*.tfvars" in gitignore
    assert "*.tfplan" in gitignore


def test_full_deploy_requires_authenticated_readback_service_canaries() -> None:
    terraform = TERRAFORM_MAIN.read_text(encoding="utf-8")
    deploy_script = PIPELINE_DEPLOY_SCRIPT.read_text(encoding="utf-8")

    assert 'output "deployed_image_digests"' in terraform
    assert "run_deployment_service_canaries.py" in deploy_script
    assert "--topology-evidence" in deploy_script
    assert "--privacy-secret-name" in deploy_script
    assert "--video-secret-name" in deploy_script
    assert "Authenticated deployment service canaries did not pass" in deploy_script
    assert "authenticated_service_canaries" in deploy_script
    assert "model task success remains separately required" in deploy_script


def test_pipeline_failure_alert_counts_failures_over_five_minutes() -> None:
    terraform = TERRAFORM_MAIN.read_text(encoding="utf-8")
    body = _terraform_resource_body(
        terraform, "google_monitoring_alert_policy", "pipeline_failures"
    )

    assert 'alignment_period     = "300s"' in body
    assert 'per_series_aligner   = "ALIGN_SUM"' in body
    assert 'cross_series_reducer = "REDUCE_SUM"' in body
    assert "threshold_value = 5" in body
    assert 'comparison      = "COMPARISON_GT"' in body
    assert 'duration        = "0s"' in body
    assert "ALIGN_RATE" not in body


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


def test_terraform_is_the_single_deployment_topology_owner() -> None:
    deploy_script = PIPELINE_DEPLOY_SCRIPT.read_text(encoding="utf-8")
    full_deploy = deploy_script[deploy_script.index("    # Full deployment") :]

    assert 'topology_owner: "terraform"' in deploy_script
    assert "terraform plan -input=false -detailed-exitcode" in deploy_script
    assert "Post-apply Terraform drift detected" in deploy_script
    assert "terraform output -json" in deploy_script
    assert "blueprint.terraform_topology_evidence.v1" in deploy_script
    assert "--function-only was removed" in deploy_script
    assert "    apply_terraform" in full_deploy
    assert "    setup_iam" not in full_deploy
    assert "    create_pubsub_topics" not in full_deploy
    assert "    create_cloud_tasks_queues" not in full_deploy
    assert "    create_cloud_run_jobs" not in full_deploy
    assert "    deploy_cloud_function" not in full_deploy


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


def test_terraform_declares_private_large_video_ingest_topic_for_extract_frames():
    text = TERRAFORM_MAIN.read_text(encoding="utf-8")

    assert 'variable "capture_extract_frames_service_account_email"' in text
    assert 'resource "google_pubsub_topic" "large_video_ingest"' in text
    assert 'name   = "blueprint-large-video-ingest"' in text
    assert (
        'resource "google_pubsub_topic_iam_member" "capture_extract_frames_large_video_ingest_publisher"'
        in text
    )
    assert 'member  = "serviceAccount:${var.capture_extract_frames_service_account_email}"' in text
    assert 'output "large_video_ingest_topic"' in text


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


def _unit_settings(name: str) -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in _read(name).splitlines()
        if line and not line.startswith(("#", "[")) and "=" in line
    )


def _span_seconds(value: str) -> int:
    match = re.fullmatch(r"(\d+)(s|min|h)", value)
    assert match is not None, f"unbounded time span: {value}"
    return int(match[1]) * {"s": 1, "min": 60, "h": 3600}[match[2]]


def test_preflight_cadence_leaves_the_host_idle_and_the_door_summary_fresh(tmp_path):
    """E-02: a ~13-minute run every 15 minutes kept the preflight busy 87% of the time.

    The operator door serves the summary only while it is at most an hour old,
    stamped when a run starts. Start-to-start spacing bounds that age whatever
    a run takes: one interval, the randomized delay, and the longest run the
    unit allows before the next summary replaces it.
    """
    import dataclasses
    import json
    import sys
    import time

    timer = _unit_settings("blueprint-control-plane-preflight.timer")
    service = _unit_settings("blueprint-control-plane-preflight.service")
    assert "OnUnitInactiveSec" not in timer, "spacing from completion would add a run to the bound"
    interval = _span_seconds(timer["OnUnitActiveSec"])
    delay = _span_seconds(timer["RandomizedDelaySec"])
    longest_run = _span_seconds(service["TimeoutStartSec"])
    assert interval >= 30 * 60
    assert longest_run < interval, "a run always ends before the next one is due"
    oldest_served = interval + delay + longest_run

    sys.path.insert(0, str(REPO_ROOT / "deploy" / "operator-door"))
    try:
        from operator_door.config import DoorConfig
        from operator_door.status import _preflight
    finally:
        sys.path.remove(str(REPO_ROOT / "deploy" / "operator-door"))
    from blueprint_pipeline.production_preflight_summary import build_safe_summary

    release = tmp_path / ("a" * 40)
    release.mkdir()
    active = tmp_path / "active"
    active.symlink_to(release)
    public = tmp_path / "preflight" / "public"
    public.mkdir(parents=True)
    config = dataclasses.replace(
        DoorConfig(), preflight_summary=str(public / "summary.json"), active_release_link=str(active)
    )

    def served(age_seconds: int) -> str:
        stamped = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - age_seconds))
        report = {"schema_version": "task_evaluation_production_chain_preflight.v1",
                  "generated_at": stamped, "active_sha": "a" * 40, "lane": "diagnostic",
                  "units": {}, "host_findings": []}
        summary = build_safe_summary(report, raw_report_bytes=json.dumps(report).encode())
        (public / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return _preflight(config)["status"]

    # A margin for scheduling jitter under load still stays inside the door's hour.
    assert served(oldest_served + 5 * 60) == "observed"
    # The bound is real: the cadence an hour apart would leave the door refusing it.
    assert served(60 * 60 + delay + longest_run) == "refused"


def test_main_deploy_image_includes_ffmpeg_for_clip_and_keyframe_lanes():
    text = ROOT_DOCKERFILE.read_text(encoding="utf-8")

    assert "ffmpeg" in text


def test_ci_workflows_checkout_the_pinned_blueprint_contracts_revision():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"BlueprintContracts\.git@([0-9a-f]{40})", pyproject)
    assert match is not None
    pinned_ref = match.group(1)

    for workflow in (
        REPO_ROOT / ".github" / "workflows" / "ci.yml",
        REPO_ROOT / ".github" / "workflows" / "full-test-lane.yml",
        REPO_ROOT / ".github" / "workflows" / "sim-only-local-gate.yml",
    ):
        text = workflow.read_text(encoding="utf-8")
        assert f"CONTRACTS_REF: {pinned_ref}" in text
        assert "CONTRACTS_REF: ${{ github.head_ref || github.ref_name }}" not in text
        assert "CONTRACTS_REF: ${{ github.ref_name }}" not in text


def test_full_test_lane_is_explicit_or_nightly_and_gates_deploy_contract():
    workflow = FULL_TEST_LANE_WORKFLOW.read_text(encoding="utf-8")
    event_block = workflow.split("jobs:", 1)[0]
    deploy_script = PIPELINE_DEPLOY_SCRIPT.read_text(encoding="utf-8")
    required_doc = CI_REQUIRED_CHECKS_DOC.read_text(encoding="utf-8")

    assert "pull_request:" not in event_block
    assert "push:" not in event_block
    assert "workflow_call:" in event_block
    assert "schedule:" in event_block
    assert 'cron: "17 8 * * *"' in event_block
    assert "workflow_dispatch:" in event_block
    assert "uv run --no-sync scripts/pytest_full.sh" in workflow
    assert (
        '--junitxml="${{ runner.temp }}/blueprint-ci/full-test-lane-shard-junit.xml"'
        in workflow
    )
    assert "scripts/full_lane_sharding.py aggregate" in workflow
    assert "Full pytest lane on CPU runner" in workflow
    assert 'FULL_TEST_LANE_REQUIRED="${FULL_TEST_LANE_REQUIRED:-true}"' in deploy_script
    assert "check_full_test_lane_deploy_gate" in deploy_script
    assert "FULL_TEST_LANE_COMMIT" in deploy_script
    assert "FULL_TEST_LANE_EVIDENCE_URI" in deploy_script
    assert "verify_deploy_release_provenance.py" in deploy_script
    assert "verify_clean_release_source" in deploy_script
    assert "status --porcelain=v1 --untracked-files=all" in deploy_script
    assert "fetch --quiet origin main" in deploy_script
    assert "refs/remotes/origin/main" in deploy_script
    assert 'merge-base --is-ancestor "$current_sha" refs/remotes/origin/main' in deploy_script
    assert "not in exact parity with origin/main" not in deploy_script
    assert "FULL_TEST_LANE_BYPASS_REASON" not in deploy_script
    assert "this deploy path has no text-only CI bypass" in deploy_script
    assert (
        "Full Test Lane / Full pytest lane on CPU runner passed for this exact commit"
        in deploy_script
    )
    assert "Full Test Lane / Full pytest lane on CPU" in required_doc
    assert "production_deployment_promotion" in required_doc
    assert "on that exact protected-main commit SHA" in required_doc
    assert "proves their disjoint union" in required_doc
    assert "No shard uses xdist" in required_doc
    assert "deploy/scripts/deploy.sh" in required_doc
    assert "nightly scheduled run is supplementary health evidence only" in required_doc


def test_release_images_are_versioned_manifested_and_rejected_if_latest():
    terraform = TERRAFORM_MAIN.read_text(encoding="utf-8")
    tfvars_example = TERRAFORM_TFVARS_EXAMPLE.read_text(encoding="utf-8")
    deploy_script = PIPELINE_DEPLOY_SCRIPT.read_text(encoding="utf-8")
    required_doc = CI_REQUIRED_CHECKS_DOC.read_text(encoding="utf-8")

    image_variables = (
        "docker_image",
        "privacy_sam3_image",
        "privacy_vip_image",
        "privacy_deepprivacy2_image",
        "video_to_world_image",
    )
    image_names = (
        "blueprint-pipeline",
        "sam3-privacy",
        "vip-privacy",
        "deepprivacy2-privacy",
        "video-to-world",
    )

    for variable in image_variables:
        body = _terraform_variable_body(terraform, variable)
        if variable == "docker_image":
            assert "default" not in body
        else:
            assert f'var.deployment_scope == "remote_cpu" && var.{variable} == ""' in body
        assert "nullable    = false" in body
        assert "^.+@sha256:[0-9a-f]{64}$" in body
        assert "non-latest versioned release tag" not in body
        assert f"var.{variable}" in body

    for image_name in image_names:
        assert f"{image_name}:latest" not in terraform
        assert f"{image_name}:latest" not in tfvars_example
        assert f"{image_name}@sha256:" in tfvars_example

    assert 'IMAGE_TAG="${IMAGE_TAG:-}"' in deploy_script
    assert 'IMAGE_TAG="${GIT_SHA:0:12}"' in deploy_script
    assert "validate_release_image_tag" in deploy_script
    assert "pin_pushed_image_digests" in deploy_script
    assert "gcloud container images describe" in deploy_script
    assert "image_summary.fully_qualified_digest" in deploy_script
    assert "require_resolved_image_matches" in deploy_script
    assert "tag/digest mismatch" in deploy_script
    assert "pin_pushed_image_digests" in deploy_script
    assert "DEPLOYMENT_MANIFEST_PATH" in deploy_script
    assert "blueprint.pipeline_deployment_manifest.v1" in deploy_script
    assert "deploy/scripts/deploy.sh --rollback --rollback-image-tag " in deploy_script
    assert "latest`/`dev`/`test`" in required_doc
    assert "pipeline-deployment-manifest.json" in required_doc
    assert "Terraform image variables require immutable `@sha256` digests" in required_doc


def test_sim_only_gate_uses_explicit_webapp_ref_with_launch_fixes():
    text = (REPO_ROOT / ".github" / "workflows" / "sim-only-local-gate.yml").read_text(
        encoding="utf-8"
    )

    assert "WEBAPP_REF: ${{ inputs.webapp_ref || 'main' }}" in text
    assert "WEBAPP_REF: ${{ inputs.webapp_ref || github.head_ref || github.ref_name }}" not in text
    assert "launch-robot-eval-delivery-forwarding-fixes" not in text


def test_sim_only_gate_uses_headless_linux_mujoco_rendering():
    text = (REPO_ROOT / ".github" / "workflows" / "sim-only-local-gate.yml").read_text(
        encoding="utf-8"
    )

    assert "runs-on: ubuntu-latest" in text
    assert "MUJOCO_GL: osmesa" in text
    assert "libosmesa6" in text
    assert "runs-on: macos-latest" not in text
    assert "MUJOCO_GL: glfw" not in text


def test_sam31_profile_registry_is_content_bound_in_the_base_units():
    """Every unit that resolves the SAM server profile -- including the progression
    service that runs the look-ahead admission replay, which never received a
    per-scene PROFILE_ENV drop-in -- reads one fixed content-addressed registry from
    its base unit file. The scene-progression factory (the registrar) mounts it
    read-write; the resolvers mount it read-only. The path lives under
    /var/lib/blueprint so the deploy's unit-sandbox installer creates it."""
    from blueprint_pipeline.task_evaluation_sam31_profile_registry import (
        DEFAULT_PROFILE_REGISTRY_ROOT as FIXED,
    )

    assert FIXED.startswith("/var/lib/blueprint/") and Path(FIXED).suffix == ""
    env_line = f"Environment=BLUEPRINT_TASK_EVALUATION_SAM31_PREPARATION_PROFILE_DIR={FIXED}"

    resolvers = (
        "blueprint-task-evaluation-configured-controls-progression.service",
        "blueprint-task-evaluation-launch-preparation.service",
        "blueprint-task-evaluation-sam31-preparation-execution.service",
        "blueprint-task-evaluation-launch-activation.service",
        "blueprint-task-evaluation-launch-supervisor.service",
        "blueprint-task-evaluation-launch-reconciler.service",
        "blueprint-task-evaluation-launch-dispatcher.service",
    )
    for name in resolvers:
        text = _read(name)
        assert env_line in text, name
        assert f"ReadOnlyPaths={FIXED}" in text, name
        assert f"ReadWritePaths={FIXED}" not in text, name

    registrar = _read("blueprint-task-evaluation-scene-progression.service")
    assert env_line in registrar
    assert f"ReadWritePaths={FIXED}" in registrar
    assert f"ReadOnlyPaths={FIXED}" not in registrar


def test_activation_loads_operator_owned_preparation_routing():
    assert "EnvironmentFile=-/etc/blueprint/task-evaluation-scene-progression.env" in _read(
        "blueprint-task-evaluation-launch-activation.service"
    )


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


def test_volume_admission_units_map_bulk_roles_to_their_writers() -> None:
    role_targets = (
        "Environment=BLUEPRINT_CONTROL_PLANE_DISK_ROLE_TARGETS="
        "launch_preparation=/var/lib/blueprint/task-evaluation-inputs/prepared-references,"
        "episode_compilation=/var/lib/blueprint/task-evaluation-inputs/compiled-episodes,"
        "launch_activation=/var/lib/blueprint/task-evaluation-inputs/launch-activations,"
        "launch_dispatch=/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-runs,"
        "policy_canary_dispatch=/var/lib/blueprint/pipeline-control-plane/task-evaluation-policy-canaries,"
        "handoff_staging=/var/lib/blueprint/pubsub-handoffs,"
        "scene_configuration_output=/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-runs,"
        "policy_canary_output=/var/lib/blueprint/pipeline-control-plane/task-evaluation-policy-canaries"
    )
    for name in (
        "blueprint-pipeline-intake.service",
        "blueprint-task-evaluation-scene-progression.service",
        "blueprint-control-plane-capacity.service",
    ):
        assert role_targets in (SYSTEMD_DIR / name).read_text(encoding="utf-8")
    capacity = (SYSTEMD_DIR / "blueprint-control-plane-capacity.service").read_text(
        encoding="utf-8"
    )
    assert "Environment=BLUEPRINT_CAPACITY_MOUNTS=/:/var/lib/blueprint:/mnt/blueprint-work" in capacity


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


def test_streamed_canary_units_bind_the_b2_artifact_store() -> None:
    """Review I4: promotion and the member view refuse, rather than borrow the staging store's
    credentials, unless the dedicated B2 store is configured. The canary dispatcher (whose
    allocator child promotes and ingests) and the existing-run continuation (which promotes
    before cleanup) bind it explicitly, exactly as the intake service does."""
    intake = (SYSTEMD_DIR / "blueprint-pipeline-intake.service").read_text(encoding="utf-8")
    bindings = [line for line in intake.splitlines()
                if line.startswith("Environment=BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_")]
    assert len(bindings) == 6 and any("_EXPECTED_BUCKET=" in line for line in bindings)
    for name in ("blueprint-task-evaluation-policy-canary-dispatcher.service",
                 "blueprint-existing-policy-canary-continuation.service"):
        unit = (SYSTEMD_DIR / name).read_text(encoding="utf-8")
        lines = unit.splitlines()
        assert [line for line in lines if line in bindings] == bindings, name
        assert "ReadOnlyPaths=/etc/blueprint/provider-secrets" in lines, name


def test_intake_reads_website_preparation_status_from_the_listener_tree() -> None:
    """Beta L3/L6: the signed status route finds a website capture that no capture-root map
    names in the listener's own tree. The intake unit binds that tree itself, exactly as the
    listener does, instead of depending on what the shared environment file happens to set."""
    binding = "Environment=BLUEPRINT_PUBSUB_HANDOFF_STORAGE_ROOT=/var/lib/blueprint/pubsub-handoffs"
    for name in ("blueprint-pipeline-intake.service", "blueprint-pubsub-handoff-listener.service"):
        assert binding in _read(name).splitlines(), name



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
