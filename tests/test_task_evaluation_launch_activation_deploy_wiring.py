from __future__ import annotations

import configparser
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_activation_worker_is_hardened_and_watches_only_its_queue() -> None:
    service = text(
        "deploy/systemd/blueprint-task-evaluation-launch-activation.service"
    )
    path = text("deploy/systemd/blueprint-task-evaluation-launch-activation.path")
    assert "User=blueprint" in service
    assert "NoNewPrivileges=true" in service
    assert "ProtectSystem=strict" in service
    assert "-m blueprint_pipeline.task_evaluation_scene_retirement_supervisor --worker blueprint_pipeline.task_evaluation_launch_activation_worker -- --max-messages 1" in service
    assert "task-evaluation-scene-constructions" in service
    assert "task-evaluation-inputs/system-runtimes" in service
    assert "BLUEPRINT_TASK_EVALUATION_SCENE_CONSTRUCTION_QUEUE_ROOT=" in service
    assert (
        "Environment='BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_ALLOWED_URI_PREFIXES_JSON="
        '["s3://blueprint/task-evaluation/production-inputs/",'
        '"s3://blueprint-task-evaluation-artifacts-prod/blueprint/arm-decision-proof-v1/'
        'configured-scenes/",'
        '"s3://blueprint/blueprint/arm-decision-proof-v1/configured-scenes/"]\''
    ) in service
    assert "task-evaluation-launch-activations/pending" in path
    assert "blueprint-task-evaluation-launch-activation.service" in path
    assert "task-evaluation-launches/pending" not in path


def test_installer_provisions_activation_roots_and_exact_unit_set() -> None:
    installer = text("scripts/install_live_pipeline_control_plane.sh")
    for unit in (
        "blueprint-task-evaluation-launch-activation.service",
        "blueprint-task-evaluation-launch-activation.path",
        "blueprint-task-evaluation-launch-activation.timer",
    ):
        assert f'deploy/systemd/{unit}' in installer
        assert f'${{SYSTEMD_DIR}}/{unit}' in installer
    assert '"${STATE_DIR}/task-evaluation-launch-activations/pending"' in installer
    assert '"${TASK_EVALUATION_INPUT_ROOT}/launch-activations"' in installer
    assert '"${TASK_EVALUATION_INPUT_ROOT}/system-runtimes"' in installer
    assert "systemctl enable --now blueprint-task-evaluation-launch-activation.path" in installer
    assert "systemctl enable --now blueprint-task-evaluation-launch-activation.timer" in installer


def _unit(path: str) -> configparser.RawConfigParser:
    parser = configparser.RawConfigParser(strict=False, interpolation=None)
    parser.optionxform = str  # keep systemd's key case
    parser.read_string(text(path))
    return parser


def test_hourly_timer_runs_the_same_worker_and_a_path_hold_still_stops_every_run() -> None:
    """E-01: only new pending/ rows started the worker, so a claim a killed worker left on an
    idle queue was never retired. The timer starts the same oneshot hourly, as the service
    account. The door parks a deploy for pending rows only while the path unit can claim
    them (activation_hold), so a hold on that path unit must stop timer-started runs too."""

    timer = _unit("deploy/systemd/blueprint-task-evaluation-launch-activation.timer")
    assert timer["Timer"]["Unit"] == "blueprint-task-evaluation-launch-activation.service"
    assert (timer["Timer"]["OnBootSec"], timer["Timer"]["OnUnitInactiveSec"]) == ("10min", "1h")
    assert timer["Install"]["WantedBy"] == "timers.target"
    spec = importlib.util.spec_from_file_location(
        "activation_hold_under_test", ROOT / "deploy/operator-door/operator_door/activation_hold.py")
    hold = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hold)
    service = _unit("deploy/systemd/blueprint-task-evaluation-launch-activation.service")
    assert service["Unit"]["ConditionPathExists"] == (
        f"!/var/lib/blueprint-operator-door/requests/holds/{hold.ACTIVATION_PATH_UNIT}.json")
    assert service["Service"]["User"] == "blueprint"
    deployer = text("scripts/deploy_control_plane_commit.py")
    assert '"blueprint-task-evaluation-launch-activation.timer"' in deployer
    assert "blueprint-task-evaluation-launch-activation.timer" in text("deploy/host/mount_work_volume.sh")


def test_activation_prefixes_are_operator_owned_and_deploy_arms_no_paid_request() -> None:
    environment = text("deploy/systemd/pipeline-control-plane.env.example")
    deployer = text("scripts/deploy_control_plane_commit.py")
    assert (
        "BLUEPRINT_TASK_EVALUATION_LAUNCH_ACTIVATION_RELEASE_WINDOW_PREFIX="
        in environment
    )
    assert (
        "BLUEPRINT_TASK_EVALUATION_LAUNCH_ACTIVATION_DESTINATION_PREFIX="
        in environment
    )
    assert (
        "BLUEPRINT_TASK_EVALUATION_SCENE_CONFIGURATION_TOOLCHAIN_ROOT="
        in environment
    )
    assert '"blueprint-task-evaluation-launch-activation.path"' in deployer
    assert "provider_allocation_performed" not in text(
        "src/blueprint_pipeline/task_evaluation_shared_mutation_window.py"
    )


def test_activation_loads_exact_release_scene_runtime_after_shared_environment() -> None:
    service = text(
        "deploy/systemd/blueprint-task-evaluation-launch-activation.service"
    )
    assert service.index("EnvironmentFile=-/etc/blueprint/pipeline-control-plane.env") < (
        service.index(
            "EnvironmentFile=-/etc/blueprint/task-evaluation-scene-configuration-release.env"
        )
    )
