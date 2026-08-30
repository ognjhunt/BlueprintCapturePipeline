from __future__ import annotations

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
    assert "task_evaluation_launch_activation_worker --max-messages 1" in service
    assert "task-evaluation-scene-constructions" in service
    assert "task-evaluation-inputs/system-runtimes" in service
    assert "BLUEPRINT_TASK_EVALUATION_SCENE_CONSTRUCTION_QUEUE_ROOT=" in service
    assert (
        "Environment='BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_ALLOWED_URI_PREFIXES_JSON="
        '["s3://blueprint/task-evaluation/production-inputs/",'
        '"s3://blueprint-task-evaluation-artifacts-prod/blueprint/arm-decision-proof-v1/'
        'configured-scenes/artifacts/",'
        '"s3://blueprint/blueprint/arm-decision-proof-v1/configured-scenes/"]\''
    ) in service
    assert "task-evaluation-launch-activations/pending" in path
    assert "blueprint-task-evaluation-launch-activation.service" in path
    assert "task-evaluation-launches/pending" not in path


def test_installer_provisions_activation_roots_and_exact_unit_pair() -> None:
    installer = text("scripts/install_live_pipeline_control_plane.sh")
    for unit in (
        "blueprint-task-evaluation-launch-activation.service",
        "blueprint-task-evaluation-launch-activation.path",
    ):
        assert f'deploy/systemd/{unit}' in installer
        assert f'${{SYSTEMD_DIR}}/{unit}' in installer
    assert '"${STATE_DIR}/task-evaluation-launch-activations/pending"' in installer
    assert '"${TASK_EVALUATION_INPUT_ROOT}/launch-activations"' in installer
    assert '"${TASK_EVALUATION_INPUT_ROOT}/system-runtimes"' in installer
    assert "systemctl enable --now blueprint-task-evaluation-launch-activation.path" in installer


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
