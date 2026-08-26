from __future__ import annotations

from pathlib import Path

import scripts.deploy_control_plane_commit as deploy


ROOT = Path(__file__).resolve().parents[1]


def text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_no_spend_preparation_worker_has_hardened_service_and_path_unit() -> None:
    service = text(
        "deploy/systemd/blueprint-task-evaluation-launch-preparation.service"
    )
    path = text("deploy/systemd/blueprint-task-evaluation-launch-preparation.path")
    assert "User=blueprint" in service
    assert "NoNewPrivileges=true" in service
    assert "ProtectSystem=strict" in service
    assert "ProtectHome=true" in service
    assert "task_evaluation_launch_preparation_worker" in service
    assert "paid_resource_allocator" not in service
    assert "provider_adapter" not in service
    assert "EnvironmentFile=-/etc/blueprint/pipeline-control-plane.env" in service
    assert "task-evaluation-launch-preparations/pending" in path
    assert "blueprint-task-evaluation-launch-preparation.service" in path
    assert "BLUEPRINT_TASK_EVALUATION_SCENE_CONSTRUCTION_QUEUE_ROOT=" in service
    assert "task-evaluation-scene-constructions" in service
    assert "BLUEPRINT_TASK_EVALUATION_EPISODE_COMPILATION_QUEUE_ROOT=" in service
    assert "task-evaluation-episode-compilations" in service


def test_canonical_environment_documents_bounded_input_prefixes() -> None:
    environment = text("deploy/systemd/pipeline-control-plane.env.example")
    assert (
        "BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_ALLOWED_URI_PREFIXES_JSON="
        '["s3://blueprint-production-inputs/"]'
    ) in environment
    assert "BLUEPRINT_WAM_OBJECT_STORE_ACCESS_KEY_ID_FILE=" in environment
    assert "BLUEPRINT_WAM_OBJECT_STORE_SECRET_ACCESS_KEY_FILE=" in environment


def test_installer_creates_both_automatic_progression_queues() -> None:
    installer = text("scripts/install_live_pipeline_control_plane.sh")
    for queue in (
        "task-evaluation-scene-constructions",
        "task-evaluation-episode-compilations",
    ):
        for state in ("pending", "processing", "completed", "blocked", "results"):
            assert f'"${{STATE_DIR}}/{queue}/{state}"' in installer


def test_canonical_installer_installs_and_enables_preparation_pair() -> None:
    installer = text("scripts/install_live_pipeline_control_plane.sh")
    for unit in (
        "blueprint-task-evaluation-launch-preparation.service",
        "blueprint-task-evaluation-launch-preparation.path",
    ):
        assert f'"${{REPO_ROOT}}/deploy/systemd/{unit}"' in installer
        assert f'"${{SYSTEMD_DIR}}/{unit}"' in installer
    assert (
        "systemctl enable --now "
        "blueprint-task-evaluation-launch-preparation.path"
    ) in installer


def test_exact_sha_deployer_installs_and_arms_both_no_spend_intake_paths() -> None:
    assert "blueprint-task-evaluation-launch-preparation.service" in (
        deploy.DEFAULT_DEPLOYED_SYSTEMD_UNITS
    )
    assert "blueprint-task-evaluation-launch-preparation.path" in (
        deploy.DEFAULT_DEPLOYED_SYSTEMD_UNITS
    )
    assert deploy.DEFAULT_ALWAYS_ARM_PATH_UNITS == (
        "blueprint-task-evaluation-launch-preparation.path",
        "blueprint-task-evaluation-launch-activation.path",
    )
    assert "blueprint-task-evaluation-launch-dispatcher.path" not in (
        deploy.DEFAULT_ALWAYS_ARM_PATH_UNITS
    )
