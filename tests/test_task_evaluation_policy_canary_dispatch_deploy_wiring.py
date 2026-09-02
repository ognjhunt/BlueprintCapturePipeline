from __future__ import annotations

from pathlib import Path

import scripts.deploy_control_plane_commit as deploy


ROOT = Path(__file__).resolve().parents[1]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_activation_remains_no_spend_and_only_emits_paid_queue() -> None:
    activation = _text(
        "deploy/systemd/blueprint-task-evaluation-launch-activation.service"
    )
    assert "task_evaluation_policy_canary_dispatcher" not in activation
    assert "paid_resource_allocator" not in activation
    assert "POLICY_CANARY_DISPATCH_QUEUE_ROOT" in activation


def test_canary_paid_dispatcher_is_installed_but_never_always_armed() -> None:
    service_name = "blueprint-task-evaluation-policy-canary-dispatcher.service"
    path_name = "blueprint-task-evaluation-policy-canary-dispatcher.path"
    service = _text(f"deploy/systemd/{service_name}")
    path = _text(f"deploy/systemd/{path_name}")
    installer = _text("scripts/install_live_pipeline_control_plane.sh")

    assert "task_evaluation_policy_canary_dispatcher" in service
    assert "--dispatch-queue-root" in service
    assert "--execution-setup-template" in service
    assert "--billing-audit-root" in service
    assert "--hotfix-overlay" in service
    assert "BLUEPRINT_TASK_EVALUATION_CANARY_HOTFIX_OVERLAY" in service
    assert "--execute" in service
    # Operator-staged inputs enter through canonical, access-checked arguments
    # rather than files hand-placed inside the service account's run directory.
    assert (
        'ARGS+=(--hotfix-overlay "$${BLUEPRINT_TASK_EVALUATION_CANARY_HOTFIX_OVERLAY}")'
        in service
    )
    assert (
        'ARGS+=(--machine-avoidlist '
        '"$${BLUEPRINT_TASK_EVALUATION_POLICY_CANARY_MACHINE_AVOIDLIST}")'
    ) in service
    assert "KillMode=process" in service
    assert (
        "PIPELINE_TASK_EVALUATION_RUN_WEBAPP_URL="
        "https://tryblueprint.io/api/internal/pipeline/capture-task-evaluation-runs"
    ) in service
    assert (
        "PIPELINE_TASK_EVALUATION_LAUNCH_PROGRESS_WEBAPP_URL="
        "https://tryblueprint.io/api/internal/pipeline/task-evaluation-launch-progress"
    ) in service
    assert "EnvironmentFile=-/etc/blueprint/pipeline-control-plane.env" in service
    assert (
        "PIPELINE_SYNC_TOKEN_FILE=/etc/blueprint/provider-secrets/pipeline_sync_token"
        in service
    )
    assert "task-evaluation-policy-canary-dispatches/pending" in path
    assert "gpu_spend_guard/billing-audit" in path
    assert service_name in deploy.DEFAULT_DEPLOYED_SYSTEMD_UNITS
    assert path_name in deploy.DEFAULT_DEPLOYED_SYSTEMD_UNITS
    assert path_name not in deploy.DEFAULT_ALWAYS_ARM_PATH_UNITS
    assert f"systemctl enable --now {path_name}" not in installer
    assert f"deploy/systemd/{service_name}" in installer
    assert f"deploy/systemd/{path_name}" in installer


def test_iteration_wrapper_prefers_signed_canary_overlay_for_eligible_deltas() -> None:
    wrapper = _text("scripts/deploy_control_plane_iteration.sh")

    assert "task_evaluation_canary_hotfix_overlay route" in wrapper
    assert "task_evaluation_canary_hotfix_overlay prepare" in wrapper
    assert "task_evaluation_canary_hotfix_overlay install" in wrapper
    assert "normal exact-main deployment remains required for promotion" in wrapper


def test_direct_launch_dispatcher_can_only_queue_canary_preparation() -> None:
    service = _text("deploy/systemd/blueprint-task-evaluation-launch-dispatcher.service")
    assert "BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_QUEUE_ROOT=" in service
    assert "task-evaluation-launch-preparations" in service
