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
    assert "--execute" in service
    assert "KillMode=process" in service
    assert "task-evaluation-policy-canary-dispatches/pending" in path
    assert service_name in deploy.DEFAULT_DEPLOYED_SYSTEMD_UNITS
    assert path_name in deploy.DEFAULT_DEPLOYED_SYSTEMD_UNITS
    assert path_name not in deploy.DEFAULT_ALWAYS_ARM_PATH_UNITS
    assert f"systemctl enable --now {path_name}" not in installer
    assert f"deploy/systemd/{service_name}" in installer
    assert f"deploy/systemd/{path_name}" in installer
