#!/usr/bin/env python3
"""Validate that BlueprintCapture Pub/Sub handoff automation is deploy-wired."""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]


def fail(message: str) -> None:
    print(f"Pub/Sub handoff infra validation failed: {message}", file=sys.stderr)
    sys.exit(1)


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def require_contains(text: str, needle: str, description: str) -> None:
    if needle not in text:
        fail(f"missing {description}: {needle}")


def has_project_runtime_dependency(text: str, package_name: str) -> bool:
    """Return whether a package is a direct production dependency.

    A package present only in an optional extra is insufficient for the
    systemd deployment, which intentionally runs ``uv sync --no-dev`` without
    extras.
    """

    try:
        payload = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return False
    project = payload.get("project")
    if not isinstance(project, dict):
        return False
    dependencies = project.get("dependencies")
    if not isinstance(dependencies, list):
        return False
    prefix = re.compile(
        rf"^\s*{re.escape(package_name)}(?:\s|$|[<>=!~;@\[])",
        flags=re.IGNORECASE,
    )
    return any(isinstance(item, str) and prefix.search(item) for item in dependencies)


def has_run_e2e_result_binding(text: str) -> bool:
    """Return whether the listener binds the canonical run_e2e result.

    The listener intentionally formats its conditional invocation across
    multiple lines.  Match Python whitespace instead of coupling the deploy
    preflight to one formatter layout.
    """

    return bool(
        re.search(
            r"\bresult\s*=\s*\(?\s*run_e2e\s*\(\s*\*\*run_kwargs\s*\)",
            text,
        )
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = repo_root / "pyproject.toml"
    listener = repo_root / "src" / "blueprint_pipeline" / "pubsub_handoff_listener.py"
    terraform = repo_root / "deploy" / "terraform" / "main.tf"
    deploy_script = repo_root / "deploy" / "scripts" / "deploy.sh"
    systemd_service = repo_root / "deploy" / "systemd" / "blueprint-pubsub-handoff-listener.service"
    systemd_timer = repo_root / "deploy" / "systemd" / "blueprint-pubsub-handoff-listener.timer"
    systemd_env_example = repo_root / "deploy" / "systemd" / "pipeline-control-plane.env.example"
    systemd_installer = repo_root / "scripts" / "install_live_pipeline_control_plane.sh"

    for path in (
        pyproject,
        listener,
        terraform,
        deploy_script,
        systemd_service,
        systemd_timer,
        systemd_env_example,
        systemd_installer,
    ):
        if not path.exists():
            fail(f"{path.relative_to(repo_root)} is missing")

    pyproject_text = pyproject.read_text(encoding="utf-8")
    listener_text = listener.read_text(encoding="utf-8")
    terraform_text = compact(terraform.read_text(encoding="utf-8"))
    deploy_text = deploy_script.read_text(encoding="utf-8")
    deploy_compact = compact(deploy_text)
    systemd_service_text = systemd_service.read_text(encoding="utf-8")
    systemd_timer_text = systemd_timer.read_text(encoding="utf-8")
    systemd_env_text = systemd_env_example.read_text(encoding="utf-8")
    systemd_installer_text = systemd_installer.read_text(encoding="utf-8")

    for needle, description in [
        (
            'blueprint-pubsub-handoff-listener = "blueprint_pipeline.pubsub_handoff_listener:main"',
            "CLI entrypoint",
        ),
    ]:
        require_contains(pyproject_text, needle, description)
    if not has_project_runtime_dependency(pyproject_text, "google-cloud-pubsub"):
        fail("google-cloud-pubsub must be a direct [project].dependencies runtime dependency")

    for needle, description in [
        ("def parse_handoff_payload", "payload parser"),
        ("def stage_handoff_capture", "GCS staging function"),
        ("def process_handoff_payload", "handoff processor"),
        ("def pull_and_process", "pull subscriber loop"),
        ("def _control_plane_handoff_payload", "control-plane payload enrichment"),
        ("stage_capture_handoff_for_control_plane", "control-plane staging helper call"),
        ("--stage-control-plane", "control-plane staging CLI flag"),
        ("--skip-run-e2e", "stage-only listener CLI flag"),
        ("from google.cloud import pubsub_v1", "Pub/Sub subscriber import"),
        ("subscriber.pull", "pull subscription call"),
        ("from .run_e2e import run_end_to_end", "pipeline entrypoint import"),
        ("run_e2e: Callable[..., dict[str, Any]] = run_end_to_end", "pipeline invocation default"),
        ("subscriber.acknowledge", "post-success ack"),
        ("run_evaluation_prep=run_evaluation_prep", "evaluation prep handoff"),
    ]:
        require_contains(listener_text, needle, description)
    if not has_run_e2e_result_binding(listener_text):
        fail("missing pipeline invocation result binding")

    for needle, description in [
        ('resource "google_pubsub_topic" "pipeline_trigger"', "descriptor topic resource"),
        ('name = "blueprint-capture-pipeline-handoff"', "descriptor topic name"),
        ('resource "google_pubsub_topic" "capture_bridge_handoff"', "dedicated handoff topic resource (XR-04)"),
        ('name = "blueprint-capture-bridge-handoff"', "dedicated handoff topic name (XR-04)"),
        ('resource "google_pubsub_topic" "pipeline_dlq"', "dead-letter topic resource"),
        ('resource "google_pubsub_subscription" "pipeline_handoff_listener"', "handoff subscription resource"),
        ('name = "blueprint-pipeline-handoff-listener"', "handoff subscription name"),
        # XR-04: listener must bind to the dedicated handoff topic, NOT the descriptor topic.
        ("topic = google_pubsub_topic.capture_bridge_handoff.id", "subscription bound to dedicated handoff topic"),
        ("ack_deadline_seconds = 600", "long ack deadline"),
        ('message_retention_duration = "604800s"', "seven-day retention"),
        ('maximum_backoff = "600s"', "valid retry maximum backoff"),
        ("dead_letter_policy", "dead-letter policy"),
        ("max_delivery_attempts = 5", "dead-letter delivery cap"),
        ('role = "roles/pubsub.subscriber"', "subscriber IAM role"),
        ("google_service_account.pipeline_runner.email", "pipeline runner subscriber principal"),
        ("SWAP_TRIGGER_HANDOFF_PUBSUB_TOPIC = google_pubsub_topic.capture_bridge_handoff.name", "storage-trigger handoff topic env var"),
        ('output "pubsub_handoff_listener_subscription"', "subscription output"),
    ]:
        require_contains(terraform_text, needle, description)

    # XR-04: listener subscription must NOT be bound to the descriptor topic.
    if "topic = google_pubsub_topic.pipeline_trigger.id" in terraform_text and (
        'resource "google_pubsub_subscription" "pipeline_handoff_listener" { name'
        ' = "blueprint-pipeline-handoff-listener" topic ='
        " google_pubsub_topic.pipeline_trigger.id" in terraform_text
    ):
        fail("handoff listener subscription is still bound to the descriptor topic (XR-04 regression)")

    for needle, description in [
        ('SWAP_TOPIC="${SWAP_TOPIC:-blueprint-capture-pipeline-handoff}"', "deploy default descriptor topic"),
        ('HANDOFF_TOPIC="${HANDOFF_TOPIC:-blueprint-capture-bridge-handoff}"', "deploy default handoff topic (XR-04)"),
        ('TOPICS=("$SWAP_TOPIC" "$HANDOFF_TOPIC" "pipeline-trigger-dlq")', "deploy topic creation list"),
        ("SWAP_TRIGGER_HANDOFF_PUBSUB_TOPIC=${HANDOFF_TOPIC}", "deploy storage-trigger handoff topic env var"),
        ("gcloud pubsub subscriptions create blueprint-pipeline-handoff-listener", "deploy subscription creation"),
        ('--topic "$HANDOFF_TOPIC"', "deploy subscription bound to dedicated handoff topic"),
        ("--ack-deadline 600", "deploy subscription ack deadline"),
        ("--message-retention-duration 7d", "deploy subscription retention"),
        ("--max-retry-delay 600s", "deploy subscription retry maximum backoff"),
        ("--dead-letter-topic pipeline-trigger-dlq", "deploy dead-letter topic"),
        ("--max-delivery-attempts 5", "deploy dead-letter delivery cap"),
        ("python3 \"$PROJECT_ROOT/scripts/validate_pubsub_handoff_infra.py\"", "deploy preflight validator"),
    ]:
        require_contains(deploy_text, needle, description)

    if "Pub/Sub Topic: pipeline-trigger" in deploy_text:
        fail("deploy summary still references stale pipeline-trigger topic")
    require_contains(deploy_compact, 'Pub/Sub Topic: ${SWAP_TOPIC}', "deploy summary canonical topic")

    for needle, description in [
        ("blueprint_pipeline.pubsub_handoff_listener", "systemd listener module entrypoint"),
        ("BLUEPRINT_PUBSUB_HANDOFF_SUBSCRIPTION=blueprint-pipeline-handoff-listener", "systemd listener subscription env"),
        ("BLUEPRINT_PUBSUB_HANDOFF_STAGE_CONTROL_PLANE=true", "systemd listener stages control-plane input"),
        ("BLUEPRINT_PUBSUB_HANDOFF_SKIP_RUN_E2E=true", "systemd listener leaves execution to control plane"),
        ("--max-messages", "systemd listener bounded batch flag"),
    ]:
        require_contains(systemd_service_text, needle, description)
    for needle, description in [
        ("OnUnitActiveSec=1min", "systemd listener timer cadence"),
        ("Unit=blueprint-pubsub-handoff-listener.service", "systemd listener timer unit binding"),
    ]:
        require_contains(systemd_timer_text, needle, description)
    for needle, description in [
        ("blueprint-pubsub-handoff-listener.service", "systemd installer copies listener service"),
        ("blueprint-pubsub-handoff-listener.timer", "systemd installer copies listener timer"),
        ("systemctl enable --now blueprint-pubsub-handoff-listener.timer", "systemd installer enables listener timer"),
        ('STATE_DIR="${STATE_DIR:-/var/lib/blueprint/pipeline-control-plane}"', "systemd installer state dir default"),
        ('HANDOFF_DIR="${HANDOFF_DIR:-/var/lib/blueprint/pubsub-handoffs}"', "systemd installer handoff dir default"),
        ('"${STATE_DIR}/robot-eval-job-requests"', "systemd installer creates request inbox"),
        ('"${STATE_DIR}/incoming_webapp_job_requests"', "systemd installer creates intake work dir"),
    ]:
        require_contains(systemd_installer_text, needle, description)
    for needle, description in [
        (
            "BLUEPRINT_ROBOT_EVAL_JOB_REQUEST_INBOX=/var/lib/blueprint/pipeline-control-plane/robot-eval-job-requests",
            "env example configured request inbox",
        ),
        (
            "BLUEPRINT_LIVE_PIPELINE_INTAKE_WORK_DIR=/var/lib/blueprint/pipeline-control-plane/incoming_webapp_job_requests",
            "env example configured intake work dir",
        ),
        (
            "BLUEPRINT_LIVE_PIPELINE_STAGED_INPUTS_PATH=/var/lib/blueprint/pipeline-control-plane/live_pipeline_staged_inputs.json",
            "env example configured staged inputs path",
        ),
    ]:
        require_contains(systemd_env_text, needle, description)

    print("Pub/Sub handoff infra validation passed: listener, Terraform subscription, IAM, DLQ, and deploy script wiring are present.")


if __name__ == "__main__":
    main()
