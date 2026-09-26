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


PUBSUB_SERVICE_AGENT_LOCAL = (
    'pubsub_service_agent = "serviceAccount:service-${data.google_project.current.number}'
    '@gcp-sa-pubsub.iam.gserviceaccount.com"'
)
DEAD_LETTER_SERVICE_AGENT_GRANTS = (
    (
        'resource "google_pubsub_topic_iam_member" "pipeline_dlq_pubsub_agent_publisher" {',
        (
            "topic = google_pubsub_topic.pipeline_dlq.name",
            'role = "roles/pubsub.publisher"',
            "member = local.pubsub_service_agent",
        ),
        "Pub/Sub service agent publisher on the dead-letter topic",
    ),
    (
        'resource "google_pubsub_subscription_iam_member" '
        '"pipeline_handoff_listener_pubsub_agent_subscriber" {',
        (
            "subscription = google_pubsub_subscription.pipeline_handoff_listener.name",
            'role = "roles/pubsub.subscriber"',
            "member = local.pubsub_service_agent",
        ),
        "Pub/Sub service agent subscriber on the handoff subscription",
    ),
)


def terraform_block_body(text: str, header: str) -> str | None:
    """Return the body of the first block opened by ``header``, or None."""

    start = text.find(header)
    if start < 0:
        return None
    body_start = start + len(header)
    depth = 1
    for index in range(body_start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[body_start:index]
    return None


DEPLOY_PROJECT_NUMBER_LOOKUP = "gcloud projects describe \"$PROJECT_ID\" --format='value(projectNumber)'"
DEPLOY_PUBSUB_SERVICE_AGENT_IDENTITY = (
    'PUBSUB_SERVICE_AGENT="serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-pubsub.iam.gserviceaccount.com"'
)
DEPLOY_PUBSUB_SERVICE_AGENT_MEMBER = '--member "$PUBSUB_SERVICE_AGENT"'
DEPLOY_DEAD_LETTER_SERVICE_AGENT_BINDINGS = (
    (
        "gcloud pubsub topics add-iam-policy-binding pipeline-trigger-dlq",
        '--role "roles/pubsub.publisher"',
        "deploy Pub/Sub service agent publisher on the dead-letter topic",
    ),
    (
        "gcloud pubsub subscriptions add-iam-policy-binding blueprint-pipeline-handoff-listener",
        '--role "roles/pubsub.subscriber"',
        "deploy Pub/Sub service agent subscriber on the handoff subscription",
    ),
)


def shell_commands(text: str) -> list[str]:
    """Logical shell lines: backslash continuations joined, whitespace collapsed."""

    return [compact(line).strip() for line in re.sub(r"\\\r?\n", " ", text).splitlines()]


def missing_deploy_dead_letter_service_agent_bindings(deploy_text: str) -> list[str]:
    """Describe each deploy.sh piece the dead-letter grants need that is missing.

    A binding counts only when one gcloud command names the resource, the
    service agent as its member and the exact role, so the listener service
    account's own subscriber grant cannot stand in for the service agent's.
    """

    commands = shell_commands(deploy_text)
    missing: list[str] = []
    if not any(DEPLOY_PROJECT_NUMBER_LOOKUP in command for command in commands):
        missing.append("deploy project number for the Pub/Sub service agent")
    if not any(command.startswith(DEPLOY_PUBSUB_SERVICE_AGENT_IDENTITY) for command in commands):
        missing.append("deploy Pub/Sub service agent identity")
    for command_prefix, role, description in DEPLOY_DEAD_LETTER_SERVICE_AGENT_BINDINGS:
        if not any(
            command.startswith(f"{command_prefix} ")
            and f" {DEPLOY_PUBSUB_SERVICE_AGENT_MEMBER} " in f" {command} "
            and f" {role} " in f" {command} "
            for command in commands
        ):
            missing.append(description)
    return missing


def missing_dead_letter_service_agent_iam(terraform_text: str) -> list[str]:
    """Describe each grant the handoff dead-letter policy needs that is missing.

    Pub/Sub dead-letters a message as its own service agent, which needs
    publisher on the dead-letter topic and subscriber on the source
    subscription. Without both, the policy never moves an exhausted handoff.
    """

    text = compact(terraform_text)
    missing: list[str] = []
    if PUBSUB_SERVICE_AGENT_LOCAL not in text:
        missing.append("Pub/Sub service agent identity (local.pubsub_service_agent)")
    for header, attributes, description in DEAD_LETTER_SERVICE_AGENT_GRANTS:
        body = terraform_block_body(text, header)
        if body is None or any(f" {attribute} " not in f" {body} " for attribute in attributes):
            missing.append(description)
    return missing


DEAD_LETTER_RETAINED_SUBSCRIPTION_HEADER = 'resource "google_pubsub_subscription" "pipeline_dlq_retained" {'
DEAD_LETTER_RETAINED_SUBSCRIPTION_ATTRIBUTES = (
    "topic = google_pubsub_topic.pipeline_dlq.id",
    'message_retention_duration = "604800s"',
    "retain_acked_messages = false",
    'expiration_policy { ttl = "" }',
)
DEAD_LETTER_PUBLISHER_WAITS_FOR_RETENTION = "depends_on = [google_pubsub_subscription.pipeline_dlq_retained]"


def missing_dead_letter_retention(terraform_text: str) -> list[str]:
    """Describe what keeps dead-lettered handoffs recoverable that is missing.

    Pub/Sub keeps a message only for the subscriptions a topic has when the
    message is published, so a dead-letter topic without one discards every
    exhausted handoff. The retained subscription keeps them for seven days and
    never expires, and the service agent may not publish (dead-letter) until it
    exists.
    """

    text = compact(terraform_text)
    missing: list[str] = []
    body = terraform_block_body(text, DEAD_LETTER_RETAINED_SUBSCRIPTION_HEADER)
    if body is None or any(
        f" {attribute} " not in f" {body} " for attribute in DEAD_LETTER_RETAINED_SUBSCRIPTION_ATTRIBUTES
    ):
        missing.append("retained dead-letter subscription (7-day retention, never expires)")
    publisher = terraform_block_body(text, DEAD_LETTER_SERVICE_AGENT_GRANTS[0][0])
    if publisher is None or f" {DEAD_LETTER_PUBLISHER_WAITS_FOR_RETENTION} " not in f" {publisher} ":
        missing.append("dead-letter publisher grant waits for the retained subscription")
    return missing


DEPLOY_DEAD_LETTER_RETAINED_SUBSCRIPTION = "gcloud pubsub subscriptions create pipeline-trigger-dlq-retained"
DEPLOY_DEAD_LETTER_RETAINED_SUBSCRIPTION_FLAGS = (
    "--topic pipeline-trigger-dlq",
    "--message-retention-duration 7d",
    "--expiration-period never",
)


def missing_deploy_dead_letter_retention(deploy_text: str) -> list[str]:
    """Describe what deploy.sh lacks to keep dead-lettered handoffs recoverable."""

    commands = shell_commands(deploy_text)
    created = [
        index
        for index, command in enumerate(commands)
        if command.startswith(f"{DEPLOY_DEAD_LETTER_RETAINED_SUBSCRIPTION} ")
        and all(f" {flag} " in f" {command} " for flag in DEPLOY_DEAD_LETTER_RETAINED_SUBSCRIPTION_FLAGS)
    ]
    if not created:
        return ["deploy retained dead-letter subscription"]
    publisher_command = DEPLOY_DEAD_LETTER_SERVICE_AGENT_BINDINGS[0][0]
    granted = [index for index, command in enumerate(commands) if command.startswith(f"{publisher_command} ")]
    if granted and min(granted) < min(created):
        return ["deploy creates the retained dead-letter subscription before granting dead-letter access"]
    return []


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


def main(repo_root: Path | None = None) -> None:
    repo_root = repo_root or Path(__file__).resolve().parents[1]
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
        ('resource "google_service_account" "pipeline_handoff_listener"', "dedicated listener identity"),
        (
            'resource "google_pubsub_subscription_iam_member" "pipeline_handoff_listener_subscriber"',
            "subscription-scoped listener IAM",
        ),
        (
            'resource "google_storage_bucket_iam_member" "pipeline_handoff_listener_capture_reader"',
            "bucket-scoped listener IAM",
        ),
        (
            "google_service_account.pipeline_handoff_listener.email",
            "dedicated listener principal",
        ),
        ("SWAP_TRIGGER_HANDOFF_PUBSUB_TOPIC = google_pubsub_topic.capture_bridge_handoff.name", "storage-trigger handoff topic env var"),
        ('output "pubsub_handoff_listener_subscription"', "subscription output"),
    ]:
        require_contains(terraform_text, needle, description)
    missing_dead_letter_iam = missing_dead_letter_service_agent_iam(terraform_text)
    if missing_dead_letter_iam:
        fail(
            "dead-letter policy cannot move exhausted handoffs; missing "
            + "; ".join(missing_dead_letter_iam)
        )
    missing_retention = missing_dead_letter_retention(terraform_text)
    if missing_retention:
        fail("dead-lettered handoffs would be discarded; missing " + "; ".join(missing_retention))
    if 'resource "google_project_iam_member" "pipeline_runner_pubsub_subscriber"' in terraform_text:
        fail("pipeline-runner must not retain project-wide Pub/Sub subscriber IAM")

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
        ('"pipeline-handoff-listener"', "deploy dedicated listener service account"),
        (
            "gcloud pubsub subscriptions add-iam-policy-binding blueprint-pipeline-handoff-listener",
            "deploy exact subscription IAM",
        ),
        (
            'gcloud storage buckets add-iam-policy-binding "gs://${STORAGE_BUCKET}"',
            "deploy exact capture-bucket IAM",
        ),
        ("python3 \"$PROJECT_ROOT/scripts/validate_pubsub_handoff_infra.py\"", "deploy preflight validator"),
    ]:
        require_contains(deploy_text, needle, description)
    missing_deploy_bindings = missing_deploy_dead_letter_service_agent_bindings(deploy_text)
    if missing_deploy_bindings:
        fail(
            "deploy script cannot enable dead-lettering; missing "
            + "; ".join(missing_deploy_bindings)
        )
    missing_deploy_retention = missing_deploy_dead_letter_retention(deploy_text)
    if missing_deploy_retention:
        fail("deploy script would discard dead-lettered handoffs; " + "; ".join(missing_deploy_retention))
    runner_grants = deploy_text[
        deploy_text.find("RUNNER_EMAIL=") : deploy_text.find("# The persistent-host listener")
    ]
    if '"roles/pubsub.subscriber"' in compact(runner_grants):
        fail("deploy script grants Pub/Sub subscriber to the broad pipeline runner")

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
        ("OnActiveSec=30s", "systemd listener initial cadence"),
        ("OnUnitInactiveSec=1min", "systemd listener repeat cadence"),
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
        (
            "GOOGLE_APPLICATION_CREDENTIALS=/etc/blueprint/credentials/pipeline-handoff-listener.json",
            "dedicated listener credential path",
        ),
        ("GOOGLE_CLOUD_PROJECT=blueprint-8c1ca", "listener project id"),
    ]:
        require_contains(systemd_env_text, needle, description)

    print("Pub/Sub handoff infra validation passed: listener, Terraform subscription, IAM, DLQ, and deploy script wiring are present.")


if __name__ == "__main__":
    main()
