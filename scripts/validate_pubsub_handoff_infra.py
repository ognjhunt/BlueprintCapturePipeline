#!/usr/bin/env python3
"""Validate that BlueprintCapture Pub/Sub handoff automation is deploy-wired."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def fail(message: str) -> None:
    print(f"Pub/Sub handoff infra validation failed: {message}", file=sys.stderr)
    sys.exit(1)


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def require_contains(text: str, needle: str, description: str) -> None:
    if needle not in text:
        fail(f"missing {description}: {needle}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = repo_root / "pyproject.toml"
    listener = repo_root / "src" / "blueprint_pipeline" / "pubsub_handoff_listener.py"
    terraform = repo_root / "deploy" / "terraform" / "main.tf"
    deploy_script = repo_root / "deploy" / "scripts" / "deploy.sh"

    for path in (pyproject, listener, terraform, deploy_script):
        if not path.exists():
            fail(f"{path.relative_to(repo_root)} is missing")

    pyproject_text = pyproject.read_text(encoding="utf-8")
    listener_text = listener.read_text(encoding="utf-8")
    terraform_text = compact(terraform.read_text(encoding="utf-8"))
    deploy_text = deploy_script.read_text(encoding="utf-8")
    deploy_compact = compact(deploy_text)

    for needle, description in [
        (
            'blueprint-pubsub-handoff-listener = "blueprint_pipeline.pubsub_handoff_listener:main"',
            "CLI entrypoint",
        ),
        ("google-cloud-pubsub", "Pub/Sub package dependency"),
    ]:
        require_contains(pyproject_text, needle, description)

    for needle, description in [
        ("def parse_handoff_payload", "payload parser"),
        ("def stage_handoff_capture", "GCS staging function"),
        ("def process_handoff_payload", "handoff processor"),
        ("def pull_and_process", "pull subscriber loop"),
        ("from google.cloud import pubsub_v1", "Pub/Sub subscriber import"),
        ("subscriber.pull", "pull subscription call"),
        ("from .run_e2e import run_end_to_end", "pipeline entrypoint import"),
        ("run_e2e: Callable[..., dict[str, Any]] = run_end_to_end", "pipeline invocation default"),
        ("result = run_e2e(", "pipeline invocation"),
        ("subscriber.acknowledge", "post-success ack"),
        ("run_evaluation_prep=run_evaluation_prep", "evaluation prep handoff"),
    ]:
        require_contains(listener_text, needle, description)

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

    print("Pub/Sub handoff infra validation passed: listener, Terraform subscription, IAM, DLQ, and deploy script wiring are present.")


if __name__ == "__main__":
    main()
