from __future__ import annotations

from scripts import validate_pubsub_handoff_infra as validator
from scripts.validate_pubsub_handoff_infra import (
    has_project_runtime_dependency,
    has_run_e2e_result_binding,
    missing_dead_letter_service_agent_iam,
)

_MAIN_TF_WITHOUT_DEAD_LETTER_GRANTS = """
data "google_project" "current" {
  project_id = var.project_id
}

resource "google_pubsub_topic" "pipeline_dlq" {
  name = "pipeline-trigger-dlq"
}

resource "google_pubsub_subscription" "pipeline_handoff_listener" {
  name  = "blueprint-pipeline-handoff-listener"
  topic = google_pubsub_topic.capture_bridge_handoff.id

  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.pipeline_dlq.id
    max_delivery_attempts = 5
  }
}
"""

_DEAD_LETTER_GRANTS = """
locals {
  pubsub_service_agent = "serviceAccount:service-${data.google_project.current.number}@gcp-sa-pubsub.iam.gserviceaccount.com"
}

resource "google_pubsub_topic_iam_member" "pipeline_dlq_pubsub_agent_publisher" {
  topic  = google_pubsub_topic.pipeline_dlq.name
  role   = "roles/pubsub.publisher"
  member = local.pubsub_service_agent
}

resource "google_pubsub_subscription_iam_member" "pipeline_handoff_listener_pubsub_agent_subscriber" {
  subscription = google_pubsub_subscription.pipeline_handoff_listener.name
  role         = "roles/pubsub.subscriber"
  member       = local.pubsub_service_agent
}
"""


def test_pubsub_must_be_a_direct_production_dependency() -> None:
    optional_only = """
[project]
dependencies = ["google-cloud-storage>=2.10.0"]

[project.optional-dependencies]
cloud = ["google-cloud-pubsub>=2.21.0"]
"""
    direct = """
[project]
dependencies = ["google-cloud-pubsub>=2.21.0"]
"""

    assert has_project_runtime_dependency(optional_only, "google-cloud-pubsub") is False
    assert has_project_runtime_dependency(direct, "google-cloud-pubsub") is True


def test_runtime_dependency_parser_fails_closed_for_invalid_toml() -> None:
    assert has_project_runtime_dependency("[project", "google-cloud-pubsub") is False


def test_run_e2e_result_binding_accepts_formatter_multiline_conditional() -> None:
    source = """
result = (
    run_e2e(**run_kwargs)
    if run_e2e_enabled
    else {"status": "skipped"}
)
"""

    assert has_run_e2e_result_binding(source) is True


def test_run_e2e_result_binding_rejects_unbound_or_different_arguments() -> None:
    assert has_run_e2e_result_binding("run_e2e(**run_kwargs)") is False
    assert has_run_e2e_result_binding("result = run_e2e(capture_root=root)") is False


def test_dead_letter_policy_without_service_agent_grants_fails(tmp_path) -> None:
    main_tf = tmp_path / "main.tf"
    main_tf.write_text(_MAIN_TF_WITHOUT_DEAD_LETTER_GRANTS, encoding="utf-8")
    assert missing_dead_letter_service_agent_iam(main_tf.read_text(encoding="utf-8")) == [
        "Pub/Sub service agent identity (local.pubsub_service_agent)",
        "Pub/Sub service agent publisher on the dead-letter topic",
        "Pub/Sub service agent subscriber on the handoff subscription",
    ]

    main_tf.write_text(_MAIN_TF_WITHOUT_DEAD_LETTER_GRANTS + _DEAD_LETTER_GRANTS, encoding="utf-8")
    assert missing_dead_letter_service_agent_iam(main_tf.read_text(encoding="utf-8")) == []


def test_dead_letter_grants_must_give_the_service_agent_the_right_role() -> None:
    wrong_role = _DEAD_LETTER_GRANTS.replace('"roles/pubsub.publisher"', '"roles/pubsub.viewer"')
    wrong_member = _DEAD_LETTER_GRANTS.replace(
        "member       = local.pubsub_service_agent",
        'member       = "serviceAccount:${google_service_account.pipeline_handoff_listener.email}"',
    )

    assert missing_dead_letter_service_agent_iam(_MAIN_TF_WITHOUT_DEAD_LETTER_GRANTS + wrong_role) == [
        "Pub/Sub service agent publisher on the dead-letter topic",
    ]
    assert missing_dead_letter_service_agent_iam(_MAIN_TF_WITHOUT_DEAD_LETTER_GRANTS + wrong_member) == [
        "Pub/Sub service agent subscriber on the handoff subscription",
    ]


def test_repository_handoff_infra_passes_validation(capsys) -> None:
    validator.main()
    assert "Pub/Sub handoff infra validation passed" in capsys.readouterr().out
