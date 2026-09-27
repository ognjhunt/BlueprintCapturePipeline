from __future__ import annotations

from pathlib import Path

import pytest

from scripts import validate_pubsub_handoff_infra as validator
from scripts.validate_pubsub_handoff_infra import (
    has_project_runtime_dependency,
    has_run_e2e_result_binding,
    missing_dead_letter_retention,
    missing_dead_letter_service_agent_iam,
    missing_deploy_dead_letter_retention,
    missing_deploy_dead_letter_service_agent_bindings,
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

  depends_on = [google_pubsub_subscription.pipeline_dlq_retained]
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


REPO_ROOT = Path(__file__).resolve().parents[1]
_VALIDATED_FILES = (
    "pyproject.toml",
    "src/blueprint_pipeline/pubsub_handoff_listener.py",
    "deploy/terraform/main.tf",
    "deploy/scripts/deploy.sh",
    "deploy/systemd/blueprint-pubsub-handoff-listener.service",
    "deploy/systemd/blueprint-pubsub-handoff-listener.timer",
    "deploy/systemd/pipeline-control-plane.env.example",
    "scripts/install_live_pipeline_control_plane.sh",
)
_DEPLOY_SERVICE_AGENT_SUBSCRIPTION_GRANT = (
    "        gcloud pubsub subscriptions add-iam-policy-binding blueprint-pipeline-handoff-listener \\\n"
    '            --project "$PROJECT_ID" \\\n'
    '            --member "$PUBSUB_SERVICE_AGENT" \\\n'
    '            --role "roles/pubsub.subscriber" \\\n'
    "            --format=none \\\n"
    "            --quiet\n"
)


def _repository_copy(root: Path) -> Path:
    for relative in _VALIDATED_FILES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO_ROOT / relative).read_bytes())
    return root


def test_the_listener_grant_does_not_stand_in_for_the_service_agent_grant(tmp_path, capsys) -> None:
    root = _repository_copy(tmp_path)
    deploy_sh = root / "deploy" / "scripts" / "deploy.sh"
    text = deploy_sh.read_text(encoding="utf-8")
    assert text.count(_DEPLOY_SERVICE_AGENT_SUBSCRIPTION_GRANT) == 1
    deploy_sh.write_text(text.replace(_DEPLOY_SERVICE_AGENT_SUBSCRIPTION_GRANT, ""), encoding="utf-8")
    # The listener service account's own subscriber grant is still there.
    assert '--member "serviceAccount:${LISTENER_EMAIL}"' in deploy_sh.read_text(encoding="utf-8")

    with pytest.raises(SystemExit):
        validator.main(root)
    assert "deploy Pub/Sub service agent subscriber on the handoff subscription" in capsys.readouterr().err


_DEPLOY_LISTENER_SUBSCRIPTION_GRANT = (
    "    gcloud pubsub subscriptions add-iam-policy-binding blueprint-pipeline-handoff-listener \\\n"
    '        --project "$PROJECT_ID" \\\n'
    '        --member "serviceAccount:${LISTENER_EMAIL}" \\\n'
    '        --role "roles/pubsub.subscriber" \\\n'
    "        --quiet\n"
)
_DEPLOY_SERVICE_AGENT_IDENTITY = (
    "        PROJECT_NUMBER=\"$(gcloud projects describe \"$PROJECT_ID\" --format='value(projectNumber)')\"\n"
    '        PUBSUB_SERVICE_AGENT="serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-pubsub.iam.gserviceaccount.com"\n'
)
_DEPLOY_SERVICE_AGENT_TOPIC_GRANT = (
    "        gcloud pubsub topics add-iam-policy-binding pipeline-trigger-dlq \\\n"
    '            --project "$PROJECT_ID" \\\n'
    '            --member "$PUBSUB_SERVICE_AGENT" \\\n'
    '            --role "roles/pubsub.publisher" \\\n'
    "            --quiet\n"
)


def test_deploy_service_agent_bindings_need_their_own_member_and_role() -> None:
    complete = (_DEPLOY_SERVICE_AGENT_IDENTITY + _DEPLOY_SERVICE_AGENT_TOPIC_GRANT
                + _DEPLOY_SERVICE_AGENT_SUBSCRIPTION_GRANT + _DEPLOY_LISTENER_SUBSCRIPTION_GRANT)
    assert missing_deploy_dead_letter_service_agent_bindings(complete) == []

    listener_grant_only = complete.replace(_DEPLOY_SERVICE_AGENT_SUBSCRIPTION_GRANT, "")
    assert missing_deploy_dead_letter_service_agent_bindings(listener_grant_only) == [
        "deploy Pub/Sub service agent subscriber on the handoff subscription",
    ]
    wrong_role = complete.replace('"roles/pubsub.publisher"', '"roles/pubsub.viewer"')
    assert missing_deploy_dead_letter_service_agent_bindings(wrong_role) == [
        "deploy Pub/Sub service agent publisher on the dead-letter topic",
    ]
    no_identity = complete.replace(_DEPLOY_SERVICE_AGENT_IDENTITY, "")
    assert missing_deploy_dead_letter_service_agent_bindings(no_identity) == [
        "deploy project number for the Pub/Sub service agent",
        "deploy Pub/Sub service agent identity",
    ]


def test_repository_deploy_script_grants_the_service_agent() -> None:
    deploy_text = (REPO_ROOT / "deploy" / "scripts" / "deploy.sh").read_text(encoding="utf-8")
    assert missing_deploy_dead_letter_service_agent_bindings(deploy_text) == []


_DEAD_LETTER_RETAINED_SUBSCRIPTION = """
resource "google_pubsub_subscription" "pipeline_dlq_retained" {
  name  = "pipeline-trigger-dlq-retained"
  topic = google_pubsub_topic.pipeline_dlq.id

  ack_deadline_seconds       = 600
  message_retention_duration = "604800s"
  retain_acked_messages      = false

  expiration_policy {
    ttl = ""
  }
}
"""


def test_dead_lettering_needs_a_retained_never_expiring_subscription() -> None:
    granted = _MAIN_TF_WITHOUT_DEAD_LETTER_GRANTS + _DEAD_LETTER_GRANTS
    assert missing_dead_letter_retention(granted) == [
        "retained dead-letter subscription (7-day retention, never expires)",
    ]
    assert missing_dead_letter_retention(granted + _DEAD_LETTER_RETAINED_SUBSCRIPTION) == []

    for weakened in (
        _DEAD_LETTER_RETAINED_SUBSCRIPTION.replace('"604800s"', '"86400s"'),
        _DEAD_LETTER_RETAINED_SUBSCRIPTION.replace("retain_acked_messages      = false",
                                                   "retain_acked_messages      = true"),
        _DEAD_LETTER_RETAINED_SUBSCRIPTION.replace('    ttl = ""\n', '    ttl = "2678400s"\n'),
        _DEAD_LETTER_RETAINED_SUBSCRIPTION.replace("  ack_deadline_seconds       = 600\n", ""),
    ):
        assert missing_dead_letter_retention(granted + weakened) == [
            "retained dead-letter subscription (7-day retention, never expires)",
        ]

    ungated = granted.replace("\n  depends_on = [google_pubsub_subscription.pipeline_dlq_retained]\n", "")
    assert missing_dead_letter_retention(ungated + _DEAD_LETTER_RETAINED_SUBSCRIPTION) == [
        "dead-letter publisher grant waits for the retained subscription",
    ]


_DEPLOY_DEAD_LETTER_RETAINED_SUBSCRIPTION = (
    "        gcloud pubsub subscriptions create pipeline-trigger-dlq-retained \\\n"
    "            --topic pipeline-trigger-dlq \\\n"
    "            --ack-deadline 600 \\\n"
    "            --message-retention-duration 7d \\\n"
    "            --expiration-period never \\\n"
    "            --quiet\n"
)


def test_deploy_creates_the_retained_dead_letter_subscription_before_granting() -> None:
    grants = _DEPLOY_SERVICE_AGENT_IDENTITY + _DEPLOY_SERVICE_AGENT_TOPIC_GRANT
    assert missing_deploy_dead_letter_retention(_DEPLOY_DEAD_LETTER_RETAINED_SUBSCRIPTION + grants) == []
    assert missing_deploy_dead_letter_retention(grants) == ["deploy retained dead-letter subscription"]
    for weakened in (
        _DEPLOY_DEAD_LETTER_RETAINED_SUBSCRIPTION.replace("--expiration-period never", "--expiration-period 31d"),
        _DEPLOY_DEAD_LETTER_RETAINED_SUBSCRIPTION.replace("            --ack-deadline 600 \\\n", ""),
    ):
        assert missing_deploy_dead_letter_retention(weakened + grants) == ["deploy retained dead-letter subscription"]
    assert missing_deploy_dead_letter_retention(grants + _DEPLOY_DEAD_LETTER_RETAINED_SUBSCRIPTION) == [
        "deploy creates the retained dead-letter subscription before granting dead-letter access",
    ]


def test_repository_retains_dead_lettered_handoffs() -> None:
    terraform = (REPO_ROOT / "deploy" / "terraform" / "main.tf").read_text(encoding="utf-8")
    deploy_text = (REPO_ROOT / "deploy" / "scripts" / "deploy.sh").read_text(encoding="utf-8")
    assert missing_dead_letter_retention(terraform) == []
    assert missing_deploy_dead_letter_retention(deploy_text) == []


def test_deploy_service_agent_grants_are_quiet_and_function_local() -> None:
    commands = validator.shell_commands((REPO_ROOT / "deploy" / "scripts" / "deploy.sh").read_text(encoding="utf-8"))
    grants = [command for command in commands if '--member "$PUBSUB_SERVICE_AGENT"' in command]
    assert len(grants) == 2
    assert all(" --format=none " in f" {command} " for command in grants)  # no IAM policy dump in deploy logs
    assert "local PROJECT_NUMBER PUBSUB_SERVICE_AGENT" in commands


def test_the_pubsub_service_agent_is_defined_in_the_locals_section() -> None:
    terraform = validator.compact((REPO_ROOT / "deploy" / "terraform" / "main.tf").read_text(encoding="utf-8"))
    assert terraform.count("locals {") == 1
    assert validator.PUBSUB_SERVICE_AGENT_LOCAL in validator.terraform_block_body(terraform, "locals {")


def test_the_documented_dead_letter_replay_republishes_the_exact_bytes() -> None:
    guide = (REPO_ROOT / "docs" / "LIVE_PIPELINE_SETUP.md").read_text(encoding="utf-8")
    assert "<decoded message.data>" not in guide  # a hand-decoded copy can change the payload digest
    assert "publisher.publish(topic, received.message.data).result()" in guide
