"""ADP-009D/day 28: managed runtime credentials cannot control dispatch transport.

Covers deploy/terraform/main.tf and scripts/remote_cpu_deployment_scope.py.
"""

from pathlib import Path

from tests.test_terraform_remote_cpu_workers import _resources


def _terraform_resource_body(source, kind, name):
    return _resources(source)[(kind, name)]


ROOT = Path(__file__).resolve().parents[1]


def test_dispatch_project_has_no_runtime_service_agent_or_project_wide_delegate():
    source = (ROOT / "deploy/terraform/main.tf").read_text()
    project = _terraform_resource_body(source, "google_project", "remote_cpu_dispatch")
    assert "var.remote_cpu_dispatch_project_id" in project
    assert "prevent_destroy = true" in project
    policy = _terraform_resource_body(source, "data.google_iam_policy", "remote_cpu_dispatch_project")
    assert 'role    = "roles/owner"' in policy
    assert '["user:ohstnhunt@gmail.com"]' in policy
    assert policy.count("binding {") == 1
    apis = _terraform_resource_body(source, "google_project_service", "remote_cpu_dispatch_apis")
    assert "run.googleapis.com" not in apis
    assert "roles/iam.denyAdmin" not in source
    assert 'resource "google_iam_deny_policy" "remote_cpu_isolation"' not in source


def test_dispatcher_and_transport_are_outside_worker_project_and_old_identity_is_retained():
    source = (ROOT / "deploy/terraform/main.tf").read_text()
    for kind, name in (("google_service_account", "remote_cpu_host_dispatcher"),
                       ("google_storage_bucket", "remote_cpu_dispatch_transport"),
                       ("google_project_iam_custom_role", "remote_cpu_dispatch_transport_reader"),
                       ("google_project_iam_custom_role", "remote_cpu_dispatch_transport_writer")):
        body = _terraform_resource_body(source, kind, name)
        assert "google_project.remote_cpu_dispatch[0].project_id" in body
    quarantine = _terraform_resource_body(source, "google_service_account", "remote_cpu_quarantined_dispatcher")
    assert "google_project.remote_cpu[0].project_id" in quarantine
    assert "prevent_destroy = true" in quarantine
    assert "from = google_service_account.remote_cpu_dispatcher[0]" in source
    assert "to   = google_service_account.remote_cpu_quarantined_dispatcher[0]" in source
    old_bucket_policy = _terraform_resource_body(source, "data.google_iam_policy", "remote_cpu_transport")
    assert "google_service_account.remote_cpu_worker" not in old_bucket_policy
    assert "remote_cpu_dispatcher" not in old_bucket_policy
    job_policy = _terraform_resource_body(source, "google_cloud_run_v2_job_iam_policy", "remote_cpu_dispatcher")
    assert "data.google_iam_policy.remote_cpu_dispatch_job" in job_policy


def test_single_cap_covers_both_projects_without_label_based_cost_omissions():
    source = (ROOT / "deploy/terraform/main.tf").read_text()
    budget = _terraform_resource_body(source, "google_billing_budget", "remote_cpu_workers")
    assert '"projects/${google_project.remote_cpu[0].number}"' in budget
    assert '"projects/${google_project.remote_cpu_dispatch[0].number}"' in budget
    assert "labels =" not in budget

