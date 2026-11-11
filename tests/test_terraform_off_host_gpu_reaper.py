# Covers (for impacted-test selection):
#   deploy/terraform/main.tf
#   deploy/terraform/terraform.tfvars.example
"""The off-host GPU reaper's Terraform is off by default, ticks every ten minutes and alerts on the existing route.

Read as text, as the other Terraform contract tests read it: there is no terraform binary or GCP credential here,
and nothing in this file plans or applies.
"""

from __future__ import annotations

import re
from pathlib import Path

from blueprint_pipeline import off_host_gpu_reaper as reaper
from tests.test_deploy_systemd_contract import _terraform_resource_body, _terraform_variable_body

MAIN = (Path(__file__).resolve().parents[1] / "deploy" / "terraform" / "main.tf").read_text(encoding="utf-8")
FLAG = "var.off_host_gpu_reaper_enabled ? 1 : 0"


def _reaper_blocks() -> list[tuple[str, str, str]]:
    return [(kind, name, match) for match, kind, name in
            ((m.group(0), m.group(1), m.group(2))
             for m in re.finditer(r'(?m)^(?:resource|data) "([a-z0-9_]+)" "(off_host_gpu_reaper[a-z0-9_]*)" \{$', MAIN))]


def test_every_reaper_resource_exists_only_while_the_flag_is_on():
    assert "default     = false" in _terraform_variable_body(MAIN, "off_host_gpu_reaper_enabled")
    blocks = _reaper_blocks()
    assert {kind for kind, _name, _header in blocks} >= {
        "google_cloud_run_v2_job", "google_cloud_scheduler_job", "google_storage_bucket",
        "google_monitoring_alert_policy", "google_secret_manager_secret"}
    for kind, name, header in blocks:
        start = MAIN.index(header)
        body = MAIN[start:MAIN.index("\n}\n", start)]
        assert f"count    = {FLAG}" in body or f"count = {FLAG}" in body, (kind, name)


def test_the_job_runs_one_unretried_reaper_pass_with_the_key_as_a_mounted_secret():
    job = _terraform_resource_body(MAIN, "google_cloud_run_v2_job", "off_host_gpu_reaper")
    assert 'command = ["python", "-m", "blueprint_pipeline.off_host_gpu_reaper"]' in job
    assert "max_retries           = 0" in job and 'timeout               = "300s"' in job
    assert 'name  = "BLUEPRINT_OFF_HOST_GPU_REAPER_BUCKET"' in job and reaper.BUCKET_ENV in job
    assert 'value = "/secrets/vast/vast_api_key"' in job and 'mount_path = "/secrets/vast"' in job
    assert "secret_key_ref" not in job  # the key is a file the provider reads, never an environment value
    assert 'variable "off_host_gpu_reaper_vast_api_key"' not in MAIN


def test_the_schedule_is_every_ten_minutes_and_starts_the_reaper_job():
    schedule = _terraform_resource_body(MAIN, "google_cloud_scheduler_job", "off_host_gpu_reaper")
    assert 'schedule         = "*/10 * * * *"' in schedule
    assert "jobs/${google_cloud_run_v2_job.off_host_gpu_reaper[0].name}:run" in schedule
    assert "google_service_account.off_host_gpu_reaper_scheduler[0].email" in schedule
    invoker = _terraform_resource_body(MAIN, "google_cloud_run_v2_job_iam_member", "off_host_gpu_reaper_scheduler")
    assert 'role     = "roles/run.invoker"' in invoker


def test_a_failed_or_missing_pass_alerts_on_the_existing_channels():
    alert = _terraform_resource_body(MAIN, "google_monitoring_alert_policy", "off_host_gpu_reaper_failures")
    assert "notification_channels = var.monitoring_notification_channels" in alert
    assert "var.allow_empty_monitoring_notification_channels || length(var.monitoring_notification_channels) > 0" in alert
    assert alert.count("resource.labels.job_name=\\\"${local.off_host_gpu_reaper_job_name}\\\"") == 2
    assert "metric.labels.result=\\\"failed\\\"" in alert and "condition_absent" in alert
    assert 'off_host_gpu_reaper_job_name = "blueprint-off-host-gpu-reaper"' in MAIN


def test_the_reaper_reads_only_and_the_host_alone_rewrites_the_state_object():
    roles = {name: re.search(r'role   = "([^"]+)"', _terraform_resource_body(MAIN, "google_storage_bucket_iam_member", name))[1]
             for name in ("off_host_gpu_reaper_reader", "off_host_gpu_reaper_report_writer",
                          "off_host_gpu_reaper_heartbeat_writer")}
    assert roles == {"off_host_gpu_reaper_reader": "roles/storage.objectViewer",
                     "off_host_gpu_reaper_report_writer": "roles/storage.objectCreator",
                     "off_host_gpu_reaper_heartbeat_writer": "roles/storage.objectUser"}
    bucket = _terraform_resource_body(MAIN, "google_storage_bucket", "off_host_gpu_reaper")
    assert 'public_access_prevention    = "enforced"' in bucket and f'matches_prefix = ["{reaper.REPORT_PREFIX}"]' in bucket
    assert 'resource "google_service_account_key"' not in MAIN
