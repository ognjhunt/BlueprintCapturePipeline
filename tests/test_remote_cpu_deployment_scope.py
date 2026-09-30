"""ADP-009D/day 28: a worker bootstrap must never adopt unrelated live resources."""
# Covers (for impacted-test selection):
#   scripts/remote_cpu_deployment_scope.py
#   deploy/scripts/remote-cpu-bootstrap.sh
#   deploy/scripts/deploy.sh
#   deploy/terraform/main.tf

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from scripts.remote_cpu_deployment_scope import (
    ALLOWED_ADDRESSES, APPROVED_BILLING_ACCOUNT, WORKER_PROJECT, check_plan, check_project_iam, check_state,
)


def _plan():
    resources = [{"address": address, "mode": "managed", "values": {}}
                 for address in sorted(ALLOWED_ADDRESSES)]
    by_address = {row["address"]: row for row in resources}
    # Saved from a real read-only Terraform 1.14 plan of the reviewed scope.
    # These are configuration expressions and computed-identity placeholders.
    fixture = json.loads((Path(__file__).parent / "fixtures" / "remote_cpu_bootstrap_security_plan.json").read_text())
    for address, values in fixture["values"].items():
        if address in by_address:
            by_address[address]["values"] = values
    by_address['google_project.remote_cpu[0]']["values"] = {
        "project_id": WORKER_PROJECT, "billing_account": APPROVED_BILLING_ACCOUNT,
    }
    by_address['google_cloud_run_v2_job.remote_cpu_worker["episode-compilation"]']["values"] = {
        "name": "blueprint-remote-cpu-episode-compilation", "location": "us-central1",
        "template": [{"task_count": 1, "parallelism": 1, "template": [{
            "max_retries": 0, "timeout": "1800s", "containers": [{
                "image": "gcr.io/example/pipeline@sha256:" + "a" * 64,
                "command": ["python", "-m", "blueprint_pipeline.remote_cpu_worker", "bootstrap"],
                "args": [], "resources": [{"limits": {"cpu": "4", "memory": "16Gi"}}],
            }],
        }]}],
    }
    by_address['google_billing_budget.remote_cpu_workers[0]']["values"] = {
        "amount": [{"specified_amount": [{"units": "25", "currency_code": "USD"}]}],
    }
    return {"configuration": fixture["configuration"],
            "planned_values": {"root_module": {"resources": resources}},
            "resource_changes": [{"address": row["address"], "mode": "managed",
                                  "change": {"actions": ["create"]}} for row in resources]}


def test_complete_worker_bootstrap_plan_is_allowed():
    assert check_plan(_plan()) == []


@pytest.mark.parametrize("field,value", [("role", "roles/artifactregistry.admin"),
                                        ("repository", "other-repo"), ("location", "europe-west1")])
def test_old_repository_binding_cannot_expand_privileges(field, value):
    plan = _plan()
    row = next(row for row in plan["planned_values"]["root_module"]["resources"]
               if row["address"].startswith("google_artifact_registry_repository_iam_member."))
    row["values"][field] = value
    assert "old_project_repository_grant_drift" in check_plan(plan)


def test_computed_project_identity_cannot_hide_an_unreviewed_principal():
    plan = _plan()
    row = next(row for row in plan["configuration"]["root_module"]["resources"]
               if row["address"] == "data.google_iam_policy.remote_cpu_project")
    row["expressions"]["binding"].append({"role": {"constant_value": "roles/editor"},
                                         "members": {"constant_value": ["user:other@example.com"]}})
    assert "project_policy_configuration_drift" in check_plan(plan)


def test_managed_service_agent_cannot_enter_dispatcher_deny_exceptions():
    plan = _plan()
    row = next(row for row in plan["planned_values"]["root_module"]["resources"]
               if row["address"] == 'google_iam_deny_policy.remote_cpu_isolation[0]')
    row["values"]["rules"][0]["deny_rule"][0]["exception_principals"].append("service-agent")
    assert "dispatcher_deny_drift" in check_plan(plan)


def test_conflicting_scope_flags_fail_before_provider_commands(tmp_path):
    marker = tmp_path / "provider-called"
    gcloud = tmp_path / "gcloud"
    gcloud.write_text('#!/bin/sh\ntouch "$PROVIDER_MARKER"\nexit 99\n')
    gcloud.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["bash", str(root / "deploy/scripts/deploy.sh"),
                             "--remote-cpu-only", "--rollback"],
                            env=dict(os.environ, PATH=str(tmp_path)+os.pathsep+os.environ["PATH"],
                                     PROVIDER_MARKER=str(marker)), capture_output=True, text=True)
    assert result.returncode == 2
    assert "cannot be combined" in result.stdout
    assert not marker.exists()


@pytest.mark.parametrize("address", ["google_service_account.pipeline_runner",
                                    "google_project_iam_member.trigger_storage_admin",
                                    'google_cloud_run_v2_job.remote_cpu_worker["unreviewed-stage"]'])
def test_target_dependencies_cannot_expand_the_scope(address):
    plan = _plan()
    plan["resource_changes"].append({"address": address, "mode": "managed",
                                     "change": {"actions": ["create"]}})
    assert "unexpected_change:" + address in check_plan(plan)


def test_bootstrap_refuses_any_existing_unrelated_state():
    state = {"values": {"root_module": {"resources": [{
        "address": "google_cloud_run_v2_job.pipeline", "mode": "managed",
    }]}}}
    assert check_state(state) == ["unrelated_state:google_cloud_run_v2_job.pipeline"]


def test_old_project_mutation_and_inherited_parent_are_refused():
    plan = _plan()
    project = next(row for row in plan["planned_values"]["root_module"]["resources"]
                   if row["address"] == 'google_project.remote_cpu[0]')
    project["values"]["org_id"] = "123"
    assert "project_isolation_drift" in check_plan(plan)
    project["values"] = {"project_id": WORKER_PROJECT, "billing_account": APPROVED_BILLING_ACCOUNT}
    job = next(row for row in plan["planned_values"]["root_module"]["resources"]
               if row["address"].startswith("google_cloud_run_v2_job.remote_cpu_worker["))
    job["values"]["project"] = "blueprint-8c1ca"
    assert "wrong_project:" + job["address"] in check_plan(plan)


def test_project_policy_rejects_editors_and_wrong_managed_service_agent():
    policy = {"bindings": [
        {"role": "roles/owner", "members": ["user:ohstnhunt@gmail.com"]},
        {"role": "roles/run.serviceAgent", "members": [
            "serviceAccount:service-123@serverless-robot-prod.iam.gserviceaccount.com"]},
    ]}
    assert check_project_iam(policy, "123") == []
    assert check_project_iam(policy, "456") == ["unexpected_project_principal"]
    policy["bindings"].append({"role": "roles/editor", "members": ["user:other@example.com"]})
    assert check_project_iam(policy, "123") == ["unexpected_project_principal"]


def test_full_deployment_refuses_a_scoped_state_without_legacy_adoption():
    state = {"values": {"outputs": {"remote_cpu_bootstrap_scope": {"value": {
        "schema_version": "remote_cpu_bootstrap_scope.v1", "scope": "remote_cpu",
    }}}}}
    assert check_state(state, full=True) == ["legacy_topology_adoption_required"]


def test_scoped_refresh_cannot_claim_missing_fences_or_allow_destroy():
    plan = _plan()
    plan["planned_values"]["root_module"]["resources"] = [
        row for row in plan["planned_values"]["root_module"]["resources"]
        if row["address"] != 'google_iam_deny_policy.remote_cpu_isolation[0]'
    ]
    assert "missing_resource:google_iam_deny_policy.remote_cpu_isolation[0]" in check_plan(plan)
    plan = _plan()
    plan["resource_changes"][0]["change"]["actions"] = ["delete", "create"]
    assert any(reason.startswith("destructive_change:") for reason in check_plan(plan))


@pytest.mark.parametrize("field,value", [("max_retries", 1), ("timeout", "3600s")])
def test_plan_refuses_paid_compute_limit_drift(field, value):
    plan = deepcopy(_plan())
    job = next(row for row in plan["planned_values"]["root_module"]["resources"]
               if row["address"].startswith("google_cloud_run_v2_job.remote_cpu_worker["))
    job["values"]["template"][0]["template"][0][field] = value
    assert "job_definition_drift" in check_plan(plan)
