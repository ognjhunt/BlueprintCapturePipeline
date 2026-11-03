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
    ALLOWED_ADDRESSES, APPROVED_BILLING_ACCOUNT, DISPATCH_PROJECT, OPTIONAL_ADDRESSES, WORKER_PROJECT, check_plan,
    check_project_iam, check_state, stage_addresses,
)

TEAM = "team-scene-packet"


def _team_job(fixture: dict, address: str) -> dict:
    """The team scene packet job's planned values, derived from the saved episode job's by renaming the stage:
    the same module and image pipeline, its own name and its 3600 s bound.  Not a saved plan of it; the owner's
    first scoped plan with the stage in tfvars is that."""

    episode = address.replace(f'["{TEAM}"]', '["episode-compilation"]')
    return json.loads(json.dumps(fixture["values"][episode]).replace("episode-compilation", TEAM))


def _plan(*, team: bool = False):
    addresses = ALLOWED_ADDRESSES | (stage_addresses(TEAM) if team else frozenset())
    resources = [{"address": address, "mode": "managed", "values": {}} for address in sorted(addresses)]
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
    if team:
        for address in stage_addresses(TEAM):
            by_address[address]["values"] = _team_job(fixture, address)
        job = deepcopy(by_address['google_cloud_run_v2_job.remote_cpu_worker["episode-compilation"]']["values"])
        job["name"] = f"blueprint-remote-cpu-{TEAM}"
        job["template"][0]["template"][0]["timeout"] = "3600s"
        by_address[f'google_cloud_run_v2_job.remote_cpu_worker["{TEAM}"]']["values"] = job
    return {"configuration": fixture["configuration"],
            "planned_values": {"root_module": {"resources": resources}},
            "resource_changes": [{"address": row["address"], "mode": "managed",
                                  "change": {"actions": ["create"]}} for row in resources]}


def test_complete_worker_bootstrap_plan_is_allowed():
    assert check_plan(_plan()) == []


@pytest.mark.parametrize("field,value", [("role", "roles/artifactregistry.admin"),
                                        ("repository", "other-repo"), ("location", "europe-west1"),
                                        ("repository", "projects/other/locations/us/repositories/gcr.io"),
                                        ("repository", "projects/blueprint-8c1ca/locations/eu/repositories/gcr.io")])
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
    assert "remote_cpu_project_configuration_drift" in check_plan(plan)


@pytest.mark.parametrize("policy_name", ["remote_cpu_dispatch_project", "remote_cpu_dispatch_transport", "remote_cpu_dispatch_job"])
def test_managed_service_agent_cannot_enter_dispatch_authority(policy_name):
    plan = _plan()
    row = next(row for row in plan["configuration"]["root_module"]["resources"]
               if row["address"] == "data.google_iam_policy." + policy_name)
    row["expressions"]["binding"].append({"role": {"constant_value": "roles/editor"},
                                         "members": {"constant_value": ["serviceAccount:runtime-agent"]}})
    assert policy_name + "_configuration_drift" in check_plan(plan)


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


@pytest.mark.parametrize("revision,accepted", [("a" * 40, True), ("b" * 40, False), (None, False)])
def test_worker_image_revision_must_match_promoted_source(tmp_path, revision, accepted):
    config = {"config": {"Labels": {}}}
    if revision is not None:
        config["config"]["Labels"]["org.opencontainers.image.revision"] = revision
    crane = tmp_path / "crane"
    crane.write_text("#!/bin/sh\ncat <<'IMAGE_CONFIG'\n" + json.dumps(config) + "\nIMAGE_CONFIG\n")
    crane.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["bash", "-c", 'source "$1"; verify_remote_cpu_image_revision "$2" "$3"',
                             "guard-test", str(root / "deploy/scripts/remote-cpu-bootstrap.sh"),
                             "gcr.io/blueprint-8c1ca/pipeline@sha256:" + "c" * 64, "a" * 40],
                            env=dict(os.environ, PATH=str(tmp_path)+os.pathsep+os.environ["PATH"]),
                            capture_output=True, text=True)
    assert result.returncode == (0 if accepted else 2)


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
    assert "project_isolation_drift:remote_cpu" in check_plan(plan)
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


def test_founder_email_casing_is_equivalent_and_dispatch_project_is_owner_only():
    policy = {"bindings": [
        {"role": "roles/owner", "members": ["user:OHSTNhunt@gmail.com"]},
    ]}
    assert check_project_iam(policy, "123", dispatch=True) == []
    assert check_project_iam(policy, "123") == ["unexpected_project_principal"]
    policy["bindings"].append({"role": "roles/run.serviceAgent", "members": [
        "serviceAccount:service-123@serverless-robot-prod.iam.gserviceaccount.com"]})
    assert check_project_iam(policy, "123") == []
    assert check_project_iam(policy, "123", dispatch=True) == ["unexpected_project_principal"]


def test_moved_keyless_dispatcher_is_allowed_in_preapply_state_but_not_active_plan():
    address = 'google_service_account.remote_cpu_dispatcher[0]'
    state = {"values": {"root_module": {"resources": [{"address": address, "mode": "managed"}]}}}
    assert check_state(state) == []
    plan = _plan()
    plan["resource_changes"].append({"address": address, "mode": "managed", "change": {"actions": ["create"]}})
    assert "unexpected_change:" + address in check_plan(plan)


@pytest.mark.parametrize("name", ["remote_cpu_dispatcher", "remote_cpu_dispatch_transport_reader", "remote_cpu_dispatch_transport_writer"])
def test_custom_role_cannot_acquire_identity_mutation_or_new_work(name):
    plan = _plan()
    row = next(row for row in plan["planned_values"]["root_module"]["resources"]
               if row["address"] == f'google_project_iam_custom_role.{name}[0]')
    row["values"]["permissions"].append("iam.serviceAccounts.getAccessToken")
    assert "custom_role_drift:" + name in check_plan(plan)


@pytest.mark.parametrize("project,folder", [("other-project", None), (DISPATCH_PROJECT, "123")])
def test_dispatch_project_cannot_move_to_an_inherited_or_shared_identity(project, folder):
    plan = _plan()
    row = next(row for row in plan["planned_values"]["root_module"]["resources"]
               if row["address"] == 'google_project.remote_cpu_dispatch[0]')
    row["values"] = {"project_id": project, "billing_account": APPROVED_BILLING_ACCOUNT, "folder_id": folder}
    assert "project_isolation_drift:remote_cpu_dispatch" in check_plan(plan)


def test_full_deployment_refuses_a_scoped_state_without_legacy_adoption():
    state = {"values": {"outputs": {"remote_cpu_bootstrap_scope": {"value": {
        "schema_version": "remote_cpu_bootstrap_scope.v1", "scope": "remote_cpu",
    }}}}}
    assert check_state(state, full=True) == ["legacy_topology_adoption_required"]


def test_scoped_refresh_cannot_claim_missing_fences_or_allow_destroy():
    plan = _plan()
    plan["planned_values"]["root_module"]["resources"] = [
        row for row in plan["planned_values"]["root_module"]["resources"]
        if row["address"] != 'google_project_iam_policy.remote_cpu_dispatch[0]'
    ]
    assert "missing_resource:google_project_iam_policy.remote_cpu_dispatch[0]" in check_plan(plan)
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


def test_the_team_scene_packet_job_is_optional_and_validated_whole_when_planned():
    """Plan 14's second stage: absent, the reviewed scope is unchanged; present, its job, dispatcher IAM policy and
    tag binding come together and its job is the 4 CPU, 16Gi, 3600 s, zero-retry bootstrap job."""

    assert stage_addresses(TEAM) == OPTIONAL_ADDRESSES and not OPTIONAL_ADDRESSES & ALLOWED_ADDRESSES
    assert check_plan(_plan()) == [] and check_plan(_plan(team=True)) == []
    job = f'google_cloud_run_v2_job.remote_cpu_worker["{TEAM}"]'
    for field, value in (("timeout", "1800s"), ("max_retries", 3)):
        plan = _plan(team=True)
        row = next(row for row in plan["planned_values"]["root_module"]["resources"] if row["address"] == job)
        row["values"]["template"][0]["template"][0][field] = value
        assert check_plan(plan) == [f"job_definition_drift:{TEAM}"], field
    plan = _plan(team=True)
    row = next(row for row in plan["planned_values"]["root_module"]["resources"] if row["address"] == job)
    row["values"]["template"][0]["template"][0]["containers"][0]["resources"] = [{"limits": {"cpu": "8",
                                                                                         "memory": "32Gi"}}]
    assert check_plan(plan) == [f"job_definition_drift:{TEAM}"]
    # A job without its dispatcher policy or isolation tag is refused, as is a policy granting anyone else.
    plan = _plan(team=True)
    policy = f'google_cloud_run_v2_job_iam_policy.remote_cpu_dispatcher["{TEAM}"]'
    plan["planned_values"]["root_module"]["resources"] = [
        row for row in plan["planned_values"]["root_module"]["resources"] if row["address"] != policy]
    assert "missing_resource:" + policy in check_plan(plan)
    plan = _plan(team=True)
    row = next(row for row in plan["planned_values"]["root_module"]["resources"] if row["address"] == policy)
    row["values"]["policy_data"] = json.dumps({"bindings": [{
        "role": f"projects/{WORKER_PROJECT}/roles/remoteCpuDispatcher", "members": ["user:other@example.com"]}]})
    assert "resource_policy_drift:" + policy in check_plan(plan)
    # The episode job keeps its own 1800 s bound: 3600 s there is still drift.
    plan = _plan(team=True)
    row = next(row for row in plan["planned_values"]["root_module"]["resources"]
               if row["address"] == 'google_cloud_run_v2_job.remote_cpu_worker["episode-compilation"]')
    row["values"]["template"][0]["template"][0]["timeout"] = "3600s"
    assert check_plan(plan) == ["job_definition_drift"]
    # State and targets know the optional job; nothing else.
    state = {"values": {"root_module": {"resources": [{"address": address, "mode": "managed"}
                                                      for address in sorted(stage_addresses(TEAM))]}}}
    assert check_state(state) == []
    targets = subprocess.run(["python3", "scripts/remote_cpu_deployment_scope.py", "targets"], check=True,
                             capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1]).stdout.split()
    assert {"-target=" + address for address in stage_addresses(TEAM)} <= set(targets)
