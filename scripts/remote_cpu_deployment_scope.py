"""ADP-009D/day 28: bound Terraform's worker-only dependency graph before apply."""

from __future__ import annotations

import argparse
import json
import re
from urllib.parse import unquote
from collections.abc import Mapping
from pathlib import Path

ALLOWED_ADDRESSES = frozenset({
    'google_project.remote_cpu[0]',
    'google_project_service_identity.remote_cpu_run[0]',
    'google_project_iam_policy.remote_cpu[0]',
    'google_artifact_registry_repository_iam_member.remote_cpu_image_reader[0]',
    'google_storage_bucket.remote_cpu_transport[0]',
    'google_storage_bucket_iam_policy.remote_cpu_transport[0]',
    'google_service_account.remote_cpu_worker[0]',
    'google_service_account.remote_cpu_dispatcher[0]',
    'google_project_iam_custom_role.remote_cpu_transport_reader[0]',
    'google_project_iam_custom_role.remote_cpu_transport_writer[0]',
    'google_project_iam_custom_role.remote_cpu_dispatcher[0]',
    'google_cloud_run_v2_job.remote_cpu_worker["episode-compilation"]',
    'google_cloud_run_v2_job_iam_member.remote_cpu_dispatcher["episode-compilation"]',
    'google_monitoring_alert_policy.remote_cpu_job_failures[0]',
    'google_monitoring_notification_channel.remote_cpu_owner[0]',
    'google_billing_budget.remote_cpu_workers[0]',
    'google_tags_tag_key.remote_cpu_isolation[0]',
    'google_iam_deny_policy.remote_cpu_isolation[0]',
    'google_tags_location_tag_binding.remote_cpu_job["episode-compilation"]',
    'google_tags_location_tag_binding.remote_cpu_transport[0]',
    *{f'google_tags_tag_value.remote_cpu_isolation["{kind}"]'
      for kind in ("worker", "dispatcher", "job", "transport")},
    *{f'google_tags_tag_binding.remote_cpu_identity["{kind}"]'
      for kind in ("worker", "dispatcher")},
    *{f'google_project_service.remote_cpu_apis["{api}.googleapis.com"]'
      for api in ("run", "iam", "iamcredentials", "cloudresourcemanager", "storage", "monitoring", "logging", "billingbudgets")},
})
JOB_ADDRESS = 'google_cloud_run_v2_job.remote_cpu_worker["episode-compilation"]'
WORKER_PROJECT = "blueprint-remote-cpu-8c1ca"
SOURCE_PROJECT = "blueprint-8c1ca"
APPROVED_BILLING_ACCOUNT = "01E907-62B1FB-2134FB"
RUN_AGENT_REFERENCES = [
    "google_project_service_identity.remote_cpu_run[0].email",
    "google_project_service_identity.remote_cpu_run[0]",
    "google_project_service_identity.remote_cpu_run",
]
DISPATCHER_DENIED_PERMISSIONS = {
    "iam.googleapis.com/serviceAccounts." + operation for operation in (
        "actAs", "getAccessToken", "getOpenIdToken", "signBlob", "signJwt", "implicitDelegation",
        "setIamPolicy", "delete", "disable", "enable", "update", "undelete",
    )
} | {"iam.googleapis.com/serviceAccountKeys.create"}


def _security_blockers(plan: Mapping, resources: Mapping) -> list[str]:
    """Computed IDs are allowed only through the reviewed identity references."""
    config = {row["address"]: row.get("expressions", {})
              for row in plan.get("configuration", {}).get("root_module", {}).get("resources", [])}
    blockers = []
    reader = resources.get('google_artifact_registry_repository_iam_member.remote_cpu_image_reader[0]', {})
    expressions = config.get("google_artifact_registry_repository_iam_member.remote_cpu_image_reader", {})
    # Artifact Registry reads back the fully qualified repository after apply.
    # Both forms must identify this exact existing repository in the old project.
    repositories = {"gcr.io", f"projects/{SOURCE_PROJECT}/locations/us/repositories/gcr.io"}
    if ((reader.get("location"), reader.get("role")) != ("us", "roles/artifactregistry.reader")
            or reader.get("repository") not in repositories
            or expressions.get("member") != {"references": RUN_AGENT_REFERENCES}):
        blockers.append("old_project_repository_grant_drift")
    project_number = resources.get('google_project.remote_cpu[0]', {}).get("number")
    if project_number and reader.get("member") != (
        f"serviceAccount:service-{project_number}@serverless-robot-prod.iam.gserviceaccount.com"
    ):
        blockers.append("old_project_repository_principal_drift")
    if project_number:
        deny_parent = resources.get('google_iam_deny_policy.remote_cpu_isolation[0]', {}).get("parent", "")
        if unquote(deny_parent) != f"cloudresourcemanager.googleapis.com/projects/{project_number}":
            blockers.append("deny_project_drift")
        try:
            projects = resources['google_billing_budget.remote_cpu_workers[0]']["budget_filter"][0]["projects"]
            if projects != [f"projects/{project_number}"]:
                blockers.append("budget_project_drift")
        except (KeyError, IndexError, TypeError):
            blockers.append("budget_project_unproven")
    policy = config.get("data.google_iam_policy.remote_cpu_project", {})
    if policy.get("binding") != [
        {"role": {"constant_value": "roles/owner"},
         "members": {"constant_value": ["user:ohstnhunt@gmail.com"]}},
        {"role": {"constant_value": "roles/iam.denyAdmin"},
         "members": {"constant_value": ["user:ohstnhunt@gmail.com"]}},
        {"role": {"constant_value": "roles/run.serviceAgent"},
         "members": {"references": RUN_AGENT_REFERENCES}},
    ]:
        blockers.append("project_policy_configuration_drift")
    policy_resource = config.get("google_project_iam_policy.remote_cpu", {})
    if policy_resource.get("policy_data") != {"references": [
        "data.google_iam_policy.remote_cpu_project[0].policy_data",
        "data.google_iam_policy.remote_cpu_project[0]", "data.google_iam_policy.remote_cpu_project",
    ]}:
        blockers.append("project_policy_source_drift")
    policy_data = resources.get('google_project_iam_policy.remote_cpu[0]', {}).get("policy_data")
    if policy_data and project_number:
        blockers.extend(check_project_iam(json.loads(policy_data), str(project_number)))
    try:
        deny = resources['google_iam_deny_policy.remote_cpu_isolation[0]']["rules"][0]["deny_rule"][0]
        if (deny["denied_principals"] != ["principalSet://goog/public:all"]
                or deny["exception_principals"] != ["principal://goog/subject/ohstnhunt@gmail.com"]
                or set(deny["denied_permissions"]) != DISPATCHER_DENIED_PERMISSIONS):
            blockers.append("dispatcher_deny_drift")
        refs = config["google_iam_deny_policy.remote_cpu_isolation"]["rules"][0]["deny_rule"][0]["denial_condition"][0]["expression"]
        if refs != {"references": [
            "google_tags_tag_key.remote_cpu_isolation[0].id",
            "google_tags_tag_key.remote_cpu_isolation[0]", "google_tags_tag_key.remote_cpu_isolation",
            'google_tags_tag_value.remote_cpu_isolation["dispatcher"].id',
            'google_tags_tag_value.remote_cpu_isolation["dispatcher"]',
            "google_tags_tag_value.remote_cpu_isolation",
        ]}:
            blockers.append("dispatcher_deny_condition_drift")
    except (KeyError, IndexError, TypeError):
        blockers.append("dispatcher_deny_unproven")
    return blockers


def check_project_iam(policy: Mapping, project_number: str) -> list[str]:
    expected = {
        "roles/owner": {"user:ohstnhunt@gmail.com"},
        "roles/iam.denyAdmin": {"user:ohstnhunt@gmail.com"},
        "roles/run.serviceAgent": {
            f"serviceAccount:service-{project_number}@serverless-robot-prod.iam.gserviceaccount.com"},
    }
    actual = {}
    for row in policy.get("bindings", []):
        if row.get("condition") or row["role"] in actual:
            return ["unexpected_project_principal"]
        # Resource Manager returns the account's original display casing.
        # Google user email addresses denote the same identity in either case;
        # keep service-account and other principal spellings exact.
        actual[row["role"]] = {member.lower() if member.startswith("user:") else member
                               for member in row.get("members", [])}
    return [] if actual == expected else ["unexpected_project_principal"]


def _resources(module: Mapping):
    yield from module.get("resources", [])
    for child in module.get("child_modules", []):
        yield from _resources(child)


def check_state(state: Mapping, *, full: bool = False) -> list[str]:
    values = state.get("values", {})
    marker = values.get("outputs", {}).get("remote_cpu_bootstrap_scope", {}).get("value")
    if full:
        return ["legacy_topology_adoption_required"] if marker else []
    return sorted("unrelated_state:" + row["address"]
                  for row in _resources(values.get("root_module", {}))
                  if row.get("mode") == "managed" and row["address"] not in ALLOWED_ADDRESSES)


def check_plan(plan: Mapping) -> list[str]:
    blockers = []
    for row in plan.get("resource_changes", []):
        if row.get("mode") != "managed":
            continue
        address = row["address"]
        if address not in ALLOWED_ADDRESSES:
            blockers.append("unexpected_change:" + address)
        if "delete" in row.get("change", {}).get("actions", []):
            blockers.append("destructive_change:" + address)
    resources = {row["address"]: row.get("values", {})
                 for row in _resources(plan.get("planned_values", {}).get("root_module", {}))
                 if row.get("mode") == "managed"}
    blockers += ["missing_resource:" + address for address in sorted(ALLOWED_ADDRESSES - resources.keys())]
    for address, values in resources.items():
        if address not in ALLOWED_ADDRESSES:
            blockers.append("unexpected_resource:" + address)
        if "project" in values:
            expected = SOURCE_PROJECT if address.startswith("google_artifact_registry_repository_iam_member.") else WORKER_PROJECT
            if values["project"] != expected:
                blockers.append("wrong_project:" + address)
    project = resources.get('google_project.remote_cpu[0]', {})
    if (project.get("project_id") != WORKER_PROJECT or project.get("org_id") or project.get("folder_id")
            or project.get("billing_account") != APPROVED_BILLING_ACCOUNT):
        blockers.append("project_isolation_drift")
    blockers.extend(_security_blockers(plan, resources))
    job = resources.get(JOB_ADDRESS, {})
    try:
        outer = job["template"][0]
        inner = outer["template"][0]
        container = inner["containers"][0]
        valid = (
            outer["task_count"] == outer["parallelism"] == 1
            and inner["max_retries"] == 0 and inner["timeout"] == "1800s"
            and container["command"] == ["python", "-m", "blueprint_pipeline.remote_cpu_worker", "bootstrap"]
            and not container.get("args")
            and container["resources"][0]["limits"] == {"cpu": "4", "memory": "16Gi"}
            and bool(re.fullmatch(r".+@sha256:[0-9a-f]{64}", container["image"]))
            and job["location"].startswith("us-")
        )
    except (KeyError, IndexError, TypeError):
        valid = False
    if not valid:
        blockers.append("job_definition_drift")
    try:
        amount = resources['google_billing_budget.remote_cpu_workers[0]']["amount"][0]["specified_amount"][0]
        if str(amount["units"]) != "25" or amount["currency_code"] != "USD":
            blockers.append("spend_limit_drift")
    except (KeyError, IndexError, TypeError):
        blockers.append("spend_limit_drift")
    return blockers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("targets", "state", "full-state", "plan", "project-iam"))
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument("--project-number")
    args = parser.parse_args()
    if args.action == "targets":
        print("\n".join("-target=" + address for address in sorted(ALLOWED_ADDRESSES)))
        return 0
    if args.path is None:
        parser.error("A saved Terraform JSON document is required.")
    document = json.loads(args.path.read_text())
    if args.action == "project-iam":
        if not args.project_number or not args.project_number.isdigit():
            parser.error("The observed project number is required for IAM validation.")
        blockers = check_project_iam(document, args.project_number)
    else:
        blockers = (check_plan(document) if args.action == "plan"
                    else check_state(document, full=args.action == "full-state"))
    print(json.dumps({"scope": "remote_cpu", "blockers": blockers}, sort_keys=True))
    return 2 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
