"""ADP-009D/day 28: bound Terraform's worker-only dependency graph before apply."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path

ALLOWED_ADDRESSES = frozenset({
    'google_project.remote_cpu[0]',
    'google_project.remote_cpu_dispatch[0]',
    'google_project_iam_policy.remote_cpu_dispatch[0]',
    'google_storage_bucket.remote_cpu_dispatch_transport[0]',
    'google_storage_bucket_iam_policy.remote_cpu_dispatch_transport[0]',
    'google_project_iam_custom_role.remote_cpu_dispatch_transport_reader[0]',
    'google_project_iam_custom_role.remote_cpu_dispatch_transport_writer[0]',
    'google_project_service_identity.remote_cpu_run[0]',
    'google_project_iam_policy.remote_cpu[0]',
    'google_artifact_registry_repository_iam_member.remote_cpu_image_reader[0]',
    'google_storage_bucket.remote_cpu_transport[0]',
    'google_storage_bucket_iam_policy.remote_cpu_transport[0]',
    'google_service_account.remote_cpu_worker[0]',
    'google_service_account.remote_cpu_quarantined_dispatcher[0]',
    'google_service_account.remote_cpu_host_dispatcher[0]',
    'google_project_iam_custom_role.remote_cpu_transport_reader[0]',
    'google_project_iam_custom_role.remote_cpu_transport_writer[0]',
    'google_project_iam_custom_role.remote_cpu_dispatcher[0]',
    'google_cloud_run_v2_job.remote_cpu_worker["episode-compilation"]',
    'google_cloud_run_v2_job_iam_policy.remote_cpu_dispatcher["episode-compilation"]',
    'google_monitoring_alert_policy.remote_cpu_job_failures[0]',
    'google_monitoring_notification_channel.remote_cpu_owner[0]',
    'google_billing_budget.remote_cpu_workers[0]',
    'google_tags_tag_key.remote_cpu_isolation[0]',
    'google_tags_location_tag_binding.remote_cpu_job["episode-compilation"]',
    'google_tags_location_tag_binding.remote_cpu_transport[0]',
    *{f'google_tags_tag_value.remote_cpu_isolation["{kind}"]'
      for kind in ("worker", "dispatcher", "job", "transport")},
    *{f'google_tags_tag_binding.remote_cpu_identity["{kind}"]'
      for kind in ("worker", "dispatcher")},
    *{f'google_project_service.remote_cpu_dispatch_apis["{api}.googleapis.com"]'
      for api in ("iam", "iamcredentials", "cloudresourcemanager", "storage", "billingbudgets")},
    *{f'google_project_service.remote_cpu_apis["{api}.googleapis.com"]'
      for api in ("run", "iam", "iamcredentials", "cloudresourcemanager", "storage", "monitoring", "logging", "billingbudgets")},
})
JOB_ADDRESS = 'google_cloud_run_v2_job.remote_cpu_worker["episode-compilation"]'
WORKER_PROJECT = "blueprint-remote-cpu-8c1ca"
DISPATCH_PROJECT = "blueprint-cpu-dispatch-8c1ca"
# Moved resources are allowed in pre-apply state only, never as active grants.
MIGRATION_ADDRESSES = {'google_service_account.remote_cpu_dispatcher[0]'}
SOURCE_PROJECT = "blueprint-8c1ca"
APPROVED_BILLING_ACCOUNT = "01E907-62B1FB-2134FB"
RUN_AGENT_REFERENCES = [
    "google_project_service_identity.remote_cpu_run[0].email",
    "google_project_service_identity.remote_cpu_run[0]",
    "google_project_service_identity.remote_cpu_run",
]
DISPATCH_PERMISSIONS = {"run.executions.cancel", "run.executions.get", "run.executions.list",
                        "run.jobs.get", "run.jobs.run", "run.jobs.runWithOverrides"}


def _refs(resource: str, attribute: str) -> dict:
    return {"references": [resource + "[0]." + attribute, resource + "[0]", resource]}


def _binding(role: str | dict, members: list[str] | dict) -> dict:
    return {"role": {"constant_value": role} if isinstance(role, str) else role,
            "members": {"constant_value": members} if isinstance(members, list) else members}


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
    dispatch_number = resources.get('google_project.remote_cpu_dispatch[0]', {}).get("number")
    if project_number and dispatch_number:
        try:
            projects = resources['google_billing_budget.remote_cpu_workers[0]']["budget_filter"][0]["projects"]
            if set(projects) != {f"projects/{project_number}", f"projects/{dispatch_number}"}:
                blockers.append("budget_project_drift")
        except (KeyError, IndexError, TypeError):
            blockers.append("budget_project_unproven")
    budget_expression = config.get("google_billing_budget.remote_cpu_workers", {}).get("budget_filter", [])
    expected_budget_refs = _refs("google_project.remote_cpu", "number")["references"] + _refs("google_project.remote_cpu_dispatch", "number")["references"]
    if (len(budget_expression) != 1 or budget_expression[0].get("projects") != {"references": expected_budget_refs}
            or "labels" in budget_expression[0]):
        blockers.append("budget_scope_configuration_drift")
    founder = _binding("roles/owner", ["user:ohstnhunt@gmail.com"])
    worker_binding = _binding(_refs("google_project_iam_custom_role.remote_cpu_dispatch_transport_reader", "name"),
                              _refs("google_service_account.remote_cpu_worker", "email"))
    dispatcher_binding = _binding(_refs("google_project_iam_custom_role.remote_cpu_dispatch_transport_writer", "name"),
                                  _refs("google_service_account.remote_cpu_host_dispatcher", "email"))
    policies = {
        "remote_cpu_project": [founder, _binding("roles/run.serviceAgent", {"references": RUN_AGENT_REFERENCES})],
        "remote_cpu_dispatch_project": [founder],
        "remote_cpu_dispatch_transport": [worker_binding, dispatcher_binding,
            _binding("roles/storage.legacyBucketOwner", {"references": ["var.remote_cpu_dispatch_project_id"]})],
        "remote_cpu_transport": [_binding("roles/storage.legacyBucketOwner", {"references": ["var.remote_cpu_project_id"]})],
        "remote_cpu_dispatch_job": [_binding(_refs("google_project_iam_custom_role.remote_cpu_dispatcher", "name"),
                                             _refs("google_service_account.remote_cpu_host_dispatcher", "email"))],
    }
    for name, bindings in policies.items():
        if config.get("data.google_iam_policy." + name, {}).get("binding") != bindings:
            blockers.append(name + "_configuration_drift")
    for address, source in (
        ("google_project_iam_policy.remote_cpu", "remote_cpu_project"),
        ("google_project_iam_policy.remote_cpu_dispatch", "remote_cpu_dispatch_project"),
        ("google_storage_bucket_iam_policy.remote_cpu_transport", "remote_cpu_transport"),
        ("google_storage_bucket_iam_policy.remote_cpu_dispatch_transport", "remote_cpu_dispatch_transport"),
        ("google_cloud_run_v2_job_iam_policy.remote_cpu_dispatcher", "remote_cpu_dispatch_job"),
    ):
        if config.get(address, {}).get("policy_data") != _refs("data.google_iam_policy." + source, "policy_data"):
            blockers.append(address + "_policy_source_drift")
    for name, number, dispatch in (("remote_cpu", project_number, False),
                                  ("remote_cpu_dispatch", dispatch_number, True)):
        policy_data = resources.get(f'google_project_iam_policy.{name}[0]', {}).get("policy_data")
        if policy_data and number:
            blockers.extend(check_project_iam(json.loads(policy_data), str(number), dispatch=dispatch))
    for name, permissions in (("remote_cpu_dispatcher", DISPATCH_PERMISSIONS),
                              ("remote_cpu_transport_reader", {"storage.objects.get"}),
                              ("remote_cpu_transport_writer", {"storage.objects.create", "storage.objects.get", "storage.objects.delete"}),
                              ("remote_cpu_dispatch_transport_reader", {"storage.objects.get"}),
                              ("remote_cpu_dispatch_transport_writer", {"storage.objects.create", "storage.objects.get", "storage.objects.delete"})):
        if set(resources.get(f'google_project_iam_custom_role.{name}[0]', {}).get("permissions", [])) != permissions:
            blockers.append("custom_role_drift:" + name)
    expected_dispatcher = f"remote-cpu-dispatcher@{DISPATCH_PROJECT}.iam.gserviceaccount.com"
    for name, expected_project in (("remote_cpu_worker", WORKER_PROJECT),
                                  ("remote_cpu_quarantined_dispatcher", WORKER_PROJECT),
                                  ("remote_cpu_host_dispatcher", DISPATCH_PROJECT)):
        values = resources.get(f'google_service_account.{name}[0]', {})
        expected_id = "remote-cpu-worker" if name == "remote_cpu_worker" else "remote-cpu-dispatcher"
        if values.get("account_id") != expected_id:
            blockers.append("service_account_drift:" + name)
        if values.get("email") and values["email"] != f"{expected_id}@{expected_project}.iam.gserviceaccount.com":
            blockers.append("service_account_identity_drift:" + name)
    # Fully resolved IAM policy data must agree with the reviewed references.
    known_policies = {
        'google_storage_bucket_iam_policy.remote_cpu_dispatch_transport[0]': {
            f"projects/{DISPATCH_PROJECT}/roles/remoteCpuTransportReader": {f"serviceAccount:remote-cpu-worker@{WORKER_PROJECT}.iam.gserviceaccount.com"},
            f"projects/{DISPATCH_PROJECT}/roles/remoteCpuTransportWriter": {"serviceAccount:" + expected_dispatcher},
            "roles/storage.legacyBucketOwner": {"projectOwner:" + DISPATCH_PROJECT}},
        'google_storage_bucket_iam_policy.remote_cpu_transport[0]': {
            "roles/storage.legacyBucketOwner": {"projectOwner:" + WORKER_PROJECT}},
        'google_cloud_run_v2_job_iam_policy.remote_cpu_dispatcher["episode-compilation"]': {
            f"projects/{WORKER_PROJECT}/roles/remoteCpuDispatcher": {"serviceAccount:" + expected_dispatcher}},
    }
    for address, expected in known_policies.items():
        text = resources.get(address, {}).get("policy_data")
        if text and _policy_members(json.loads(text)) != expected:
            blockers.append("resource_policy_drift:" + address)
    return blockers


def _policy_members(policy: Mapping) -> dict | None:
    actual = {}
    for row in policy.get("bindings", []):
        if row.get("condition") or row["role"] in actual:
            return None
        actual[row["role"]] = {member.lower() if member.startswith("user:") else member
                               for member in row.get("members", [])}
    return actual


def check_project_iam(policy: Mapping, project_number: str, *, dispatch: bool = False) -> list[str]:
    expected = {"roles/owner": {"user:ohstnhunt@gmail.com"}}
    if not dispatch:
        expected["roles/run.serviceAgent"] = {
            f"serviceAccount:service-{project_number}@serverless-robot-prod.iam.gserviceaccount.com"}
    return [] if _policy_members(policy) == expected else ["unexpected_project_principal"]


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
                  if row.get("mode") == "managed" and row["address"] not in ALLOWED_ADDRESSES | MIGRATION_ADDRESSES)


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
            expected = (SOURCE_PROJECT if address.startswith("google_artifact_registry_repository_iam_member.")
                        else DISPATCH_PROJECT if ("remote_cpu_dispatch" in address and "remote_cpu_dispatcher" not in address)
                            or "remote_cpu_host_dispatcher" in address else WORKER_PROJECT)
            if values["project"] != expected:
                blockers.append("wrong_project:" + address)
    for name, expected_id in (("remote_cpu", WORKER_PROJECT), ("remote_cpu_dispatch", DISPATCH_PROJECT)):
        project = resources.get(f'google_project.{name}[0]', {})
        if (project.get("project_id") != expected_id or project.get("org_id") or project.get("folder_id")
                or project.get("billing_account") != APPROVED_BILLING_ACCOUNT):
            blockers.append("project_isolation_drift:" + name)
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
    parser.add_argument("--dispatch-project", action="store_true")
    args = parser.parse_args()
    if args.action == "targets":
        print("\n".join("-target=" + address for address in sorted(ALLOWED_ADDRESSES | {"google_service_account.remote_cpu_dispatcher"})))
        return 0
    if args.path is None:
        parser.error("A saved Terraform JSON document is required.")
    document = json.loads(args.path.read_text())
    if args.action == "project-iam":
        if not args.project_number or not args.project_number.isdigit():
            parser.error("The observed project number is required for IAM validation.")
        blockers = check_project_iam(document, args.project_number, dispatch=args.dispatch_project)
    else:
        blockers = (check_plan(document) if args.action == "plan"
                    else check_state(document, full=args.action == "full-state"))
    print(json.dumps({"scope": "remote_cpu", "blockers": blockers}, sort_keys=True))
    return 2 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
