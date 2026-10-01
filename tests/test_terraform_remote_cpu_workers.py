# Covers (for impacted-test selection):
#   deploy/terraform/main.tf
#   deploy/terraform/terraform.tfvars.example
#   deploy/scripts/deploy.sh
#   .github/workflows/terraform-static.yml
"""ADP-009D/day-28, plan 14 PR 5: the remote CPU worker infrastructure is off by default and fenced.

Terraform is read as text, as the deploy contract tests read it: CI has no terraform binary and no
GCP credentials here, and nothing in this file plans or applies.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from blueprint_pipeline import cloud_run_jobs_client
from blueprint_pipeline import remote_cpu_job_contract as contract
from blueprint_pipeline.cloud_run_jobs_client import BOOTSTRAP_COMMAND, job_definition_blockers
from blueprint_pipeline.remote_cpu_worker import PREFIX_VARIABLE, STAGE_VARIABLE, _object_prefix
from tests.test_deploy_systemd_contract import _terraform_resource_body, _terraform_variable_body

REPO_ROOT = Path(__file__).resolve().parents[1]
TERRAFORM_MAIN = REPO_ROOT / "deploy" / "terraform" / "main.tf"
TFVARS_EXAMPLE = REPO_ROOT / "deploy" / "terraform" / "terraform.tfvars.example"
DEPLOY_SCRIPT = REPO_ROOT / "deploy" / "scripts" / "deploy.sh"
TERRAFORM_STATIC_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "terraform-static.yml"
GIB = 1024**3
IMAGE = "gcr.io/blueprint-8c1ca/blueprint-pipeline@sha256:" + "d" * 64
# Plan 14 §14: run with overrides and read or cancel executions; nothing that operates on the project.
# jobs.run is authorized on run.jobs.run, with runWithOverrides checked on top when overrides are sent.
DISPATCHER_PERMISSIONS = [
    "run.executions.cancel",
    "run.executions.get",
    "run.executions.list",
    "run.jobs.get",
    "run.jobs.run",
    "run.jobs.runWithOverrides",
]
TRANSPORT_WRITER_PERMISSIONS = ["storage.objects.create", "storage.objects.delete", "storage.objects.get"]
IMAGE_VARIABLES = {
    "docker_image",
    "privacy_sam3_image",
    "privacy_vip_image",
    "privacy_deepprivacy2_image",
    "video_to_world_image",
}


def _main() -> str:
    return TERRAFORM_MAIN.read_text(encoding="utf-8")


def _braced(text: str, opening: int) -> str:
    """The body of the block whose ``{`` is at ``opening``."""
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index]
    raise AssertionError("unclosed Terraform block")


def _top_level(body: str, position: int) -> bool:
    prefix = body[:position]
    return prefix.count("{") == prefix.count("}")


def _children(body: str, header: str) -> list[str]:
    """Bodies of the blocks ``header { ... }`` directly inside ``body``, such as ``env`` or
    ``dynamic "condition"``, or of a map attribute written ``name = {``."""
    words = r"[ \t]*".join(re.escape(word) for word in header.split())
    pattern = re.compile(rf"(?m)^[ \t]*{words}[ \t]*\{{[ \t]*$")
    return [_braced(body, match.end() - 1) for match in pattern.finditer(body)
            if _top_level(body, match.start())]


def _child(body: str, header: str) -> str:
    children = _children(body, header)
    assert len(children) == 1, (header, len(children))
    return children[0]


def _attr(body: str, name: str) -> str | None:
    """The single-line value of the attribute ``name`` directly inside ``body``."""
    pattern = re.compile(rf"(?m)^[ \t]*{re.escape(name)}[ \t]*=[ \t]*(.+?)[ \t]*$")
    values = [match.group(1) for match in pattern.finditer(body) if _top_level(body, match.start())]
    assert len(values) <= 1, (name, values)
    return values[0] if values else None


def _strings(value: str | None) -> list[str]:
    """The decoded string literals in an HCL expression, in order."""
    return [json.loads(literal) for literal in re.findall(r'"(?:[^"\\]|\\.)*"', value or "")]


def _list(body: str, name: str) -> list[str]:
    """The strings of the list attribute ``name`` directly inside ``body``."""
    pattern = re.compile(rf"(?ms)^[ \t]*{re.escape(name)}[ \t]*=[ \t]*\[(.*?)\]")
    values = [match.group(1) for match in pattern.finditer(body) if _top_level(body, match.start())]
    assert len(values) == 1, (name, values)
    return _strings(values[0])


def _resources(text: str) -> dict[tuple[str, str], str]:
    """Every resource by ``(kind, name)``, and every data source by ``("data.<kind>", name)``."""
    blocks = {(kind, name): _terraform_resource_body(text, kind, name)
              for kind, name in re.findall(r'(?m)^resource "([a-z0-9_]+)" "([a-z0-9_]+)" \{$', text)}
    for kind, name in re.findall(r'(?m)^data "([a-z0-9_]+)" "([a-z0-9_]+)" \{$', text):
        header = f'data "{kind}" "{name}" {{'
        blocks[(f"data.{kind}", name)] = _braced(text, text.index(header) + len(header) - 1)
    return blocks


def _hcl_regex(body: str, variable: str) -> re.Pattern[str]:
    """The pattern a validation passes to ``regex(..., var.<variable>)``; RE2 and ``re`` agree on it."""
    match = re.search(rf'regex\(("(?:[^"\\]|\\.)*"), var\.{variable}\)', body)
    assert match is not None, variable
    return re.compile(json.loads(match.group(1)))


def _project_grants(main: str, role_prefix: str) -> dict[str, str]:
    """Every project-wide ``google_project_iam_member`` whose role starts with ``role_prefix``."""
    return {name: body for (kind, name), body in _resources(main).items()
            if kind == "google_project_iam_member"
            and (_strings(_attr(body, "role")) or [""])[0].startswith(role_prefix)}


def _remote_cpu_exclusion(body: str) -> str:
    """The condition a project-wide grant gains when, and only when, remote CPU workers exist."""
    condition = _child(body, 'dynamic "condition"')
    assert _attr(condition, "for_each") == "var.remote_cpu_workers_enabled ? [1] : []"
    content = _child(condition, "content")
    assert _strings(_attr(content, "title"))
    # A new condition replaces the binding: create the conditioned one before removing the old.
    assert _attr(_child(body, "lifecycle"), "create_before_destroy") == "true"
    return _strings(_attr(content, "expression"))[0]


def test_remote_cpu_jobs_are_off_by_default_and_us_only() -> None:
    main = _main()
    flag = _terraform_variable_body(main, "remote_cpu_workers_enabled")
    assert (_attr(flag, "type"), _attr(flag, "default"), _attr(flag, "nullable")) == (
        "bool", "false", "false")
    # deploy.sh exports the flag and the prefix, and a terraform.tfvars value overrides an export:
    # the example shows them commented out, so a copy of it cannot pin either.
    example = TFVARS_EXAMPLE.read_text(encoding="utf-8")
    for name in ("remote_cpu_workers_enabled", "remote_cpu_worker_object_prefix"):
        assert not re.search(rf"(?m)^{name}\s*=", example), name
        assert re.search(rf"(?m)^# {name}\s*=", example), name

    # Every remote CPU resource exists only while the flag is on: off, its count is 0 or its map
    # empty. Anything with a location sits in the primary region, which must be a US region.
    remote = {address: body for address, body in _resources(main).items()
              if address[1].startswith("remote_cpu")}
    assert ("google_cloud_run_v2_job", "remote_cpu_worker") in remote
    for address, body in sorted(remote.items()):
        gate = _attr(body, "count") or _attr(body, "for_each") or ""
        assert gate.startswith("var.remote_cpu_workers_enabled ?"), address
        assert _attr(body, "location") in (None, "var.primary_region", '"us"'), address
    assert 'startswith(var.primary_region, "us-")' in _terraform_variable_body(main, "primary_region")

    # The object prefix, which bounds every presigned URL a worker may use, is a US B2 endpoint.
    prefix = _terraform_variable_body(main, "remote_cpu_worker_object_prefix")
    assert _attr(prefix, "default") == '""'
    pattern = _hcl_regex(prefix, "remote_cpu_worker_object_prefix")
    accepted = ("https://s3.us-west-004.backblazeb2.com/b2-bucket/"
                "blueprint/arm-decision-proof-v1/configured-scenes/")
    assert pattern.fullmatch(accepted) and _object_prefix(accepted)
    for refused in (
        "https://s3.eu-central-003.backblazeb2.com/b2-bucket/prefix/",
        "http://s3.us-west-004.backblazeb2.com/b2-bucket/prefix/",
        "https://s3.us-west-004.backblazeb2.com.example.com/b2-bucket/prefix/",
        "https://user@s3.us-west-004.backblazeb2.com/b2-bucket/prefix/",
        "https://s3.us-west-004.backblazeb2.com/b2-bucket/prefix",
        "https://s3.us-west-004.backblazeb2.com/b2-bucket/../prefix/",
        "https://s3.us-west-004.backblazeb2.com/b2-bucket//prefix/",
        "https://s3.us-west-004.backblazeb2.com/b2-bucket/prefix/?x=1",
    ):
        assert not pattern.fullmatch(refused), refused


def test_remote_cpu_job_command_is_the_bootstrap_with_zero_retries_and_bounded_timeout() -> None:
    main = _main()
    job = _terraform_resource_body(main, "google_cloud_run_v2_job", "remote_cpu_worker")
    execution = _child(job, "template")
    task = _child(execution, "template")
    container = _child(task, "containers")

    assert _attr(job, "provider") == "google-beta"
    assert _attr(job, "for_each") == (
        "var.remote_cpu_workers_enabled ? var.remote_cpu_worker_stages : {}"
    )
    assert _attr(job, "name") == '"blueprint-remote-cpu-${each.key}"'
    # The capture job's own image with the worker bootstrap as its command. Execution overrides
    # can only append args, and the bootstrap refuses any.
    assert _attr(container, "image") == "var.docker_image"
    assert _strings(_attr(container, "command")) == list(BOOTSTRAP_COMMAND)
    assert _attr(container, "args") is None
    assert (_attr(execution, "task_count"), _attr(execution, "parallelism")) == ("1", "1")
    assert _attr(task, "max_retries") == "0"
    assert _attr(task, "timeout") == '"${each.value.timeout_seconds}s"'
    assert _attr(task, "execution_environment") == '"EXECUTION_ENVIRONMENT_GEN2"'
    assert _attr(task, "service_account") == "google_service_account.remote_cpu_worker[0].email"
    # The worker holds no secret: the template names its stage and pins the object prefix every
    # presigned URL must stay under; the dispatcher's overrides carry the attempt identifiers.
    env = {_strings(_attr(block, "name"))[0]: _attr(block, "value")
           for block in _children(container, "env")}
    assert env == {STAGE_VARIABLE: "each.key", PREFIX_VARIABLE: "var.remote_cpu_worker_object_prefix"}
    assert "value_source" not in job and "secret_key_ref" not in job
    precondition = _child(_child(job, "lifecycle"), "precondition")
    assert _attr(precondition, "condition") == 'var.remote_cpu_worker_object_prefix != ""'
    # /var/lib/blueprint is in memory and nothing else is mounted: no capture bucket.
    (volume,) = _children(task, "volumes")
    mount = _child(container, "volume_mounts")
    assert _attr(mount, "name") == _attr(volume, "name")
    assert _strings(_attr(mount, "mount_path")) == [contract.PERMITTED_PATH_ROOTS[0].rstrip("/")]
    empty_dir = _child(volume, "empty_dir")
    assert _attr(empty_dir, "medium") == '"MEMORY"'
    assert _attr(empty_dir, "size_limit") == "each.value.ephemeral_size_limit"
    limits = _child(_child(container, "resources"), "limits =")
    assert (_attr(limits, "cpu"), _attr(limits, "memory")) == ("each.value.cpu", "each.value.memory")

    stages = _terraform_variable_body(main, "remote_cpu_worker_stages")
    stage = _child(_child(stages, "default ="), "episode-compilation =")
    values = {name: _attr(stage, name)
              for name in ("cpu", "memory", "timeout_seconds", "ephemeral_size_limit")}
    assert values == {"cpu": '"4"', "memory": '"16Gi"', "timeout_seconds": "1800",
                      "ephemeral_size_limit": '"10Gi"'}
    assert _attr(stages, "nullable") == "false"
    # 1800 s is fetch + stage + seal/upload plus the margin (plan 14 §10); every stage is bounded
    # by the contract's hard limits, and the default stage is one the worker knows.
    assert sum(contract.STAGE_LIMITS["phase_seconds"].values()) + contract.PHASE_MARGIN_SECONDS == 1800
    assert f"<= {contract.MAX_TASK_TIMEOUT_SECONDS}" in stages
    assert f"<= {contract.MAX_MEMORY_BYTES // GIB}" in stages
    # Cloud Run's per-CPU memory range is checked at plan time, not first met at apply
    # (cpu "1" with 32Gi would pass a plan); the default stage sits inside it.
    per_cpu = [check for check in _children(stages, "validation") if "4 * tonumber(limits.cpu)" in check]
    assert len(per_cpu) == 1
    for bound in ('<= 4 * tonumber(limits.cpu)', '(tonumber(limits.cpu) < 4 ||', '>= 2)',
                  '(tonumber(limits.cpu) < 6 ||', '>= 4)'):
        assert bound in per_cpu[0], bound
    cpu, memory_gi = int(_strings(values["cpu"])[0]), int(_strings(values["memory"])[0].removesuffix("Gi"))
    assert memory_gi <= 4 * cpu and (cpu < 4 or memory_gi >= 2) and (cpu < 6 or memory_gi >= 4)
    assert "episode-compilation".replace("-", "_") in contract.STAGES
    assert contract._JOB.fullmatch("blueprint-remote-cpu-episode-compilation")
    # What jobs.get returns for this definition passes the dispatcher's own job check.
    definition = {"etag": "etag-1", "template": {
        "taskCount": int(_attr(execution, "task_count")),
        "parallelism": int(_attr(execution, "parallelism")),
        "template": {"maxRetries": int(_attr(task, "max_retries")),
                     "timeout": f"{values['timeout_seconds']}s",
                     "containers": [{"image": IMAGE, "command": _strings(_attr(container, "command")),
                                     "resources": {"limits": {"cpu": _strings(values["cpu"])[0],
                                                              "memory": _strings(values["memory"])[0]}}}]}}}
    assert job_definition_blockers(definition, image=IMAGE, timeout_seconds=1800, vcpu=4,
                                   memory_bytes=16 * GIB) == []
    # No new image and no new registry: the image variables are the existing five.
    assert set(re.findall(r'(?m)^variable "([a-z0-9_]*image[a-z0-9_]*)" \{$', main)) == IMAGE_VARIABLES


def test_transport_bucket_is_private_and_deletes_objects_after_a_day() -> None:
    """The transport bucket holds live presigned links: private, unversioned, gone within a day."""
    main = _main()
    bucket = _terraform_resource_body(main, "google_storage_bucket", "remote_cpu_transport")

    assert _attr(bucket, "count") == "var.remote_cpu_workers_enabled ? 1 : 0"
    assert _attr(bucket, "name") == '"${var.remote_cpu_project_id}-transport"'
    project = _strings(_attr(_terraform_variable_body(main, "project_id"), "default"))[0]
    assert re.fullmatch(cloud_run_jobs_client._BUCKET, f"{project}-remote-cpu-transport")
    assert _attr(bucket, "location") == "var.primary_region"
    assert _attr(bucket, "uniform_bucket_level_access") == "true"
    assert _attr(bucket, "public_access_prevention") == '"enforced"'
    assert _attr(_child(bucket, "versioning"), "enabled") == "false"
    assert _attr(_child(bucket, "soft_delete_policy"), "retention_duration_seconds") == "0"
    (rule,) = _children(bucket, "lifecycle_rule")
    assert _attr(_child(rule, "condition"), "age") == "1"
    assert _attr(_child(rule, "action"), "type") == '"Delete"'
    assert _attr(bucket, "force_destroy") is None
    for exposure in ("website", "cors", "retention_policy"):
        assert not _children(bucket, exposure), exposure


def test_remote_cpu_worker_identity_has_no_project_roles() -> None:
    main = _main()
    worker = _terraform_resource_body(main, "google_service_account", "remote_cpu_worker")
    assert _attr(worker, "account_id") == '"remote-cpu-worker"'

    # It runs the job and may get transport objects (the bucket policy's reader binding). Nothing
    # else names it; no project binding does.
    holders = {address for address, body in _resources(main).items()
               if "google_service_account.remote_cpu_worker[" in body}
    assert holders == {("google_cloud_run_v2_job", "remote_cpu_worker"),
                       ("data.google_iam_policy", "remote_cpu_dispatch_transport"),
                       ("google_tags_tag_binding", "remote_cpu_identity")}
    reader = _terraform_resource_body(main, "google_project_iam_custom_role",
                                      "remote_cpu_transport_reader")
    assert _list(reader, "permissions") == ["storage.objects.get"]
    # No authoritative project policy exists that could hand it a role either.
    assert 'resource "google_project_iam_binding"' not in main
    policy = _terraform_resource_body(main, "google_project_iam_policy", "remote_cpu")
    assert _attr(policy, "project") == "google_project.remote_cpu[0].project_id"
    project_policy = _resources(main)[("data.google_iam_policy", "remote_cpu_project")]
    assert "google_service_account.remote_cpu_worker" not in project_policy
    assert "google_service_account.remote_cpu_dispatcher" not in project_policy


def test_isolated_project_blocks_inherited_authority_before_worker_dependencies() -> None:
    resources = _resources(_main())
    project = resources[("google_project", "remote_cpu")]
    lifecycle = _child(project, "lifecycle")
    assert _attr(lifecycle, "prevent_destroy") == "true"
    postcondition = _child(lifecycle, "postcondition")
    assert "self.org_id" in _attr(postcondition, "condition")
    assert "self.folder_id" in _attr(postcondition, "condition")
    for name in ("remote_cpu_worker", "remote_cpu_quarantined_dispatcher"):
        assert _list(resources[("google_service_account", name)], "depends_on") == []
        assert "google_project_service.remote_cpu_apis" in resources[("google_service_account", name)]
        assert "google_project_iam_policy.remote_cpu" in resources[("google_service_account", name)]
    project_policy = resources[("data.google_iam_policy", "remote_cpu_project")]
    bindings = _children(project_policy, "binding")
    assert len(bindings) == 2
    assert {_attr(binding, "role"): _strings(_attr(binding, "members")) for binding in bindings} == {
        '"roles/owner"': ["user:ohstnhunt@gmail.com"],
        '"roles/run.serviceAgent"': [
            "serviceAccount:${google_project_service_identity.remote_cpu_run[0].email}"],
    }
    job = resources[("google_cloud_run_v2_job", "remote_cpu_worker")]
    assert "google_storage_bucket_iam_policy.remote_cpu_dispatch_transport" in job
    assert "google_project_iam_policy.remote_cpu" in job


def test_dispatcher_roles_are_custom_minimal_and_resource_scoped() -> None:
    resources = _resources(_main())
    dispatcher = resources[("google_service_account", "remote_cpu_host_dispatcher")]
    assert _attr(dispatcher, "account_id") == '"remote-cpu-dispatcher"'
    for name, permissions, project in (
        ("remote_cpu_dispatcher", DISPATCHER_PERMISSIONS, "remote_cpu"),
        ("remote_cpu_dispatch_transport_writer", TRANSPORT_WRITER_PERMISSIONS, "remote_cpu_dispatch"),
    ):
        role = resources[("google_project_iam_custom_role", name)]
        assert _attr(role, "project") == f"google_project.{project}[0].project_id"
        assert sorted(_list(role, "permissions")) == permissions
    holders = {address for address, body in resources.items()
               if "google_service_account.remote_cpu_host_dispatcher[" in body}
    assert holders == {("data.google_iam_policy", "remote_cpu_dispatch_job"),
                       ("data.google_iam_policy", "remote_cpu_dispatch_transport")}
    job = resources[("google_cloud_run_v2_job", "remote_cpu_worker")]
    policy = resources[("google_cloud_run_v2_job_iam_policy", "remote_cpu_dispatcher")]
    assert _attr(policy, "for_each") == _attr(job, "for_each")
    assert _attr(policy, "name") == "google_cloud_run_v2_job.remote_cpu_worker[each.key].name"
    assert _attr(policy, "location") == _attr(job, "location")
    document = resources[("data.google_iam_policy", "remote_cpu_dispatch_job")]
    (binding,) = _children(document, "binding")
    assert _attr(binding, "role") == "google_project_iam_custom_role.remote_cpu_dispatcher[0].name"
    assert _attr(binding, "members") == '["serviceAccount:${google_service_account.remote_cpu_host_dispatcher[0].email}"]'


def test_transport_bucket_policy_is_authoritative_and_exact() -> None:
    """Only the worker reads transport objects and only the dispatcher writes them.

    A new bucket starts with the project convenience bindings: Viewers read every object, and
    Editors (default service accounts among them) and Owners own them. Additive members would
    leave those in place, so the bucket's whole policy is set, and Owners keep only bucket
    management, which reads no object.
    """
    main = _main()
    resources = _resources(main)
    policy = resources[("google_storage_bucket_iam_policy", "remote_cpu_dispatch_transport")]
    assert _attr(policy, "count") == "var.remote_cpu_workers_enabled ? 1 : 0"
    assert _attr(policy, "bucket") == "google_storage_bucket.remote_cpu_dispatch_transport[0].name"
    assert _attr(policy, "policy_data") == (
        "data.google_iam_policy.remote_cpu_dispatch_transport[0].policy_data")
    document = resources[("data.google_iam_policy", "remote_cpu_dispatch_transport")]
    assert _attr(document, "count") == "var.remote_cpu_workers_enabled ? 1 : 0"
    bindings = _children(document, "binding")
    assert len(bindings) == 3
    # Raw HCL: each binding names one member.
    assert {_attr(binding, "role"): _attr(binding, "members") for binding in bindings} == {
        "google_project_iam_custom_role.remote_cpu_dispatch_transport_reader[0].name":
            '["serviceAccount:${google_service_account.remote_cpu_worker[0].email}"]',
        "google_project_iam_custom_role.remote_cpu_dispatch_transport_writer[0].name":
            '["serviceAccount:${google_service_account.remote_cpu_host_dispatcher[0].email}"]',
        '"roles/storage.legacyBucketOwner"': '["projectOwner:${var.remote_cpu_dispatch_project_id}"]',
    }
    # The policy is the bucket's only binding: an additive member or binding would fight it.
    on_bucket = {address for address, body in resources.items()
                 if "google_storage_bucket.remote_cpu_dispatch_transport[" in body}
    assert on_bucket == {("google_storage_bucket_iam_policy", "remote_cpu_dispatch_transport")}
    assert "google_storage_bucket_iam_member" not in "".join(
        body for (kind, name), body in resources.items() if name.startswith("remote_cpu"))


def test_no_service_account_key_is_managed_by_terraform() -> None:
    """The owner creates the dispatcher's key, and only the paid unit loads it (LoadCredential=).

    A key managed here would sit in Terraform state and could reach its outputs.
    """
    terraform = "\n".join(path.read_text(encoding="utf-8")
                          for path in sorted(TERRAFORM_MAIN.parent.glob("*.tf")))
    assert 'resource "google_service_account" "remote_cpu_host_dispatcher"' in terraform
    assert "google_service_account_key" not in terraform
    assert "private_key" not in terraform
    assert "keys create" not in DEPLOY_SCRIPT.read_text(encoding="utf-8")


def test_existing_storage_grants_exclude_the_transport_bucket() -> None:
    """Plan 14 C2: no project-wide storage role reaches the bucket that holds live presigned links."""
    main = _main()
    bucket = _terraform_resource_body(main, "google_storage_bucket", "remote_cpu_transport")
    transport = "projects/_/buckets/" + _strings(_attr(bucket, "name"))[0]

    grants = _project_grants(main, "roles/storage.")
    assert sorted(grants) == [
        "pipeline_runner_storage",  # objectAdmin
        "privacy_services_storage",  # objectAdmin, four privacy services
        "storage_trigger_storage",  # objectViewer
    ]
    for name, body in sorted(grants.items()):
        # The bucket itself and everything in it, and nothing else: a bucket merely sharing the
        # name as a prefix is unaffected.
        assert _remote_cpu_exclusion(body) == (
            f'resource.name != "{transport}" && !resource.name.startsWith("{transport}/")'), name


def test_existing_run_grants_exclude_remote_cpu_jobs() -> None:
    """Plan 14 C2: no project-wide run role can start, or invoke, a remote CPU job."""
    main = _main()
    job = _terraform_resource_body(main, "google_cloud_run_v2_job", "remote_cpu_worker")
    prefix = _strings(_attr(job, "name"))[0].removesuffix("${each.key}")
    assert prefix == "blueprint-remote-cpu-"
    assert _attr(job, "location") == "var.primary_region"

    grants = _project_grants(main, "roles/run.")
    assert sorted(grants) == [
        "pipeline_invoker_run",  # run.invoker
        "storage_trigger_run",  # run.invoker
        "storage_trigger_run_jobs",  # run.jobsExecutorWithOverrides
    ]
    # A job's resource name may carry the project id or its number. Either way every remote CPU
    # job, and each of its executions, falls under one of these prefixes; no other job does.
    jobs = [f"projects/{project}/locations/${{var.primary_region}}/jobs/{prefix}"
            for project in ("${var.project_id}", "${data.google_project.current.number}")]
    expected = " && ".join(f'!resource.name.startsWith("{name}")' for name in jobs)
    for name, body in sorted(grants.items()):
        assert _remote_cpu_exclusion(body) == expected, name


def test_pipeline_failure_alert_excludes_remote_cpu_jobs_which_have_their_own() -> None:
    main = _main()
    job = _terraform_resource_body(main, "google_cloud_run_v2_job", "remote_cpu_worker")
    remote_jobs = 'resource.labels.job_name = starts_with("{}")'.format(
        _strings(_attr(job, "name"))[0].removesuffix("${each.key}"))
    failed_attempts = [
        'resource.type="cloud_run_job"',
        'metric.type="run.googleapis.com/job/completed_task_attempt_count"',
        'metric.labels.result="failed"',
    ]

    # The capture pipeline's alert: its filter is unchanged while the workers are off, and drops
    # the remote CPU jobs once they exist.
    existing = _terraform_resource_body(main, "google_monitoring_alert_policy", "pipeline_failures")
    threshold = _child(_child(existing, "conditions"), "condition_threshold")
    expression = threshold[threshold.index("filter") : threshold.index("duration")]
    assert 'join(" AND ", concat(' in expression
    assert _strings(expression) == [" AND ", *failed_attempts, "NOT " + remote_jobs]
    gated = re.search(r'var\.remote_cpu_workers_enabled \? \[("[^\]]*")\] : \[\]', expression)
    assert gated is not None and _strings(gated.group(1)) == ["NOT " + remote_jobs]
    assert " AND ".join(failed_attempts) == (
        'resource.type="cloud_run_job" AND '
        'metric.type="run.googleapis.com/job/completed_task_attempt_count" AND '
        'metric.labels.result="failed"')

    # Their own alert fires on any failed attempt: with no retries, none is retried away.
    own = _terraform_resource_body(main, "google_monitoring_alert_policy", "remote_cpu_job_failures")
    assert _attr(own, "count") == "var.remote_cpu_workers_enabled ? 1 : 0"
    own_threshold = _child(_child(own, "conditions"), "condition_threshold")
    assert _strings(_attr(own_threshold, "filter")) == [" AND ".join([*failed_attempts, remote_jobs])]
    assert (_attr(own_threshold, "comparison"), _attr(own_threshold, "threshold_value"),
            _attr(own_threshold, "duration")) == ('"COMPARISON_GT"', "0", '"0s"')
    assert _child(own_threshold, "aggregations") == _child(threshold, "aggregations")
    # The same receivers, and the same refusal to exist without one, as the existing policies.
    assert _attr(own, "notification_channels") == "[google_monitoring_notification_channel.remote_cpu_owner[0].name]"
    assert _attr(_child(_child(own, "lifecycle"), "precondition"), "condition") == 'google_monitoring_notification_channel.remote_cpu_owner[0].name != ""'


def test_remote_cpu_budget_alerts_at_the_approved_cap() -> None:
    """Owner decision 2: the lane's first month runs under a $25 cap, backed by a budget alert."""
    main = _main()
    budget = _terraform_resource_body(main, "google_billing_budget", "remote_cpu_workers")
    assert _attr(budget, "count") == "var.remote_cpu_workers_enabled ? 1 : 0"
    # Required, not optional, once the workers exist.
    precondition = _child(_child(budget, "lifecycle"), "precondition")
    assert _attr(precondition, "condition") == 'var.billing_account_id != ""'
    assert _attr(budget, "billing_account") == "var.billing_account_id"

    amount = _terraform_variable_body(main, "remote_cpu_workers_budget_usd")
    thresholds = _terraform_variable_body(main, "remote_cpu_workers_budget_thresholds")
    assert (_attr(amount, "default"), _attr(thresholds, "default")) == ("25", "[0.5, 0.9, 1.0]")
    specified = _child(_child(budget, "amount"), "specified_amount")
    assert _attr(specified, "currency_code") == '"USD"'
    assert _attr(specified, "units") == "tostring(var.remote_cpu_workers_budget_usd)"
    rules = _child(budget, 'dynamic "threshold_rules"')
    assert _attr(rules, "for_each") == "var.remote_cpu_workers_budget_thresholds"
    assert _attr(_child(rules, "content"), "threshold_percent") == "threshold_rules.value"
    example = TFVARS_EXAMPLE.read_text(encoding="utf-8")
    assert re.search(r"(?m)^remote_cpu_workers_budget_usd\s*=\s*25\s*$", example)

    # Scoped by label to the lane's own resources, which all carry it.
    budget_filter = _child(budget, "budget_filter")
    assert _attr(budget_filter, "projects") == '["projects/${google_project.remote_cpu[0].number}", "projects/${google_project.remote_cpu_dispatch[0].number}"]'
    assert _attr(budget_filter, "labels") is None
    resources = _resources(main)
    for address in (("google_cloud_run_v2_job", "remote_cpu_worker"),
                    ("google_storage_bucket", "remote_cpu_transport")):
        assert _attr(resources[address], "labels") == (
            'merge(local.common_labels, { cost-center = "remote-cpu-workers" })'), address


def test_deploy_script_defaults_remote_cpu_workers_on_and_exports_the_flag() -> None:
    """Enablement persists only as a committed default (review finding I7), never as a one-off
    environment value that the next deploy would silently undo by destroying the workers."""
    deploy = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    configuration = deploy[: deploy.index("# Directories")].splitlines()
    default = 'REMOTE_CPU_WORKERS_ENABLED="${REMOTE_CPU_WORKERS_ENABLED:-true}"'
    assert configuration.count(default) == 1
    # Beside the other committed flag defaults, the privacy flags.
    fail_closed = configuration.index('PRIVACY_FAIL_CLOSED="${PRIVACY_FAIL_CLOSED:-true}"')
    assert fail_closed < configuration.index(default) < configuration.index(
        'PRIVACY_SAM3_URL="${PRIVACY_SAM3_URL:-}"')
    assert ('REMOTE_CPU_WORKER_OBJECT_PREFIX="${REMOTE_CPU_WORKER_OBJECT_PREFIX:-'
            'https://s3.us-east-005.backblazeb2.com/blueprint-task-evaluation-artifacts-prod/'
            'blueprint/arm-decision-proof-v1/configured-scenes/}"') in configuration
    assert "REMOTE_CPU_WORKERS_ENABLED:-false" not in deploy
    # setup_iam is outside the deploy flow, and its unconditional project grants would undo plan 14
    # C2's conditions: it refuses before any grant once the workers are enabled.
    setup_iam = deploy[deploy.index("setup_iam() {"):]
    refusal = setup_iam.index('if [[ "$REMOTE_CPU_WORKERS_ENABLED" == "true" ]]; then')
    assert refusal < setup_iam.index("gcloud ") and "return 1" in setup_iam[refusal:setup_iam.index("fi\n", refusal)]

    # Exported with the rest of the fixed TF_VAR set, before the backend is touched or any plan.
    start = deploy.index("apply_terraform() {")
    exports = [line.strip() for line in
               deploy[start : deploy.index("    validate_terraform_state_backend", start)].splitlines()]
    flag = 'export TF_VAR_remote_cpu_workers_enabled="$REMOTE_CPU_WORKERS_ENABLED"'
    prefix = 'export TF_VAR_remote_cpu_worker_object_prefix="$REMOTE_CPU_WORKER_OBJECT_PREFIX"'
    assert exports.index('export TF_VAR_privacy_fail_closed="$PRIVACY_FAIL_CLOSED"') < exports.index(flag)
    assert prefix in exports
    # A bare Terraform invocation remains conservative; the deploy script is the committed opt-in.
    flag_variable = _terraform_variable_body(_main(), "remote_cpu_workers_enabled")
    assert _attr(flag_variable, "default") == "false"


def test_terraform_static_workflow_pins_setup_terraform_and_never_plans() -> None:
    """Plan 14 PR 5.2: CI checks syntax and provider schema with no credentials and no plan."""
    text = TERRAFORM_STATIC_WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    events = workflow.get("on", workflow.get(True))  # YAML 1.1 reads the bare key on as true
    for event in ("pull_request", "push"):
        assert "deploy/terraform/**" in events[event]["paths"], event
    assert events["push"]["branches"] == ["main"]
    assert workflow["permissions"] == {"contents": "read"}

    job = workflow["jobs"]["terraform-static"]
    assert 0 < job["timeout-minutes"] <= 15
    assert job["defaults"]["run"]["working-directory"] == "deploy/terraform"
    steps = job["steps"]
    checkout = [step for step in steps if str(step.get("uses", "")).startswith("actions/checkout@")]
    assert len(checkout) == 1 and checkout[0]["with"]["persist-credentials"] is False
    assert re.fullmatch(r"actions/checkout@[0-9a-f]{40}", checkout[0]["uses"])
    setup = [step for step in steps
             if str(step.get("uses", "")).startswith("hashicorp/setup-terraform@")]
    assert len(setup) == 1
    assert re.fullmatch(r"hashicorp/setup-terraform@[0-9a-f]{40}", setup[0]["uses"])
    # An exact release, never latest or a range, and one main.tf accepts.
    version = str(setup[0]["with"]["terraform_version"])
    assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version)
    required = re.search(r'required_version = ">= ([0-9]+)\.([0-9]+)\.([0-9]+)"', _main())
    assert required is not None
    assert tuple(map(int, version.split("."))) >= tuple(map(int, required.groups()))
    assert setup[0]["with"]["terraform_wrapper"] is False

    # fmt, then init with the committed lock and no backend, then validate: nothing else runs.
    commands = [" ".join(str(step["run"]).split()) for step in steps if "run" in step]
    assert commands == [
        "terraform fmt -check -diff -recursive",
        "terraform init -backend=false -input=false -lockfile=readonly",
        "terraform validate -no-color",
    ]
    invoked = re.findall(r"(?m)(?<![\w./-])terraform[ \t]+([a-z-]+)", text)
    assert set(invoked) == {"fmt", "init", "validate"}
    for credential in ("secrets.", "google-github-actions/auth", "GOOGLE_", "id-token",
                       "cli_config_credentials_token"):
        assert credential not in text, credential
