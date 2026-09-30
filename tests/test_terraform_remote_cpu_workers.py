# Covers (for impacted-test selection):
#   deploy/terraform/main.tf
#   deploy/terraform/terraform.tfvars.example
#   deploy/scripts/deploy.sh
"""ADP-009D/day-28, plan 14 PR 5: the remote CPU worker infrastructure is off by default and fenced.

Terraform is read as text, as the deploy contract tests read it: CI has no terraform binary and no
GCP credentials here, and nothing in this file plans or applies.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from blueprint_pipeline import cloud_run_jobs_client
from blueprint_pipeline import remote_cpu_job_contract as contract
from blueprint_pipeline.cloud_run_jobs_client import BOOTSTRAP_COMMAND, job_definition_blockers
from blueprint_pipeline.remote_cpu_worker import PREFIX_VARIABLE, STAGE_VARIABLE, _object_prefix
from tests.test_deploy_systemd_contract import _terraform_resource_body, _terraform_variable_body

REPO_ROOT = Path(__file__).resolve().parents[1]
TERRAFORM_MAIN = REPO_ROOT / "deploy" / "terraform" / "main.tf"
TFVARS_EXAMPLE = REPO_ROOT / "deploy" / "terraform" / "terraform.tfvars.example"
GIB = 1024**3
IMAGE = "gcr.io/blueprint-8c1ca/blueprint-pipeline@sha256:" + "d" * 64
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


def _resources(text: str) -> dict[tuple[str, str], str]:
    return {(kind, name): _terraform_resource_body(text, kind, name)
            for kind, name in re.findall(r'(?m)^resource "([a-z0-9_]+)" "([a-z0-9_]+)" \{$', text)}


def _hcl_regex(body: str, variable: str) -> re.Pattern[str]:
    """The pattern a validation passes to ``regex(..., var.<variable>)``; RE2 and ``re`` agree on it."""
    match = re.search(rf'regex\(("(?:[^"\\]|\\.)*"), var\.{variable}\)', body)
    assert match is not None, variable
    return re.compile(json.loads(match.group(1)))


def test_remote_cpu_jobs_are_off_by_default_and_us_only() -> None:
    main = _main()
    flag = _terraform_variable_body(main, "remote_cpu_workers_enabled")
    assert (_attr(flag, "type"), _attr(flag, "default"), _attr(flag, "nullable")) == (
        "bool", "false", "false")
    example = TFVARS_EXAMPLE.read_text(encoding="utf-8")
    assert re.findall(r"(?m)^remote_cpu_workers_enabled\s*=\s*(\S+)\s*$", example) == ["false"]

    # Every remote CPU resource exists only while the flag is on: off, its count is 0 or its map
    # empty. Anything with a location sits in the primary region, which must be a US region.
    remote = {address: body for address, body in _resources(main).items()
              if address[1].startswith("remote_cpu")}
    assert ("google_cloud_run_v2_job", "remote_cpu_worker") in remote
    for address, body in sorted(remote.items()):
        gate = _attr(body, "count") or _attr(body, "for_each") or ""
        assert re.fullmatch(r"var\.remote_cpu_workers_enabled \? \S+ : (?:0|\{\})", gate), address
        assert _attr(body, "location") in (None, "var.primary_region"), address
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
    assert _attr(bucket, "name") == '"${var.project_id}-remote-cpu-transport"'
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
