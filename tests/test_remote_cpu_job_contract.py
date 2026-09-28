# Covers (for impacted-test selection):
#   src/blueprint_pipeline/remote_cpu_job_contract.py
#   src/blueprint_pipeline/remote_cpu_job_lease.py
"""ADP-009D/day-28, plan 14 PR 1: remote CPU job records carry no URL or credential."""

from __future__ import annotations

import copy
import json
import stat
from pathlib import Path

import pytest

from blueprint_pipeline import remote_cpu_job_contract as contract
from blueprint_pipeline import remote_cpu_job_lease as lease
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

GIB = 1024**3
ROOT = "/var/lib/blueprint"
IMAGE = "gcr.io/blueprint-8c1ca/blueprint-pipeline@sha256:" + "d" * 64
ENVELOPE_HEX = "e" * 64
QUEUE_NAME = f"prep-1-{ENVELOPE_HEX}.json"
SOURCE_HEX = "5" * 64
CAS = "s3://b2-bucket/blueprint/arm-decision-proof-v1/configured-scenes/artifacts"
EXECUTION = "blueprint-remote-cpu-episode-compilation-x7k2p"
CPU_CLASS = "sha256:" + "c" * 64


def _seal(value: dict, field: str) -> dict:
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _config(**changes) -> dict:
    config = {
        "schema_version": "remote_cpu_workers_config.v1",
        "project": "blueprint-8c1ca",
        "region": "us-central1",
        "transport_bucket": "blueprint-8c1ca-remote-cpu-transport",
        "stages": {
            "episode_compilation": {
                "job": "blueprint-remote-cpu-episode-compilation",
                "image": IMAGE,
                "vcpu": 4,
                "memory_bytes": 16 * GIB,
                "ephemeral_bytes": 10 * GIB,
                "task_timeout_seconds": 1800,
            }
        },
        "rate_table": {
            "source": "cloud-run-list-prices",
            "observed_on": "2026-09-28",
            "usd_per_vcpu_second": 0.000018,
            "usd_per_gib_second": 0.000002,
        },
        "max_live_executions": 2,
        "max_attempts": 2,
        "config_digest": "",
    }
    config.update(changes)
    return _seal(config, "config_digest")


def _limits(**changes) -> dict:
    limits = {
        "vcpu": 4,
        "memory_bytes": 16 * GIB,
        "ephemeral_bytes": 10 * GIB,
        "task_timeout_seconds": 1800,
        "phase_seconds": {"fetch": 300, "stage": 900, "seal_upload": 420},
        "start_allowance_seconds": 600,
        "heartbeat_interval_seconds": 30,
        "heartbeat_stale_seconds": 180,
        "max_input_bytes": 6 * GIB,
        "max_output_bytes": 4 * GIB,
        "max_output_paths": 20000,
        "allowed_path_roots": ["/var/lib/blueprint/"],
        "max_attempts": 2,
        "allowed_cpu_classes": [CPU_CLASS],
    }
    limits.update(changes)
    return limits


def _input(role: str, name: str, digit: str, size: int, materialize_at: str) -> dict:
    return {
        "role": role,
        "contract_path": name,
        "digest": "sha256:" + digit * 64,
        "size_bytes": size,
        "mode": "0440",
        "materialize_at": materialize_at,
        "uri": f"{CAS}/remote-cpu-input/sha256/{digit * 64}/{name.rsplit('.', 1)[-1]}.bin",
    }


def _inputs() -> list[dict]:
    return [
        _input(
            "queue_envelope", "queue_envelope", "1", 2048,
            f"{ROOT}/pipeline-control-plane/task-evaluation-episode-compilations/processing/{QUEUE_NAME}",
        ),
        _input(
            "materialized_reference", "execution_adapter.runtime_source_bundle", "2", 4096,
            f"{ROOT}/task-evaluation-inputs/prepared-references/prep-1/runtime.zip",
        ),
    ]


def _code(**changes) -> dict:
    code = {
        "source_commit": "a" * 40,
        "source_archive": {
            "digest": "sha256:" + SOURCE_HEX,
            "size_bytes": 1234,
            "uri": f"{CAS}/remote-cpu-source/sha256/{SOURCE_HEX}/source.tar",
        },
        "image": IMAGE,
        "environment_digest": "sha256:" + "4" * 64,
    }
    code.update(changes)
    return code


def _descriptor(config: dict | None = None, **changes) -> dict:
    config = config or _config()
    arguments = {
        "config": config,
        "stage": "episode_compilation",
        "mode": "shadow",
        "attempt": 1,
        "queue_row": {
            "queue": "task-evaluation-episode-compilations",
            "name": QUEUE_NAME,
            "envelope_digest": "sha256:" + ENVELOPE_HEX,
        },
        "code": _code(),
        "environment": {},
        "inputs": _inputs(),
        "outputs": {
            "output_root": f"{ROOT}/task-evaluation-inputs/compiled-episodes/prep-1",
            "declared_scratch": [f"{ROOT}/task-evaluation-inputs/compiled-episodes/content-addressed/"],
            "object_prefix": "s3://b2-bucket/blueprint/arm-decision-proof-v1",
        },
        "limits": _limits(),
        "closure": {"class": "not_applicable", "source_appearance_digest": None},
        "spend": {"worst_case_usd": 0.67, "rate_table_digest": canonical_digest(config["rate_table"])},
        "nonce": "0" * 32,
    }
    for name in ("output_root", "declared_scratch"):
        if name in changes:
            arguments["outputs"] = {**arguments["outputs"], name: changes.pop(name)}
    arguments.update(changes)
    return contract.build_descriptor(**arguments)


def _result(descriptor: dict, *, blocker: str | None = None) -> dict:
    result = {
        "schema_version": "task_evaluation_episode_compilation_result.v1",
        "compilation_id": "prep-1",
        "source_commit": descriptor["code"]["source_commit"],
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "result_digest": "",
    }
    if blocker is None:
        result.update(status="compiled_for_production_launch", blockers=[])
    else:
        result.update(status="blocked", blockers=[blocker], automatic_retry_performed=False)
    return _seal(result, "result_digest")


def _output() -> dict:
    return {
        "format": "remote_cpu_output.v1",
        "index": {"digest": "sha256:" + "6" * 64, "size_bytes": 900},
        "archive": {"digest": "sha256:" + "7" * 64, "size_bytes": 10240},
        "paths_total": 12,
        "bytes_total": 50000,
        "host_known": {"count": 8, "bytes": 40000},
    }


def _receipt(descriptor: dict, **changes) -> dict:
    status = changes.pop("status", "succeeded")
    receipt = {
        "schema_version": "remote_cpu_job_receipt.v1",
        "job_id": descriptor["job_id"],
        "attempt": descriptor["attempt"],
        "attempt_id": descriptor["attempt_id"],
        "stage": descriptor["stage"],
        "descriptor_digest": descriptor["descriptor_digest"],
        "execution_name": EXECUTION,
        "status": status,
        "result": _result(descriptor) if status == "succeeded" else None,
        "infrastructure_failures": [],
        "release_path_misses": [],
        "environment": {"environment_digest": descriptor["code"]["environment_digest"], "cpu_class": CPU_CLASS},
        "phases": {"bootstrap": 3.0, "fetch": 20.5, "stage": 300.0, "seal_upload": 40.0},
        "bytes_fetched": 6144,
        "bytes_uploaded": 11140,
        "output": _output() if status == "succeeded" else None,
        "private_url_recorded": False,
        "receipt_digest": "",
    }
    receipt.update(changes)
    return _seal(receipt, "receipt_digest")


def _reasons(call) -> tuple[str, ...]:
    with pytest.raises(contract.RemoteCpuContractError) as caught:
        call()
    return caught.value.reasons


def test_descriptor_digest_binds_code_inputs_outputs_limits_and_attempt() -> None:
    config = _config()
    descriptor = _descriptor(config)

    assert contract.validate_descriptor(descriptor, config=config) == descriptor
    assert descriptor["job_id"] == contract.job_id_for("episode_compilation", QUEUE_NAME)
    assert descriptor["job_id"].startswith("rcj-ec-") and len(descriptor["job_id"]) == 7 + 24
    assert descriptor["attempt_id"] == f"{descriptor['job_id']}-a1-{'0' * 32}"
    assert descriptor["outputs"]["staging_prefix"] == (
        "s3://b2-bucket/blueprint/arm-decision-proof-v1/remote-cpu/staging/"
        f"{descriptor['job_id']}/{descriptor['attempt_id']}/"
    )
    assert descriptor["private_url_recorded"] is False
    assert _descriptor(config) == descriptor
    second = _descriptor(config, attempt=2, nonce="f" * 32)
    assert second["job_id"] == descriptor["job_id"]
    assert second["attempt_id"] == f"{descriptor['job_id']}-a2-{'f' * 32}"
    assert second["descriptor_digest"] != descriptor["descriptor_digest"]
    random_one = _descriptor(config, nonce=None)
    random_two = _descriptor(config, nonce=None)
    assert random_one["attempt_id"] != random_two["attempt_id"]

    mutations = {
        "code": lambda d: d["code"].update(source_commit="b" * 40),
        "source_archive": lambda d: d["code"]["source_archive"].update(size_bytes=1235),
        "environment_digest": lambda d: d["code"].update(environment_digest="sha256:" + "5" * 64),
        "image": lambda d: d["code"].update(image=IMAGE[:-1] + "0"),
        "inputs": lambda d: d["inputs"][1].update(digest="sha256:" + "9" * 64),
        "input_mode": lambda d: d["inputs"][1].update(mode="0640"),
        "outputs": lambda d: d["outputs"].update(output_root=f"{ROOT}/task-evaluation-inputs/compiled-episodes/prep-2"),
        "limits": lambda d: d["limits"].update(max_output_paths=19999),
        "phases": lambda d: d["limits"]["phase_seconds"].update(stage=901),
        "attempt": lambda d: d.update(attempt=2),
        "attempt_id": lambda d: d.update(attempt_id=f"{d['job_id']}-a1-{'1' * 32}"),
        "closure": lambda d: d["closure"].update({"class": "shipped"}),
        "spend": lambda d: d["spend"].update(worst_case_usd=0.68),
        "mode": lambda d: d.update(mode="authoritative"),
    }
    for label, mutate in mutations.items():
        changed = copy.deepcopy(descriptor)
        mutate(changed)
        assert canonical_digest(changed, digest_field="descriptor_digest") != descriptor["descriptor_digest"], label
        reasons = _reasons(lambda: contract.validate_descriptor(changed, config=config))
        assert "remote_cpu_descriptor_digest_mismatch" in reasons, label

    extra = copy.deepcopy(descriptor)
    extra["inputs"][0]["note"] = "smuggled"
    _seal(extra, "descriptor_digest")
    assert "remote_cpu_field_unexpected:inputs[0].note" in _reasons(
        lambda: contract.validate_descriptor(extra, config=config)
    )
    other_image = _config(stages={"episode_compilation": {**config["stages"]["episode_compilation"], "image": IMAGE[:-1] + "0"}})
    assert "remote_cpu_image_mismatch" in _reasons(
        lambda: contract.validate_descriptor(descriptor, config=other_image)
    )


def test_descriptor_refuses_paths_outside_allowed_roots_and_non_us_regions() -> None:
    def refused(**changes) -> tuple[str, ...]:
        return _reasons(lambda: _descriptor(**changes))

    def with_input(index: int, **row_changes) -> list[dict]:
        rows = _inputs()
        rows[index].update(row_changes)
        return rows

    assert "remote_cpu_path_outside_allowed_roots:inputs[1].materialize_at" in refused(
        inputs=with_input(1, materialize_at="/tmp/runtime.zip")
    )
    assert "remote_cpu_path_not_normalized:inputs[1].materialize_at" in refused(
        inputs=with_input(1, materialize_at=f"{ROOT}/task-evaluation-inputs/../../../etc/passwd")
    )
    assert "remote_cpu_path_not_absolute:inputs[1].materialize_at" in refused(
        inputs=with_input(1, materialize_at="var/lib/blueprint/x.zip")
    )
    assert "remote_cpu_path_outside_allowed_roots:inputs[1].materialize_at" in refused(
        inputs=with_input(1, materialize_at="/var/lib/blueprint")
    )
    duplicate = _inputs()
    duplicate[1]["materialize_at"] = duplicate[0]["materialize_at"]
    assert "remote_cpu_descriptor_materialize_at_duplicate:inputs[1]" in refused(inputs=duplicate)
    assert "remote_cpu_descriptor_input_inside_output_root:inputs[1]" in refused(
        inputs=with_input(1, materialize_at=f"{ROOT}/task-evaluation-inputs/compiled-episodes/prep-1/x.zip")
    )
    assert "remote_cpu_path_outside_allowed_roots:outputs.output_root" in refused(
        output_root="/srv/compiled-episodes/prep-1"
    )
    assert "remote_cpu_path_outside_allowed_roots:outputs.declared_scratch[0]" in refused(
        declared_scratch=["/tmp/scratch/"]
    )
    assert "remote_cpu_descriptor_output_root_unbound" in refused(
        output_root=f"{ROOT}/task-evaluation-inputs/compiled-episodes/another-row"
    )
    assert "remote_cpu_field_invalid:limits.allowed_path_roots" in refused(
        limits=_limits(allowed_path_roots=["/"])
    )
    assert "remote_cpu_path_outside_allowed_roots:environment.BLUEPRINT_PARTICLEFIELD_RUNTIME_ASSET_CACHE_ROOT" in refused(
        environment={"BLUEPRINT_PARTICLEFIELD_RUNTIME_ASSET_CACHE_ROOT": "/opt/cache"}
    )
    assert "remote_cpu_descriptor_environment_variable_not_allowed:PYTHONPATH" in refused(
        environment={"PYTHONPATH": f"{ROOT}/src"}
    )
    assert "remote_cpu_descriptor_input_bytes_exceed_limit" in refused(
        inputs=with_input(1, size_bytes=6 * GIB)
    )
    assert "remote_cpu_descriptor_limits_do_not_match_job:vcpu" in refused(limits=_limits(vcpu=8))
    assert "remote_cpu_descriptor_phase_budget_exceeds_task_timeout" in refused(
        limits=_limits(phase_seconds={"fetch": 600, "stage": 900, "seal_upload": 420})
    )
    assert "remote_cpu_descriptor_closure_requires_qualified_cpu_classes" in refused(
        closure={"class": "absent_inline_only", "source_appearance_digest": "sha256:" + "8" * 64},
        limits=_limits(allowed_cpu_classes=[]),
    )
    assert "remote_cpu_uri_not_content_addressed:inputs[1].uri" in refused(
        inputs=with_input(1, uri=f"{CAS}/remote-cpu-input/sha256/{'3' * 64}/runtime.zip")
    )

    non_us = _config(region="europe-west1")
    reasons = refused(config=non_us, spend={"worst_case_usd": 0.67, "rate_table_digest": canonical_digest(non_us["rate_table"])})
    assert "remote_cpu_config_region_not_us" in reasons
    assert "remote_cpu_descriptor_region_not_us" in reasons
    config = _config()
    descriptor = _descriptor(config)
    moved = copy.deepcopy(descriptor)
    moved["execution"]["region"] = "europe-west1"
    _seal(moved, "descriptor_digest")
    assert "remote_cpu_descriptor_region_not_us" in _reasons(
        lambda: contract.validate_descriptor(moved, config=config)
    )
    unsealed = dict(config, region="asia-east1")
    assert "remote_cpu_config_invalid:config_digest" in _reasons(
        lambda: contract.validate_descriptor(descriptor, config=unsealed)
    )


@pytest.mark.parametrize(
    "key, value",
    [
        ("api_key", "harmless"),
        ("session_token", "harmless"),
        ("aws_secret", "harmless"),
        ("password", "harmless"),
        ("note", "https://b2.example.test/bucket/object"),
        ("note", "s3://b2-bucket/object?X-Amz-Signature=abc"),
        ("note", "X-Amz-Credential=AKIDEXAMPLE/20260928"),
        ("note", "gs://transport/object.json?generation=1"),
        ("note", "AKIAABCDEFGHIJKLMNOP"),
        ("note", "Bearer ya29.a0AfH6SM"),
    ],
)
def test_records_refuse_url_and_credential_shaped_keys_and_values(tmp_path: Path, key: str, value: str) -> None:
    config = _config()
    descriptor = _descriptor(config)

    def expect_refused(call) -> None:
        reasons = _reasons(call)
        assert any(
            reason.startswith(("remote_cpu_record_credential_shaped_key:", "remote_cpu_record_url_or_credential_value:"))
            for reason in reasons
        ), reasons
        assert value not in " ".join(reasons)

    inputs = _inputs()
    inputs[1][key] = value
    expect_refused(lambda: _descriptor(config, inputs=inputs))
    tampered = copy.deepcopy(descriptor)
    tampered["queue_row"][key] = value
    _seal(tampered, "descriptor_digest")
    expect_refused(lambda: contract.validate_descriptor(tampered, config=config))

    result = _result(descriptor)
    result[key] = value
    _seal(result, "result_digest")
    receipt = _receipt(descriptor, result=result)
    expect_refused(lambda: contract.validate_receipt(receipt, descriptor=descriptor, execution_name=EXECUTION))

    heartbeat = {
        "schema_version": "remote_cpu_job_heartbeat.v1",
        "attempt_id": descriptor["attempt_id"],
        "execution_name": EXECUTION,
        "sequence": 1,
        "phase": "fetch",
        "elapsed_seconds": 1.5,
        "bytes_fetched": 0,
        "bytes_uploaded": 0,
        key: value,
    }
    expect_refused(
        lambda: contract.validate_heartbeat(heartbeat, attempt_id=descriptor["attempt_id"], execution_name=EXECUTION)
    )
    teardown = {
        "descriptor": descriptor,
        "worker_identity": contract.worker_identity_for(descriptor["execution"], EXECUTION),
        "outcome": "completed",
        "compute": _compute(),
        "provider": _provider(),
        "observed_at_epoch": 2_000_000_000.0,
    }
    if key == "note":
        teardown["outcome"] = value
    else:
        teardown["provider"] = {**_provider(), key: value}
    expect_refused(lambda: contract.teardown_record(**teardown))
    fields = _pointer_fields(descriptor)
    fields["code"] = {**fields["code"], key: value}
    expect_refused(lambda: contract.pointer_record(fields))
    target = tmp_path / "record.json"
    expect_refused(lambda: contract.write_remote_cpu_record(target, {"schema_version": "x.v1", key: value}))
    assert not target.exists() and list(tmp_path.iterdir()) == []


def test_transport_schema_is_refused_by_every_record_writer(tmp_path: Path) -> None:
    descriptor = _descriptor()
    transport = {
        "schema_version": "remote_cpu_job_transport.v1",
        "descriptor": descriptor,
        "inputs": [{"digest": descriptor["inputs"][0]["digest"], "get": "presigned-get-redacted"}],
    }
    root = tmp_path / "remote-cpu-jobs"
    target = root / "descriptors" / f"{descriptor['attempt_id']}.json"

    assert "remote_cpu_transport_never_persisted:$" in _reasons(
        lambda: contract.write_remote_cpu_record(target, transport)
    )
    assert "remote_cpu_transport_never_persisted:shadow.transport" in _reasons(
        lambda: contract.write_remote_cpu_record(target, {"schema_version": "x.v1", "shadow": {"transport": transport}})
    )
    assert not root.exists() or not any(path.is_file() for path in root.rglob("*"))

    assert contract.write_remote_cpu_record(target, descriptor) is True
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert json.loads(target.read_text(encoding="utf-8")) == descriptor
    assert contract.write_remote_cpu_record(target, descriptor) is False
    changed = dict(descriptor, mode="authoritative")
    assert f"remote_cpu_record_conflict:{target.name}" in _reasons(
        lambda: contract.write_remote_cpu_record(target, changed)
    )
    assert sorted(path.name for path in target.parent.iterdir()) == [target.name]

    config, now = _config(), 2_000_000_000.0
    job, attempt = descriptor["job_id"], descriptor["attempt_id"]
    assert "remote_cpu_transport_never_persisted:$" in _reasons(
        lambda: lease.claim_handoff(root, descriptor=transport, config=config, now=now)
    )
    lease.claim_handoff(root, descriptor=descriptor, config=config, now=now)
    assert "remote_cpu_transport_never_persisted:$" in _reasons(lambda: lease.transition(
        root, job, attempt_id=attempt, to_state=None, now=now, updates={"teardown": transport}
    ))
    assert "remote_cpu_record_url_or_credential_value:outcome" in _reasons(lambda: lease.transition(
        root, job, attempt_id=attempt, to_state=None, now=now, updates={"outcome": "https://b2.example.test/object"}
    ))
    assert "remote_cpu_transport_never_persisted:$" in _reasons(
        lambda: lease.observe_heartbeat(root, job, transport, execution_running=True, now=now)
    )
    for path in (item for item in tmp_path.rglob("*") if item.is_file()):
        assert b"remote_cpu_job_transport.v1" not in path.read_bytes(), path
        assert b"https://" not in path.read_bytes(), path


def test_receipt_must_echo_descriptor_attempt_and_execution_identity() -> None:
    descriptor = _descriptor()
    verdict = contract.validate_receipt(_receipt(descriptor), descriptor=descriptor, execution_name=EXECUTION)
    assert verdict["outcome"] == "succeeded" and verdict["terminal"] is True
    assert verdict["infrastructure_failures"] == []

    other_attempt = _descriptor(attempt=2, nonce="f" * 32)
    for field, replacement in {
        "job_id": "rcj-ec-" + "0" * 24,
        "attempt": 2,
        "attempt_id": other_attempt["attempt_id"],
        "stage": "environment_probe",
        "descriptor_digest": other_attempt["descriptor_digest"],
        "execution_name": "blueprint-remote-cpu-episode-compilation-zzzzz",
    }.items():
        receipt = _receipt(descriptor, **{field: replacement})
        assert f"remote_cpu_receipt_fenced:{field}" in _reasons(
            lambda: contract.validate_receipt(receipt, descriptor=descriptor, execution_name=EXECUTION)
        ), field
    assert "remote_cpu_receipt_fenced:execution_name" in _reasons(
        lambda: contract.validate_receipt(
            _receipt(descriptor), descriptor=descriptor, execution_name="blueprint-remote-cpu-episode-compilation-other"
        )
    )
    assert "remote_cpu_receipt_fenced:attempt_id" in _reasons(
        lambda: contract.validate_receipt(_receipt(descriptor), descriptor=other_attempt, execution_name=EXECUTION)
    )
    unsealed = _receipt(descriptor)
    unsealed["bytes_uploaded"] += 1
    assert "remote_cpu_receipt_digest_mismatch" in _reasons(
        lambda: contract.validate_receipt(unsealed, descriptor=descriptor, execution_name=EXECUTION)
    )
    result = _result(descriptor)
    result["compilation_id"] = "prep-2"
    assert "remote_cpu_receipt_result_digest_mismatch" in _reasons(
        lambda: contract.validate_receipt(
            _receipt(descriptor, result=result), descriptor=descriptor, execution_name=EXECUTION
        )
    )
    wide = _output()
    wide["paths_total"] = 20001
    assert "remote_cpu_receipt_output_exceeds_limits" in _reasons(
        lambda: contract.validate_receipt(_receipt(descriptor, output=wide), descriptor=descriptor, execution_name=EXECUTION)
    )


def test_blocked_receipt_with_release_path_misses_is_not_terminal() -> None:
    descriptor = _descriptor()

    def verdict(**changes) -> dict:
        return contract.validate_receipt(_receipt(descriptor, **changes), descriptor=descriptor, execution_name=EXECUTION)

    blocked = verdict(status="blocked", result=_result(descriptor, blocker="episode_compilation_request_invalid"))
    assert blocked["outcome"] == "blocked" and blocked["terminal"] is True
    missed = verdict(
        status="blocked",
        result=_result(descriptor, blocker="launch_preparation_schema_unavailable"),
        release_path_misses=["docs/schemas/task_evaluation_launch_preparation.schema.json"],
    )
    assert missed["outcome"] == "infrastructure_failed" and missed["terminal"] is False
    assert missed["infrastructure_failures"] == [
        "infrastructure_failed:release_path_missing:docs/schemas/task_evaluation_launch_preparation.schema.json"
    ]
    for blocker in ("episode_compilation_failed:OSError:errno_28", "episode_compilation_failed:MemoryError"):
        retry = verdict(status="blocked", result=_result(descriptor, blocker=blocker))
        assert retry["outcome"] == "infrastructure_failed" and retry["terminal"] is False, blocker
    succeeded_with_miss = verdict(release_path_misses=["docs/schemas/x.schema.json"])
    assert succeeded_with_miss["terminal"] is False
    drifted = verdict(environment={"environment_digest": "sha256:" + "0" * 64, "cpu_class": CPU_CLASS})
    assert drifted["infrastructure_failures"] == ["infrastructure_failed:environment_mismatch"]
    timed_out = verdict(status="infrastructure_failed", infrastructure_failures=["infrastructure_failed:phase_deadline:stage"])
    assert timed_out["terminal"] is False

    inline = _descriptor(closure={"class": "absent_inline_only", "source_appearance_digest": "sha256:" + "8" * 64})
    unqualified = contract.validate_receipt(
        _receipt(inline, environment={"environment_digest": inline["code"]["environment_digest"], "cpu_class": "sha256:" + "9" * 64}),
        descriptor=inline,
        execution_name=EXECUTION,
    )
    assert unqualified["infrastructure_failures"] == ["infrastructure_failed:cpu_class_unqualified"]
    assert "remote_cpu_receipt_infrastructure_failure_unnamed" in _reasons(
        lambda: verdict(status="infrastructure_failed")
    )
    assert "remote_cpu_field_invalid:release_path_misses[0]" in _reasons(
        lambda: verdict(status="blocked", result=_result(descriptor, blocker="x"), release_path_misses=["../etc/passwd"])
    )


def _compute(**changes) -> dict:
    compute = {
        "execution_completed": True,
        "running_count": 0,
        "listing_complete": True,
        "listing_pages": 3,
        "executions_for_attempt": 1,
        "unfinished_executions_for_attempt": 0,
        "transport_deleted": True,
        "transport_absent_at_generation": True,
    }
    compute.update(changes)
    return compute


def _provider(**changes) -> dict:
    provider = {
        "staging_versions_deleted": 4,
        "staging_versions_remaining": 0,
        "staging_listing_complete": True,
        "urls_expire_at_epoch": 1_999_999_000.0,
        "named_objects_absent": False,
    }
    provider.update(changes)
    return provider


def _pointer_fields(descriptor: dict) -> dict:
    return {
        "stage": descriptor["stage"],
        "compilation_id": "prep-1",
        "queue_row": dict(descriptor["queue_row"]),
        "attempt_id": descriptor["attempt_id"],
        "execution": {
            "provider": "gcp_cloud_run_job",
            "job": descriptor["execution"]["job"],
            "worker_identity": contract.worker_identity_for(descriptor["execution"], EXECUTION),
            "allocation_binding_digest": "sha256:" + "a" * 64,
            "spend_consumption": "sha256:" + "b" * 64,
        },
        "descriptor_digest": descriptor["descriptor_digest"],
        "receipt_digest": "sha256:" + "c" * 64,
        "code": {
            "source_commit": descriptor["code"]["source_commit"],
            "source_archive_digest": descriptor["code"]["source_archive"]["digest"],
            "image": descriptor["code"]["image"],
            "environment_digest": descriptor["code"]["environment_digest"],
        },
        "archive": {"uri": f"{CAS}/remote-cpu-output/sha256/{'7' * 64}/blobs.tar", "digest": "sha256:" + "7" * 64, "size_bytes": 10240},
        "index": {"uri": f"{CAS}/remote-cpu-output/sha256/{'6' * 64}/index.json", "digest": "sha256:" + "6" * 64, "size_bytes": 900},
        "output_root": descriptor["outputs"]["output_root"],
        "paths_total": 12,
        "bytes_total": 50000,
        "host_known": {"count": 8, "bytes": 40000},
        "landed": {"subset": "episode_compilation_consumer.v1", "paths": 3, "bytes": 2000},
        "state": "landed",
    }


def test_pointer_record_is_resealed_on_state_change_and_never_drops_fields() -> None:
    descriptor = _descriptor()
    identity = contract.worker_identity_for(descriptor["execution"], EXECUTION)
    assert identity == f"gcp-cloud-run:blueprint-8c1ca/us-central1/blueprint-remote-cpu-episode-compilation/executions/{EXECUTION}"

    landed = contract.pointer_record(_pointer_fields(descriptor))
    assert landed["schema_version"] == "remote_cpu_output_pointer.v1"
    assert landed["teardown_receipt_digest"] is None and landed["provider_zero_proven"] is False
    assert landed["pointer_digest"] == canonical_digest(landed, digest_field="pointer_digest")

    unproven = contract.teardown_record(
        descriptor=descriptor, worker_identity=identity, outcome="completed",
        compute=_compute(), provider=_provider(), observed_at_epoch=1_999_998_000.0,
    )
    assert unproven["compute_zero_proven"] is True and unproven["provider_zero_proven"] is False
    assert "remote_cpu_pointer_teardown_not_provider_zero" in _reasons(
        lambda: contract.pointer_record({}, previous=landed, teardown=unproven)
    )
    still_running = contract.teardown_record(
        descriptor=descriptor, worker_identity=identity, outcome="completed",
        compute=_compute(unfinished_executions_for_attempt=1), provider=_provider(), observed_at_epoch=2_000_000_000.0,
    )
    assert still_running["compute_zero_proven"] is False and still_running["provider_zero_proven"] is False
    hidden_only = contract.teardown_record(
        descriptor=descriptor, worker_identity=identity, outcome="completed",
        compute=_compute(), provider=_provider(staging_versions_remaining=2), observed_at_epoch=2_000_000_000.0,
    )
    assert hidden_only["provider_zero_proven"] is False
    teardown = contract.teardown_record(
        descriptor=descriptor, worker_identity=identity, outcome="completed",
        compute=_compute(), provider=_provider(), observed_at_epoch=2_000_000_000.0,
    )
    assert teardown["provider_zero_proven"] is True

    proven = contract.pointer_record({}, previous=landed, teardown=teardown)
    assert proven["teardown_receipt_digest"] == teardown["teardown_digest"]
    assert proven["provider_zero_proven"] is True and proven["state"] == "landed"
    assert proven["pointer_digest"] != landed["pointer_digest"]
    restored = contract.pointer_record({"state": "restored_full"}, previous=proven)
    assert restored["state"] == "restored_full"
    assert restored["pointer_digest"] == canonical_digest(restored, digest_field="pointer_digest")
    for field, value in proven.items():
        if field not in {"state", "pointer_digest"}:
            assert restored[field] == value, field

    assert "remote_cpu_pointer_state_regression" in _reasons(
        lambda: contract.pointer_record({"state": "landed"}, previous=restored)
    )
    assert "remote_cpu_pointer_field_immutable:paths_total" in _reasons(
        lambda: contract.pointer_record({"paths_total": 13}, previous=landed)
    )
    assert "remote_cpu_pointer_field_immutable:teardown_receipt_digest" in _reasons(
        lambda: contract.pointer_record({"teardown_receipt_digest": None}, previous=proven)
    )
    dropped = dict(landed)
    del dropped["host_known"]
    _seal(dropped, "pointer_digest")
    assert "remote_cpu_field_missing:host_known" in _reasons(
        lambda: contract.pointer_record({"state": "restored_full"}, previous=dropped)
    )
    forged = dict(landed, provider_zero_proven=True)
    _seal(forged, "pointer_digest")
    assert "remote_cpu_pointer_provider_zero_unbound" in _reasons(
        lambda: contract.pointer_record({"state": "restored_full"}, previous=forged)
    )
    assert "remote_cpu_pointer_initial_state_invalid" in _reasons(
        lambda: contract.pointer_record({**_pointer_fields(descriptor), "state": "restored_full"})
    )
    other = _descriptor(attempt=2, nonce="f" * 32)
    foreign = contract.teardown_record(
        descriptor=other, worker_identity=identity, outcome="completed",
        compute=_compute(), provider=_provider(), observed_at_epoch=2_000_000_000.0,
    )
    assert "remote_cpu_pointer_teardown_attempt_mismatch" in _reasons(
        lambda: contract.pointer_record({}, previous=landed, teardown=foreign)
    )
