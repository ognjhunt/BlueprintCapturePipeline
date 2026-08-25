from __future__ import annotations

import hashlib
import json
import os
import pwd
import sys
import types
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_preparation_queue import (
    launch_preparation_status,
    stage_launch_preparation_request,
)
from blueprint_pipeline.task_evaluation_launch_preparation_worker import (
    TaskEvaluationLaunchPreparationWorkerError,
    default_reference_fetcher,
    main,
    materialize_preparation_references,
    process_launch_preparation_queue,
    validate_allowed_uri_prefixes,
)
from tests.test_task_evaluation_launch_preparation_contract import request


SERVICE_ACCOUNT = pwd.getpwuid(os.geteuid()).pw_name


def request_with_fetchable_bytes() -> tuple[dict[str, object], dict[str, bytes]]:
    value = request()
    payloads: dict[str, bytes] = {}

    def replace(node) -> None:
        if isinstance(node, dict):
            if set(node) == {"uri", "digest", "size_bytes"}:
                payload = ("payload-for-" + node["uri"]).encode()
                node["digest"] = "sha256:" + hashlib.sha256(payload).hexdigest()
                node["size_bytes"] = len(payload)
                payloads[node["uri"]] = payload
                return
            for child in node.values():
                replace(child)
        elif isinstance(node, list):
            for child in node:
                replace(child)

    replace(value)
    return value, payloads


def fetcher(payloads: dict[str, bytes]):
    def fetch(uri: str, destination: Path, maximum_bytes: int) -> None:
        assert len(payloads[uri]) <= maximum_bytes
        destination.write_bytes(payloads[uri])

    return fetch


def test_materializes_every_reference_and_full_byte_reads_back(tmp_path) -> None:
    value, payloads = request_with_fetchable_bytes()
    result = materialize_preparation_references(
        request=value,
        input_root=tmp_path / "inputs",
        allowed_uri_prefixes=["s3://blueprint-production-inputs/"],
        service_account=SERVICE_ACCOUNT,
        fetcher=fetcher(payloads),
    )
    assert result["status"] == "inputs_materialized_awaiting_construction_adapter"
    assert result["reference_count"] == len(payloads)
    assert result["full_byte_service_account_readback_passed"] is True
    assert result["provider_mutation_performed"] is False
    assert result["catalog_mutation_performed"] is False
    assert result["result_digest"] == canonical_digest(
        result, digest_field="result_digest"
    )
    for row in result["references"]:
        path = Path(row["materialized_path"])
        assert path.read_bytes() == payloads[row["uri"]]
        assert path.stat().st_mode & 0o777 == 0o440


def test_worker_claims_queue_and_seals_terminal_no_spend_result(tmp_path) -> None:
    value, payloads = request_with_fetchable_bytes()
    queue = tmp_path / "queue"
    intake = stage_launch_preparation_request(
        value=value, queue_root=queue, submitted_by="blueprint-webapp"
    )
    run = process_launch_preparation_queue(
        queue_root=queue,
        input_root=tmp_path / "inputs",
        allowed_uri_prefixes=["s3://blueprint-production-inputs/"],
        service_account=SERVICE_ACCOUNT,
        fetcher=fetcher(payloads),
    )
    assert run["status"] == "processed"
    assert run["processed_count"] == 1
    assert run["provider_mutation_performed"] is False
    assert launch_preparation_status(
        preparation_id=value["preparation_id"], queue_root=queue
    ) == {
        "schema_version": "task_evaluation_launch_preparation_status.v1",
        "status": "materialized",
        "preparation_id": value["preparation_id"],
        "run_id": value["run_id"],
        "team_namespace": value["team_namespace"],
        "request_digest": intake["request_digest"],
        "provider_mutation_performed_by_status_read": False,
        "worker_status": "inputs_materialized_awaiting_construction_adapter",
        "result_digest": run["results"][0]["result_digest"],
        "reference_count": len(payloads),
        "full_byte_service_account_readback_passed": True,
        "blockers": [],
        "provider_mutation_performed_by_worker": False,
        "catalog_mutation_performed_by_worker": False,
        "paid_execution_requested": False,
    }
    result_files = list((queue / "results").glob("*.json"))
    assert len(result_files) == 1
    sealed = json.loads(result_files[0].read_text())
    assert sealed["result_digest"] == canonical_digest(
        sealed, digest_field="result_digest"
    )


def test_worker_blocks_unapproved_storage_prefix_before_fetch(tmp_path) -> None:
    value, payloads = request_with_fetchable_bytes()
    queue = tmp_path / "queue"
    stage_launch_preparation_request(
        value=value, queue_root=queue, submitted_by="blueprint-webapp"
    )

    def forbidden_fetch(*_args):
        raise AssertionError("unapproved prefix must block before fetch")

    run = process_launch_preparation_queue(
        queue_root=queue,
        input_root=tmp_path / "inputs",
        allowed_uri_prefixes=["s3://different-team/"],
        service_account=SERVICE_ACCOUNT,
        fetcher=forbidden_fetch,
    )
    assert run["results"][0]["status"] == "blocked"
    assert run["results"][0]["blockers"] == [
        "launch_preparation_reference_prefix_not_allowed"
    ]
    assert run["results"][0]["preparation_id"] == value["preparation_id"]
    assert launch_preparation_status(
        preparation_id=value["preparation_id"], queue_root=queue
    )["status"] == "blocked"


def test_worker_cli_fails_closed_without_production_configuration(capsys) -> None:
    assert main([]) == 2
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "blocked"
    assert receipt["blockers"] == [
        "launch_preparation_worker_configuration_invalid"
    ]
    assert receipt["provider_mutation_performed"] is False


@pytest.mark.parametrize(
    "prefix",
    [
        "s3://blueprint-production-inputs",
        "s3://user@example.test/team/",
        "file:///var/lib/blueprint/",
        "https://example.test/team/?token=secret",
    ],
)
def test_operator_storage_prefixes_fail_closed_at_directory_boundaries(
    prefix: str,
) -> None:
    with pytest.raises(
        TaskEvaluationLaunchPreparationWorkerError,
        match="launch_preparation_allowed_uri_prefix_invalid",
    ):
        validate_allowed_uri_prefixes([prefix])


def test_s3_fetch_uses_canonical_private_secret_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = {
        "BLUEPRINT_WAM_OBJECT_STORE_ACCESS_KEY_ID_FILE": "access",
        "BLUEPRINT_WAM_OBJECT_STORE_SECRET_ACCESS_KEY_FILE": "secret",
        "BLUEPRINT_WAM_OBJECT_STORE_ENDPOINT_URL_FILE": "https://objects.example.test",
        "BLUEPRINT_WAM_OBJECT_STORE_REGION_FILE": "region-1",
    }
    for environment_name, value in values.items():
        path = tmp_path / environment_name.lower()
        path.write_text(value + "\n", encoding="utf-8")
        path.chmod(0o600)
        monkeypatch.setenv(environment_name, str(path))

    observed: dict[str, object] = {}

    class Body:
        def __init__(self) -> None:
            self.payload = bytearray(b"payload")

        def read(self, count: int) -> bytes:
            chunk = bytes(self.payload[:count])
            del self.payload[:count]
            return chunk

    class Client:
        def get_object(self, **kwargs):
            observed["request"] = kwargs
            return {"ContentLength": 7, "Body": Body()}

    def client(name: str, **kwargs):
        observed["name"] = name
        observed["kwargs"] = kwargs
        return Client()

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=client))
    destination = tmp_path / "download"
    default_reference_fetcher(
        "s3://blueprint-production-inputs/object.bin", destination, 7
    )

    assert destination.read_bytes() == b"payload"
    assert observed == {
        "name": "s3",
        "kwargs": {
            "aws_access_key_id": "access",
            "aws_secret_access_key": "secret",
            "endpoint_url": "https://objects.example.test",
            "region_name": "region-1",
        },
        "request": {
            "Bucket": "blueprint-production-inputs",
            "Key": "object.bin",
        },
    }


def test_existing_different_result_is_preserved_and_conflict_is_sealed(
    tmp_path: Path,
) -> None:
    value, payloads = request_with_fetchable_bytes()
    queue = tmp_path / "queue"
    receipt = stage_launch_preparation_request(
        value=value, queue_root=queue, submitted_by="blueprint-webapp"
    )
    queue_name = Path(receipt["queue_path"]).name
    results = queue / "results"
    results.mkdir(mode=0o750)
    existing = {
        "schema_version": "task_evaluation_launch_preparation_result.v1",
        "status": "blocked",
        "preparation_id": value["preparation_id"],
        "blockers": ["prior-immutable-result"],
        "result_digest": "",
    }
    existing["result_digest"] = canonical_digest(
        existing, digest_field="result_digest"
    )
    (results / queue_name).write_text(
        json.dumps(existing, sort_keys=True) + "\n", encoding="utf-8"
    )

    run = process_launch_preparation_queue(
        queue_root=queue,
        input_root=tmp_path / "inputs",
        allowed_uri_prefixes=["s3://blueprint-production-inputs/"],
        service_account=SERVICE_ACCOUNT,
        fetcher=fetcher(payloads),
    )

    assert json.loads((results / queue_name).read_text()) == existing
    assert run["results"][0]["status"] == "blocked"
    assert run["results"][0]["blockers"] == [
        "launch_preparation_immutable_result_conflict"
    ]
    conflicts = list((results / "conflicts").glob("*.json"))
    assert len(conflicts) == 1
    status = launch_preparation_status(
        preparation_id=value["preparation_id"], queue_root=queue
    )
    assert status["status"] == "blocked"
    assert status["worker_status"] == "blocked"
    assert status["blockers"] == [
        "launch_preparation_immutable_result_conflict"
    ]
