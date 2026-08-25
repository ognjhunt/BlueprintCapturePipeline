from __future__ import annotations

import hashlib
import json
import os
import pwd
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_launch_activation_worker as worker
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_activation_contract import (
    launch_activation_intent_digest,
)
from blueprint_pipeline.task_evaluation_launch_activation_queue import (
    launch_activation_status,
    stage_launch_activation_request,
)
from blueprint_pipeline.task_evaluation_launch_activation_worker import (
    TaskEvaluationLaunchActivationWorkerError,
    process_launch_activation_queue,
    validate_release_window_uri,
)
from blueprint_pipeline.task_evaluation_launch_preparation_contract import (
    launch_preparation_request_digest,
)
from blueprint_pipeline.task_evaluation_launch_preparation_queue import (
    stage_launch_preparation_request,
    write_launch_preparation_record_exclusive,
)
from tests.test_task_evaluation_launch_activation_contract import (
    request as activation_request,
)
from tests.test_task_evaluation_launch_preparation_contract import (
    request as preparation_request,
)
from tests.test_task_evaluation_configured_scene_revision import revision


SERVICE_ACCOUNT = pwd.getpwuid(os.geteuid()).pw_name


def test_release_window_must_use_coordinator_owned_prefix() -> None:
    prefix = "s3://blueprint-production-inputs/coordinator-release-windows/"
    assert validate_release_window_uri(
        prefix + "window.json", prefix=prefix
    ) == prefix
    with pytest.raises(
        TaskEvaluationLaunchActivationWorkerError,
        match="launch_activation_release_window_prefix_not_authorized",
    ):
        validate_release_window_uri(
            "s3://blueprint-production-inputs/team-a/forged-window.json",
            prefix=prefix,
        )


def test_configured_revision_rights_substitution_blocks_before_activation(
    tmp_path: Path,
) -> None:
    preparation, result, _payloads, queue, input_root = (
        _stage_verified_preparation(tmp_path)
    )
    result_path = next((queue / "results").glob("*.json"))
    sealed = json.loads(result_path.read_text(encoding="utf-8"))
    evidence = next(
        row
        for row in sealed["references"]
        if row["contract_path"]
        == "scene.configured_revision.source.rights_evidence.0.artifact"
    )
    replacement = b"substituted-rights-evidence"
    Path(evidence["materialized_path"]).write_bytes(replacement)
    evidence["digest"] = _bytes_digest(replacement)
    evidence["size_bytes"] = len(replacement)
    sealed["result_digest"] = canonical_digest(
        sealed, digest_field="result_digest"
    )
    result_path.chmod(0o640)
    result_path.write_text(json.dumps(sealed), encoding="utf-8")
    activation_request = {
        "preparation": {
            "preparation_id": preparation["preparation_id"],
            "request_digest": launch_preparation_request_digest(preparation),
            "result_digest": sealed["result_digest"],
        },
        "team_namespace": preparation["team_namespace"],
        "expected_production_commit": preparation[
            "expected_production_commit"
        ],
    }

    with pytest.raises(
        TaskEvaluationLaunchActivationWorkerError,
        match="launch_activation_preparation_reference_invalid",
    ):
        worker._load_verified_preparation(
            activation_request=activation_request,
            preparation_queue_root=queue,
            preparation_input_root=input_root,
        )


def _bytes_digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _reference(uri: str, payload: bytes) -> dict[str, object]:
    return {"uri": uri, "digest": _bytes_digest(payload), "size_bytes": len(payload)}


def _sealed_claim(schema: str, status: str, scene_id: str, field: str) -> bytes:
    value = {
        "schema_version": schema,
        "status": status,
        "scene_id": scene_id,
        field: "",
    }
    value[field] = canonical_digest(value, digest_field=field)
    return json.dumps(value, sort_keys=True).encode()


def _stage_verified_preparation(tmp_path: Path):
    request = preparation_request()
    request["preparation_id"] = "preparation-scene-841007-v1"
    request["run_id"] = "run-scene-841007-v1"
    payloads: dict[str, bytes] = {}

    def replace(node) -> None:
        if isinstance(node, dict):
            if set(node) == {"uri", "digest", "size_bytes"}:
                payload = ("payload:" + str(node["uri"])).encode()
                node.update(_reference(str(node["uri"]), payload))
                payloads[str(node["uri"])] = payload
                return
            for child in node.values():
                replace(child)
        elif isinstance(node, list):
            for child in node:
                replace(child)

    replace(request)
    scene_id = request["scene"]["identity"]["id"]
    configured = revision()
    configured["configuration_run_id"] = "configuration-scene-841007-v1"
    configured["team_namespace"] = request["team_namespace"]
    configured["scene_identity"] = request["scene"]["identity"]
    configured["source_commit"] = request["expected_production_commit"]
    configured["task_template"]["identity"] = request["task"]["identity"]
    configured["replacement"]["identity"] = request["task"]["subject"][
        "identity"
    ]
    source_bytes = _sealed_claim(
        "task_evaluation_scene_source_manifest.v1",
        "retained",
        scene_id,
        "source_manifest_digest",
    )
    rights_bytes = _sealed_claim(
        "task_evaluation_scene_rights_admission.v1",
        "admitted",
        scene_id,
        "rights_admission_digest",
    )
    for field, payload in (("manifest", source_bytes), ("rights_admission", rights_bytes)):
        reference = configured["source"][field]
        payloads[str(reference["uri"])] = payload
        reference.update(_reference(str(reference["uri"]), payload))
    for index, evidence in enumerate(configured["source"]["rights_evidence"]):
        payload = f"rights-evidence:{index}:{evidence['role']}".encode()
        reference = evidence["artifact"]
        payloads[str(reference["uri"])] = payload
        reference.update(_reference(str(reference["uri"]), payload))
    bundle_payload = b"configured-scene-bundle"
    configured["configured_scene_bundle"].update(
        _reference(
            str(configured["configured_scene_bundle"]["uri"]),
            bundle_payload,
        )
    )
    payloads[str(configured["configured_scene_bundle"]["uri"])] = bundle_payload
    configured["revision_digest"] = canonical_digest(
        configured, digest_field="revision_digest"
    )
    configured_bytes = json.dumps(configured, sort_keys=True).encode()
    request["scene"]["configured_revision"].update(
        _reference(
            str(request["scene"]["configured_revision"]["uri"]),
            configured_bytes,
        )
    )
    request["task"]["configured_scene_revision_digest"] = configured[
        "revision_digest"
    ]
    payloads[str(request["scene"]["configured_revision"]["uri"])] = (
        configured_bytes
    )

    queue = tmp_path / "preparation-queue"
    intake = stage_launch_preparation_request(
        value=request, queue_root=queue, submitted_by="blueprint-webapp"
    )
    filename = (
        f"{request['preparation_id']}-"
        f"{intake['request_digest'].removeprefix('sha256:')}.json"
    )
    os.replace(queue / "pending" / filename, queue / "materialized" / filename)

    input_root = tmp_path / "preparation-inputs"
    preparation_root = input_root / request["preparation_id"]
    preparation_root.mkdir(parents=True)
    reference_rows: list[dict[str, object]] = []

    def materialize(node, path: tuple[str, ...] = ()) -> None:
        if isinstance(node, dict):
            if set(node) == {"uri", "digest", "size_bytes"}:
                target = preparation_root / str(node["digest"]).removeprefix("sha256:")
                target.write_bytes(payloads[str(node["uri"])])
                reference_rows.append({
                    "contract_path": ".".join(path),
                    "uri": node["uri"],
                    "digest": node["digest"],
                    "size_bytes": node["size_bytes"],
                    "materialized_path": str(target),
                    "full_byte_service_account_readback_passed": True,
                })
                return
            for key, child in node.items():
                materialize(child, (*path, str(key)))
        elif isinstance(node, list):
            for index, child in enumerate(node):
                materialize(child, (*path, str(index)))

    materialize(request)
    transitive_references = [
        (
            "scene.configured_revision.configured_scene_bundle",
            configured["configured_scene_bundle"],
        ),
        (
            "scene.configured_revision.source.manifest",
            configured["source"]["manifest"],
        ),
        (
            "scene.configured_revision.source.rights_admission",
            configured["source"]["rights_admission"],
        ),
        *[
            (
                "scene.configured_revision.source.rights_evidence."
                f"{index}.artifact",
                evidence["artifact"],
            )
            for index, evidence in enumerate(
                configured["source"]["rights_evidence"]
            )
        ],
    ]
    for contract_path, reference in transitive_references:
        target = preparation_root / str(reference["digest"]).removeprefix(
            "sha256:"
        )
        target.write_bytes(payloads[str(reference["uri"])])
        reference_rows.append(
            {
                "contract_path": contract_path,
                "uri": reference["uri"],
                "digest": reference["digest"],
                "size_bytes": reference["size_bytes"],
                "materialized_path": str(target),
                "full_byte_service_account_readback_passed": True,
            }
        )
    adapter_root = input_root / request["preparation_id"] / "native-arena-adapter"
    packet_root = adapter_root / "construction-packet"
    runtime_root = adapter_root / "runtime-source"
    packet_root.mkdir(parents=True)
    runtime_root.mkdir()
    robot = {"robot_id": request["robot"]["identity"]["id"], "joint_count": 7}
    (packet_root / "native_task_runtime_contract.v1.json").write_text(
        json.dumps({"task_spec_digest": "sha256:" + "8" * 64, "robot": robot}),
        encoding="utf-8",
    )
    runtime_receipt = runtime_root / "native_task_runtime_source_packet.v1.json"
    runtime_receipt.write_text("{}\n", encoding="utf-8")
    adapter = {
        "schema_version": "task_evaluation_native_arena_adapter_result.v1",
        "status": "native_arena_adapter_materialized",
        "preparation_id": request["preparation_id"],
        "source_commit": request["expected_production_commit"],
        "packet_receipt_digest": "sha256:" + "6" * 64,
        "runtime_source_receipt_digest": "sha256:" + "7" * 64,
        "packet_root": str(packet_root),
        "runtime_source_receipt": str(runtime_receipt),
        "result_digest": "",
    }
    adapter["result_digest"] = canonical_digest(adapter, digest_field="result_digest")
    write_launch_preparation_record_exclusive(
        adapter_root / "task_evaluation_native_arena_adapter_result.v1.json", adapter
    )
    result = {
        "schema_version": "task_evaluation_launch_preparation_result.v1",
        "status": "native_arena_inputs_verified_awaiting_profile_authority",
        "preparation_id": request["preparation_id"],
        "run_id": request["run_id"],
        "team_namespace": request["team_namespace"],
        "source_commit": request["expected_production_commit"],
        "full_byte_service_account_readback_passed": True,
        "references": reference_rows,
        "adapter_result_digest": adapter["result_digest"],
        "provider_mutation_performed": False,
        "catalog_mutation_performed": False,
        "paid_execution_requested": False,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    (queue / "results").mkdir()
    write_launch_preparation_record_exclusive(queue / "results" / filename, result)
    assert intake["request_digest"] == launch_preparation_request_digest(request)
    return request, result, payloads, queue, input_root


def _release_window(request: dict[str, object], now: datetime) -> bytes:
    value = {
        "schema_version": "task_evaluation_shared_mutation_window.v1",
        "status": "released",
        "window_id": "window-scene-841007-001",
        "activation_id": request["activation_id"],
        "activation_intent_digest": launch_activation_intent_digest(request),
        "team_namespace": request["team_namespace"],
        "expected_production_commit": request["expected_production_commit"],
        "allowed_mutations": [
            "profile_publication",
            "catalog_synchronization",
            "standing_authorization",
        ],
        "provider_allowlist": ["vast"],
        "maximum_hard_cap_usd": 1.0,
        "issued_at": (now - timedelta(minutes=1)).isoformat(),
        "expires_at": (now + timedelta(minutes=10)).isoformat(),
        "released_by": "policy-lead-001",
        "release_reference": "coordinated release window",
        "provider_resource_allocation_allowed": False,
        "paid_request_allowed": False,
        "window_digest": "",
    }
    value["window_digest"] = canonical_digest(value, digest_field="window_digest")
    return json.dumps(value, sort_keys=True).encode()


def test_worker_cross_binds_preparation_window_and_no_execution_publication(tmp_path) -> None:
    preparation, preparation_result, payloads, preparation_queue, input_root = (
        _stage_verified_preparation(tmp_path)
    )
    request = activation_request()
    request["activation_id"] = "activation-scene-841007-construction"
    request["preparation"] = {
        "preparation_id": preparation["preparation_id"],
        "request_digest": launch_preparation_request_digest(preparation),
        "result_digest": preparation_result["result_digest"],
    }
    request["authorization"]["authorized_on"] = datetime.now(timezone.utc).isoformat()
    request["authorization"]["standing_authorization_expires_at"] = (
        datetime.now(timezone.utc) + timedelta(hours=1)
    ).isoformat()
    for name, reference in request["lineage"].items():
        if name == "kind":
            continue
        content = json.dumps({"kind": name}).encode()
        reference.update(_reference(str(reference["uri"]), content))
        payloads[str(reference["uri"])] = content
    window_bytes = _release_window(request, datetime.now(timezone.utc))
    request["release_window"] = _reference(
        "s3://blueprint-production-inputs/coordinator-release-windows/release-window.json",
        window_bytes,
    )
    payloads[request["release_window"]["uri"]] = window_bytes

    queue = tmp_path / "activation-queue"
    stage_launch_activation_request(
        value=request, queue_root=queue, submitted_by="blueprint-webapp"
    )
    profile_dir = tmp_path / "profiles"
    authorization_dir = tmp_path / "standing-authorizations"

    def fetch(uri: str, destination: Path, maximum_bytes: int) -> None:
        assert len(payloads[uri]) == maximum_bytes
        destination.write_bytes(payloads[uri])

    observed_context: dict[str, object] = {}

    def prepare(**kwargs) -> dict[str, object]:
        context = json.loads(Path(kwargs["context_path"]).read_text())
        observed_context.update(context)
        set_root = Path(context["operations"]["set_root"])
        set_root.mkdir(parents=True)
        profile = {
            "profile_id": "scene-841007-construction-r1",
            "profile_digest": "sha256:" + "d" * 64,
        }
        profile_path = set_root / "live_profile-r1.v1.json"
        profile_path.write_text(json.dumps(profile), encoding="utf-8")
        publication_path = set_root / "profile_publication_receipt.v1.json"
        publication_path.write_text('{"status":"published"}\n', encoding="utf-8")
        authorization_dir.mkdir(parents=True)
        authorization_path = authorization_dir / f"{profile['profile_id']}.json"
        authorization_path.write_text(
            json.dumps({
                "profile_id": profile["profile_id"],
                "profile_digest": profile["profile_digest"],
                "provider_mutation_performed": False,
            }),
            encoding="utf-8",
        )
        steps = []
        for step_id, path in (
            ("live_profile", profile_path),
            ("profile_publication", publication_path),
            ("standing_authorization", authorization_path),
        ):
            steps.append({
                "step_id": step_id,
                "artifact_path": str(path),
                "artifact_sha256": _bytes_digest(path.read_bytes()),
            })
        receipt = {
            "schema_version": "paid_lane_launch_preparation.v1",
            "status": "prepared",
            "source_commit": request["expected_production_commit"],
            "completed_steps": steps,
            "provider_allocation_performed": False,
            "paid_inference_performed": False,
        }
        Path(kwargs["receipt_path"]).write_text(json.dumps(receipt), encoding="utf-8")
        return receipt

    run = process_launch_activation_queue(
        queue_root=queue,
        preparation_queue_root=preparation_queue,
        preparation_input_root=input_root,
        activation_root=tmp_path / "activations",
        allowed_uri_prefixes=["s3://blueprint-production-inputs/"],
        service_account=SERVICE_ACCOUNT,
        service_group=pwd.getpwuid(os.geteuid()).pw_name,
        repository_root=tmp_path,
        destination_prefix="s3://blueprint-production-inputs/activated",
        release_window_prefix=(
            "s3://blueprint-production-inputs/coordinator-release-windows/"
        ),
        profile_dir=profile_dir,
        webapp_catalog=tmp_path / "catalog.json",
        standing_authorization_dir=authorization_dir,
        source_commit=request["expected_production_commit"],
        fetcher=fetch,
        preparer=prepare,
    )

    assert run["status"] == "processed"
    assert run["results"][0]["status"] == "profile_authority_materialized_no_execution", json.dumps(run, sort_keys=True)
    assert run["results"][0]["provider_mutation_performed"] is False
    assert run["results"][0]["paid_execution_requested"] is False
    assert observed_context["lane"] == "native_task_arena_construction"
    assert observed_context["operations"]["provider"] == "vast"
    assert observed_context["references"]["scene"]["scene_id"] == (
        preparation["scene"]["identity"]["id"]
    )
    status = launch_activation_status(
        activation_id=request["activation_id"], queue_root=queue
    )
    assert status["status"] == "prepared"
    assert status["profile_id"] == "scene-841007-construction-r1"
    assert status["provider_mutation_performed_by_worker"] is False
    assert status["paid_execution_requested"] is False


def test_worker_blocks_wrong_window_before_preparer(tmp_path) -> None:
    preparation, preparation_result, payloads, preparation_queue, input_root = (
        _stage_verified_preparation(tmp_path)
    )
    request = activation_request()
    request["activation_id"] = "activation-scene-841007-construction"
    request["preparation"] = {
        "preparation_id": preparation["preparation_id"],
        "request_digest": launch_preparation_request_digest(preparation),
        "result_digest": preparation_result["result_digest"],
    }
    for name, reference in request["lineage"].items():
        if name != "kind":
            content = json.dumps({"kind": name}).encode()
            reference.update(_reference(str(reference["uri"]), content))
            payloads[str(reference["uri"])] = content
    window_bytes = _release_window(request, datetime.now(timezone.utc))
    window = json.loads(window_bytes)
    window["activation_id"] = "different-activation"
    window["window_digest"] = canonical_digest(window, digest_field="window_digest")
    window_bytes = json.dumps(window, sort_keys=True).encode()
    request["release_window"] = _reference(
        "s3://blueprint-production-inputs/coordinator-release-windows/release-window.json",
        window_bytes,
    )
    payloads[request["release_window"]["uri"]] = window_bytes
    queue = tmp_path / "activation-queue"
    stage_launch_activation_request(
        value=request, queue_root=queue, submitted_by="blueprint-webapp"
    )

    def fetch(uri: str, destination: Path, maximum_bytes: int) -> None:
        destination.write_bytes(payloads[uri])

    def forbidden_preparer(**_kwargs):
        raise AssertionError("invalid release window must block before mutation")

    run = process_launch_activation_queue(
        queue_root=queue,
        preparation_queue_root=preparation_queue,
        preparation_input_root=input_root,
        activation_root=tmp_path / "activations",
        allowed_uri_prefixes=["s3://blueprint-production-inputs/"],
        service_account=SERVICE_ACCOUNT,
        service_group=SERVICE_ACCOUNT,
        repository_root=tmp_path,
        destination_prefix="s3://blueprint-production-inputs/activated",
        release_window_prefix=(
            "s3://blueprint-production-inputs/coordinator-release-windows/"
        ),
        profile_dir=tmp_path / "profiles",
        webapp_catalog=tmp_path / "catalog.json",
        standing_authorization_dir=tmp_path / "standing-authorizations",
        source_commit=request["expected_production_commit"],
        fetcher=fetch,
        preparer=forbidden_preparer,
    )
    assert run["results"][0]["status"] == "blocked"
    assert run["results"][0]["blockers"] == [
        "shared_mutation_window_binding_mismatch"
    ]
