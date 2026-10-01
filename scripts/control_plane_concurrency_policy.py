"""Fake external policy bytes joined to real Plan 13c ingestion and delivery.

ADP-009D/day 28. The fixture result is development-only and is never evidence
of provider execution, physical outcomes, billing, or worker qualification.
All local compilation bindings, archive publication, selective ingestion,
reservations, returned-byte checks, and result projections use production code.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlsplit

from blueprint_pipeline.control_plane_disk_budget import reserve_control_plane_disk
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.policy_canary_output_members import POLICY_CANARY_OUTPUT_CONTRACT
from blueprint_pipeline.provider_output_member_index import (
    LocalArchiveRangeSource, build_member_index, seal_durable_reference,
)
from blueprint_pipeline.provider_output_member_view import write_member_view_descriptor
from blueprint_pipeline.provider_output_range_ingestion import CasArchiveSource, ingest_selected_members
from blueprint_pipeline.task_evaluation_configured_scene_object_store import publish_configured_scene_stream
from blueprint_pipeline.task_evaluation_policy_canary_result_projection import build_policy_canary_result_projection
from blueprint_pipeline.task_evaluation_launch_preparation_contract import (
    launch_preparation_request_digest, validate_launch_preparation_request,
)
from blueprint_pipeline.task_evaluation_result_delivery import materialize_policy_canary_result_delivery
from scripts.control_plane_concurrency_fixture import FilesystemObjectStore

FIXTURE_URL = "https://storage.example.invalid/private/fixture.zip"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    return "sha256:" + digest.hexdigest()


def _compiled_contract(compiled: dict) -> dict:
    if (compiled.get("schema_version") != "task_evaluation_episode_compilation_result.v1"
            or compiled.get("status") != "compiled_for_production_launch"
            or compiled.get("compiled_by_production") is not True
            or compiled.get("provider_mutation_performed") is not False
            or compiled.get("paid_execution_requested") is not False
            or compiled.get("result_digest") != canonical_digest(compiled, digest_field="result_digest")):
        raise ValueError("harness_policy_compilation_binding_invalid")
    path = Path(compiled["compiled_episode_packet_path"])
    if (path.is_symlink() or not path.is_file()
            or path.stat().st_size != compiled["compiled_episode_packet_size_bytes"]
            or _sha(path) != compiled["compiled_episode_packet_digest"]):
        raise ValueError("harness_policy_compiled_packet_changed")
    with zipfile.ZipFile(path) as archive:
        matches = [row for row in archive.infolist()
                   if Path(row.filename).name == "native_task_runtime_contract.v1.json"]
        if len(matches) != 1 or not 0 < matches[0].file_size <= 16 * 1024**2:
            raise ValueError("harness_policy_runtime_contract_invalid")
        contract = json.loads(archive.read(matches[0]))
    if (contract.get("schema_version") != "native_task_runtime_contract.v1"
            or not isinstance(contract.get("task_spec", {}).get("task_success_contract"), dict)):
        raise ValueError("harness_policy_task_success_contract_missing")
    return contract


def prepare_policy_fixture(*, compiled: dict, preparation_request: dict,
                           object_root: Path, worker_root: Path) -> dict:
    """Publish real fixture bytes outside the control plane, preserving CPU joins."""
    contract = _compiled_contract(compiled)
    request = validate_launch_preparation_request(preparation_request)
    if (request["preparation_id"] != compiled["compilation_id"]
            or request["expected_production_commit"] != compiled["source_commit"]
            or request["scene"]["identity"]["id"] != contract["scene_id"]
            or request["task"]["configured_scene_revision_digest"] != compiled["configured_scene_revision_digest"]):
        raise ValueError("harness_policy_preparation_binding_mismatch")
    worker_root.mkdir(mode=0o700)
    evidence = worker_root / "provider-evidence"
    evidence.mkdir(mode=0o700)
    # Pure data builder for this external boundary. Never run a mocked compiler,
    # ingest/delivery function, or production qualification validator here.
    from tests.test_task_evaluation_policy_canary_result_delivery import _result
    result = _result(evidence)
    task = contract["task_spec"]["task_success_contract"]
    result.update(run_id=compiled["run_id"], configuration_digest=compiled["result_digest"],
                  scene_revision_digest=compiled["configured_scene_revision_digest"],
                  task_success_contract=task, task_success_contract_digest=task["contract_digest"],
                  official_total_usd=0.0, provider="fixture", provider_instance_ids=[],
                  fixture_provider=True, actual_provider_calls=0,
                  fixture_claim_ceiling="development_only")
    prototype = result["episodes"][0]
    video = next(row for row in result["artifact_inventory"] if row["role"] == "review_video")
    prototype["evidence_artifacts"]["review_video"] = video
    episodes = []
    for cell in range(10):
        for candidate in ("pi05_droid", "groot_n17_droid"):
            row = deepcopy(prototype)
            row.update(candidate_id=candidate, cell_id=f"quick-cell-{cell}", seed=3100 + cell,
                       scene_revision_digest=result["scene_revision_digest"], fixture_provider=True)
            row["episode"]["episode_id"] = f"fixture-{cell}-{candidate}"
            episodes.append(row)
    result["episodes"] = episodes
    result["artifact_inventory_digest"] = canonical_digest({"value": result["artifact_inventory"]})
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    (evidence / "native_task_arena_policy_canary_session_result.v1.json").write_text(json.dumps(result))
    # A real, allocated, deliberately unneeded member demonstrates selective
    # transport and disk savings without synthetic sparse-file accounting.
    with (evidence / "unneeded-provider-buffer.bin").open("xb") as stream:
        for _ in range(8):
            stream.write(bytes(1024 * 1024))
    archive_path = worker_root / "provider-output.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(evidence.iterdir()):
            archive.write(path, path.name)
    store = FilesystemObjectStore(object_root)

    def write_archive(destination):
        with archive_path.open("rb") as source:
            shutil.copyfileobj(source, destination, 1024 * 1024)

    reference = publish_configured_scene_stream(write_stream=write_archive, digest=_sha(archive_path),
        size_bytes=archive_path.stat().st_size, filename="provider-output.zip",
        artifact_kind="provider-output", client=store, bucket="fixtures")
    with LocalArchiveRangeSource(archive_path, block_bytes=128 * 1024) as source:
        index = build_member_index(source, maximum_expanded_bytes=64 * 1024**2)
    index = seal_durable_reference(index, reference)
    setup = {"scene_id": contract["scene_id"], "request_digest": launch_preparation_request_digest(request),
             "scene_revision_digest": result["scene_revision_digest"],
             "task_success_contract": task, "task_success_contract_digest": task["contract_digest"]}
    return {"fixture_provider": True, "claim_ceiling": "development_only", "actual_provider_calls": 0,
            "result": result, "setup": setup, "index": index, "archive": reference,
            "worker_publication_readback_bytes": store.read_bytes}


class _Response:
    def __init__(self, row: dict):
        self.body = row["Body"]
        self.status = row["ResponseMetadata"]["HTTPStatusCode"]
        self.headers = {"Content-Length": str(row["ContentLength"]),
                        "Content-Range": row["ContentRange"], "ETag": row["ETag"]}

    def geturl(self):
        return FIXTURE_URL

    def read(self, size=-1):
        return self.body.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.body.close()


def ingest_policy_fixture(*, provider: dict, object_root: Path, output_root: Path,
                          reservation_root: Path, on_reserved=None) -> dict:
    """Fetch only production-selected members using the actual range transport."""
    output_root.mkdir(mode=0o700)
    index, reference = provider["index"], provider["archive"]
    index_path = output_root / "provider-member-index.json"
    index_path.write_text(json.dumps(index))
    store = FilesystemObjectStore(object_root)
    location = urlsplit(reference["uri"])

    def opener(request, timeout, policy):
        if request.full_url != FIXTURE_URL or request.get_method() != "GET":
            raise ValueError("harness_fixture_transport_request_invalid")
        headers = {key.lower(): value for key, value in request.header_items()}
        return _Response(store.get_object(Bucket=location.netloc, Key=location.path.lstrip("/"),
            Range=headers.get("range"), IfMatch=headers.get("if-match")))

    source = CasArchiveSource(reference, presign=lambda: FIXTURE_URL, opener=opener,
                              block_bytes=128 * 1024)
    selection = POLICY_CANARY_OUTPUT_CONTRACT.selection(index)
    overhead = POLICY_CANARY_OUTPUT_CONTRACT.hold_bytes(needed_bytes=0,
        member_count=len(selection["members"]), index_file_bytes=index_path.stat().st_size)
    hold = None

    def reserve(needed):
        nonlocal hold
        expected = max(1, needed + overhead)
        if hold is None:
            hold = reserve_control_plane_disk("policy_canary_output", target_root=output_root,
                expected_bytes=expected, reservation_root=reservation_root)
            if on_reserved is not None:
                on_reserved(hold)
        else:
            hold.resize(expected)
        return hold

    evidence, metadata = output_root / "immutable_execution", output_root / "ingestion"
    try:
        ingestion = ingest_selected_members(source=source, index=index, selection=selection,
            members_root=evidence, metadata_root=metadata, reserve=reserve)
    finally:
        if hold is not None:
            hold.release()
    if ingestion["status"] == "materialized":
        write_member_view_descriptor(evidence_root=evidence, index_path=index_path,
                                    ingestion_receipt_path=metadata / "receipt.json")
    return {**provider, "evidence_root": evidence, "run_root": output_root,
            "ingestion": ingestion, "host_transport_bytes": store.read_bytes}


def deliver_policy_fixture(*, collected: dict) -> dict:
    """Validate actual returned bytes and project an explicitly fictional run."""
    root, result = Path(collected["run_root"]), collected["result"]
    closures = {}
    for name, flag in (("billing", "official_billing_sealed"), ("teardown", "teardown_completed"),
                       ("provider_zero", "provider_zero_verified")):
        path = root / ("fixture-" + name + ".json")
        payload = {"status": "completed", "fixture_provider": True,
                   "actual_provider_calls": 0, "claim_ceiling": "development_only"}
        if not path.exists():
            with path.open("x") as stream:
                json.dump(payload, stream)
        elif path.is_symlink() or json.loads(path.read_text()) != payload:
            raise ValueError("harness_fixture_closure_changed")
        closures[name] = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha(path), flag: True}
    delivery = materialize_policy_canary_result_delivery(run_root=root, run_id=result["run_id"],
        result_status=result["status"], session_result=result,
        evidence_root=collected["evidence_root"], closure_records=closures)
    projection = build_policy_canary_result_projection(setup=collected["setup"], result=result, delivery=delivery)
    return {"delivery": delivery, "projection": projection, "fixture_provider": True,
            "claim_ceiling": "development_only", "actual_provider_calls": 0}
