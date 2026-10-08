"""Disposable CPU bridge for the existing Node/Python signed-handler replay.

Owned synthetic JSON in; actual selected staging/birth/lease/FastAPI out; fake
object/owner/context transports. No providers, native results or production data.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--resume-only", action="store_true")
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    assert fixture["schema_version"] == "website_preparation_joined_fixture.v1"
    callback = urlsplit(fixture["callback_base_url"])
    assert callback.scheme == "http" and callback.hostname == "127.0.0.1" and callback.port
    assert not callback.username and not callback.password and callback.path in {"", "/"}
    original_connect = socket.socket.connect
    def isolated_connect(sock, address):
        assert isinstance(address, tuple) and address[0] == "127.0.0.1" and address[1] == callback.port, "Non-sandbox network refused"
        return original_connect(sock, address)
    socket.socket.connect = isolated_connect

    import pytest
    import uvicorn
    from blueprint_pipeline import capture_original_owner_observer as observer
    from blueprint_pipeline import live_pipeline_intake_service as service
    from blueprint_pipeline import pubsub_handoff_listener as listener
    from blueprint_pipeline import task_evaluation_scene_retirement_access as access
    from blueprint_pipeline import website_task_context as context_reader
    from blueprint_pipeline.common import PipelineError, StageError, write_json
    from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
    from blueprint_pipeline.website_preparation_status import read_preparation_status, reconcile_preparation_wakeups
    from blueprint_pipeline.website_scene_handoff import prepare_website_scene_handoff
    from blueprint_pipeline.webapp_sync import _pipeline_sync_headers
    from tests.test_scene_retirement_real_participants import access_fixture

    state_root = args.state_root.resolve()
    state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    monkeypatch = pytest.MonkeyPatch()
    admission = state_root / "admission"
    policy = admission / "policy.json"
    if policy.is_file():
        assert policy.stat().st_uid == os.getuid()
        stored = json.loads(policy.read_text())
        storage_root = Path(stored["roots"][0]["root"])
        assert storage_root.is_relative_to(state_root)
        monkeypatch.setenv("BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE", str(policy))
        monkeypatch.setattr(access, "_INSTALLED_POLICY", policy)
        monkeypatch.setattr(access, "_POLICY_UID", os.getuid())
        monkeypatch.setattr(access, "_SERVICE_IDENTITY", (os.getuid(), os.getgid()))
    else:
        admission.mkdir(mode=0o700)
        _, _, storage_root = access_fixture(admission, monkeypatch)
    owner = fixture["owner_observation"]
    context = fixture["task_context"]
    root = storage_root / owner["bucket"] / "scenes" / owner["scene_id"] / "captures" / owner["capture_id"]
    assert root.is_relative_to(storage_root)
    context_reader.validate_website_task_context(context, request_id=owner["request_id"], scene_id=owner["scene_id"], capture_id=owner["capture_id"])
    def fresh_owner(**kwargs):
        current = json.loads(args.fixture.read_text())["owner_observation"]
        current["observed_at_epoch"] = int(time.time())
        current["valid_until_epoch"] = current["observed_at_epoch"] + 60
        current["observation_digest"] = cross_runtime_canonical_digest(current, digest_field="observation_digest")
        return observer.validate_observation(current, **kwargs)
    monkeypatch.setattr(observer, "load_original_owner_observation", fresh_owner)
    monkeypatch.setattr(context_reader, "load_current_website_task_context", lambda **_: json.loads(args.fixture.read_text())["task_context"])

    rows = {}
    contents = {}
    prefix = f"scenes/{owner['scene_id']}/captures/{owner['capture_id']}"
    for row in fixture["objects"]:
        data = base64.b64decode(row["bytes_base64"], validate=True)
        assert row["size_bytes"] == len(data) and row["sha256"] == "sha256:" + hashlib.sha256(data).hexdigest()
        if row["object_name"].startswith(prefix + "/raw/"):
            rows[row["object_name"]] = {key: row[key] for key in ("object_name", "generation", "size_bytes", "crc32c", "sha256")}
            rows[row["object_name"]]["relative_path"] = "raw/" + row["object_name"].removeprefix(prefix + "/raw/")
            contents[row["object_name"]] = data
    assert json.loads(contents[prefix + "/raw/manifest.json"])["capture_source"] == "browser_self_capture"
    marker = owner["completion_marker"]
    source = {"bucket": owner["bucket"], "object_name": marker["object_name"], "generation": marker["generation"]}
    key = hashlib.sha256(json.dumps([source["bucket"], source["object_name"], source["generation"]], separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    producer = owner["producer_delivery"]
    membership = {"schema_version": "capture_delivery_membership.v1", "delivery_key": key, "source_finalize": source,
        "producer_delivery": {"kind": "browser", "receipt_object_name": producer["server_record"]["object_name"],
            "receipt_generation": producer["server_record"]["generation"], "receipt_size_bytes": producer["server_record"]["size_bytes"],
            "receipt_sha256": producer["server_record"]["sha256"]}, "raw": list(rows.values()), "derived": []}
    raw = json.dumps(membership, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    member = {"object_name": prefix + f"/deliveries/{key}/capture_delivery_membership.json", "generation": "17000000000000000003",
              "size_bytes": len(raw), "sha256": "sha256:" + hashlib.sha256(raw).hexdigest()}
    contents[member["object_name"]] = raw
    class Blob:
        def __init__(self, name, generation):
            self.name, self.generation, self.size = name, generation, len(contents[name])
            self.crc32c = rows[name]["crc32c"] if name in rows else "AAAAAA=="
        def reload(self, **kwargs):
            assert kwargs["if_generation_match"] == self.generation and kwargs["retry"] is None
        def download_as_bytes(self, **kwargs):
            assert kwargs["if_generation_match"] == self.generation
            return contents[self.name]
    class Bucket:
        def blob(self, name, generation):
            expected = rows[name]["generation"] if name in rows else member["generation"]
            assert str(generation) == expected
            return Blob(name, generation)
    class Client:
        def bucket(self, name):
            assert name == owner["bucket"]
            return Bucket()
        def list_blobs(self, *_, **__):
            raise AssertionError("No prefix listing allowed")
    def download(*, downloads, **_):
        for blob, destination in downloads:
            destination.write_bytes(contents[blob.name])
    monkeypatch.setattr(listener, "download_with_reservation", download)
    payload = {"bucket": owner["bucket"], "scene_id": owner["scene_id"], "capture_id": owner["capture_id"], "raw_prefix_uri": owner["raw_prefix_uri"],
               "source_finalize": {**source, "event_id": "synthetic-joined-source", "event_source": "isolated-fixture"}, "source_membership_selector": member}
    selectors = {key: owner[key] for key in ("request_id", "scene_id", "capture_id")}
    selectors.update(completion_marker_generation=marker["generation"], producer_delivery_key=producer["delivery_key"],
                     source_payload_sha256=listener.payload_sha256(payload), task_context_digest=context["context_digest"])
    def held(**kwargs):
        assert kwargs["pipeline_lane"] == "qualification" and kwargs["run_evaluation_prep"] is False
        write_json(root / "pipeline/website_task_context.json", context)
        result = prepare_website_scene_handoff(descriptor={"scene_id": owner["scene_id"], "capture_id": owner["capture_id"], "metadata": {"site_task_context": context}},
            clean_plate={"privacy_verified": True, "status": "noop"}, provider_run={"status": "not_requested"}, capture_root=root, now=0)
        assert result["status"] == "awaiting_inputs" and result["blockers"] == ["website_reconstruction_pending"]
        raise PipelineError("website_reconstruction_pending") from StageError("website_scene_preparation", "website_reconstruction_pending")
    def process():
        try:
            listener.process_handoff_payload(payload, storage_root=storage_root, provider="local", storage_client=Client(), run_e2e=held,
                run_e2e_enabled=False, stage_control_plane=True)
        except PipelineError as exc:
            assert str(exc) == "website_reconstruction_pending"
    if not args.resume_only:
        process()
    else:
        assert listener._read_job_ledger(root)["status"] in {"failed_retryable", "processing"}

    def send_wakeup(*, capture_id, operation, payload):
        assert operation == "preparation-status" and capture_id == owner["capture_id"]
        body = json.dumps(payload, separators=(",", ":")).encode()
        req = Request(fixture["callback_base_url"].rstrip("/") + f"/api/internal/pipeline/creator-captures/{capture_id}/preparation-status",
                      data=body, method="POST", headers={**_pipeline_sync_headers(fixture["callback_secret"], body), "Content-Type": "application/json"})
        with urlopen(req, timeout=10) as response:
            return json.load(response)
    monkeypatch.setattr(context_reader, "website_webapp_request", send_wakeup)
    monkeypatch.setenv(service.INTAKE_CLIENT_SECRETS_ENV, json.dumps({"blueprint-webapp": fixture["forward_token"]}))
    monkeypatch.setenv(service.INTAKE_CLIENT_ROOTS_ENV, json.dumps({"blueprint-webapp": {owner["request_id"]: str(root)}}))
    monkeypatch.setenv(service.INTAKE_NONCE_STORE_DIR_ENV, str(state_root / "nonces"))
    monkeypatch.setenv("BLUEPRINT_LIVE_PIPELINE_INTAKE_WORK_DIR", str(state_root / "work"))
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_OUTPUT_PATH", str(state_root / "absent.json"))
    monkeypatch.setenv(service.INTAKE_ALLOW_LEGACY_BEARER_ENV, "false")
    server_socket = socket.socket()
    server_socket.bind(("127.0.0.1", 0))
    server_socket.listen()
    port = server_socket.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(service.create_app(), host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [server_socket]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started
    ready = {"event": "ready", "schema_version": "website_preparation_joined_ready.v1", "base_url": f"http://127.0.0.1:{port}",
             "payload": payload, "selectors": selectors, "ledger": listener._read_job_ledger(root),
             "provider_calls": 0, "simulation": ["synthetic owned object store", "owner/context transport", "qualification runner refusal seam"], "new_unique_case_credit": 0}
    write_json(args.ready_file, ready)
    args.ready_file.chmod(0o600)
    print(json.dumps(ready), flush=True)
    try:
        for line in sys.stdin:
            command = json.loads(line)["command"]
            if command == "stop":
                print(json.dumps({"command": command, "event": "stopped", "result": {"stopped": True}}), flush=True)
                break
            if command == "deliver":
                counts = reconcile_preparation_wakeups(storage_root)
                delivery_path = root / "website_preparation_status_delivery.json"
                delivery = json.loads(delivery_path.read_text()) if delivery_path.is_file() else {}
                result = {"delivered": delivery.get("state") == "delivered", "remaining_pending": int(delivery.get("state") == "pending"),
                          "delivery_counts": counts, "delivery": delivery, "acceptance": delivery.get("acceptance")}
            elif command == "retry":
                process()
                result = listener._read_job_ledger(root)
            elif command == "status":
                result = read_preparation_status(capture_root=root, selectors=selectors)
            else:
                raise ValueError("Unknown isolated test command")
            print(json.dumps({"command": command, "event": "delivered" if command == "deliver" else command, "result": result}), flush=True)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        monkeypatch.undo()


if __name__ == "__main__":
    main()
