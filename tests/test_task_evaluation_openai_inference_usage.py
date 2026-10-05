from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from blueprint_pipeline import task_evaluation_openai_inference_usage as usage_projection

from blueprint_pipeline.task_evaluation_openai_inference_usage import (
    build_placement_inference_usage_packet,
    sync_inference_usage_to_webapp,
)


def _placement_receipt() -> dict:
    usage = {
        "model": "gpt-6.1-sol",
        "input_tokens": 2_000,
        "cached_tokens": 1_200,
        "cache_write_tokens": 0,
        "uncached_input_tokens": 800,
        "output_tokens": 20,
        "reasoning_tokens": 5,
        "cache_hit_ratio": 0.6,
        "uncached_input_cost_usd": 0.0032,
        "cache_write_cost_usd": 0.0,
        "cached_read_cost_usd": 0.00048,
        "output_cost_usd": 0.0004,
        "estimated_total_cost_usd": 0.00408,
        "estimated_cost_without_caching_usd": 0.0084,
        "estimated_savings_usd": 0.00432,
        "cost_status": "model_pricing_estimate_not_official_billing",
        "provider_response_id": "resp_cache_read",
        "provider_request_id": "req_cache_read",
        "usage_receipt_digest": "sha256:" + "b" * 64,
        "breakpoint_digests": [
            "sha256:" + "c" * 64,
            "sha256:" + "9" * 64,
        ],
        "cache_policy": {
            "status": "enabled",
            "model_family": "gpt-6.1-sol",
            "family": "task_aware_robot_placement_proposal",
            "contract_version": "robot-placement-proposal-v2",
            "stable_prefix_digest": "sha256:" + "c" * 64,
            "policy_digest": "sha256:" + "d" * 64,
            "privacy_scope": "task_evaluation_rights_admitted",
            "processing_region": "default",
            "decision_reason": "expected_cached_cost_lower",
            "cache_key_digest": "sha256:" + "a" * 64,
            "economics": {"stable_prefix_tokens": 1_200},
        },
    }
    return {
        "run_id": "placement-run",
        "receipt_digest": "sha256:" + "e" * 64,
        "rounds": [{"proposal_usage": usage}],
    }


def test_packet_projects_only_digest_and_usage_evidence() -> None:
    packet = build_placement_inference_usage_packet(
        placement_receipt=_placement_receipt(),
        packet_run_id="website-run",
        launch_id="launch-1",
        source_commit="f" * 40,
    )
    call = packet["calls"][0]

    assert call["cached_tokens"] == 1_200
    assert call["cache_write_tokens"] == 0
    assert call["uncached_input_tokens"] == 800
    assert call["cache_key_digest"].startswith("sha256:")
    assert "cache_key" not in call
    assert call["raw_prompt_recorded"] is False
    assert call["dynamic_content_before_breakpoint"] is False
    assert "blueprint:cache:v1" not in json.dumps(packet)


def test_signed_sync_requires_exact_response_binding(monkeypatch) -> None:
    packet = build_placement_inference_usage_packet(
        placement_receipt=_placement_receipt(),
        packet_run_id="website-run",
        launch_id=None,
        source_commit="f" * 40,
    )
    captured: dict = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(
                {
                    "schema_version": "blueprint_openai_inference_usage_ingest_receipt.v1",
                    "status": "created",
                    "run_id": packet["run_id"],
                    "launch_id": packet["launch_id"],
                    "source_commit": packet["source_commit"],
                    "packet_digest": packet["packet_digest"],
                    "call_count": len(packet["calls"]),
                }
            ).encode()

    def urlopen(request, *, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_openai_inference_usage.urllib_request.urlopen",
        urlopen,
    )
    result = sync_inference_usage_to_webapp(
        packet=packet,
        endpoint_url="https://tryblueprint.io/api/internal/pipeline/openai-inference-usage",
        token="test-sync-token",
    )

    assert result["status"] == "succeeded"
    assert captured["request"].headers["X-blueprint-pipeline-signature"].startswith(
        "sha256="
    )
    assert captured["timeout"] == 10.0


def _project(root, **overrides):
    arguments = {
        "placement_receipt": _placement_receipt(),
        "packet_run_id": "website-run",
        "launch_id": "launch-1",
        "source_commit": "f" * 40,
        "output_root": root,
        "require_sync": True,
    }
    return usage_projection.materialize_placement_usage_projection(**(arguments | overrides))


def _consumer(monkeypatch, *, lose_first_response=False):
    """Exercise the real signed adapter; emulate the consumer's transaction keys."""
    bodies, records = [], {}

    class Response:
        def __init__(self, receipt):
            self.receipt = receipt

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(self.receipt).encode()

    def urlopen(request, *, timeout):
        bodies.append(request.data)
        packet = json.loads(request.data)
        assert request.headers["X-blueprint-pipeline-signature"].startswith("sha256=")
        status = "replayed" if all(c["call_id"] in records for c in packet["calls"]) else "created"
        for call in packet["calls"]:
            assert records.setdefault(call["call_id"], packet["packet_digest"]) == packet["packet_digest"]
        if lose_first_response and len(bodies) == 1:
            raise TimeoutError("synthetic acknowledgement loss")
        return Response({
            "schema_version": "blueprint_openai_inference_usage_ingest_receipt.v1",
            "status": status,
            **{key: packet[key] for key in ("run_id", "launch_id", "source_commit", "packet_digest")},
            "call_count": len(packet["calls"]),
        })

    monkeypatch.setattr(usage_projection, "load_pipeline_sync_token", lambda **_: "synthetic-token")
    monkeypatch.setattr(usage_projection.urllib_request, "urlopen", urlopen)
    return bodies, records


def test_timeout_after_acceptance_replays_identical_packet(tmp_path, monkeypatch):
    bodies, records = _consumer(monkeypatch, lose_first_response=True)
    with pytest.raises(usage_projection.OpenAIInferenceUsageError, match="sync_required"):
        _project(tmp_path)
    packet_path = tmp_path / "openai_inference_usage_packet.v1.json"
    retained = packet_path.read_bytes()
    result = _project(tmp_path)
    assert len(records) == 1
    assert len(bodies) == 2 and bodies[0] == bodies[1]
    assert packet_path.read_bytes() == retained
    assert usage_projection.result_projection_valid(result)
    receipt = json.loads(Path(
        result["openai_inference_usage_webapp_sync"]["artifact"]["path"]
    ).read_text())
    assert receipt["response"]["status"] == "replayed"


def test_optional_failed_sync_does_not_poison_required_recovery(tmp_path, monkeypatch):
    _consumer(monkeypatch, lose_first_response=True)
    failed = _project(tmp_path, require_sync=False)
    failed_path = Path(failed["openai_inference_usage_webapp_sync"]["artifact"]["path"])
    failed_bytes = failed_path.read_bytes()
    recovered = _project(tmp_path)
    assert usage_projection.result_projection_valid(recovered)
    assert failed_path.read_bytes() == failed_bytes
    assert recovered["openai_inference_usage_webapp_sync"]["artifact"]["path"] != str(failed_path)


@pytest.mark.parametrize("field,value", [
    ("packet_run_id", "other-run"), ("launch_id", "other-launch"),
    ("source_commit", "a" * 40), ("placement_receipt", {"receipt_digest": "changed"}),
    ("usage", 99),
])
def test_retained_projection_refuses_changed_binding(tmp_path, monkeypatch, field, value):
    bodies, _ = _consumer(monkeypatch)
    _project(tmp_path)
    overrides = {field: value}
    if field in {"placement_receipt", "usage"}:
        receipt = _placement_receipt()
        if field == "usage":
            receipt["rounds"][0]["proposal_usage"]["output_tokens"] = value
        else:
            receipt.update(value)
        overrides = {"placement_receipt": receipt}
    with pytest.raises(usage_projection.OpenAIInferenceUsageError, match="artifact_conflict"):
        _project(tmp_path, **overrides)
    assert len(bodies) == 1


@pytest.mark.parametrize("corruption", ["truncated", "digest", "timestamp", "symlink", "directory"])
def test_invalid_retained_packet_never_dispatches(tmp_path, monkeypatch, corruption):
    bodies, _ = _consumer(monkeypatch)
    _project(tmp_path)
    path = tmp_path / "openai_inference_usage_packet.v1.json"
    original = path.read_bytes()
    path.chmod(0o640)
    if corruption == "truncated":
        path.write_bytes(b'{"schema_version":')
    elif corruption in {"digest", "timestamp"}:
        packet = json.loads(original)
        packet["packet_digest" if corruption == "digest" else "generated_at_utc"] = "invalid"
        path.write_text(json.dumps(packet))
    else:
        path.unlink()
        if corruption == "symlink":
            target = tmp_path / "elsewhere.json"
            target.write_bytes(original)
            path.symlink_to(target)
        else:
            path.mkdir()
    with pytest.raises(usage_projection.OpenAIInferenceUsageError, match="artifact_conflict"):
        _project(tmp_path)
    assert len(bodies) == 1


def test_concurrent_first_writers_reconcile_to_one_packet(tmp_path, monkeypatch):
    bodies, records = _consumer(monkeypatch)
    barrier = Barrier(2)
    build = usage_projection.build_placement_inference_usage_packet

    def simultaneous_build(**kwargs):
        packet = build(**kwargs)
        barrier.wait(timeout=5)
        return packet

    monkeypatch.setattr(usage_projection, "build_placement_inference_usage_packet", simultaneous_build)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: _project(tmp_path), range(2)))
    assert all(usage_projection.result_projection_valid(result) for result in results)
    assert len(records) == 1
    assert bodies[0] == bodies[1]


@pytest.mark.parametrize("crash_after_publish", [False, True])
def test_crash_around_publication_recovers_without_partial_packet(tmp_path, monkeypatch, crash_after_publish):
    bodies, _ = _consumer(monkeypatch)
    link = usage_projection.os.link

    def crash(source, target):
        if crash_after_publish:
            link(source, target)
        raise OSError("synthetic crash around atomic publication")

    monkeypatch.setattr(usage_projection.os, "link", crash)
    with pytest.raises(OSError, match="synthetic crash"):
        _project(tmp_path)
    path = tmp_path / "openai_inference_usage_packet.v1.json"
    assert path.exists() == crash_after_publish
    assert not bodies
    monkeypatch.setattr(usage_projection.os, "link", link)
    assert usage_projection.result_projection_valid(_project(tmp_path))


def test_legacy_optional_receipt_and_artifact_reference_remain_immutable(tmp_path, monkeypatch):
    _consumer(monkeypatch)
    legacy_path = tmp_path / "openai_inference_usage_webapp_sync.v1.json"
    legacy_path.write_text('{"status":"skipped","reason":"sync_not_configured"}\n')
    legacy_record = usage_projection._artifact_record(legacy_path)
    first = _project(tmp_path)
    second = _project(tmp_path)
    assert usage_projection.result_projection_valid(first)
    assert usage_projection.result_projection_valid(second)
    assert usage_projection.artifact_record_valid(legacy_record)


def test_crash_after_sync_before_receipt_persistence_replays(tmp_path, monkeypatch):
    bodies, records = _consumer(monkeypatch)
    write = usage_projection._write_immutable_json

    def fail_receipt(path, value):
        if value.get("schema_version") == "blueprint_openai_inference_usage_sync_result.v1":
            raise OSError("synthetic receipt persistence failure")
        write(path, value)

    monkeypatch.setattr(usage_projection, "_write_immutable_json", fail_receipt)
    with pytest.raises(OSError, match="receipt persistence"):
        _project(tmp_path)
    monkeypatch.setattr(usage_projection, "_write_immutable_json", write)
    assert usage_projection.result_projection_valid(_project(tmp_path))
    assert bodies[0] == bodies[1]
    assert len(records) == 1
