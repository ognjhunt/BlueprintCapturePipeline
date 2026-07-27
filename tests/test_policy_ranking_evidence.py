from __future__ import annotations

import json
import sys
import threading
import types
from pathlib import Path

import pytest

from blueprint_pipeline.policy_ranking_evidence import (
    EvidenceStore,
    InventoryMismatchError,
    scan_for_secrets,
    utc_now,
)
from blueprint_pipeline.policy_ranking_thesis_judge_openai import (
    GATE_ENV,
    _provider_error_details,
    run_inventory_v2,
)


def _store(path: Path, inventory: str = "a" * 64) -> EvidenceStore:
    return EvidenceStore(
        path,
        experiment_id="experiment-2-test",
        inventory_sha256=inventory,
        configuration_sha256="b" * 64,
    )


def _request(request_id: str = "r1") -> dict[str, str]:
    return {
        "request_id": request_id,
        "session_id": "s1",
        "policy_id": "p1",
        "task_id": "t1",
        "deterministic_input_hash": "c" * 64,
        "method": "temporal",
    }


def _accept(
    store: EvidenceStore,
    request: dict[str, str],
    cost: float = 0.01,
    claim: str | None = None,
) -> None:
    claim = claim or store.claim(
        request,
        arm_id="temporal",
        provider="test",
        model_snapshot="model-v1",
        attempt_type="scientific_request",
        lease_seconds=30,
    )
    assert claim
    store.complete(
        request=request,
        claim_id=claim,
        arm_id="temporal",
        attempt_type="scientific_request",
        provider="test",
        model_snapshot="model-v1",
        started_at=utc_now(),
        elapsed_seconds=0.1,
        structured_response={"score": 0.75},
        validation_result="valid",
        usage={"input_tokens": 10, "output_tokens": 5},
        estimated_cost_usd=cost,
        actual_cost_usd=cost,
        response_id="response-1",
        consumed_scientific_response=True,
    )


def _inventory() -> dict[str, object]:
    return {
        "status": "ready",
        "inventory_sha256": "d" * 64,
        "requests": [_request()],
    }


def _fake_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "openai",
        types.SimpleNamespace(OpenAI=lambda **kwargs: object()),
    )
    monkeypatch.setenv(GATE_ENV, "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-not-persisted")


def test_crash_during_write_leaves_no_visible_partial_event(tmp_path: Path) -> None:
    store = _store(tmp_path / "evidence")
    partial = store.journal_dir / ".00000002-crashed.tmp"
    partial.write_text('{"partial":')
    reopened = _store(tmp_path / "evidence")
    reopened.append("preflight_failed", {"provider_called": False})
    reopened.verify()
    assert len(reopened.events()) == 2
    assert partial.read_text() == '{"partial":'


def test_kill_and_resume_reclaims_expired_unaccepted_request(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    store = _store(root)
    first_claim = store.claim(
        _request(),
        arm_id="temporal",
        provider="test",
        model_snapshot="m",
        attempt_type="scientific_request",
        lease_seconds=0,
    )
    assert first_claim
    resumed = _store(root)
    second_claim = resumed.claim(
        _request(),
        arm_id="temporal",
        provider="test",
        model_snapshot="m",
        attempt_type="infrastructure_retry",
        lease_seconds=30,
    )
    assert second_claim and second_claim != first_claim
    _accept(resumed, _request(), claim=second_claim)
    assert resumed.rebuild()["accepted_request_count"] == 1


def test_missing_credential_and_gate_preserve_nonempty_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "evidence"
    _fake_openai(monkeypatch)
    monkeypatch.setattr(
        "blueprint_pipeline.policy_ranking_thesis_judge_openai._score_one",
        lambda client, request: (
            {
                "request_id": request["request_id"],
                "response_id": "ok",
                "usage": {"estimated_cost_usd_conservative": 0.01},
            },
            {},
        ),
    )
    first = run_inventory_v2(
        _inventory(),
        evidence_root=root,
        experiment_id="experiment-2-test",
        max_estimated_cost_usd=1,
        projected_total_cost_usd=0.02,
    )
    assert first["accepted_request_count"] == 1
    accepted_paths = sorted((root / "requests" / "r1").glob("*.json"))
    snapshots = {path: path.read_bytes() for path in accepted_paths}

    monkeypatch.delenv("OPENAI_API_KEY")
    missing_key = run_inventory_v2(
        _inventory(),
        evidence_root=root,
        experiment_id="experiment-2-test",
        max_estimated_cost_usd=1,
        projected_total_cost_usd=0.02,
    )
    assert missing_key["status"] == "blocked"
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.delenv(GATE_ENV)
    missing_gate = run_inventory_v2(
        _inventory(),
        evidence_root=root,
        experiment_id="experiment-2-test",
        max_estimated_cost_usd=1,
        projected_total_cost_usd=0.02,
    )
    assert missing_gate["status"] == "blocked"
    assert all(path.read_bytes() == content for path, content in snapshots.items())


def test_mismatched_inventory_cannot_merge(tmp_path: Path) -> None:
    _store(tmp_path / "evidence")
    with pytest.raises(InventoryMismatchError):
        _store(tmp_path / "evidence", inventory="e" * 64)


def test_duplicate_completion_is_recorded_but_never_reaccepted(tmp_path: Path) -> None:
    store = _store(tmp_path / "evidence")
    request = _request()
    claim = store.claim(
        request,
        arm_id="temporal",
        provider="test",
        model_snapshot="m",
        attempt_type="scientific_request",
        lease_seconds=30,
    )
    assert claim
    arguments = dict(
        request=request,
        claim_id=claim,
        arm_id="temporal",
        attempt_type="scientific_request",
        provider="test",
        model_snapshot="m",
        started_at=utc_now(),
        elapsed_seconds=0.1,
        structured_response={"score": 1},
        validation_result="valid",
        usage={},
        estimated_cost_usd=0.0,
        actual_cost_usd=0.0,
    )
    first = store.complete(**arguments)
    second = store.complete(**arguments)
    assert first["event_type"] == "response_accepted"
    assert second["event_type"] == "duplicate_completion_ignored"
    assert store.rebuild()["accepted_request_count"] == 1


def test_concurrent_completion_order_is_hash_chained(tmp_path: Path) -> None:
    store = _store(tmp_path / "evidence")
    claims: list[tuple[dict[str, str], str]] = []
    for index in range(8):
        request = _request(f"r{index}")
        claim = store.claim(
            request,
            arm_id="temporal",
            provider="test",
            model_snapshot="m",
            attempt_type="scientific_request",
            lease_seconds=30,
        )
        assert claim
        claims.append((request, claim))

    threads = [
        threading.Thread(target=_accept, args=(store, request, 0.001, claim))
        for request, claim in claims
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    store.verify()
    sequences = [event["sequence"] for event in store.events()]
    assert sequences == list(range(1, len(sequences) + 1))
    assert store.rebuild()["accepted_request_count"] == 8


def test_provider_429_preserves_retry_after_and_reset_metadata() -> None:
    class RateLimitError(Exception):
        status_code = 429
        response = types.SimpleNamespace(
            headers={"Retry-After": "12", "x-ratelimit-reset-requests": "2s"}
        )

    details = _provider_error_details(RateLimitError())
    assert details["category"].startswith("http_429")
    assert details["retry_after_seconds"] == 12
    assert details["reset_metadata"]["x-ratelimit-reset-requests"] == "2s"


def test_provider_429_without_retry_after_is_explicit() -> None:
    class RateLimitError(Exception):
        status_code = 429
        response = types.SimpleNamespace(headers={})

    details = _provider_error_details(RateLimitError())
    assert details["retry_after_seconds"] is None
    assert details["category"] == "http_429:RateLimitError"


def test_invalid_structured_response_consumes_scientific_response(tmp_path: Path) -> None:
    store = _store(tmp_path / "evidence")
    request = _request()
    claim = store.claim(
        request,
        arm_id="temporal",
        provider="test",
        model_snapshot="m",
        attempt_type="scientific_request",
        lease_seconds=30,
    )
    assert claim
    event = store.complete(
        request=request,
        claim_id=claim,
        arm_id="temporal",
        attempt_type="scientific_request",
        provider="test",
        model_snapshot="m",
        started_at=utc_now(),
        elapsed_seconds=0.2,
        structured_response=None,
        validation_result="unparseable_structured_output",
        usage={"output_tokens": 50},
        estimated_cost_usd=0.01,
        actual_cost_usd=None,
        provider_error_category="invalid_structured_response",
        consumed_scientific_response=True,
    )
    assert event["event_type"] == "attempt_failed"
    assert event["payload"]["consumed_scientific_response"] is True
    assert store.rebuild()["accepted_request_count"] == 0


def test_valid_response_is_not_resampled_on_retry_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_openai(monkeypatch)
    calls = 0

    def score(client, request):
        nonlocal calls
        calls += 1
        return (
            {
                "request_id": request["request_id"],
                "response_id": "ok",
                "usage": {"estimated_cost_usd_conservative": 0.01},
            },
            {},
        )

    monkeypatch.setattr("blueprint_pipeline.policy_ranking_thesis_judge_openai._score_one", score)
    kwargs = dict(
        evidence_root=tmp_path / "evidence",
        experiment_id="experiment-2-test",
        max_estimated_cost_usd=1,
        projected_total_cost_usd=0.02,
    )
    assert run_inventory_v2(_inventory(), **kwargs)["status"] == "completed"
    assert run_inventory_v2(_inventory(), **kwargs)["status"] == "completed"
    assert calls == 1


def test_quota_failure_before_first_acceptance_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_openai(monkeypatch)

    class RateLimitError(Exception):
        status_code = 429
        response = types.SimpleNamespace(headers={})

    monkeypatch.setattr(
        "blueprint_pipeline.policy_ranking_thesis_judge_openai._score_one",
        lambda client, request: (_ for _ in ()).throw(RateLimitError()),
    )
    result = run_inventory_v2(
        _inventory(),
        evidence_root=tmp_path / "evidence",
        experiment_id="experiment-2-test",
        max_estimated_cost_usd=1,
        projected_total_cost_usd=0.02,
        infrastructure_retries_per_request=1,
        systemic_rejection_threshold=10,
        sleep_function=lambda seconds: None,
    )
    aggregate = json.loads((tmp_path / "evidence" / "derived_aggregate.json").read_text())
    assert result["status"] == "blocked"
    assert aggregate["accepted_request_count"] == 0
    assert aggregate["failed_attempt_count"] == 2
    assert aggregate["provider_called"] is True


def test_exact_cost_recomputation_counts_each_attempt_once(tmp_path: Path) -> None:
    store = _store(tmp_path / "evidence")
    _accept(store, _request("r1"), 0.0125)
    _accept(store, _request("r2"), 0.0075)
    aggregate = store.rebuild()
    assert aggregate["estimated_cost_usd_recomputed"] == pytest.approx(0.02)
    assert aggregate["actual_cost_usd_recomputed"] == pytest.approx(0.02)


def test_deterministic_aggregate_rebuild_and_manifest(tmp_path: Path) -> None:
    store = _store(tmp_path / "evidence")
    _accept(store, _request())
    first = store.rebuild()
    first_bytes = store.aggregate_path.read_bytes()
    second = store.rebuild()
    assert second["aggregate_sha256"] == first["aggregate_sha256"]
    assert store.aggregate_path.read_bytes() == first_bytes
    store.verify_manifest()


def test_secret_redaction_and_scan(tmp_path: Path) -> None:
    store = _store(tmp_path / "evidence")
    store.append(
        "preflight_failed",
        {
            "api_key": "sk-abcdefghijklmnopqrstuvwxyz123456",
            "authorization": "Bearer abcdefghijklmnopqrstuvwxyz",
            "message": "key sk-abcdefghijklmnopqrstuvwxyz123456 was rejected",
        },
    )
    assert scan_for_secrets(store.root) == []
    text = "".join(path.read_text() for path in store.root.rglob("*.json"))
    assert "abcdefghijklmnopqrstuvwxyz123456" not in text
    assert "[REDACTED]" in text


def test_append_only_event_files_never_change(tmp_path: Path) -> None:
    store = _store(tmp_path / "evidence")
    existing = {path: path.read_bytes() for path in store.journal_dir.glob("*.json")}
    store.append("preflight_failed", {"blocker": "test", "provider_called": False})
    assert all(path.read_bytes() == content for path, content in existing.items())
    assert len(list(store.journal_dir.glob("*.json"))) == len(existing) + 1
