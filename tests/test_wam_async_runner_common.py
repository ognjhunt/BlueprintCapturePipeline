from __future__ import annotations

import json

import pytest

from blueprint_pipeline.wam_async_runner_common import (
    AsyncPollDeadline,
    decide_async_teardown,
    deadline_capped_wait_seconds,
    download_url_to_file,
    read_json_mapping,
    read_sensitive_url_file,
    redact_provider_url,
    regex_json_number_field,
    regex_json_string_field,
)


def test_async_poll_deadline_centralizes_retry_and_expiry_semantics(
    monkeypatch,
) -> None:
    deadline = AsyncPollDeadline.start(
        max_wait_seconds=10,
        retry_interval_seconds=3,
        started_monotonic=100.0,
    )
    waits: list[float] = []
    monkeypatch.setattr("blueprint_pipeline.wam_async_runner_common.time.sleep", waits.append)

    assert deadline.is_open(109.0) is True
    assert deadline.can_retry(107.0) is True
    assert deadline.can_retry(107.1) is False
    deadline.wait_for_retry()
    assert waits == [3]
    assert deadline.elapsed_seconds(111.0) == 11.0
    assert deadline.expired(110.0) is True


def test_paid_deadline_caps_provider_poll_wait() -> None:
    capped, remaining, applied = deadline_capped_wait_seconds(
        state={"max_live_deadline_epoch": 125.8},
        requested_max_wait_seconds=60,
        now_epoch=100.0,
    )
    assert capped == 25
    assert remaining == pytest.approx(25.8)
    assert applied is True
    assert deadline_capped_wait_seconds(
        state={},
        requested_max_wait_seconds=60,
        now_epoch=100.0,
    ) == (60, None, False)


def test_async_teardown_waits_for_explicit_request_readiness() -> None:
    decision = decide_async_teardown(
        explicit_requested=True,
        requested_ready=False,
        requested_action="stop",
        allocation_actionable=True,
    )

    assert decision.effective_requested is True
    assert decision.should_teardown is False
    assert decision.action == "stop"
    assert decision.teardown_pending is False


def test_async_teardown_automatic_failure_forces_delete_when_actionable() -> None:
    decision = decide_async_teardown(
        explicit_requested=False,
        requested_ready=False,
        requested_action="keep_on_success",
        automatic_reasons=("runtime_stall", "runtime_stall", "output_validation_failed"),
        automatic_action="delete",
        allocation_actionable=True,
    )

    assert decision.should_teardown is True
    assert decision.action == "delete"
    assert decision.teardown_pending is True
    assert decision.automatic_reasons == (
        "runtime_stall",
        "output_validation_failed",
    )


def test_async_teardown_keeps_decision_but_blocks_mutation_without_allocation() -> None:
    decision = decide_async_teardown(
        explicit_requested=True,
        requested_ready=True,
        requested_action="destroy",
        allocation_actionable=False,
        blockers_present=True,
    )

    assert decision.should_teardown is True
    assert decision.teardown_pending is False
    assert decision.as_manifest()["blockers_present"] is True


def test_redact_provider_url_removes_query_and_fragment() -> None:
    value = redact_provider_url("https://objects.example/bundle.zip?secret=1#token")

    assert value == "https://objects.example/bundle.zip?REDACTED_QUERY#REDACTED_FRAGMENT"
    assert "secret=1" not in value
    assert "token" not in value


def test_read_json_mapping_rejects_non_object_shape(tmp_path) -> None:
    path = tmp_path / "value.json"
    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    assert read_json_mapping(path) == {}


def test_partial_json_field_salvage_is_bounded_to_requested_fields() -> None:
    partial = 'prefix "instance_id": 42, "bundle_path": "/tmp/bundle.zip" trailing'

    assert regex_json_number_field(partial, "instance_id") == 42.0
    assert regex_json_string_field(partial, "bundle_path") == "/tmp/bundle.zip"
    assert regex_json_number_field(partial, "missing") is None
    assert regex_json_string_field(partial, "missing") == ""


def test_read_sensitive_url_file_reports_metadata_without_echoing_value(tmp_path) -> None:
    path = tmp_path / "signed-url.txt"
    secret_url = "https://objects.example/bundle.zip?credential=do-not-record"
    path.write_text(secret_url, encoding="utf-8")
    path.chmod(0o600)

    value, metadata = read_sensitive_url_file(str(path), label="provider_bundle_url")

    assert value == secret_url
    assert metadata["mode_is_0600"] is True
    assert metadata["value_present"] is True
    assert metadata["raw_secret_values_recorded"] is False
    assert secret_url not in json.dumps(metadata)


def test_download_url_to_file_does_not_record_signed_url(tmp_path, monkeypatch) -> None:
    class Response:
        status = 200
        headers = {
            "Content-Length": str(len(b"provider-output")),
            "Last-Modified": "Wed, 29 Jul 2026 11:37:00 GMT",
            "ETag": '"safe-object-etag"',
        }

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, size=-1) -> bytes:
            import io
            if not hasattr(self, "body"):
                self.body = io.BytesIO(b"provider-output")
            return self.body.read(size)

    monkeypatch.setattr(
        "blueprint_pipeline.wam_async_runner_common.urllib.request.urlopen",
        lambda *args, **kwargs: Response(),
    )
    url = "https://objects.example/output.zip?credential=do-not-record"
    output = tmp_path / "output.zip"

    result = download_url_to_file(
        url=url,
        output_path=output,
        user_agent="BlueprintTest/1.0",
        timeout_seconds=10,
    )

    assert result["status"] == "completed"
    assert result["downloaded_size_bytes"] == len(b"provider-output")
    assert result["response_last_modified"] == "Wed, 29 Jul 2026 11:37:00 GMT"
    assert result["response_etag_present"] is True
    assert output.read_bytes() == b"provider-output"
    assert url not in json.dumps(result)


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/provider-output.zip",
        "http://objects.example/output.zip",
        "https://user:password@objects.example/output.zip",
    ],
)
def test_download_url_to_file_rejects_unsafe_url_schemes_and_credentials(
    tmp_path,
    url: str,
) -> None:
    result = download_url_to_file(
        url=url,
        output_path=tmp_path / "output.zip",
        user_agent="BlueprintTest/1.0",
        timeout_seconds=10,
    )

    assert result["status"] == "blocked"
    assert result["error_type"] == "ValueError"


def test_provider_download_streams_bounded_reads_and_atomically_keeps_complete_bytes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from blueprint_pipeline import wam_async_runner_common as common
    total = 20 * 1024**2 + 17
    class Response:
        status = 200
        headers = {"Content-Length": str(total)}
        remaining = total
        reads = []
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, size=-1):
            assert 0 < size <= 1024**2, "unbounded provider-output read"
            self.reads.append(size)
            count = min(size, self.remaining)
            self.remaining -= count
            return b"x" * count
    response = Response()
    monkeypatch.setattr(common.urllib.request, 'urlopen', lambda *a, **kw: response)
    monkeypatch.setattr('shutil.disk_usage', lambda _path: SimpleNamespace(free=total + 1024**2))
    output = tmp_path / 'result.zip'
    result = common.download_url_to_file(url='https://example.invalid/result', output_path=output,
                                        user_agent='test', timeout_seconds=1)
    assert result['status'] == 'completed'
    assert output.stat().st_size == total and result['downloaded_size_bytes'] == total
    with output.open('rb') as stream:
        assert all(chunk == b'x' * len(chunk) for chunk in iter(lambda: stream.read(1024**2), b''))
    assert len(response.reads) > 20 and list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize('failure', ['short_body', 'network', 'capacity_before', 'capacity_during'])
def test_provider_download_failure_preserves_previous_file_and_removes_partial(tmp_path, monkeypatch, failure):
    from types import SimpleNamespace
    from blueprint_pipeline import wam_async_runner_common as common
    output = tmp_path / 'result.zip'
    output.write_bytes(b'previous')
    class Response:
        status = 200
        headers = {'Content-Length': str(2 * 1024**2)}
        reads = 0
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, size):
            self.reads += 1
            if self.reads == 1:
                return b'x' * size
            if failure == 'network':
                raise OSError('network failure')
            return b''
    response = Response()
    checks = []
    def capacity(_path):
        checks.append(True)
        free = 0 if failure == 'capacity_before' or (failure == 'capacity_during' and len(checks) > 1) else 10 * 1024**2
        return SimpleNamespace(free=free)
    monkeypatch.setattr(common.urllib.request, 'urlopen', lambda *a, **kw: response)
    monkeypatch.setattr('shutil.disk_usage', capacity)
    result = common.download_url_to_file(url='https://example.invalid/result?secret=hidden',
        output_path=output, user_agent='test', timeout_seconds=1)
    assert result['status'] == 'blocked'
    assert output.read_bytes() == b'previous' and list(tmp_path.iterdir()) == [output]
    assert 'hidden' not in json.dumps(result)
    if failure == 'capacity_before':
        assert response.reads == 0


def test_provider_download_refuses_unknown_size_before_any_body_read(tmp_path, monkeypatch):
    from blueprint_pipeline import wam_async_runner_common as common
    class Response:
        status = 200
        headers = {}
        reads = 0
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, size):
            self.reads += 1
            return b''
    response = Response()
    monkeypatch.setattr(common.urllib.request, 'urlopen', lambda *a, **kw: response)
    result = common.download_url_to_file(url='https://example.invalid/unknown',
        output_path=tmp_path / 'result', user_agent='test', timeout_seconds=1)
    assert result['status'] == 'blocked' and response.reads == 0
    assert not list(tmp_path.iterdir())


def test_provider_download_respects_other_live_disk_reservations(tmp_path, monkeypatch, _isolated_control_plane_disk_reservation_root):
    from types import SimpleNamespace
    from blueprint_pipeline import control_plane_disk_budget as budget
    from blueprint_pipeline import wam_async_runner_common as common
    usage = SimpleNamespace(total=200 * 1024**3, free=100 * 1024**3, used=100 * 1024**3)
    expected = 20 * 1024**2
    ledger = _isolated_control_plane_disk_reservation_root
    real_reserve = budget.reserve_control_plane_disk
    other_bytes = usage.free - budget.floor_bytes(usage.total, role='policy_canary_output') - expected // 2
    class Response:
        status = 200
        headers = {'Content-Length': str(expected)}
        reads = 0
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, size):
            self.reads += 1
            return b''
    response = Response()
    monkeypatch.setattr(common.urllib.request, 'urlopen', lambda *a, **kw: response)
    monkeypatch.setattr('shutil.disk_usage', lambda _path: usage)
    def reserve(*args, **kwargs):
        kwargs['disk_usage'] = lambda _path: usage
        return real_reserve(*args, **kwargs)
    monkeypatch.setattr(budget, 'reserve_control_plane_disk', reserve)
    with real_reserve('policy_canary_output', target_root=tmp_path, expected_bytes=other_bytes,
                      reservation_root=ledger, disk_usage=lambda _path: usage):
        result = common.download_url_to_file(url='https://example.invalid/big',
            output_path=tmp_path / 'result', user_agent='test', timeout_seconds=1)
    assert result['status'] == 'blocked' and response.reads == 0
    assert not (tmp_path / 'result').exists() and not list(tmp_path.glob('*.partial'))
