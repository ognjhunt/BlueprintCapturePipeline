"""Hermetic tests for the persistent warm-render serve loop (NO Isaac, NO GPU).

The serve loop is the heart of "keep it warm": boot Isaac + load the scene ONCE, then serve a stream
of task render jobs so each rerun skips image pull + Isaac boot + stage load + most settle. The Isaac
render is INJECTED (``render_one``), so the control flow — poll, render, publish, stop / idle-timeout /
max-jobs, error isolation — is testable with fakes.
"""
from __future__ import annotations

import io
import json
import urllib.error
import zipfile
from collections import deque

from blueprint_pipeline.warm_render_server import (
    FileJobSource,
    PresignedUrlAccessError,
    SignedUrlJobSource,
    WarmJob,
    WarmPoolClient,
    serve_render_loop,
    submit_warm_render_batch,
)


class _FakeSource:
    def __init__(self, jobs):
        self._jobs = deque(jobs)
        self.results: dict[str, dict] = {}

    def poll(self):
        return self._jobs.popleft() if self._jobs else None

    def publish_result(self, request_id, result):
        self.results[request_id] = result


def _advancing_clock(step):
    t = {"v": 0.0}

    def clock():
        v = t["v"]
        t["v"] += step
        return v

    return clock


def test_serve_loop_renders_queued_jobs_then_stops_on_sentinel() -> None:
    src = _FakeSource([
        WarmJob("r1", {"description": "open the refrigerator"}, session_nonce="sess-1"),
        WarmJob("r2", {"description": "turn on the faucet"}),
        WarmJob("stop", {}, stop=True),
    ])
    rendered = []

    def render_one(scenario):
        rendered.append(scenario)
        return {"status": "ok", "task": scenario.get("description")}

    res = serve_render_loop(render_one=render_one, job_source=src, sleep=lambda s: None)

    assert res["jobs_served"] == 2
    assert res["exit_reason"] == "stop_requested"
    assert [s["description"] for s in rendered] == ["open the refrigerator", "turn on the faucet"]
    assert src.results["r1"]["status"] == "ok"
    assert src.results["r1"]["task"] == "open the refrigerator"
    assert src.results["r1"]["request_id"] == "r1"
    assert src.results["r1"]["warm_session_nonce"] == "sess-1"
    assert src.results["r2"]["status"] == "ok"


def test_serve_loop_exits_on_idle_timeout() -> None:
    src = _FakeSource([])  # never any work

    res = serve_render_loop(
        render_one=lambda sc: {"status": "ok"},
        job_source=src,
        idle_timeout_s=100.0,
        clock=_advancing_clock(60.0),
        sleep=lambda s: None,
    )
    assert res["jobs_served"] == 0
    assert res["exit_reason"] == "idle_timeout"


def test_serve_loop_exits_on_max_jobs() -> None:
    src = _FakeSource([WarmJob(f"r{i}", {"i": i}) for i in range(5)])
    res = serve_render_loop(
        render_one=lambda sc: {"status": "ok"},
        job_source=src,
        max_jobs=2,
        sleep=lambda s: None,
    )
    assert res["jobs_served"] == 2
    assert res["exit_reason"] == "max_jobs"


def test_serve_loop_isolates_render_errors_and_keeps_serving() -> None:
    src = _FakeSource([
        WarmJob("bad", {"description": "explode"}),
        WarmJob("good", {"description": "open the refrigerator"}),
        WarmJob("stop", {}, stop=True),
    ])

    def render_one(scenario):
        if scenario.get("description") == "explode":
            raise RuntimeError("isaac boom")
        return {"status": "ok"}

    res = serve_render_loop(render_one=render_one, job_source=src, sleep=lambda s: None)

    assert res["jobs_served"] == 2  # both jobs consumed; the bad one did not kill the loop
    assert src.results["bad"]["status"] == "error"
    assert "isaac boom" in src.results["bad"]["error"]
    assert src.results["good"]["status"] == "ok"


def test_file_job_source_round_trips_job_and_result(tmp_path) -> None:
    root = tmp_path / "warm"
    src = FileJobSource(root)
    assert src.poll() is None  # empty queue

    src.submit("req-1", {"scenario_id": "open_fridge", "description": "open the refrigerator"})
    job = src.poll()
    assert job is not None
    assert job.request_id == "req-1"
    assert job.scenario["description"] == "open the refrigerator"
    assert src.poll() is None  # claimed, not re-served

    src.publish_result("req-1", {"status": "ok", "pose": [-1.04, 0.66, 0.84]})
    result = json.loads((root / "results" / "req-1.json").read_text())
    assert result["status"] == "ok"


def test_file_job_source_reads_stop_sentinel(tmp_path) -> None:
    src = FileJobSource(tmp_path / "warm")
    src.submit_stop()
    job = src.poll()
    assert job is not None and job.stop is True


# --- signed-URL transport (the real remote pod <-> control plane channel) ---

def test_signed_url_job_source_polls_dedups_and_reads_stop(tmp_path) -> None:
    # Pod side: poll one inbox key (GET); a job is claimed once (by monotonic seq), and the same seq
    # is not re-served. A 404/empty inbox (no job yet) yields None.
    state = {"payload": None}

    def http_get(url):
        if state["payload"] is None:
            raise FileNotFoundError("404")
        return json.dumps(state["payload"]).encode()

    src = SignedUrlJobSource("http://inbox", tmp_path / "out", http_get=http_get)
    assert src.poll() is None
    state["payload"] = {"seq": 1, "request_id": "r1", "scenario": {"description": "open the refrigerator"}}
    job = src.poll()
    assert job is not None and job.request_id == "r1"
    assert job.scenario["description"] == "open the refrigerator"
    assert src.poll() is None  # same seq -> not re-served
    state["payload"] = {"seq": 2, "stop": True}
    job2 = src.poll()
    assert job2 is not None and job2.stop is True


def test_signed_url_job_source_treats_404_as_empty_inbox(tmp_path) -> None:
    def http_get(url):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    src = SignedUrlJobSource("http://inbox", tmp_path / "out", http_get=http_get)

    assert src.poll() is None
    assert src.consecutive_failures == 0
    assert src.last_error is None


def test_signed_url_job_source_surfaces_forbidden_inbox(tmp_path) -> None:
    def http_get(url):
        raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)

    src = SignedUrlJobSource("http://inbox", tmp_path / "out", http_get=http_get)

    try:
        src.poll()
    except PresignedUrlAccessError as exc:
        assert exc.classification == "presigned_url_expired_or_forbidden"
        assert exc.operation == "warm_inbox_get"
        assert exc.status_code == 403
    else:
        raise AssertionError("expected PresignedUrlAccessError")
    assert src.consecutive_failures == 1
    assert src.last_error == "presigned_url_expired_or_forbidden"


def test_serve_loop_exits_on_unrecoverable_inbox(tmp_path) -> None:
    src = SignedUrlJobSource(
        "http://inbox",
        tmp_path / "out",
        http_get=lambda _url: b"{not json",
        max_consecutive_failures=2,
    )
    logs: list[str] = []

    res = serve_render_loop(
        render_one=lambda sc: {"status": "ok"},
        job_source=src,
        idle_timeout_s=100.0,
        clock=_advancing_clock(1.0),
        sleep=lambda s: None,
        log=logs.append,
    )

    assert res["exit_reason"] == "inbox_unrecoverable"
    assert res["blocker"] == "warm_inbox_malformed_json"
    assert res["consecutive_failures"] == 2
    assert any("inbox unrecoverable" in line for line in logs)


def test_signed_url_job_source_publishes_result_into_out_dir(tmp_path) -> None:
    # Results ride the EXISTING output channel: the pod writes them into its out dir, which the
    # worker's heartbeat already uploads. No second presigned channel needed.
    src = SignedUrlJobSource("http://inbox", tmp_path / "out", http_get=lambda u: b"")
    src.publish_result("r1", {"status": "ok", "pose": [-1.04, 0.66, 0.84]})
    p = tmp_path / "out" / "warm_results" / "r1.json"
    assert json.loads(p.read_text())["status"] == "ok"


def test_warm_pool_client_submit_puts_incrementing_seq() -> None:
    puts = []
    cli = WarmPoolClient("http://inbox/put", "http://out/get",
                         http_put=lambda u, d: puts.append((u, json.loads(d))),
                         http_get=lambda u: b"")
    rid = cli.submit({"description": "open the refrigerator"}, request_id="r1")
    assert rid == "r1"
    cli.submit({"description": "open the microwave"})
    assert puts[0][1]["seq"] == 1 and puts[0][1]["request_id"] == "r1"
    assert puts[0][1]["scenario"]["description"] == "open the refrigerator"
    assert puts[0][1]["warm_session_nonce"] == cli.session_nonce
    assert puts[1][1]["seq"] == 2


def test_warm_pool_client_poll_result_reads_from_output_zip() -> None:
    session_nonce = "fresh-session"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("warm_results/r1.json", json.dumps({
            "status": "ok",
            "request_id": "r1",
            "warm_session_nonce": session_nonce,
        }))
    data = buf.getvalue()
    cli = WarmPoolClient("http://inbox/put", "http://out/get",
                         http_put=lambda u, d: None, http_get=lambda u: data,
                         session_nonce=session_nonce)
    res = cli.poll_result("r1", timeout_s=5.0, interval_s=0.0,
                          clock=_advancing_clock(1.0), sleep=lambda s: None)
    assert res is not None and res["status"] == "ok"


def test_warm_pool_client_poll_result_rejects_stale_session_result() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("warm_results/job-1.json", json.dumps({
            "status": "ok",
            "request_id": "job-1",
            "warm_session_nonce": "old-session",
        }))
    data = buf.getvalue()
    cli = WarmPoolClient(
        "http://inbox/put",
        "http://out/get",
        http_put=lambda u, d: None,
        http_get=lambda u: data,
        session_nonce="new-session",
    )

    res = cli.poll_result(
        "job-1",
        timeout_s=3.0,
        interval_s=0.0,
        clock=_advancing_clock(1.0),
        sleep=lambda s: None,
    )

    assert res is None


def test_warm_pool_client_poll_result_surfaces_expired_output_url() -> None:
    def http_get(url):
        raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)

    cli = WarmPoolClient(
        "http://inbox/put",
        "http://out/get",
        http_put=lambda u, d: None,
        http_get=http_get,
        session_nonce="fresh-session",
    )

    try:
        cli.poll_result(
            "job-1",
            timeout_s=3.0,
            interval_s=0.0,
            clock=_advancing_clock(1.0),
            sleep=lambda s: None,
        )
    except PresignedUrlAccessError as exc:
        assert exc.classification == "presigned_url_expired_or_forbidden"
        assert exc.operation == "warm_output_get"
    else:
        raise AssertionError("expected PresignedUrlAccessError")


def test_warm_pool_client_poll_result_times_out_when_absent() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("other.json", "{}")
    data = buf.getvalue()
    cli = WarmPoolClient("http://inbox/put", "http://out/get",
                         http_put=lambda u, d: None, http_get=lambda u: data)
    res = cli.poll_result("missing", timeout_s=3.0, interval_s=0.0,
                          clock=_advancing_clock(1.0), sleep=lambda s: None)
    assert res is None


def test_submit_warm_render_batch_keeps_one_monotonic_client_session(tmp_path) -> None:
    inbox_put_file = tmp_path / "warm_inbox_put_url.txt"
    output_get_file = tmp_path / "provider_output_get_url.txt"
    inbox_put_file.write_text("http://inbox/put", encoding="utf-8")
    output_get_file.write_text("http://out/get", encoding="utf-8")
    manifest_path = tmp_path / "isaac_g1_kitchen_parity_job_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "status": "serving",
            "warm_serve": {
                "inbox_put_url_file": str(inbox_put_file),
                "output_get_url_file": str(output_get_file),
            },
        }),
        encoding="utf-8",
    )
    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(
        json.dumps({
            "scenarios": [
                {"scenario_id": "sink_faucet", "description": "turn on the faucet"},
                {"scenario_id": "stovetop_knob", "description": "turn the stove knob"},
            ],
        }),
        encoding="utf-8",
    )
    puts: list[dict] = []

    def http_put(url, data):
        assert url == "http://inbox/put"
        puts.append(json.loads(data))

    def http_get(url):
        assert url == "http://out/get"
        payload = puts[-1]
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr(
                f"warm_results/{payload['request_id']}.json",
                json.dumps({
                    "status": "completed",
                    "request_id": payload["request_id"],
                    "warm_session_nonce": payload["warm_session_nonce"],
                }),
            )
        return buf.getvalue()

    summary = submit_warm_render_batch(
        manifest_path=manifest_path,
        scenarios_path=scenarios_path,
        out_dir=tmp_path / "results",
        timeout_s=5.0,
        interval_s=0.0,
        stop_after=True,
        session_nonce="batch-session",
        http_put=http_put,
        http_get=http_get,
        clock=_advancing_clock(1.0),
        sleep=lambda s: None,
    )

    assert summary["status"] == "completed"
    assert summary["results_collected"] == 2
    assert [payload["seq"] for payload in puts] == [1, 2, 3]
    assert [payload.get("request_id") for payload in puts[:2]] == ["sink_faucet", "stovetop_knob"]
    assert puts[-1]["stop"] is True
    assert (tmp_path / "results" / "sink_faucet.json").is_file()
    assert (tmp_path / "results" / "warm_render_batch_results.json").is_file()
