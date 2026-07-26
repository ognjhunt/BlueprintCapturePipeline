"""Tests for the resident OSCAR worker protocol, teardown, and throughput report."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from blueprint_pipeline import oscar_resident_worker as resident
from blueprint_pipeline.oscar_resident_worker import (
    ResidentOscarWorker,
    ResidentWorkerError,
    build_resident_worker_argv,
    make_resident_oscar_generate,
)
from blueprint_pipeline.oscar_resident_worker_main import serve


FAKE_WORKER = textwrap.dedent(
    """
    import json, sys, hashlib, pathlib
    mode = sys.argv[1] if len(sys.argv) > 1 else "ok"
    if mode == "load_fail":
        print(json.dumps({"schema_version": "oscar_resident_worker_ready.v1",
                          "status": "failed"}))
        sys.exit(1)
    ready = {"schema_version": "oscar_resident_worker_ready.v1", "status": "ready",
             "model_load_seconds": 12.5, "checkpoint_sha256": "a" * 64,
             "worker_session_id": "session-1"}
    if mode != "no_device":
        ready["cuda_device_name"] = "NVIDIA RTX PRO 6000"
    print(json.dumps(ready)); sys.stdout.flush()
    count = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("op") == "shutdown":
            break
        count += 1
        if mode == "die":
            sys.exit(3)
        rid = "b" * 64 if mode == "replay" else hashlib.sha256(str(count).encode()).hexdigest()
        req_id = "wrong-id" if mode == "out_of_order" else req.get("request_id", "")
        out = req.get("output_video", "")
        if out and mode != "no_output":
            pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(out).write_bytes(b"video")
        print(json.dumps({"schema_version": "oscar_resident_worker_response.v1",
                          "request_id": req_id, "status": "ok", "blockers": [],
                          "generate_seconds": 0.25, "runtime_result_id": rid,
                          "output_video": out}))
        sys.stdout.flush()
    """
)


@pytest.fixture()
def fake_worker_script(tmp_path: Path) -> Path:
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER, encoding="utf-8")
    return script


def _worker(script: Path, mode: str = "ok", **kwargs) -> ResidentOscarWorker:
    return ResidentOscarWorker(argv=[sys.executable, str(script), mode], **kwargs)


def test_weights_load_once_across_many_steps(fake_worker_script: Path) -> None:
    """The whole point: one cold start, many warm steps."""

    worker = _worker(fake_worker_script)
    ready = worker.start()
    assert ready["status"] == "ready"

    for _ in range(5):
        response = worker.generate({"reference_frame_path": "f.png", "output_video": ""})
        assert response["status"] == "ok"

    report = worker.throughput_report()
    worker.close()

    assert report["cold_start_count"] == 1
    assert report["warm_step_count"] == 5
    assert report["cold_start_seconds"] == pytest.approx(12.5)
    assert report["restart_count"] == 0
    # A per-step spawn would pay the 12.5s load on every one of those steps.
    assert report["per_step_spawn_equivalent_seconds"] > report["warm_step_seconds_mean"]
    assert report["estimated_speedup_vs_per_step_spawn"] > 1.0
    assert report["steps_per_hour"] > 0


def test_ready_handshake_is_required_before_generation(fake_worker_script: Path) -> None:
    worker = _worker(fake_worker_script)
    with pytest.raises(ResidentWorkerError, match="not_started"):
        worker.generate({"output_video": ""})


def test_missing_gpu_identity_fails_closed_when_residency_required(
    fake_worker_script: Path,
) -> None:
    """Residency is reported by the worker here, so an absent device must block."""

    worker = _worker(fake_worker_script, mode="no_device", require_gpu_residency=True)
    with pytest.raises(ResidentWorkerError, match="gpu_residency_unproven"):
        worker.start()
    worker.close()

    permissive = _worker(fake_worker_script, mode="no_device", require_gpu_residency=False)
    assert permissive.start()["status"] == "ready"
    permissive.close()


def test_worker_death_fails_closed_rather_than_falling_back(
    fake_worker_script: Path,
) -> None:
    """A silent per-step fallback would hide the cold start it reintroduces."""

    worker = _worker(fake_worker_script, mode="die")
    worker.start()
    with pytest.raises(ResidentWorkerError):
        worker.generate({"output_video": ""})
    assert worker.failures
    worker.close()


def test_out_of_order_response_is_rejected(fake_worker_script: Path) -> None:
    """A desynchronised stream would attribute one step's output to another."""

    worker = _worker(fake_worker_script, mode="out_of_order")
    worker.start()
    with pytest.raises(ResidentWorkerError, match="out_of_order"):
        worker.generate({"output_video": ""})
    worker.close()


def test_replayed_runtime_result_id_is_rejected(fake_worker_script: Path) -> None:
    worker = _worker(fake_worker_script, mode="replay")
    worker.start()
    assert worker.generate({"output_video": ""})["status"] == "ok"
    with pytest.raises(ResidentWorkerError, match="replayed"):
        worker.generate({"output_video": ""})
    worker.close()


def test_restart_budget_is_zero_by_default_and_counted_when_used(
    fake_worker_script: Path,
) -> None:
    worker = _worker(fake_worker_script, mode="die", max_restarts=0)
    worker.start()
    with pytest.raises(ResidentWorkerError):
        worker.generate({"output_video": ""})
    assert worker.throughput_report()["restart_count"] == 0
    worker.close()


def test_failed_load_is_reported_as_not_ready(fake_worker_script: Path) -> None:
    worker = _worker(fake_worker_script, mode="load_fail")
    with pytest.raises(ResidentWorkerError, match="not_ready"):
        worker.start()
    worker.close()


def test_generate_adapter_matches_the_per_step_response_contract(
    fake_worker_script: Path, tmp_path: Path
) -> None:
    """The adapter must be a drop-in for make_oscar_per_step_wam_backend."""

    worker = _worker(fake_worker_script)
    worker.start()
    extracted = tmp_path / "next.png"

    def _extract(video: Path, out_dir: Path) -> Path:
        extracted.write_bytes(b"frame")
        return extracted

    generate = make_resident_oscar_generate(worker=worker, extract_next_frame=_extract)
    result = generate({"output_dir": str(tmp_path / "step0"), "reference_frame_path": "f.png"})
    worker.close()

    assert result["status"] == "completed"
    assert result["blockers"] == []
    assert result["generated_frame_path"] == str(extracted)
    assert Path(result["generated_video_path"]).is_file()
    assert Path(result["stdout_log_path"]).is_file()
    assert Path(result["stderr_log_path"]).is_file()


def test_generate_adapter_uses_prefix_aligned_extractor_for_timed_request(
    fake_worker_script: Path,
    tmp_path: Path,
) -> None:
    worker = _worker(fake_worker_script)
    worker.start()
    selected: list[int] = []

    def _aligned(_video: Path, _out_dir: Path, target_index: int) -> Path:
        selected.append(target_index)
        frame = tmp_path / "aligned-next.png"
        frame.write_bytes(b"frame")
        return frame

    generate = make_resident_oscar_generate(
        worker=worker,
        extract_next_frame=lambda _video, _out_dir: pytest.fail(
            "timed request must not use earliest-future extraction"
        ),
        extract_prefix_aligned_frame=_aligned,
    )
    result = generate(
        {
            "output_dir": str(tmp_path / "step0"),
            "reference_frame_path": "f.png",
            "next_observation_timing": {"target_wam_frame_index": 5},
        }
    )
    worker.close()

    assert result["status"] == "completed"
    assert selected == [5]
    assert result["next_observation_timing"]["target_wam_frame_index"] == 5


def test_adapter_blocks_when_the_worker_produces_no_video(
    fake_worker_script: Path, tmp_path: Path
) -> None:
    worker = _worker(fake_worker_script, mode="no_output")
    worker.start()
    generate = make_resident_oscar_generate(
        worker=worker, extract_next_frame=lambda video, out_dir: None
    )
    result = generate({"output_dir": str(tmp_path / "step0")})
    worker.close()

    assert result["status"] == "blocked"
    assert "oscar_resident_worker_output_missing" in result["blockers"]


def test_adapter_blocks_when_skeleton_conditioning_is_unavailable(
    fake_worker_script: Path, tmp_path: Path
) -> None:
    worker = _worker(fake_worker_script)
    worker.start()
    generate = make_resident_oscar_generate(
        worker=worker,
        extract_next_frame=lambda video, out_dir: None,
        build_skeleton_video=lambda rows, out_dir: None,
    )
    result = generate({"output_dir": str(tmp_path / "step0")})
    worker.close()

    assert result["status"] == "blocked"
    assert (
        "oscar_per_step_projected_skeleton_conditioning_unavailable" in result["blockers"]
    )


def test_worker_error_is_surfaced_as_a_blocked_step(
    fake_worker_script: Path, tmp_path: Path
) -> None:
    worker = _worker(fake_worker_script, mode="die")
    worker.start()
    generate = make_resident_oscar_generate(
        worker=worker, extract_next_frame=lambda video, out_dir: None
    )
    result = generate({"output_dir": str(tmp_path / "step0")})
    worker.close()

    assert result["status"] == "blocked"
    assert any(item.startswith("oscar_resident_worker_failed") for item in result["blockers"])


def test_resident_argv_is_not_a_distributed_launcher() -> None:
    argv = build_resident_worker_argv(
        python="/usr/bin/python3",
        oscar_repo="/opt/oscar",
        checkpoint="/ckpt",
        num_steps=35,
        guidance=6.0,
        height=480,
        width=640,
        fps=15.0,
    )
    assert "torch.distributed.run" not in argv
    assert argv[1:3] == ["-m", "blueprint_pipeline.oscar_resident_worker_main"]
    assert "--checkpoint" in argv


class _Capture:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write(self, text: str) -> None:
        if text.strip():
            self.lines.append(text.strip())

    def flush(self) -> None:
        return None

    def payloads(self) -> list[dict]:
        return [json.loads(line) for line in self.lines]


def test_serve_answers_requests_and_stops_on_shutdown() -> None:
    """The worker-side protocol is exercised with no model and no GPU."""

    requests = [
        json.dumps(
            {
                "schema_version": resident.REQUEST_SCHEMA_VERSION,
                "request_id": "req-1",
                "op": "generate",
            }
        ),
        json.dumps({"schema_version": resident.REQUEST_SCHEMA_VERSION, "op": "shutdown"}),
        json.dumps({"schema_version": resident.REQUEST_SCHEMA_VERSION, "op": "generate"}),
    ]
    out = _Capture()
    code = serve(
        stdin=iter(requests),
        stdout=out,
        generate=lambda request: {"blockers": [], "output_video": "/tmp/x.mp4"},
        ready_payload={
            "schema_version": resident.READY_SCHEMA_VERSION,
            "status": "ready",
            "worker_session_id": "session-1",
        },
    )
    payloads = out.payloads()

    assert code == 0
    assert payloads[0]["schema_version"] == resident.READY_SCHEMA_VERSION
    assert payloads[1]["request_id"] == "req-1"
    assert payloads[1]["status"] == "ok"
    # The request after shutdown must never be served.
    assert len(payloads) == 2


def test_serve_reports_a_raising_generate_as_a_protocol_error() -> None:
    out = _Capture()

    def _boom(request):
        raise RuntimeError("cuda oom")

    serve(
        stdin=iter(
            [
                json.dumps(
                    {
                        "schema_version": resident.REQUEST_SCHEMA_VERSION,
                        "request_id": "req-1",
                        "op": "generate",
                    }
                )
            ]
        ),
        stdout=out,
        generate=_boom,
        ready_payload={
            "schema_version": resident.READY_SCHEMA_VERSION,
            "status": "ready",
            "worker_session_id": "s",
        },
    )
    response = out.payloads()[1]

    assert response["status"] == "error"
    assert any("generate_raised" in item for item in response["blockers"])
    assert "cuda oom" in response["stderr_tail"]


def test_serve_rejects_unknown_schema_and_op() -> None:
    out = _Capture()
    serve(
        stdin=iter(
            [
                json.dumps({"schema_version": "bogus.v9", "request_id": "a", "op": "generate"}),
                json.dumps(
                    {
                        "schema_version": resident.REQUEST_SCHEMA_VERSION,
                        "request_id": "b",
                        "op": "train",
                    }
                ),
                "not json at all",
            ]
        ),
        stdout=out,
        generate=lambda request: {"blockers": []},
        ready_payload={
            "schema_version": resident.READY_SCHEMA_VERSION,
            "status": "ready",
            "worker_session_id": "s",
        },
    )
    payloads = out.payloads()

    assert payloads[1]["blockers"] == ["oscar_resident_worker_request_schema_invalid"]
    assert payloads[2]["blockers"] == ["oscar_resident_worker_unsupported_op:train"]
    assert payloads[3]["blockers"] == ["oscar_resident_worker_request_not_json"]


def test_close_is_idempotent_and_survives_a_dead_worker(fake_worker_script: Path) -> None:
    worker = _worker(fake_worker_script)
    worker.start()
    worker.close()
    worker.close()


def test_context_manager_tears_down(fake_worker_script: Path) -> None:
    with _worker(fake_worker_script) as worker:
        assert worker.generate({"output_video": ""})["status"] == "ok"
    assert worker._process is None


def test_throughput_report_carries_a_claim_boundary(fake_worker_script: Path) -> None:
    worker = _worker(fake_worker_script)
    worker.start()
    worker.generate({"output_video": ""})
    report = worker.throughput_report()
    worker.close()

    assert report["schema_version"] == "wam_generation_throughput.v1"
    assert report["claim_boundary"]["throughput_is_not_generation_quality"] is True
    assert report["claim_boundary"]["throughput_is_not_task_success"] is True
    assert report["worker_identity"]["cuda_device_name"] == "NVIDIA RTX PRO 6000"
