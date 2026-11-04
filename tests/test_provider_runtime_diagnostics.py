# Covers (for impacted-test selection):
#   src/blueprint_pipeline/provider_runtime_diagnostics.py
#   src/blueprint_pipeline/provider_output_remote_collection.py
#   src/blueprint_pipeline/beta_policy_screening_worker.py
#   src/blueprint_pipeline/bounded_error.py
"""A failed provider run's own account is summarized, bounded and redacted, at inspection time."""

from __future__ import annotations

import io
import json
import re
import urllib.error
import zipfile
from pathlib import Path

from blueprint_pipeline import provider_runtime_diagnostics as diagnostics
from blueprint_pipeline.beta_policy_screening_worker import backfill_runtime_diagnostics
from blueprint_pipeline.provider_output_remote_collection import RemoteProviderOutputCollector
from tests.provider_output_fixtures import (
    DEFLATED,
    SECRET,
    URL,
    Entry,
    RangeStore,
    VirtualFile,
    Zeros,
    build_zip,
    no_disk_writes,
    quick10_shaped_archive,
)

RESULT = "native_task_arena_policy_canary_session_result.v1.json"
FAKE_TOKEN = "sk-live-FAKEtoken0123456789abcdefXYZ"
FAKE_BEARER = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJmYWtlIn0.Zm9vYmFyQkFaMTIz"
FAKE_BLOB = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZnaGlqa2xtbm9wcXJzdHV2d3h5ejAxMjM0NTY3ODk="
PNG_MARKER = b"PNG_BYTES_MUST_NOT_BE_READ"


def _cell_result(index: int) -> dict:
    return {
        "schema_version": "native_task_arena_policy_canary_session_result.v1",
        "status": "blocked", "selected_cell_index": index, "episodes": [], "policy_loads": [],
        "candidate_policy_queried": False, "preload_observation_gate": None,
        "session_failure_type": "LaunchFailure",
        "session_failure_codes": ["runtime_launch_post_start_identity_failed"],
        "session_failure_source_trace": [
            {"file": "provider_session.py", "function": "execute_paired_session", "line": 975},
            {"file": "/workspace/bundle/provider_runtime/launch.py", "function": "launch_runtime", "line": 342},
            {"file": "launch.py", "function": "not a name()", "line": 7},
        ],
        "session_closeout": {"status": "runtime_closed", "failure_type": "RuntimeError"},
        "diagnostics": {"reason": f"identity readback failed; Authorization: Bearer {FAKE_BEARER}"},
        "task_success_contract": {"criteria": {"large": ["content", "never", "copied"]}},
    }


def _worker_console(index: int) -> str:
    noise = [f"[{line:04d}] [Info] kit loading extension {line}" for line in range(200)]
    traceback = [
        "Traceback (most recent call last):",
        '  File "/workspace/bundle/provider_runtime/launch.py", line 342, in launch_runtime',
        f"RuntimeError: post-start identity failed (cell {index}) api_key={FAKE_TOKEN}",
    ]
    tail = [f"[{line:04d}] [Info] shutting down {line}" for line in range(120)]
    return "\n".join([*noise[:100], *traceback, *noise[100:], *tail[:60], f"payload {FAKE_BLOB}",
                      *tail[60:]]) + "\n"


def _failed_run_files(cells: int = 2) -> dict[str, bytes]:
    files: dict[str, bytes] = {
        RESULT: json.dumps({"status": "blocked", "blockers": ["policy_canary_worker_failed_without_result"],
                            "worker_exit_code": 1}).encode(),
        "policy_server.candidate.log": (
            "server starting\nCUDA Error: out of memory while loading weights\n"
            + f"token={FAKE_TOKEN}\nserver stopped\n").encode(),
        "policy_server_receipt.candidate.json": json.dumps({"status": "passed", "blockers": []}).encode(),
    }
    for index in range(cells):
        root = f"cell_runs/{index:02d}"
        files[f"{root}/{RESULT}"] = json.dumps(_cell_result(index)).encode()
        files[f"{root}/worker_console.log"] = _worker_console(index).encode()
        files[f"{root}/cell_progress.log"] = b"stage=static_preflight_passed\nstage=launch_started\n"
        files[f"{root}/telemetry.jsonl"] = b'{"message": 1}\n'
        files[f"{root}/episodes/e0.score_receipt.json"] = json.dumps({"status": "scored"}).encode()
        files[f"{root}/frames/000001.png"] = b"\x89PNG" + PNG_MARKER
    return files


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def _write_zip(path: Path, files: dict[str, bytes]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_zip_bytes(files))
    return path


def test_summary_keeps_cell_failures_and_log_errors_and_never_reads_images(tmp_path, monkeypatch):
    archive = _write_zip(tmp_path / "out.zip", _failed_run_files())
    opened: list[str] = []
    real_open = zipfile.ZipFile.open

    def recording_open(self, name, *args, **kwargs):
        opened.append(name.filename if isinstance(name, zipfile.ZipInfo) else name)
        return real_open(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", recording_open)
    summary = diagnostics.summarize_zip_path(archive)

    assert summary["schema_version"] == "provider_runtime_diagnostics.v1"
    assert summary["status"] == "completed" and summary["claims_success"] is False
    # Images, telemetry streams and per-episode JSON are never opened.
    assert not [name for name in opened if not name.endswith((".json", ".log", ".txt"))]
    assert not [name for name in opened if "/episodes/" in name]
    text = json.dumps(summary)
    assert PNG_MARKER.decode() not in text
    unread = {row["member"]: row["reason"] for row in summary["unread_members"]}
    assert unread["cell_runs/00/frames/000001.png"] == "not_a_text_member"
    assert unread["cell_runs/00/telemetry.jsonl"] == "not_a_text_member"
    assert "cell_runs/00/episodes/e0.score_receipt.json" not in unread
    assert summary["unread_member_counts"]["nested_json_member"] == 2

    rows = {row["member"]: row for row in summary["results"]}
    cell = rows[f"cell_runs/01/{RESULT}"]
    assert cell["directory"] == "cell_runs/01" and cell["selected_cell_index"] == 1
    assert (cell["status"], cell["session_failure_type"]) == ("blocked", "LaunchFailure")
    assert cell["session_failure_codes"] == ["runtime_launch_post_start_identity_failed"]
    assert cell["episode_count"] == 0 and cell["candidate_policy_queried"] is False
    # Function, file name and line only: the directory is dropped, an invalid frame skipped.
    assert cell["session_failure_source_trace"] == [
        {"function": "execute_paired_session", "file": "provider_session.py", "line": 975},
        {"function": "launch_runtime", "file": "launch.py", "line": 342}]
    fields = {row["field"]: row["message"] for row in cell["error_messages"]}
    assert fields["session_closeout.failure_type"] == "RuntimeError"
    assert fields["diagnostics.reason"].startswith("identity readback failed")
    assert "content" not in json.dumps(cell)  # other fields are never copied
    assert rows[RESULT]["blockers"] == ["policy_canary_worker_failed_without_result"]

    logs = {row["member"]: row for row in summary["logs"]}
    console = logs["cell_runs/00/worker_console.log"]
    assert console["line_count"] == 324 and len(console["tail"]) == diagnostics.TAIL_LINES
    assert f"payload {FAKE_BLOB}" not in console["tail"] and "payload <redacted>" in console["tail"]
    assert console["tail"][-1] == "[0119] [Info] shutting down 119"
    # The traceback precedes the tail window, so it is kept as numbered error lines.
    assert [line.split(": ", 1)[0] for line in console["error_lines"]] == ["101", "103"]
    assert console["error_lines"][0] == "101: Traceback (most recent call last):"
    assert "RuntimeError: post-start identity failed (cell 0)" in console["error_lines"][1]
    server = logs["policy_server.candidate.log"]
    assert server["error_line_count"] == 1 and "CUDA Error: out of memory" in " ".join(server["tail"])

    # Bearer tokens, key/token assignments and long base64 blobs are redacted.
    for secret in (FAKE_TOKEN, FAKE_BEARER, FAKE_BLOB):
        assert secret not in text
    assert summary["redacted_line_count"] >= 4 and "<redacted>" in text


def test_each_excerpt_and_the_whole_summary_stay_within_their_limits(tmp_path, monkeypatch):
    noisy = ("".join(f"Error {line} " + "x" * 1200 + "\n" for line in range(3000))
             + "".join(f"info {line} " + "z" * 1200 + "\n" for line in range(100)))
    files = {f"cell_runs/{index:02d}/worker_console.log": noisy.encode() for index in range(3)}
    files["huge_line.log"] = b"A" * 20_000 + b"\nRuntimeError: after a long line\n"
    files["big_result.json"] = json.dumps({"status": "blocked", "padding": "y" * (9 * 1024**2)}).encode()
    summary = diagnostics.summarize_zip_path(_write_zip(tmp_path / "out.zip", files))

    for row in summary["logs"]:
        excerpt = [*row["error_lines"], *row["tail"]]
        assert len(excerpt) <= diagnostics.EXCERPT_LINES
        assert len(json.dumps(row["error_lines"])) + len(json.dumps(row["tail"])) <= diagnostics.EXCERPT_BYTES
        assert all(len(line) <= diagnostics.LINE_CHARS + 8 for line in excerpt)
    console = next(row for row in summary["logs"] if row["member"] == "cell_runs/00/worker_console.log")
    assert console["excerpt_trimmed"] is True and console["error_line_count"] == 3000
    long_line = next(row for row in summary["logs"] if row["member"] == "huge_line.log")
    assert long_line["over_long_line_count"] == 1 and long_line["tail"][-1] == "RuntimeError: after a long line"
    assert {"member": "big_result.json", "reason": "over_member_size_cap"} in summary["unread_members"]
    assert len(json.dumps(summary)) <= diagnostics.MAX_SUMMARY_BYTES

    # The whole summary is fitted to its cap by shortening the longest excerpts.
    monkeypatch.setattr(diagnostics, "MAX_SUMMARY_BYTES", 40 * 1024)
    small = diagnostics.summarize_zip_path(tmp_path / "out.zip")
    assert len(json.dumps(small, sort_keys=True).encode()) <= 40 * 1024
    assert all(row["excerpt_trimmed"] for row in small["logs"] if row["member"].endswith("worker_console.log"))

    compact = diagnostics.compact(summary)
    assert len(json.dumps(compact, sort_keys=True).encode()) <= diagnostics.COMPACT_MAX_BYTES


def test_compact_keeps_failing_cells_and_last_errors_for_the_command_result(tmp_path):
    summary = diagnostics.summarize_zip_path(_write_zip(tmp_path / "out.zip", _failed_run_files(cells=10)))
    compact = diagnostics.compact(summary)

    assert len(json.dumps(compact, sort_keys=True).encode()) <= diagnostics.COMPACT_MAX_BYTES
    failing = {row["member"]: row for row in compact["failing_results"]}
    assert "policy_server_receipt.candidate.json" not in failing  # passed, no blockers
    assert {row.get("selected_cell_index") for row in failing.values()} >= set(range(10))
    cell = failing[f"cell_runs/04/{RESULT}"]
    assert cell["session_failure_source_trace"][-1] == {"function": "launch_runtime", "file": "launch.py",
                                                        "line": 342}
    logs = {row["member"]: row for row in compact["log_errors"]}
    assert any("post-start identity failed (cell 4)" in line
               for line in logs["cell_runs/04/worker_console.log"]["last_error_lines"])
    assert FAKE_TOKEN not in json.dumps(compact)


def test_an_unreadable_or_absent_archive_is_recorded_not_raised(tmp_path):
    broken = tmp_path / "broken.zip"
    broken.write_bytes(b"not a zip")
    assert diagnostics.summarize_zip_path(broken)["status"] == "failed"
    assert diagnostics.summarize_zip_path(tmp_path / "absent.zip")["status"] == "missing"
    target = _write_zip(tmp_path / "real.zip", _failed_run_files(cells=1))
    (tmp_path / "link.zip").symlink_to(target)
    assert diagnostics.summarize_zip_path(tmp_path / "link.zip")["status"] == "missing"
    unsafe = _write_zip(tmp_path / "unsafe.zip", {"../escape.log": b"Error outside\n", "ok.log": b"fine\n"})
    summary = diagnostics.summarize_zip_path(unsafe)
    assert [row["member"] for row in summary["logs"]] == ["ok.log"]
    assert summary["unread_members"] == [{"member": "../escape.log", "reason": "unsafe_member_name"}]
    assert not (tmp_path.parent / "escape.log").exists()


def test_the_adapter_keeps_a_summary_only_for_an_archive_it_observed(tmp_path):
    archive = _write_zip(tmp_path / "local" / diagnostics.OUTPUT_ZIP_NAME, _failed_run_files(cells=1))
    kept = diagnostics.retain_for_transfer(tmp_path / "local", {"status": "completed"}, archive)
    assert kept["written"] is True and kept["path"] == str(tmp_path / "local" / diagnostics.FILENAME)
    assert json.loads((tmp_path / "local" / diagnostics.FILENAME).read_text())["archive_source"] == "local_file"

    # A remote observation carries its own summary; nothing is read from disk.
    remote = {"status": "completed", "delivery": "remote_only",
              "runtime_diagnostics": diagnostics.summarize_zip_path(archive)}
    (tmp_path / "remote").mkdir()
    compact = diagnostics.retain_for_transfer(tmp_path / "remote", remote, tmp_path / "remote" / "absent.zip")
    assert compact["failing_results"] and (tmp_path / "remote" / diagnostics.FILENAME).is_file()
    receipt = diagnostics.receipt_transfer(remote, compact)
    assert receipt["runtime_diagnostics"] == compact and "results" not in receipt["runtime_diagnostics"]
    assert remote["runtime_diagnostics"]["results"]  # the caller's transfer is unchanged

    blocked = {"status": "blocked", "blockers": ["provider_output_not_ready"]}
    assert diagnostics.retain_for_transfer(tmp_path / "local", blocked, archive) is None
    (tmp_path / "linked").mkdir()
    (tmp_path / "linked" / diagnostics.OUTPUT_ZIP_NAME).symlink_to(archive)
    assert diagnostics.retain_for_transfer(tmp_path / "linked", {}, tmp_path / "linked" /
                                           diagnostics.OUTPUT_ZIP_NAME) is None
    assert not (tmp_path / "linked" / diagnostics.FILENAME).exists()


def _run_dir(root: Path, name: str, *, command_result: bool = True, archive: bool = True) -> Path:
    run = root / name / "vast_provider_run"
    run.mkdir(parents=True)
    if archive:
        _write_zip(run / diagnostics.OUTPUT_ZIP_NAME, _failed_run_files(cells=1))
    if command_result:
        (run / diagnostics.COMMAND_RESULT_NAME).write_text("{}")
    return run


def test_backfill_is_bounded_and_idempotent(tmp_path):
    runs = [_run_dir(tmp_path, f"run{index}") for index in range(7)]
    unfinished = _run_dir(tmp_path, "unfinished", command_result=False)
    no_archive = _run_dir(tmp_path, "no_archive", archive=False)
    done = _run_dir(tmp_path, "done")
    (done / diagnostics.FILENAME).write_text('{"status": "completed"}')
    every = sorted([*runs, unfinished, no_archive, done])

    first = diagnostics.backfill(every, max_writes=5)
    assert (len(first["written"]), first["pending"]) == (5, True)
    second = diagnostics.backfill(every, max_writes=5)
    assert (len(second["written"]), second["pending"]) == (2, False)
    assert sorted(first["written"] + second["written"]) == sorted(str(run) for run in runs)
    before = {run: (run / diagnostics.FILENAME).read_bytes() for run in runs}
    assert diagnostics.backfill(every, max_writes=5) == {"status": "completed", "written": [], "pending": False}
    assert {run: (run / diagnostics.FILENAME).read_bytes() for run in runs} == before
    assert (done / diagnostics.FILENAME).read_text() == '{"status": "completed"}'
    assert not (unfinished / diagnostics.FILENAME).exists() and not (no_archive / diagnostics.FILENAME).exists()
    written = json.loads((runs[0] / diagnostics.FILENAME).read_text())
    assert written["backfilled"] is True and written["status"] == "completed"
    assert any(row.get("session_failure_type") == "LaunchFailure" for row in written["results"])


def test_beta_worker_backfills_local_and_work_volume_attempts(tmp_path):
    root = tmp_path / "queue"
    attempts = root / "attempts"
    attempts.mkdir(parents=True)
    _run_dir(attempts / "screen-local" / "allocator" / "attempts", "attempt_001")
    work = tmp_path / "work" / "attempts"
    _run_dir(work / "screen-remote" / "allocator" / "attempts", "attempt_001")
    (attempts / "screen-remote.attempt_dir.json").write_text(json.dumps(
        {"attempt_id": "screen-remote", "attempt_dir": str(work / "screen-remote")}))
    # A pointer whose directory does not carry its own attempt id is ignored.
    _run_dir(work / "screen-other" / "allocator" / "attempts", "attempt_001")
    (attempts / "screen-forged.attempt_dir.json").write_text(json.dumps(
        {"attempt_id": "screen-forged", "attempt_dir": str(work / "screen-other")}))

    result = backfill_runtime_diagnostics(root)

    assert result["status"] == "completed" and result["pending"] is False
    assert sorted(Path(path).parents[3].name for path in result["written"]) == ["screen-local", "screen-remote"]
    assert not (work / "screen-other/allocator/attempts/attempt_001/vast_provider_run" / diagnostics.FILENAME).exists()
    assert backfill_runtime_diagnostics(root)["written"] == []
    assert backfill_runtime_diagnostics(tmp_path / "missing-root")["status"] == "failed"


def _collect(store, **options):
    options.setdefault("maximum_archive_bytes", store.object.size)
    collector = RemoteProviderOutputCollector(expected_video_count=0, opener=options.pop("opener", store.opener),
                                              **options)
    return collector(url=URL, output_path=Path("/nonexistent/vast_provider_runtime_output.zip"))


def test_remote_collection_reads_failure_diagnostics_by_range_without_writing(tmp_path, monkeypatch):
    store = RangeStore(_zip_bytes(_failed_run_files()))
    monkeypatch.chdir(tmp_path)
    with no_disk_writes(monkeypatch) as attempts:
        transfer = _collect(store)

    assert attempts == [] and list(tmp_path.iterdir()) == []
    assert transfer["status"] == "completed" and transfer["inspection"]["runtime_result_status"] == "blocked"
    summary = transfer["runtime_diagnostics"]
    assert (summary["status"], summary["archive_source"]) == ("completed", "remote_range")
    assert any(row.get("selected_cell_index") == 1 for row in summary["results"])
    text = json.dumps(transfer)
    assert SECRET not in text and FAKE_TOKEN not in text and PNG_MARKER.decode() not in text


def test_remote_collection_skips_diagnostics_for_a_clean_completion(tmp_path):
    q10 = quick10_shaped_archive(cells=1)
    transfer = _collect(RangeStore(q10.archive))
    assert transfer["status"] == "completed"
    assert transfer["runtime_diagnostics"]["status"] == "not_collected"
    assert transfer["runtime_diagnostics"]["reason"] == "runtime_result_completed_without_blockers"


def _spread_archive():
    """A blocked result, then bulk, then a log, then more bulk: the log has its own blocks."""
    return build_zip([
        Entry(RESULT, json.dumps({"status": "blocked", "blockers": ["worker_failed"]}).encode(), method=DEFLATED),
        Entry("cell_runs/00/frames/000001.png", Zeros(2 * 1024**2)),
        Entry("cell_runs/00/worker_console.log", b"Traceback (most recent call last):\nRuntimeError: x\n",
              method=DEFLATED),
        Entry("cell_runs/01/frames/000001.png", Zeros(2 * 1024**2)),
        Entry("cell_runs/01/worker_console.log", b"RuntimeError: y\n", method=DEFLATED),
        Entry("cell_runs/02/frames/000001.png", Zeros(2 * 1024**2)),
    ])


def test_a_transport_failure_reading_diagnostics_keeps_the_observation(tmp_path):
    archive = _spread_archive()
    with zipfile.ZipFile(VirtualFile(archive)) as opened:
        log_offset = opened.getinfo("cell_runs/00/worker_console.log").header_offset
    store = RangeStore(archive)

    def failing(request, timeout, policy):
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", dict(request.header_items()).get("Range", ""))
        if match and int(match[1]) <= log_offset <= int(match[2]):
            raise urllib.error.HTTPError("redacted", 500, "Internal Error", {}, None)
        return store.opener(request, timeout, policy)

    transfer = _collect(store, opener=failing, block_bytes=128 * 1024)

    assert transfer["status"] == "completed" and transfer["blockers"] == []
    assert transfer["inspection"]["runtime_result_status"] == "blocked"
    summary = transfer["runtime_diagnostics"]
    assert summary["status"] == "partial"
    assert "provider_runtime_diagnostics_transport_failed" in summary["blockers"]
    assert [row["member"] for row in summary["results"]] == [RESULT]


def test_remote_diagnostics_stop_at_their_transfer_budget(tmp_path):
    store = RangeStore(_spread_archive())
    transfer = _collect(store, block_bytes=128 * 1024, diagnostics_transfer_bytes=1)

    summary = transfer["runtime_diagnostics"]
    assert transfer["status"] == "completed" and summary["status"] == "partial"
    assert summary["unread_member_counts"]["read_budget_exhausted"] >= 1
    assert summary["transferred_bytes"] <= 2 * 128 * 1024
