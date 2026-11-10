"""The committed replay set replays every record to the first stop it declares, at $0 and offline.

Beta readiness E3 ("fixes generalize"): a fix to the post-completion chain is checked against several
job records of different kinds at once instead of the one job that stopped. The set's records are
synthetic (built by ``tests/replay_set_v1_build.py`` from the replay tools' own unit-test builders);
they prove the harness and the job-kind paths, not real-scene variety (see the set's README).
"""
# Covers: src/blueprint_pipeline/replay_set.py
# Covers: src/blueprint_pipeline/task_evaluation_downstream_replay.py
from __future__ import annotations

import contextlib
import datetime
import json
import shutil
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from blueprint_pipeline import replay_set
from blueprint_pipeline.atlas_preview_replay import _swap
from blueprint_pipeline.task_evaluation_downstream_replay import STAGES, STOP_KINDS
from tests.replay_set_v1_build import grant_windows, replay_at

SET = Path(__file__).resolve().parent / "fixtures" / "replay_set_v1"


def _manifest(path: Path = SET) -> dict:
    return json.loads((path / replay_set.MANIFEST_FILENAME).read_text(encoding="utf-8"))


def test_the_set_declares_synthetic_records_of_both_job_kinds_and_the_input_variety():
    manifest = _manifest()
    assert manifest["schema_version"] == replay_set.MANIFEST_SCHEMA and manifest["synthetic"] is True
    records = manifest["records"]
    assert len(records) >= 4
    assert {row["job_kind"] for row in records} == {"rigid_pick_and_place", "navigation"}
    assert {row["input_variety"]["video_resolution"] for row in records} == {"1080p", "4k"}
    assert {row["input_variety"]["clip_length"] for row in records} == {"short", "long"}
    for row in records:
        # Never presented as a retained job: every record says it is synthetic and where it came from.
        assert row["synthetic"] is True and row["provenance"].startswith("synthetic:")
        assert (SET / "records" / row["record_id"]).is_dir()
        assert row["tools"], row["record_id"]
        for tool in row["tools"]:
            assert tool["tool"] in replay_set.TOOLS
            assert tool["until"] in STAGES
            assert tool["expected_stop"]["kind"] in STOP_KINDS
    # Small enough to commit and replay on every relevant pull request.
    size = sum(path.stat().st_size for path in SET.rglob("*") if path.is_file())
    assert size < 400_000, size


def test_every_record_replays_to_its_expected_first_stop_without_network_or_spend(tmp_path):
    lines: list[str] = []
    result = replay_set.run_set(SET, replay_root=tmp_path / "replays", echo=lines.append)
    assert result["passed"] is True, "\n".join(lines)
    assert len(result["rows"]) == sum(len(row["tools"]) for row in _manifest()["records"])
    assert all(row["breaches"] == [] and row["error"] is None for row in result["rows"])
    assert [line.split()[0] for line in lines[:-1]] == ["PASS"] * len(result["rows"])
    # Every record's copy is removed once it has been replayed.
    root = Path(_manifest()["materialize_root"])
    with replay_set._locked(root):  # another worker may be replaying a record right now
        assert not any((root / row["record_id"]).exists() for row in _manifest()["records"])


def test_a_changed_expected_stop_fails_its_record_and_the_command_exits_non_zero(tmp_path, capsys):
    copy = tmp_path / "replay_set_v1"
    shutil.copytree(SET, copy)
    manifest = _manifest(copy)
    record = manifest["records"][0]
    record["tools"][0]["expected_stop"]["stage"] = "team_packet"
    (copy / replay_set.MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    code = replay_set.main(["--set", str(copy), "--record", record["record_id"]])
    out = capsys.readouterr().out
    assert code == 1
    assert out.splitlines()[0].startswith(f"FAIL  {record['record_id']}  ")
    assert out.splitlines()[-1].startswith("0/1 passed")


def test_materialize_puts_a_record_back_byte_for_byte_and_makes_its_empty_directories(tmp_path):
    record = next(row for row in _manifest()["records"] if row["empty_dirs"])
    target = tmp_path / "host"
    replay_set.materialize(SET / "records" / record["record_id"], target, empty_dirs=record["empty_dirs"])
    assert all((target / relative).is_dir() for relative in record["empty_dirs"])
    assert not list(target.rglob("*" + replay_set.GZIP_SUFFIX))


def test_every_record_declares_the_time_it_replays_at_inside_its_own_grant_window():
    manifest = _manifest()
    assert manifest["schema_version"] == "blueprint_replay_set_manifest.v2"
    for row in manifest["records"]:
        record = SET / "records" / row["record_id"]
        declared = replay_set.replay_time(row)
        assert declared is not None, (
            f"{row['record_id']} declares no {replay_set.REPLAY_AT_FIELD}: a committed record must replay at its own "
            "time, or its grants expire with the calendar (rebuild with tests/replay_set_v1_build.py)")
        # Derived from the record's own files, and inside the latest grant's window.
        assert (declared, row["replay_at_basis"]) == replay_at(record), row["record_id"]
        accepted, expires, _ = max(grant_windows(record))
        assert accepted <= declared < expires, row["record_id"]


def test_a_record_with_no_declared_replay_time_fails_and_is_not_replayed_at_todays_clock(tmp_path, monkeypatch):
    copy = tmp_path / "replay_set_v1"
    shutil.copytree(SET, copy)
    manifest = _manifest(copy)
    record = manifest["records"][0]
    del record[replay_set.REPLAY_AT_FIELD]
    (copy / replay_set.MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setitem(replay_set.TOOLS, "task_evaluation_downstream_replay",
                        lambda *args, **kwargs: pytest.fail("replayed without a declared time"))
    lines: list[str] = []
    result = replay_set.run_set(copy, records=[record["record_id"]], echo=lines.append)
    assert result["passed"] is False
    assert result["rows"][0]["error"] == "replay_at_epoch_not_declared"
    assert lines[0].startswith(f"FAIL  {record['record_id']}  ") and lines[0].endswith("replay_at_epoch_not_declared")


class _AnyDatetime(type):
    def __instancecheck__(cls, instance):
        return isinstance(instance, _REAL_DATETIME)


_REAL_DATETIME = datetime.datetime


@contextlib.contextmanager
def _wall_clock_at(epoch: float) -> Iterator[None]:
    """Move the real wall clock (``time.time``/``time_ns`` and ``datetime.now``/``utcnow``) to ``epoch``."""
    offset = epoch - time.time()
    real_time, real_ns = time.time, time.time_ns

    class Moved(_REAL_DATETIME, metaclass=_AnyDatetime):
        @classmethod
        def now(cls, tz=None):
            return _REAL_DATETIME.fromtimestamp(real_time() + offset, tz)

        @classmethod
        def utcnow(cls):
            return _REAL_DATETIME.fromtimestamp(real_time() + offset, datetime.timezone.utc).replace(tzinfo=None)

    with contextlib.ExitStack() as stack:
        _swap(stack, time, "time", lambda: real_time() + offset)
        _swap(stack, time, "time_ns", lambda: real_ns() + int(offset * 1_000_000_000))
        _swap(stack, datetime, "datetime", Moved)
        yield


def _stops(result: dict) -> dict:
    return {row["record_id"]: row["actual"] for row in result["rows"]}


def test_the_set_gives_the_same_first_stops_whatever_day_it_runs(tmp_path):
    """No date bomb: the set's answer does not move with the calendar, ahead or back."""
    manifest = _manifest()
    expected = {row["record_id"]: replay_set._expected(row["tools"][0]["expected_stop"],
                                                       Path(manifest["materialize_root"]) / row["record_id"] / "host")
                for row in manifest["records"]}
    built = min(replay_set.replay_time(row) for row in manifest["records"])
    now, day = time.time(), 86_400
    for label, epoch in (("+1 day", now + day), ("+1 year", now + 365 * day), ("+10 years", now + 3_652 * day),
                         ("before the records were built", built - 30 * day)):
        lines: list[str] = []
        with _wall_clock_at(epoch):
            moved_time, moved_datetime = time.time, datetime.datetime
            result = replay_set.run_set(SET, replay_root=tmp_path / label.replace(" ", "-"), echo=lines.append)
            # The runner gave the clock back as it found it.
            assert time.time is moved_time and datetime.datetime is moved_datetime
        assert _stops(result) == expected, f"wall clock {label}:\n" + "\n".join(lines)
        assert result["passed"] is True, f"wall clock {label}:\n" + "\n".join(lines)


def test_the_pinned_clock_reads_the_declared_time_and_is_restored_also_when_the_body_raises():
    from blueprint_pipeline.core import common

    real = (time.time, time.time_ns, time.gmtime, datetime.datetime)
    with pytest.raises(RuntimeError, match="boom"), replay_set.pinned_clock(1_800_000_060.0):
        assert 1_800_000_060.0 <= time.time() < 1_800_000_070.0
        assert time.time_ns() // 1_000_000_000 == 1_800_000_060
        assert time.gmtime()[:3] == (2027, 1, 15)
        assert datetime.datetime.now(datetime.timezone.utc).date() == datetime.date(2027, 1, 15)
        assert common.utc_now_iso().startswith("2027-01-15T08:01")
        assert isinstance(_REAL_DATETIME.now(), datetime.datetime)
        assert time.monotonic() > 0  # durations are not pinned
        raise RuntimeError("boom")
    assert (time.time, time.time_ns, time.gmtime, datetime.datetime) == real
    assert abs(time.time() - _REAL_DATETIME.now().timestamp()) < 5
