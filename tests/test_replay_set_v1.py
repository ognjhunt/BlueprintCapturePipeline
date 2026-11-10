"""The committed replay set replays every record to the first stop it declares, at $0 and offline.

Beta readiness E3 ("fixes generalize"): a fix to the post-completion chain is checked against several
job records of different kinds at once instead of the one job that stopped. The set's records are
synthetic (built by ``tests/replay_set_v1_build.py`` from the replay tools' own unit-test builders);
they prove the harness and the job-kind paths, not real-scene variety (see the set's README).
"""
# Covers: src/blueprint_pipeline/replay_set.py
# Covers: src/blueprint_pipeline/task_evaluation_downstream_replay.py
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest

from blueprint_pipeline import replay_set
from blueprint_pipeline.task_evaluation_downstream_replay import STAGES, STOP_KINDS

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

def test_the_set_replays_at_its_recorded_clock_whatever_today_is(tmp_path, monkeypatch):
    """The records carry real grant times; a month later they must still replay the same."""
    clock = _manifest()[replay_set.CLOCK_FIELD]
    assert isinstance(clock, int) and clock > 0
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + 30 * 86400)
    seen: list[float] = []
    runner = replay_set.TOOLS["task_evaluation_downstream_replay"]

    def watching(inputs, **kwargs):
        seen.append(time.time())
        return runner(inputs, **kwargs)

    monkeypatch.setitem(replay_set.TOOLS, "task_evaluation_downstream_replay", watching)
    lines: list[str] = []
    result = replay_set.run_set(SET, replay_root=tmp_path / "replays", echo=lines.append)
    assert result["passed"] is True, "\n".join(lines)
    # Each record ran at the recorded instant, and the shifted wall clock is back afterwards.
    assert seen and all(clock <= moment < clock + 3600 for moment in seen)
    assert time.time() > clock + 29 * 86400


def test_a_manifest_clock_that_is_not_a_time_is_refused(tmp_path):
    copy = tmp_path / "set"
    shutil.copytree(SET, copy)
    manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
    manifest[replay_set.CLOCK_FIELD] = "yesterday"
    (copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="replay_clock_epoch"):
        replay_set.run_set(copy, replay_root=tmp_path / "replays", echo=lambda _line: None)
