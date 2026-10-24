"""Count-only FLORA monitoring evidence; never a benchmark or grade verdict."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Callable


STAGES = ("startup", "preflight", "terminal", "dense", "rubric")
PREFIX = "BLUEPRINT_FLORA_PROGRESS:"
_MARKER = re.compile(
    r"BLUEPRINT_FLORA_PROGRESS:stage=(startup|preflight|terminal|dense|rubric)"
    r":records=([0-9]+):total=([0-9]+):benchmark_complete=false"
)
_MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024


def progress_marker(stage: str, records: int, total: int) -> str:
    if (stage not in STAGES or type(records) is not int or type(total) is not int
            or not 0 <= records <= total or total <= 0):
        raise ValueError("flora_progress_count_invalid")
    return f"{PREFIX}stage={stage}:records={records}:total={total}:benchmark_complete=false"


def progress_cursor(text: str) -> tuple[int, int] | None:
    """Ignore duplicates, reordered/truncated tails, and unrelated log text."""
    cursors = []
    for line in text.splitlines():
        match = _MARKER.fullmatch(line.strip())
        if match is None:
            continue
        try:
            records, total = int(match[2]), int(match[3])
        except ValueError:
            continue
        if 0 <= records <= total and total > 0:
            cursors.append((STAGES.index(match[1]), records))
    return max(cursors, default=None)


class FloraArtifactProgress:
    """Observe the exact admitted cohort while its existing subprocess runs.

    Completed JSON records show output advancement, not valid grades. Terminal
    receipts and the benchmark report remain the only execution/result verdicts.
    """

    def __init__(self, *, batch_root: Path, episode_ids: set[str], instruction: str,
                 emit: Callable[[str], None] | None = None) -> None:
        if not episode_ids or any(not isinstance(value, str) or not value for value in episode_ids):
            raise ValueError("flora_progress_cohort_invalid")
        self.batch_root = batch_root
        self.episode_ids = episode_ids
        self.instruction = instruction
        self.emit = emit or (lambda line: print(line, flush=True))
        self._counts: dict[str, int] = {}
        self._snapshots: dict[Path, tuple[tuple[int, int], str]] = {}

    def _read(self, path: Path) -> str | None:
        try:
            if any(parent.is_symlink() for parent in (path, path.parent, self.batch_root)):
                return None
            stat = path.stat()
            if not path.is_file() or stat.st_size > _MAX_SNAPSHOT_BYTES:
                return None
            signature = (stat.st_size, stat.st_mtime_ns)
            saved = self._snapshots.get(path)
            if saved is not None and saved[0] == signature:
                return saved[1]
            with path.open("rb") as stream:
                data = stream.read(_MAX_SNAPSHOT_BYTES + 1)
            if len(data) > _MAX_SNAPSHOT_BYTES:
                return None
            text = data.decode("utf-8")
            self._snapshots[path] = (signature, text)
            return text
        except (OSError, UnicodeError):
            return None

    def _unique_ids(self, rows: list, *, require_task: bool) -> int | None:
        seen = set()
        for row in rows:
            if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                    or row["id"] not in self.episode_ids or row["id"] in seen
                    or (require_task and row.get("task") != self.instruction)):
                return None
            seen.add(row["id"])
        return len(seen)

    def _count(self, stage: str) -> int | None:
        path = (self.batch_root / "preflight/diagnostic.json" if stage == "preflight"
                else self.batch_root / (stage + ".jsonl"))
        text = self._read(path)
        if text is None:
            return None
        try:
            if stage == "preflight":
                record = json.loads(text)
                if (not isinstance(record, dict) or type(record.get("requested_episode_count")) is not int
                        or record["requested_episode_count"] != len(self.episode_ids)
                        or not isinstance(record.get("media"), list)):
                    return None
                return self._unique_ids(record["media"], require_task=False)
            # The last line may still be in flight. Only newline-terminated
            # records count; byte growth, partial JSON, and duplicate IDs do not.
            lines = text.splitlines(keepends=True)
            if lines and not lines[-1].endswith("\n"):
                lines.pop()
            rows = [json.loads(line) for line in lines if line.strip()]
            return self._unique_ids(rows, require_task=True)
        except (TypeError, ValueError, RecursionError):
            return None

    def poll(self) -> None:
        for stage in STAGES[1:]:
            count = self._count(stage)
            if count is not None and count > self._counts.get(stage, -1):
                self._counts[stage] = count
                self.emit(progress_marker(stage, count, len(self.episode_ids)))
