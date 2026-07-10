from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from blueprint_pipeline import common


def test_write_json_replace_failure_preserves_previous_complete_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "ledger.json"
    target.write_text('{"generation": "old"}', encoding="utf-8")

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(common.os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected replace failure"):
        common.write_json(target, {"generation": "new"})

    assert json.loads(target.read_text(encoding="utf-8")) == {"generation": "old"}
    assert list(tmp_path.glob(".ledger.json.*.tmp")) == []


def test_concurrent_write_json_never_mixes_payloads(tmp_path: Path) -> None:
    target = tmp_path / "ledger.json"
    barrier = threading.Barrier(8)
    failures: list[BaseException] = []

    def writer(index: int) -> None:
        try:
            barrier.wait()
            common.write_json(target, {"writer": index, "values": [index] * 100})
        except BaseException as exc:  # pragma: no cover - assertion reports worker failures
            failures.append(exc)

    threads = [threading.Thread(target=writer, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["values"] == [payload["writer"]] * 100
    assert list(tmp_path.glob(".ledger.json.*.tmp")) == []


def test_write_text_replaces_symlink_itself_without_overwriting_target(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    target = tmp_path / "result.txt"
    target.symlink_to(outside)

    common.write_text(target, "inside")

    assert not target.is_symlink()
    assert target.read_text(encoding="utf-8") == "inside"
    assert outside.read_text(encoding="utf-8") == "outside"
