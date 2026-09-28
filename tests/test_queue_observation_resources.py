"""Covers: bounded queue observation JSON work, bytes, clocks and output limits."""
import importlib
import inspect
import json

import pytest


def _module():
    return importlib.import_module("blueprint_pipeline.control_plane_queue_observation")


def _observe(module, root, clock=lambda: 0):
    return module.observe_queue_states([module.QueueRootContract(str(root), ("pending",))],
                                       observed_at_epoch=10, monotonic=clock)


def _write(root, name="a.json", raw=b'{"x":1}'):
    (root / "pending").mkdir(exist_ok=True)
    path = root / "pending" / name
    path.write_bytes(raw)
    return path


@pytest.mark.parametrize("raw", [b'{"x":[[0]]}', b'{"x":{"y":{}}}'])
def test_depth_limit_refuses_before_json_parser(tmp_path, monkeypatch, raw):
    module = _module()
    _write(tmp_path, raw=raw)
    monkeypatch.setattr(module, "MAX_DEPTH", 2)
    monkeypatch.setattr(module.json, "loads", lambda *a, **kw: pytest.fail("parser reached"))
    result = _observe(module, tmp_path)
    assert not result.complete and "queue_depth_limit" in result.blockers


def test_value_limit_is_global_and_precedes_parser(tmp_path, monkeypatch):
    module = _module()
    _write(tmp_path)
    _write(tmp_path, "b.json")
    monkeypatch.setattr(module, "MAX_VALUES", 5)  # Each object+key+number is three tokens.
    real = module.json.loads
    calls = []
    monkeypatch.setattr(module.json, "loads", lambda *a, **kw: (calls.append(True), real(*a, **kw))[1])
    result = _observe(module, tmp_path)
    assert not result.complete and "queue_values_limit" in result.blockers
    assert len(calls) == 1


@pytest.mark.parametrize("raw", [b'{"x":"long whitespace {} [ ] \\\" end"}', b'{"x":"\\u007b\\u005b"}', b'  {"x": [true,null, false,-1.2e2]}\n'])
def test_preflight_does_not_count_structure_inside_strings_as_depth(tmp_path, monkeypatch, raw):
    module = _module()
    _write(tmp_path, raw=raw)
    monkeypatch.setattr(module, "MAX_DEPTH", 2)
    assert _observe(module, tmp_path).complete


@pytest.mark.parametrize("content", ["a" * 40, " " * 40])
def test_mid_preflight_deadline_stops_before_loads(tmp_path, monkeypatch, content):
    module = _module()
    raw = json.dumps({"x": content}).encode() if content[0] == "a" else (content + "{}").encode()
    _write(tmp_path, raw=raw)
    monkeypatch.setattr(module, "PREFLIGHT_CHECK_CHARS", 4)
    calls = []
    real_loads = module.json.loads
    monkeypatch.setattr(module.json, "loads", lambda *a, **kw: (calls.append(True), real_loads(*a, **kw))[1])
    checkpoints = 0

    def clock():
        nonlocal checkpoints
        frame = inspect.currentframe().f_back
        if frame.f_back.f_code.co_name == "preflight":
            checkpoints += 1
        return 6 if checkpoints >= 3 else 0

    result = _observe(module, tmp_path, clock)
    assert not result.complete and "queue_deadline_exceeded" in result.blockers
    assert calls == [] and checkpoints == 3 and result.rows == ()


@pytest.mark.parametrize("cap_name,cap,blocker", [("MAX_ROW_BYTES", 6, "queue_row_bytes_limit"), ("MAX_TOTAL_BYTES", 6, "queue_bytes_limit"), ("MAX_ROWS", 0, "queue_rows_limit"), ("MAX_ENTRIES", 0, "queue_entries_limit"), ("MAX_OUTPUT_BYTES", 1, "queue_output_limit")])
def test_resource_caps_never_return_complete_truncation(tmp_path, monkeypatch, cap_name, cap, blocker):
    module = _module()
    _write(tmp_path)
    monkeypatch.setattr(module, cap_name, cap)
    result = _observe(module, tmp_path)
    assert not result.complete and blocker in result.blockers


@pytest.mark.parametrize("cap_name", ["MAX_ROW_BYTES", "MAX_TOTAL_BYTES"])
def test_exact_inclusive_read_size_cap_passes(tmp_path, monkeypatch, cap_name):
    module = _module()
    raw = b'{"x":1}'
    _write(tmp_path, raw=raw)
    monkeypatch.setattr(module, cap_name, len(raw))
    assert _observe(module, tmp_path).complete


def test_growth_sentinel_counts_against_aggregate_cap_before_parse(tmp_path, monkeypatch):
    module = _module()
    path = _write(tmp_path)
    monkeypatch.setattr(module, "MAX_TOTAL_BYTES", path.stat().st_size)
    real = module.os.read
    changed = False

    def read(fd, n):
        nonlocal changed
        if not changed:
            changed = True
            with path.open("ab") as stream:
                stream.write(b" ")
        return real(fd, n)

    monkeypatch.setattr(module.os, "read", read)
    monkeypatch.setattr(module.json, "loads", lambda *a, **kw: pytest.fail("parser reached"))
    result = _observe(module, tmp_path)
    assert not result.complete and "queue_bytes_limit" in result.blockers


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, None, -1])
def test_invalid_or_backwards_clock_gives_incomplete_evidence(tmp_path, value):
    module = _module()
    clock_values = iter([0, value])
    result = _observe(module, tmp_path, lambda: next(clock_values, value))
    assert not result.complete and result.rows == () and "queue_clock_invalid" in result.blockers


@pytest.mark.parametrize("stage", ["second_row", "result"])
def test_accepted_positive_then_deadline_has_empty_incomplete_fallback(tmp_path, monkeypatch, stage):
    module = _module()
    _write(tmp_path)
    _write(tmp_path, "b.json")
    expired = False
    if stage == "second_row":
        real = module._Scan.read_row

        def read(self, *args):
            nonlocal expired
            if args[-1] == "b.json":
                assert len(self.rows) == 1
                expired = True
            return real(self, *args)

        monkeypatch.setattr(module._Scan, "read_row", read)
    else:
        real = module._Scan.result

        def result(self):
            nonlocal expired
            assert len(self.rows) == 2
            expired = True
            return real(self)

        monkeypatch.setattr(module._Scan, "result", result)
    result = _observe(module, tmp_path, lambda: 6 if expired else 0)
    assert not result.complete and result.rows == () and "queue_deadline_exceeded" in result.blockers


def test_output_cap_accounts_for_raw_text_escaping(tmp_path, monkeypatch):
    module = _module()
    _write(tmp_path, raw=b'{"x":"\\n\\n\\n"}')
    ordinary = _observe(module, tmp_path)
    from dataclasses import asdict
    exact = len(json.dumps(asdict(ordinary), ensure_ascii=False, allow_nan=False).encode())
    monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", exact)
    assert _observe(module, tmp_path).complete
    monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", exact - 1)
    result = _observe(module, tmp_path)
    assert not result.complete and result.rows == () and "queue_output_limit" in result.blockers


def test_missing_root_still_retains_explicit_selected_scope(tmp_path):
    module = _module()
    result = _observe(module, tmp_path / "missing")
    assert len(result.roots) == 1
    assert result.roots[0].selected_states == ("pending",)
    assert result.roots[0].root_identity is None and result.roots[0].missing_states == ()
