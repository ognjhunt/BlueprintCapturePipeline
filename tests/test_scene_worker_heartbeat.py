from __future__ import annotations

import importlib.util
from pathlib import Path


def _worker():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_adp009d_articulated_scene_worker.py"
    )
    spec = importlib.util.spec_from_file_location("scene_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_each_phase_announces_itself_on_stdout(capsys) -> None:
    """A silent worker and a hung worker look identical from outside.

    The transport watches stdout for progress and kills a run that goes quiet.
    The scene worker wrote its phases only into the result file, so a slow
    Arena boot tripped the heartbeat and came back as
    vast_heartbeat_no_log_progress_timeout with an empty container log - a
    launch spent on an ambiguity rather than an answer.
    """

    module = _worker()
    result = {"phase_reached": "start"}

    module._phase(result, "arena_imported")

    captured = capsys.readouterr().out
    assert "BLUEPRINT_WAM_RUNTIME_PHASE" in captured
    assert "arena_imported" in captured
    assert result["phase_reached"] == "arena_imported"


def test_the_phase_line_is_flushed_immediately(capsys) -> None:
    """Buffered progress reaches the watcher after the timeout, which is never."""

    module = _worker()
    module._phase({"phase_reached": "start"}, "environment_built")

    assert "environment_built" in capsys.readouterr().out


def test_every_phase_the_worker_reaches_is_announced() -> None:
    """A phase recorded but not printed is invisible to the heartbeat."""

    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_adp009d_articulated_scene_worker.py"
    ).read_text(encoding="utf-8")

    # Exactly one assignment, the one inside _phase. Any other is a phase the
    # heartbeat will never see.
    assert source.count('result["phase_reached"] =') == 1, (
        "phases must go through _phase so they reach stdout"
    )
    for phase in (
        "arena_imported",
        "assets_resolved",
        "embodiment_configured",
        "environment_built",
        "articulation_bound",
        "adapter_wired",
        "controls_complete",
    ):
        assert f'"{phase}"' in source, phase
