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


def test_the_worker_satisfies_the_arena_lane_runner_contract() -> None:
    """The transport reads the runner before it will launch it.

    Five tokens, and none of them are arbitrary: the output filename the
    collector looks for, the two revisions whose provenance the receipt has to
    carry, the teardown obligation, and whether a policy was queried. A worker
    missing any of them is refused at bundle-contract time with an empty
    container log, which is a launch spent on a filename.
    """

    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_adp009d_articulated_scene_worker.py"
    ).read_text(encoding="utf-8")

    for token in (
        "adp009d_native_microcheck.json",
        "ARENA_REVISION",
        "ISAAC_LAB_REVISION",
        "provider_zero_required_after_return",
        "candidate_policy_queried",
    ):
        assert token in source, token


def test_the_worker_reports_no_policy_was_queried() -> None:
    """This composition runs scripted controls only; saying otherwise overclaims."""

    module = _worker()
    assert module.CANDIDATE_POLICY_QUERIED is False


def test_the_worker_runs_with_no_arguments_like_the_lane_invokes_it(
    tmp_path, monkeypatch
) -> None:
    """The Arena entrypoint calls its runner bare; config comes from the env.

    Requiring --spec and --output meant the entrypoint reached the worker,
    argparse rejected the empty command line, and the run died with
    adp009d_worker_failed_without_runtime_result - after downloading the
    bundle, booting Isaac and clearing every other gate.
    """

    module = _worker()
    spec = tmp_path / "native" / "adp009d_articulated_scene_spec.json"
    spec.parent.mkdir(parents=True)
    spec.write_text("{}", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("BLUEPRINT_ADP009D_OUTPUT_DIR", str(out))
    monkeypatch.chdir(tmp_path)

    resolved = module._resolve_paths([])

    assert resolved["spec"] == spec.resolve()
    assert resolved["output"].name == "adp009d_native_microcheck.json"
    assert resolved["output"].parent == out.resolve()


def test_explicit_arguments_still_win(tmp_path) -> None:
    """Local debugging invokes it directly; that must keep working."""

    module = _worker()
    spec = tmp_path / "s.json"
    spec.write_text("{}", encoding="utf-8")

    resolved = module._resolve_paths(
        ["--spec", str(spec), "--output", str(tmp_path / "r.json")]
    )

    assert resolved["spec"] == spec.resolve()
    assert resolved["output"].name == "r.json"


def test_a_missing_spec_with_no_arguments_fails_by_name(tmp_path, monkeypatch) -> None:
    """Silently inventing a spec path would hide a packaging mistake."""

    module = _worker()
    monkeypatch.setenv("BLUEPRINT_ADP009D_OUTPUT_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    resolved = module._resolve_paths([])

    assert resolved["spec"] is None
