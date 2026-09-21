"""The agent repairs its own model inside one bounded, evidenced session.

The properties under test are the ones that make an iterating model safe to
put in front of paid authoring: it cannot declare its own success, it cannot
outlast its declared turn budget, and every turn it took is recoverable
afterwards.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_object_authoring_session as session
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "attempt"
    root.mkdir()
    return root


def _executor(script):
    """Our validators' verdicts, in order, plus a record of what was proposed."""
    seen: list[str] = []

    def execute(program: str, index: int) -> session.AttemptOutcome:
        seen.append(program)
        accepted, report = script[min(index, len(script)) - 1]
        return session.AttemptOutcome(accepted=accepted, report=report,
                                      artifacts={"stl": "/attempt/candidate.stl"})

    return execute, seen


def _drive(binding, programs, *, stop_on_accept=True):
    """Stand in for the model: call the tool with each program in turn."""
    def invoke_session(bound, turns):
        for program in programs:
            result = bound.invoke({"program": program, "rationale": "revised"})
            if stop_on_accept and result.get("accepted"):
                return result
        return None
    return invoke_session


def test_the_agent_reads_the_real_failure_and_a_later_candidate_is_accepted(tmp_path):
    root = _root(tmp_path)
    execute, seen = _executor([
        (False, {"failure": "geometry_readback_out_of_tolerance: depth 0.31 > 0.22"}),
        (False, {"failure": "appearance_review_failed: handle missing on left face"}),
        (True, {"measurements": {"depth_m": 0.221}}),
    ])
    binding = None

    def invoke_session(bound, turns):
        nonlocal binding
        binding = bound
        assert turns == 3
        first = bound.invoke({"program": "v1", "rationale": "initial"})
        # The model sees exactly what our validators said, so it can repair.
        assert first["accepted"] is False and "0.31" in first["failure"]
        second = bound.invoke({"program": "v2", "rationale": "scaled depth"})
        assert "handle missing" in second["failure"]
        return bound.invoke({"program": "v3", "rationale": "added handle"})

    record = session.run_authoring_session(invoke_session=invoke_session, execute_attempt=execute,
                                           max_turns=3, output_root=root)
    assert seen == ["v1", "v2", "v3"]
    assert record["status"] == "accepted" and record["turns_used"] == 3
    assert record["accepted_turn_index"] == 3
    assert record["accepted_program_digest"] == canonical_digest({"program": "v3"})
    assert record["acceptance_decided_by"] == "blueprint_validators"
    assert binding.tool_id == session.TOOL_ID
    # Every turn is recoverable without re-running the model.
    assert [t["index"] for t in record["turns"]] == [1, 2, 3]
    assert "0.31" in record["turns"][0]["report"]["failure"]
    sealed = json.loads((root / "authoring_session.v1.json").read_text())
    assert sealed == record
    assert sealed["session_digest"] == canonical_digest(sealed, digest_field="session_digest")


def test_a_model_claiming_success_without_an_accepted_candidate_is_a_refusal(tmp_path):
    """Acceptance comes from our validators; the model's own verdict is ignored."""
    root = _root(tmp_path)
    execute, _ = _executor([(False, {"failure": "appearance_review_failed"})])

    def invoke_session(bound, turns):
        bound.invoke({"program": "v1", "rationale": "initial"})
        return {"status": "completed", "summary": "the asset looks correct"}

    record = session.run_authoring_session(invoke_session=invoke_session, execute_attempt=execute,
                                           max_turns=3, output_root=root)
    assert record["status"] == "blocked"
    assert record["blockers"] == ["authoring_session_no_accepted_candidate"]
    assert record["accepted_program_digest"] is None
    assert record["model_declared_success_honoured"] is False


def test_the_turn_budget_holds_even_if_the_model_keeps_calling(tmp_path):
    root = _root(tmp_path)
    execute, seen = _executor([(False, {"failure": "still wrong"})])

    def invoke_session(bound, turns):
        for index in range(10):
            bound.invoke({"program": f"v{index}", "rationale": "again"})

    record = session.run_authoring_session(invoke_session=invoke_session, execute_attempt=execute,
                                           max_turns=3, output_root=root)
    # The executor ran exactly the budgeted number of times; the refusal that
    # stopped the model is recorded rather than raised away.
    assert len(seen) == 3 and record["turns_used"] == 3
    assert record["status"] == "blocked"
    assert "turn_budget_exhausted" in record["invocation_failure"]


def test_a_candidate_that_passes_before_the_budget_ends_the_session(tmp_path):
    root = _root(tmp_path)
    execute, seen = _executor([(True, {"measurements": {"depth_m": 0.22}})])
    record = session.run_authoring_session(
        invoke_session=_drive(None, ["v1", "v2", "v3"]), execute_attempt=execute,
        max_turns=3, output_root=root)
    assert seen == ["v1"] and record["turns_used"] == 1 and record["status"] == "accepted"


def test_the_report_handed_back_is_bounded_and_carries_no_artifact_paths(tmp_path):
    root = _root(tmp_path)
    execute, _ = _executor([(False, {"failure": "x" * (session.MAX_REPORT_BYTES * 2),
                                     "artifacts": {"stl": "/secret/path.stl"}})])
    captured = {}

    def invoke_session(bound, turns):
        captured["report"] = bound.invoke({"program": "v1", "rationale": "initial"})

    session.run_authoring_session(invoke_session=invoke_session, execute_attempt=execute,
                                  max_turns=3, output_root=root)
    report = captured["report"]
    assert report["truncated"] is True and len(json.dumps(report)) < session.MAX_REPORT_BYTES
    assert "artifacts" not in report and "/secret/path.stl" not in json.dumps(report)


def test_an_empty_candidate_is_refused_without_spending_a_turn(tmp_path):
    root = _root(tmp_path)
    execute, seen = _executor([(True, {})])

    def invoke_session(bound, turns):
        with pytest.raises(session.AuthoringSessionError, match="candidate_program_invalid"):
            bound.invoke({"program": "   ", "rationale": "empty"})
        bound.invoke({"program": "v1", "rationale": "real"})

    record = session.run_authoring_session(invoke_session=invoke_session, execute_attempt=execute,
                                           max_turns=3, output_root=root)
    assert seen == ["v1"] and record["turns_used"] == 1


@pytest.mark.parametrize("value,expected", [("", 0), ("3", 3), ("8", 8)])
def test_the_session_is_off_unless_its_budget_is_configured(value, expected):
    assert session.authoring_session_turns({session.TURNS_ENV: value} if value else {}) == expected


@pytest.mark.parametrize("value", ["1", "2", "9", "0", "-1", "many"])
def test_an_unusable_turn_budget_is_refused_rather_than_clamped(value):
    with pytest.raises(session.AuthoringSessionError):
        session.authoring_session_turns({session.TURNS_ENV: value})


def test_out_of_range_budgets_are_refused_at_the_session_boundary_too(tmp_path):
    root = _root(tmp_path)
    execute, _ = _executor([(True, {})])
    for turns in (2, 9):
        with pytest.raises(session.AuthoringSessionError, match="turn_budget_out_of_range"):
            session.run_authoring_session(invoke_session=_drive(None, ["v1"]),
                                          execute_attempt=execute, max_turns=turns, output_root=root)


def test_the_declared_image_ceiling_grows_with_the_stills_actually_sent():
    one = session.session_initial_multimodal_tokens(frames=1)
    four = session.session_initial_multimodal_tokens(frames=4)
    assert four > one and four == 4 * 1_600 + 4_000
    with pytest.raises(session.AuthoringSessionError, match="frame_count_invalid"):
        session.session_initial_multimodal_tokens(frames=0)
