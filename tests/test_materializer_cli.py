"""The flag-table skeleton two entry-point scripts now share.

These pin behaviour that was moved, not invented: it used to live inline in
`scripts/prepare_artifixer3d_inputs.py`, where nothing exercised it directly --
the contract test there reads the table and never calls it. Now that a second
script depends on the same code, an edit to it can break both at once, so the
subtle parts get held down here.

The subtle part is `accumulate`. A repeatable selector flag distinguishes three
states, and two of them look identical on a command line: "every task" (the
flag was omitted) and "no tasks" (the flag was omitted, and empty means empty).
Collapsing them silently produces a receipt covering nothing, which is a
successful-looking run that did no work. The default the lane declares is what
separates them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.materializer_cli import (
    Param,
    Step,
    build_parser,
    call_arguments,
    run,
)


def _received(**kwargs):
    """A materializer that just reports the keywords it was handed."""

    return dict(kwargs)


def _arguments(step: Step, argv: list[str]) -> dict:
    parser = build_parser({"only": step})
    return call_arguments(step, parser.parse_args(["only", *argv]))


def test_a_repeatable_flag_collects_every_occurrence() -> None:
    step = Step("", _received, {"task_ids": Param("--task-id", accumulate=True)})

    assert _arguments(step, ["--task-id", "a", "--task-id", "b"]) == {
        "task_ids": ("a", "b")
    }


def test_an_omitted_selector_defaulting_to_none_means_every() -> None:
    """`None` reaches the materializer, which reads it as "no restriction"."""

    step = Step("", _received, {"task_ids": Param("--task-id", accumulate=True)})

    assert _arguments(step, []) == {"task_ids": None}


def test_an_omitted_selector_defaulting_to_empty_means_none() -> None:
    """The other reading of the same absent flag, and the lane picks which."""

    step = Step(
        "", _received, {"task_ids": Param("--task-id", accumulate=True, default=())}
    )

    assert _arguments(step, []) == {"task_ids": ()}


def test_a_repeatable_flag_with_an_empty_default_can_actually_be_passed() -> None:
    """`action="append"` appends to whatever default argparse is handed.

    Give it the declared `()` and the first occurrence of the flag calls
    `().append(...)`. The declared default has to be restored afterwards
    instead, or the only way to use the flag is to never use it.
    """

    step = Step(
        "",
        _received,
        {"references": Param("--reference", accumulate=True, default=())},
    )

    assert _arguments(step, ["--reference", "a", "--reference", "b"]) == {
        "references": ("a", "b")
    }


def test_a_json_file_flag_is_read_rather_than_passed_as_a_path(
    tmp_path: Path,
) -> None:
    """Provenance belongs in one signed object, not reassembled from flags."""

    identity = tmp_path / "editor-identity.json"
    identity.write_text(json.dumps({"model": "gpt-image-2"}), encoding="utf-8")
    step = Step("", _received, {"editor": Param("--editor", json_file=True)})

    assert _arguments(step, ["--editor", str(identity)]) == {
        "editor": {"model": "gpt-image-2"}
    }


def test_a_typed_flag_reaches_the_materializer_already_converted() -> None:
    step = Step("", _received, {"radius": Param("--radius", type=int)})

    assert _arguments(step, ["--radius", "12"]) == {"radius": 12}


def test_a_missing_required_flag_is_refused_before_anything_runs() -> None:
    step = Step("", _received, {"output": Param("--output", required=True)})

    with pytest.raises(SystemExit):
        build_parser({"only": step}).parse_args(["only"])


def test_a_successful_step_reports_what_it_wrote(capsys) -> None:
    step = Step(
        "",
        lambda **_: {
            "schema_version": "example.v1",
            "status": "materialized",
            "receipt_digest": "sha256:abc",
            "ignored": "not echoed",
        },
        {"output": Param("--output", required=True)},
    )

    assert run({"only": step}, ["only", "--output", "/tmp/x.json"]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "materialized"
    assert summary["step"] == "only"
    assert summary["provider_mutation_performed"] is False
    # The receipt's own status is renamed so it cannot shadow the summary's.
    assert summary["receipt_status"] == "materialized"
    assert summary["receipt_digest"] == "sha256:abc"
    assert "ignored" not in summary


@pytest.mark.parametrize(
    "failure",
    [ValueError("refused"), OSError("gone"), KeyError("missing"), TypeError("wrong")],
    ids=["value", "os", "key", "type"],
)
def test_a_refused_step_exits_two_and_states_zero_provider_mutation(
    capsys, failure: Exception
) -> None:
    """Fail-closed. A raised traceback would read as a crash, not a refusal."""

    def _refuse(**_):
        raise failure

    step = Step("", _refuse, {"output": Param("--output", required=True)})

    assert run({"only": step}, ["only", "--output", "/tmp/x.json"]) == 2

    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "blocked"
    assert summary["provider_mutation_performed"] is False
    assert summary["blockers"] == [f"{type(failure).__name__}:{failure}"]


def test_an_unexpected_failure_is_not_swallowed_as_a_refusal() -> None:
    """Only the failures a materializer states are turned into `blocked`."""

    def _crash(**_):
        raise RuntimeError("this is a bug, not a refusal")

    step = Step("", _crash, {"output": Param("--output", required=True)})

    with pytest.raises(RuntimeError):
        run({"only": step}, ["only", "--output", "/tmp/x.json"])


def test_naming_no_step_is_refused_rather_than_defaulted() -> None:
    step = Step("", _received, {})

    with pytest.raises(SystemExit):
        build_parser({"only": step}).parse_args([])
