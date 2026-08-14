"""The semantic-teacher image-edit lane has to be closeable from a command line.

`semantic_teacher_image_edit_paid_lane` produces the three receipts that make a
paid run terminal -- the retained result, the provider-zero proof, and the
no-allocation closeout -- and none of them could be produced by any script or by
any module carrying a `main()`. So a run could be *started* from a production
path and only *closed* from a Python session, which is the state that leaves an
attempt with no terminal artifact and a provider bill nobody reconciled.

That is the same defect as #512 (lanes), #520 (bundle modules), #523 (authority
materializers) and the input chain, in a fifth scope. It is also what pushed
`tests/test_materializer_reachability.py` from 73 to 76: the lane landed with
three unreachable materializers, and the budget only ratchets down.

The flag table in the closer *is* the call, and the contract below derives the
left column from each function's own signature -- 30 keyword-only parameters
across the three, every one of them required. A hand-listed table is how #523's
first cut silently dropped four of six, and on a closeout a dropped path is a
piece of evidence the receipt claims to bind and does not.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "retain_semantic_teacher_image_edit_receipts.py"


def _load():
    name = "retain_semantic_teacher_image_edit_receipts"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # `dataclass` resolves annotations through `sys.modules[cls.__module__]`,
    # so a file-loaded module has to be registered before it is executed.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


closer = _load()

STEP_NAMES = ["result", "provider-zero", "no-allocation-closeout"]


def test_every_step_of_the_lane_is_exposed() -> None:
    """All three terminal receipts, or the lane is closeable only in part."""

    assert sorted(closer.STEPS) == sorted(STEP_NAMES)


@pytest.mark.parametrize("step", STEP_NAMES)
def test_every_materializer_keyword_has_a_flag(step: str) -> None:
    """Derives the requirement from the signature, so upstream cannot outrun it.

    A new keyword on any of the three fails here until a flag supplies it,
    rather than being defaulted away inside a receipt that claims to bind it.
    """

    entry = closer.STEPS[step]
    upstream = {
        name
        for name, parameter in inspect.signature(entry.materialize).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }

    missing = upstream - set(entry.params)
    assert not missing, f"{step} cannot supply {sorted(missing)} from a command line"


@pytest.mark.parametrize("step", STEP_NAMES)
def test_no_flag_invents_a_keyword_the_materializer_will_not_take(step: str) -> None:
    entry = closer.STEPS[step]
    upstream = set(inspect.signature(entry.materialize).parameters)

    assert not set(entry.params) - upstream


@pytest.mark.parametrize("step", STEP_NAMES)
def test_flags_are_distinct_within_a_step(step: str) -> None:
    """Two keywords sharing a flag would let one silently overwrite the other."""

    flags = [param.flag for param in closer.STEPS[step].params.values()]

    assert len(flags) == len(set(flags))


@pytest.mark.parametrize("step", STEP_NAMES)
def test_every_evidence_path_is_required(step: str) -> None:
    """Upstream takes no defaults, so an optional flag here would be a `None` path.

    Every one of the 30 keyword-only parameters across the three functions is
    required upstream. A flag that could be omitted would reach a materializer
    that has no default for it.
    """

    entry = closer.STEPS[step]
    signature = inspect.signature(entry.materialize)

    for keyword, param in entry.params.items():
        assert signature.parameters[keyword].default is inspect.Parameter.empty
        assert param.required, f"{step} {param.flag} is optional but upstream requires it"


def test_the_two_counts_are_parsed_as_integers() -> None:
    """`expected_task_count` compares against a length; a string never matches."""

    params = closer.STEPS["result"].params

    assert params["expected_task_count"].type is int
    assert params["expected_camera_count"].type is int


@pytest.mark.parametrize("step", STEP_NAMES)
def test_missing_evidence_is_refused_without_touching_a_provider(
    step: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A closer that rents nothing is the whole point; it only reads and seals."""

    entry = closer.STEPS[step]
    output = tmp_path / f"{step}.json"
    argv: list[str] = [step]
    for keyword, param in entry.params.items():
        if keyword == "output_path":
            argv += [param.flag, str(output)]
        elif param.type is int:
            argv += [param.flag, "1"]
        elif keyword == "reason":
            argv += [param.flag, "allocation never became possible"]
        else:
            argv += [param.flag, str(tmp_path / "absent.json")]

    code = closer.main(argv)

    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["status"] == "blocked"
    assert payload["provider_mutation_performed"] is False
    assert payload["blockers"], "a refusal has to name its cause"
    assert not output.exists()
