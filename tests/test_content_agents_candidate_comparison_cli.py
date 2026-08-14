"""The Content Agents candidate comparison has to be sealable from a command line.

`materialize_content_agents_candidate_comparison` seals the terminal evidence of
several paid Content Agents runs into the one receipt a human reviews before any
backend is preferred over another. It landed with no entry point: no script named
it, its own module carries no `main()`, and no sibling module exposed it. So the
comparison could be *produced* only from a Python session, and
`tests/test_materializer_reachability.py` went to 74 against a budget of 73 that
is documented to fall and never rise.

Same defect as #512 (lanes), #520 (bundle modules), #523 (authority
materializers), the ArtiFixer3D input chain, and the semantic-teacher image-edit
closer, in a sixth scope.

The flag table *is* the call, and the contract below derives the left column from
the function's own signature, so a new keyword upstream fails here until a flag
supplies it rather than being defaulted away inside a receipt that claims to bind
it.

One thing is deliberately not flag-per-field. A candidate is a nine-key mapping
with a nested list of review frames, and there are up to one per replacement slot
per admitted backend. Spreading that across flags would put the operator in the
business of reassembling a document the comparison then validates as a whole, so
`--candidates` reads one JSON array instead. Reading it as JSON rather than as a
path is what the contract below pins: the materializer takes
`Sequence[Mapping[str, Any]]`, and handing it a path string produces a refusal
that reads like bad evidence rather than a bad command line.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "seal_content_agents_candidate_comparison.py"


def _load():
    name = "seal_content_agents_candidate_comparison"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # `dataclass` resolves annotations through `sys.modules[cls.__module__]`,
    # so a file-loaded module has to be registered before it is executed.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sealer = _load()

STEP = "comparison"


def test_the_step_is_exposed() -> None:
    assert sorted(sealer.STEPS) == [STEP]


def test_every_materializer_keyword_has_a_flag() -> None:
    """Derived from the signature, so upstream cannot outrun the table."""

    entry = sealer.STEPS[STEP]
    upstream = {
        name
        for name, parameter in inspect.signature(entry.materialize).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }

    missing = upstream - set(entry.params)
    assert not missing, f"comparison cannot supply {sorted(missing)} from a command line"


def test_no_flag_invents_a_keyword_the_materializer_will_not_take() -> None:
    entry = sealer.STEPS[STEP]

    assert not set(entry.params) - set(inspect.signature(entry.materialize).parameters)


def test_flags_are_distinct() -> None:
    flags = [param.flag for param in sealer.STEPS[STEP].params.values()]

    assert len(flags) == len(set(flags))


def test_a_keyword_upstream_requires_is_required_here() -> None:
    """And one it defaults is not, so an omitted flag cannot become a `None` path.

    `generated_at` defaults upstream and is genuinely optional -- the receipt
    stamps its own time. `candidates` and `output_path` do not, so a flag that
    could be omitted would reach a materializer with no default for it.
    """

    entry = sealer.STEPS[STEP]
    signature = inspect.signature(entry.materialize)

    for keyword, param in entry.params.items():
        upstream_required = signature.parameters[keyword].default is inspect.Parameter.empty
        assert param.required is upstream_required, (
            f"{param.flag} is {'optional' if not param.required else 'required'} but "
            f"upstream {'requires' if upstream_required else 'defaults'} it"
        )


def test_the_candidates_flag_is_read_as_json_not_passed_as_a_path() -> None:
    """Upstream takes `Sequence[Mapping]`; a path string is a silently wrong shape.

    A string is a `Sequence`, so handing the path through would iterate its
    characters and refuse deep inside candidate normalization, blaming the
    evidence for what was a command-line error.
    """

    assert sealer.STEPS[STEP].params["candidates"].json_file is True


def test_missing_evidence_is_refused_without_touching_a_provider(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reads retained bytes and seals; it allocates nothing and rents nothing."""

    candidates = tmp_path / "candidates.json"
    candidates.write_text(
        json.dumps([{"bundle_receipt_path": str(tmp_path / "absent.json")}]),
        encoding="utf-8",
    )
    output = tmp_path / "comparison.json"

    code = sealer.main(
        [STEP, "--candidates", str(candidates), "--output", str(output)]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["status"] == "blocked"
    assert payload["provider_mutation_performed"] is False
    assert payload["blockers"], "a refusal has to name its cause"
    assert not output.exists()


def test_a_real_candidate_matrix_seals_through_the_command_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The entry point has to seal, not only refuse.

    A CLI proven only by its refusals is one that could reject every input and
    still pass. This drives the real fixture matrix -- two replacement slots by
    both admitted backends -- and checks the receipt on disk is the one the
    summary reports.
    """

    from tests.test_adp_content_agents_candidate_comparison import _matrix

    from blueprint_pipeline.adp_content_agents_candidate_comparison import (
        validate_content_agents_candidate_comparison,
    )

    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps(_matrix(tmp_path)), encoding="utf-8")
    output = tmp_path / "comparison.json"

    code = sealer.main(
        [
            STEP,
            "--candidates",
            str(candidates),
            "--output",
            str(output),
            "--generated-at",
            "2026-08-14T00:00:00Z",
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert code == 0
    sealed = json.loads(output.read_text(encoding="utf-8"))
    assert sealed["candidate_count"] == 4
    assert sealed["status"] == "completed_candidates_ready_for_within_task_visual_review"
    # The receipt the operator records must be the one that was written.
    assert summary["receipt_digest"] == sealed["receipt_digest"]
    assert summary["provider_mutation_performed"] is False
    # And it has to survive the module's own validator, not just be well-formed.
    assert validate_content_agents_candidate_comparison(sealed) == sealed


def test_the_optional_timestamp_flag_actually_reaches_the_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An optional flag that is parsed and dropped is worse than no flag."""

    from tests.test_adp_content_agents_candidate_comparison import _matrix

    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps(_matrix(tmp_path)), encoding="utf-8")
    output = tmp_path / "comparison.json"

    sealer.main(
        [
            STEP,
            "--candidates",
            str(candidates),
            "--output",
            str(output),
            "--generated-at",
            "2026-08-14T00:00:00Z",
        ]
    )
    capsys.readouterr()

    assert json.loads(output.read_text(encoding="utf-8"))["generated_at"] == (
        "2026-08-14T00:00:00Z"
    )


def test_an_empty_candidate_set_is_refused_rather_than_sealed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty array would otherwise seal a comparison covering nothing."""

    candidates = tmp_path / "candidates.json"
    candidates.write_text("[]", encoding="utf-8")
    output = tmp_path / "comparison.json"

    code = sealer.main(
        [STEP, "--candidates", str(candidates), "--output", str(output)]
    )

    assert code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
    assert not output.exists()
