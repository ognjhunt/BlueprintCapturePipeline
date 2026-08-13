"""The rehearsal has to catch the defect that cost a GPU run, or it is theatre.

SimReady rented a card, ran the Isaac probe, tore the instance down with a 200,
and reported `completed, blockers: []` -- and the launch was blocked, because
the lane sealed under the job root while its provider run lives one directory
deeper. That is a path bug. No GPU was required to find it, only the launch's
own question asked one step earlier.

So the load-bearing test here is the one that reproduces that lane's layout and
asserts the rehearsal would have said `would_block` before any money moved.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "rehearse_lane_terminal_contract",
    REPO_ROOT / "scripts" / "rehearse_lane_terminal_contract.py",
)
rehearsal = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(rehearsal)


def _profile(tmp_path: Path, **overrides) -> Path:
    profile = {
        "schema_version": "task_evaluation_launch_profile.v1",
        "profile_id": "rehearsal-probe",
        "terminal_contract": {
            "result_path": "{launch_run_root}/allocator/result.json",
            "success_statuses": ["completed"],
            "required_values": {"continuing_spend_from_this_run": False, "retry_cap": 0},
            "required_path_fields": [
                "teardown_manifest_path",
                "artifact_manifest_path",
            ],
        },
    }
    profile.update(overrides)
    path = tmp_path / "launch_profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    return path


def test_a_lane_that_seals_where_its_evidence_is_would_pass(tmp_path: Path) -> None:
    receipt = rehearsal.rehearse_lane_terminal_contract(
        profile_path=_profile(tmp_path),
        lane_module="adp_content_agents_vast.py",
    )

    assert receipt["status"] == "would_pass"
    assert receipt["blockers"] == []
    assert receipt["seals_under_nested_attempt"] is False
    # A rehearsal that rented something would be a contradiction in terms.
    assert receipt["provider_mutation_performed"] is False
    assert receipt["paid_resource_allocated"] is False


def test_the_nested_attempt_lane_is_rehearsed_where_it_actually_writes(
    tmp_path: Path,
) -> None:
    """SimReady numbers its attempts; the rehearsal has to follow it there."""

    receipt = rehearsal.rehearse_lane_terminal_contract(
        profile_path=_profile(tmp_path),
        lane_module="public_scene_simready_isaac_vast.py",
    )

    assert receipt["seals_under_nested_attempt"] is True
    assert receipt["status"] == "would_pass"


def test_the_rehearsal_reproduces_the_defect_that_cost_a_gpu_run(
    tmp_path: Path, monkeypatch
) -> None:
    """The load-bearing case.

    With the pre-#501 behaviour -- sealing the job root while the provider run
    sits under `attempts/attempt_001/` -- the rehearsal must refuse, naming the
    same artifacts the live launch named after the money was spent.
    """

    # Force the layout question to answer the way the broken lane behaved: the
    # evidence goes under a numbered attempt, the sealer is pointed at the job.
    monkeypatch.setattr(
        rehearsal, "lane_seals_under_a_nested_attempt", lambda module: False
    )
    original = rehearsal._write_provider_evidence

    def evidence_one_level_deeper(attempt_root: Path) -> None:
        original(attempt_root / "attempts" / "attempt_001")

    monkeypatch.setattr(rehearsal, "_write_provider_evidence", evidence_one_level_deeper)

    receipt = rehearsal.rehearse_lane_terminal_contract(
        profile_path=_profile(tmp_path),
        lane_module="public_scene_simready_isaac_vast.py",
    )

    assert receipt["status"] == "would_block"
    assert sorted(receipt["blockers"]) == [
        "allocator_terminal_artifact_missing:artifact_manifest_path",
        "allocator_terminal_artifact_missing:teardown_manifest_path",
        "allocator_terminal_status_not_success",
    ]
    # And the sealer says why, rather than shrugging -- the #501 half of it.
    assert any(
        item.startswith("terminal_artifacts_not_found_under_attempt_root")
        for item in receipt["sealed_result_blockers"]
    )


def test_a_profile_demanding_a_field_no_lane_seals_would_block(tmp_path: Path) -> None:
    """A terminal contract can be wrong too, and that is just as expensive."""

    profile = _profile(tmp_path)
    value = json.loads(profile.read_text(encoding="utf-8"))
    value["terminal_contract"]["required_path_fields"].append("nonexistent_manifest_path")
    profile.write_text(json.dumps(value), encoding="utf-8")

    receipt = rehearsal.rehearse_lane_terminal_contract(
        profile_path=profile, lane_module="adp_content_agents_vast.py"
    )

    assert receipt["status"] == "would_block"
    assert "allocator_terminal_artifact_missing:nonexistent_manifest_path" in (
        receipt["blockers"]
    )


def test_a_lane_whose_layout_cannot_be_read_is_refused_not_guessed(
    tmp_path: Path,
) -> None:
    with pytest.raises(rehearsal.LaneRehearsalError) as excinfo:
        rehearsal.rehearse_lane_terminal_contract(
            profile_path=_profile(tmp_path), lane_module="common.py"
        )

    assert str(excinfo.value).startswith("lane_provider_run_root_not_found:")


def test_a_missing_lane_module_is_named(tmp_path: Path) -> None:
    with pytest.raises(rehearsal.LaneRehearsalError) as excinfo:
        rehearsal.rehearse_lane_terminal_contract(
            profile_path=_profile(tmp_path), lane_module="no_such_lane_vast.py"
        )

    assert str(excinfo.value) == "lane_module_missing:no_such_lane_vast.py"
