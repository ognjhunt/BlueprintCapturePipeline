"""Negative metadata checks avoid geometry work but cannot admit a prefix."""
import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_sam31_prefix_adoption as adoption
from tests.test_sam31_prefix_adoption import prefix as source_prefix, write, NEW


@pytest.fixture
def prefix(tmp_path):
    return source_prefix.__wrapped__(tmp_path)


def _materialize(value, profile, tmp_path, through_phase):
    zero = write(tmp_path / "zero.json", {"provider": "vast", "status": "observed", "api_confirmed": True,
        "name_prefix": "", "live_resource_count": 0, "resources": [], "http": 200, "observed_at_epoch": 1000.})
    return adoption.materialize_completed_prefix_adoption(
        source_plan_path=value["source_plan"]["path"], source_profile_path=value["source_profile"]["path"],
        parent_request_digest=value["original_parent_request_digest"], through_phase=through_phase,
        current_host_inputs={}, current_provider_profile_path=profile["artifact_references"]["sam31_provider_profile"]["path"],
        current_repo_root=tmp_path, expected_source_commit=NEW, provider_zero_path=zero["path"],
        output_path=tmp_path / "must-not-publish.json", approved_roots=(tmp_path,), queue_root=tmp_path / "queue",
        execution_root=tmp_path / "executions", now_epoch=1001.)


@pytest.mark.parametrize("fault", ["missing", "failed", "other_parent", "other_plan", "other_phase", "ambiguous"])
def test_incomplete_tail_is_refused_before_inherited_science(prefix, tmp_path, monkeypatch, fault):
    value, _plan, profile, _ = prefix
    path = Path(value["phase_records"][-1]["result"]["path"])
    result = json.loads(path.read_text())
    if fault == "missing":
        path.unlink()
    elif fault == "ambiguous":
        duplicate = {**result, "child_id": "sam31-forged"}
        write(path.parent / "sam31-forged.json", duplicate, "result_digest")
    else:
        key, changed = {"failed": ("status", "failed"), "other_parent": ("parent_request_digest", "other"),
            "other_plan": ("plan_digest", "other"), "other_phase": ("phase", "other")}[fault]
        result[key] = changed
        write(path, result, "result_digest")
    monkeypatch.setattr(adoption, "_seed", lambda *args: pytest.fail("negative tail performed inherited validation"))
    with pytest.raises(ValueError, match="sam31_adoption_(prefix_not_terminal|tail_identity_ambiguous)"):
        _materialize(value, profile, tmp_path, "sam31_tracking")
    assert not (tmp_path / "must-not-publish.json").exists()


@pytest.mark.parametrize("inherited_phase", ["sam31_tracking", "segment_cutout"])
def test_nonextending_bound_prefix_is_refused_before_seed(prefix, tmp_path, monkeypatch, inherited_phase):
    value, _plan, profile, _ = prefix
    profile["completed_prefix_adoption"] = write(tmp_path / "inherited.json",
        {"through_phase": inherited_phase}, "adoption_digest")
    value["source_profile"] = write(Path(value["source_profile"]["path"]), profile, "profile_digest")
    monkeypatch.setattr(adoption, "_seed", lambda *args: pytest.fail("nonextending prefix reopened geometry"))
    with pytest.raises(ValueError, match="sam31_adoption_prefix_not_extended"):
        _materialize(value, profile, tmp_path, "sam31_tracking")
    assert not (tmp_path / "must-not-publish.json").exists()


def test_forged_completed_tail_still_requires_full_identity_validation(prefix, tmp_path):
    value, _plan, profile, _ = prefix
    row = value["phase_records"][-1]
    result = json.loads(Path(row["result"]["path"]).read_text())
    result["source_commit"] = NEW
    row["result"] = write(Path(row["result"]["path"]), result, "result_digest")
    # Metadata can justify doing the full work, never accepting its result.
    adoption._require_possible_extension(profile, queue=tmp_path / "queue",
        parent_digest=value["original_parent_request_digest"], plan_digest=value["source_plan"]["sha256"],
        through_phase="sam31_tracking", roots=(tmp_path,))
    with pytest.raises(ValueError, match="sam31_adoption_terminal_result_invalid"):
        adoption._phase_chain(value, (tmp_path,))


def test_completed_tail_cannot_bypass_inherited_validation(prefix, tmp_path, monkeypatch):
    value, _plan, profile, _ = prefix
    def refusing_seed(*args):
        raise ValueError("inherited_science_refused")
    monkeypatch.setattr(adoption, "_seed", refusing_seed)
    with pytest.raises(ValueError, match="inherited_science_refused"):
        _materialize(value, profile, tmp_path, "sam31_tracking")
    assert not (tmp_path / "must-not-publish.json").exists()
