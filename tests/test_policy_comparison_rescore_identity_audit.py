"""Ensure a source-bound correction changes identity when its scorer changes."""
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_policy_canary_rescore as rescore


@pytest.mark.parametrize("changed", ["adp_rigid_task_scoring.py", "adp_rigid_retreat_scoring.py", "task_evaluation_surface_target.py"])
def test_dirty_rigid_scorer_cannot_reuse_clean_correction_identity(tmp_path, monkeypatch, changed):
    root = Path(rescore.__file__).resolve().parents[2]
    sources = [
        "adp_task_scoring.py", "adp009d_task_scoring.py",
        "articulation_graph_contract.py", "decision_evidence_contracts.py",
        "adp_rigid_task_scoring.py", "adp_rigid_retreat_scoring.py",
        "task_evaluation_surface_target.py",
    ]
    package = tmp_path / "src/blueprint_pipeline"
    package.mkdir(parents=True)
    for name in sources:
        (package / name).write_bytes((root / "src/blueprint_pipeline" / name).read_bytes())

    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init", "--quiet")
    git("add", "src")
    git("-c", "user.name=Offline Audit", "-c", "user.email=audit@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "Synthetic scorer identity")
    commit = git("rev-parse", "HEAD")
    monkeypatch.setattr(rescore, "__file__", str(package / "task_evaluation_policy_canary_rescore.py"))
    clean = rescore.resolve_scorer_identity(expected_commit=commit)
    with (package / changed).open("a") as stream:
        stream.write("\n# Changed deterministic scoring implementation\n")
    with pytest.raises(rescore.PolicyCanaryRescoreError, match="sources_dirty"):
        rescore.resolve_scorer_identity(expected_commit=commit)
    git("add", "src")
    git("-c", "user.name=Offline Audit", "-c", "user.email=audit@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "Changed scorer")
    revised = rescore.resolve_scorer_identity(expected_commit=git("rev-parse", "HEAD"))
    assert clean["source_files_digest"] != revised["source_files_digest"]
