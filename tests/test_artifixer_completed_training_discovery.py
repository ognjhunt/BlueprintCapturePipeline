"""A successor discovers this intent's closed appearance launches; nothing else is a candidate."""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

from blueprint_pipeline import task_evaluation_artifixer_pretraining as pretraining
from blueprint_pipeline.artifixer_completed_training_reuse import CANDIDATES_ENV, SOURCE_ENV

INTENT = "sha256:" + "1" * 64
FOREIGN = "sha256:" + "2" * 64


def _launch(root: Path, name: str, *, intent: str = INTENT, closed: bool = True, terminal: bool = True,
            archive: bool = True, profile: bool = True, age: int = 0) -> Path:
    launch = root / name
    job = launch / "allocator" / "scene-configuration-job"
    (job / "vast_provider_run").mkdir(parents=True)
    if profile:
        (launch / "launch_profile.json").write_text(json.dumps({
            "schema_version": "task_evaluation_launch_profile.v1",
            "scene_attempt_binding": {"intent_digest": intent, "intent_id": "scene-x"}}))
    if terminal:
        (launch / "launch_receipt.json").write_text(json.dumps({"status": "completed"}))
    if closed:
        zero = launch / "post_teardown_provider_zero_receipt.json"
        zero.write_text(json.dumps({"status": "provider_zero_confirmed", "provider_zero_verified": True,
                                    "continuing_spend_from_this_run": False}))
        os.utime(zero, (1_000_000 - age, 1_000_000 - age))
    (job / "api_pretraining_receipt.json").write_text("{}")
    with zipfile.ZipFile(job / "api_pretraining_capsule.zip", "w") as z:
        z.writestr("capsule_manifest.json", "{}")
    if archive:
        with zipfile.ZipFile(job / "vast_provider_run" / "vast_provider_runtime_output.zip", "w") as z:
            z.writestr("x", "y")
    return launch


def test_discovery_returns_only_this_intents_closed_launches_newest_first(tmp_path):
    runs = tmp_path / "task-evaluation-launch-runs"
    me = _launch(runs, "self-launch")
    older = _launch(runs, "same-older", age=100)
    newer = _launch(runs, "same-newer", age=10)
    _launch(runs, "foreign", intent=FOREIGN)
    _launch(runs, "still-open", closed=False)
    _launch(runs, "not-terminal", terminal=False)
    _launch(runs, "no-archive", archive=False)
    _launch(runs, "no-profile", profile=False)
    (runs / "same-newer-symlink").symlink_to(newer, target_is_directory=True)
    job = me / "allocator" / "scene-configuration-job"
    assert pretraining.discover_completed_training_candidates(job) == [newer, older]
    assert pretraining.discover_completed_training_candidates(job, limit=1) == [newer]
    # A job outside the launch layout, or a launch without its own intent, discovers nothing.
    assert pretraining.discover_completed_training_candidates(tmp_path / "elsewhere" / "job") == []
    orphan = _launch(tmp_path / "other-root", "orphan", profile=False)
    assert pretraining.discover_completed_training_candidates(orphan / "allocator" / "scene-configuration-job") == []


def test_explicit_source_or_candidate_environment_suppresses_discovery(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pretraining, "discover_completed_training_candidates",
                        lambda job, **kw: calls.append(job) or [tmp_path / "found"])
    # Mirror of the injection rule in prepare_semantics_before_gpu.
    def injected(environment):
        discovered = ([] if environment.get(SOURCE_ENV) or environment.get(CANDIDATES_ENV)
                      else pretraining.discover_completed_training_candidates(tmp_path / "job"))
        return {CANDIDATES_ENV: os.pathsep.join(str(p) for p in discovered)} if discovered else {}
    assert injected({}) == {CANDIDATES_ENV: str(tmp_path / "found")}
    assert injected({SOURCE_ENV: "/explicit"}) == {}
    assert injected({CANDIDATES_ENV: "/explicit-a"}) == {}
    assert len(calls) == 1
