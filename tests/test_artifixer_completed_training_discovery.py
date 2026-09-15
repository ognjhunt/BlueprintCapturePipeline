"""A successor discovers this intent's closed appearance launches; nothing else is a candidate."""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_artifixer_pretraining as pretraining
from blueprint_pipeline.artifixer_completed_training_reuse import CANDIDATES_ENV, SOURCE_ENV

INTENT = "sha256:" + "1" * 64
FOREIGN = "sha256:" + "2" * 64


def _launch(root: Path, name: str, *, intent: str = INTENT, closed: bool = True, terminal: bool = True,
            archive: bool = True, profile: bool = True, age: int = 0, scene: str | None = None,
            namespace: str = "team") -> Path:
    launch = root / name
    job = launch / "allocator" / "scene-configuration-job"
    (job / "vast_provider_run").mkdir(parents=True)
    if profile:
        (launch / "launch_profile.json").write_text(json.dumps({
            "schema_version": "task_evaluation_launch_profile.v1",
            "scene_attempt_binding": {"intent_digest": intent, "intent_id": "scene-x"},
            **({"task_evaluation_run": {"team_namespace": namespace, "scene_id": scene, "task_id": "t"}} if scene else {})}))
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


def test_explicit_source_naming_another_scene_is_ignored_and_recorded(tmp_path):
    """A global pin from another scene's run (2026-09-13: 841757's r24 drop-in) must never block this scene."""
    root = tmp_path / "launch-runs"
    own = _launch(root, "own", scene="840938")
    foreign = _launch(root, "foreign", intent=FOREIGN, scene="841757")
    job = own / "allocator" / "scene-configuration-job"
    environment = {SOURCE_ENV: str(foreign), "BLUEPRINT_ARTIFIXER_COMPLETED_TRAINING_REVIEW_ROOT": "/r24/review",
                   "OTHER": "kept"}
    values, ignored = pretraining.scoped_completed_training_environment(environment, job)
    assert values == {"OTHER": "kept"}
    assert ignored == {"schema_version": pretraining.SOURCE_IGNORED_SCHEMA,
                       "reason": "completed_training_source_out_of_scope",
                       "source_launch_root": str(foreign), "source_scene": ["team", "841757"],
                       "current_scene": ["team", "840938"],
                       "ignored_environment": [SOURCE_ENV, "BLUEPRINT_ARTIFIXER_COMPLETED_TRAINING_REVIEW_ROOT"]}
    # Discovery then runs for this intent as if nothing had been pinned.
    sibling = _launch(root, "sibling", scene="840938")
    assert pretraining.discover_completed_training_candidates(job) == [sibling]


def test_explicit_source_of_this_scene_is_honoured(tmp_path):
    root = tmp_path / "launch-runs"
    own = _launch(root, "own", scene="840938")
    same_scene = _launch(root, "earlier-intent", intent=FOREIGN, scene="840938")
    job = own / "allocator" / "scene-configuration-job"
    environment = {SOURCE_ENV: str(same_scene)}
    assert pretraining.scoped_completed_training_environment(environment, job) == (environment, None)
    # Without a scene on this launch there is nothing to scope against: the explicit source stays authoritative.
    unscoped = _launch(root, "unscoped")
    assert pretraining.scoped_completed_training_environment(
        {SOURCE_ENV: str(tmp_path / "missing")}, unscoped / "allocator" / "scene-configuration-job",
    ) == ({SOURCE_ENV: str(tmp_path / "missing")}, None)


@pytest.mark.parametrize("source_intent,source_scene,retained", [
    (INTENT, "840938", True),
    (FOREIGN, "840938", False),
    (INTENT, "841757", False),
    (None, "840938", False),
])
def test_successor_namespace_preserves_exact_intent_human_approval(
    tmp_path, source_intent, source_scene, retained,
):
    from blueprint_pipeline.task_evaluation_scene_configuration_appearance_review import HUMAN_REVIEW_ENV

    root = tmp_path / "launch-runs"
    own = _launch(root, "successor", scene="840938", namespace="scene-new-attempt")
    source = _launch(root, "completed", intent=source_intent, scene=source_scene,
                     namespace="scene-old-attempt")
    environment = {SOURCE_ENV: str(source), HUMAN_REVIEW_ENV: str(source / "owner-approval"),
                   "OTHER": "kept"}
    values, ignored = pretraining.scoped_completed_training_environment(
        environment, own / "allocator" / "scene-configuration-job")
    if retained:
        assert values == environment
        assert ignored is None
    else:
        assert values == {"OTHER": "kept"}
        assert ignored["ignored_environment"] == [SOURCE_ENV, HUMAN_REVIEW_ENV]
