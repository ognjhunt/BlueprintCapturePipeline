"""A deploy does not retire a website job whose paid launch already left progression.

Before: on the first tick under a new release, ``_release_successor`` retired the
attempt once the preparation queue row was terminal and ownership read zero. A
launch still pending in the launch queue, or still in transit through the
WebApp, is neither, so the job was rebuilt on the new release and could reserve
and launch the same paid construction again while the dispatcher ran the first.
Now the attempt waits on its own release (``running`` / ``previous_release``)
until that launch has a terminal receipt; the completion join closes it.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from blueprint_pipeline import public_scene_host_input_intake
from blueprint_pipeline import task_evaluation_scene_configuration_activation_automation as automation
from blueprint_pipeline import task_evaluation_scene_configuration_submission_publication as publication
from blueprint_pipeline import task_evaluation_scene_preparation_completion as completion
from blueprint_pipeline import task_evaluation_scene_progression as progression
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_release_hold import previous_launch_in_flight
from tests.test_task_evaluation_scene_configuration_activation_automation import _provider_zero, _sealed, _write
from tests.test_website_scene_configuration_launch_in_progression import (
    _Tick, _activation_enabled, _authority_for, _launch_environment,
)

NEW_COMMIT = "b" * 40


def _launched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A website job on release A whose tick submitted its one paid launch (today's code path)."""
    from tests.test_task_evaluation_scene_configuration_submission import SHA
    from tests.test_website_native_submission import prepare_website_construction

    prepare_website_construction(tmp_path, monkeypatch, False)
    config_path, moment = _activation_enabled(tmp_path)
    config = json.loads(config_path.read_text())
    launches = tmp_path / "launches"
    for state in ("pending", "processing"):
        (launches / state).mkdir(parents=True)
    config["launch_queue_root"] = str(launches)
    _write(config_path, _sealed(config, "config_digest"))
    _launch_environment(tmp_path, monkeypatch)
    now = datetime.now(timezone.utc)
    tick = _Tick(tmp_path / "preparations", tmp_path / "activation-intents")
    calls: list[dict] = []

    def launcher(**kwargs):
        calls.append(kwargs)
        return automation.advance_owned_scene_configuration(
            **kwargs, provider_zero_collector=lambda: _provider_zero(now - timedelta(seconds=20)),
            lineage_publisher_factory=lambda: tick.lineage, release_window_publisher_factory=lambda: tick.window,
            submitter=tick.submit, now=now)

    def run(at: float) -> dict:
        [row] = progression.process_scene_intents(
            config_path=config_path, now=at, configuration_launcher=launcher,
            activation_provisioner=automation.provision_scene_configuration_activation_intent)["results"]
        return row

    run(moment)
    _authority_for(tmp_path, json.loads(_projection(tmp_path).read_text())["state"]["scene_configuration_launch"],
                   commit=SHA, now=now)
    row = run(moment + 60)
    launched = json.loads(_projection(tmp_path).read_text())["state"]["scene_configuration_launch"]
    assert (row["phase"], launched["status"]) == ("scene_configuration", "scene_configuration_launch_queued")
    assert len(tick.submissions) == 1 and len(calls) == 2
    return config, run, moment, tick, calls, launched["launch_id"]


def _projection(tmp_path: Path) -> Path:
    return next((tmp_path / "intents").glob("scene-*/progression.json"))


def _deploy(config: dict, monkeypatch: pytest.MonkeyPatch, commit: str) -> None:
    """Move the live release: a new commit and a new runtime digest, as a door deploy does."""
    path = Path(config["release_binding_path"])
    release = json.loads(path.read_text())
    release.update(source_commit=commit, runtime_digest="sha256:" + "e" * 64)
    release["release_digest"] = canonical_digest(release, digest_field="release_digest")
    path.write_text(json.dumps(release))
    monkeypatch.setattr(public_scene_host_input_intake, "_verified_checkout_head", lambda: commit)
    monkeypatch.setattr(publication, "_verified_checkout_head", lambda: commit)


def _footprint(tmp_path: Path) -> dict:
    intent = _projection(tmp_path).parent
    return {"preparation_attempts": sorted(p.name for p in (intent / "preparation-attempts").glob("*.json")),
            "paid_rows": sorted(p.name for p in (intent / "attempts").glob("*.json")),
            "activation_links": sorted(p.name for p in (intent / "preparations").glob("*.activation.json")),
            "prepared_requests": sorted(p.name for p in (tmp_path / "preparations").rglob("*.json")
                                        if p.parent.name in {"pending", "processing"})}


@pytest.mark.parametrize("where", ["queued", "running", "in_transit"])
def test_a_deploy_never_rebuilds_or_relaunches_a_job_whose_launch_left(tmp_path, monkeypatch, where):
    config, run, moment, tick, calls, launch_id = _launched(tmp_path, monkeypatch)
    if where == "queued":
        _write(Path(config["launch_queue_root"]) / "pending" / (launch_id + "-request.json"), {"launch_id": launch_id})
    elif where == "running":
        (Path(config["launch_execution_root"]) / launch_id).mkdir(parents=True)
    # What production reads for a launch that is pending or in transit: provider
    # zero, no live lease, nothing in launch processing. (A running launch would
    # refuse here; the hold answers before this is reached.)
    from blueprint_pipeline import task_evaluation_scene_progression_recovery as recovery
    from blueprint_pipeline.task_evaluation_public_scene_attempt_factory import record
    monkeypatch.setattr(recovery, "reconcile_ownership", lambda *, failure_path, **_: {
        "failure": record(failure_path), "provider_guard": record(failure_path),
        "ownership_reconciliation": record(failure_path)})
    before = _footprint(tmp_path)
    state_before = json.loads(_projection(tmp_path).read_text())["state"]

    _deploy(config, monkeypatch, NEW_COMMIT)
    for later in (120, 180):  # a fake clock, inside the fixture intent's authority window
        row = run(moment + later)
        assert (row["status"], row["phase"], row["blockers"]) == (
            "running", "previous_release", ["previous_release_attempt_not_terminal"]), row
    state = json.loads(_projection(tmp_path).read_text())["state"]
    assert state["attempt_id"] == state_before["attempt_id"]
    assert state["attempt_commit"] != NEW_COMMIT
    assert "release_predecessors" not in state
    assert _footprint(tmp_path) == before  # no new attempt, reservation, activation or preparation
    assert len(tick.submissions) == 1 and len(calls) == 2  # the paid launch was never fired again


def test_the_held_job_completes_from_its_own_launch_under_the_new_release(tmp_path, monkeypatch):
    config, run, moment, tick, calls, launch_id = _launched(tmp_path, monkeypatch)
    _write(Path(config["launch_queue_root"]) / "pending" / (launch_id + "-request.json"), {"launch_id": launch_id})
    _deploy(config, monkeypatch, NEW_COMMIT)
    assert run(moment + 120)["phase"] == "previous_release"
    # The dispatcher ran the launch on its own profile; the commit-agnostic join
    # (tested in test_scene_preparation_completion) now finds its receipt.
    (Path(config["launch_queue_root"]) / "pending" / (launch_id + "-request.json")).unlink()
    receipt = _write(Path(config["launch_execution_root"]) / launch_id / "launch_receipt.json",
                     {"launch_id": launch_id, "status": "completed"})

    def joined(**_kwargs):
        return {"status": "completed", "phase": "scene_prepared", "blockers": [],
                "result_reference": {"path": str(receipt)}, "state": {"scene_preparation_completion": {}}}
    monkeypatch.setattr(completion, "reconcile_preparation_completion", joined)
    row = run(moment + 180)
    assert (row["status"], row["phase"]) == ("completed", "scene_prepared"), row
    assert len(tick.submissions) == 1 and len(calls) == 2


def test_only_a_launch_that_left_holds_the_attempt(tmp_path):
    from blueprint_pipeline.task_evaluation_terminal_scene_attempt_settlement import launch_id_for_preparation
    preparation_id = "website-0123456789abcdef01234567-0a1b2c3d-20261031t120000z-preparation"
    link = _write(tmp_path / "link.json", _sealed({"preparation_id": preparation_id}, "link_digest"))
    launch_id = launch_id_for_preparation(preparation_id)
    config = {"launch_execution_root": str(tmp_path / "runs"), "launch_queue_root": str(tmp_path / "queue")}
    for state in ("pending", "processing"):
        (tmp_path / "queue" / state).mkdir(parents=True)
    queued = {"status": "scene_configuration_launch_queued", "launch_id": launch_id}

    def held(record=None, path=link):
        return previous_launch_in_flight(activation_link_path=path, launch_record=record, config=config)
    # No paid reservation, or a reservation whose launch was never submitted: rebuild as before.
    assert held(queued, path=None) is None
    assert held({"status": "awaiting_scene_configuration_authority", "launch_id": None}) is None
    assert held({**queued, "launch_id": "another-launch"}) is None
    # Submitted, then queued, claimed, running: held.
    assert held(queued) == "previous_release_launch_in_transit"
    _write(tmp_path / "queue" / "processing" / (launch_id + "-x.json"), {})
    assert held(queued) == "previous_release_launch_not_terminal"
    (tmp_path / "queue" / "processing" / (launch_id + "-x.json")).unlink()
    (tmp_path / "runs" / launch_id).mkdir(parents=True)
    assert held(queued) == "previous_release_launch_not_terminal"
    # A terminal receipt, whatever its outcome, releases the attempt to the existing successor rules.
    for status in ("completed", "failed", "blocked"):
        _write(tmp_path / "runs" / launch_id / "launch_receipt.json", {"launch_id": launch_id, "status": status})
        assert held(queued) is None
