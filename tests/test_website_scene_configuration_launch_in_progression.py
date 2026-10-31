"""A Website job stages its scene-configuration activation and fires its paid launch
from its own scene progression tick, not from the paused configured-controls worker.

Before this, the activation and then the launch each waited for an operator to run
the configured-controls worker by hand. The progression now drives the same
activation automation for its one preparation, on the same state root, owned queue
and WebApp submitter, and the controls worker leaves Website jobs alone so the
write-once state keeps one writer. Every provider-zero, lookahead, release-window,
standing-authorization and spend gate stays inside the automation.
"""

from __future__ import annotations

import json
from datetime import timedelta
from functools import partial
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_controls_autoprovision as autoprovision
from blueprint_pipeline import task_evaluation_configured_controls_progression_worker as worker
from blueprint_pipeline import task_evaluation_scene_configuration_activation_automation as automation
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import task_evaluation_scene_progression as progression
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_task_evaluation_scene_configuration_activation_automation import (
    COMMIT,
    NOW,
    PREPARATION_ID,
    _intent,
    _materialized_activation,
    _preparation,
    _provider_zero,
    _publisher,
    _sealed,
    _write,
)
from tests.test_task_evaluation_scene_intake import request as owner_request


def _owned_preparation(tmp_path: Path, queue: Path, *, scene_intent_digest: str,
                       preparation_id: str = PREPARATION_ID) -> Path:
    """One awaiting preparation in the owned queue, bound to its scene intent."""

    staging = tmp_path / ("staging-" + preparation_id)
    source = _preparation(staging)
    envelope = json.loads((staging / "preparations" / "materialized" / source.name).read_text())
    result = json.loads(source.read_text())
    request = dict(envelope["request"], preparation_id=preparation_id,
                   scene_intent_digest=scene_intent_digest)
    digest = canonical_digest(request)
    envelope.update(request=request, request_digest=digest)
    name = f"{preparation_id}-{digest.removeprefix('sha256:')}.json"
    _write(queue / "materialized" / name, _sealed(envelope, "envelope_digest"))
    return _write(queue / "results" / name, _sealed(dict(result, preparation_id=preparation_id), "result_digest"))


def _launch_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The scene-progression unit's roots, named as the controls unit names them."""

    roots = {
        "BLUEPRINT_TASK_EVALUATION_CONFIGURED_CONTROLS_STATE_ROOT": tmp_path / "configured-controls",
        "BLUEPRINT_TASK_EVALUATION_LAUNCH_ACTIVATION_QUEUE_ROOT": tmp_path / "activations",
        "BLUEPRINT_TASK_EVALUATION_LAUNCH_PROFILE_DIR": tmp_path / "profiles",
        "BLUEPRINT_TASK_EVALUATION_STANDING_AUTHORIZATION_DIR": tmp_path / "standing",
        "BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_REPO": tmp_path / "repo",
        "BLUEPRINT_TASK_EVALUATION_WEBAPP_SUBMISSION_SECRET_FILE": tmp_path / "secret",
    }
    for name, value in roots.items():
        monkeypatch.setenv(name, str(value))
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_WEBAPP_SUBMISSION_URL",
                       "https://tryblueprint.io/api/internal/task-evaluation-launch-submissions")
    return tmp_path / "configured-controls"


class _Tick:
    """One scene progression tick's launch step, with only external transports faked."""

    def __init__(self, queue: Path, intent_root: Path) -> None:
        self.config = {"preparation_queue_root": str(queue), "activation_intent_root": str(intent_root)}
        self.zeros: list[object] = []
        self.submissions: list[dict] = []
        self.lineage = _publisher("scene-configuration-activation-lineage")
        self.window = _publisher("coordinator-release-windows")

    def collect_zero(self):
        self.zeros.append(object())
        return _provider_zero(NOW - timedelta(seconds=20))

    def submit(self, request):
        self.submissions.append(dict(request))
        return {"status": "submitted", "launch_id": request["launch_id"],
                "provider_mutation_performed_inside_web_request": False}

    def __call__(self, result_path: Path, *, now) -> dict:
        launcher = partial(automation.advance_owned_scene_configuration,
                           provider_zero_collector=self.collect_zero,
                           lineage_publisher_factory=lambda: self.lineage,
                           release_window_publisher_factory=lambda: self.window,
                           submitter=self.submit, now=now)
        return progression._website_configuration_launch(
            observed={"result_filename": result_path.name}, config=self.config,
            release={"source_commit": COMMIT}, launcher=launcher)


def test_website_job_stages_then_launches_across_two_ticks_without_the_controls_worker(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_root = _launch_environment(tmp_path, monkeypatch)
    queue = tmp_path / "task-evaluation-owned-scene-preparations"
    intent_root, _ = _intent(tmp_path)
    result_path = _owned_preparation(tmp_path, queue, scene_intent_digest="sha256:" + "1" * 64)
    tick = _Tick(queue, intent_root)

    first = tick(result_path, now=NOW)
    assert first["status"] == "awaiting_scene_configuration_authority", first
    assert first["activation_status"] == "scene_configuration_activation_queued"
    assert "blockers" not in first and tick.submissions == [] and len(tick.zeros) == 1
    # The same state root and layout the configured-controls worker writes, so a
    # job that worker already staged continues here without restaging.
    staged = state_root / automation.STATE_DIRECTORY / PREPARATION_ID / "activation_progression.json"
    assert json.loads(staged.read_text())["activation_id"] == first["activation_id"]
    assert len(list((tmp_path / "activations" / "pending").glob("*.json"))) == 1

    # The activation worker publishes the profile and the standing authorization.
    profile, standing, _ = _materialized_activation(tmp_path, first)

    second = tick(result_path, now=NOW + timedelta(minutes=1))
    assert second["status"] == "scene_configuration_launch_queued", second
    assert second["activation_id"] == first["activation_id"]
    assert len(tick.submissions) == 1 and len(tick.zeros) == 1
    request = tick.submissions[0]
    assert request["launch_id"] == second["launch_id"]
    assert request["profile_digest"] == profile["profile_digest"]
    assert request["spend"] == {"max_spend_usd": standing["max_total_spend_usd"],
                                "expires_at": standing["expires_at"]}
    launched = state_root / automation.STATE_DIRECTORY / PREPARATION_ID / "launch_progression.json"
    assert json.loads(launched.read_text())["launch_id"] == second["launch_id"]


def test_repeated_ticks_after_the_launch_never_submit_again(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _launch_environment(tmp_path, monkeypatch)
    queue = tmp_path / "owned"
    intent_root, _ = _intent(tmp_path)
    result_path = _owned_preparation(tmp_path, queue, scene_intent_digest="sha256:" + "1" * 64)
    tick = _Tick(queue, intent_root)
    first = tick(result_path, now=NOW)
    _materialized_activation(tmp_path, first)
    launched = tick(result_path, now=NOW + timedelta(minutes=1))
    assert launched["status"] == "scene_configuration_launch_queued"
    # Each later tick, and a fresh process after a restart, returns the sealed
    # launch row and neither observes the provider nor submits again.
    for minutes in (2, 3, 30):
        assert tick(result_path, now=NOW + timedelta(minutes=minutes)) == launched
    restarted = _Tick(queue, intent_root)
    restarted.submit = lambda request: pytest.fail("a restart must not resubmit")  # type: ignore[method-assign]
    restarted.collect_zero = lambda: pytest.fail("a staged activation is not re-observed")  # type: ignore[method-assign]
    assert restarted(result_path, now=NOW + timedelta(hours=1)) == launched
    assert len(tick.submissions) == 1 and len(tick.zeros) == 1


def test_production_submitter_uses_the_controls_state_root_and_webapp_channel(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_root = _launch_environment(tmp_path, monkeypatch)
    queue = tmp_path / "owned"
    intent_root, _ = _intent(tmp_path)
    result_path = _owned_preparation(tmp_path, queue, scene_intent_digest="sha256:" + "1" * 64)
    built: dict = {}
    monkeypatch.setattr(automation, "webapp_submitter", lambda **kwargs: built.update(kwargs) or (
        lambda request: pytest.fail("no authority yet")))
    row = automation.advance_owned_scene_configuration(
        preparation_result_path=result_path, preparation_queue_root=queue, intent_root=intent_root,
        running_commit=COMMIT, provider_zero_collector=lambda: _provider_zero(NOW - timedelta(seconds=5)),
        lineage_publisher_factory=lambda: _publisher("lineage"),
        release_window_publisher_factory=lambda: _publisher("windows"), now=NOW)
    assert row["status"] == "awaiting_scene_configuration_authority", row
    assert built == {
        "repo_root": str(tmp_path / "repo"),
        "secret_file": str(tmp_path / "secret"),
        "endpoint": "https://tryblueprint.io/api/internal/task-evaluation-launch-submissions",
        "state_root": state_root / automation.STATE_DIRECTORY / "webapp-submissions",
    }


def test_an_unconfigured_unit_is_flagged_on_the_row_and_touches_nothing(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in automation.OWNED_LAUNCH_ENVIRONMENT.values():
        monkeypatch.delenv(name, raising=False)
    queue = tmp_path / "owned"
    intent_root, _ = _intent(tmp_path)
    result_path = _owned_preparation(tmp_path, queue, scene_intent_digest="sha256:" + "1" * 64)
    row = progression._website_configuration_launch(
        observed={"result_filename": result_path.name},
        config={"preparation_queue_root": str(queue), "activation_intent_root": str(intent_root)},
        release={"source_commit": COMMIT}, launcher=None)
    assert row["status"] == "blocked"
    assert row["blockers"][0].startswith("scene_configuration_launch_unconfigured:")
    assert not (tmp_path / "activations").exists()


def test_a_failing_launch_step_is_reported_and_never_stops_the_tick(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _launch_environment(tmp_path, monkeypatch)
    queue = tmp_path / "owned"
    intent_root, _ = _intent(tmp_path)
    result_path = _owned_preparation(tmp_path, queue, scene_intent_digest="sha256:" + "1" * 64)
    tick = _Tick(queue, intent_root)
    tick.collect_zero = lambda: _provider_zero(NOW - timedelta(seconds=20), live=1)  # type: ignore[method-assign]
    row = tick(result_path, now=NOW)
    assert row == {"status": "blocked", "blockers": ["scene_configuration_activation_provider_not_zero"]}
    assert tick.submissions == []

    def unavailable(**_kwargs):
        raise OSError(5, "object store unreachable")
    row = progression._website_configuration_launch(
        observed={"result_filename": result_path.name}, config=tick.config,
        release={"source_commit": COMMIT}, launcher=unavailable)
    assert row == {"status": "blocked", "blockers": ["scene_configuration_launch_step_failed:OSError"]}


def _website_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, queue: Path) -> dict:
    """A real Website intent in the owner store the controls autoprovision reads."""

    moment = NOW.timestamp()
    owner = owner_request()
    owner["execution"].update(purpose="scene_preparation", policy_candidates=[],
                              expires_at_epoch=moment + 7200)
    owner["consent"]["accepted_at_epoch"] = moment - 1
    staged = intake.stage_scene_intent(value=owner, queue_root=tmp_path / "scenes",
                                       authenticated_client="webapp", trusted_clients={"webapp"}, now=moment)
    config = tmp_path / "autoprovision.json"
    config.write_text(json.dumps({"scene_root": str(tmp_path / "scenes"), "trusted_clients": ["webapp"],
                                  "preparation_queue_root": str(queue)}))
    monkeypatch.setenv(autoprovision.CONFIG_ENV, str(config))
    return staged


def test_controls_worker_reports_website_jobs_and_writes_nothing_for_them(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    queue = tmp_path / "task-evaluation-owned-scene-preparations"
    staged = _website_scope(tmp_path, monkeypatch, queue)
    _owned_preparation(tmp_path, queue, scene_intent_digest=staged["intent_digest"])
    scope = autoprovision.progression_owner_scope(COMMIT)
    assert scope.scene_progression_intent_digests == {staged["intent_digest"]}
    assert not scope.unresolved
    monkeypatch.setattr(automation, "advance_scene_configuration_activation",
                        lambda **_: pytest.fail("the controls worker must not stage a Website job"))
    monkeypatch.setattr(worker, "running_release_commit", lambda: COMMIT)
    monkeypatch.setattr(automation, "running_release_commit", lambda: COMMIT)
    intent_root = tmp_path / "scene-configuration-intents"
    intent_root.mkdir()
    report = worker.process_plans(
        plan_root=tmp_path / "plans", autostart_intent_root=tmp_path / "intents",
        launch_state_root=tmp_path / "launch", progression_root=tmp_path / "progression",
        preparation_queue_root=tmp_path / "cli-launch-preparations",
        episode_compilation_queue_root=tmp_path / "compilations",
        activation_queue_root=tmp_path / "activations", repo_root=tmp_path / "repo",
        webapp_secret_file=tmp_path / "secret",
        webapp_endpoint="https://tryblueprint.io/api/internal/task-evaluation-launch-submissions",
        scene_configuration_activation_intent_root=intent_root,
        profile_dir=tmp_path / "profiles", standing_authorization_dir=tmp_path / "standing")
    rows = [row for row in report["rows"] if row.get("lane") == automation.LANE]
    assert rows == [{"lane": automation.LANE, "preparation_id": PREPARATION_ID,
                     "status": automation.SCENE_PROGRESSION_OWNED_STATUS}]
    assert report["status"] == "completed", report
    assert not (tmp_path / "progression" / automation.STATE_DIRECTORY).exists()
    assert not (tmp_path / "activations").exists()


def test_controls_worker_still_advances_a_robot_team_configuration(tmp_path: Path) -> None:
    """A non-Website preparation beside a Website one keeps the existing controls flow."""

    queue = tmp_path / "owned"
    intent_root, _ = _intent(tmp_path)
    website = "sha256:" + "1" * 64
    _owned_preparation(tmp_path, queue, scene_intent_digest=website,
                       preparation_id="website-0123456789abcdef01234567-0a1b2c3d-20261031t120000z-preparation")
    _owned_preparation(tmp_path, queue, scene_intent_digest="sha256:" + "2" * 64)
    lineage, window = _publisher("lineage"), _publisher("windows")
    rows = automation.process_scene_configuration_activations(
        preparation_queue_root=queue, activation_queue_root=tmp_path / "activations",
        progression_root=tmp_path / "progression", intent_root=intent_root,
        profile_dir=tmp_path / "profiles", standing_authorization_dir=tmp_path / "standing",
        provider_zero_collector=lambda: _provider_zero(NOW - timedelta(seconds=20)),
        lineage_publisher_factory=lambda: lineage, release_window_publisher_factory=lambda: window,
        submitter=lambda request: pytest.fail("no authority yet"), now=NOW,
        scene_progression_intent_digests={website})
    by_id = {row["preparation_id"]: row for row in rows}
    assert by_id[PREPARATION_ID]["status"] == "awaiting_scene_configuration_authority", rows
    assert by_id[PREPARATION_ID]["activation_status"] == "scene_configuration_activation_queued"
    skipped = by_id["website-0123456789abcdef01234567-0a1b2c3d-20261031t120000z-preparation"]
    assert skipped == {"preparation_id": skipped["preparation_id"],
                       "status": automation.SCENE_PROGRESSION_OWNED_STATUS}
    staged = sorted(path.name for path in (tmp_path / "progression" / automation.STATE_DIRECTORY).iterdir())
    assert staged == [PREPARATION_ID]


def test_a_release_successor_starts_without_the_previous_launch_row() -> None:
    state = {"attempt_id": "source-1", "scene_configuration_launch": {"status": "x"}, "binding_digest": "b"}
    progression._clear_attempt(state)
    assert state == {"binding_digest": "b"}


# ------------------------------------------------- real scene progression ticks


def _authority_for(roots: Path, row: dict, *, commit: str, now) -> dict:
    """What the launch-activation worker publishes: profile, standing grant, result."""

    profile_id = f"task-evaluation-scene-configuration-{commit}-r1-binding-0a1b2c3d4e5f"
    profile = _sealed({
        "schema_version": "task_evaluation_launch_profile.v1", "profile_id": profile_id,
        "source_commit": commit, "claim_ceiling": "development_only",
        "evaluation_run_spec": {"uri": "s3://blueprint/task-evaluation/production-inputs/ns/evaluation_run_spec.json",
                                "digest": "sha256:" + "e" * 64, "size_bytes": 512},
        "allocator": {"max_spend_usd": 11.0, "retry_cap": 0, "hard_ttl_seconds": 27_000}}, "profile_digest")
    _write(roots / "profiles" / f"{profile_id}.json", profile)
    standing = {
        "schema_version": "task_evaluation_standing_launch_authorization.v1",
        "authorization_reference": "scene intent", "authorized_by": "owner-1",
        "issued_at": now.isoformat(), "expires_at": (now + timedelta(hours=1)).isoformat(),
        "max_launches": 1, "max_total_spend_usd": 11.0, "profile_id": profile_id,
        "profile_digest": profile["profile_digest"], "provider_mutation_performed": False}
    _write(roots / "standing" / f"{profile_id}.json", standing)
    _write(roots / "activations" / "results" / f"{row['activation_id']}-{'f' * 64}.json", _sealed({
        "schema_version": "task_evaluation_launch_activation_result.v1",
        "status": "profile_authority_materialized_no_execution", "activation_id": row["activation_id"],
        "preparation_id": row["preparation_id"], "lane": automation.LANE, "source_commit": commit,
        "profile_id": profile_id, "profile_digest": profile["profile_digest"],
        "provider_mutation_performed": False, "paid_execution_requested": False, "blockers": []},
        "result_digest"))
    return standing


def _activation_enabled(tmp_path: Path) -> tuple[Path, float]:
    """Arm the activation on-ramp exactly as the installer's activation config does."""

    import time
    from datetime import datetime, timezone
    from blueprint_pipeline.project_spend_reconciliation import materialize_project_spend_reconciliation
    from blueprint_pipeline.task_evaluation_scene_spend import publish_current_scene_project_spend
    from tests.test_project_spend_reconciliation import _human_baseline

    baseline, _ = _human_baseline(tmp_path / "activation-baseline.json")
    spend = tmp_path / "activation-project-spend.json"
    materialize_project_spend_reconciliation(
        baseline_authority_path=baseline, posted_reconciliation_paths=[], expected_coverage_ids=[],
        completeness_reference=str(baseline), authorized_by="fixture-owner",
        authorized_on=datetime.now(timezone.utc).isoformat(), output_path=spend)
    moment = time.time()
    current = tmp_path / "activation-project-spend-current.json"
    publish_current_scene_project_spend(scene_root=tmp_path / "intents", seed_reconciliation_path=spend,
        output_root=tmp_path / "activation-spend-snapshots", current_path=current, now=moment)
    for name in ("activation-intents", "launch-runs"):
        (tmp_path / name).mkdir()
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config.update(activation_enabled=True, activation_intent_root=str(tmp_path / "activation-intents"),
                  project_spend_current_path=str(current), launch_execution_root=str(tmp_path / "launch-runs"))
    _write(config_path, _sealed(config, "config_digest"))
    return config_path, moment


@pytest.mark.parametrize("website", [True, False])
def test_scene_progression_ticks_stage_then_launch_a_website_job(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, website: bool) -> None:
    from datetime import datetime, timezone
    from blueprint_pipeline import task_evaluation_scene_scope_restriction as scope
    from tests.test_task_evaluation_scene_configuration_submission import SHA
    from tests.test_website_native_submission import prepare_website_construction

    prepare_website_construction(tmp_path, monkeypatch, False)
    config_path, moment = _activation_enabled(tmp_path)
    state_root = _launch_environment(tmp_path, monkeypatch)
    if not website:
        # The same prepared scene under a robot team's intent: its activation and
        # launch stay with the configured-controls worker.
        monkeypatch.setattr(scope, "preparation_only", lambda **_: False)
    now = datetime.now(timezone.utc)
    tick = _Tick(tmp_path / "preparations", tmp_path / "activation-intents")
    calls: list[dict] = []

    def launcher(**kwargs):
        calls.append(kwargs)
        return automation.advance_owned_scene_configuration(
            **kwargs, provider_zero_collector=lambda: _provider_zero(now - timedelta(seconds=20)),
            lineage_publisher_factory=lambda: tick.lineage, release_window_publisher_factory=lambda: tick.window,
            submitter=tick.submit, now=now)

    def progress(at: float) -> dict:
        run = progression.process_scene_intents(
            config_path=config_path, now=at, configuration_launcher=launcher,
            activation_provisioner=automation.provision_scene_configuration_activation_intent)
        [row] = run["results"]
        assert (row["status"], row["phase"]) == ("awaiting_execution", "scene_configuration"), row
        projection = json.loads(next((tmp_path / "intents").glob("scene-*/progression.json")).read_text())
        return projection

    first = progress(moment)
    if not website:
        assert calls == [] and first["blockers"] == []
        assert "scene_configuration_launch" not in first["state"]
        assert not (state_root / automation.STATE_DIRECTORY).exists()
        return
    [call] = calls
    result_path = next((tmp_path / "preparations" / "results").glob("*.json"))
    assert call == {"preparation_result_path": result_path, "preparation_queue_root": tmp_path / "preparations",
                    "intent_root": str(tmp_path / "activation-intents"), "running_commit": SHA}
    staged = first["state"]["scene_configuration_launch"]
    assert staged["status"] == "awaiting_scene_configuration_authority", staged
    assert staged["activation_status"] == "scene_configuration_activation_queued"
    assert first["blockers"] == [] and tick.submissions == []
    assert (state_root / automation.STATE_DIRECTORY / staged["preparation_id"] / "activation_progression.json").is_file()

    standing = _authority_for(tmp_path, staged, commit=SHA, now=now)
    second = progress(moment + 60)
    launched = second["state"]["scene_configuration_launch"]
    assert launched["status"] == "scene_configuration_launch_queued", launched
    assert [request["launch_id"] for request in tick.submissions] == [launched["launch_id"]]
    assert tick.submissions[0]["spend"]["max_spend_usd"] == standing["max_total_spend_usd"]

    third = progress(moment + 120)
    assert third["state"]["scene_configuration_launch"] == launched
    assert third["event_sequence"] == second["event_sequence"]
    assert len(tick.submissions) == 1
