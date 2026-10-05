"""Project freshness comes from reopening evidence, not manually restamping totals."""

import json
import time
from pathlib import Path

import pytest

from blueprint_pipeline.project_spend_reconciliation import (
    materialize_project_spend_reconciliation, validate_project_spend_reconciliation,
    project_spend_dependency_records,
)
from blueprint_pipeline.task_evaluation_scene_spend import publish_current_scene_project_spend
from blueprint_pipeline import task_evaluation_production_chain_preflight as preflight
from blueprint_pipeline.task_evaluation_scene_intake import revoke_scene_intent
from tests.test_project_spend_reconciliation import _human_baseline
from tests.test_task_evaluation_scene_intake import stage, attempt, request


def seed(root):
    baseline, _ = _human_baseline(root / "baseline.json")
    path = root / "seed.json"
    materialize_project_spend_reconciliation(baseline_authority_path=baseline,
        posted_reconciliation_paths=[], expected_coverage_ids=[], completeness_reference="fixture-scope",
        authorized_by="fixture-owner", authorized_on="2026-09-05", output_path=path)
    return path


def test_pointer_includes_full_caps_and_does_not_double_count_refresh(tmp_path):
    prior = seed(tmp_path)
    root = tmp_path / "intents"
    intent = stage(root)
    attempt(root, intent)
    args = dict(scene_root=root, seed_reconciliation_path=prior, output_root=tmp_path / "spend",
                current_path=tmp_path / "current.json")
    first = publish_current_scene_project_spend(**args, now=200)
    assert first["total_cost_usd"] == pytest.approx(43.197914 + 2)
    second = publish_current_scene_project_spend(**args, now=300)
    assert second["pointer"]["path"] == first["pointer"]["path"]
    assert second["total_cost_usd"] == first["total_cost_usd"]
    assert second["pointer"]["observed_at_epoch"] == 300
    attempt(root, intent, "a2")
    third = publish_current_scene_project_spend(**args, now=400)
    assert third["total_cost_usd"] == pytest.approx(43.197914 + 4)
    receipt, _ = validate_project_spend_reconciliation(third["pointer"]["path"])
    dependencies = project_spend_dependency_records(receipt)
    assert sum(name.startswith("unposted_owner_intent_") for name, _ in dependencies) == 2
    assert json.loads((tmp_path / "current.json").read_text())["digest"] == third["pointer"]["digest"]


def test_revocation_never_implies_a_zero_bill_and_corruption_does_not_refresh(tmp_path):
    prior = seed(tmp_path)
    root = tmp_path / "intents"
    intent = stage(root)
    attempt(root, intent)
    revoke_scene_intent(queue_root=root, intent_id=intent["intent_id"], intent_digest=intent["intent_digest"],
                       owner=request()["owner"], now=103)
    args = dict(scene_root=root, seed_reconciliation_path=prior, output_root=tmp_path / "spend",
                current_path=tmp_path / "current.json")
    report = publish_current_scene_project_spend(**args, now=2000)
    assert report["total_cost_usd"] == pytest.approx(43.197914 + 2)
    assert report["reserved_caps_are_not_actual_billing"] is True
    old = (tmp_path / "current.json").read_bytes()
    (root / intent["intent_id"] / "attempts" / "a1.json").chmod(0o640)
    (root / intent["intent_id"] / "attempts" / "a1.json").write_text("{}")
    with pytest.raises(ValueError):
        publish_current_scene_project_spend(**args, now=3000)
    assert (tmp_path / "current.json").read_bytes() == old


def test_activation_preflight_requires_the_configured_monitor(tmp_path, monkeypatch):
    scene_config = tmp_path / "scene-progression.json"
    scene_config.write_text(json.dumps({"activation_enabled": True}))
    monkeypatch.setattr(preflight, "SCENE_PROGRESSION_CONFIG_PATH", scene_config)
    units = {
        preflight.SCENE_PROGRESSION_UNIT: {
            "effective_environment": {},
        }
    }
    findings = preflight.project_spend_checks(units, (0, 0))
    assert [row["code"] for row in findings] == ["scene_project_spend_config_unset"]


def test_refresh_honours_the_activation_tick_now_so_the_freshness_gate_never_inverts(tmp_path, monkeypatch):
    """R6 regression. ``_activation`` captures ONE ``now`` for the whole progression
    tick, calls ``refresh_configured_scene_project_spend()``, then gates activation on
    ``0 <= now - pointer.observed_at_epoch <= 900``. If the refresh stamps
    ``time.time()`` (which is necessarily LATER than the tick's ``now``), that delta is
    negative and activation fails ``project_spend_stale`` on every tick forever, so the
    hands-off chain stalls permanently at activation. The refresh must stamp the
    caller's ``now`` so the same-``now`` gate holds."""
    from blueprint_pipeline.task_evaluation_scene_spend import refresh_configured_scene_project_spend
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    root = tmp_path / "intents"
    intent = stage(root)
    attempt(root, intent)
    prior = seed(tmp_path)
    monitor = {"schema_version": "task_evaluation_scene_project_spend_monitor.v1",
               "scene_root": str(root), "seed_reconciliation_path": str(prior),
               "output_root": str(tmp_path / "spend"), "current_path": str(tmp_path / "current.json"),
               "config_digest": ""}
    monitor["config_digest"] = canonical_digest(monitor, digest_field="config_digest")
    config_path = tmp_path / "monitor.json"
    config_path.write_text(json.dumps(monitor))
    monkeypatch.setenv("BLUEPRINT_SCENE_PROJECT_SPEND_CONFIG", str(config_path))
    # The tick's ``now`` is captured a moment BEFORE the refresh actually runs.
    tick_now = time.time() - 1.0
    result = refresh_configured_scene_project_spend(now=tick_now)
    observed = result["pointer"]["observed_at_epoch"]
    assert observed == tick_now, "refresh must stamp the caller's now, not time.time()"
    assert 0 <= tick_now - observed <= 900  # the exact _activation freshness gate now holds


def test_refresh_without_now_still_stamps_wall_clock(tmp_path, monkeypatch):
    """The capacity controller and installer call refresh with no ``now`` and rely on
    real wall-clock freshness; that default must be unchanged."""
    from blueprint_pipeline.task_evaluation_scene_spend import refresh_configured_scene_project_spend
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    root = tmp_path / "intents"
    intent = stage(root)
    attempt(root, intent)
    prior = seed(tmp_path)
    monitor = {"schema_version": "task_evaluation_scene_project_spend_monitor.v1",
               "scene_root": str(root), "seed_reconciliation_path": str(prior),
               "output_root": str(tmp_path / "spend"), "current_path": str(tmp_path / "current.json"),
               "config_digest": ""}
    monitor["config_digest"] = canonical_digest(monitor, digest_field="config_digest")
    config_path = tmp_path / "monitor.json"
    config_path.write_text(json.dumps(monitor))
    monkeypatch.setenv("BLUEPRINT_SCENE_PROJECT_SPEND_CONFIG", str(config_path))
    before = time.time()
    observed = refresh_configured_scene_project_spend()["pointer"]["observed_at_epoch"]
    assert before <= observed <= time.time()


def observed_monitor(tmp_path, monkeypatch):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    root = tmp_path / "intents"
    intent = stage(root)
    attempt(root, intent)
    prior = seed(tmp_path)
    args = dict(scene_root=root, seed_reconciliation_path=prior,
                output_root=tmp_path / "spend", current_path=tmp_path / "current.json")
    publication = publish_current_scene_project_spend(**args, now=1000)
    monitor = {"schema_version": "task_evaluation_scene_project_spend_monitor.v1",
               **{key: str(value) for key, value in args.items()}}
    monitor["config_digest"] = canonical_digest(monitor, digest_field="config_digest")
    config = tmp_path / "monitor.json"
    config.write_text(json.dumps(monitor))
    monkeypatch.setenv("BLUEPRINT_SCENE_PROJECT_SPEND_CONFIG", str(config))
    return publication, args


def test_capacity_observation_reopens_evidence_without_writes_or_restamping(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_spend as spend
    publication, args = observed_monitor(tmp_path, monkeypatch)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    monkeypatch.setattr(spend, "publish_current_scene_project_spend",
                        lambda **kwargs: pytest.fail("read-only capacity observation tried publication"))
    observed = spend.observe_configured_scene_project_spend(now=1100)
    assert observed["total_cost_usd"] == publication["total_cost_usd"]
    assert observed["pointer"] == publication["pointer"]
    assert observed["pointer"]["observed_at_epoch"] == 1000
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert args["current_path"].is_file()


@pytest.mark.parametrize("now", [999, 1901, float("nan")])
def test_capacity_observation_refuses_stale_or_future_publication(tmp_path, monkeypatch, now):
    from blueprint_pipeline.task_evaluation_scene_spend import observe_configured_scene_project_spend
    observed_monitor(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="scene_spend_pointer_invalid_or_stale"):
        observe_configured_scene_project_spend(now=now)


def test_capacity_observation_refuses_source_changed_after_publication(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_spend import observe_configured_scene_project_spend
    publication, _ = observed_monitor(tmp_path, monkeypatch)
    source = Path(publication["pointer"]["path"])
    source.chmod(0o640)
    source.write_text("{}")
    with pytest.raises(ValueError, match="scene_spend_pointer_source_changed"):
        observe_configured_scene_project_spend(now=1100)


@pytest.mark.parametrize("escape", ["traversal", "symlink"])
def test_capacity_observation_rejects_resealed_pointer_escape(tmp_path, monkeypatch, escape):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.task_evaluation_scene_spend import observe_configured_scene_project_spend
    publication, args = observed_monitor(tmp_path, monkeypatch)
    outside = tmp_path / "project_spend_reconciliation.json"
    outside.write_bytes(Path(publication["pointer"]["path"]).read_bytes())
    if escape == "traversal":
        candidate = args["output_root"] / ".." / outside.name
    else:
        link = args["output_root"] / "linked"
        link.symlink_to(tmp_path, target_is_directory=True)
        candidate = link / outside.name
    pointer = {**publication["pointer"], "path": str(candidate)}
    pointer["receipt_digest"] = canonical_digest(pointer, digest_field="receipt_digest")
    args["current_path"].chmod(0o640)
    args["current_path"].write_text(json.dumps(pointer))
    with pytest.raises(ValueError, match="scene_spend_pointer_outside_output_root"):
        observe_configured_scene_project_spend(now=1100)


def test_capacity_observation_rejects_nonobject_monitor_and_reopens_baseline(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_spend import observe_configured_scene_project_spend
    observed_monitor(tmp_path, monkeypatch)
    baseline = tmp_path / "baseline.json"
    baseline.chmod(0o640)
    baseline.write_text("{}")
    with pytest.raises(ValueError):
        observe_configured_scene_project_spend(now=1100)
    (tmp_path / "monitor.json").write_text("[]")
    with pytest.raises(ValueError, match="scene_spend_monitor_config_invalid"):
        observe_configured_scene_project_spend(now=1100)


def test_monitor_rejects_present_null_seed_reference(tmp_path, monkeypatch):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.task_evaluation_scene_spend import observe_configured_scene_project_spend
    observed_monitor(tmp_path, monkeypatch)
    path = tmp_path / "monitor.json"
    monitor = json.loads(path.read_text())
    monitor["seed_reconciliation_reference"] = None
    monitor["config_digest"] = canonical_digest(monitor, digest_field="config_digest")
    path.write_text(json.dumps(monitor))
    with pytest.raises(ValueError, match="scene_spend_monitor_seed_reference_invalid"):
        observe_configured_scene_project_spend(now=1100)


def test_publisher_reuses_large_artifact_hashes_only_within_one_pass(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_retained_controls_evidence as retained
    from blueprint_pipeline import task_evaluation_configured_controls_autostart_support as support
    from blueprint_pipeline import validation_file_digests as digests
    root = tmp_path / "intents"
    intent = stage(root)
    attempt(root, intent, "first")
    attempt(root, intent, "second")
    robot = tmp_path / "robot.usd"
    robot.write_bytes(b"a" * digests.MINIMUM_BYTES)
    observations = []

    def cancellation(_directory, _attempt):
        support._sha256(robot)
        observations.append(digests.digest_scope_stats())
        return None

    monkeypatch.setattr(retained, "validated_cancellation", cancellation)
    args = dict(scene_root=root, seed_reconciliation_path=seed(tmp_path),
                output_root=tmp_path / "spend", current_path=tmp_path / "current.json")
    for now in (1000, 1100):
        result = publish_current_scene_project_spend(**args, now=now)
        assert result["scene_reservation_count"] == 2
        assert digests.digest_scope_stats() is None
    assert [row["bytes_hashed"] for row in observations] == [robot.stat().st_size] * 12
    assert sum(row["cache_hits"] == 0 for row in observations) == 2


@pytest.mark.parametrize("launch_status, retained", [(None, 4.5), ("blocked", 21.26), ("completed", 24.72)])
def test_terminal_settlement_does_not_erase_unreconciled_project_cost(tmp_path, launch_status, retained):
    from tests.test_terminal_scene_attempt_settlement import _fixture, _settle
    fx = _fixture(tmp_path, launch_status=launch_status)
    _settle(fx)
    args = dict(scene_root=fx["root"], seed_reconciliation_path=seed(tmp_path),
                output_root=tmp_path / "spend", current_path=tmp_path / "current.json")
    first = publish_current_scene_project_spend(**args, now=300)
    assert first["total_cost_usd"] == pytest.approx(43.197914 + retained)
    second = publish_current_scene_project_spend(**args, now=400)
    assert second["total_cost_usd"] == first["total_cost_usd"]
    assert second["pointer"]["path"] == first["pointer"]["path"]


def test_posted_identity_does_not_erase_partially_covered_retained_cost():
    from blueprint_pipeline.task_evaluation_scene_spend import _require_reservation_posted_coverage
    reservation = {"authorization_digest": "sha256:attempt", "hard_attempt_spend_cap_usd": 4.5}
    with pytest.raises(ValueError, match="partial_coverage"):
        _require_reservation_posted_coverage(reservation, {"authority_digest": "sha256:attempt", "cost_usd": 0.5})
    with pytest.raises(ValueError, match="identity_mismatch"):
        _require_reservation_posted_coverage(reservation, {"authority_digest": "sha256:other", "cost_usd": 4.5})
    _require_reservation_posted_coverage(reservation, {"authority_digest": "sha256:attempt", "cost_usd": 4.5})
