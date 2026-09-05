"""The scene-configuration activation runs itself from one registered owner intent.

The 839873 rehearsal needed five hand-written scripts between "preparation
result landed" and "Website launch fired": observe provider zero, copy the
project spend baseline, author the activation request and its release window,
publish three governance objects, then sign a WebApp launch.  Every one of them
is a per-run decision an operator could get wrong.  This module owns those
joins.  It never allocates a provider and never executes anything paid; the
dispatcher still requires the Website record and the standing authorization
the activation worker publishes.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from blueprint_pipeline import (
    task_evaluation_scene_configuration_activation_automation as automation,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_activation_contract import (
    validate_launch_activation_request,
)
from blueprint_pipeline.vast_evidence_contracts import VAST_PROVIDER_ZERO_API_CALL


COMMIT = "8ea8b32a6" + "0" * 31
TEAM = "blueprint-adp"
SCENE_ID = "interiorgs-841757"
TASK_ID = "scene-841757-book-to-tray"
PREPARATION_ID = "adp-new-scene-book-to-tray-841757-8ea8b32a-20260905t030000z-preparation"
NOW = datetime(2026, 9, 5, 3, 0, 30, tzinfo=timezone.utc)


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    return path


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _sealed(value: dict, field: str) -> dict:
    value[field] = ""
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _release_window_template(path: Path, *, commit: str = COMMIT) -> Path:
    return _write(
        path,
        _sealed(
            {
                "schema_version": (
                    "task_evaluation_configured_controls_release_window_template.v1"
                ),
                "status": "authorized_for_dynamic_release",
                "team_namespace": TEAM,
                "expected_production_commit": commit,
                "allowed_mutations": [
                    "catalog_synchronization",
                    "profile_publication",
                    "standing_authorization",
                ],
                "provider_allowlist": ["vast"],
                "maximum_hard_cap_usd": 12.0,
                "valid_for_seconds": 21_600,
                "released_by": "Blueprint-owner",
                "release_reference": "Scene 841757 scene-configuration automatic activation",
                "provider_resource_allocation_allowed": False,
                "paid_request_allowed": False,
            },
            "template_digest",
        ),
    )


def _project_spend(path: Path) -> Path:
    return _write(
        path,
        {
            "schema_version": "adp_project_spend_reconciliation.v1",
            "status": "project_spend_conservatively_reconciled",
            "goal_id": "arm-decision-proof-v1",
            "total_cost_usd": 61.42,
        },
    )


def _provider_zero(observed_at: datetime, *, live: int = 0) -> dict:
    return _sealed(
        {
            "api_command": list(VAST_PROVIDER_ZERO_API_CALL),
            "api_confirmed": True,
            "global_live_resource_count": live,
            "inventory": [] if live == 0 else [{"id": 1}],
            "observed_at_utc": observed_at.isoformat(),
            "provider": "vast",
            "provider_zero": live == 0,
            "raw_secret_values_recorded": False,
            "schema_version": "adp_paid_provider_zero.v1",
            "stderr_present": False,
        },
        "provider_zero_digest",
    )


def _intent(tmp_path: Path, *, commit: str = COMMIT) -> tuple[Path, dict]:
    template = _release_window_template(
        tmp_path / "intent-inputs" / "release_window_template.v1.json", commit=commit
    )
    spend = _project_spend(tmp_path / "intent-inputs" / "project_spend_reconciliation.json")
    intent_root = tmp_path / "intents"
    intent_root.mkdir()
    intent = automation.materialize_scene_configuration_activation_intent(
        expected_production_commit=commit,
        team_namespace=TEAM,
        scene_id=SCENE_ID,
        task_id=TASK_ID,
        authorization_template={
            "reference": "Blueprint owner direction 2026-09-04: scene 841757 book-to-tray end to end",
            "authorized_by": "nijelhunt_1",
            "profile_revision": "r1",
            "valid_for_seconds": 21_600,
        },
        release_window_template_path=template,
        project_spend_reconciliation_path=spend,
        rights_scope="internal_noncommercial_research_only",
        output_path=intent_root
        / automation.scene_configuration_activation_registry_name(
            team_namespace=TEAM, scene_id=SCENE_ID, task_id=TASK_ID
        ),
    )
    return intent_root, intent


def _preparation(tmp_path: Path, *, status: str = "queued_for_production_scene_configuration") -> Path:
    queue = tmp_path / "preparations"
    request = {
        "schema_version": "task_evaluation_launch_preparation_request.v1",
        "run_mode": "scene_configuration",
        "expected_production_commit": COMMIT,
        "preparation_id": PREPARATION_ID,
        "team_namespace": TEAM,
        "run_id": PREPARATION_ID.removesuffix("-preparation") + "-scene-configuration",
        "scene": {"identity": {"id": SCENE_ID, "version": "book-tray-v1"}},
        "task": {"identity": {"id": TASK_ID, "version": "v1"}},
        "spend": {
            "maximum_hourly_rate_usd": 0.8,
            "hard_cap_usd": 12.0,
            "hard_ttl_seconds": 27_000,
            "provider_compute_spend_cap_usd": 6.0,
            "retry_cap": 0,
            "selected_provider": "vast",
            "provider_allowlist": ["vast"],
        },
        "publication": {"input_namespace": "adp-new-scene-book-to-tray-841757-ns"},
    }
    request_digest = canonical_digest(request)
    envelope = _sealed(
        {
            "schema_version": "task_evaluation_launch_preparation_intake_envelope.v1",
            "request_digest": request_digest,
            "request": request,
            "submitted_by": "blueprint-production-runner",
            "submitted_at_iso": "2026-09-05T02:58:00+00:00",
            "provider_mutation_performed_inside_intake": False,
            "catalog_mutation_performed_inside_intake": False,
        },
        "envelope_digest",
    )
    suffix = request_digest.removeprefix("sha256:")
    _write(queue / "materialized" / f"{PREPARATION_ID}-{suffix}.json", envelope)
    result = _sealed(
        {
            "schema_version": "task_evaluation_launch_preparation_result.v1",
            "status": status,
            "preparation_id": PREPARATION_ID,
            "run_id": request["run_id"],
            "run_mode": "scene_configuration",
            "team_namespace": TEAM,
            "source_commit": COMMIT,
            "provider_mutation_performed": False,
            "paid_execution_requested": False,
            "references": [],
        },
        "result_digest",
    )
    return _write(queue / "results" / f"{PREPARATION_ID}-{suffix}.json", result)


def _publisher(prefix: str):
    published: dict[str, bytes] = {}

    def publish(*, path: Path, object_name: str):
        payload = path.read_bytes()
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        uri = f"s3://blueprint/task-evaluation/production-inputs/{prefix}/{object_name}"
        published[uri] = payload
        return {
            "uri": uri,
            "digest": digest,
            "size_bytes": len(payload),
            "full_byte_service_account_readback_passed": True,
            "readback_digest": digest,
            "readback_size_bytes": len(payload),
        }

    publish.published = published  # type: ignore[attr-defined]
    return publish


def _controls_intent_root(tmp_path: Path, *, registered: bool = True) -> Path:
    """A controls-continuation registry as the provisioner leaves it (identity-named file)."""

    from blueprint_pipeline.task_evaluation_configured_controls_autostart import (
        configured_controls_autostart_registry_name,
    )

    root = tmp_path / "controls-intents"
    root.mkdir(exist_ok=True)
    if registered:
        name = configured_controls_autostart_registry_name(
            team_namespace=TEAM, scene_id=SCENE_ID, task_id=TASK_ID
        )
        _write(root / name, {"schema_version": "task_evaluation_configured_controls_autostart_intent.v2"})
    return root


def _advance(
    tmp_path: Path, result_path: Path, intent_root: Path, *, zero=None, now=NOW, controls_registered=True
):
    lineage_publisher = _publisher("scene-configuration-activation-lineage")
    window_publisher = _publisher("coordinator-release-windows")
    observed = automation.advance_scene_configuration_activation(
        preparation_result_path=result_path,
        preparation_queue_root=tmp_path / "preparations",
        activation_queue_root=tmp_path / "activations",
        progression_root=tmp_path / "progression",
        intent_root=intent_root,
        configured_controls_intent_root=_controls_intent_root(tmp_path, registered=controls_registered),
        provider_zero_collector=lambda: zero
        if zero is not None
        else _provider_zero(now - timedelta(seconds=20)),
        lineage_publisher_factory=lambda: lineage_publisher,
        release_window_publisher_factory=lambda: window_publisher,
        now=now,
    )
    return observed, lineage_publisher, window_publisher


def test_intent_is_sealed_and_bound_to_its_template_and_baseline_bytes(tmp_path: Path) -> None:
    intent_root, intent = _intent(tmp_path)
    assert intent["schema_version"] == automation.INTENT_SCHEMA_VERSION
    assert intent["configuration_source_commit"] == COMMIT
    assert intent["requested_mutations"] == {
        "profile_publication": True,
        "catalog_synchronization": True,
        "standing_authorization": True,
    }
    assert intent["provider_zero"] == {"mode": "observe_live_before_authorization"}
    assert intent["provider_mutation_performed"] is False
    assert intent["paid_execution_requested"] is True
    for name in ("release_window_template", "project_spend_reconciliation"):
        row = intent["artifact_inventory"][name]
        assert row["digest"] == _sha256(Path(row["path"]))
    stored = json.loads(next(intent_root.glob("*.json")).read_text())
    assert automation.validate_scene_configuration_activation_intent(stored) == intent
    tampered = dict(stored, rights_scope="commercial_use")
    with pytest.raises(
        automation.SceneConfigurationActivationAutomationError,
        match="scene_configuration_activation_intent_invalid",
    ):
        automation.validate_scene_configuration_activation_intent(tampered)


def test_intent_refuses_a_short_authorization_window_or_partial_mutations(tmp_path: Path) -> None:
    template = _release_window_template(tmp_path / "t.json")
    spend = _project_spend(tmp_path / "p.json")
    with pytest.raises(
        automation.SceneConfigurationActivationAutomationError,
        match="scene_configuration_activation_intent_authorization_invalid",
    ):
        automation.materialize_scene_configuration_activation_intent(
            expected_production_commit=COMMIT,
            team_namespace=TEAM,
            scene_id=SCENE_ID,
            task_id=TASK_ID,
            authorization_template={
                "reference": "x",
                "authorized_by": "nijelhunt_1",
                "profile_revision": "r1",
                "valid_for_seconds": 60,
            },
            release_window_template_path=template,
            project_spend_reconciliation_path=spend,
            rights_scope="internal_noncommercial_research_only",
            output_path=tmp_path / "intent.json",
        )


def test_activation_is_staged_once_from_the_owner_intent_and_a_fresh_provider_zero(
    tmp_path: Path,
) -> None:
    intent_root, intent = _intent(tmp_path)
    result_path = _preparation(tmp_path)
    observed, lineage_publisher, window_publisher = _advance(tmp_path, result_path, intent_root)

    assert observed["status"] == "scene_configuration_activation_queued"
    assert observed["preparation_id"] == PREPARATION_ID
    assert observed["intent_digest"] == intent["intent_digest"]
    assert observed["provider_mutation_performed"] is False
    assert observed["paid_execution_requested"] is False
    pending = list((tmp_path / "activations" / "pending").glob("*.json"))
    assert len(pending) == 1
    request = json.loads(pending[0].read_text())["request"]
    validate_launch_activation_request(request)
    assert request["lane"] == "task_evaluation_scene_configuration"
    assert request["expected_production_commit"] == COMMIT
    assert request["team_namespace"] == TEAM
    assert request["activation_id"] != PREPARATION_ID
    assert request["activation_id"] == observed["activation_id"]
    assert request["preparation"]["preparation_id"] == PREPARATION_ID
    result = json.loads(result_path.read_text())
    assert request["preparation"]["result_digest"] == result["result_digest"]
    envelope = json.loads(next((tmp_path / "preparations" / "materialized").glob("*.json")).read_text())
    assert request["preparation"]["request_digest"] == envelope["request_digest"]
    assert request["requested_mutations"] == intent["requested_mutations"]
    assert request["authorization"]["reference"] == intent["authorization_template"]["reference"]
    assert request["authorization"]["authorized_by"] == "nijelhunt_1"
    assert request["authorization"]["profile_revision"] == "r1"
    authorized_on = datetime.fromisoformat(request["authorization"]["authorized_on"].replace("Z", "+00:00"))
    expires = datetime.fromisoformat(
        request["authorization"]["standing_authorization_expires_at"].replace("Z", "+00:00")
    )
    assert expires - authorized_on == timedelta(seconds=21_600)
    # Lineage objects are the exact published bytes, and the zero was observed
    # before the authorization it precedes (the paid authority refuses otherwise).
    lineage = request["lineage"]
    assert lineage["kind"] == "initial_project"
    for name in ("project_spend_reconciliation", "initial_provider_zero"):
        reference = lineage[name]
        payload = lineage_publisher.published[reference["uri"]]
        assert reference["digest"] == "sha256:" + hashlib.sha256(payload).hexdigest()
        assert reference["size_bytes"] == len(payload)
    zero = json.loads(lineage_publisher.published[lineage["initial_provider_zero"]["uri"]])
    assert zero["schema_version"] == "adp_paid_provider_zero.v1"
    observed_at = datetime.fromisoformat(zero["observed_at_utc"])
    assert observed_at < authorized_on
    assert (authorized_on - observed_at).total_seconds() <= 900
    window_payload = window_publisher.published[request["release_window"]["uri"]]
    window = json.loads(window_payload)
    assert window["schema_version"] == "task_evaluation_shared_mutation_window.v1"
    assert window["activation_id"] == request["activation_id"]
    assert window["maximum_hard_cap_usd"] >= 12.0
    assert request["release_window"]["digest"] == "sha256:" + hashlib.sha256(window_payload).hexdigest()

    # Re-running is safe: the sealed progression is returned and nothing is re-queued.
    again, _lineage, _window = _advance(tmp_path, result_path, intent_root, now=NOW + timedelta(minutes=5))
    assert again == observed
    assert len(list((tmp_path / "activations" / "pending").glob("*.json"))) == 1


def test_activation_waits_without_an_intent_and_refuses_a_foreign_commit(tmp_path: Path) -> None:
    result_path = _preparation(tmp_path)
    empty_root = tmp_path / "no-intents"
    empty_root.mkdir()
    observed, _l, _w = _advance(tmp_path, result_path, empty_root)
    assert observed["status"] == "awaiting_scene_configuration_activation_intent"
    assert not (tmp_path / "activations" / "pending").exists()

    intent_root, _intent_value = _intent(tmp_path, commit="1" * 40)
    with pytest.raises(
        automation.SceneConfigurationActivationAutomationError,
        match="scene_configuration_activation_intent_commit_mismatch",
    ):
        _advance(tmp_path, result_path, intent_root)


def test_activation_waits_until_the_controls_continuation_intent_is_registered(tmp_path: Path) -> None:
    """The activation worker binds the controls intent into the profile; activating first would strand it."""

    intent_root, _intent_value = _intent(tmp_path)
    result_path = _preparation(tmp_path)
    observed, _l, _w = _advance(tmp_path, result_path, intent_root, controls_registered=False)
    assert observed["status"] == "awaiting_configured_controls_continuation_intent"
    assert not (tmp_path / "activations" / "pending").exists()
    ready, _l2, _w2 = _advance(tmp_path, result_path, intent_root, controls_registered=True)
    assert ready["status"] == "scene_configuration_activation_queued"


def test_provision_intent_authors_its_own_release_window_template(tmp_path: Path) -> None:
    spend = _project_spend(tmp_path / "spend.json")
    intent_root = tmp_path / "intents"
    intent_root.mkdir()
    intent = automation.provision_scene_configuration_activation_intent(
        expected_production_commit=COMMIT,
        team_namespace=TEAM,
        scene_id=SCENE_ID,
        task_id=TASK_ID,
        authorization_reference="Blueprint owner direction 2026-09-04: scene 841757 book-to-tray end to end",
        authorized_by="nijelhunt_1",
        profile_revision="r1",
        valid_for_seconds=21_600,
        project_spend_reconciliation_path=spend,
        rights_scope="internal_noncommercial_research_only",
        maximum_hard_cap_usd=12.0,
        release_reference="Scene 841757 scene-configuration automatic activation",
        intent_root=intent_root,
        materialization_root=tmp_path / "activation-intent-inputs",
    )
    template_path = Path(intent["artifact_inventory"]["release_window_template"]["path"])
    template = json.loads(template_path.read_text())
    from blueprint_pipeline.task_evaluation_shared_mutation_window import (
        validate_shared_mutation_window_template,
    )

    validate_shared_mutation_window_template(template, team_namespace=TEAM, expected_production_commit=COMMIT)
    assert template["maximum_hard_cap_usd"] == 12.0
    assert template["released_by"] == "nijelhunt_1"
    registered = json.loads(next(intent_root.glob("*.json")).read_text())
    assert registered == intent
    # Re-provisioning is a no-op that returns the identical sealed intent.
    again = automation.provision_scene_configuration_activation_intent(
        expected_production_commit=COMMIT,
        team_namespace=TEAM,
        scene_id=SCENE_ID,
        task_id=TASK_ID,
        authorization_reference="Blueprint owner direction 2026-09-04: scene 841757 book-to-tray end to end",
        authorized_by="nijelhunt_1",
        profile_revision="r1",
        valid_for_seconds=21_600,
        project_spend_reconciliation_path=spend,
        rights_scope="internal_noncommercial_research_only",
        maximum_hard_cap_usd=12.0,
        release_reference="Scene 841757 scene-configuration automatic activation",
        intent_root=intent_root,
        materialization_root=tmp_path / "activation-intent-inputs",
    )
    assert again == intent


def test_activation_refuses_a_live_provider_and_stages_nothing(tmp_path: Path) -> None:
    intent_root, _intent_value = _intent(tmp_path)
    result_path = _preparation(tmp_path)
    with pytest.raises(
        automation.SceneConfigurationActivationAutomationError,
        match="scene_configuration_activation_provider_not_zero",
    ):
        _advance(tmp_path, result_path, intent_root, zero=_provider_zero(NOW - timedelta(seconds=5), live=1))
    assert not (tmp_path / "activations" / "pending").exists()


def test_activation_skips_a_preparation_that_is_not_awaiting_configuration(tmp_path: Path) -> None:
    intent_root, _intent_value = _intent(tmp_path)
    result_path = _preparation(tmp_path, status="blocked")
    observed, _l, _w = _advance(tmp_path, result_path, intent_root)
    assert observed["status"] == "preparation_not_awaiting_scene_configuration"


def _materialized_activation(tmp_path: Path, observed: dict) -> tuple[dict, dict, dict]:
    profile_id = f"task-evaluation-scene-configuration-{COMMIT}-r1-binding-0a1b2c3d4e5f"
    profile = _sealed(
        {
            "schema_version": "task_evaluation_launch_profile.v1",
            "profile_id": profile_id,
            "source_commit": COMMIT,
            "claim_ceiling": "development_only",
            "evaluation_run_spec": {
                "uri": "s3://blueprint/task-evaluation/production-inputs/ns/evaluation_run_spec.json",
                "digest": "sha256:" + "e" * 64,
                "size_bytes": 512,
            },
            "allocator": {"max_spend_usd": 12.0, "retry_cap": 0, "hard_ttl_seconds": 27_000},
        },
        "profile_digest",
    )
    _write(tmp_path / "profiles" / f"{profile_id}.json", profile)
    standing = {
        "schema_version": "task_evaluation_standing_launch_authorization.v1",
        "authorization_reference": "Scene 841757 automatic activation",
        "authorized_by": "nijelhunt_1",
        "issued_at": NOW.isoformat(),
        "expires_at": (NOW + timedelta(hours=6)).isoformat(),
        "max_launches": 1,
        "max_total_spend_usd": 12.0,
        "profile_id": profile_id,
        "profile_digest": profile["profile_digest"],
        "provider_mutation_performed": False,
    }
    _write(tmp_path / "standing" / f"{profile_id}.json", standing)
    activation_result = _sealed(
        {
            "schema_version": "task_evaluation_launch_activation_result.v1",
            "status": "profile_authority_materialized_no_execution",
            "activation_id": observed["activation_id"],
            "preparation_id": PREPARATION_ID,
            "team_namespace": TEAM,
            "lane": "task_evaluation_scene_configuration",
            "source_commit": COMMIT,
            "profile_id": profile_id,
            "profile_digest": profile["profile_digest"],
            "standing_authorization_published": True,
            "catalog_mutation_performed": True,
            "provider_mutation_performed": False,
            "paid_execution_requested": False,
            "blockers": [],
        },
        "result_digest",
    )
    _write(
        tmp_path / "activations" / "results" / f"{observed['activation_id']}-{'f' * 64}.json",
        activation_result,
    )
    return profile, standing, activation_result


def test_launch_is_fired_once_through_the_webapp_after_the_authority_is_materialized(
    tmp_path: Path,
) -> None:
    intent_root, intent = _intent(tmp_path)
    result_path = _preparation(tmp_path)
    observed, _l, _w = _advance(tmp_path, result_path, intent_root)
    submissions: list[dict] = []

    def submitter(request):
        submissions.append(dict(request))
        return {
            "status": "submitted",
            "launch_id": request["launch_id"],
            "provider_mutation_performed_inside_web_request": False,
        }

    waiting = automation.advance_scene_configuration_launch(
        progression=observed,
        activation_queue_root=tmp_path / "activations",
        profile_dir=tmp_path / "profiles",
        standing_authorization_dir=tmp_path / "standing",
        progression_root=tmp_path / "progression",
        intent=intent,
        submitter=submitter,
        now=NOW + timedelta(minutes=2),
    )
    assert waiting["status"] == "awaiting_scene_configuration_authority"
    assert submissions == []

    profile, standing, _activation_result = _materialized_activation(tmp_path, observed)
    launched = automation.advance_scene_configuration_launch(
        progression=observed,
        activation_queue_root=tmp_path / "activations",
        profile_dir=tmp_path / "profiles",
        standing_authorization_dir=tmp_path / "standing",
        progression_root=tmp_path / "progression",
        intent=intent,
        submitter=submitter,
        now=NOW + timedelta(minutes=3),
    )
    assert launched["status"] == "scene_configuration_launch_queued"
    assert launched["profile_id"] == profile["profile_id"]
    assert launched["submitted_through_webapp"] is True
    assert launched["paid_execution_requested"] is True
    assert len(submissions) == 1
    request = submissions[0]
    assert set(request) == {
        "launch_id",
        "run_id",
        "profile_id",
        "profile_digest",
        "rights",
        "spend",
        "confirm_execution",
    }
    assert request["confirm_execution"] is True
    assert request["launch_id"] == request["run_id"] == launched["launch_id"]
    assert request["launch_id"].endswith("-launch")
    assert request["profile_id"] == profile["profile_id"]
    assert request["profile_digest"] == profile["profile_digest"]
    assert request["rights"] == {
        "scope": "internal_noncommercial_research_only",
        "evidence": {
            "uri": profile["evaluation_run_spec"]["uri"],
            "digest": profile["evaluation_run_spec"]["digest"],
        },
    }
    assert request["spend"] == {
        "max_spend_usd": standing["max_total_spend_usd"],
        "expires_at": standing["expires_at"],
    }

    # A second pass returns the sealed launch state and never submits twice.
    again = automation.advance_scene_configuration_launch(
        progression=observed,
        activation_queue_root=tmp_path / "activations",
        profile_dir=tmp_path / "profiles",
        standing_authorization_dir=tmp_path / "standing",
        progression_root=tmp_path / "progression",
        intent=intent,
        submitter=submitter,
        now=NOW + timedelta(minutes=9),
    )
    assert again == launched
    assert len(submissions) == 1


def test_launch_refuses_a_standing_authorization_that_disagrees_with_the_profile(
    tmp_path: Path,
) -> None:
    intent_root, intent = _intent(tmp_path)
    result_path = _preparation(tmp_path)
    observed, _l, _w = _advance(tmp_path, result_path, intent_root)
    profile, standing, _activation_result = _materialized_activation(tmp_path, observed)
    standing_path = tmp_path / "standing" / f"{profile['profile_id']}.json"
    _write(standing_path, dict(standing, profile_digest="sha256:" + "1" * 64))
    with pytest.raises(
        automation.SceneConfigurationActivationAutomationError,
        match="scene_configuration_launch_authority_mismatch",
    ):
        automation.advance_scene_configuration_launch(
            progression=observed,
            activation_queue_root=tmp_path / "activations",
            profile_dir=tmp_path / "profiles",
            standing_authorization_dir=tmp_path / "standing",
            progression_root=tmp_path / "progression",
            intent=intent,
            submitter=lambda request: pytest.fail("must not submit"),
            now=NOW + timedelta(minutes=3),
        )


def test_process_scans_the_preparation_queue_and_reports_one_row_per_configuration(
    tmp_path: Path,
) -> None:
    intent_root, _intent_value = _intent(tmp_path)
    _preparation(tmp_path)
    lineage_publisher = _publisher("scene-configuration-activation-lineage")
    window_publisher = _publisher("coordinator-release-windows")
    rows = automation.process_scene_configuration_activations(
        preparation_queue_root=tmp_path / "preparations",
        activation_queue_root=tmp_path / "activations",
        progression_root=tmp_path / "progression",
        intent_root=intent_root,
        configured_controls_intent_root=_controls_intent_root(tmp_path),
        profile_dir=tmp_path / "profiles",
        standing_authorization_dir=tmp_path / "standing",
        provider_zero_collector=lambda: _provider_zero(NOW - timedelta(seconds=20)),
        lineage_publisher_factory=lambda: lineage_publisher,
        release_window_publisher_factory=lambda: window_publisher,
        submitter=lambda request: pytest.fail("no authority yet"),
        now=NOW,
    )
    assert [row["status"] for row in rows] == ["awaiting_scene_configuration_authority"]
    assert rows[0]["preparation_id"] == PREPARATION_ID
    assert rows[0]["activation_status"] == "scene_configuration_activation_queued"
