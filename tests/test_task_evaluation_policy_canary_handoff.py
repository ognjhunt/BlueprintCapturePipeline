"""Automatic policy-canary hand-off after the configured controls run completes.

The hand-off composes only existing production producers: the presubmission
setup, the profile materializer, the profile/catalog publisher and the WebApp
service channel.  These tests drive it with a fake object store, a fake
publisher and a fake WebApp so no network, secret or paid boundary is touched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_policy_canary_handoff as handoff
from blueprint_pipeline import (
    task_evaluation_scene_configuration_activation_automation as activation_automation,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_task_evaluation_launch_dispatcher import _profile as base_launch_profile
from tests.test_task_evaluation_policy_canary_scene_setup import _inputs as presubmission_inputs
from tests.test_task_evaluation_policy_run_contract import _template as policy_template

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGURATION_COMMIT = "1359447d4" + "1" * 31
COMMIT = "f50061836" + "2" * 31
TEAM = "blueprint-internal"
SCENE_ID = "interiorgs-841007"
TASK_ID = "scene-841007-book-to-tray"
SOURCE_LAUNCH_ID = "adp-new-scene-book-to-tray-841007-1359447d-20260905t080000z-scene-configuration"
CONFIGURATION_RUN_ID = SOURCE_LAUNCH_ID
CONSTRUCTION_LAUNCH_ID = SOURCE_LAUNCH_ID + "-franka-construction"
CONTROLS_LAUNCH_ID = SOURCE_LAUNCH_ID + "-franka-controls"
PREPARATION_ID = SOURCE_LAUNCH_ID + "-franka-controls-1359447d4111-episode-preparation"
REVISION_DIGEST = "sha256:" + "7" * 64
OFFERING_DIGEST = "sha256:" + "8" * 64


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value if isinstance(value, bytes) else (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    return path


def _sealed(value: dict, field: str) -> dict:
    value[field] = ""
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


class _Publisher:
    """An object-store publisher that seals whatever it is handed."""

    def __init__(self) -> None:
        self.published: dict[str, bytes] = {}

    def __call__(self, *, path: Path, object_name: str) -> dict:
        payload = Path(path).read_bytes()
        uri = f"s3://blueprint/task-evaluation/production-inputs/{object_name}"
        self.published[uri] = payload
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        return {
            "uri": uri,
            "digest": digest,
            "size_bytes": len(payload),
            "full_byte_service_account_readback_passed": True,
            "readback_digest": digest,
            "readback_size_bytes": len(payload),
        }


def _terminal_run(tmp_path: Path, *, launch_id: str, result: dict) -> Path:
    """A completed native-arena run directory as the dispatcher leaves it."""

    run_root = tmp_path / "launch-runs" / launch_id
    authority_path = _write(
        tmp_path / "inputs" / launch_id / "attempt-authority.json",
        {"schema_version": "native_task_arena_paid_attempt_authority.v1"},
    )
    reconciliation_path = _write(
        tmp_path / "inputs" / launch_id / "spend-reconciliation.json",
        {"schema_version": "adp_same_goal_spend_reconciliation.v1"},
    )
    native_result_path = _write(run_root / "allocator" / "native-result.json", _sealed(dict(result), "result_digest"))
    native_result = json.loads(native_result_path.read_text())
    allocator_path = _write(
        run_root / "allocator" / "result.json",
        {
            "schema_version": "native_task_arena_vast_run.v1",
            "status": "completed",
            "blockers": [],
            "native_control_result_path": str(native_result_path),
            "native_control_result_digest": native_result["result_digest"],
        },
    )
    profile = _sealed(
        {
            "schema_version": "task_evaluation_launch_profile.v1",
            "profile_id": f"{launch_id}-profile",
            "immutable_inputs": [
                {"name": "native_task_arena_attempt_authority", "path": str(authority_path), "digest": _digest(authority_path)},
                {
                    "name": "native_task_arena_attempt_authority_prior_spend_reconciliation",
                    "path": str(reconciliation_path),
                    "digest": _digest(reconciliation_path),
                },
            ],
        },
        "profile_digest",
    )
    _write(run_root / "launch_profile.json", profile)
    receipt = _sealed(
        {
            "schema_version": "task_evaluation_launch_receipt.v1",
            "status": "completed",
            "launch_id": launch_id,
            "run_id": launch_id,
            "request_digest": "sha256:" + "1" * 64,
            "launch_profile_digest": profile["profile_digest"],
            "terminal_evidence": {
                "status": "passed",
                "result": {"path": str(allocator_path), "exists": True, "digest": _digest(allocator_path)},
            },
        },
        "receipt_digest",
    )
    _write(run_root / "launch_receipt.json", receipt)
    sync = _sealed(
        {
            "schema_version": "task_evaluation_launch_webapp_sync_result.v1",
            "status": "succeeded",
            "launch_id": launch_id,
            "run_id": launch_id,
            "request_digest": receipt["request_digest"],
            "receipt_digest": receipt["receipt_digest"],
            "attempt_number": 1,
            "attempted_at": "2026-09-05T08:30:00+00:00",
            "provider_mutation_performed": False,
            "response": {
                "schema_version": "task_evaluation_launch_web_sync_receipt.v1",
                "status": "completed",
                "already_exists": False,
                "launch_id": launch_id,
                "run_id": launch_id,
                "request_digest": receipt["request_digest"],
                "receipt_digest": receipt["receipt_digest"],
            },
        },
        "sync_result_digest",
    )
    _write(run_root / "webapp_sync_succeeded.json", sync)
    zero = _sealed(
        {
            "schema_version": "task_evaluation_post_teardown_provider_zero.v1",
            "status": "provider_zero_confirmed",
            "launch_id": launch_id,
            "run_id": launch_id,
            "request_digest": receipt["request_digest"],
            "receipt_digest": receipt["receipt_digest"],
            "launch_profile_digest": profile["profile_digest"],
            "provider_zero_verified": True,
            "continuing_spend_from_this_run": False,
            "allocator_invoked": False,
            "provider_mutation_performed": False,
            "automatic_retry_performed": False,
            "blockers": [],
        },
        "provider_zero_receipt_digest",
    )
    _write(run_root / "post_teardown_provider_zero_receipt.json", zero)
    return run_root


def _configured_run(tmp_path: Path) -> Path:
    """The Website-started scene-configuration run: request, base profile, terminal evidence."""

    run_root = tmp_path / "launch-runs" / SOURCE_LAUNCH_ID
    (tmp_path / "configured-profile-inputs").mkdir(parents=True, exist_ok=True)
    profile = base_launch_profile(tmp_path / "configured-profile-inputs")
    profile.update(
        {
            "profile_id": f"{SOURCE_LAUNCH_ID}-profile",
            "source_commit": CONFIGURATION_COMMIT,
            "task_evaluation_run": {
                "run_mode": "scene_configuration",
                "team_namespace": TEAM,
                "scene_id": SCENE_ID,
                "task_id": TASK_ID,
                "configuration_run_id": CONFIGURATION_RUN_ID,
                "evaluation_episode_executed": False,
            },
        }
    )
    _sealed(profile, "profile_digest")
    _write(run_root / "launch_profile.json", profile)
    request = _sealed(
        {
            "schema_version": "task_evaluation_launch_request.v1",
            "launch_id": SOURCE_LAUNCH_ID,
            "run_id": SOURCE_LAUNCH_ID,
            "launch_profile_id": profile["profile_id"],
            "launch_profile_digest": profile["profile_digest"],
            "source_commit": CONFIGURATION_COMMIT,
        },
        "request_digest",
    )
    _write(run_root / "launch_request.json", request)
    return run_root


def _progression_state(tmp_path: Path, *, controls_terminal: bool = True) -> Path:
    """The configured-controls progression after both phases were launched."""

    state = tmp_path / "progression" / SOURCE_LAUNCH_ID / f"franka-controls-{COMMIT[:12]}"
    template = policy_template()
    episode_request = {
        field: template[field]
        for field in ("scene", "construction", "robot", "controller", "task", "sensors", "runtime", "execution_adapter", "spend")
    }
    episode_request.update(
        {
            "schema_version": "task_evaluation_launch_preparation_request.v1",
            "run_mode": "episode_evaluation",
            "expected_production_commit": COMMIT,
            "preparation_id": PREPARATION_ID,
            "team_namespace": TEAM,
            "run_id": PREPARATION_ID.removesuffix("-preparation"),
            "publication": {"input_namespace": SOURCE_LAUNCH_ID, "service_account_readback_required": True},
        }
    )
    episode_request["scene"] = {**episode_request["scene"], "identity": {"id": SCENE_ID, "version": "book-tray-v2"}}
    episode_request["task"] = {**episode_request["task"], "identity": {"id": TASK_ID, "version": "v1"}}
    _write(
        state / "configured_controls_progression.v1.json",
        _sealed(
            {
                "schema_version": "task_evaluation_configured_controls_progression.v1",
                "status": "episode_preparation_queued",
                "configuration_run_id": CONFIGURATION_RUN_ID,
                "configured_scene_revision_digest": REVISION_DIGEST,
                "configured_scene_offering_digest": OFFERING_DIGEST,
                "expected_production_commit": COMMIT,
                "episode_preparation_request": episode_request,
                "episode_preparation_request_digest": canonical_digest(episode_request),
            },
            "progression_digest",
        ),
    )
    _write(
        state / "construction_launch_progression.json",
        _sealed({"schema_version": "task_evaluation_configured_controls_progression.v1", "status": "construction_launch_queued", "launch_id": CONSTRUCTION_LAUNCH_ID}, "progression_digest"),
    )
    _write(
        state / "controls_launch_progression.json",
        _sealed({"schema_version": "task_evaluation_configured_controls_progression.v1", "status": "controls_pair_launch_queued", "launch_id": CONTROLS_LAUNCH_ID}, "progression_digest"),
    )
    _terminal_run(
        tmp_path,
        launch_id=CONSTRUCTION_LAUNCH_ID,
        result={
            "schema_version": "native_task_arena_construction_result.v1",
            "status": "completed",
            "construction_gate_qualified": True,
            "candidate_policy_queried": False,
            "blockers": [],
        },
    )
    if controls_terminal:
        _terminal_run(
            tmp_path,
            launch_id=CONTROLS_LAUNCH_ID,
            result={
                "schema_version": "native_task_arena_control_result.v1",
                "status": "completed",
                "controls_qualified": True,
                "candidate_policy_queried": False,
                "blockers": [],
            },
        )
    return state


def _compiled_packet(tmp_path: Path) -> Path:
    """The no-spend construction compilation: adapter result, packet root, runtime receipt."""

    inputs = presubmission_inputs(tmp_path / "presubmission-inputs")
    scene_plan = json.loads(Path(inputs["scene"]).read_text())
    scene_plan.update({"scene_id": SCENE_ID, "task_id": TASK_ID})
    scene_plan["task_spec"].update(
        {"instruction_subject_label": "open book", "visible_target_label": "blue document tray"}
    )
    _sealed(scene_plan, "plan_digest")
    compiled = tmp_path / "compiled-episodes" / PREPARATION_ID
    packet_root = compiled / "native-task-packet"
    _write(packet_root / "native_task_arena_scene_plan.v1.json", scene_plan)
    packet_receipt = _write(
        packet_root / "native_task_arena_packet_receipt.v1.json",
        {
            "schema_version": "native_task_arena_packet_receipt.v1",
            "scene_id": SCENE_ID,
            "task_id": TASK_ID,
            "arena_scene_plan_digest": scene_plan["plan_digest"],
        },
    )
    runtime_receipt = _write(compiled / "runtime-source" / "native_task_runtime_source_packet.v1.json", json.loads(Path(inputs["runtime"]).read_text()))
    adapter = _sealed(
        {
            "schema_version": "task_evaluation_native_arena_adapter_result.v1",
            "status": "native_arena_adapter_materialized",
            "preparation_id": PREPARATION_ID,
            "source_commit": COMMIT,
            "packet_root": str(packet_root),
            "runtime_source_receipt": str(runtime_receipt),
            "packet_receipt_digest": _digest(packet_receipt),
            "runtime_source_receipt_digest": _digest(runtime_receipt),
        },
        "result_digest",
    )
    adapter_path = _write(compiled / "task_evaluation_native_arena_adapter_result.v1.json", adapter)
    result = _sealed(
        {
            "schema_version": "task_evaluation_episode_compilation_result.v1",
            "status": "compiled_for_production_launch",
            "compilation_id": PREPARATION_ID,
            "run_id": PREPARATION_ID.removesuffix("-preparation"),
            "team_namespace": TEAM,
            "source_commit": COMMIT,
            "configured_scene_revision_digest": REVISION_DIGEST,
            "adapter_result_path": str(adapter_path),
            "adapter_result_digest": adapter["result_digest"],
            "compiled_by_production": True,
            "provider_mutation_performed": False,
            "paid_execution_requested": False,
            "blockers": [],
        },
        "result_digest",
    )
    queue = tmp_path / "compilations"
    _write(queue / "results" / f"{PREPARATION_ID}-{'9' * 64}.json", result)
    return queue


def _activation_intent(tmp_path: Path) -> Path:
    intent_root = tmp_path / "activation-intents"
    intent_root.mkdir()
    spend = _write(
        tmp_path / "spend.json",
        {"schema_version": "adp_project_spend_reconciliation.v1", "total_cost_usd": 0.0},
    )
    activation_automation.provision_scene_configuration_activation_intent(
        expected_production_commit=COMMIT,
        team_namespace=TEAM,
        scene_id=SCENE_ID,
        task_id=TASK_ID,
        authorization_reference="Blueprint owner direction 2026-09-05: scene 841007 book-to-tray end to end",
        authorized_by="nijelhunt_1",
        profile_revision="r1",
        valid_for_seconds=21_600,
        project_spend_reconciliation_path=spend,
        rights_scope="internal_noncommercial_research_only",
        maximum_hard_cap_usd=12.0,
        release_reference="Scene 841007 scene-configuration automatic activation",
        intent_root=intent_root,
        materialization_root=tmp_path / "activation-intent-inputs",
    )
    return intent_root


class _WebApp:
    def __init__(self, *, status: int = 202) -> None:
        self.calls: list[dict] = []
        self.status = status

    def __call__(self, *, endpoint: str, headers: dict, body: bytes) -> tuple[int, bytes]:
        selection = json.loads(body)
        self.calls.append({"endpoint": endpoint, "headers": dict(headers), "selection": selection})
        receipt = {
            "schema_version": "task_evaluation_policy_canary_web_receipt.v1",
            "status": "forward_pending",
            "already_exists": False,
            "submission_channel": "production_webapp_service_api",
            "run": {"run_id": selection["run_id"], "state": "forward_pending", "request_digest": "sha256:" + "a" * 64},
            "forward": {"status": "forwarded"},
            "warning": "Controls pending — results are unqualified.",
        }
        return self.status, json.dumps(receipt).encode()


def _profile_publisher(calls: list[dict]):
    def publish(*, profile_path: Path, profile_dir: Path, webapp_catalog_out: Path) -> dict:
        calls.append({"profile_path": Path(profile_path), "profile_dir": Path(profile_dir), "catalog": Path(webapp_catalog_out)})
        return {"schema_version": "task_evaluation_launch_profile_publication.v1", "status": "published", "webapp_catalog_digest": "sha256:" + "c" * 64}

    return publish


def _model_rights(*, template_path: str, repo_root: str, source_commit: str, scene_id: str, task_id: str, output_path: str) -> dict:
    """The rights producer re-bound to a checkout that is not this test's release."""

    template = json.loads(Path(template_path).read_text())
    document = {**template, "scene_id": scene_id, "task_id": task_id, "source_commit": source_commit}
    _sealed(document, "rights_digest")
    _write(Path(output_path), document)
    return document


def _advance(tmp_path: Path, *, state: Path, webapp: _WebApp, publisher: _Publisher, profile_calls: list[dict], **overrides) -> dict:
    return handoff.advance_policy_canary_handoff(
        **overrides,
        model_rights_materializer=_model_rights,
        state_root=state,
        source_launch_id=SOURCE_LAUNCH_ID,
        expected_production_commit=COMMIT,
        launch_state_root=tmp_path / "launch-runs",
        episode_compilation_queue_root=tmp_path / "compilations",
        activation_intent_root=tmp_path / "activation-intents",
        repo_root=REPO_ROOT,
        profile_dir=tmp_path / "profiles",
        webapp_catalog_out=tmp_path / "catalog.json",
        webapp_secret=b"service-secret",
        webapp_endpoint="https://tryblueprint.io/api/internal/task-evaluation-launch-submissions",
        notification_email="ohstnhunt@gmail.com",
        publisher_factory=lambda: publisher,
        profile_publisher=_profile_publisher(profile_calls),
        poster=webapp,
    )


def _prepared(tmp_path: Path, *, controls_terminal: bool = True) -> Path:
    _configured_run(tmp_path)
    _compiled_packet(tmp_path)
    _activation_intent(tmp_path)
    return _progression_state(tmp_path, controls_terminal=controls_terminal)


@pytest.mark.parametrize("scene_id", ["841007", "site-capture-ae539f2c-f6aa-4cbd-9f99-3ed72017791e-development"])
def test_controller_configuration_is_rebound_to_the_scene_and_its_quick10_matrix(tmp_path: Path, scene_id: str) -> None:
    from blueprint_pipeline.task_evaluation_policy_canary_scene_setup import _quick_cells

    document = handoff.rebind_policy_controller_configuration_to_scene(
        template_path=REPO_ROOT / "config/trial/scene839873_policy_canary_controller_configuration.v1.json",
        scene_id=scene_id,
        task_id=TASK_ID,
        scene_revision_digest=REVISION_DIGEST,
    )
    cells = _quick_cells(REVISION_DIGEST, scene_id=scene_id)
    assert document["scene_id"] == scene_id
    assert document["task_id"] == TASK_ID
    assert document["schema_version"] == f"scene{scene_id}_policy_canary_controller_configuration.v1"
    assert [cell["cell_id"] for cell in document["quick_10"]["cells"]] == [cell["cell_id"] for cell in cells]
    assert document["quick_10"]["matrix_digest"] == canonical_digest({"cells": cells})
    assert document["configuration_digest"] == canonical_digest(document, digest_field="configuration_digest")
    assert "839873" not in json.dumps(document)


def test_controller_rebind_keeps_declared_scene_mounting_in_its_matrix():
    plan = {"objects": [{"task_subject": True,
        "articulation_adaptation": {"scene_mounting": {"mode": "fixed_in_scene"}}}]}
    document = handoff.rebind_policy_controller_configuration_to_scene(
        template_path=REPO_ROOT / handoff.MANIFESTS["policy_controller_template"],
        scene_id="another-site", task_id="another-task", scene_revision_digest=REVISION_DIGEST,
        scene_plan=plan)
    expected = handoff._quick_cells(REVISION_DIGEST, scene_id="another-site", scene_plan=plan)
    assert document["quick_10"]["cells"] == expected
    assert document["quick_10"]["matrix_digest"] == canonical_digest({"cells": expected})
    assert expected[2]["resolved_scenario"]["runtime_coverage_gaps"]


def _fail_final_seal(monkeypatch):
    original = handoff._seal_state
    def fail(path, value):
        if value["status"] == "canary_launch_submitted":
            raise OSError("injected final seal failure")
        return original(path, value)
    monkeypatch.setattr(handoff, "_seal_state", fail)
    return original


