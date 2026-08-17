from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.scene_strict_family_supervisor import (
    SceneStrictFamilyError,
    audit_scene_families,
    derive_governed_families,
)
from blueprint_pipeline.task_evaluation_artifact_manifest import (
    build_task_evaluation_artifact_manifest,
)
from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    CANONICAL_ALLOCATOR_ENTRYPOINT,
)


SCENE = "840920"
COMMIT = "c" * 40
TASKS = ("task_a_washer_door_open", "task_b_notebook_relocation")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "supervise_scene_strict_family_ledger.py"


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _record(path: Path, *, role: str, schema: str | None) -> dict:
    return {
        "role": role,
        "schema_version": schema,
        "digest_field": None,
        "legacy_digest_gap": "exact_source_bytes_sha256_bound_no_canonical_digest",
        "record": {
            "path": str(path.resolve()),
            "sha256": _sha(path),
            "size_bytes": path.stat().st_size,
        },
    }


def _profile_files(input_root: Path) -> tuple[Path, Path]:
    source = _write(input_root / "source.json", {"scene_id": SCENE})
    spec = _write(input_root / "spec.json", {"scene_id": SCENE})
    return source, spec


def _make_launch(
    launch_root: Path,
    *,
    probe_kind: str,
    task_marker: str | None = None,
    scene_id: str = SCENE,
    input_root: Path | None = None,
) -> Path:
    suffix = f"-{task_marker}" if task_marker else ""
    launch_id = f"scene{SCENE}-{probe_kind}{suffix}"
    run_root = launch_root / launch_id
    source_manifest, evaluation_spec = _profile_files(input_root or run_root / "inputs")
    source_bundle = {
        "bundle_id": f"scene{scene_id}-{probe_kind}{suffix}",
        "source_kind": "interiorgs_sage",
        "uri": f"r2://blueprint/scene{scene_id}/{probe_kind}{suffix}.json",
        "digest": "sha256:" + "a" * 64,
    }
    run_spec = {
        "uri": f"r2://blueprint/scene{scene_id}/run-spec{suffix}.json",
        "digest": "sha256:" + "b" * 64,
    }
    result_path = run_root / "allocator" / "result.json"
    profile = {
        "schema_version": "task_evaluation_launch_profile.v1",
        "profile_id": f"scene{scene_id}-{probe_kind}{suffix}-{COMMIT}",
        "program_id": "arm-decision-proof-v1",
        "source_bundle": source_bundle,
        "evaluation_run_spec": run_spec,
        "immutable_inputs": [
            {
                "name": "source_bundle_manifest",
                "path": str(source_manifest.resolve()),
                "digest": _sha(source_manifest),
            },
            {
                "name": "evaluation_run_spec",
                "path": str(evaluation_spec.resolve()),
                "digest": _sha(evaluation_spec),
            },
        ],
        "allocator": {
            "entrypoint": CANONICAL_ALLOCATOR_ENTRYPOINT,
            "subcommand": "gpu-canary",
            "argv": [
                "--expected-source-commit",
                COMMIT,
                "--probe-kind",
                probe_kind,
                "--adapter-output",
                str(result_path),
            ],
            "max_spend_usd": 2.0,
            "hard_ttl_seconds": 600,
            "retry_cap": 0,
        },
        "runtime_environment": {},
        "execution_admission": {
            "live_enabled": True,
            "readiness_receipt": {
                "uri": f"r2://blueprint/scene{scene_id}/readiness.json",
                "digest": "sha256:" + "d" * 64,
            },
            "blockers": [],
        },
        "reconciliation": {"required_providers": ["vast"], "max_guard_age_seconds": 300},
        "webapp_sync": {"max_attempts": 20},
        "terminal_contract": {
            "result_path": str(result_path),
            "success_statuses": ["completed"],
            "required_values": {"continuing_spend_from_this_run": False, "retry_cap": 0},
            "required_path_fields": ["teardown_manifest_path", "artifact_manifest_path"],
        },
        "required_controls": {
            "canonical_allocator": CANONICAL_ALLOCATOR_ENTRYPOINT,
            "secret_profile_id": "canonical-vast-adp",
            "watchdog_required": True,
            "artifact_storage_required": True,
            "teardown_required": True,
            "provider_zero_required": True,
            "webapp_status_sync_required": True,
            "retry_cap": 0,
        },
        "claim_ceiling": "development_only",
    }
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    _write(run_root / "launch_profile.json", profile)
    request = {
        "schema_version": "task_evaluation_launch_request.v1",
        "launch_id": launch_id,
        "run_id": launch_id,
        "launch_profile_id": profile["profile_id"],
        "launch_profile_digest": profile["profile_digest"],
        "source_bundle": source_bundle,
        "evaluation_run_spec": run_spec,
        "authorization": {
            "actor": {"id": "production-runner", "role": "ops"},
            "authorized_at": datetime.now(timezone.utc).isoformat(),
            "rights": {
                "approved": True,
                "scope": "internal_noncommercial_research_only",
                "evidence": {
                    "uri": "gs://blueprint/rights.json",
                    "digest": "sha256:" + "e" * 64,
                },
            },
            "spend": {
                "approved": True,
                "currency": "USD",
                "max_spend_usd": 2.0,
                "expires_at": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
            },
            "execution": {"approved": True},
        },
        "required_controls": profile["required_controls"],
        "claim_ceiling": "development_only",
        "idempotency_key": launch_id,
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    _write(run_root / "launch_request.json", request)
    binding = {
        "schema_version": "task_evaluation_launch_binding.v1",
        "launch_id": launch_id,
        "run_id": launch_id,
        "request_digest": request["request_digest"],
        "profile_digest": profile["profile_digest"],
        "source_bundle_digest": source_bundle["digest"],
        "evaluation_run_spec_digest": run_spec["digest"],
        "canonical_allocator": CANONICAL_ALLOCATOR_ENTRYPOINT,
        "execute_requested": True,
        "execute_launch_id": "",
        "execute_env_allowed": True,
        "secret_profile_id_match": True,
        "profile_live_enabled": True,
    }
    binding["binding_digest"] = canonical_digest(binding, digest_field="binding_digest")
    _write(run_root / "launch_binding.json", binding)

    provider_run = run_root / "allocator" / "job" / "vast_provider_run"
    execution = _write(
        run_root / "allocator" / "job" / "immutable_execution" / "execution.json",
        {"schema_version": "fixture_execution.v1", "status": "completed"},
    )
    teardown = _write(
        provider_run / "vast_teardown_manifest.json",
        {
            "schema_version": "vast_teardown_manifest.v1",
            "generated_at": "2026-08-17T01:00:00+00:00",
            "status": "completed",
            "vast_instance_ids": [100],
            "teardown_actions_performed": [
                {"instance_id": 100, "action": "destroy_instance", "status": "completed"}
            ],
            "continuing_spend_from_this_run": False,
        },
    )
    _write(
        provider_run / "vast_provider_adapter_result.json",
        {"schema_version": "vast_provider_adapter_result.v1", "status": "completed"},
    )
    manifest = build_task_evaluation_artifact_manifest(
        attempt_root=run_root / "allocator" / "job",
        artifact_roots={
            "provider_runtime_evidence": execution.parent,
            "allocator_adapter_result": provider_run / "vast_provider_adapter_result.json",
            "teardown_manifest": teardown,
        },
        required_roles=(
            "provider_runtime_evidence",
            "allocator_adapter_result",
            "teardown_manifest",
        ),
        binding={"allocator_lane": probe_kind, "retry_cap": 0},
    )
    manifest_path = run_root / "allocator" / "job" / "artifact_manifest.json"
    result = {
        "schema_version": "fixture_terminal_result.v1",
        "status": "completed",
        "blockers": [],
        "continuing_spend_from_this_run": False,
        "retry_cap": 0,
        "bundle_sha256": "sha256:" + "f" * 64,
        "artifact_manifest_path": str(manifest_path.resolve()),
        "teardown_manifest_path": str(teardown.resolve()),
    }
    _write(result_path, result)
    receipt = {
        "schema_version": "task_evaluation_launch_receipt.v1",
        "status": "completed",
        "launch_id": launch_id,
        "run_id": launch_id,
        "request_digest": request["request_digest"],
        "launch_profile_digest": profile["profile_digest"],
        "binding_digest": binding["binding_digest"],
        "canonical_allocator": CANONICAL_ALLOCATOR_ENTRYPOINT,
        "allocator_exit_code": 0,
        "execute_requested": True,
        "execute_launch_id": "",
        "provider_mutation_attempted": True,
        "prelaunch_skill_execution": {"status": "not_configured", "blockers": []},
        "terminal_evidence": {
            "status": "passed",
            "result": {"path": str(result_path.resolve()), "exists": True, "digest": _sha(result_path)},
            "artifacts": {
                "artifact_manifest_path": {
                    "path": str(manifest_path.resolve()),
                    "exists": True,
                    "digest": _sha(manifest_path),
                },
                "teardown_manifest_path": {
                    "path": str(teardown.resolve()),
                    "exists": True,
                    "digest": _sha(teardown),
                },
            },
            "blockers": [],
        },
        "blockers": [],
        "raw_secret_values_recorded": False,
        "agent_operator_used": False,
        "claim_ceiling": "development_only",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    _write(run_root / "launch_receipt.json", receipt)
    sync = {
        "schema_version": "task_evaluation_launch_webapp_sync_result.v1",
        "status": "succeeded",
        "launch_id": launch_id,
        "run_id": launch_id,
        "request_digest": request["request_digest"],
        "receipt_digest": receipt["receipt_digest"],
        "response": {
            "launch_id": launch_id,
            "run_id": launch_id,
            "request_digest": request["request_digest"],
            "receipt_digest": receipt["receipt_digest"],
        },
        "attempt_number": 1,
        "attempted_at": "2026-08-17T01:01:00+00:00",
        "provider_mutation_performed": False,
    }
    sync["sync_result_digest"] = canonical_digest(sync, digest_field="sync_result_digest")
    _write(run_root / "webapp_sync_succeeded.json", sync)
    zero = {
        "schema_version": "task_evaluation_post_teardown_provider_zero.v1",
        "status": "provider_zero_confirmed",
        "launch_id": launch_id,
        "run_id": launch_id,
        "request_digest": request["request_digest"],
        "receipt_digest": receipt["receipt_digest"],
        "launch_profile_digest": profile["profile_digest"],
        "required_providers": ["vast"],
        "provider_zero_verified": True,
        "continuing_spend_from_this_run": False,
        "allocator_invoked": False,
        "provider_mutation_performed": False,
        "automatic_retry_performed": False,
        "blockers": [],
        "observed_at": "2026-08-17T01:02:00+00:00",
        "teardown_manifest": {"digest": _sha(teardown)},
    }
    guard = {
        "schema_version": "task_evaluation_provider_zero_guard_snapshot.v1",
        "source_guard_report_path": "/retained/read-only-guard.json",
        "source_guard_report_sha256": "sha256:" + "7" * 64,
        "source_guard_generated_at": "2026-08-17T01:01:30+00:00",
        "guard": {"provider_zero_verified": True, "live_instance_count": 0},
    }
    guard["snapshot_digest"] = canonical_digest(guard, digest_field="snapshot_digest")
    guard_path = _write(
        run_root
        / "provider_zero_guard_snapshots"
        / f"{guard['snapshot_digest'][7:]}.json",
        guard,
    )
    zero["independent_guard_snapshot"] = {
        "path": str(guard_path.resolve()),
        "snapshot_digest": guard["snapshot_digest"],
        "source_guard_generated_at": guard["source_guard_generated_at"],
        "source_guard_report_sha256": guard["source_guard_report_sha256"],
    }
    zero["provider_zero_receipt_digest"] = canonical_digest(
        zero, digest_field="provider_zero_receipt_digest"
    )
    zero_path = _write(run_root / "post_teardown_provider_zero_receipt.json", zero)
    allocation = {
        "program_id": "arm-decision-proof-v1",
        "probe_kind": probe_kind,
        "orchestrator_source_commit": COMMIT,
        "paid_attempt_authority_digest": "sha256:" + hashlib.sha256(launch_id.encode()).hexdigest(),
        "retry_cap": 0,
    }
    admission = {
        "schema_version": "paid_lane_admission.v1",
        "status": "admitted",
        "blockers": [],
        "probe_kind": probe_kind,
        "retry_cap": 0,
        "control_plane_identity": {
            "orchestrator_source_commit": COMMIT,
            "checkout_clean": True,
            "identity_probe_ran": True,
            "origin_main_commit": COMMIT,
            "remote_main_commit": COMMIT,
            "orchestrator_equals_origin_main": True,
            "orchestrator_equals_remote_main": True,
        },
        "allocation_binding": allocation,
        "allocation_binding_digest": canonical_digest(allocation),
    }
    admission_path = _write(run_root / "allocator" / "admission.json", admission)
    billing_response = _write(
        run_root / "official_billing" / "response.json",
        {"results": [{"source": "instance-100", "amount": 0.25}]},
    )
    entry = {
        "schema_version": "adp_same_goal_spend_entry.v1",
        "goal_id": "arm-decision-proof-v1",
        "attempt_id": launch_id,
        "lane": probe_kind,
        "evidence_kind": "fully_bound_official_billing",
        "provider_instance_id": 100,
        "cost_usd": 0.25,
        "authority_digest": allocation["paid_attempt_authority_digest"],
        "bundle_sha256": result["bundle_sha256"],
        "continuing_spend_from_this_run": False,
        "provider_zero_confirmed": True,
        "source_receipts": [
            _record(result_path, role="terminal_result", schema=result["schema_version"]),
            _record(teardown, role="teardown_manifest", schema=teardown and "vast_teardown_manifest.v1"),
            _record(zero_path, role="provider_zero", schema=zero["schema_version"]),
            _record(billing_response, role="official_billing_response", schema=None),
            _record(admission_path, role="admission", schema=admission["schema_version"]),
        ],
        "bindings": [
            {
                "kind": "cost_usd",
                "source_role": "official_billing_response",
                "json_path": ["results", 0, "amount"],
                "expected_value": 0.25,
            },
            {
                "kind": "authority_digest",
                "source_role": "admission",
                "json_path": ["allocation_binding", "paid_attempt_authority_digest"],
                "expected_value": allocation["paid_attempt_authority_digest"],
            },
            {
                "kind": "provider_zero",
                "source_role": "provider_zero",
                "json_path": ["provider_zero_verified"],
                "expected_value": True,
            },
        ],
    }
    entry["entry_digest"] = canonical_digest(entry, digest_field="entry_digest")
    reconciliation = {
        "schema_version": "adp_same_goal_spend_reconciliation.v1",
        "status": "all_same_goal_paid_attempts_terminal_and_provider_zero",
        "goal_id": "arm-decision-proof-v1",
        "entries": [entry],
        "entry_count": 1,
        "total_cost_usd": 0.25,
        "provider_mutation_performed": False,
        "paid_resource_allocated": False,
        "raw_secret_values_recorded": False,
        "blockers": [],
    }
    reconciliation["receipt_digest"] = canonical_digest(
        reconciliation, digest_field="receipt_digest"
    )
    _write(run_root / "official_billing" / "same_goal_reconciliation.json", reconciliation)
    assert manifest["status"] == "completed"
    return run_root


@pytest.fixture
def scene_840920_five_family_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    launches = tmp_path / "launches"
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    roots: dict[str, Path] = {}
    for probe in ("semantic-sam31-source-tracks", "adp-gaussian-excision", "adp-usd-content-agents"):
        for marker in ("task-a", "task-b"):
            roots[f"{probe}:{marker}"] = _make_launch(
                launches, probe_kind=probe, task_marker=marker
            )
    for probe in (
        "adp-retained-scene-gpu-render",
        "adp-artifixer3d-exact-support",
        "adp-paired-target-native-import",
    ):
        roots[probe] = _make_launch(launches, probe_kind=probe)
    return evidence, launches, roots


@pytest.fixture
def scene_840920_host_layout_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, Path], dict[str, Path]]:
    """Mirror production: launches and lane input roots are disjoint siblings."""

    launches = tmp_path / "pipeline-control-plane" / "task-evaluation-launch-runs"
    inputs = tmp_path / "task-evaluation-inputs"
    evidence = inputs / "scene840920-native-assets-53b84015-r1"
    evidence.mkdir(parents=True)
    roots: dict[str, Path] = {}
    billings: dict[str, Path] = {}
    for probe in (
        "semantic-sam31-source-tracks",
        "adp-gaussian-excision",
        "adp-usd-content-agents",
    ):
        for marker in ("task-a", "task-b"):
            key = f"{probe}:{marker}"
            lane_input = inputs / f"{probe}-scene840920-{marker}-host-layout"
            roots[key] = _make_launch(
                launches,
                probe_kind=probe,
                task_marker=marker,
                input_root=lane_input,
            )
            source = roots[key] / "official_billing" / "same_goal_reconciliation.json"
            destination = lane_input / "terminal_official_reconciliation.v1.json"
            source.rename(destination)
            billings[key] = destination
    for probe in (
        "adp-retained-scene-gpu-render",
        "adp-artifixer3d-exact-support",
        "adp-paired-target-native-import",
    ):
        lane_input = (
            evidence
            if probe in {
                "adp-artifixer3d-exact-support",
                "adp-paired-target-native-import",
            }
            else inputs / f"retained-scene-render-{SCENE}-host-layout"
        )
        roots[probe] = _make_launch(
            launches,
            probe_kind=probe,
            input_root=lane_input / probe,
        )
        source = roots[probe] / "official_billing" / "same_goal_reconciliation.json"
        destination = lane_input / f"{probe}-terminal-official-reconciliation.v1.json"
        source.rename(destination)
        billings[probe] = destination
    return evidence, launches, roots, billings


def test_denominator_is_derived_as_fifteen_from_seventeen_reachable_probes() -> None:
    families, derivation = derive_governed_families()

    assert len(families) == 15
    assert derivation["website_reachable_probe_count"] == 17
    assert len(next(row for row in families if row.family_id == "native_task_arena").probe_kinds) == 3
    assert len(
        next(
            row
            for row in families
            if row.family_id == "artifixer3d_paired_native_import"
        ).probe_kinds
    ) == 2


def test_supervise_scene_strict_family_ledger_cli_delegates_to_read_only_module() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "blueprint_pipeline.scene_strict_family_supervisor import main" in source
    assert "paid_resource_allocator" not in source


def test_scene_840920_shape_reports_honest_five_of_fifteen_and_next_checkpoint(
    scene_840920_five_family_fixture: tuple[Path, Path, dict[str, Path]],
) -> None:
    evidence, launches, _roots = scene_840920_five_family_fixture

    result = audit_scene_families(
        scene_id=SCENE,
        task_ids=TASKS,
        evidence_root=evidence,
        launch_state_root=launches,
        observed_at="2026-08-17T02:00:00+00:00",
    )

    assert result["strict_completed_family_count"] == 5
    assert result["strict_family_denominator"] == 15
    assert result["status"] == "in_progress"
    assert result["next_unproven_checkpoint"]["family_id"] == "production_ai_visual_review"
    assert {
        row["family_id"]
        for row in result["families"]
        if row["status"] == "strict_terminal_complete"
    } == {
        "semantic_source_tracks",
        "gaussian_excision",
        "retained_scene_render",
        "artifixer3d_paired_native_import",
        "usd_content_agents",
    }
    assert result["authority_boundary"]["provider_mutation_performed"] is False


def test_exact_host_layout_discovers_five_families_outside_one_input_root(
    scene_840920_host_layout_fixture: tuple[
        Path, Path, dict[str, Path], dict[str, Path]
    ],
) -> None:
    evidence, launches, _roots, _billings = scene_840920_host_layout_fixture

    result = audit_scene_families(
        scene_id=SCENE,
        task_ids=TASKS,
        evidence_root=evidence,
        launch_state_root=launches,
    )

    assert result["strict_completed_family_count"] == 5
    assert result["trusted_evidence_catalog"]["source"] == (
        "canonical_launch_state_and_allowed_scene_task_roots"
    )
    assert result["trusted_evidence_catalog"]["arbitrary_evidence_root_files_admitted"] is False


def test_other_root_billing_noise_cannot_replace_missing_allowed_scene_evidence(
    scene_840920_host_layout_fixture: tuple[
        Path, Path, dict[str, Path], dict[str, Path]
    ],
    tmp_path: Path,
) -> None:
    evidence, launches, _roots, billings = scene_840920_host_layout_fixture
    target = billings["adp-retained-scene-gpu-render"]
    noise = tmp_path / "arbitrary-other-root" / "copied-official-reconciliation.json"
    noise.parent.mkdir()
    shutil.copy2(target, noise)
    target.unlink()

    result = audit_scene_families(
        scene_id=SCENE,
        task_ids=TASKS,
        evidence_root=evidence,
        launch_state_root=launches,
    )

    assert result["strict_completed_family_count"] == 4
    retained = next(
        row for row in result["families"] if row["family_id"] == "retained_scene_render"
    )
    assert retained["status"] == "unproven"


def test_digest_mismatched_allowed_root_billing_does_not_count(
    scene_840920_host_layout_fixture: tuple[
        Path, Path, dict[str, Path], dict[str, Path]
    ],
) -> None:
    evidence, launches, _roots, billings = scene_840920_host_layout_fixture
    target = billings["adp-retained-scene-gpu-render"]
    value = json.loads(target.read_text())
    value["total_cost_usd"] = 0.01
    _write(target, value)

    result = audit_scene_families(
        scene_id=SCENE,
        task_ids=TASKS,
        evidence_root=evidence,
        launch_state_root=launches,
    )

    assert result["strict_completed_family_count"] == 4
    assert any(
        "official_billing_entry_invalid" in row["blockers"]
        for row in result["rejected_candidate_launches"]
        if row["probe_kind"] == "adp-retained-scene-gpu-render"
    )


def test_completed_wrong_scene_and_task_launch_is_never_admitted(
    scene_840920_host_layout_fixture: tuple[
        Path, Path, dict[str, Path], dict[str, Path]
    ],
) -> None:
    evidence, launches, _roots, _billings = scene_840920_host_layout_fixture
    wrong_input = evidence.parent / "gaussian-excision-scene840313-task-c"
    _make_launch(
        launches,
        probe_kind="adp-gaussian-excision",
        task_marker="task-c",
        scene_id="840313",
        input_root=wrong_input,
    )

    result = audit_scene_families(
        scene_id=SCENE,
        task_ids=TASKS,
        evidence_root=evidence,
        launch_state_root=launches,
    )

    assert result["strict_completed_family_count"] == 5
    assert all("task-c" not in row["qualified_launch_ids"] for row in result["families"])


def test_missing_official_billing_refuses_one_task_split_family(
    scene_840920_five_family_fixture: tuple[Path, Path, dict[str, Path]],
) -> None:
    evidence, launches, roots = scene_840920_five_family_fixture
    (roots["adp-usd-content-agents:task-b"] / "official_billing" / "same_goal_reconciliation.json").unlink()

    result = audit_scene_families(
        scene_id=SCENE,
        task_ids=TASKS,
        evidence_root=evidence,
        launch_state_root=launches,
    )

    assert result["strict_completed_family_count"] == 4
    rejected = next(
        row
        for row in result["rejected_candidate_launches"]
        if row["probe_kind"] == "adp-usd-content-agents"
        and "task-b" in row["launch_root"]
    )
    assert "official_billing_reconciliation_missing" in rejected["blockers"]


def test_wrong_scene_profile_cannot_close_scene_840920(tmp_path: Path) -> None:
    launches = tmp_path / "launches"
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    run = _make_launch(
        launches,
        probe_kind="adp-retained-scene-gpu-render",
        scene_id="840313",
    )
    # Keep the operator-facing launch identity 840920-shaped: the profile and
    # immutable request remain bound to 840313 and must win over the filename.
    renamed = launches / "scene840920-retained-wrong-input"
    run.rename(renamed)

    result = audit_scene_families(
        scene_id=SCENE,
        task_ids=TASKS,
        evidence_root=evidence,
        launch_state_root=launches,
    )

    retained = next(row for row in result["families"] if row["family_id"] == "retained_scene_render")
    assert retained["status"] == "unproven"
    assert result["strict_completed_family_count"] == 0


def test_self_valid_but_wrong_webapp_digest_is_rejected(
    scene_840920_five_family_fixture: tuple[Path, Path, dict[str, Path]],
) -> None:
    evidence, launches, roots = scene_840920_five_family_fixture
    run = roots["adp-retained-scene-gpu-render"]
    sync_path = run / "webapp_sync_succeeded.json"
    sync = json.loads(sync_path.read_text())
    sync["response"]["request_digest"] = "sha256:" + "9" * 64
    sync["sync_result_digest"] = canonical_digest(sync, digest_field="sync_result_digest")
    _write(sync_path, sync)

    result = audit_scene_families(
        scene_id=SCENE,
        task_ids=TASKS,
        evidence_root=evidence,
        launch_state_root=launches,
    )

    assert result["strict_completed_family_count"] == 4
    rejected = next(
        row
        for row in result["rejected_candidate_launches"]
        if row["probe_kind"] == "adp-retained-scene-gpu-render"
    )
    assert rejected["blockers"] == ["webapp_terminal_binding_invalid"]


def test_checkpoint_ledger_is_append_only_self_digested_and_chained(
    scene_840920_five_family_fixture: tuple[Path, Path, dict[str, Path]],
    tmp_path: Path,
) -> None:
    evidence, launches, _roots = scene_840920_five_family_fixture
    ledger = tmp_path / "ledger"
    first = audit_scene_families(
        scene_id=SCENE,
        task_ids=TASKS,
        evidence_root=evidence,
        launch_state_root=launches,
        ledger_root=ledger,
        observed_at="2026-08-17T02:00:00+00:00",
    )
    second = audit_scene_families(
        scene_id=SCENE,
        task_ids=TASKS,
        evidence_root=evidence,
        launch_state_root=launches,
        ledger_root=ledger,
        observed_at="2026-08-17T02:01:00+00:00",
    )

    assert first["sequence"] == 1
    assert second["sequence"] == 2
    assert second["previous_checkpoint_digest"] == first["checkpoint_digest"]
    first_path = sorted((ledger / SCENE).glob("*.json"))[0]
    tampered = json.loads(first_path.read_text())
    tampered["strict_completed_family_count"] = 15
    first_path.chmod(0o640)
    _write(first_path, tampered)
    with pytest.raises(SceneStrictFamilyError, match="checkpoint_chain_invalid"):
        audit_scene_families(
            scene_id=SCENE,
            task_ids=TASKS,
            evidence_root=evidence,
            launch_state_root=launches,
            ledger_root=ledger,
        )
