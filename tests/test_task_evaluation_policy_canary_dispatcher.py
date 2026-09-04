from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from blueprint_pipeline.decision_evidence_contracts import (
    canonical_digest,
    cross_runtime_canonical_digest,
)
from blueprint_pipeline.task_evaluation_policy_canary_dispatcher import (
    _join_session_closeout,
    _partial_policy_canary_result,
    _projection,
    _recovered_complete_policy_canary_result,
    _resume_materialized_policy_canary_delivery,
    TaskEvaluationPolicyCanaryDispatchError,
    dispatch_policy_canary_activation,
    process_policy_canary_dispatch_queue,
)
from tests.test_task_evaluation_policy_canary_setup import _setup as public_setup


COMMIT = "a" * 40


def test_join_uses_terminal_watchdog_as_provider_allocation_lineage() -> None:
    inner = {
        "episodes": [
            {"status": "blocked"}
            for _ in range(20)
        ],
        "blockers": ["episode_gap"],
    }
    adapter = {
        "continuing_spend_from_this_run": False,
        "independent_watchdog": {
            "status": "provider_terminal",
            "instance_ids": [49_609_705],
            "provider_absence_confirmed": True,
        },
        "provider_closeout": {
            "provider_zero_confirmed": True,
            "warm_session_retained": False,
            "all_staged_objects_absent": True,
        },
    }
    zero = {
        "schema_version": "task_evaluation_policy_canary_vast_provider_zero.v1",
        "provider_zero_verified": True,
        "live_instance_count": 0,
        "blockers": [],
    }

    joined = _join_session_closeout(
        inner=inner,
        adapter=adapter,
        provider_zero=zero,
    )

    assert joined["provider_allocations_observed"] == 1
    assert "policy_canary_provider_allocation_count_invalid" not in joined["blockers"]


def test_join_lifts_exact_mixed_episode_failure_taxonomy_to_run_blockers() -> None:
    episodes = [{"status": "completed"} for _ in range(12)]
    episodes.extend(
        {
            "status": "blocked",
            "typed_harness_failure": "TaskNeutralScoringError",
            "episode": {"score": {"blockers": ["policy_outcome_uninterpretable"]}},
        }
        for _ in range(7)
    )
    episodes.append(
        {
            "status": "blocked",
            "typed_harness_failure": "DroidActionExecutionError",
            "episode": {"score": {"blockers": ["policy_outcome_uninterpretable"]}},
        }
    )
    joined = _join_session_closeout(
        inner={"episodes": episodes, "blockers": []},
        adapter={
            "vast_instance_ids": [49_629_253],
            "continuing_spend_from_this_run": False,
            "provider_closeout": {
                "provider_zero_confirmed": True,
                "warm_session_retained": False,
                "all_staged_objects_absent": True,
            },
        },
        provider_zero={
            "schema_version": "task_evaluation_policy_canary_vast_provider_zero.v1",
            "provider_zero_verified": True,
            "live_instance_count": 0,
            "blockers": [],
        },
    )

    assert joined["status"] == "blocked"
    assert joined["blockers"] == [
        "policy_canary_episode_failure:DroidActionExecutionError",
        "policy_canary_episode_failure:TaskNeutralScoringError",
    ]


def _public_artifact(character: str, artifact_id: str) -> dict[str, object]:
    return {
        "artifact_id": artifact_id,
        "digest": "sha256:" + character * 64,
        "size_bytes": 10,
    }


def test_projection_derives_reset_identity_only_for_blocked_legacy_episodes() -> None:
    contract = public_setup()["task_success_contract"]
    episodes = []
    for candidate_id in ("pi05_droid", "groot_n17_droid"):
        for index in range(10):
            episodes.append(
                {
                    "candidate_id": candidate_id,
                    "cell_id": f"quick-cell-{index}",
                    "seed": 3100 + index,
                    "resolved_scenario": {
                        "family": "canonical_anchor",
                        "ordinal": index,
                    },
                    "status": "blocked",
                    "candidate_policy_queried": False,
                    "actions_reached_robot": False,
                    "arm_moved": False,
                    "policy_outcome_interpretable": False,
                    "typed_harness_failure": "RuntimeError",
                    "checkpoint_digest": "sha256:" + "a" * 64,
                    "runtime_identity_digest": "sha256:" + "b" * 64,
                    "visual_evidence": {
                        "media_gap": {
                            "type": "before_first_observation",
                            "reason": "policy_canary_episode_runner_failed",
                        }
                    },
                }
            )
    report = {
        "run_id": "scene-839873-canary-legacy",
        "result_status": "blocked",
        "delivery_digest": "sha256:" + "c" * 64,
        "report": {
            "machine_readable_report": _public_artifact("d", "report"),
            "evidence_manifest": _public_artifact("e", "manifest"),
        },
        "closure": {
            "billing": _public_artifact("f", "billing"),
            "teardown": _public_artifact("1", "teardown"),
            "provider_zero": _public_artifact("2", "provider-zero"),
        },
        "candidate_results": [],
        "artifacts": [],
        "task_success_contract": contract,
        "task_success_contract_digest": contract["contract_digest"],
    }
    result = {
        "run_id": report["run_id"],
        "configuration_digest": "sha256:" + "3" * 64,
        "result_digest": "sha256:" + "4" * 64,
        "status": "blocked",
        "episodes": episodes,
        "blockers": ["policy_canary_episode_runner_failed"],
        "task_success_contract": contract,
        "task_success_contract_digest": contract["contract_digest"],
    }

    projected = _projection(
        setup={
            "scene_id": "839873",
            "request_digest": "sha256:" + "5" * 64,
            "task_success_contract": contract,
            "task_success_contract_digest": contract["contract_digest"],
        },
        result=result,
        delivery=report,
    )

    expected = canonical_digest(
        {
            "resolved_scenario": episodes[0]["resolved_scenario"],
            "seed": episodes[0]["seed"],
            "execution_performed": False,
        }
    )
    assert projected["episodes"][0]["evidence"]["reset_state_digest"] == expected
    assert projected["counts"]["completed_learned_policy_rollout_count"] == 0


def test_materialized_delivery_resume_retries_only_website_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = public_setup()["task_success_contract"]
    root = tmp_path / "dispatch"
    episodes = []
    for candidate_id in ("pi05_droid", "groot_n17_droid"):
        for index in range(10):
            episodes.append(
                {
                    "candidate_id": candidate_id,
                    "cell_id": f"quick-cell-{index}",
                    "seed": 3100 + index,
                    "resolved_scenario": {"family": "canonical_anchor", "ordinal": index},
                    "status": "blocked",
                    "candidate_policy_queried": False,
                    "actions_reached_robot": False,
                    "arm_moved": False,
                    "policy_outcome_interpretable": False,
                    "typed_harness_failure": "RuntimeError",
                    "checkpoint_digest": "sha256:" + "a" * 64,
                    "runtime_identity_digest": "sha256:" + "b" * 64,
                    "reset_state_digest": "sha256:" + "c" * 64,
                    "visual_evidence": {
                        "media_gap": {
                            "type": "before_first_observation",
                            "reason": "policy_canary_episode_runner_failed",
                        }
                    },
                }
            )
    joined = {
        "run_id": "scene-839873-canary-resume",
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "status": "blocked",
        "configuration_digest": "sha256:" + "d" * 64,
        "episodes": episodes,
        "blockers": ["policy_canary_episode_runner_failed"],
        "task_success_contract": contract,
        "task_success_contract_digest": contract["contract_digest"],
        "result_digest": "",
    }
    joined["result_digest"] = canonical_digest(joined, digest_field="result_digest")
    joined_path = _write(root / "policy_canary_terminal_result.json", joined)
    report_path = _write(
        root / "artifacts/result_delivery/policy_canary_full_report.json", joined
    )
    billing_path = _write(root / "official_billing_reconciliation.json", {"status": "ok"})
    teardown_path = _write(root / "teardown.json", {"status": "completed"})
    provider_zero_path = _write(
        root / "post_teardown_global_provider_zero.json",
        {"provider_zero_verified": True},
    )

    def descriptor(path: Path, artifact_id: str) -> dict[str, object]:
        return {
            "artifact_id": artifact_id,
            "digest": _sha(path),
            "size_bytes": path.stat().st_size,
        }

    delivery = {
        "schema_version": "task_evaluation_result_delivery.v2",
        "run_id": joined["run_id"],
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "result_status": "blocked",
        "status": "ready",
        "blockers": list(joined["blockers"]),
        "report": {
            "machine_readable_report": descriptor(report_path, "report"),
            "evidence_manifest": _public_artifact("e", "manifest"),
        },
        "closure": {
            "billing": descriptor(billing_path, "billing"),
            "teardown": descriptor(teardown_path, "teardown"),
            "provider_zero": {
                **descriptor(provider_zero_path, "provider-zero"),
                "provider_zero_verified": True,
            },
        },
        "candidate_results": [],
        "artifacts": [],
        "task_success_contract": contract,
        "task_success_contract_digest": contract["contract_digest"],
        "delivery_digest": "",
    }
    delivery["delivery_digest"] = cross_runtime_canonical_digest(
        delivery, digest_field="delivery_digest"
    )
    _write(root / "artifacts/result_delivery/delivery.json", delivery)
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.validate_vast_official_same_goal_reconciliation",
        lambda _path: {},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.materialize_policy_canary_website_delivery",
        lambda *, run_root, delivery: dict(delivery),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.policy_canary_episode_interpretation_closeout.materialize_policy_canary_episode_interpretations",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("interpreter unavailable")),
    )
    sync_calls = []

    receipt = _resume_materialized_policy_canary_delivery(
        root=root,
        setup={
            "scene_id": "839873",
            "capture_session_id": "capture-839873",
            "intake_id": "intake-839873",
            "request_digest": "sha256:" + "f" * 64,
            "task_success_contract": contract,
            "task_success_contract_digest": contract["contract_digest"],
        },
        runtime_inputs={
            "configuration_digest": joined["configuration_digest"],
            "task_success_contract": contract,
            "task_success_contract_digest": contract["contract_digest"],
        },
        authority={"authority_digest": "sha256:" + "1" * 64},
        bundle={"bundle_sha256": "sha256:" + "2" * 64},
        adapter={"teardown_manifest_path": str(teardown_path)},
        sync_runner=lambda **kwargs: (
            sync_calls.append(kwargs)
            or {
                "status": "succeeded",
                "notification_delivery": {
                    "status": "failed",
                    "attempts": 1,
                    "provider": "email_transport_unavailable",
                    "message_id": None,
                    "delivered_at": None,
                    "run_result_digest": joined["result_digest"],
                },
            }
        ),
    )

    assert receipt is not None
    assert receipt["status"] == "blocked"
    assert receipt["allocator_invoked"] is False
    assert len(sync_calls) == 1
    assert (root / "dispatch_receipt.json").is_file()
    assert json.loads(joined_path.read_text(encoding="utf-8")) == joined


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_partial_provider_result_preserves_completed_cell_and_types_remaining_gaps(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "immutable_execution"
    native_path = _write(
        evidence_root / "native_task_arena_policy_canary_session_result.v1.json",
        {
            "schema_version": "native_task_arena_policy_canary_session_result.v1",
            "status": "blocked",
            "blockers": ["policy_canary_worker_failed_without_result"],
        },
    )
    cells = [
        {
            "cell_id": f"cell-{index}",
            "seed": 3100 + index,
            "resolved_scenario": {"ordinal": index},
        }
        for index in range(10)
    ]
    episodes = [
        {
            "candidate_id": candidate,
            "cell_id": "cell-0",
            "seed": 3100,
            "status": "completed",
            "candidate_policy_queried": True,
            "actions_reached_robot": True,
            "arm_moved": True,
            "policy_outcome_interpretable": False,
            "evidence_artifacts": {
                "review_video": {
                    "relative_path": f"episodes/{candidate}.mp4",
                    "sha256": "sha256:" + "1" * 64,
                    "size_bytes": 10,
                }
            },
        }
        for candidate in ("pi05_droid", "groot_n17_droid")
    ]
    child = {
        "schema_version": "native_task_arena_policy_canary_session_result.v1",
        "status": "runtime_selected_cell_completed_pending_aggregation",
        "selected_cell_index": 0,
        "episodes": episodes,
        "artifact_inventory": [
            {
                "role": "review_video",
                "relative_path": "episodes/pi05.mp4",
                "sha256": "sha256:" + "2" * 64,
                "size_bytes": 10,
            },
            {
                "role": "runtime_supporting_evidence",
                "relative_path": "worker_console.log",
                "sha256": "sha256:" + "3" * 64,
                "size_bytes": 100,
            },
        ],
        "result_digest": "",
    }
    child["result_digest"] = canonical_digest(child, digest_field="result_digest")
    _write(
        evidence_root
        / "cell_runs/00/native_task_arena_policy_canary_session_result.v1.json",
        child,
    )

    partial = _partial_policy_canary_result(
        native_path=native_path,
        fallback=json.loads(native_path.read_text(encoding="utf-8")),
        runtime_inputs={
            "cells": cells,
            "task_success_contract": public_setup()["task_success_contract"],
            "task_success_contract_digest": public_setup()[
                "task_success_contract_digest"
            ],
        },
        specs={
            candidate: {
                "checkpoint_digest": "sha256:" + character * 64,
                "runtime_identity_digest": "sha256:" + character.upper() * 64,
            }
            for candidate, character in (
                ("pi05_droid", "a"),
                ("groot_n17_droid", "b"),
            )
        },
    )

    assert partial is not None
    result, result_path = partial
    assert result_path.is_file()
    assert result["status"] == "blocked"
    assert result["candidate_policy_queried"] is True
    assert result["completed_cell_count"] == 1
    assert result["incomplete_cell_count"] == 9
    assert len(result["episodes"]) == 20
    preserved = [row for row in result["episodes"] if row["cell_id"] == "cell-0"]
    missing = [row for row in result["episodes"] if row["cell_id"] != "cell-0"]
    assert len(preserved) == 2 and len(missing) == 18
    assert all(row["candidate_policy_queried"] is True for row in preserved)
    assert all(
        row["typed_harness_failure"]
        == "cell_not_completed_before_terminal_failure"
        for row in missing
    )
    assert preserved[0]["evidence_artifacts"]["review_video"][
        "relative_path"
    ].startswith("cell_runs/00/")
    assert not any(
        row["relative_path"].endswith("worker_console.log")
        for row in result["artifact_inventory"]
    )


def test_complete_provider_result_is_not_rebuilt_from_child_receipts(
    tmp_path: Path,
) -> None:
    native_path = tmp_path / "native_task_arena_policy_canary_session_result.v1.json"
    complete = {
        "status": "runtime_completed_unqualified_pending_closeout",
        "episodes": [{} for _ in range(20)],
        "artifact_inventory": [],
    }
    _write(native_path, complete)

    assert (
        _partial_policy_canary_result(
            native_path=native_path,
            fallback=complete,
            runtime_inputs={"cells": []},
            specs={},
        )
        is None
    )


def test_complete_pinned_ssh_recovery_adopts_all_ten_cells_without_provider_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "dispatch"
    attempt = tmp_path / "attempt"
    evidence = attempt / "immutable_execution"
    archive = attempt / "vast_provider_run" / "provider-output.zip"
    children = []
    for index in range(10):
        child = {
            "selected_cell_index": index,
            "status": "runtime_selected_cell_completed_pending_aggregation",
            "construction_lineage_mode": "compiled_configured_scene_diagnostic",
            "episodes": [],
            "result_digest": "",
        }
        child["result_digest"] = canonical_digest(child, digest_field="result_digest")
        path = _write(
            evidence
            / f"cell_runs/{index:02d}/native_task_arena_policy_canary_session_result.v1.json",
            child,
        )
        children.append((path, child))
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as output:
        for index, (path, _child) in enumerate(children):
            output.write(
                path,
                f"cell_runs/{index:02d}/native_task_arena_policy_canary_session_result.v1.json",
            )
        for index in range(120):
            output.writestr(f"cell_runs/media/{index:03d}.mp4", b"video")
    command = {
        "provider_bundle_kind": "native_task_arena_policy_canary_session",
        "provider_runtime_output_zip_received": True,
        "provider_runtime_output_zip_path": str(archive),
        "provider_output_download_manifest": {
            "ssh_recovery": {
                "status": "completed",
                "strict_host_key_checking": True,
                "streamed_to_disk": True,
                "recovered_size_bytes": archive.stat().st_size,
                "recovered_sha256": _sha(archive),
                "known_hosts_sha256": "a" * 64,
            }
        },
        "provider_runtime_output_zip_inspection": {
            "zip_present": True,
            "mp4_count": 120,
        },
    }
    _write(attempt / "vast_provider_run/vast_provider_command_result.json", command)
    aggregate_calls = []

    def aggregate(**kwargs):
        aggregate_calls.append(kwargs)
        result = {
            "status": "runtime_completed_unqualified_pending_closeout",
            "episodes": [{"status": "completed"} for _ in range(20)],
            "result_digest": "",
        }
        result["result_digest"] = canonical_digest(result, digest_field="result_digest")
        return result

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher._aggregate_isolated_cell_results",
        aggregate,
    )
    native_path = evidence / "native_task_arena_policy_canary_session_result.v1.json"
    recovered = _recovered_complete_policy_canary_result(
        root=root,
        native_path=native_path,
        adapter={"attempt_root": str(attempt)},
        authority={"run_id": "scene-839873-recovered"},
        runtime_inputs={"cells": [{"cell_id": f"cell-{i}"} for i in range(10)]},
    )

    assert recovered is not None
    result, result_path = recovered
    assert len(result["episodes"]) == 20
    assert result_path.is_file()
    assert len(aggregate_calls) == 1
    receipt = json.loads(
        (root / "recovered_provider_output_adoption.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "adopted_complete_provider_output"
    assert receipt["episode_count"] == 20
    assert receipt["mp4_count"] == 120
    assert receipt["provider_mutation_performed"] is False
    assert receipt["automatic_retry_performed"] is False


def test_partial_provider_result_preserves_prepolicy_blocked_cell_evidence(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "immutable_execution"
    native_path = _write(
        evidence_root / "native_task_arena_policy_canary_session_result.v1.json",
        {
            "schema_version": "native_task_arena_policy_canary_session_result.v1",
            "status": "blocked",
            "blockers": ["policy_canary_worker_failed_without_result"],
        },
    )
    cells = [
        {
            "cell_id": f"cell-{index}",
            "seed": 3100 + index,
            "resolved_scenario": {"ordinal": index},
        }
        for index in range(10)
    ]
    child = {
        "schema_version": "native_task_arena_policy_canary_session_result.v1",
        "status": "blocked",
        "selected_cell_index": 0,
        "episodes": [],
        "candidate_policy_queried": False,
        "blockers": ["policy_canary_task_semantic_visibility_failed"],
        "artifact_inventory": [
            {
                "role": "runtime_supporting_evidence",
                "relative_path": "prepolicy_observation_gate/external.png",
                "sha256": "sha256:" + "4" * 64,
                "size_bytes": 123,
            }
        ],
        "result_digest": "",
    }
    child["result_digest"] = canonical_digest(child, digest_field="result_digest")
    _write(
        evidence_root
        / "cell_runs/00/native_task_arena_policy_canary_session_result.v1.json",
        child,
    )

    partial = _partial_policy_canary_result(
        native_path=native_path,
        fallback=json.loads(native_path.read_text(encoding="utf-8")),
        runtime_inputs={
            "cells": cells,
            "task_success_contract": public_setup()["task_success_contract"],
            "task_success_contract_digest": public_setup()[
                "task_success_contract_digest"
            ],
        },
        specs={
            candidate: {
                "checkpoint_digest": "sha256:" + character * 64,
                "runtime_identity_digest": "sha256:" + character.upper() * 64,
            }
            for candidate, character in (
                ("pi05_droid", "a"),
                ("groot_n17_droid", "b"),
            )
        },
    )

    assert partial is not None
    result, _ = partial
    assert result["candidate_policy_queried"] is False
    assert result["completed_cell_count"] == 0
    assert result["incomplete_cell_count"] == 10
    assert len(result["episodes"]) == 20
    assert result["artifact_inventory"][0]["relative_path"] == (
        "cell_runs/00/prepolicy_observation_gate/external.png"
    )


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    public = public_setup()
    task_success_contract = public["task_success_contract"]
    task_success_contract_digest = public["task_success_contract_digest"]
    run_id = "scene-839873-canary-1"
    units = [
        {
            "campaign_unit_id": f"unit-{index}",
            "cell_id": f"quick-cell-{index}",
            "seed": 3100 + index,
            "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        }
        for index in range(10)
    ]
    activation: dict[str, object] = {
        "schema_version": "task_evaluation_policy_campaign_activation.v1",
        "run_id": run_id,
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "task_success_contract": task_success_contract,
        "task_success_contract_digest": task_success_contract_digest,
        "campaign_unit_count": 10,
        "campaign_units": units,
        "activation_digest": "",
    }
    activation["activation_digest"] = canonical_digest(
        activation, digest_field="activation_digest"
    )
    activation_root = tmp_path / "activation"
    activation_path = _write(
        activation_root / "task_evaluation_policy_campaign_activation.v1.json",
        activation,
    )
    packet = _write(
        tmp_path / "packet" / "native_task_arena_packet_receipt.v1.json", {}
    )
    runtime_source = _write(tmp_path / "runtime-source.json", {})
    construction = _write(tmp_path / "construction.json", {})
    cells = []
    for index in range(10):
        scenario = {"family": "canonical", "ordinal": index}
        cells.append(
            {
                "cell_id": f"quick-cell-{index}",
                "seed": 3100 + index,
                "family": (
                    "canonical_anchor" if index < 2 else "placement_approach"
                ),
                "cell_spec_digest": "sha256:" + f"{index:064x}",
                "resolved_scenario": scenario,
                "resolved_scenario_digest": canonical_digest(scenario),
                "control_diagnostic": {
                    "mode": "nonblocking_diagnostic_pending",
                    "typed_gap": "controls_pending_at_submission",
                    "policy_execution_blocked": False,
                },
            }
        )
    runtime: dict[str, object] = {
        "schema_version": "task_evaluation_policy_canary_runtime_inputs.v1",
        "run_id": run_id,
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "scene_revision_digest": "sha256:" + "9" * 64,
        "matrix_digest": "sha256:" + "8" * 64,
        "configuration_digest": "sha256:" + "1" * 64,
        "plan_digest": "sha256:" + "2" * 64,
        "activation_digest": activation["activation_digest"],
        "base_native_packet": _record(packet),
        "runtime_source": _record(runtime_source),
        "construction_result": _record(construction),
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "task_success_contract": task_success_contract,
        "task_success_contract_digest": task_success_contract_digest,
        "cells": cells,
        "execution_authority": {
            "maximum_provider_allocations": 1,
            "retry_cap": 0,
            "single_warm_provider_session_required": True,
            "caller_surviving_watchdog_required": True,
            "billing_teardown_provider_zero_required": True,
        },
        "resource_authority": {
            "resource_name": "blueprint-native-task-policy-canary-0123456789abcdef",
            "maximum_hourly_rate_usd": 0.8,
            "hard_cap_usd": 4.0,
            "hard_ttl_seconds": 14_400,
            "user_confirmed": True,
        },
        "runtime_inputs_digest": "",
    }
    runtime["runtime_inputs_digest"] = canonical_digest(
        runtime, digest_field="runtime_inputs_digest"
    )
    runtime_path = _write(
        activation_root / "task_evaluation_policy_canary_runtime_inputs.v1.json",
        runtime,
    )
    activation_result: dict[str, object] = {
        "schema_version": "task_evaluation_launch_activation_result.v1",
        "status": "policy_campaign_queue_materialized_no_execution",
        "activation_id": "activation-1",
        "source_commit": COMMIT,
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "policy_canary_runtime_inputs_path": str(runtime_path),
        "task_success_contract": task_success_contract,
        "task_success_contract_digest": task_success_contract_digest,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "result_digest": "",
    }
    activation_result["result_digest"] = canonical_digest(
        activation_result, digest_field="result_digest"
    )
    activation_result_path = _write(tmp_path / "activation-result.json", activation_result)
    records = {}
    for name in (
        "pi05_execution_spec",
        "groot_execution_spec",
        "pi05_checkpoint_inventory",
    ):
        records[name] = _record(
            _write(
                tmp_path / f"{name}.json",
                {
                    "name": name,
                    **(
                        {
                            "task_success_contract": task_success_contract,
                            "task_success_contract_digest": (
                                task_success_contract_digest
                            ),
                        }
                        if name.endswith("execution_spec")
                        else {}
                    ),
                },
            )
        )
    setup: dict[str, object] = {
        "schema_version": "task_evaluation_policy_canary_execution_setup.v1",
        "status": "verified_runnable",
        "scene_id": "839873",
        "configured_source_launch_id": "configured-scene-839873-r4",
        "scene_revision_digest": runtime["scene_revision_digest"],
        "activation_digest": activation["activation_digest"],
        "source_commit": COMMIT,
        "provider": "vast",
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "capture_session_id": "capture-839873",
        "intake_id": "intake-839873",
        "request_digest": "sha256:" + "7" * 64,
        "task_success_contract": task_success_contract,
        "task_success_contract_digest": task_success_contract_digest,
        "records": records,
        "setup_digest": "",
    }
    setup["setup_digest"] = canonical_digest(setup, digest_field="setup_digest")
    setup_path = _write(tmp_path / "execution-setup.json", setup)
    return activation_result_path, setup_path, activation_path


def test_dispatcher_materializes_one_authority_bundle_and_allocator_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    activation_result, setup_path, _activation = _inputs(tmp_path)
    output = tmp_path / "dispatch"
    observed: dict[str, object] = {}
    progress_updates = []

    def fake_bundle(**kwargs):
        observed["bundle"] = kwargs
        job = Path(kwargs["job_dir"])
        job.mkdir(parents=True, exist_ok=True)
        receipt = {
            "bundle_sha256": "sha256:" + "b" * 64,
            "bundle_path": str(job / "bundle.zip"),
        }
        _write(
            job / "native_task_arena_policy_canary_session_bundle_receipt.v1.json",
            receipt,
        )
        return receipt

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.build_policy_canary_session_bundle",
        fake_bundle,
    )

    def fake_allocator(argv):
        observed["argv"] = list(argv)
        adapter = Path(argv[argv.index("--adapter-output") + 1])
        _write(adapter, {"status": "dry_run_ready", "provider_mutations_performed": 0})
        return 0

    receipt = dispatch_policy_canary_activation(
        activation_result_path=activation_result,
        execution_setup_path=setup_path,
        output_root=output,
        implementation_commit=COMMIT,
        allocator_runner=fake_allocator,
        progress_sync_runner=lambda **kwargs: progress_updates.append(kwargs["progress"])
        or {"status": "succeeded"},
    )

    argv = observed["argv"]
    assert receipt["status"] == "prepared_no_execution"
    assert argv.count("native-task-arena-policy-canary-session") == 1
    assert "--execute" not in argv
    assert argv[0] == "gpu-canary"
    assert receipt["retry_cap"] == 0
    assert receipt["provider_mutation_performed"] is False
    assert Path(observed["bundle"]["session_authority_path"]).is_file()
    assert [update["phase"] for update in progress_updates] == [
        "queued",
        "preparing",
        "preparing",
        "provider_allocating",
    ]
    assert progress_updates[-1]["phase_status"] == "running"
    assert (output / "status_progress_sync.jsonl").is_file()


def test_dispatcher_refuses_absent_scene839873_setup_before_allocator(
    tmp_path: Path,
) -> None:
    activation_result, _setup_path, _activation = _inputs(tmp_path)
    with pytest.raises(
        TaskEvaluationPolicyCanaryDispatchError,
        match="policy_canary_scene839873_setup_receipt_missing",
    ):
        dispatch_policy_canary_activation(
            activation_result_path=activation_result,
            execution_setup_path=tmp_path / "missing.json",
            output_root=tmp_path / "dispatch",
            implementation_commit=COMMIT,
            allocator_runner=lambda _argv: pytest.fail("allocator must not run"),
        )


def test_allocator_invocation_marker_prevents_unrecorded_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation_result, setup_path, _activation = _inputs(tmp_path)
    output = tmp_path / "dispatch-crashed-allocator"

    def fake_bundle(**kwargs):
        job = Path(kwargs["job_dir"])
        job.mkdir(parents=True, exist_ok=True)
        receipt = {"bundle_sha256": "sha256:" + "b" * 64}
        _write(
            job / "native_task_arena_policy_canary_session_bundle_receipt.v1.json",
            receipt,
        )
        return receipt

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.build_policy_canary_session_bundle",
        fake_bundle,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.validate_provider_bundle",
        lambda value, **_kwargs: value,
    )

    with pytest.raises(
        TaskEvaluationPolicyCanaryDispatchError,
        match="policy_canary_allocator_exit_17_without_result",
    ):
        dispatch_policy_canary_activation(
            activation_result_path=activation_result,
            execution_setup_path=setup_path,
            output_root=output,
            implementation_commit=COMMIT,
            execute=True,
            allocator_runner=lambda _argv: 17,
        )

    started = json.loads(
        (output / "allocator_invocation_started.json").read_text(encoding="utf-8")
    )
    finished = json.loads(
        (output / "allocator_invocation_finished.json").read_text(encoding="utf-8")
    )
    assert started["allocator_invoked"] is True
    assert started["automatic_retry_authorized"] is False
    assert finished["exit_code"] == 17
    assert finished["adapter_result_present"] is False
    with pytest.raises(
        TaskEvaluationPolicyCanaryDispatchError,
        match="policy_canary_allocator_previous_invocation_without_result",
    ):
        dispatch_policy_canary_activation(
            activation_result_path=activation_result,
            execution_setup_path=setup_path,
            output_root=output,
            implementation_commit=COMMIT,
            execute=True,
            allocator_runner=lambda _argv: pytest.fail("allocator invoked twice"),
        )


def test_live_shaped_result_waits_for_billing_and_never_launches_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    activation_result, setup_path, _activation = _inputs(tmp_path)
    output = tmp_path / "dispatch-live"
    calls = {"allocator": 0, "bundle": 0}

    def fake_bundle(**kwargs):
        calls["bundle"] += 1
        job = Path(kwargs["job_dir"])
        job.mkdir(parents=True, exist_ok=True)
        receipt = {"bundle_sha256": "sha256:" + "b" * 64}
        _write(
            job / "native_task_arena_policy_canary_session_bundle_receipt.v1.json",
            receipt,
        )
        return receipt

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.build_policy_canary_session_bundle",
        fake_bundle,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.validate_provider_bundle",
        lambda value, **_kwargs: value,
    )

    def fake_allocator(argv):
        calls["allocator"] += 1
        adapter_path = Path(argv[argv.index("--adapter-output") + 1])
        attempt = output / "allocator" / "attempts" / "attempt-1"
        evidence = attempt / "immutable_execution"
        evidence.mkdir(parents=True, exist_ok=True)
        inner = {
            "schema_version": "native_task_arena_policy_canary_session_result.v1",
            "status": "runtime_completed_unqualified_pending_closeout",
            "run_kind": "internal_policy_canary",
            "claim_ceiling": "diagnostic_policy_execution",
            "episodes": [
                {
                    "candidate_id": candidate,
                    "cell_id": f"quick-cell-{index}",
                    "seed": 3100 + index,
                    "status": "completed",
                }
                for candidate in ("pi05_droid", "groot_n17_droid")
                for index in range(10)
            ],
            "artifact_inventory": [],
            "result_digest": "",
        }
        inner["result_digest"] = canonical_digest(
            inner, digest_field="result_digest"
        )
        inner_path = _write(
            evidence / "native_task_arena_policy_canary_session_result.v1.json",
            inner,
        )
        teardown = _write(attempt / "vast_teardown_manifest.json", {"status": "completed"})
        _write(
            adapter_path,
            {
                "schema_version": "native_task_arena_policy_canary_session_result.v1",
                "status": "completed",
                "vast_instance_ids": [49247792],
                "native_control_result_path": str(inner_path),
                "teardown_manifest_path": str(teardown),
                "continuing_spend_from_this_run": False,
                "all_staged_objects_absent": True,
                "provider_closeout": {
                    "provider_zero_confirmed": True,
                    "warm_session_retained": False,
                    "all_staged_objects_absent": True,
                },
            },
        )
        return 0

    zero = {
        "schema_version": "task_evaluation_policy_canary_vast_provider_zero.v1",
        "status": "provider_zero_confirmed",
        "api_confirmed": True,
        "provider_zero_verified": True,
        "live_instance_count": 0,
        "blockers": [],
        "receipt_digest": "",
    }
    zero["receipt_digest"] = canonical_digest(zero, digest_field="receipt_digest")
    zero_collections = 0

    def collect_zero():
        nonlocal zero_collections
        zero_collections += 1
        return zero
    progress_updates = []

    def sync_progress(**kwargs):
        progress_updates.append(kwargs["progress"])
        return {"status": "succeeded", "response": {"status": "recorded"}}

    first = dispatch_policy_canary_activation(
        activation_result_path=activation_result,
        execution_setup_path=setup_path,
        output_root=output,
        implementation_commit=COMMIT,
        execute=True,
        allocator_runner=fake_allocator,
        provider_zero_collector=collect_zero,
        progress_sync_runner=sync_progress,
    )
    second = dispatch_policy_canary_activation(
        activation_result_path=activation_result,
        execution_setup_path=setup_path,
        output_root=output,
        implementation_commit=COMMIT,
        execute=True,
        allocator_runner=lambda _argv: pytest.fail("allocator invoked twice"),
        provider_zero_collector=collect_zero,
        progress_sync_runner=sync_progress,
    )

    assert first["status"] == second["status"] == "awaiting_official_billing"
    assert first["allocator_invoked"] is True
    assert second["allocator_invoked"] is False
    assert first["website_progress_sync"]["status"] == "succeeded"
    assert progress_updates[-1]["phase"] == "awaiting_official_billing"
    assert "provider_allocating" in [update["phase"] for update in progress_updates]
    assert calls == {"allocator": 1, "bundle": 1}

    def post_billing(**kwargs):
        _write(Path(kwargs["output_path"]), {"status": "reconciled_official_posted_charges"})
        return True

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher._materialize_official_billing_if_posted",
        post_billing,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.validate_vast_official_same_goal_reconciliation",
        lambda _path: {},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.materialize_policy_canary_result_delivery",
        lambda **kwargs: {
            "run_id": kwargs["run_id"],
            "result_status": kwargs["result_status"],
            "delivery_digest": "sha256:" + "d" * 64,
            "report": {},
            "closure": {},
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher._projection",
        lambda **_kwargs: {"projection_digest": "sha256:" + "e" * 64},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.materialize_policy_canary_website_delivery",
        lambda *, run_root, delivery: dict(delivery),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.policy_canary_episode_interpretation_closeout.materialize_policy_canary_episode_interpretations",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("interpreter unavailable")),
    )
    third = dispatch_policy_canary_activation(
        activation_result_path=activation_result,
        execution_setup_path=setup_path,
        output_root=output,
        implementation_commit=COMMIT,
        execute=True,
        allocator_runner=lambda _argv: pytest.fail("allocator invoked on billing resume"),
        provider_zero_collector=collect_zero,
        sync_runner=lambda **_kwargs: {
            "status": "succeeded",
            "notification_delivery": {
                "status": "failed",
                "run_result_digest": "sha256:" + "e" * 64,
            },
        },
    )

    assert third["status"] == "completed_unqualified"
    assert third["allocator_invoked"] is False
    assert third["notification_delivery"]["status"] == "failed"
    assert (output / "dispatch_receipt.json").is_file()
    terminal = json.loads((output / "policy_canary_terminal_result.json").read_text())
    assert terminal["episode_interpretation"]["status"] == "abstained"
    assert terminal["episode_interpretation"]["closeout_error_type"] == "RuntimeError"
    assert terminal["session_closeout"]["provider_zero_confirmed"] is True
    assert calls == {"allocator": 1, "bundle": 1}
    assert zero_collections == 1


def test_invalid_envelope_is_quarantined_without_allocator(tmp_path: Path) -> None:
    queue = tmp_path / "queue"
    for name in ("pending", "processing", "completed", "blocked"):
        (queue / name).mkdir(parents=True, exist_ok=True)
    pending = _write(queue / "pending" / "000-invalid.json", {"secret": "redacted"})
    setups = tmp_path / "setups"
    setups.mkdir()

    observed = process_policy_canary_dispatch_queue(
        dispatch_queue_root=queue,
        execution_setup_root=setups,
        dispatch_root=tmp_path / "dispatches",
        implementation_commit=COMMIT,
        execute=True,
    )

    result = observed["results"][0]
    assert result["status"] == "blocked_invalid_envelope"
    assert result["allocator_invoked"] is False
    assert result["provider_mutation_performed"] is False
    assert "secret" not in json.dumps(result)
    assert not pending.exists()
    assert (queue / "blocked" / pending.name).is_file()


def test_proven_zero_allocation_terminalizes_without_billing_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation_result, setup_path, _activation = _inputs(tmp_path)
    output = tmp_path / "dispatch-no-allocation"

    def fake_bundle(**kwargs):
        job = Path(kwargs["job_dir"])
        job.mkdir(parents=True, exist_ok=True)
        receipt = {"bundle_sha256": "sha256:" + "b" * 64}
        _write(
            job / "native_task_arena_policy_canary_session_bundle_receipt.v1.json",
            receipt,
        )
        return receipt

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.build_policy_canary_session_bundle",
        fake_bundle,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.validate_provider_bundle",
        lambda value, **_kwargs: value,
    )

    def allocator_without_instance(argv):
        adapter_path = Path(argv[argv.index("--adapter-output") + 1])
        _write(
            adapter_path,
            {
                "status": "blocked",
                "vast_instance_ids": [],
                "provider_mutations_performed": 0,
                "provider_create_attempted": False,
                "vast_side_effects_may_have_occurred": False,
                "continuing_spend_from_this_run": False,
                "blockers": ["policy_canary_provider_capacity_unavailable"],
            },
        )
        return 2

    synced = []
    result = dispatch_policy_canary_activation(
        activation_result_path=activation_result,
        execution_setup_path=setup_path,
        output_root=output,
        implementation_commit=COMMIT,
        execute=True,
        allocator_runner=allocator_without_instance,
        blocked_sync_runner=lambda **kwargs: synced.append(kwargs)
        or {
            "status": "succeeded",
            "notification_delivery": {"status": "accepted"},
        },
    )

    assert result["status"] == "blocked_without_provider_allocation"
    assert result["provider_allocation_performed"] is False
    assert result["provider_mutation_performed"] is False
    assert result["terminal_sync"]["status"] == "succeeded"
    assert synced[0]["blockers"] == ["policy_canary_provider_capacity_unavailable"]
    assert not (output / "official_billing_reconciliation.json").exists()


def test_post_allocator_failure_is_not_labeled_preprovider_or_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation_result, setup_path, _activation = _inputs(tmp_path)
    queue = tmp_path / "queue"
    for name in ("pending", "processing", "completed", "blocked"):
        (queue / name).mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": "task_evaluation_policy_canary_dispatch_envelope.v1",
        "activation_id": "activation-1",
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "source_commit": COMMIT,
        "activation_result": _record(activation_result),
        "capture_session_id": "capture-839873",
        "intake_id": "intake-839873",
        "request_digest": "sha256:" + "7" * 64,
        "maximum_provider_allocations": 1,
        "retry_cap": 0,
        "automatic_retry_authorized": False,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "envelope_digest": "",
    }
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    pending = _write(queue / "pending" / "activation-1.json", envelope)
    setups = tmp_path / "setups"
    setups.mkdir()
    (setups / "activation-1.json").write_bytes(setup_path.read_bytes())

    def crash_after_allocator(**kwargs):
        output = Path(kwargs["output_root"])
        output.mkdir(parents=True, exist_ok=True)
        marker = {
            "schema_version": "task_evaluation_policy_canary_allocator_invocation.v1",
            "status": "started",
            "run_id": "scene-839873-canary-1",
            "allocator_invoked": True,
            "invocation_digest": "sha256:" + "a" * 64,
        }
        _write(output / "allocator_invocation_started.json", marker)
        raise TaskEvaluationPolicyCanaryDispatchError(
            "policy_canary_allocator_exit_17_without_result"
        )

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.dispatch_policy_canary_activation",
        crash_after_allocator,
    )
    zero = {
        "status": "provider_zero_confirmed",
        "provider_zero_verified": True,
        "live_instance_count": 0,
    }
    observed = process_policy_canary_dispatch_queue(
        dispatch_queue_root=queue,
        execution_setup_root=setups,
        dispatch_root=tmp_path / "dispatches",
        implementation_commit=COMMIT,
        execute=True,
        blocked_sync_runner=lambda **_kwargs: pytest.fail(
            "post-allocator failure cannot use preprovider sync"
        ),
        provider_zero_collector=lambda: zero,
    )

    result = observed["results"][0]
    assert result["status"] == "blocked_after_allocator_invocation_provider_zero"
    assert result["allocator_invoked"] is True
    assert result["provider_mutation_status"] == "unknown_after_allocator_invocation"
    assert result["automatic_retry_performed"] is False
    assert not pending.exists()
    assert (queue / "blocked" / pending.name).is_file()


def test_paid_queue_waits_for_setup_without_invoking_dispatcher(tmp_path: Path) -> None:
    activation_result, _setup_path, _activation = _inputs(tmp_path)
    queue = tmp_path / "queue"
    for name in ("pending", "processing", "completed", "blocked"):
        (queue / name).mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": "task_evaluation_policy_canary_dispatch_envelope.v1",
        "activation_id": "activation-1",
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "source_commit": COMMIT,
        "activation_result": _record(activation_result),
        "capture_session_id": "capture-839873",
        "intake_id": "intake-839873",
        "request_digest": "sha256:" + "7" * 64,
        "maximum_provider_allocations": 1,
        "retry_cap": 0,
        "automatic_retry_authorized": False,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "envelope_digest": "",
    }
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    envelope_path = _write(queue / "pending" / "activation-1.json", envelope)
    setups = tmp_path / "setups"
    setups.mkdir()

    observed = process_policy_canary_dispatch_queue(
        dispatch_queue_root=queue,
        execution_setup_root=setups,
        dispatch_root=tmp_path / "dispatches",
        implementation_commit=COMMIT,
        execute=True,
    )

    assert observed["results"][0]["status"] == (
        "waiting_for_scene839873_execution_setup"
    )
    assert observed["results"][0]["allocator_invoked"] is False
    assert envelope_path.is_file()
    assert (
        tmp_path / "dispatches/activation-1/preprovider_waiting.json"
    ).is_file()


def test_nonretryable_setup_refusal_moves_queue_only_after_blocked_email_sync(
    tmp_path: Path,
) -> None:
    activation_result, _setup_path, _activation = _inputs(tmp_path)
    queue = tmp_path / "queue"
    for name in ("pending", "processing", "completed", "blocked"):
        (queue / name).mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": "task_evaluation_policy_canary_dispatch_envelope.v1",
        "activation_id": "activation-1",
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "source_commit": COMMIT,
        "activation_result": _record(activation_result),
        "capture_session_id": "capture-839873",
        "intake_id": "intake-839873",
        "request_digest": "sha256:" + "7" * 64,
        "maximum_provider_allocations": 1,
        "retry_cap": 0,
        "automatic_retry_authorized": False,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "envelope_digest": "",
    }
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    pending = _write(queue / "pending" / "activation-1.json", envelope)
    setups = tmp_path / "setups"
    setups.mkdir()
    invalid_template = _write(tmp_path / "invalid-template.json", {})

    observed = process_policy_canary_dispatch_queue(
        dispatch_queue_root=queue,
        execution_setup_root=setups,
        execution_setup_template_path=invalid_template,
        dispatch_root=tmp_path / "dispatches",
        implementation_commit=COMMIT,
        execute=True,
        blocked_sync_runner=lambda **_kwargs: {
            "status": "succeeded",
            "notification_delivery": {"status": "delivered"},
        },
    )

    assert observed["results"][0]["status"] == "blocked_before_paid_dispatch"
    assert observed["results"][0]["allocator_invoked"] is False
    assert not pending.exists()
    assert (queue / "blocked" / pending.name).is_file()
    assert (
        tmp_path / "dispatches/activation-1/preprovider_blocked.json"
    ).is_file()


def test_stale_commit_setup_is_terminalized_before_allocator(
    tmp_path: Path,
) -> None:
    activation_result, setup_path, _activation = _inputs(tmp_path)
    queue = tmp_path / "queue"
    for name in ("pending", "processing", "completed", "blocked"):
        (queue / name).mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": "task_evaluation_policy_canary_dispatch_envelope.v1",
        "activation_id": "activation-1",
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "source_commit": COMMIT,
        "activation_result": _record(activation_result),
        "capture_session_id": "capture-839873",
        "intake_id": "intake-839873",
        "request_digest": "sha256:" + "7" * 64,
        "maximum_provider_allocations": 1,
        "retry_cap": 0,
        "automatic_retry_authorized": False,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "envelope_digest": "",
    }
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    pending = _write(queue / "pending" / "activation-1.json", envelope)
    setups = tmp_path / "setups"
    setups.mkdir()
    (setups / "activation-1.json").write_bytes(setup_path.read_bytes())

    observed = process_policy_canary_dispatch_queue(
        dispatch_queue_root=queue,
        execution_setup_root=setups,
        dispatch_root=tmp_path / "dispatches",
        implementation_commit="b" * 40,
        execute=True,
        blocked_sync_runner=lambda **_kwargs: {
            "status": "succeeded",
            "notification_delivery": {"status": "delivered"},
        },
    )

    result = observed["results"][0]
    assert result["status"] == "blocked_before_paid_dispatch"
    assert result["allocator_invoked"] is False
    assert result["blockers"] == ["policy_canary_dispatch_activation_setup_mismatch"]
    assert not pending.exists()
    assert (queue / "blocked" / pending.name).is_file()


def test_paid_queue_materializes_setup_from_staged_template_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    activation_result, source_setup, _activation = _inputs(tmp_path)
    queue = tmp_path / "queue"
    for name in ("pending", "processing", "completed", "blocked"):
        (queue / name).mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": "task_evaluation_policy_canary_dispatch_envelope.v1",
        "activation_id": "activation-1",
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "source_commit": COMMIT,
        "activation_result": _record(activation_result),
        "capture_session_id": "capture-839873",
        "intake_id": "intake-839873",
        "request_digest": "sha256:" + "7" * 64,
        "maximum_provider_allocations": 1,
        "retry_cap": 0,
        "automatic_retry_authorized": False,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "envelope_digest": "",
    }
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    pending = _write(queue / "pending" / "activation-1.json", envelope)
    template = _write(tmp_path / "template.json", {"static": True})
    setups = tmp_path / "setups"
    setups.mkdir()
    dispatches = tmp_path / "dispatches"

    def materialize(*, output_dir, **_kwargs):
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / "task_evaluation_policy_canary_execution_setup.v1.json"
        target.write_bytes(source_setup.read_bytes())
        return json.loads(target.read_text(encoding="utf-8"))

    def dispatch(**kwargs):
        output = Path(kwargs["output_root"])
        output.mkdir(parents=True, exist_ok=True)
        _write(output / "dispatch_receipt.json", {"status": "prepared_no_execution"})
        return {"status": "prepared_no_execution", "allocator_invoked": True}

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.materialize_scene839873_policy_canary_setup_from_template",
        materialize,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.dispatch_policy_canary_activation",
        dispatch,
    )

    observed = process_policy_canary_dispatch_queue(
        dispatch_queue_root=queue,
        execution_setup_root=setups,
        execution_setup_template_path=template,
        dispatch_root=dispatches,
        implementation_commit=COMMIT,
        execute=False,
    )

    assert observed["results"][0]["status"] == "prepared_no_execution"
    assert (
        setups
        / "activation-1/task_evaluation_policy_canary_execution_setup.v1.json"
    ).is_file()
    assert not pending.exists()
    assert (queue / "completed" / pending.name).is_file()


def _denying(*unusable: Path):
    """An access checker that refuses exactly the given paths for the service identity."""

    refused = {Path(path).resolve() for path in unusable}

    def access(path: Path, mode: int) -> bool:
        del mode
        return Path(path).resolve() not in refused

    return access


def test_service_access_preflight_blocks_unreadable_inputs_before_the_allocator(
    tmp_path: Path,
) -> None:
    activation_result, setup_path, _activation = _inputs(tmp_path)
    setup = json.loads(setup_path.read_text(encoding="utf-8"))
    runtime_inputs_path = Path(
        json.loads(activation_result.read_text(encoding="utf-8"))[
            "policy_canary_runtime_inputs_path"
        ]
    )
    overlay = _write(tmp_path / "overlay.zip", {"fixture": True})
    avoidlist = _write(tmp_path / "avoidlist.json", {"machine_ids": [144209]})
    cases = {
        "execution_setup": setup_path,
        "activation_result": activation_result,
        "runtime_inputs": runtime_inputs_path,
        "pi05_checkpoint_inventory": Path(setup["records"]["pi05_checkpoint_inventory"]["path"]),
        "hotfix_overlay": overlay,
        "machine_avoidlist": avoidlist,
    }
    for role, unusable in cases.items():
        with pytest.raises(TaskEvaluationPolicyCanaryDispatchError) as denied:
            dispatch_policy_canary_activation(
                activation_result_path=activation_result,
                execution_setup_path=setup_path,
                output_root=tmp_path / f"dispatch-{role}",
                implementation_commit=COMMIT,
                execute=True,
                hotfix_overlay_path=overlay,
                machine_avoidlist_path=avoidlist,
                allocator_runner=lambda _argv: pytest.fail("allocator must not run"),
                access=_denying(unusable),
            )
        assert str(denied.value) == (
            "policy_canary_dispatch_service_access_denied:"
            f"policy_canary_dispatch_input_unreadable:{role}"
        ), role
        assert not (tmp_path / f"dispatch-{role}").exists()
        assert str(unusable) not in str(denied.value)


def test_service_access_preflight_blocks_an_unwritable_run_directory(
    tmp_path: Path,
) -> None:
    activation_result, setup_path, _activation = _inputs(tmp_path)
    output = tmp_path / "dispatch"
    output.mkdir()

    with pytest.raises(
        TaskEvaluationPolicyCanaryDispatchError,
        match=(
            "^policy_canary_dispatch_service_access_denied:"
            "policy_canary_dispatch_output_unwritable:output_root$"
        ),
    ):
        dispatch_policy_canary_activation(
            activation_result_path=activation_result,
            execution_setup_path=setup_path,
            output_root=output,
            implementation_commit=COMMIT,
            execute=True,
            allocator_runner=lambda _argv: pytest.fail("allocator must not run"),
            access=_denying(output),
        )

    assert not (output / "status_events.jsonl").exists()


def test_missing_setup_still_reports_its_own_typed_code_after_the_access_preflight(
    tmp_path: Path,
) -> None:
    activation_result, _setup_path, _activation = _inputs(tmp_path)
    with pytest.raises(
        TaskEvaluationPolicyCanaryDispatchError,
        match="^policy_canary_scene839873_setup_receipt_missing$",
    ):
        dispatch_policy_canary_activation(
            activation_result_path=activation_result,
            execution_setup_path=tmp_path / "absent-setup.json",
            output_root=tmp_path / "dispatch",
            implementation_commit=COMMIT,
            allocator_runner=lambda _argv: pytest.fail("allocator must not run"),
        )


def test_machine_avoidlist_is_forwarded_to_the_allocator_after_access_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    activation_result, setup_path, _activation = _inputs(tmp_path)
    avoidlist = _write(tmp_path / "avoidlist.json", {"machine_ids": [144209]})
    observed: dict[str, object] = {}

    def fake_bundle(**kwargs):
        job = Path(kwargs["job_dir"])
        job.mkdir(parents=True, exist_ok=True)
        receipt = {"bundle_sha256": "sha256:" + "b" * 64, "bundle_path": str(job / "b.zip")}
        _write(job / "native_task_arena_policy_canary_session_bundle_receipt.v1.json", receipt)
        return receipt

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.build_policy_canary_session_bundle",
        fake_bundle,
    )

    def fake_allocator(argv):
        observed["argv"] = list(argv)
        adapter = Path(argv[argv.index("--adapter-output") + 1])
        _write(adapter, {"status": "dry_run_ready", "provider_mutations_performed": 0})
        return 0

    receipt = dispatch_policy_canary_activation(
        activation_result_path=activation_result,
        execution_setup_path=setup_path,
        output_root=tmp_path / "dispatch",
        implementation_commit=COMMIT,
        machine_avoidlist_path=avoidlist,
        allocator_runner=fake_allocator,
    )

    argv = observed["argv"]
    assert receipt["status"] == "prepared_no_execution"
    assert argv[argv.index("--adp-machine-avoidlist") + 1] == str(avoidlist.resolve())
    assert argv.count("--adp-machine-avoidlist") == 1


def test_paid_queue_reports_service_access_denial_as_a_pre_provider_block(
    tmp_path: Path,
) -> None:
    activation_result, setup_path, _activation = _inputs(tmp_path)
    queue = tmp_path / "queue"
    for name in ("pending", "processing", "completed", "blocked"):
        (queue / name).mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": "task_evaluation_policy_canary_dispatch_envelope.v1",
        "activation_id": "activation-1",
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "source_commit": COMMIT,
        "activation_result": _record(activation_result),
        "capture_session_id": "capture-839873",
        "intake_id": "intake-839873",
        "request_digest": "sha256:" + "7" * 64,
        "maximum_provider_allocations": 1,
        "retry_cap": 0,
        "automatic_retry_authorized": False,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "envelope_digest": "",
    }
    envelope["envelope_digest"] = canonical_digest(envelope, digest_field="envelope_digest")
    envelope_path = _write(queue / "pending" / "activation-1.json", envelope)
    setups = tmp_path / "setups"
    setups.mkdir()
    (setups / "activation-1.json").write_bytes(setup_path.read_bytes())
    dispatches = tmp_path / "dispatches"
    # The exact production failure: the run directory pre-created by root.
    run_root = dispatches / "activation-1"
    run_root.mkdir(parents=True)
    synced: list[dict[str, object]] = []

    def blocked_sync(**kwargs):
        synced.append(dict(kwargs))
        return {
            "status": "succeeded",
            "notification_delivery": {"status": "accepted", "terminal_state": "blocked"},
        }

    observed = process_policy_canary_dispatch_queue(
        dispatch_queue_root=queue,
        execution_setup_root=setups,
        dispatch_root=dispatches,
        implementation_commit=COMMIT,
        execute=True,
        blocked_sync_runner=blocked_sync,
        access=_denying(run_root),
    )

    result = observed["results"][0]
    assert result["status"] == "blocked_before_paid_dispatch"
    assert result["allocator_invoked"] is False
    assert result["blockers"] == [
        "policy_canary_dispatch_service_access_denied:"
        "policy_canary_dispatch_output_unwritable:output_root"
    ]
    assert synced[0]["blockers"] == result["blockers"]
    assert not envelope_path.exists()
    assert (queue / "blocked" / envelope_path.name).is_file()
    assert (run_root / "preprovider_blocked.json").is_file()


def test_paid_queue_retains_the_block_beside_the_queue_when_the_run_root_rejects_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    activation_result, setup_path, _activation = _inputs(tmp_path)
    queue = tmp_path / "queue"
    for name in ("pending", "processing", "completed", "blocked"):
        (queue / name).mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": "task_evaluation_policy_canary_dispatch_envelope.v1",
        "activation_id": "activation-1",
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "source_commit": COMMIT,
        "activation_result": _record(activation_result),
        "capture_session_id": "capture-839873",
        "intake_id": "intake-839873",
        "request_digest": "sha256:" + "7" * 64,
        "maximum_provider_allocations": 1,
        "retry_cap": 0,
        "automatic_retry_authorized": False,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "envelope_digest": "",
    }
    envelope["envelope_digest"] = canonical_digest(envelope, digest_field="envelope_digest")
    envelope_path = _write(queue / "pending" / "activation-1.json", envelope)
    setups = tmp_path / "setups"
    setups.mkdir()
    (setups / "activation-1.json").write_bytes(setup_path.read_bytes())
    dispatches = tmp_path / "dispatches"
    run_root = dispatches / "activation-1"
    run_root.mkdir(parents=True)
    real_write_json = json.dumps

    import blueprint_pipeline.task_evaluation_policy_canary_dispatcher as module

    original = module.write_json

    def rejecting_write(path, value):
        if Path(path).parent == run_root:
            raise PermissionError(str(path))
        return original(path, value)

    monkeypatch.setattr(module, "write_json", rejecting_write)
    del real_write_json

    observed = process_policy_canary_dispatch_queue(
        dispatch_queue_root=queue,
        execution_setup_root=setups,
        dispatch_root=dispatches,
        implementation_commit=COMMIT,
        execute=True,
        blocked_sync_runner=lambda **_kwargs: {
            "status": "succeeded",
            "notification_delivery": {"status": "accepted", "terminal_state": "blocked"},
        },
        access=_denying(run_root),
    )

    assert observed["results"][0]["status"] == "blocked_before_paid_dispatch"
    assert (dispatches / "unwritable-runs" / "activation-1" / "preprovider_blocked.json").is_file()
    assert (queue / "blocked" / envelope_path.name).is_file()


def test_legacy_activation_results_mode_honours_overlay_avoidlist_and_service_access(
    tmp_path: Path,
) -> None:
    """The third CLI mode forwards the same operator inputs as the queue mode.

    A flag the CLI accepts but one mode silently drops is a fail-open surface:
    the operator believes the avoidlist or signed overlay applies when it does
    not.  The access preflight names each role, which proves the path arrived.
    """

    import blueprint_pipeline.task_evaluation_policy_canary_dispatcher as module

    activation_result, setup_path, _activation = _inputs(tmp_path)
    results = tmp_path / "activation-results"
    results.mkdir()
    (results / "activation-1.json").write_bytes(activation_result.read_bytes())
    setups = tmp_path / "setups"
    setups.mkdir()
    (setups / "activation-1.json").write_bytes(setup_path.read_bytes())
    overlay = _write(tmp_path / "overlay.zip", {"fixture": True})
    avoidlist = _write(tmp_path / "avoidlist.json", {"machine_ids": [144209]})

    for role, unusable in (("hotfix_overlay", overlay), ("machine_avoidlist", avoidlist)):
        with pytest.raises(
            TaskEvaluationPolicyCanaryDispatchError,
            match=(
                "^policy_canary_dispatch_service_access_denied:"
                f"policy_canary_dispatch_input_unreadable:{role}$"
            ),
        ):
            module.process_policy_canary_activation_results(
                activation_results_root=results,
                execution_setup_root=setups,
                dispatch_root=tmp_path / "dispatches",
                implementation_commit=COMMIT,
                execute=True,
                hotfix_overlay_path=overlay,
                machine_avoidlist_path=avoidlist,
                access=_denying(unusable),
            )
        assert not (tmp_path / "dispatches" / "activation-1").exists()


def test_cli_legacy_mode_forwards_overlay_avoidlist_and_billing_audit_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import blueprint_pipeline.task_evaluation_policy_canary_dispatcher as module

    observed: dict[str, object] = {}

    def fake_legacy(**kwargs):
        observed.update(kwargs)
        return {"status": "idle", "processed_count": 0, "results": []}

    monkeypatch.setattr(module, "process_policy_canary_activation_results", fake_legacy)

    exit_code = module.main(
        [
            "--activation-results-root",
            str(tmp_path / "results"),
            "--execution-setup-root",
            str(tmp_path / "setups"),
            "--dispatch-root",
            str(tmp_path / "dispatches"),
            "--implementation-commit",
            COMMIT,
            "--hotfix-overlay",
            str(tmp_path / "overlay.zip"),
            "--machine-avoidlist",
            str(tmp_path / "avoidlist.json"),
            "--billing-audit-root",
            str(tmp_path / "billing-audit"),
            "--execute",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "idle"
    assert observed["hotfix_overlay_path"] == str(tmp_path / "overlay.zip")
    assert observed["machine_avoidlist_path"] == str(tmp_path / "avoidlist.json")
    assert observed["billing_audit_root"] == str(tmp_path / "billing-audit")
    assert observed["execute"] is True
