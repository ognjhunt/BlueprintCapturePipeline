from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_native_construction_optuna_ledger import (
    NativeConstructionOptunaLedgerError,
    NativeConstructionOptunaSearchLedger,
)

RUN_ID = "scene-839873-native-construction"


def _sealed(value: dict, field: str) -> dict:
    value[field] = ""
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _candidate(candidate_id: str, rank: int) -> dict:
    reset = _sealed(
        {
            "schema_version": "task_evaluation_native_robot_reset_variant.v1",
            "robot_joint_reset_positions_rad": {"panda_joint1": 0.1 + rank},
        },
        "reset_variant_digest",
    )
    entry = _sealed(
        {
            "schema_version": "task_evaluation_native_entry_trajectory_variant.v1",
            "waypoints": [
                {
                    "waypoint_id": "entry",
                    "position_world_m": [1.0, 2.0, 3.0],
                    "orientation_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            ],
        },
        "entry_trajectory_variant_digest",
    )
    camera = _sealed(
        {
            "schema_version": "task_evaluation_native_camera_variant.v1",
            "cameras": [{"role": "external", "position_world_m": [1, 2, 3]}],
        },
        "camera_variant_digest",
    )
    return _sealed(
        {
            "schema_version": "task_evaluation_native_construction_candidate.v1",
            "candidate_id": candidate_id,
            "deterministic_rank": rank,
            "support_surface_id": "table-top",
            "robot_base_pose_world": {
                "position_world_m": [float(rank), 0.0, 0.75],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "reset_variant": reset,
            "entry_trajectory_variant": entry,
            "camera_variant": camera,
            "addressed_feedback_codes": ["collision:precontact:robot_task"],
            "maximum_incremental_cost_usd": 0.2,
            "maximum_runtime_seconds": 60.0,
        },
        "candidate_digest",
    )


def _inventory(round_index: int, *candidates: dict) -> dict:
    return _sealed(
        {
            "schema_version": (
                "task_evaluation_native_construction_candidate_inventory.v1"
            ),
            "run_id": RUN_ID,
            "round_index": round_index,
            "source_native_feedback_digest": None,
            "model_authored_candidates": False,
            "candidates": list(candidates),
        },
        "inventory_digest",
    )


def _round_record(
    *, inventory: dict, candidate: dict, passed: bool, search_state: str
) -> dict:
    native_result = _sealed(
        {
            "schema_version": "native_task_arena_construction_result.v1",
            "status": "completed" if passed else "blocked",
            "blockers": [] if passed else ["native_rigid_construction_gate_failed:push_path"],
        },
        "result_digest",
    )
    feedback = _sealed(
        {
            "schema_version": "task_evaluation_native_construction_feedback.v1",
            "native_result_digest": native_result["result_digest"],
            "passed": passed,
            "native_blockers": list(native_result["blockers"]),
            "first_failed_phase": None if passed else "push_contact",
            "first_collision": None,
            "phase_measurements": [
                {
                    "phase_id": "push_contact",
                    "target_reached": passed,
                    "terminal_position_error_m": 0.003 if passed else 0.14,
                    "contacts": {"task_robot_contact_peak_force_n": 3.2},
                }
            ],
            "camera_measurements": {
                "external": {"passed": True, "pixel_count": 314}
            },
        },
        "feedback_digest",
    )
    execution = _sealed(
        {
            "schema_version": (
                "task_evaluation_native_construction_candidate_execution.v1"
            ),
            "status": "passed" if passed else "rejected",
            "candidate_id": candidate["candidate_id"],
            "candidate_digest": candidate["candidate_digest"],
            "inventory_digest": inventory["inventory_digest"],
            "provider_instance_id": 123,
            "provider_allocations_performed": 0,
            "runtime_seconds": 12.5,
            "incremental_cost_upper_bound_usd": 0.11,
            "native_result": native_result,
        },
        "execution_result_digest",
    )
    return {
        "round_index": inventory["round_index"],
        "inventory_digest": inventory["inventory_digest"],
        "candidate": candidate,
        "execution": execution,
        "native_feedback": feedback,
        "controller_search_state": search_state,
    }


def test_journal_records_rejected_trial_and_reopens_idempotently(tmp_path: Path) -> None:
    first, second = _candidate("candidate-a", 0), _candidate("candidate-b", 1)
    inventory = _inventory(0, first, second)
    ledger = NativeConstructionOptunaSearchLedger(
        root=tmp_path / "ledger", run_id=RUN_ID, seed=839873
    )

    inventory_receipt = ledger.record_inventory(inventory=inventory)
    record = _round_record(
        inventory=inventory,
        candidate=first,
        passed=False,
        search_state="continuing",
    )
    attempt_receipt = ledger.record_attempt(round_record=record)

    assert inventory_receipt["event"] == "inventory_recorded"
    assert inventory_receipt["candidate_authoring_performed"] is False
    assert inventory_receipt["grading_performed"] is False
    assert attempt_receipt["optuna_trial"]["state"] == "pruned"
    assert attempt_receipt["candidate_disposition"] == "discard"
    assert attempt_receipt["candidate_inventory_exhausted"] is False
    assert attempt_receipt["prune_reasons"] == [
        "native_rigid_construction_gate_failed:push_path"
    ]
    assert attempt_receipt["runtime_seconds"] == 12.5
    assert attempt_receipt["incremental_cost_upper_bound_usd"] == 0.11
    assert (
        attempt_receipt["native_outcome_metrics"]["phase_measurements"][0][
            "terminal_position_error_m"
        ]
        == 0.14
    )
    assert first["candidate_digest"] in attempt_receipt["optuna_trial"]["params"].values()
    assert inventory_receipt["optimizer_version"] == "4.9.0"
    assert inventory_receipt["optimizer_license"] == "MIT"

    reopened = NativeConstructionOptunaSearchLedger(
        root=tmp_path / "ledger", run_id=RUN_ID, seed=839873
    )
    assert reopened.reopen_receipt(attempt_receipt) == attempt_receipt
    assert reopened.record_inventory(inventory=inventory) == inventory_receipt
    assert reopened.record_attempt(round_record=record) == attempt_receipt
    assert len(reopened._study().trials) == 1


def test_passed_trial_is_kept_and_completed(tmp_path: Path) -> None:
    candidate = _candidate("candidate-pass", 0)
    inventory = _inventory(0, candidate)
    ledger = NativeConstructionOptunaSearchLedger(root=tmp_path, run_id=RUN_ID)
    ledger.record_inventory(inventory=inventory)

    receipt = ledger.record_attempt(
        round_record=_round_record(
            inventory=inventory,
            candidate=candidate,
            passed=True,
            search_state="qualified",
        )
    )

    assert receipt["optuna_trial"]["state"] == "complete"
    assert receipt["optuna_trial"]["value"] == 1.0
    assert receipt["candidate_disposition"] == "keep"
    assert receipt["candidate_inventory_exhausted"] is True
    assert receipt["controller_search_state"] == "qualified"


def test_attempt_receipt_accumulates_native_runtime_and_cost(tmp_path: Path) -> None:
    first, second = _candidate("candidate-a", 0), _candidate("candidate-b", 1)
    ledger = NativeConstructionOptunaSearchLedger(root=tmp_path, run_id=RUN_ID)
    initial = _inventory(0, first)
    ledger.record_inventory(inventory=initial)
    ledger.record_attempt(
        round_record=_round_record(
            inventory=initial,
            candidate=first,
            passed=False,
            search_state="continuing",
        )
    )
    following = _inventory(1, second)
    ledger.record_inventory(inventory=following)

    receipt = ledger.record_attempt(
        round_record=_round_record(
            inventory=following,
            candidate=second,
            passed=True,
            search_state="qualified",
        )
    )

    assert receipt["attempted_candidate_count"] == 2
    assert receipt["cumulative_runtime_seconds"] == 25.0
    assert receipt["cumulative_incremental_cost_upper_bound_usd"] == 0.22


def test_resume_after_trial_tell_before_receipt_write_does_not_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _candidate("candidate-resume", 0)
    inventory = _inventory(0, candidate)
    ledger = NativeConstructionOptunaSearchLedger(root=tmp_path, run_id=RUN_ID)
    ledger.record_inventory(inventory=inventory)
    record = _round_record(
        inventory=inventory,
        candidate=candidate,
        passed=False,
        search_state="exhausted_round_cap",
    )
    import blueprint_pipeline.task_evaluation_native_construction_optuna_ledger as module

    original_write = module._atomic_immutable_write

    def fail_attempt_write(path: Path, value: dict) -> None:
        if "attempt" in path.name:
            raise OSError("simulated publication interruption")
        original_write(path, value)

    monkeypatch.setattr(module, "_atomic_immutable_write", fail_attempt_write)
    with pytest.raises(OSError, match="publication interruption"):
        ledger.record_attempt(round_record=record)
    assert len(ledger._study().trials) == 1
    assert ledger._study().trials[0].state.name == "PRUNED"

    monkeypatch.setattr(module, "_atomic_immutable_write", original_write)
    resumed = NativeConstructionOptunaSearchLedger(root=tmp_path, run_id=RUN_ID)
    receipt = resumed.record_attempt(round_record=record)
    assert receipt["controller_search_state"] == "exhausted_round_cap"
    assert len(resumed._study().trials) == 1


def test_rejects_nonmember_and_repeated_candidate(tmp_path: Path) -> None:
    first, outsider = _candidate("candidate-a", 0), _candidate("candidate-z", 9)
    inventory = _inventory(0, first)
    ledger = NativeConstructionOptunaSearchLedger(root=tmp_path, run_id=RUN_ID)
    ledger.record_inventory(inventory=inventory)

    with pytest.raises(
        NativeConstructionOptunaLedgerError,
        match="native_construction_search_attempt_nonmember",
    ):
        ledger.record_attempt(
            round_record=_round_record(
                inventory=inventory,
                candidate=outsider,
                passed=False,
                search_state="continuing",
            )
        )

    ledger.record_attempt(
        round_record=_round_record(
            inventory=inventory,
            candidate=first,
            passed=False,
            search_state="continuing",
        )
    )
    repeated = _inventory(1, first)
    with pytest.raises(
        NativeConstructionOptunaLedgerError,
        match="native_construction_search_inventory_repeats_attempted_candidate",
    ):
        ledger.record_inventory(inventory=repeated)


def test_rejects_mutated_inventory_and_receipt_history(tmp_path: Path) -> None:
    candidate = _candidate("candidate-a", 0)
    inventory = _inventory(0, candidate)
    ledger = NativeConstructionOptunaSearchLedger(root=tmp_path, run_id=RUN_ID)

    mutated = dict(inventory)
    mutated["candidates"] = [dict(candidate, support_surface_id="floor")]
    with pytest.raises(
        NativeConstructionOptunaLedgerError,
        match="native_construction_search_inventory_invalid",
    ):
        ledger.record_inventory(inventory=mutated)

    ledger.record_inventory(inventory=inventory)
    receipt = ledger.record_attempt(
        round_record=_round_record(
            inventory=inventory,
            candidate=candidate,
            passed=True,
            search_state="qualified",
        )
    )
    tampered = dict(receipt)
    tampered["candidate_disposition"] = "discard"
    with pytest.raises(
        NativeConstructionOptunaLedgerError,
        match="native_construction_search_ledger_receipt_invalid",
    ):
        ledger.reopen_receipt(tampered)


def test_candidate_id_cannot_rebind_to_new_digest_between_rounds(tmp_path: Path) -> None:
    first = _candidate("candidate-a", 0)
    ledger = NativeConstructionOptunaSearchLedger(root=tmp_path, run_id=RUN_ID)
    ledger.record_inventory(inventory=_inventory(0, first))
    rebound = _candidate("candidate-a", 1)

    with pytest.raises(
        NativeConstructionOptunaLedgerError,
        match="native_construction_search_candidate_identity_rebound",
    ):
        ledger.record_inventory(inventory=_inventory(1, rebound))


def test_seed_is_deterministic_and_binds_study(tmp_path: Path) -> None:
    first = NativeConstructionOptunaSearchLedger(
        root=tmp_path / "first", run_id=RUN_ID
    )
    second = NativeConstructionOptunaSearchLedger(
        root=tmp_path / "second", run_id=RUN_ID
    )
    assert first.seed == second.seed
    assert first.study_name == second.study_name

    first.record_inventory(inventory=_inventory(0, _candidate("candidate-a", 0)))
    with pytest.raises(
        NativeConstructionOptunaLedgerError,
        match="native_construction_search_ledger_study_conflict",
    ):
        NativeConstructionOptunaSearchLedger(
            root=tmp_path / "first", run_id=RUN_ID, seed=1
        )._study()
