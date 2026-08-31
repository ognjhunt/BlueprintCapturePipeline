"""Durable Optuna ask/tell history for bounded native construction recovery.

This ledger is deliberately not a candidate generator and not a grader.  The
deterministic geometry/IK compiler supplies a self-digested candidate inventory,
the controller/model selects one exact member, and the native worker supplies
the sealed outcome.  Optuna persists that experiment history so a restarted
controller can resume without trying the same candidate twice.

The JournalStorage log is mutable by design.  Every public operation therefore
also emits an immutable, self-digested Blueprint receipt containing the exact
Optuna study/trial snapshot needed to reopen and verify the event.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import optuna
from optuna.distributions import CategoricalDistribution, distribution_to_json
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import FrozenTrial, TrialState

from .decision_evidence_contracts import canonical_digest, canonical_json

INVENTORY_SCHEMA_VERSION = (
    "task_evaluation_native_construction_candidate_inventory.v1"
)
INVENTORY_RECEIPT_SCHEMA_VERSION = (
    "task_evaluation_native_construction_optuna_inventory_ledger_receipt.v1"
)
ATTEMPT_RECEIPT_SCHEMA_VERSION = (
    "task_evaluation_native_construction_optuna_attempt_ledger_receipt.v1"
)
OPTUNA_VERSION = "4.9.0"
OPTUNA_LICENSE = "MIT"
STUDY_CONTRACT_VERSION = "blueprint_native_construction_optuna_study.v1"

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")


class NativeConstructionOptunaLedgerError(ValueError):
    """The persistent search history or one of its bindings was invalid."""


def _copy(value: Mapping[str, Any], *, blocker: str) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise NativeConstructionOptunaLedgerError(blocker) from exc
    if not isinstance(result, dict):
        raise NativeConstructionOptunaLedgerError(blocker)
    return result


def _digest(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


def _identifier(value: object) -> bool:
    return isinstance(value, str) and bool(_IDENTIFIER.fullmatch(value))


def _finite_nonnegative(value: object, *, blocker: str) -> float:
    if isinstance(value, bool):
        raise NativeConstructionOptunaLedgerError(blocker)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise NativeConstructionOptunaLedgerError(blocker) from exc
    if not math.isfinite(number) or number < 0.0:
        raise NativeConstructionOptunaLedgerError(blocker)
    return number


def _iso(value: object) -> str | None:
    return value.isoformat() if value is not None else None


def _atomic_immutable_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = (canonical_json(value) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise NativeConstructionOptunaLedgerError(
                "native_construction_search_ledger_immutable_conflict"
            )
        return
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o440)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != payload:
                raise NativeConstructionOptunaLedgerError(
                    "native_construction_search_ledger_immutable_conflict"
                )
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _load_receipt(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NativeConstructionOptunaLedgerError(
            "native_construction_search_ledger_receipt_invalid"
        ) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise NativeConstructionOptunaLedgerError(
            "native_construction_search_ledger_receipt_invalid"
        )
    return dict(value)


def _validate_inventory(value: Mapping[str, Any], *, run_id: str) -> dict[str, Any]:
    inventory = _copy(
        value, blocker="native_construction_search_inventory_invalid"
    )
    candidates = inventory.get("candidates")
    source_feedback = inventory.get("source_native_feedback_digest")
    if (
        inventory.get("schema_version") != INVENTORY_SCHEMA_VERSION
        or inventory.get("run_id") != run_id
        or not isinstance(inventory.get("round_index"), int)
        or int(inventory["round_index"]) < 0
        or (source_feedback is not None and not _digest(source_feedback))
        or inventory.get("model_authored_candidates") is not False
        or not isinstance(candidates, list)
        or not candidates
        or inventory.get("inventory_digest")
        != canonical_digest(inventory, digest_field="inventory_digest")
    ):
        raise NativeConstructionOptunaLedgerError(
            "native_construction_search_inventory_invalid"
        )
    ids: list[str] = []
    digests: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise NativeConstructionOptunaLedgerError(
                "native_construction_search_inventory_candidate_invalid"
            )
        candidate = dict(candidate)
        candidate_id = candidate.get("candidate_id")
        candidate_digest = candidate.get("candidate_digest")
        if (
            not _identifier(candidate_id)
            or not _digest(candidate_digest)
            or candidate_digest
            != canonical_digest(candidate, digest_field="candidate_digest")
        ):
            raise NativeConstructionOptunaLedgerError(
                "native_construction_search_inventory_candidate_invalid"
            )
        ids.append(str(candidate_id))
        digests.append(str(candidate_digest))
    if len(set(ids)) != len(ids) or len(set(digests)) != len(digests):
        raise NativeConstructionOptunaLedgerError(
            "native_construction_search_inventory_duplicate_candidate"
        )
    return inventory


def _candidate_dimensions(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only pre-admitted, execution-relevant search dimensions."""

    pose = candidate.get("robot_base_pose_world")
    reset = candidate.get("reset_variant")
    entry = candidate.get("entry_trajectory_variant")
    camera = candidate.get("camera_variant")
    dimensions: dict[str, Any] = {
        "candidate_id": candidate["candidate_id"],
        "candidate_digest": candidate["candidate_digest"],
        "deterministic_rank": candidate.get("deterministic_rank"),
        "support_surface_id": candidate.get("support_surface_id"),
        "robot_base_pose_world": dict(pose) if isinstance(pose, Mapping) else None,
        "reset_variant_digest": (
            reset.get("reset_variant_digest") if isinstance(reset, Mapping) else None
        ),
        "entry_trajectory_variant_digest": (
            entry.get("entry_trajectory_variant_digest")
            if isinstance(entry, Mapping)
            else None
        ),
        "camera_variant_digest": (
            camera.get("camera_variant_digest") if isinstance(camera, Mapping) else None
        ),
        "addressed_feedback_codes": sorted(
            str(row) for row in candidate.get("addressed_feedback_codes") or []
        ),
    }
    # Candidate bytes are already self-digested.  This round trip rejects NaN,
    # provider objects, and other values Optuna's journal could not preserve.
    return _copy(
        dimensions,
        blocker="native_construction_search_candidate_dimensions_invalid",
    )


def _trial_parameters(
    candidate: Mapping[str, Any], dimensions: Mapping[str, Any]
) -> dict[str, str]:
    suffix = str(candidate["candidate_digest"])[7:23]
    return {
        f"candidate_id__{suffix}": str(candidate["candidate_id"]),
        f"candidate_digest__{suffix}": str(candidate["candidate_digest"]),
        f"candidate_dimensions_json__{suffix}": canonical_json(dimensions),
    }


def _trial_snapshot(trial: FrozenTrial) -> dict[str, Any]:
    distributions = {
        name: json.loads(distribution_to_json(distribution))
        for name, distribution in sorted(trial.distributions.items())
    }
    return {
        "number": int(trial.number),
        "state": trial.state.name.lower(),
        "value": trial.value,
        "params": dict(sorted(trial.params.items())),
        "distributions": distributions,
        "user_attrs": dict(sorted(trial.user_attrs.items())),
        "system_attrs": dict(sorted(trial.system_attrs.items())),
        "datetime_start": _iso(trial.datetime_start),
        "datetime_complete": _iso(trial.datetime_complete),
    }


class NativeConstructionOptunaSearchLedger:
    """Journal-backed implementation of the controller's ``SearchLedger`` API."""

    def __init__(self, *, root: Path, run_id: str, seed: int | None = None) -> None:
        if not _identifier(run_id):
            raise NativeConstructionOptunaLedgerError(
                "native_construction_search_ledger_run_id_invalid"
            )
        resolved = Path(root).expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True, mode=0o750)
        if resolved.is_symlink() or not resolved.is_dir():
            raise NativeConstructionOptunaLedgerError(
                "native_construction_search_ledger_root_invalid"
            )
        derived_seed = int.from_bytes(
            hashlib.sha256(run_id.encode("utf-8")).digest()[:4], "big"
        )
        selected_seed = derived_seed if seed is None else int(seed)
        if not 0 <= selected_seed <= 2**32 - 1:
            raise NativeConstructionOptunaLedgerError(
                "native_construction_search_ledger_seed_invalid"
            )
        if optuna.__version__ != OPTUNA_VERSION:
            raise NativeConstructionOptunaLedgerError(
                "native_construction_search_ledger_optuna_version_invalid"
            )
        self.root = resolved
        self.run_id = run_id
        self.seed = selected_seed
        self.journal_path = self.root / "optuna-journal.v1.log"
        self.receipt_root = self.root / "immutable_receipts"
        self.lock_path = self.root / ".native-construction-search.lock"
        self.study_name = (
            "blueprint-native-construction-"
            + hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:24]
        )

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        self.lock_path.touch(mode=0o640, exist_ok=True)
        with self.lock_path.open("r+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _study(self) -> optuna.study.Study:
        if self.journal_path.is_symlink():
            raise NativeConstructionOptunaLedgerError(
                "native_construction_search_ledger_journal_invalid"
            )
        storage = JournalStorage(JournalFileBackend(str(self.journal_path)))
        study = optuna.create_study(
            storage=storage,
            study_name=self.study_name,
            direction="maximize",
            sampler=TPESampler(seed=self.seed),
            load_if_exists=True,
        )
        self.journal_path.chmod(0o640)
        expected_attrs: dict[str, Any] = {
            "study_contract_version": STUDY_CONTRACT_VERSION,
            "run_id": self.run_id,
            "deterministic_seed": self.seed,
            "candidate_authoring_performed": False,
            "grading_performed": False,
            "optimizer_package": "optuna",
            "optimizer_version": OPTUNA_VERSION,
            "optimizer_license": OPTUNA_LICENSE,
            "storage_backend": "JournalStorage(JournalFileBackend)",
            "sampler": "TPESampler",
        }
        for key, expected in expected_attrs.items():
            existing = study.user_attrs.get(key)
            if existing is not None and existing != expected:
                raise NativeConstructionOptunaLedgerError(
                    "native_construction_search_ledger_study_conflict"
                )
            if existing is None:
                study.set_user_attr(key, expected)
        return study

    def _inventory_path(self, round_index: int) -> Path:
        return self.receipt_root / f"round-{round_index:03d}-inventory.v1.json"

    def _attempt_path(self, round_index: int) -> Path:
        return self.receipt_root / f"round-{round_index:03d}-attempt.v1.json"

    def record_inventory(self, *, inventory: Mapping[str, Any]) -> Mapping[str, Any]:
        """Persist one exact digest-bound inventory without asking for new bytes."""

        admitted = _validate_inventory(inventory, run_id=self.run_id)
        round_index = int(admitted["round_index"])
        receipt_path = self._inventory_path(round_index)
        with self._locked():
            if receipt_path.exists():
                receipt = _load_receipt(receipt_path)
                self.reopen_receipt(receipt)
                if receipt.get("inventory_digest") != admitted["inventory_digest"]:
                    raise NativeConstructionOptunaLedgerError(
                        "native_construction_search_ledger_immutable_conflict"
                    )
                return receipt
            study = self._study()
            attempted = {
                str(trial.user_attrs.get("candidate_digest") or "")
                for trial in study.get_trials(deepcopy=False)
                if trial.user_attrs.get("event") == "attempt_recorded"
            }
            candidate_digests = [
                str(row["candidate_digest"]) for row in admitted["candidates"]
            ]
            if attempted.intersection(candidate_digests):
                raise NativeConstructionOptunaLedgerError(
                    "native_construction_search_inventory_repeats_attempted_candidate"
                )
            known_ids: dict[str, str] = {}
            known_digests: dict[str, str] = {}
            for prior_path in sorted(self.receipt_root.glob("round-*-inventory.v1.json")):
                prior = _load_receipt(prior_path)
                self.reopen_receipt(prior)
                for candidate_id, candidate_digest in zip(
                    prior["candidate_ids"], prior["candidate_digests"], strict=True
                ):
                    known_ids[str(candidate_id)] = str(candidate_digest)
                    known_digests[str(candidate_digest)] = str(candidate_id)
            for candidate in admitted["candidates"]:
                candidate_id = str(candidate["candidate_id"])
                candidate_digest = str(candidate["candidate_digest"])
                if (
                    candidate_id in known_ids
                    and known_ids[candidate_id] != candidate_digest
                ) or (
                    candidate_digest in known_digests
                    and known_digests[candidate_digest] != candidate_id
                ):
                    raise NativeConstructionOptunaLedgerError(
                        "native_construction_search_candidate_identity_rebound"
                    )
            dimension_records = []
            for candidate in admitted["candidates"]:
                dimensions = _candidate_dimensions(candidate)
                dimension_records.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "candidate_digest": candidate["candidate_digest"],
                        "dimensions": dimensions,
                        "dimensions_digest": canonical_digest(dimensions),
                    }
                )
            optuna_inventory_record = {
                "round_index": round_index,
                "inventory_digest": admitted["inventory_digest"],
                "source_native_feedback_digest": admitted.get(
                    "source_native_feedback_digest"
                ),
                "candidate_digests": candidate_digests,
                "candidate_dimension_digests": [
                    row["dimensions_digest"] for row in dimension_records
                ],
            }
            attr_name = f"inventory_round_{round_index:03d}"
            attr_value = canonical_json(optuna_inventory_record)
            existing = study.user_attrs.get(attr_name)
            if existing is not None and existing != attr_value:
                raise NativeConstructionOptunaLedgerError(
                    "native_construction_search_ledger_study_conflict"
                )
            if existing is None:
                study.set_user_attr(attr_name, attr_value)
            receipt: dict[str, Any] = {
                "schema_version": INVENTORY_RECEIPT_SCHEMA_VERSION,
                "event": "inventory_recorded",
                "run_id": self.run_id,
                "round_index": round_index,
                "inventory_digest": admitted["inventory_digest"],
                "candidate_digest": None,
                "execution_result_digest": None,
                "native_feedback_digest": None,
                "source_native_feedback_digest": admitted.get(
                    "source_native_feedback_digest"
                ),
                "candidate_count": len(admitted["candidates"]),
                "candidate_ids": [
                    str(row["candidate_id"]) for row in admitted["candidates"]
                ],
                "candidate_digests": candidate_digests,
                "candidate_dimensions": dimension_records,
                "optuna_study_name": self.study_name,
                "optuna_inventory_attr": attr_name,
                "optuna_inventory_record": optuna_inventory_record,
                "deterministic_seed": self.seed,
                "storage_backend": "JournalStorage(JournalFileBackend)",
                "optimizer_package": "optuna",
                "optimizer_version": OPTUNA_VERSION,
                "optimizer_license": OPTUNA_LICENSE,
                "candidate_authoring_performed": False,
                "grading_performed": False,
                "ledger_receipt_digest": "",
            }
            receipt["ledger_receipt_digest"] = canonical_digest(
                receipt, digest_field="ledger_receipt_digest"
            )
            _atomic_immutable_write(receipt_path, receipt)
            return receipt

    def record_attempt(
        self, *, round_record: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Ask one fixed Optuna trial and tell it the sealed native outcome."""

        row = _copy(
            round_record, blocker="native_construction_search_attempt_invalid"
        )
        try:
            round_index = int(row.get("round_index"))
        except (TypeError, ValueError) as exc:
            raise NativeConstructionOptunaLedgerError(
                "native_construction_search_attempt_invalid"
            ) from exc
        inventory_digest = row.get("inventory_digest")
        candidate = row.get("candidate")
        execution = row.get("execution")
        feedback = row.get("native_feedback")
        if (
            round_index < 0
            or not _digest(inventory_digest)
            or not isinstance(candidate, Mapping)
            or not isinstance(execution, Mapping)
            or not isinstance(feedback, Mapping)
        ):
            raise NativeConstructionOptunaLedgerError(
                "native_construction_search_attempt_invalid"
            )
        candidate = dict(candidate)
        execution = dict(execution)
        feedback = dict(feedback)
        candidate_digest = candidate.get("candidate_digest")
        execution_digest = execution.get("execution_result_digest")
        feedback_digest = feedback.get("feedback_digest")
        status = execution.get("status")
        controller_state = row.get("controller_search_state", "continuing")
        native_result = execution.get("native_result")
        if (
            not _digest(candidate_digest)
            or candidate_digest
            != canonical_digest(candidate, digest_field="candidate_digest")
            or not _digest(execution_digest)
            or execution_digest
            != canonical_digest(execution, digest_field="execution_result_digest")
            or not _digest(feedback_digest)
            or feedback_digest
            != canonical_digest(feedback, digest_field="feedback_digest")
            or execution.get("inventory_digest") != inventory_digest
            or execution.get("candidate_digest") != candidate_digest
            or execution.get("candidate_id") != candidate.get("candidate_id")
            or not isinstance(native_result, Mapping)
            or native_result.get("result_digest")
            != canonical_digest(native_result, digest_field="result_digest")
            or feedback.get("native_result_digest")
            != native_result.get("result_digest")
            or status not in {"passed", "rejected"}
            or (status == "passed") != (feedback.get("passed") is True)
            or controller_state
            not in {"continuing", "qualified", "exhausted_round_cap"}
            or (status == "passed") != (controller_state == "qualified")
        ):
            raise NativeConstructionOptunaLedgerError(
                "native_construction_search_attempt_invalid"
            )
        runtime_seconds = _finite_nonnegative(
            execution.get("runtime_seconds"),
            blocker="native_construction_search_attempt_runtime_invalid",
        )
        cost_usd = _finite_nonnegative(
            execution.get("incremental_cost_upper_bound_usd"),
            blocker="native_construction_search_attempt_cost_invalid",
        )
        inventory_receipt_path = self._inventory_path(round_index)
        attempt_receipt_path = self._attempt_path(round_index)
        with self._locked():
            if not inventory_receipt_path.exists():
                raise NativeConstructionOptunaLedgerError(
                    "native_construction_search_attempt_inventory_missing"
                )
            inventory_receipt = _load_receipt(inventory_receipt_path)
            self.reopen_receipt(inventory_receipt)
            members = dict(
                zip(
                    inventory_receipt["candidate_ids"],
                    inventory_receipt["candidate_digests"],
                    strict=True,
                )
            )
            if (
                inventory_receipt.get("inventory_digest") != inventory_digest
                or members.get(candidate.get("candidate_id")) != candidate_digest
            ):
                raise NativeConstructionOptunaLedgerError(
                    "native_construction_search_attempt_nonmember"
                )
            if attempt_receipt_path.exists():
                receipt = _load_receipt(attempt_receipt_path)
                self.reopen_receipt(receipt)
                if (
                    receipt.get("candidate_digest") != candidate_digest
                    or receipt.get("execution_result_digest") != execution_digest
                    or receipt.get("native_feedback_digest") != feedback_digest
                ):
                    raise NativeConstructionOptunaLedgerError(
                        "native_construction_search_ledger_immutable_conflict"
                    )
                return receipt

            study = self._study()
            dimensions = _candidate_dimensions(candidate)
            dimension_digest = canonical_digest(dimensions)
            parameters = _trial_parameters(candidate, dimensions)
            prune_reasons = (
                sorted(str(item) for item in feedback.get("native_blockers") or [])
                if status == "rejected"
                else []
            )
            if status == "rejected" and not prune_reasons:
                prune_reasons = ["native_construction_rejected"]
            disposition = "keep" if status == "passed" else "discard"
            inventory_remaining = [
                digest
                for digest in inventory_receipt["candidate_digests"]
                if digest != candidate_digest
            ]
            outcome_metrics = _copy(
                feedback,
                blocker="native_construction_search_attempt_metrics_invalid",
            )
            outcome_metrics_digest = canonical_digest(outcome_metrics)
            event_record = {
                "run_id": self.run_id,
                "round_index": round_index,
                "inventory_digest": inventory_digest,
                "candidate_digest": candidate_digest,
                "execution_result_digest": execution_digest,
                "native_feedback_digest": feedback_digest,
                "candidate_dimensions_digest": dimension_digest,
                "native_outcome_metrics_digest": outcome_metrics_digest,
                "runtime_seconds": runtime_seconds,
                "incremental_cost_upper_bound_usd": cost_usd,
                "disposition": disposition,
                "controller_search_state": controller_state,
            }
            event_digest = canonical_digest(event_record)
            matching: FrozenTrial | None = None
            for trial in study.get_trials(deepcopy=False):
                recorded_candidate = trial.user_attrs.get("candidate_digest")
                if recorded_candidate != candidate_digest:
                    continue
                if trial.user_attrs.get("attempt_event_digest") != event_digest:
                    raise NativeConstructionOptunaLedgerError(
                        "native_construction_search_candidate_repeated"
                    )
                matching = trial
                break
            user_attrs: dict[str, Any] = {
                "event": "attempt_recorded",
                "attempt_event_digest": event_digest,
                **event_record,
                "candidate_id": candidate["candidate_id"],
                "candidate_dimensions_json": canonical_json(dimensions),
                "native_outcome_metrics_json": canonical_json(outcome_metrics),
                "native_status": status,
                "prune_reasons_json": canonical_json({"reasons": prune_reasons}),
                "candidate_authoring_performed": False,
                "grading_performed": False,
            }
            if matching is None:
                distributions = {
                    name: CategoricalDistribution([value])
                    for name, value in parameters.items()
                }
                # Enqueue the exact admitted member and all lineage attrs before
                # ask transitions it to RUNNING. A crash at either boundary can
                # therefore reopen the same trial instead of duplicating it.
                study.enqueue_trial(
                    parameters,
                    user_attrs=user_attrs,
                    skip_if_exists=True,
                )
                live_trial = study.ask(fixed_distributions=distributions)
                if status == "passed":
                    study.tell(live_trial, values=1.0)
                else:
                    study.tell(live_trial, state=TrialState.PRUNED)
                frozen = study.trials[live_trial.number]
            else:
                if matching.state in {TrialState.WAITING, TrialState.RUNNING}:
                    if matching.state == TrialState.WAITING:
                        distributions = {
                            name: CategoricalDistribution([value])
                            for name, value in parameters.items()
                        }
                        live_trial = study.ask(fixed_distributions=distributions)
                        if live_trial.number != matching.number:
                            raise NativeConstructionOptunaLedgerError(
                                "native_construction_search_trial_invalid"
                            )
                        matching = study.trials[live_trial.number]
                    if dict(matching.params) != parameters:
                        raise NativeConstructionOptunaLedgerError(
                            "native_construction_search_trial_invalid"
                        )
                    for key, value in user_attrs.items():
                        if matching.user_attrs.get(key) != value:
                            raise NativeConstructionOptunaLedgerError(
                                "native_construction_search_trial_invalid"
                            )
                    if status == "passed":
                        study.tell(matching.number, values=1.0)
                    else:
                        study.tell(matching.number, state=TrialState.PRUNED)
                    frozen = study.trials[matching.number]
                else:
                    frozen = matching
            expected_state = (
                TrialState.COMPLETE if status == "passed" else TrialState.PRUNED
            )
            if (
                frozen.state != expected_state
                or dict(frozen.params) != parameters
                or any(frozen.user_attrs.get(key) != value for key, value in user_attrs.items())
            ):
                raise NativeConstructionOptunaLedgerError(
                    "native_construction_search_trial_invalid"
                )
            snapshot = _trial_snapshot(frozen)
            receipt = {
                "schema_version": ATTEMPT_RECEIPT_SCHEMA_VERSION,
                "event": "attempt_recorded",
                "run_id": self.run_id,
                "round_index": round_index,
                "inventory_digest": inventory_digest,
                "candidate_digest": candidate_digest,
                "execution_result_digest": execution_digest,
                "native_feedback_digest": feedback_digest,
                "native_result_digest": feedback["native_result_digest"],
                "candidate_id": candidate["candidate_id"],
                "candidate_dimensions": dimensions,
                "candidate_dimensions_digest": dimension_digest,
                "native_status": status,
                "native_outcome_metrics": outcome_metrics,
                "native_outcome_metrics_digest": outcome_metrics_digest,
                "runtime_seconds": runtime_seconds,
                "incremental_cost_upper_bound_usd": cost_usd,
                "candidate_disposition": disposition,
                "prune_reasons": prune_reasons,
                "candidate_inventory_remaining_digests": inventory_remaining,
                "candidate_inventory_exhausted": not inventory_remaining,
                "controller_search_state": controller_state,
                "optuna_study_name": self.study_name,
                "optuna_trial": snapshot,
                "optuna_trial_snapshot_digest": canonical_digest(snapshot),
                "deterministic_seed": self.seed,
                "storage_backend": "JournalStorage(JournalFileBackend)",
                "optimizer_package": "optuna",
                "optimizer_version": OPTUNA_VERSION,
                "optimizer_license": OPTUNA_LICENSE,
                "candidate_authoring_performed": False,
                "grading_performed": False,
                "ledger_receipt_digest": "",
            }
            receipt["ledger_receipt_digest"] = canonical_digest(
                receipt, digest_field="ledger_receipt_digest"
            )
            _atomic_immutable_write(attempt_receipt_path, receipt)
            return receipt

    def reopen_receipt(self, receipt: Mapping[str, Any]) -> Mapping[str, Any]:
        """Validate an immutable receipt against the current Optuna journal."""

        value = _copy(
            receipt,
            blocker="native_construction_search_ledger_receipt_invalid",
        )
        if (
            value.get("schema_version")
            not in {INVENTORY_RECEIPT_SCHEMA_VERSION, ATTEMPT_RECEIPT_SCHEMA_VERSION}
            or value.get("run_id") != self.run_id
            or value.get("optuna_study_name") != self.study_name
            or value.get("deterministic_seed") != self.seed
            or value.get("optimizer_package") != "optuna"
            or value.get("optimizer_version") != OPTUNA_VERSION
            or value.get("optimizer_license") != OPTUNA_LICENSE
            or value.get("storage_backend")
            != "JournalStorage(JournalFileBackend)"
            or value.get("candidate_authoring_performed") is not False
            or value.get("grading_performed") is not False
            or value.get("ledger_receipt_digest")
            != canonical_digest(value, digest_field="ledger_receipt_digest")
        ):
            raise NativeConstructionOptunaLedgerError(
                "native_construction_search_ledger_receipt_invalid"
            )
        study = self._study()
        if value.get("event") == "inventory_recorded":
            if (
                value.get("schema_version") != INVENTORY_RECEIPT_SCHEMA_VERSION
                or value.get("candidate_digest") is not None
                or value.get("execution_result_digest") is not None
                or value.get("native_feedback_digest") is not None
                or study.user_attrs.get(value.get("optuna_inventory_attr"))
                != canonical_json(value.get("optuna_inventory_record") or {})
            ):
                raise NativeConstructionOptunaLedgerError(
                    "native_construction_search_ledger_receipt_invalid"
                )
            return value
        if (
            value.get("event") != "attempt_recorded"
            or value.get("schema_version") != ATTEMPT_RECEIPT_SCHEMA_VERSION
            or not _digest(value.get("candidate_digest"))
            or not _digest(value.get("execution_result_digest"))
            or not _digest(value.get("native_feedback_digest"))
            or value.get("optuna_trial_snapshot_digest")
            != canonical_digest(value.get("optuna_trial") or {})
        ):
            raise NativeConstructionOptunaLedgerError(
                "native_construction_search_ledger_receipt_invalid"
            )
        trial_snapshot = value.get("optuna_trial")
        if not isinstance(trial_snapshot, Mapping):
            raise NativeConstructionOptunaLedgerError(
                "native_construction_search_ledger_receipt_invalid"
            )
        number = trial_snapshot.get("number")
        if not isinstance(number, int):
            raise NativeConstructionOptunaLedgerError(
                "native_construction_search_ledger_receipt_invalid"
            )
        trials = {trial.number: trial for trial in study.get_trials(deepcopy=False)}
        trial = trials.get(number)
        if trial is None or _trial_snapshot(trial) != dict(trial_snapshot):
            raise NativeConstructionOptunaLedgerError(
                "native_construction_search_ledger_receipt_history_mismatch"
            )
        return value
