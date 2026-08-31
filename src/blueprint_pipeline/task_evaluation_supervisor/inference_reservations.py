"""Persistent worst-case inference reservations for interruption-safe SDK budgets."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from ..common import read_json, write_json
from ..decision_evidence_contracts import canonical_digest


INFERENCE_RESERVATION_SCHEMA_VERSION = "task_evaluation_inference_reservation.v1"
INFERENCE_COMPLETION_SCHEMA_VERSION = "task_evaluation_inference_completion.v1"
INFERENCE_RESERVATION_MANIFEST_SCHEMA_VERSION = (
    "task_evaluation_inference_reservation_manifest.v1"
)


class InferenceReservationError(ValueError):
    """Raised when inference reservation evidence is missing or inconsistent."""


class InferenceReservationAudit:
    """Write reservation intent before a provider call and completion afterward."""

    def __init__(self, *, run_root: str | Path, run_id: str) -> None:
        self.run_root = Path(run_root).expanduser().resolve()
        self.run_id = run_id
        self.root = self.run_root / "inference_reservations"
        self.reserved_root = self.root / "reserved"
        self.completed_root = self.root / "completed"

    @staticmethod
    def _file_token(reservation_id: str) -> str:
        if not reservation_id.startswith("sha256:") or len(reservation_id) != 71:
            raise InferenceReservationError("inference_reservation_id_invalid")
        return reservation_id.removeprefix("sha256:")

    def _reservation_path(self, reservation_id: str) -> Path:
        return self.reserved_root / f"{self._file_token(reservation_id)}.json"

    def _completion_path(self, reservation_id: str) -> Path:
        return self.completed_root / f"{self._file_token(reservation_id)}.json"

    def record_reservation(self, value: Mapping[str, Any]) -> None:
        reservation = dict(value)
        if reservation.get("schema_version") != INFERENCE_RESERVATION_SCHEMA_VERSION:
            raise InferenceReservationError("inference_reservation_schema_invalid")
        if reservation.get("run_id") != self.run_id:
            raise InferenceReservationError("inference_reservation_run_mismatch")
        projected = reservation.get("projected_max_cost_usd")
        if (
            not isinstance(projected, (int, float))
            or isinstance(projected, bool)
            or not math.isfinite(float(projected))
            or float(projected) <= 0
        ):
            raise InferenceReservationError("inference_reservation_cost_invalid")
        reservation_id = str(reservation.get("reservation_id") or "")
        reservation_identity = {
            "run_id": reservation.get("run_id"),
            "capability": reservation.get("capability"),
            "model": reservation.get("model"),
            "input_digest": reservation.get("input_digest"),
            "max_turns": reservation.get("max_turns"),
            "max_output_tokens": reservation.get("max_output_tokens"),
        }
        if "reasoning_effort" in reservation:
            reservation_identity["reasoning_effort"] = reservation.get(
                "reasoning_effort"
            )
        if "cache_policy_digest" in reservation:
            reservation_identity["cache_policy_digest"] = reservation.get(
                "cache_policy_digest"
            )
        expected_id = canonical_digest(reservation_identity)
        if reservation_id != expected_id:
            raise InferenceReservationError("inference_reservation_identity_mismatch")
        expected_digest = canonical_digest(
            reservation,
            digest_field="inference_reservation_digest",
        )
        if reservation.get("inference_reservation_digest") != expected_digest:
            raise InferenceReservationError("inference_reservation_digest_mismatch")
        path = self._reservation_path(reservation_id)
        if path.exists():
            existing = read_json(path)
            if existing != reservation:
                raise InferenceReservationError("inference_reservation_reuse_mismatch")
            raise InferenceReservationError(
                "prior_inference_reservation_requires_operator_review"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, reservation)

    def record_completion(self, value: Mapping[str, Any]) -> None:
        completion = dict(value)
        if completion.get("schema_version") != INFERENCE_COMPLETION_SCHEMA_VERSION:
            raise InferenceReservationError("inference_completion_schema_invalid")
        reservation_id = str(completion.get("reservation_id") or "")
        reservation_path = self._reservation_path(reservation_id)
        if not reservation_path.is_file():
            raise InferenceReservationError("inference_completion_reservation_missing")
        reservation = dict(read_json(reservation_path))
        expected_reservation_digest = canonical_digest(
            reservation,
            digest_field="inference_reservation_digest",
        )
        if reservation.get("inference_reservation_digest") != expected_reservation_digest:
            raise InferenceReservationError("inference_reservation_digest_mismatch")
        for field in ("run_id", "capability", "model"):
            if completion.get(field) != reservation.get(field):
                raise InferenceReservationError(
                    f"inference_completion_{field}_mismatch"
                )
        if completion.get("provider") != "openai":
            raise InferenceReservationError("inference_completion_provider_mismatch")
        if completion.get("cache_policy") != reservation.get("cache_policy"):
            raise InferenceReservationError("inference_completion_cache_policy_mismatch")
        completion_policy = completion.get("cache_policy")
        if (
            not isinstance(completion_policy, Mapping)
            or completion_policy.get("policy_digest")
            != reservation.get("cache_policy_digest")
        ):
            raise InferenceReservationError(
                "inference_completion_cache_policy_digest_mismatch"
            )
        if completion.get("breakpoint_digests") != reservation.get(
            "breakpoint_digests"
        ):
            raise InferenceReservationError(
                "inference_completion_breakpoint_digests_mismatch"
            )
        projected = float(reservation.get("projected_max_cost_usd") or 0.0)
        completion_projected = completion.get("projected_max_cost_usd")
        reconciled = completion.get("reconciled_actual_cost_usd")
        released = completion.get("released_reservation_usd")
        for label, raw in (
            ("projected_cost", completion_projected),
            ("reconciled_cost", reconciled),
            ("released_reservation", released),
        ):
            if (
                not isinstance(raw, (int, float))
                or isinstance(raw, bool)
                or not math.isfinite(float(raw))
                or float(raw) < 0
            ):
                raise InferenceReservationError(
                    f"inference_completion_{label}_invalid"
                )
        if abs(float(completion_projected) - projected) > 1.0e-12:
            raise InferenceReservationError(
                "inference_completion_projected_cost_mismatch"
            )
        if float(reconciled) > projected + 1.0e-12:
            raise InferenceReservationError(
                "inference_completion_reconciled_cost_exceeds_reservation"
            )
        expected_released = max(0.0, projected - float(reconciled))
        if abs(float(released) - expected_released) > 1.0e-12:
            raise InferenceReservationError(
                "inference_completion_released_reservation_mismatch"
            )
        expected_digest = canonical_digest(
            completion,
            digest_field="inference_completion_digest",
        )
        if completion.get("inference_completion_digest") != expected_digest:
            raise InferenceReservationError("inference_completion_digest_mismatch")
        path = self._completion_path(reservation_id)
        if path.exists():
            existing = read_json(path)
            if existing != completion:
                raise InferenceReservationError("inference_completion_reuse_mismatch")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, completion)

    def manifest(self) -> dict[str, Any]:
        reservations: list[dict[str, Any]] = []
        completions: dict[str, dict[str, Any]] = {}
        for path in sorted(self.completed_root.glob("*.json")):
            value = dict(read_json(path))
            expected = canonical_digest(value, digest_field="inference_completion_digest")
            if value.get("inference_completion_digest") != expected:
                raise InferenceReservationError("inference_completion_digest_mismatch")
            reservation_id = str(value.get("reservation_id") or "")
            if path != self._completion_path(reservation_id):
                raise InferenceReservationError("inference_completion_path_mismatch")
            completions[reservation_id] = value
        total = 0.0
        projected_total = 0.0
        for path in sorted(self.reserved_root.glob("*.json")):
            value = dict(read_json(path))
            if value.get("run_id") != self.run_id:
                raise InferenceReservationError("inference_reservation_run_mismatch")
            expected = canonical_digest(value, digest_field="inference_reservation_digest")
            if value.get("inference_reservation_digest") != expected:
                raise InferenceReservationError("inference_reservation_digest_mismatch")
            reservation_id = str(value.get("reservation_id") or "")
            if path != self._reservation_path(reservation_id):
                raise InferenceReservationError("inference_reservation_path_mismatch")
            projected = float(value.get("projected_max_cost_usd") or 0.0)
            if not math.isfinite(projected) or projected <= 0:
                raise InferenceReservationError("inference_reservation_cost_invalid")
            completion = completions.pop(reservation_id, None)
            reconciled = projected
            if completion is not None and "reconciled_actual_cost_usd" in completion:
                raw_reconciled = completion.get("reconciled_actual_cost_usd")
                if (
                    not isinstance(raw_reconciled, (int, float))
                    or isinstance(raw_reconciled, bool)
                    or not math.isfinite(float(raw_reconciled))
                    or float(raw_reconciled) < 0
                ):
                    raise InferenceReservationError(
                        "inference_completion_reconciled_cost_invalid"
                    )
                reconciled = float(raw_reconciled)
                if reconciled > projected + 1.0e-12:
                    raise InferenceReservationError(
                        "inference_completion_reconciled_cost_exceeds_reservation"
                    )
            projected_total += projected
            total += reconciled if completion is not None else projected
            reservations.append(
                {
                    "reservation_id": reservation_id,
                    "reservation_digest": value["inference_reservation_digest"],
                    "reservation_path": str(path.relative_to(self.run_root)),
                    "projected_max_cost_usd": projected,
                    "reconciled_actual_cost_usd": (
                        reconciled if completion is not None else None
                    ),
                    "released_reservation_usd": (
                        max(0.0, projected - reconciled)
                        if completion is not None
                        else 0.0
                    ),
                    "status": "completed" if completion is not None else "in_flight_unknown",
                    "completion_digest": (
                        None if completion is None else completion["inference_completion_digest"]
                    ),
                    "completion_path": (
                        None
                        if completion is None
                        else str(self._completion_path(reservation_id).relative_to(self.run_root))
                    ),
                }
            )
        if completions:
            raise InferenceReservationError("orphan_inference_completion")
        manifest: dict[str, Any] = {
            "schema_version": INFERENCE_RESERVATION_MANIFEST_SCHEMA_VERSION,
            "run_id": self.run_id,
            "reservations": reservations,
            "reservation_count": len(reservations),
            "in_flight_unknown_count": sum(
                row["status"] == "in_flight_unknown" for row in reservations
            ),
            "reserved_max_cost_usd": total,
            "projected_max_cost_usd_total": projected_total,
            "proof_effect": "none",
        }
        manifest["inference_reservation_manifest_digest"] = canonical_digest(
            manifest,
            digest_field="inference_reservation_manifest_digest",
        )
        return manifest

    def write_manifest(self) -> dict[str, Any]:
        manifest = self.manifest()
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.root / "manifest.json", manifest)
        return manifest


__all__ = [
    "INFERENCE_COMPLETION_SCHEMA_VERSION",
    "INFERENCE_RESERVATION_MANIFEST_SCHEMA_VERSION",
    "INFERENCE_RESERVATION_SCHEMA_VERSION",
    "InferenceReservationAudit",
    "InferenceReservationError",
]
