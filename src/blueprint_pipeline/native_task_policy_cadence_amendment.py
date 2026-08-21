"""Derive a 15 Hz DROID policy packet from a sealed 20 Hz construction packet.

The Scene 840920 task freeze predates the learned-policy runtime and froze a
20 Hz control cadence.  Both frozen candidates consume the DROID temporal
contract at 15 Hz; silently executing their rows at 20 Hz changes the policy.
This provider-free amendment binds one exact source freeze and packet request,
changes only the cadence fields, preserves the task prompt byte-for-byte, and
materializes the complete replacement packet so the derived scene plan proves
``physics=120 Hz / control=15 Hz / decimation=8`` before any allocation.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .dual_task_rehearsal_contract import (
    FROZEN_CANDIDATES,
    DualTaskRehearsalContractError,
    validate_task_freeze,
)
from .freeze_amendment_carry_forward import (
    FreezeAmendmentCarryForwardError,
    changed_document_paths,
    evaluate_freeze_amendment_carry_forward,
    validate_freeze_amendment_carry_forward_content,
)
from .native_task_arena_packet import (
    NativeTaskArenaPacketError,
    materialize_native_task_arena_packet,
    validate_native_task_arena_packet_request,
)


REQUEST_SCHEMA_VERSION = "native_task_policy_cadence_amendment_request.v1"
RECEIPT_SCHEMA_VERSION = "native_task_policy_cadence_amendment.v1"
PROGRAM_ID = "arm-decision-proof-v1"
SOURCE_CONTROL_FREQUENCY_HZ = 20
TARGET_CONTROL_FREQUENCY_HZ = 15
PHYSICS_FREQUENCY_HZ = 120
TARGET_CONTROL_DECIMATION = 8
OFFICIAL_DROID_CADENCE_SOURCE = (
    "https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/main.py"
)
_RFC3339_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_CONSTRUCTION_BINDING_CHANGE = re.compile(
    r"^construction_bindings\.bindings\[\d+\]\."
    r"(?:task_freeze_digest|evidence_receipts\.task_freeze\."
    r"(?:path|size_bytes|sha256|canonical_digest))$"
)


class NativeTaskPolicyCadenceAmendmentError(ValueError):
    """Stable provider-free refusal for an unsafe cadence amendment."""

    def __init__(self, errors: list[str] | tuple[str, ...]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _read(path: str | Path, *, error: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeTaskPolicyCadenceAmendmentError([error]) from exc
    if source.is_symlink() or not isinstance(payload, dict):
        raise NativeTaskPolicyCadenceAmendmentError([error])
    return source, payload


def _source_pair(
    *, task_freeze_path: str | Path, packet_request_path: str | Path
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    freeze_path, raw_freeze = _read(
        task_freeze_path, error="policy_cadence_source_freeze_invalid"
    )
    request_path, raw_request = _read(
        packet_request_path, error="policy_cadence_source_packet_request_invalid"
    )
    try:
        freeze = validate_task_freeze(raw_freeze)
        packet = validate_native_task_arena_packet_request(raw_request)
    except (DualTaskRehearsalContractError, NativeTaskArenaPacketError) as exc:
        raise NativeTaskPolicyCadenceAmendmentError(
            [f"policy_cadence_source_contract_invalid:{type(exc).__name__}"]
        ) from exc

    execution = freeze.get("execution_contract") or {}
    task_spec = packet.get("task_spec") or {}
    errors: list[str] = []
    if tuple(freeze.get("candidate_ids") or ()) != FROZEN_CANDIDATES:
        errors.append("policy_cadence_frozen_candidates_invalid")
    if freeze.get("learned_policy_outcomes_accessed") is not False:
        errors.append("policy_cadence_outcome_leakage")
    if freeze.get("task_id") != packet.get("task_id"):
        errors.append("policy_cadence_task_id_mismatch")
    if packet.get("task_freeze_digest") != freeze.get("task_freeze_digest"):
        errors.append("policy_cadence_packet_freeze_binding_mismatch")
    prompt = freeze.get("prompt")
    if not isinstance(prompt, str) or not prompt or task_spec.get("prompt") != prompt:
        errors.append("policy_cadence_prompt_mismatch")
    if execution.get("control_frequency_hz") != SOURCE_CONTROL_FREQUENCY_HZ:
        errors.append("policy_cadence_source_freeze_frequency_invalid")
    if task_spec.get("control_frequency_hz") != SOURCE_CONTROL_FREQUENCY_HZ:
        errors.append("policy_cadence_source_packet_frequency_invalid")
    if packet.get("physics_frequency_hz") != PHYSICS_FREQUENCY_HZ:
        errors.append("policy_cadence_physics_frequency_invalid")
    amendments = freeze.get("freeze_amendments") or []
    if not isinstance(amendments, list) or any(
        isinstance(row, Mapping)
        and row.get("field") == "execution_contract.control_frequency_hz"
        for row in amendments
    ):
        errors.append("policy_cadence_source_already_amended")
    if errors:
        raise NativeTaskPolicyCadenceAmendmentError(errors)
    return freeze_path, freeze, request_path, packet


def materialize_native_task_policy_cadence_amendment_request(
    *,
    task_freeze_path: str | Path,
    packet_request_path: str | Path,
    authorized_by: str,
    authorized_at: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Bind operator authority to one exact source pair without changing it."""

    freeze_path, freeze, request_path, packet = _source_pair(
        task_freeze_path=task_freeze_path,
        packet_request_path=packet_request_path,
    )
    authority = str(authorized_by or "").strip()
    if not authority or not _RFC3339_UTC.fullmatch(str(authorized_at or "")):
        raise NativeTaskPolicyCadenceAmendmentError(
            ["policy_cadence_amendment_authority_invalid"]
        )
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise NativeTaskPolicyCadenceAmendmentError(
            ["policy_cadence_amendment_request_output_exists"]
        )
    payload: dict[str, Any] = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "task_id": freeze["task_id"],
        "candidate_ids": list(FROZEN_CANDIDATES),
        "source_task_freeze_digest": freeze["task_freeze_digest"],
        "source_task_freeze_sha256": _sha256(freeze_path),
        "source_packet_request_digest": packet["request_digest"],
        "source_packet_request_sha256": _sha256(request_path),
        "source_prompt_digest": canonical_digest({"prompt": freeze["prompt"]}),
        "source_control_frequency_hz": SOURCE_CONTROL_FREQUENCY_HZ,
        "target_control_frequency_hz": TARGET_CONTROL_FREQUENCY_HZ,
        "physics_frequency_hz": PHYSICS_FREQUENCY_HZ,
        "target_control_decimation": TARGET_CONTROL_DECIMATION,
        "official_droid_cadence_source": OFFICIAL_DROID_CADENCE_SOURCE,
        "authorized_by": authority,
        "authorized_at": authorized_at,
        "learned_policy_outcomes_accessed": False,
        "provider_mutation_performed": False,
        "request_digest": "",
    }
    payload["request_digest"] = canonical_digest(payload, digest_field="request_digest")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def validate_native_task_policy_cadence_amendment_request(
    value: Mapping[str, Any],
    *,
    freeze_path: Path,
    freeze: Mapping[str, Any],
    packet_path: Path,
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject a tampered or stale amendment request against its source bytes."""

    try:
        payload = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise NativeTaskPolicyCadenceAmendmentError(
            ["policy_cadence_amendment_request_invalid"]
        ) from exc
    expected = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "task_id": freeze["task_id"],
        "candidate_ids": list(FROZEN_CANDIDATES),
        "source_task_freeze_digest": freeze["task_freeze_digest"],
        "source_task_freeze_sha256": _sha256(freeze_path),
        "source_packet_request_digest": packet["request_digest"],
        "source_packet_request_sha256": _sha256(packet_path),
        "source_prompt_digest": canonical_digest({"prompt": freeze["prompt"]}),
        "source_control_frequency_hz": SOURCE_CONTROL_FREQUENCY_HZ,
        "target_control_frequency_hz": TARGET_CONTROL_FREQUENCY_HZ,
        "physics_frequency_hz": PHYSICS_FREQUENCY_HZ,
        "target_control_decimation": TARGET_CONTROL_DECIMATION,
        "official_droid_cadence_source": OFFICIAL_DROID_CADENCE_SOURCE,
        "learned_policy_outcomes_accessed": False,
        "provider_mutation_performed": False,
    }
    errors = [
        f"policy_cadence_amendment_request_field_invalid:{field}"
        for field, expected_value in expected.items()
        if payload.get(field) != expected_value
    ]
    if not str(payload.get("authorized_by") or "").strip() or not _RFC3339_UTC.fullmatch(
        str(payload.get("authorized_at") or "")
    ):
        errors.append("policy_cadence_amendment_authority_invalid")
    if payload.get("request_digest") != canonical_digest(
        payload, digest_field="request_digest"
    ):
        errors.append("policy_cadence_amendment_request_digest_invalid")
    if errors:
        raise NativeTaskPolicyCadenceAmendmentError(errors)
    return payload


def validate_native_task_policy_cadence_amendment(
    value: Mapping[str, Any],
    *,
    source_freeze: Mapping[str, Any],
    source_packet_request: Mapping[str, Any],
    amended_freeze: Mapping[str, Any],
    amended_packet_request: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute every carry-forward relation from actual before/after documents."""

    payload = json.loads(json.dumps(dict(value)))
    try:
        validated_source_freeze = validate_task_freeze(source_freeze)
        validated_amended_freeze = validate_task_freeze(amended_freeze)
        validated_source_packet = validate_native_task_arena_packet_request(
            source_packet_request
        )
        validated_amended_packet = validate_native_task_arena_packet_request(
            amended_packet_request
        )
    except (DualTaskRehearsalContractError, NativeTaskArenaPacketError) as exc:
        raise NativeTaskPolicyCadenceAmendmentError(
            ["policy_cadence_amendment_document_contract_invalid"]
        ) from exc
    source_freeze = validated_source_freeze
    amended_freeze = validated_amended_freeze
    source_packet_request = validated_source_packet
    amended_packet_request = validated_amended_packet
    freeze_changes = sorted(
        path
        for path in changed_document_paths(source_freeze, amended_freeze)
        if path != "task_freeze_digest"
    )
    request_changes = sorted(
        path
        for path in changed_document_paths(source_packet_request, amended_packet_request)
        if path != "request_digest"
    )
    errors: list[str] = []
    if payload.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        errors.append("policy_cadence_amendment_receipt_schema_invalid")
    if payload.get("status") != "materialized":
        errors.append("policy_cadence_amendment_receipt_status_invalid")
    source_execution = source_freeze.get("execution_contract") or {}
    amended_execution = amended_freeze.get("execution_contract") or {}
    source_task_spec = source_packet_request.get("task_spec") or {}
    amended_task_spec = amended_packet_request.get("task_spec") or {}
    if (
        source_execution.get("control_frequency_hz")
        != SOURCE_CONTROL_FREQUENCY_HZ
        or source_task_spec.get("control_frequency_hz")
        != SOURCE_CONTROL_FREQUENCY_HZ
        or amended_execution.get("control_frequency_hz")
        != TARGET_CONTROL_FREQUENCY_HZ
        or amended_task_spec.get("control_frequency_hz")
        != TARGET_CONTROL_FREQUENCY_HZ
        or source_packet_request.get("physics_frequency_hz")
        != PHYSICS_FREQUENCY_HZ
        or amended_packet_request.get("physics_frequency_hz")
        != PHYSICS_FREQUENCY_HZ
    ):
        errors.append("policy_cadence_amendment_frequency_relation_invalid")
    amendment_rows = amended_freeze.get("freeze_amendments") or []
    amendment = amendment_rows[-1] if amendment_rows else {}
    expected_amendment = {
        "schema_version": "native_task_policy_cadence_amendment_record.v1",
        "field": "execution_contract.control_frequency_hz",
        "from": SOURCE_CONTROL_FREQUENCY_HZ,
        "to": TARGET_CONTROL_FREQUENCY_HZ,
        "reason": "match_the_frozen_DROID_policy_temporal_contract",
        "official_source": OFFICIAL_DROID_CADENCE_SOURCE,
        "candidate_ids": list(FROZEN_CANDIDATES),
        "superseded_task_freeze_digest": source_freeze.get("task_freeze_digest"),
        "source_packet_request_digest": source_packet_request.get("request_digest"),
        "task_semantics_changed": False,
        "learned_policy_outcomes_accessed": False,
        "provider_mutation_performed": False,
    }
    if not isinstance(amendment, Mapping) or any(
        amendment.get(field) != expected_value
        for field, expected_value in expected_amendment.items()
    ):
        errors.append("policy_cadence_amendment_record_invalid")
    if not str(amendment.get("authorized_by") or "").strip() or not _RFC3339_UTC.fullmatch(
        str(amendment.get("authorized_at") or "")
    ):
        errors.append("policy_cadence_amendment_record_authority_invalid")
    if not _digest(payload.get("amendment_request_digest")):
        errors.append("policy_cadence_amendment_request_binding_invalid")
    if freeze_changes != [
        "execution_contract.control_frequency_hz",
        "freeze_amendments",
    ]:
        errors.append("policy_cadence_amended_freeze_changes_invalid")
    required_request_changes = {
        "task_freeze_digest",
        "task_spec.control_frequency_hz",
        "construction_bindings.construction_digest",
    }
    if not required_request_changes.issubset(request_changes) or any(
        path not in required_request_changes
        and _CONSTRUCTION_BINDING_CHANGE.fullmatch(path) is None
        for path in request_changes
    ):
        errors.append("policy_cadence_amended_packet_changes_invalid")
    relations = {
        "source_task_freeze_digest": source_freeze.get("task_freeze_digest"),
        "amended_task_freeze_digest": amended_freeze.get("task_freeze_digest"),
        "source_packet_request_digest": source_packet_request.get("request_digest"),
        "amended_packet_request_digest": amended_packet_request.get("request_digest"),
        "prompt": source_freeze.get("prompt"),
        "freeze_changed_paths": freeze_changes,
        "packet_request_changed_paths": request_changes,
    }
    errors.extend(
        f"policy_cadence_amendment_relation_invalid:{field}"
        for field, expected_value in relations.items()
        if payload.get(field) != expected_value
    )
    if (
        amended_freeze.get("prompt") != source_freeze.get("prompt")
        or (amended_packet_request.get("task_spec") or {}).get("prompt")
        != source_freeze.get("prompt")
    ):
        errors.append("policy_cadence_amendment_prompt_changed")
    if payload.get("prompt_preserved_exactly") is not True:
        errors.append("policy_cadence_amendment_prompt_proof_missing")
    if payload.get("cadence") != {
        "physics_frequency_hz": PHYSICS_FREQUENCY_HZ,
        "control_frequency_hz": TARGET_CONTROL_FREQUENCY_HZ,
        "control_decimation": TARGET_CONTROL_DECIMATION,
    }:
        errors.append("policy_cadence_amendment_cadence_invalid")
    if (
        payload.get("learned_policy_outcomes_accessed") is not False
        or payload.get("provider_mutation_performed") is not False
        or payload.get("spend_incurred_usd") != 0.0
    ):
        errors.append("policy_cadence_amendment_claim_boundary_invalid")
    try:
        validate_freeze_amendment_carry_forward_content(
            payload.get("scenario_suite_carry_forward") or {},
            sealed_schema="third_scene_task_scenario_suite.v1",
            superseded_digest=str(source_freeze.get("task_freeze_digest") or ""),
            amended_digest=str(amended_freeze.get("task_freeze_digest") or ""),
        )
    except FreezeAmendmentCarryForwardError:
        errors.append("policy_cadence_scenario_carry_forward_stale")
    if payload.get("receipt_digest") != canonical_digest(
        payload, digest_field="receipt_digest"
    ):
        errors.append("policy_cadence_amendment_receipt_digest_invalid")
    if errors:
        raise NativeTaskPolicyCadenceAmendmentError(errors)
    return payload


def materialize_native_task_policy_cadence_amendment(
    *,
    amendment_request_path: str | Path,
    task_freeze_path: str | Path,
    packet_request_path: str | Path,
    evidence_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Emit amended freeze, request, packet, and verified 15 Hz scene plan."""

    freeze_path, freeze, request_path, packet = _source_pair(
        task_freeze_path=task_freeze_path,
        packet_request_path=packet_request_path,
    )
    _, raw_authority = _read(
        amendment_request_path, error="policy_cadence_amendment_request_invalid"
    )
    authority = validate_native_task_policy_cadence_amendment_request(
        raw_authority,
        freeze_path=freeze_path,
        freeze=freeze,
        packet_path=request_path,
        packet=packet,
    )
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise NativeTaskPolicyCadenceAmendmentError(
            ["policy_cadence_amendment_output_exists"]
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    try:
        amended_freeze = json.loads(json.dumps(freeze))
        amendment_record = {
            "schema_version": "native_task_policy_cadence_amendment_record.v1",
            "field": "execution_contract.control_frequency_hz",
            "from": SOURCE_CONTROL_FREQUENCY_HZ,
            "to": TARGET_CONTROL_FREQUENCY_HZ,
            "reason": "match_the_frozen_DROID_policy_temporal_contract",
            "official_source": OFFICIAL_DROID_CADENCE_SOURCE,
            "candidate_ids": list(FROZEN_CANDIDATES),
            "authorized_by": authority["authorized_by"],
            "authorized_at": authority["authorized_at"],
            "superseded_task_freeze_digest": freeze["task_freeze_digest"],
            "source_packet_request_digest": packet["request_digest"],
            "task_semantics_changed": False,
            "learned_policy_outcomes_accessed": False,
            "provider_mutation_performed": False,
        }
        amended_freeze.setdefault("freeze_amendments", []).append(amendment_record)
        amended_freeze["execution_contract"][
            "control_frequency_hz"
        ] = TARGET_CONTROL_FREQUENCY_HZ
        amended_freeze["task_freeze_digest"] = canonical_digest(
            amended_freeze, digest_field="task_freeze_digest"
        )
        validate_task_freeze(amended_freeze)

        freeze_output = destination / "native_task_policy_15hz_freeze.v1.json"
        freeze_output.write_text(
            json.dumps(amended_freeze, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        amended_packet = json.loads(json.dumps(packet))
        amended_packet["task_freeze_digest"] = amended_freeze["task_freeze_digest"]
        amended_packet["task_spec"][
            "control_frequency_hz"
        ] = TARGET_CONTROL_FREQUENCY_HZ
        construction = amended_packet.get("construction_bindings")
        if isinstance(construction, dict):
            matching = [
                row
                for row in construction.get("bindings") or []
                if isinstance(row, dict) and row.get("task_id") == freeze["task_id"]
            ]
            if len(matching) != 1:
                raise NativeTaskPolicyCadenceAmendmentError(
                    ["policy_cadence_construction_binding_missing"]
                )
            binding = matching[0]
            binding["task_freeze_digest"] = amended_freeze["task_freeze_digest"]
            evidence = binding.get("evidence_receipts") or {}
            if not isinstance(evidence, dict) or not isinstance(
                evidence.get("task_freeze"), dict
            ):
                raise NativeTaskPolicyCadenceAmendmentError(
                    ["policy_cadence_construction_freeze_evidence_missing"]
                )
            evidence["task_freeze"] = {
                "path": str(freeze_output),
                "size_bytes": freeze_output.stat().st_size,
                "sha256": _sha256(freeze_output),
                "schema_version": "dual_task_task_freeze.v1",
                "canonical_digest": amended_freeze["task_freeze_digest"],
            }
            construction["construction_digest"] = canonical_digest(
                construction, digest_field="construction_digest"
            )
        amended_packet["request_digest"] = canonical_digest(
            amended_packet, digest_field="request_digest"
        )
        validate_native_task_arena_packet_request(amended_packet)

        request_output = (
            destination / "native_task_arena_packet_request.policy_15hz.v1.json"
        )
        request_output.write_text(
            json.dumps(amended_packet, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        packet_output = destination / "native_task_arena_packet"
        packet_receipt = materialize_native_task_arena_packet(
            request=amended_packet,
            evidence_root=evidence_root,
            output_dir=packet_output,
        )
        scene_plan_path = packet_output / "native_task_arena_scene_plan.v1.json"
        scene_plan = json.loads(scene_plan_path.read_text(encoding="utf-8"))
        expected_cadence = {
            "physics_frequency_hz": float(PHYSICS_FREQUENCY_HZ),
            "control_frequency_hz": float(TARGET_CONTROL_FREQUENCY_HZ),
            "control_decimation": TARGET_CONTROL_DECIMATION,
        }
        cadence = scene_plan.get("cadence") or {}
        if any(cadence.get(field) != value for field, value in expected_cadence.items()):
            raise NativeTaskPolicyCadenceAmendmentError(
                ["policy_cadence_scene_plan_cadence_invalid"]
            )
        if (scene_plan.get("task_spec") or {}).get("prompt") != freeze["prompt"]:
            raise NativeTaskPolicyCadenceAmendmentError(
                ["policy_cadence_scene_plan_prompt_changed"]
            )
        freeze_changes = sorted(
            path
            for path in changed_document_paths(freeze, amended_freeze)
            if path != "task_freeze_digest"
        )
        request_changes = sorted(
            path
            for path in changed_document_paths(packet, amended_packet)
            if path != "request_digest"
        )
        carry_forward = evaluate_freeze_amendment_carry_forward(
            superseded_freeze=freeze,
            amended_freeze=amended_freeze,
            sealed_schema="third_scene_task_scenario_suite.v1",
            superseded_file_sha256=_sha256(freeze_path),
            amended_file_sha256=_sha256(freeze_output),
        )
        receipt: dict[str, Any] = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "program_id": PROGRAM_ID,
            "status": "materialized",
            "task_id": freeze["task_id"],
            "candidate_ids": list(FROZEN_CANDIDATES),
            "source_task_freeze_digest": freeze["task_freeze_digest"],
            "amended_task_freeze_digest": amended_freeze["task_freeze_digest"],
            "source_packet_request_digest": packet["request_digest"],
            "amended_packet_request_digest": amended_packet["request_digest"],
            "prompt": freeze["prompt"],
            "prompt_preserved_exactly": True,
            "freeze_changed_paths": freeze_changes,
            "packet_request_changed_paths": request_changes,
            "cadence": {
                "physics_frequency_hz": PHYSICS_FREQUENCY_HZ,
                "control_frequency_hz": TARGET_CONTROL_FREQUENCY_HZ,
                "control_decimation": TARGET_CONTROL_DECIMATION,
            },
            "amendment_request_digest": authority["request_digest"],
            "amended_task_freeze": {
                "path": str(freeze_output),
                "sha256": _sha256(freeze_output),
            },
            "amended_packet_request": {
                "path": str(request_output),
                "sha256": _sha256(request_output),
            },
            "packet_receipt_digest": packet_receipt["receipt_digest"],
            "scene_plan_digest": scene_plan["plan_digest"],
            "scenario_suite_carry_forward": carry_forward,
            "learned_policy_outcomes_accessed": False,
            "provider_mutation_performed": False,
            "spend_incurred_usd": 0.0,
            "receipt_digest": "",
        }
        receipt["receipt_digest"] = canonical_digest(
            receipt, digest_field="receipt_digest"
        )
        validate_native_task_policy_cadence_amendment(
            receipt,
            source_freeze=freeze,
            source_packet_request=packet,
            amended_freeze=amended_freeze,
            amended_packet_request=amended_packet,
        )
        (destination / "native_task_policy_cadence_amendment.v1.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return receipt
    except Exception:
        shutil.rmtree(destination)
        raise


__all__ = [
    "OFFICIAL_DROID_CADENCE_SOURCE",
    "PHYSICS_FREQUENCY_HZ",
    "RECEIPT_SCHEMA_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "SOURCE_CONTROL_FREQUENCY_HZ",
    "TARGET_CONTROL_DECIMATION",
    "TARGET_CONTROL_FREQUENCY_HZ",
    "NativeTaskPolicyCadenceAmendmentError",
    "materialize_native_task_policy_cadence_amendment",
    "materialize_native_task_policy_cadence_amendment_request",
    "validate_native_task_policy_cadence_amendment",
    "validate_native_task_policy_cadence_amendment_request",
]
