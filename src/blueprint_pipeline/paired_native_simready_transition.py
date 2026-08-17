"""Bind one SimReady probe to a terminal paired-native predecessor.

The paired-native lane is the first native Isaac consumer of the replacement
asset.  A later SimReady probe must not be built from a similarly named asset
or from another scene's retained profile.  This module reopens the exact
paired-native bundle, request, terminal result, and per-candidate native
readback, checks their internal and byte identities, then emits one compact
binding for the downstream probe spec.

It performs no provider mutation and makes no new simulator claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "paired_native_simready_predecessor_binding.v1"


class PairedNativeSimReadyTransitionError(ValueError):
    """Stable, sorted transition failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, *, error: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PairedNativeSimReadyTransitionError([error])
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedNativeSimReadyTransitionError([error]) from exc
    if not isinstance(value, Mapping):
        raise PairedNativeSimReadyTransitionError([error])
    return dict(value)


def _record(path: Path, *, identity: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if identity is not None:
        record["internal_digest"] = identity
    return record


def bind_paired_native_simready_predecessor(
    *,
    scene_id: str,
    task_id: str,
    asset_id: str,
    candidate_usd_path: str | Path,
    paired_bundle_receipt_path: str | Path,
    paired_request_path: str | Path,
    paired_terminal_result_path: str | Path,
    paired_runtime_result_path: str | Path,
    paired_candidate_probe_path: str | Path,
) -> dict[str, Any]:
    """Verify and bind the exact paired-native predecessor for one candidate."""

    normalized_scene_id = str(scene_id or "").strip()
    normalized_task_id = str(task_id or "").strip()
    normalized_asset_id = str(asset_id or "").strip()
    candidate = Path(candidate_usd_path).expanduser().resolve()
    receipt_path = Path(paired_bundle_receipt_path).expanduser().resolve()
    request_path = Path(paired_request_path).expanduser().resolve()
    terminal_path = Path(paired_terminal_result_path).expanduser().resolve()
    result_path = Path(paired_runtime_result_path).expanduser().resolve()
    probe_path = Path(paired_candidate_probe_path).expanduser().resolve()
    if (
        not normalized_scene_id
        or Path(normalized_scene_id).name != normalized_scene_id
        or not normalized_task_id
        or not normalized_asset_id
        or candidate.is_symlink()
        or not candidate.is_file()
    ):
        raise PairedNativeSimReadyTransitionError(
            ["paired_native_simready_candidate_identity_invalid"]
        )

    receipt = _read(
        receipt_path, error="paired_native_simready_bundle_receipt_invalid"
    )
    request = _read(request_path, error="paired_native_simready_request_invalid")
    terminal = _read(
        terminal_path, error="paired_native_simready_terminal_result_invalid"
    )
    result = _read(result_path, error="paired_native_simready_result_invalid")
    probe = _read(probe_path, error="paired_native_simready_probe_invalid")
    errors: list[str] = []

    if (
        receipt.get("schema_version")
        != "paired_target_native_import_provider_bundle.v1"
        or receipt.get("status") != "ready"
        or receipt.get("receipt_digest")
        != canonical_digest(receipt, digest_field="receipt_digest")
    ):
        errors.append("paired_native_simready_bundle_receipt_invalid")
    if (
        request.get("schema_version") != "paired_target_native_import_request.v1"
        or request.get("scene_id") != normalized_scene_id
        or request.get("request_digest")
        != canonical_digest(request, digest_field="request_digest")
        or request.get("request_digest") != receipt.get("request_digest")
    ):
        errors.append("paired_native_simready_request_binding_invalid")
    source_render = request.get("source_native_render_request") or {}
    request_input_rows = [
        row
        for row in receipt.get("input_files") or []
        if isinstance(row, Mapping)
        and Path(str(row.get("relative_path") or "")).name
        == "paired_target_native_import_request.v1.json"
    ]
    if (
        receipt.get("source_request_digest")
        != source_render.get("receipt_digest")
        or len(request_input_rows) != 1
        or request_input_rows[0].get("size_bytes") != request_path.stat().st_size
        or request_input_rows[0].get("sha256") != _sha256(request_path)
    ):
        errors.append("paired_native_simready_request_file_binding_invalid")
    if (
        result.get("schema_version")
        != "paired_target_native_import_runtime_result.v1"
        or result.get("status") != "completed"
        or result.get("blockers") != []
        or result.get("scene_id") != normalized_scene_id
        or result.get("all_replacements_import_qualified") is not True
        or result.get("request_digest") != request.get("request_digest")
        or result.get("result_digest")
        != canonical_digest(result, digest_field="result_digest")
    ):
        errors.append("paired_native_simready_result_binding_invalid")
    terminal_native_result = Path(
        str(terminal.get("native_result_path") or "")
    ).expanduser().resolve()
    if (
        terminal.get("schema_version") != "paired_target_native_import_vast_run.v1"
        or terminal.get("status") != "completed"
        or terminal.get("blockers") != []
        or terminal.get("bundle_sha256") != receipt.get("bundle_sha256")
        or terminal.get("request_digest") != request.get("request_digest")
        or terminal.get("replacement_count") != result.get("replacement_count")
        or terminal_native_result != result_path
        or terminal.get("continuing_spend_from_this_run") is not False
        or not str(terminal.get("teardown_manifest_path") or "").strip()
        or not str(terminal.get("artifact_manifest_path") or "").strip()
    ):
        errors.append("paired_native_simready_terminal_result_binding_invalid")
    if (
        probe.get("schema_version")
        != "simready_replacement_native_import_probe_result.v1"
        or probe.get("status") != "completed"
        or probe.get("blockers") != []
        or probe.get("asset_id") != normalized_asset_id
        or probe.get("native_isaac_executed") is not True
        or probe.get("native_simulator_import_qualified") is not True
        or probe.get("physical_equivalence_claimed") is not False
        or probe.get("result_digest")
        != canonical_digest(probe, digest_field="result_digest")
    ):
        errors.append("paired_native_simready_probe_binding_invalid")
    probe_readback = probe.get("native_readback") or {}
    probe_registration_digest = probe.get(
        "asset_frame_registration_digest",
        probe_readback.get("asset_frame_registration_digest"),
    )

    candidate_sha256 = _sha256(candidate)
    request_replacements = [
        row
        for row in request.get("replacements") or []
        if isinstance(row, Mapping)
        and row.get("task_id") == normalized_task_id
        and row.get("asset_id") == normalized_asset_id
    ]
    receipt_replacements = [
        row
        for row in receipt.get("replacements") or []
        if isinstance(row, Mapping)
        and row.get("task_id") == normalized_task_id
        and row.get("asset_id") == normalized_asset_id
    ]
    result_replacements = [
        row
        for row in result.get("replacements") or []
        if isinstance(row, Mapping)
        and row.get("task_id") == normalized_task_id
        and row.get("asset_id") == normalized_asset_id
    ]
    if not (
        len(request_replacements)
        == len(receipt_replacements)
        == len(result_replacements)
        == 1
    ):
        errors.append("paired_native_simready_replacement_identity_invalid")
    else:
        request_row = dict(request_replacements[0])
        receipt_row = dict(receipt_replacements[0])
        result_row = dict(result_replacements[0])
        stable_fields = (
            "task_id",
            "asset_id",
            "size_bytes",
            "sha256",
            "asset_frame_registration_digest",
            "registered_static_qualification_digest",
        )
        if any(
            request_row.get(field) != receipt_row.get(field)
            for field in stable_fields
        ):
            errors.append("paired_native_simready_bundle_request_candidate_mismatch")
        if (
            candidate.stat().st_size != receipt_row.get("size_bytes")
            or candidate_sha256 != receipt_row.get("sha256")
            or probe.get("replacement_asset_sha256") != candidate_sha256
            or probe_registration_digest
            != receipt_row.get("asset_frame_registration_digest")
            or probe.get("registered_static_qualification_digest")
            != receipt_row.get("registered_static_qualification_digest")
        ):
            errors.append("paired_native_simready_candidate_digest_mismatch")
        expected_probe = (
            result_path.parent / str(result_row.get("probe_result_path") or "")
        ).resolve()
        if (
            expected_probe != probe_path
            or result_path.parent not in probe_path.parents
            or probe_path.stat().st_size <= 0
            or _sha256(probe_path) != result_row.get("probe_result_sha256")
            or probe.get("result_digest") != result_row.get("probe_result_digest")
            or result_row.get("native_simulator_import_qualified") is not True
            or result_row.get("blockers") != []
        ):
            errors.append("paired_native_simready_result_probe_mismatch")
    if errors:
        raise PairedNativeSimReadyTransitionError(errors)

    binding: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scene_id": normalized_scene_id,
        "task_id": normalized_task_id,
        "asset_id": normalized_asset_id,
        "candidate_usd_sha256": candidate_sha256,
        "asset_frame_registration_digest": probe_registration_digest,
        "registered_static_qualification_digest": probe[
            "registered_static_qualification_digest"
        ],
        "paired_request_digest": request["request_digest"],
        "paired_bundle_sha256": receipt.get("bundle_sha256"),
        "source_native_render_request_digest": source_render.get("receipt_digest"),
        "bundle_receipt": _record(
            receipt_path, identity=str(receipt.get("receipt_digest") or "")
        ),
        "request": _record(
            request_path, identity=str(request.get("request_digest") or "")
        ),
        "terminal_result": _record(terminal_path),
        "runtime_result": _record(
            result_path, identity=str(result.get("result_digest") or "")
        ),
        "candidate_probe": _record(
            probe_path, identity=str(probe.get("result_digest") or "")
        ),
        "claim_boundary": {
            "paired_native_import_is_predecessor_not_simready_result": True,
            "simready_native_execution_still_required": True,
            "physical_equivalence_proven": False,
        },
        "binding_digest": "",
    }
    binding["binding_digest"] = canonical_digest(
        binding, digest_field="binding_digest"
    )
    return binding


def materialize_paired_native_simready_probe(
    *,
    scene_id: str,
    task_id: str,
    asset_id: str,
    candidate_usd_path: str | Path,
    paired_bundle_receipt_path: str | Path,
    paired_request_path: str | Path,
    paired_terminal_result_path: str | Path,
    paired_runtime_result_path: str | Path,
    paired_candidate_probe_path: str | Path,
    destination: str | Path,
    task_joint_prim_path: str,
    locked_joint_prim_paths: Sequence[str],
    commanded_sweep_degrees: Sequence[float],
    reset_joint_positions_rad: Mapping[str, float],
    locked_joint_motion_tolerance_rad: float,
    settle_samples: int,
    control_frequency_hz: float,
    validation_mode: str = "commanded_articulation",
    probe_drive_stiffness: float = 0.0,
    probe_drive_damping: float = 0.0,
    probe_drive_max_force: float = 0.0,
    fixed_step_seconds: float = 1.0 / 120.0,
) -> dict[str, Any]:
    """Build the frozen probe only after its paired predecessor is exact."""

    from .articulated_native_probe import materialize_articulated_native_probe

    binding = bind_paired_native_simready_predecessor(
        scene_id=scene_id,
        task_id=task_id,
        asset_id=asset_id,
        candidate_usd_path=candidate_usd_path,
        paired_bundle_receipt_path=paired_bundle_receipt_path,
        paired_request_path=paired_request_path,
        paired_terminal_result_path=paired_terminal_result_path,
        paired_runtime_result_path=paired_runtime_result_path,
        paired_candidate_probe_path=paired_candidate_probe_path,
    )
    return materialize_articulated_native_probe(
        candidate_usd_path=candidate_usd_path,
        destination=destination,
        task_joint_prim_path=task_joint_prim_path,
        locked_joint_prim_paths=locked_joint_prim_paths,
        commanded_sweep_degrees=commanded_sweep_degrees,
        reset_joint_positions_rad=reset_joint_positions_rad,
        locked_joint_motion_tolerance_rad=locked_joint_motion_tolerance_rad,
        settle_samples=settle_samples,
        control_frequency_hz=control_frequency_hz,
        probe_drive_stiffness=probe_drive_stiffness,
        probe_drive_damping=probe_drive_damping,
        probe_drive_max_force=probe_drive_max_force,
        fixed_step_seconds=fixed_step_seconds,
        validation_mode=validation_mode,
        scene_id=scene_id,
        paired_native_predecessor=binding,
    )


__all__ = [
    "PairedNativeSimReadyTransitionError",
    "SCHEMA_VERSION",
    "bind_paired_native_simready_predecessor",
    "materialize_paired_native_simready_probe",
]
