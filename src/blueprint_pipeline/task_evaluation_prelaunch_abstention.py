"""Seal a Task Evaluation Run blocked before external bytes may be uploaded."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .adp_task_evaluation_abstention import PROVIDER_ZERO_SCHEMA_VERSION
from .decision_evidence_contracts import canonical_digest, canonical_json

FREEZE_SCHEMA_VERSION = "task_evaluation_prelaunch_freeze.v1"
ABSTENTION_SCHEMA_VERSION = "task_evaluation_prelaunch_external_input_abstention.v1"
OWNER_AUTHORITY_SCHEMA_VERSION = "external_asset_owner_processing_authority.v1"
ABSTENTION_SUPERSESSION_SCHEMA_VERSION = "task_evaluation_prelaunch_abstention_supersession.v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,159}")


class TaskEvaluationPrelaunchAbstentionError(ValueError):
    """Stable failure at the prelaunch abstention trust boundary."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(error) for error in errors if str(error)}))
        super().__init__(";".join(self.errors))


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _read_once(path_value: str | Path, *, label: str) -> tuple[bytes, Path]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise TaskEvaluationPrelaunchAbstentionError([f"{label}_no_follow_unavailable"])
    path = Path(os.path.abspath(Path(path_value).expanduser()))
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 8 * 1024 * 1024:
            raise TaskEvaluationPrelaunchAbstentionError([f"{label}_file_invalid"])
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
        )
        if before_identity != after_identity or len(content) != before.st_size:
            raise TaskEvaluationPrelaunchAbstentionError([f"{label}_changed_while_reading"])
        return content, path
    except TaskEvaluationPrelaunchAbstentionError:
        raise
    except OSError as exc:
        raise TaskEvaluationPrelaunchAbstentionError([f"{label}_file_invalid"]) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _json_file(path: str | Path, *, label: str) -> tuple[dict[str, Any], bytes, Path]:
    content, resolved = _read_once(path, label=label)

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(content, object_pairs_hook=unique_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise TaskEvaluationPrelaunchAbstentionError([f"{label}_json_invalid"]) from exc
    if not isinstance(value, dict):
        raise TaskEvaluationPrelaunchAbstentionError([f"{label}_json_invalid"])
    return value, content, resolved


def _identifier(value: Any) -> bool:
    return isinstance(value, str) and bool(_IDENTIFIER.fullmatch(value))


def _sha(value: Any) -> bool:
    return isinstance(value, str) and bool(_DIGEST.fullmatch(value))


def materialize_task_evaluation_prelaunch_freeze(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the smallest task-neutral identity needed for a prelaunch stop."""

    try:
        raw = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TaskEvaluationPrelaunchAbstentionError(["prelaunch_freeze_invalid"]) from exc
    expected = {
        "schema_version",
        "program_id",
        "run_id",
        "scene_id",
        "task_id",
        "task_kind",
        "prompt",
        "candidate_ids",
        "cell_ids",
        "external_asset_id",
        "external_asset_archive_sha256",
        "selection_file_sha256",
        "placement_manifest_digest",
        "observation_file_sha256",
        "repository_commit",
        "claim_ceiling",
        "freeze_digest",
    }
    errors: list[str] = []
    if set(raw) != expected or raw.get("schema_version") != FREEZE_SCHEMA_VERSION:
        errors.append("prelaunch_freeze_fields_invalid")
    if any(
        not _identifier(raw.get(name))
        for name in (
            "program_id",
            "run_id",
            "scene_id",
            "task_id",
            "task_kind",
            "external_asset_id",
        )
    ):
        errors.append("prelaunch_freeze_identity_invalid")
    prompt = raw.get("prompt")
    ceiling = raw.get("claim_ceiling")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 500:
        errors.append("prelaunch_freeze_prompt_invalid")
    if not isinstance(ceiling, str) or not ceiling.strip() or len(ceiling) > 1000:
        errors.append("prelaunch_freeze_claim_ceiling_invalid")
    candidates = raw.get("candidate_ids")
    cells = raw.get("cell_ids")
    if (
        not isinstance(candidates, list)
        or len(candidates) != 2
        or len(set(candidates)) != 2
        or any(not _identifier(item) for item in candidates)
    ):
        errors.append("prelaunch_freeze_candidate_pair_invalid")
    if (
        not isinstance(cells, list)
        or not cells
        or len(cells) != len(set(cells))
        or any(not _identifier(item) for item in cells)
    ):
        errors.append("prelaunch_freeze_cells_invalid")
    for name in (
        "external_asset_archive_sha256",
        "selection_file_sha256",
        "placement_manifest_digest",
        "observation_file_sha256",
    ):
        if not _sha(raw.get(name)):
            errors.append(f"prelaunch_freeze_digest_invalid:{name}")
    if not isinstance(raw.get("repository_commit"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", raw["repository_commit"]
    ):
        errors.append("prelaunch_freeze_repository_commit_invalid")
    if errors:
        raise TaskEvaluationPrelaunchAbstentionError(errors)
    normalized = {key: raw[key] for key in expected if key != "freeze_digest"}
    normalized["candidate_ids"] = list(candidates)
    normalized["cell_ids"] = list(cells)
    normalized["freeze_digest"] = canonical_digest(normalized, digest_field="freeze_digest")
    if raw.get("freeze_digest") not in ("", normalized["freeze_digest"]):
        raise TaskEvaluationPrelaunchAbstentionError(["prelaunch_freeze_digest_mismatch"])
    return normalized


def _provider_zero(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        receipt = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TaskEvaluationPrelaunchAbstentionError(["provider_zero_invalid"]) from exc
    if (
        receipt.get("schema_version") != PROVIDER_ZERO_SCHEMA_VERSION
        or receipt.get("provider") != "vast"
        or receipt.get("api_command") != ["vastai", "show", "instances", "--raw"]
        or receipt.get("api_confirmed") is not True
        or receipt.get("global_live_resource_count") != 0
        or receipt.get("provider_zero") is not True
        or receipt.get("inventory") != []
        or receipt.get("provider_zero_digest")
        != canonical_digest(receipt, digest_field="provider_zero_digest")
    ):
        raise TaskEvaluationPrelaunchAbstentionError(["provider_zero_invalid"])
    return receipt


def materialize_external_asset_owner_processing_authority(
    value: Mapping[str, Any], *, owner_statement_path: str | Path
) -> dict[str, Any]:
    """Bind a direct owner representation to one exact asset and private provider scope.

    This admits processing of the owner's bytes.  It deliberately does not infer
    public redistribution rights, provider-native qualification, or physical truth.
    """

    statement, _ = _read_once(owner_statement_path, label="owner_statement")
    try:
        raw = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TaskEvaluationPrelaunchAbstentionError(["owner_authority_invalid"]) from exc
    expected = {
        "schema_version",
        "authority_id",
        "authority_kind",
        "authority_reference",
        "asset_id",
        "asset_archive_sha256",
        "commissioned_and_paid_by_authority",
        "authority_represents_asset_ownership_or_processing_control",
        "private_processing_authorized",
        "permitted_provider_ids",
        "public_redistribution_authorized",
        "statement_file_sha256",
        "statement_size_bytes",
        "claim_ceiling",
        "receipt_digest",
    }
    errors: list[str] = []
    if set(raw) != expected or raw.get("schema_version") != OWNER_AUTHORITY_SCHEMA_VERSION:
        errors.append("owner_authority_fields_invalid")
    for name in ("authority_id", "asset_id"):
        if not _identifier(raw.get(name)):
            errors.append(f"owner_authority_identity_invalid:{name}")
    if raw.get("authority_kind") != "direct_asset_owner_attestation":
        errors.append("owner_authority_kind_invalid")
    reference = raw.get("authority_reference")
    if not isinstance(reference, str) or not reference.strip() or len(reference) > 500:
        errors.append("owner_authority_reference_invalid")
    if not _sha(raw.get("asset_archive_sha256")):
        errors.append("owner_authority_asset_digest_invalid")
    if (
        raw.get("commissioned_and_paid_by_authority") is not True
        or raw.get("authority_represents_asset_ownership_or_processing_control") is not True
        or raw.get("private_processing_authorized") is not True
    ):
        errors.append("owner_authority_private_processing_not_granted")
    if raw.get("permitted_provider_ids") != ["vast"]:
        errors.append("owner_authority_provider_scope_invalid")
    if raw.get("public_redistribution_authorized") is not False:
        errors.append("owner_authority_public_redistribution_scope_invalid")
    if raw.get("statement_file_sha256") not in ("", _digest(statement)):
        errors.append("owner_authority_statement_digest_mismatch")
    if raw.get("statement_size_bytes") not in (0, len(statement)):
        errors.append("owner_authority_statement_size_mismatch")
    ceiling = raw.get("claim_ceiling")
    if not isinstance(ceiling, str) or not ceiling.strip() or len(ceiling) > 1000:
        errors.append("owner_authority_claim_ceiling_invalid")
    if errors:
        raise TaskEvaluationPrelaunchAbstentionError(errors)
    receipt = {key: raw[key] for key in expected if key != "receipt_digest"}
    receipt["statement_file_sha256"] = _digest(statement)
    receipt["statement_size_bytes"] = len(statement)
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    if raw.get("receipt_digest") not in ("", receipt["receipt_digest"]):
        raise TaskEvaluationPrelaunchAbstentionError(["owner_authority_receipt_digest_mismatch"])
    return receipt


def materialize_task_evaluation_prelaunch_abstention_supersession(
    *,
    freeze_path: str | Path,
    abstention_path: str | Path,
    owner_authority_path: str | Path,
) -> dict[str, Any]:
    """Supersede only a rights blocker while preserving the historical abstention."""

    freeze_raw, freeze_bytes, _ = _json_file(freeze_path, label="prelaunch_freeze")
    freeze = materialize_task_evaluation_prelaunch_freeze(freeze_raw)
    if freeze != freeze_raw:
        raise TaskEvaluationPrelaunchAbstentionError(["prelaunch_freeze_replay_mismatch"])
    abstention, abstention_bytes, _ = _json_file(abstention_path, label="abstention")
    authority, authority_bytes, _ = _json_file(owner_authority_path, label="owner_authority")
    blocker = abstention.get("smallest_missing_external_input")
    if (
        abstention.get("schema_version") != ABSTENTION_SCHEMA_VERSION
        or abstention.get("status") != "typed_evidence_backed_abstention"
        or abstention.get("receipt_digest")
        != canonical_digest(abstention, digest_field="receipt_digest")
        or abstention.get("freeze_digest") != freeze["freeze_digest"]
        or abstention.get("all_terminal_blockers") != [blocker]
        or not _identifier(blocker)
    ):
        raise TaskEvaluationPrelaunchAbstentionError(["abstention_replay_invalid"])
    if (
        authority.get("schema_version") != OWNER_AUTHORITY_SCHEMA_VERSION
        or authority.get("receipt_digest")
        != canonical_digest(authority, digest_field="receipt_digest")
        or authority.get("asset_id") != freeze["external_asset_id"]
        or authority.get("asset_archive_sha256") != freeze["external_asset_archive_sha256"]
        or authority.get("commissioned_and_paid_by_authority") is not True
        or authority.get("authority_represents_asset_ownership_or_processing_control") is not True
        or authority.get("private_processing_authorized") is not True
        or authority.get("permitted_provider_ids") != ["vast"]
        or authority.get("public_redistribution_authorized") is not False
    ):
        raise TaskEvaluationPrelaunchAbstentionError(["owner_authority_replay_invalid"])
    receipt: dict[str, Any] = {
        "schema_version": ABSTENTION_SUPERSESSION_SCHEMA_VERSION,
        "status": "historical_rights_abstention_superseded_for_private_processing",
        "freeze_digest": freeze["freeze_digest"],
        "asset_id": freeze["external_asset_id"],
        "asset_archive_sha256": freeze["external_asset_archive_sha256"],
        "superseded_blocker": blocker,
        "historical_abstention_binding": {
            "receipt_digest": abstention["receipt_digest"],
            "file_sha256": _digest(abstention_bytes),
            "size_bytes": len(abstention_bytes),
        },
        "owner_authority_binding": {
            "receipt_digest": authority["receipt_digest"],
            "file_sha256": _digest(authority_bytes),
            "size_bytes": len(authority_bytes),
            "statement_file_sha256": authority["statement_file_sha256"],
        },
        "freeze_binding": {
            "file_sha256": _digest(freeze_bytes),
            "size_bytes": len(freeze_bytes),
        },
        "private_upload_to_vast_permitted": True,
        "public_redistribution_permitted": False,
        "rights_gate_passed_for_private_vast_canary": True,
        "paid_launch_authorized_by_this_receipt": False,
        "canonical_paid_resource_admission_still_required": True,
        "native_asset_qualified": False,
        "controls_or_policy_outcomes_created": False,
        "next_action": "Run fresh provider inventory and canonical paid-resource admission for one immutable blank-stage native canary.",
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def materialize_task_evaluation_prelaunch_external_input_abstention(
    *,
    freeze_path: str | Path,
    rights_receipt_path: str | Path,
    provider_zero_receipt: Mapping[str, Any],
    output_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Seal path B when rights forbid the first provider disclosure."""

    freeze_raw, freeze_bytes, _ = _json_file(freeze_path, label="prelaunch_freeze")
    freeze = materialize_task_evaluation_prelaunch_freeze(freeze_raw)
    if freeze != freeze_raw:
        raise TaskEvaluationPrelaunchAbstentionError(["prelaunch_freeze_replay_mismatch"])
    rights, rights_bytes, _ = _json_file(rights_receipt_path, label="rights_receipt")
    source = rights.get("source_asset")
    admission = rights.get("admission")
    blocker = rights.get("typed_blocker")
    if (
        not isinstance(source, dict)
        or not isinstance(admission, dict)
        or rights.get("receipt_digest") != canonical_digest(rights, digest_field="receipt_digest")
        or rights.get("asset_id") != freeze["external_asset_id"]
        or source.get("archive_sha256") != freeze["external_asset_archive_sha256"]
        or rights.get("status") != "blocked_missing_generated_output_rights"
        or not _identifier(blocker)
        or admission.get("private_upload_to_vast_permitted") is not False
        or admission.get("redistribution_permitted") is not False
    ):
        raise TaskEvaluationPrelaunchAbstentionError(["rights_receipt_not_blocking_upload"])
    zero = _provider_zero(provider_zero_receipt)
    receipt: dict[str, Any] = {
        "schema_version": ABSTENTION_SCHEMA_VERSION,
        "status": "typed_evidence_backed_abstention",
        "program_id": freeze["program_id"],
        "run_id": freeze["run_id"],
        "scene_id": freeze["scene_id"],
        "task_id": freeze["task_id"],
        "task_kind": freeze["task_kind"],
        "freeze_digest": freeze["freeze_digest"],
        "candidate_ids": freeze["candidate_ids"],
        "cell_ids": freeze["cell_ids"],
        "smallest_missing_external_input": blocker,
        "all_terminal_blockers": [blocker],
        "freeze_binding": {"file_sha256": _digest(freeze_bytes), "size_bytes": len(freeze_bytes)},
        "rights_binding": {
            "file_sha256": _digest(rights_bytes),
            "size_bytes": len(rights_bytes),
            "receipt_digest": rights["receipt_digest"],
            "asset_id": rights["asset_id"],
            "asset_archive_sha256": source["archive_sha256"],
        },
        "provider_zero": zero,
        "provider_upload_performed": False,
        "provider_mutation_performed": False,
        "paid_gpu_attempts": 0,
        "paid_gpu_cost_usd": 0.0,
        "automatic_paid_retry_executed": False,
        "native_asset_qualified": False,
        "controls_executed": False,
        "learned_candidate_episodes_executed": False,
        "episode_media_exists": False,
        "comparison_exists": False,
        "claim_ceiling": freeze["claim_ceiling"],
        "next_action": rights.get("smallest_external_resolution"),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    if output_path is not None:
        if repo_root is None:
            raise TaskEvaluationPrelaunchAbstentionError(["output_repo_root_missing"])
        root = Path(repo_root).expanduser().resolve()
        output = Path(output_path).expanduser().resolve()
        if not output.is_relative_to(root) or output.exists() or output.is_symlink():
            raise TaskEvaluationPrelaunchAbstentionError(["output_path_invalid"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


__all__ = [
    "ABSTENTION_SUPERSESSION_SCHEMA_VERSION",
    "ABSTENTION_SCHEMA_VERSION",
    "FREEZE_SCHEMA_VERSION",
    "OWNER_AUTHORITY_SCHEMA_VERSION",
    "TaskEvaluationPrelaunchAbstentionError",
    "materialize_external_asset_owner_processing_authority",
    "materialize_task_evaluation_prelaunch_abstention_supersession",
    "materialize_task_evaluation_prelaunch_external_input_abstention",
    "materialize_task_evaluation_prelaunch_freeze",
]
