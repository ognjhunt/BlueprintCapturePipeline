"""Provider-free validation of durable cancellation evidence for admission reads."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

SCHEMA = "task_evaluation_unstarted_controls_cancellation.v1"
DIRECTORY = "cancelled-unstarted-controls"


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("unstarted_controls_evidence_unsafe")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("unstarted_controls_evidence_invalid")
    return value


def _file(path: Path) -> dict[str, Any]:
    _read(path)
    return {"path": str(path), "digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}


def validated_cancellation(directory: Path, attempt: Mapping[str, Any]) -> dict[str, Any] | None:
    path = directory / DIRECTORY / (str(attempt["attempt_id"]) + ".json")
    if not path.exists() and not path.is_symlink():
        return None
    receipt = _read(path)
    if receipt.get('schema_version') == 'task_evaluation_unmaterialized_adoption_cancellation.v1':
        validate_retirement(receipt=receipt, attempt=attempt)
        return receipt
    original = receipt.get("original_blocked_launch_receipt") or {}
    if (
        receipt.get("schema_version") != SCHEMA
        or receipt.get("status") != "cancelled_before_controls_eligibility"
        or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
        or receipt.get("attempt_digest") != attempt.get("attempt_digest")
        or receipt.get("intent_digest") != attempt.get("intent_digest")
        or receipt.get("attempt_id") != attempt.get("attempt_id")
        or receipt.get("maximum_spend_usd") != attempt.get("maximum_spend_usd")
        or receipt.get("provider") != attempt.get("provider")
        or original.get("schema_version") != "task_evaluation_launch_receipt.v1"
        or original.get("status") != "blocked"
        or original.get("receipt_digest") != cross_runtime_canonical_digest(original, digest_field="receipt_digest")
        or original.get("source_commit") != attempt.get("source_commit")
        or receipt.get("downstream_execution_eligible") is not False
        or receipt.get("provider_mutation_performed") is not False
    ):
        raise ValueError("unstarted_controls_cancellation_invalid")
    return receipt


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError("controls_autoprovision_" + code)


def validate_retirement(*, receipt: Mapping[str, Any], attempt: Mapping[str, Any]) -> None:
    ref = receipt.get('source_adoption_intent') or {}
    intent = _read(Path(str(ref.get('path') or '')))
    _require(receipt.get('schema_version') == 'task_evaluation_unmaterialized_adoption_cancellation.v1'
        and receipt.get('status') == 'cancelled_before_adoption_materialization'
        and receipt.get('receipt_digest') == canonical_digest(receipt, digest_field='receipt_digest')
        and all(receipt.get(k) == attempt.get(k) for k in ('attempt_id', 'attempt_digest', 'intent_digest', 'provider', 'maximum_spend_usd'))
        and _file(Path(ref['path'])) == ref
        and intent.get('intent_digest') == canonical_digest(intent, digest_field='intent_digest')
        and intent.get('expected_production_commit') == attempt.get('source_commit')
        and (intent.get('configuration_adoption') or {}).get('mode') == 'explicit_terminal_adoption'
        and re.fullmatch(r'[0-9a-f]{40}', str(receipt.get('superseding_release')))
        and receipt['superseding_release'] != attempt['source_commit']
        and receipt.get('materialization_started') is False
        and receipt.get('provider_mutation_performed') is False,
        'terminal_adoption_retirement_invalid')
    owner = _read(Path(intent['phases']['construction']['authorization_path']))['scene_owner_attempt']['scene_attempt_binding']
    stem = owner['attempt_id'].removesuffix('-construction')
    _require(attempt['attempt_id'] in {stem+'-'+p for p in ('construction', 'controls', 'placement')}
        and all(owner[k] == attempt[k] for k in ('intent_id', 'intent_digest', 'source_commit', 'input_digest', 'runtime_digest')),
        'terminal_adoption_retirement_owner_mismatch')


