"""Append-only cumulative budget grants; historical reservations are never refunded.

Pure retained-evidence validation preserves cumulative spend/count ceilings,
per-action limits, original consent and exact historical attempt grants.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = 'task_evaluation_scene_execution_budget_extension.v1'
DIRECTORY = 'execution-budget-extensions'
ACK = 'extend-scene-execution-budget'
LIMIT_FIELDS = {'max_total_spend_usd', 'max_paid_attempts'}
# Ceiling for an explicit owner grant; original intake stays capped at 32.
# Raised 64 -> 100 on 2026-09-16 at the owner's instruction for scene 840938,
# whose contract-drift repairs consumed attempts without reaching the policy
# runs. Every other bound is unchanged: the $1000 cumulative spend ceiling
# still applies, grants must still be monotonic, and each attempt is still
# charged against the same per-attempt caps.
MAX_EXTENDED_PAID_ATTEMPTS = 100
ATTEMPT_GRANT_FIELD = 'execution_budget_extension_digest'


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError('scene_execution_budget_' + code)


def _safe(path: Path) -> None:
    _require(path.is_absolute() and '..' not in path.parts
             and not any(p.is_symlink() for p in (path, *path.parents)), 'store_unsafe')


def _limits(value: Mapping[str, Any]) -> dict[str, Any]:
    from .task_evaluation_scene_intent_contracts import _number
    spend, count = value.get('max_total_spend_usd'), value.get('max_paid_attempts')
    _require(_number(spend) and 0 < spend <= 1000
             and type(count) is int and 1 <= count <= MAX_EXTENDED_PAID_ATTEMPTS, 'limits_invalid')
    return {'max_total_spend_usd': spend, 'max_paid_attempts': count}


def _chain(directory: Path, intent: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from . import task_evaluation_scene_intent_contracts as intake
    from .task_evaluation_scene_execution_window_evidence import effective_execution_expiry
    _safe(directory)
    root = directory / DIRECTORY
    _safe(root)
    execution = intent['request']['execution']
    original = _limits(execution)
    unchanged = {k: v for k, v in execution.items() if k not in LIMIT_FIELDS}
    expiry = effective_execution_expiry(directory, intent)
    records = []
    if root.exists():
        _require(root.is_dir(), 'store_unsafe')
        for path in root.iterdir():
            _safe(path)
            _require(path.is_file() and path.suffix == '.json', 'record_invalid')
            row = intake._read(path, 'extension_digest')
            _require(set(row) == {'schema_version', 'scope', 'intent_id', 'intent_digest', 'owner',
                'authenticated_issuer', 'authorization_reference', 'original_limits', 'prior_limits',
                'limits', 'unchanged_execution_bounds', 'sequence', 'predecessor_digest',
                'issued_at_epoch', 'provider_mutation_performed', 'historical_reservations_released',
                'extension_digest'}, 'record_invalid')
            _require(row['schema_version'] == SCHEMA and row['scope'] == 'cumulative_budget_and_attempts_only'
                and row['intent_id'] == intent['intent_id'] and row['intent_digest'] == intent['intent_digest']
                and row['owner'] == intent['request']['owner']
                and row['authenticated_issuer'] == intent['authenticated_issuer']
                and row['original_limits'] == original and row['unchanged_execution_bounds'] == unchanged
                and isinstance(row['authorization_reference'], str) and bool(row['authorization_reference'].strip())
                and intake._number(row['issued_at_epoch'])
                and intent['accepted_at_epoch'] <= row['issued_at_epoch'] < expiry
                and type(row['sequence']) is int and row['sequence'] > 0
                and row['provider_mutation_performed'] is False
                and row['historical_reservations_released'] is False
                and path.name == row['extension_digest'].removeprefix('sha256:') + '.json', 'record_invalid')
            _require(isinstance(row['limits'], dict) and isinstance(row['prior_limits'], dict)
                     and set(row['limits']) == LIMIT_FIELDS and set(row['prior_limits']) == LIMIT_FIELDS, 'limits_invalid')
            _limits(row['limits'])
            records.append(row)
    current, predecessor, issued = original, intent['intent_digest'], intent['accepted_at_epoch']
    for index, row in enumerate(sorted(records, key=lambda r: r['sequence']), 1):
        limits = row['limits']
        _require(row['sequence'] == index and row['predecessor_digest'] == predecessor
            and row['prior_limits'] == current and row['issued_at_epoch'] >= issued
            and all(limits[k] >= current[k] for k in LIMIT_FIELDS)
            and any(limits[k] > current[k] for k in LIMIT_FIELDS), 'chain_invalid')
        current, predecessor, issued = limits, row['extension_digest'], row['issued_at_epoch']
    return {**current, 'extension_digest': predecessor if records else None,
            'extension_count': len(records)}, records


def effective_execution_budget(directory: Path, intent: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every grant; expose cumulative limits separately from sealed consent."""
    return _chain(directory, intent)[0]


def validate_attempt_execution_budget(directory: Path, intent: Mapping[str, Any],
                                      attempt: Mapping[str, Any]) -> None:
    """Reopen an attempt's exact grant before use, including after later amendments."""
    _, records = _chain(directory, intent)
    if ATTEMPT_GRANT_FIELD not in attempt:
        return  # Original reservations retain their original schema and full holds.
    matching = [r for r in records if r['extension_digest'] == attempt[ATTEMPT_GRANT_FIELD]]
    _require(len(matching) == 1 and attempt.get('intent_digest') == intent['intent_digest']
             and attempt.get('intent_id') == intent['intent_id']
             and attempt.get('reserved_at_epoch', -1) >= matching[0]['issued_at_epoch'], 'attempt_grant_invalid')
