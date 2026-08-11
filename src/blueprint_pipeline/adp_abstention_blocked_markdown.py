"""Generate BLOCKED.md placeholders from typed ADP task abstentions.

When construction stops before controls or learned policies, the directory tree
still needs human-readable placeholders.  Those files must not drift from the
typed abstention receipt.  This module validates the abstention digest and
materializes controls/candidate BLOCKED.md files from that receipt.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:  # flat provider-bundle layout
    from decision_evidence_contracts import canonical_digest
except ModuleNotFoundError:  # repository package
    from .decision_evidence_contracts import canonical_digest


ABSTENTION_SCHEMA_VERSION = "adp_task_evaluation_run_abstention.v1"


class AbstentionBlockedMarkdownError(ValueError):
    """Fail-closed abstention placeholder generation error."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AbstentionBlockedMarkdownError("abstention_receipt_unreadable") from exc
    if not isinstance(value, dict):
        raise AbstentionBlockedMarkdownError("abstention_receipt_not_mapping")
    return value


def validate_task_abstention(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = json.loads(json.dumps(dict(value), allow_nan=False))
    errors: list[str] = []
    if receipt.get("schema_version") != ABSTENTION_SCHEMA_VERSION:
        errors.append("abstention_schema_invalid")
    if receipt.get("status") != "typed_evidence_backed_abstention":
        errors.append("abstention_status_invalid")
    if receipt.get("receipt_digest") != canonical_digest(
        receipt, digest_field="receipt_digest"
    ):
        errors.append("abstention_digest_invalid")
    if receipt.get("controls_executed") is not False:
        errors.append("abstention_controls_state_invalid")
    if receipt.get("learned_candidate_episodes_executed") is not False:
        errors.append("abstention_policy_state_invalid")
    candidates = receipt.get("candidate_ids")
    if (
        not isinstance(candidates, list)
        or not candidates
        or any(not isinstance(candidate, str) or not candidate for candidate in candidates)
        or len(candidates) != len(set(candidates))
    ):
        errors.append("abstention_candidate_ids_invalid")
    for field in (
        "task_id",
        "scene_id",
        "smallest_missing_capability",
        "task_freeze_digest",
        "gaussian_excision_freeze_digest",
    ):
        if not str(receipt.get(field) or ""):
            errors.append(f"abstention_{field}_missing")
    if not isinstance(receipt.get("historical_attempt_blockers"), list):
        errors.append("abstention_historical_blockers_invalid")
    if errors:
        raise AbstentionBlockedMarkdownError(",".join(errors))
    return receipt


def _historical_line(receipt: Mapping[str, Any]) -> str:
    historical = [str(row) for row in receipt.get("historical_attempt_blockers") or []]
    if not historical:
        return "Historical attempt blockers: none recorded."
    return "Historical attempt blockers retained as historical only: " + ", ".join(
        f"`{row}`" for row in historical
    ) + "."


def _controls_markdown(receipt: Mapping[str, Any], *, task_label: str) -> str:
    blocker = str(receipt["smallest_missing_capability"])
    return "\n".join(
        [
            "# Controls not admitted",
            "",
            f"Task: `{receipt['task_id']}` ({task_label}); scene `{receipt['scene_id']}`.",
            "",
            "No zero-action or scripted-positive episode was run. The typed "
            "abstention receipt says controls were not executed and learned "
            "candidate episodes were not executed.",
            "",
            f"Current smallest blocker: `{blocker}`.",
            _historical_line(receipt),
            "",
            f"Typed abstention digest: `{receipt['receipt_digest']}`.",
            f"Task freeze digest: `{receipt['task_freeze_digest']}`.",
            f"Gaussian excision freeze digest: `{receipt['gaussian_excision_freeze_digest']}`.",
            "",
            "Gaussian ownership/removal, replacement-depth coverage, native import, "
            "construction joins, controls, policy episodes, and media are therefore "
            "unqualified. This is a construction/admission blocker, not a control failure.",
            "",
        ]
    )


def _candidate_markdown(
    receipt: Mapping[str, Any], *, candidate_id: str, task_label: str
) -> str:
    blocker = str(receipt["smallest_missing_capability"])
    return "\n".join(
        [
            f"# {candidate_id} not admitted",
            "",
            f"Task: `{receipt['task_id']}` ({task_label}); scene `{receipt['scene_id']}`.",
            "",
            "No learned-policy episode exists for this candidate. The candidate was "
            "not launched because construction and canonical controls were not "
            "admitted for this task.",
            "",
            f"Current smallest blocker: `{blocker}`.",
            _historical_line(receipt),
            "",
            f"Typed abstention digest: `{receipt['receipt_digest']}`.",
            "",
            "This is not a policy failure, not a `never_moved` result, and not a "
            "candidate comparison. It is an unranked admission blocker.",
            "",
        ]
    )


def materialize_abstention_blocked_markdown(
    *,
    task_root: str | Path,
    abstention_receipt_path: str | Path,
    task_label: str,
    replace_existing: bool = False,
) -> dict[str, Any]:
    """Write controls/candidate ``BLOCKED.md`` files from one abstention receipt."""

    root = Path(task_root).expanduser().resolve()
    receipt_path = Path(abstention_receipt_path).expanduser()
    if not receipt_path.is_absolute():
        receipt_path = root / receipt_path
    receipt_path = receipt_path.resolve()
    try:
        receipt_path.relative_to(root)
    except ValueError as exc:
        raise AbstentionBlockedMarkdownError(
            "abstention_receipt_outside_task_root"
        ) from exc
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise AbstentionBlockedMarkdownError("abstention_receipt_missing")
    receipt = validate_task_abstention(_load_json(receipt_path))
    if not task_label:
        raise AbstentionBlockedMarkdownError("abstention_task_label_missing")
    outputs: list[dict[str, Any]] = []
    files = {
        "controls/BLOCKED.md": _controls_markdown(receipt, task_label=task_label),
    }
    for candidate_id in receipt["candidate_ids"]:
        files[f"{candidate_id}/BLOCKED.md"] = _candidate_markdown(
            receipt,
            candidate_id=candidate_id,
            task_label=task_label,
        )
    for relative_path, content in files.items():
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise AbstentionBlockedMarkdownError(
                "blocked_markdown_output_outside_task_root"
            ) from exc
        if path.is_symlink():
            raise AbstentionBlockedMarkdownError("blocked_markdown_symlink")
        if path.exists() and not replace_existing:
            raise AbstentionBlockedMarkdownError("blocked_markdown_exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        outputs.append({"relative_path": relative_path, "size_bytes": path.stat().st_size})
    return {
        "schema_version": "adp_abstention_blocked_markdown_materialization.v1",
        "status": "materialized",
        "task_id": receipt["task_id"],
        "scene_id": receipt["scene_id"],
        "candidate_ids": list(receipt["candidate_ids"]),
        "current_smallest_missing_capability": receipt["smallest_missing_capability"],
        "abstention_receipt_digest": receipt["receipt_digest"],
        "outputs": outputs,
    }


__all__ = [
    "AbstentionBlockedMarkdownError",
    "materialize_abstention_blocked_markdown",
    "validate_task_abstention",
]

