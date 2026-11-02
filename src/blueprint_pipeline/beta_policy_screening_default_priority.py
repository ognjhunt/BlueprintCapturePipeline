"""The screening order used when a job has no intake ranking (owner decision D-20).

The intake ranking (``website_candidate_ranking``) orders a job's candidates when
it exists. Without one, the plan fell back to identifier order, which put
untested policies ahead of the ones that have run in production. This list names
the learned policies to try first; it orders only, removes nothing, and the
site's screening budget still decides how far down the order a plan reaches.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SCHEMA_VERSION = "beta_screening_default_priority.v1"
DEFAULT_PRIORITY_PATH = Path(__file__).resolve().parents[2] / "config" / "beta_screening_default_priority.v1.json"
_CANDIDATE_ID = re.compile(r"[a-z][a-z0-9_]{1,127}")


class DefaultPriorityError(ValueError):
    """The default order is unreadable or malformed."""


def default_screening_priority(path: Path = DEFAULT_PRIORITY_PATH) -> list[str]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DefaultPriorityError("beta_screening_default_priority_unreadable") from exc
    rows = document.get("priority") if isinstance(document, dict) else None
    if (document.get("schema_version") != SCHEMA_VERSION or not isinstance(rows, list) or not rows
            or any(not isinstance(row, dict) or not _CANDIDATE_ID.fullmatch(str(row.get("candidate_id")))
                   or not isinstance(row.get("basis"), str) or not row["basis"].strip() for row in rows)):
        raise DefaultPriorityError("beta_screening_default_priority_invalid")
    order = [row["candidate_id"] for row in rows]
    if len(set(order)) != len(order):
        raise DefaultPriorityError("beta_screening_default_priority_invalid")
    return order
