"""The set-up flags a beta screening publication carries beside its estimate (publication v3).

Beta journey: a finished run's counts reach the site as a bounded estimate. A
count alone cannot say whether the robots or the scene Blueprint built decided
it: a run can score no success in a scene whose carry the free CPU check had
already flagged, and that flag used to stay in the plan. v3 publishes what
Blueprint's own checks recorded, as typed codes only:

- per candidate ``observation_quality_flags``: the flags the run's pre-policy
  camera gate recorded (``candidate_result_summary``);
- per candidate ``setup_quality_flags``: a humanoid (G1) row's spawn and floor
  flags (``beta_policy_screening_g1.setup_quality_flags``);
- per candidate ``scene_check_flags``: what the free CPU reach check found for
  that row at planning (``website_candidate_feasibility.scene_check``);
- per campaign ``scene_feasibility``: ``clear``, ``flagged`` or ``unknown`` over
  its rows, with their flags. Unknown stays unknown: a plan that recorded no
  verdict publishes ``unknown`` and no flag.

A list is present only when it has a code. Every code matches ``FLAG_CODE``; a
value that does not is carried as ``UNTYPED``, so no message, path or measured
number is published. Flags inform: nothing here refuses, reorders or funds.

v2 stays writable. ``as_v2`` is the same publication without these fields, with
exactly the bytes the v2 writer produced, for a website that does not accept v3
yet (``beta_policy_screening_worker.deliver_publications``).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import re
from typing import Any

from .decision_evidence_contracts import cross_runtime_canonical_digest
from .policy_episode_lifecycle import OBSERVATION_QUALITY_FLAG_UNTYPED as UNTYPED
from .website_candidate_feasibility import scene_check

PUBLICATION_SCHEMA_V2 = "beta_policy_screening_publication.v2"
PUBLICATION_SCHEMA_V3 = "beta_policy_screening_publication.v3"
#: One flag on the wire: a lowercase code with at most three typed qualifiers.
FLAG_CODE = re.compile(r"[a-z][a-z0-9_]{0,127}(?::[a-z0-9_]{1,96}){0,3}")
MAX_FLAGS = 64
ROW_FLAG_FIELDS = ("observation_quality_flags", "setup_quality_flags", "scene_check_flags")
CAMPAIGN_FLAG_FIELD = "scene_feasibility"
#: The ledger's record that a revision went out as v2 because the website did not accept its v3 form.
DOWNGRADE_FIELD = "publication_downgrade"


def typed_flags(values: Any) -> list[str]:
    """Sorted, unique typed codes; anything that is not one is kept as ``UNTYPED``, never as text."""
    if not isinstance(values, (list, tuple, set, frozenset)):
        return []
    return sorted({value if isinstance(value, str) and FLAG_CODE.fullmatch(value) else UNTYPED
                   for value in values})[:MAX_FLAGS]


def row_flags(configured: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, list[str]]:
    """The flag lists one candidate row carries: only those with a code."""
    execution = configured.get("execution_spec")
    execution = execution if isinstance(execution, Mapping) else {}
    lists = {"observation_quality_flags": summary.get("observation_quality_flags"),
             # The plan descriptor has a G1 row's flags before it runs; its summary restates them.
             "setup_quality_flags": execution.get("setup_quality_flags") or summary.get("setup_quality_flags"),
             "scene_check_flags": scene_check(configured)[1]}
    flags = {key: typed_flags(value) for key, value in lists.items()}
    return {key: value for key, value in flags.items() if value}


def scene_feasibility(configured_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The campaign's free CPU scene verdict: flagged when any row is, clear when one cleared and none flagged."""
    verdicts = [scene_check(row) for row in configured_rows]
    statuses = {status for status, _ in verdicts}
    return {"status": "flagged" if "flagged" in statuses else "clear" if "clear" in statuses else "unknown",
            "flags": typed_flags({code for _, codes in verdicts for code in codes})}


def flags_valid(value: Any, *, empty: bool = False) -> bool:
    """A sorted, unique list of typed codes within the wire's bound; empty only where the field allows it."""
    return (isinstance(value, list) and (bool(value) or empty) and len(value) <= MAX_FLAGS
            and value == sorted(set(value))
            and all(isinstance(code, str) and FLAG_CODE.fullmatch(code) for code in value))


def publication_flags_valid(publication: Mapping[str, Any]) -> bool:
    """The v3 fields of a publication, as the website reads them (WebApp ``betaPolicyScreeningContract``)."""
    verdict = publication.get(CAMPAIGN_FLAG_FIELD)
    return (isinstance(verdict, Mapping) and set(verdict) == {"status", "flags"}
            and verdict["status"] in {"clear", "flagged", "unknown"} and flags_valid(verdict["flags"], empty=True)
            and (verdict["status"] == "flagged") == bool(verdict["flags"])
            and all(flags_valid(row[key]) for row in publication.get("candidates") or []
                    if isinstance(row, Mapping) for key in ROW_FLAG_FIELDS if key in row))


def as_v2(publication: Mapping[str, Any]) -> dict[str, Any]:
    """The same result as publication v2: the v3 fields removed and the digest taken again."""
    value = {key: item for key, item in publication.items() if key != CAMPAIGN_FLAG_FIELD}
    value["schema_version"] = PUBLICATION_SCHEMA_V2
    value["candidates"] = [{key: item for key, item in row.items() if key not in ROW_FLAG_FIELDS}
                           for row in publication["candidates"]]
    value["publication_digest"] = cross_runtime_canonical_digest(value, digest_field="publication_digest")
    return value


def goes_out_as_v2(state: Mapping[str, Any], legacy_digest: str) -> bool:
    """Whether this revision's bytes are its v2 form.

    Bytes the website acknowledged stand: a revision it holds as v2 is never rewritten as v3 (it refuses
    the same revision with another digest), so a campaign published before v3 changes nothing until its
    next revision. A revision whose v3 form the website did not accept goes out as v2 for good
    (``DOWNGRADE_FIELD``); the next revision is offered as v3 again.
    """
    revision = state.get("revision")
    downgrade = state.get(DOWNGRADE_FIELD)
    if isinstance(downgrade, Mapping) and downgrade.get("revision") == revision:
        return True
    return state.get("published_revision") == revision and state.get("published_digest") == legacy_digest


__all__ = ["CAMPAIGN_FLAG_FIELD", "DOWNGRADE_FIELD", "FLAG_CODE", "MAX_FLAGS", "PUBLICATION_SCHEMA_V2",
           "PUBLICATION_SCHEMA_V3", "ROW_FLAG_FIELDS", "UNTYPED", "as_v2", "flags_valid", "goes_out_as_v2",
           "publication_flags_valid", "row_flags", "scene_feasibility", "typed_flags"]
