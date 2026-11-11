"""The estimates a website scene was built on, named once, so they reach the site's result as flags.

Reconstruction steps that used to refuse now continue on a labelled estimate (2026-11-10). Each
estimate stays in its own record (levelling, mesh registration, the scene review, the view
selection); this module reads those records, gives each estimate one typed code, and keeps the list
on the base scene (``scene_estimates``). Screening planning copies the codes into its plan, and
publication v3 lists them in ``scene_feasibility.flags`` (``beta_policy_screening_publication_flags``),
where the website names each in its own sentence and withholds any derived headline band. Nothing
measured is adjusted: an estimate only adds a flag.

Codes, each ``scene_estimate:<kind>``:

- ``level``: nothing observed gave the up direction; the provider's +Y stands in, unverified.
- ``splat_view_dependence_dropped``: the levelled splat keeps its base colour only.
- ``floor``: a flat floor stands in for an uneven or missing scanned one (the levelling's
  ``fallback_floor`` or ``walked_floor``).
- ``collider_registration``: the collider's frame, or the faces kept, are an estimate.
- ``review_citations_dropped``: the scene review cited frames that could not be inputs; those
  citations were left out.
- ``surrounding_views_dropped``: surrounding views were left out to keep the frame budget.

Pure apart from reading the retained JSON records named. Records written before these estimates
existed carry none of their fields and give no code.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

__all__ = ["CODES", "FIELD", "base_scene_flags", "capture_scene_flags", "scene_estimates"]

FIELD = "scene_estimates"
LEVEL, APPEARANCE, FLOOR = "scene_estimate:level", "scene_estimate:splat_view_dependence_dropped", "scene_estimate:floor"
COLLIDER = "scene_estimate:collider_registration"
CITATIONS, SURROUNDING = "scene_estimate:review_citations_dropped", "scene_estimate:surrounding_views_dropped"
CODES = frozenset({LEVEL, APPEARANCE, FLOOR, COLLIDER, CITATIONS, SURROUNDING})


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def scene_estimates(*, levelling: Mapping[str, Any], registration: Mapping[str, Any],
                    review: Mapping[str, Any] | None = None,
                    selection: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """One row per estimate the scene rests on: its code, the record it is read from, and its evidence."""
    rows: list[dict[str, Any]] = []
    for key, code in (("level_estimate", LEVEL), ("appearance_estimate", APPEARANCE)):
        if isinstance(levelling.get(key), Mapping):
            rows.append({"code": code, "source": "levelling", "evidence": dict(levelling[key])})
    floors = {key: dict(levelling[key]) for key in ("fallback_floor", "walked_floor")
              if isinstance(levelling.get(key), Mapping)}
    if floors:
        rows.append({"code": FLOOR, "source": "levelling", "evidence": floors})
    for key in ("registration_estimate", "support_estimate"):
        if isinstance(registration.get(key), Mapping):
            rows.append({"code": COLLIDER, "source": "mesh_registration", "evidence": dict(registration[key])})
    dropped = _mapping(review).get("automation_citations_dropped")
    if isinstance(dropped, list) and dropped:
        rows.append({"code": CITATIONS, "source": "scene_review", "evidence": {"frame_ids": list(dropped)}})
    from .atlas_video_inputs import OVER_FRAME_BUDGET
    left_out = [row.get("frame_id") for row in _mapping(selection).get("diagnostics") or []
                if isinstance(row, Mapping) and row.get("decision") == OVER_FRAME_BUDGET]
    if left_out:
        rows.append({"code": SURROUNDING, "source": "view_selection", "evidence": {"frame_ids": left_out}})
    return rows


def _codes(rows: Iterable[Any]) -> set[str]:
    return {row["code"] for row in rows if isinstance(row, Mapping) and row.get("code") in CODES}


def base_scene_flags(base_scene: Mapping[str, Any]) -> list[str]:
    """The estimate codes of a base scene: its ``scene_estimates``, and the floor stand-ins of one written before them."""
    rows = base_scene.get(FIELD)
    codes = _codes(rows if isinstance(rows, list) else [])
    levelling = _mapping(base_scene.get("levelling"))
    if isinstance(levelling.get("fallback_floor"), Mapping) or isinstance(levelling.get("walked_floor"), Mapping):
        codes.add(FLOOR)
    return sorted(codes)


def capture_scene_flags(capture_root: Path) -> list[str]:
    """The estimate codes of the base scene a website capture's preparation built; none when it has none."""
    from .website_preparation_revision import workspace
    path = workspace(Path(capture_root)) / "base_scene.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() and not path.is_symlink() else {}
    except (OSError, ValueError):
        return []
    return base_scene_flags(value) if isinstance(value, Mapping) else []
