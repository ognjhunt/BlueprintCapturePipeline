"""The surface an object visibly rests on, when no surface lies at its estimated bottom.

Internal test job 4 ("move the lamp", hotel-room footage, 2026-11-17) stopped in compile on
``support_surface_not_found_under_subject``. The lamp's masks had spilled onto the round table it stood
on and a few of its lifted points fell below the table, so its box reached 22 cm under the table top:
the box-fit check flagged offset and size (overlap 0.19), and its mask bounds overlapped the table's.
The search looked only the registration's residual (7.5 cm) up from that bottom, and judged the
footprint by the box's corners, three of four of which missed the round top.

This runs when ``support_under`` found no surface at the estimated bottom, so a job whose support was
found there finds the same one. It runs before the nearest-surface guess (``nearest_support_under``),
which takes no evidence that the object touches what it is put on: job 4's floor under the table did
not pass that guess's own test, and a flat floor would have taken the lamp
(``tests/test_website_resting_support``). Where the frames show no contact with any surface, that guess
decides as before. It reads the object's own points, one array per frame
(``website_task_masks.masked_view_points``):

- **How far up.** As far as the object's own estimate is uncertain: when the box-fit check
  (``website_task_placement_check``, run on the box as estimated) flags the box, the share of the box its
  masks do not cover, ``1 - median overlap``, of the box's height; else the search window. Never above
  where the object visibly rests: the top of the support the analysis named under it
  (``target_role`` ``support``), plus the search window. Down: ``NEAREST_SUPPORT_MAX_DROP_M``, as before.
- **Footprint, contact, preference.** ``website_support_geometry.resting_surface_under``: the object's
  own outline, an area-overlap threshold, contact in the frames, nearest its lowest observed points.
- **Correction.** A surface above the box's bottom: the object's points below a surface it rests on
  cannot be the object, so its bottom is raised to the surface and its observed top kept. A surface
  below: the object is moved down onto it, as ``nearest_support_under`` moves it. Either way the move is
  recorded and the support carries ``scene_estimate:support_guessed``.

Nothing found: the caller tries the nearest-surface guess, and with none keeps
``support_surface_not_found_under_subject`` (so the room package still ships). The record
(``support_search`` in the preparation) is kept either way. Estimates only.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .atlas_scene_estimates import SUPPORT_GUESSED
from .website_support_geometry import NEAREST_SUPPORT_MAX_DROP_M, resting_surface_under, search_window_m

SCHEMA_VERSION = "website_resting_support_search.v1"
GUESS_BASIS = "surface_the_object_visibly_rests_on"


def _in_frame(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _box_in_frame(low: Sequence[float], high: Sequence[float], matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])],
                       dtype=float)
    moved = _in_frame(corners, matrix)
    return moved.min(axis=0), moved.max(axis=0)


def estimate_uncertainty(*, subject_min: Sequence[float], subject_max: Sequence[float], track: Mapping[str, Any],
                         frames: Sequence[Mapping[str, Any]], matrix: np.ndarray, runtime_to_sim: np.ndarray, up: int,
                         meters_per_unit: float, placement_residual_m: float | None) -> dict[str, Any]:
    """How far the object's estimated bottom may be wrong, upward, from its own box-fit check (metres)."""
    from .website_task_placement_check import check_video_placement
    from .website_track_mask_codec import decode_track_mask

    window_m = search_window_m(placement_residual_m)
    height_m = float(subject_max[up] - subject_min[up]) * meters_per_unit
    record: dict[str, Any] = {"search_window_m": window_m, "box_height_m": height_m}
    try:
        sim_min, sim_max = _box_in_frame(subject_min, subject_max, runtime_to_sim)
        fit = check_video_placement(
            subject_min_sim=sim_min.tolist(), subject_max_sim=sim_max.tolist(), frames=frames,
            masks={row["source_frame_id"]: decode_track_mask(row) for row in track.get("observations") or []},
            source_to_runtime=matrix, runtime_to_simulator=runtime_to_sim, placement_uncertainty_m=placement_residual_m)
    except Exception as exc:  # noqa: BLE001 - an unassessed fit leaves the search window, never a stop
        record.update(box_fit={"status": "not_assessed", "reason": type(exc).__name__ + ":" + str(exc)[:200]},
                      rise_m=window_m, basis="search_window")
        return record
    overlap = (fit.get("summary") or {}).get("median_overlap")
    record["box_fit"] = {"status": fit["status"], "flags": list(fit["flags"]), "median_overlap": overlap,
                         "median_size_ratio": (fit.get("summary") or {}).get("median_size_ratio")}
    if fit["status"] == "assessed" and fit["flags"] and isinstance(overlap, float) and np.isfinite(overlap):
        # The masks cover only ``overlap`` of the box: its bottom may lie up to the rest of its height too low.
        record.update(rise_m=max(window_m, (1.0 - overlap) * height_m), basis="box_fit_masks_do_not_cover_the_box")
    else:
        record.update(rise_m=window_m, basis="search_window")
    return record


def visible_rest(*, subject_min: Sequence[float], subject_max: Sequence[float], targets: Sequence[Mapping[str, Any]],
                 matrix: np.ndarray, up: int, up_sign: int, meters_per_unit: float) -> dict[str, Any] | None:
    """The top of the support the analysis named under the object, above its estimated bottom (metres).

    The named supports (``target_role`` ``support``) whose visible bounds overlap the object's box in plan;
    of those, the one overlapping it most. None when the analysis named none there.
    """
    plan = [axis for axis in range(3) if axis != up]
    best = None
    for row in targets:
        bounds = row.get("estimated_visible_bounds")
        if row.get("target_role") != "support" or not isinstance(bounds, Mapping):
            continue
        low, high = _box_in_frame(bounds["minimum"], bounds["maximum"], matrix)
        overlap = float(np.prod([max(0.0, min(high[a], subject_max[a]) - max(low[a], subject_min[a])) for a in plan]))
        if overlap > 0 and (best is None or overlap > best[0]):
            best = (overlap, row, low, high)
    if best is None:
        return None
    _, row, low, high = best
    top = high[up] if up_sign == 1 else low[up]
    bottom = subject_min[up] if up_sign == 1 else subject_max[up]
    return {"target_id": row.get("target_id"), "label": row.get("semantic_label"),
            "top_above_lowest_observed_points_m": float((top - bottom) * up_sign * meters_per_unit),
            "basis": "named_support_visible_bounds"}


def visible_resting_support(*, collider: Any, subject_min: Sequence[float], subject_max: Sequence[float],
                            track: Mapping[str, Any], frames: Sequence[Mapping[str, Any]], matrix: Any,
                            runtime_to_sim: Any, targets: Sequence[Mapping[str, Any]], up: int, up_sign: int,
                            meters_per_unit: float, placement_residual_m: float | None
                            ) -> tuple[dict[str, Any] | None, dict[str, Any], list[float], list[float]]:
    """The support the object visibly rests on, the record, and the object's box resting on it (runtime units).

    The box comes back unchanged when no support is found, and when the support lies below (the
    caller's snap moves it down, as for every support).
    """
    from .website_task_masks import masked_view_points

    matrix, runtime_to_sim = np.asarray(matrix, dtype=float), np.asarray(runtime_to_sim, dtype=float)
    low, high = [float(v) for v in subject_min], [float(v) for v in subject_max]
    record: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "status": "not_found",
                              "searched_because": "no_surface_at_the_estimated_bottom_under_the_box"}
    try:
        views = [_in_frame(view, matrix) for view in masked_view_points(track, list(frames))]
    except ValueError as exc:
        record.update(status="not_searched", reason=str(exc))
        return None, record, low, high
    uncertainty = estimate_uncertainty(subject_min=low, subject_max=high, track=track, frames=frames, matrix=matrix,
                                       runtime_to_sim=runtime_to_sim, up=up, meters_per_unit=meters_per_unit,
                                       placement_residual_m=placement_residual_m)
    rest = visible_rest(subject_min=low, subject_max=high, targets=targets, matrix=matrix, up=up, up_sign=up_sign,
                        meters_per_unit=meters_per_unit)
    rise_m = uncertainty["rise_m"]
    if rest is not None:
        rest["bound_m"] = rest["top_above_lowest_observed_points_m"] + uncertainty["search_window_m"]
        rise_m = max(0.0, min(rise_m, rest["bound_m"]))
    record.update(estimate_uncertainty=uncertainty, visible_rest=rest,
                  range_m={"up": rise_m, "down": NEAREST_SUPPORT_MAX_DROP_M})
    try:
        support, search = resting_surface_under(collider, low, high, views=views, up=up,
                                                meters_per_unit=meters_per_unit, up_sign=up_sign,
                                                placement_residual_m=placement_residual_m, rise_m=rise_m)
    except Exception as exc:  # noqa: BLE001 - a search that cannot run keeps today's code, recorded
        record.update(status="not_searched", reason=type(exc).__name__ + ":" + str(exc)[:200])
        return None, record, low, high
    record["search"] = search
    if support is None:
        return None, record, low, high
    bottom = low[up] if up_sign == 1 else high[up]
    moved_m = float((support["top_runtime_units"] - bottom) * up_sign * meters_per_unit)
    height_before_m = float(high[up] - low[up]) * meters_per_unit
    if moved_m > 0:
        # Its points below a surface it rests on cannot be the object: the bottom rises, the observed top stays.
        (low if up_sign == 1 else high)[up] = float(support["top_runtime_units"])
        basis = "bottom_raised_to_the_surface_observed_top_kept"
    else:
        basis = "moved_down_onto_the_surface"
    correction = {"basis": basis, "bottom_moved_m": moved_m, "height_before_m": height_before_m,
                  "height_after_m": float(high[up] - low[up]) * meters_per_unit, "physical_measurement": False}
    record.update(status="found", correction=correction)
    support = {**support, "guess": {"basis": GUESS_BASIS, "flag": SUPPORT_GUESSED, "bottom_moved_m": moved_m,
                                    "correction": basis, "maximum_rise_m": rise_m,
                                    "maximum_drop_m": NEAREST_SUPPORT_MAX_DROP_M}}
    return support, record, low, high


__all__ = ["GUESS_BASIS", "SCHEMA_VERSION", "estimate_uncertainty", "visible_rest", "visible_resting_support"]
