"""The surface an object visibly rests on, when no surface lies at its estimated bottom.

Internal test job 4 ("move the lamp", hotel-room footage, 2026-11-17) stopped in compile on
``support_surface_not_found_under_subject``. The lamp's masks had spilled onto the round table it stood
on and a few of its lifted points fell below the table, so its box reached 22 cm under the table top:
the box-fit check flagged offset and size (overlap 0.19), and its mask bounds overlapped the table's.
The search looked only the registration's residual (7.5 cm) up from that bottom, and judged the
footprint by the box's corners, three of four of which missed the round top.

This runs when ``support_under`` found no surface at the estimated bottom, so a job whose support was
found there finds the same one. Compile places the object through it (``rest_where_seen``) before it
flattens the surfaces objects rest on (``website_collider_conditioning``), so the surface found is
flattened under the corrected box. It runs before the nearest-surface guess (``nearest_support_under``),
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
- **Footprint, room, contact, preference.** ``website_support_geometry.resting_surface_under``: the
  object's own outline over an upward-facing surface (an area-overlap threshold), room above it, its
  lowest points over the surface at it in its frames, nearest its lowest observed points. Contact is
  read within the search window grown by the depth noise the named support's own points show about
  the surface (their spread about its plane), so noisy depth does not refuse the table it stands on.
- **Correction.** Within the search window of the box's bottom, the whole box moves onto the surface,
  its height kept, as every support snap moves it. Farther up, the box's points below a surface it
  rests on cannot be the object, so its bottom rises to the surface and its observed top stays; a
  surface at or above the observed top is refused. Below: the box moves down onto it. The move is
  recorded and the support carries ``scene_estimate:support_guessed``.

Nothing found: compile's nearest-surface guess, and with none ``support_surface_not_found_under_subject``
(so the room package still ships). The record (``support_search`` in the preparation) is kept either way.
Estimates only.
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


def _support_points(targets: Sequence[Mapping[str, Any]], target_id: Any, frames: Sequence[Mapping[str, Any]],
                    matrix: np.ndarray) -> np.ndarray | None:
    """The named support's own observed points (its masks lifted through the frames), in the runtime frame."""
    from .website_task_masks import masked_view_points

    row = next((row for row in targets if row.get("target_id") == target_id), None)
    track = None if row is None else row.get("track")
    if not isinstance(track, Mapping) or not track.get("observations"):
        return None
    try:
        return _in_frame(np.concatenate(masked_view_points(track, list(frames))), matrix)
    except (KeyError, ValueError, OSError):
        return None


def visible_resting_support(*, collider: Any, subject_min: Sequence[float], subject_max: Sequence[float],
                            track: Mapping[str, Any], frames: Sequence[Mapping[str, Any]], matrix: Any,
                            runtime_to_sim: Any, targets: Sequence[Mapping[str, Any]], up: int, up_sign: int,
                            meters_per_unit: float, placement_residual_m: float | None
                            ) -> tuple[dict[str, Any] | None, dict[str, Any], list[float], list[float]]:
    """The support the object visibly rests on, the record, and the object's box resting on it (runtime units).

    The box is new lists, corrected as the support's ``resting_correction`` says; unchanged when no
    support is found.
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
    surface_points = None
    if rest is not None:
        rest["bound_m"] = rest["top_above_lowest_observed_points_m"] + uncertainty["search_window_m"]
        rise_m = max(0.0, min(rise_m, rest["bound_m"]))
        # The support's own points: how far a point lying on it scatters in this job's depth (the contact test).
        surface_points = _support_points(targets, rest["target_id"], frames, matrix)
        rest["support_points"] = 0 if surface_points is None else int(len(surface_points))
    record.update(estimate_uncertainty=uncertainty, visible_rest=rest,
                  range_m={"up": rise_m, "down": NEAREST_SUPPORT_MAX_DROP_M})
    try:
        support, search = resting_surface_under(collider, low, high, views=views, up=up,
                                                meters_per_unit=meters_per_unit, up_sign=up_sign,
                                                placement_residual_m=placement_residual_m, rise_m=rise_m,
                                                surface_points=surface_points)
    except Exception as exc:  # noqa: BLE001 - a search that cannot run keeps today's code, recorded
        record.update(status="not_searched", reason=type(exc).__name__ + ":" + str(exc)[:200])
        return None, record, low, high
    record["search"] = search
    if support is None:
        return None, record, low, high
    correction = support["resting_correction"]
    top = float(support["top_runtime_units"])
    bottom = low[up] if up_sign == 1 else high[up]
    if correction["basis"] == "bottom_raised_to_the_surface_observed_top_kept":
        # Its points below a surface it rests on cannot be the object: the bottom rises, the observed top stays.
        (low if up_sign == 1 else high)[up] = top
    else:
        # Within the window (or down onto a surface below): the whole box moves, its height kept.
        low[up], high[up] = low[up] + top - bottom, high[up] + top - bottom
    record.update(status="found", correction=correction)
    support = {**support, "guess": {"basis": GUESS_BASIS, "flag": SUPPORT_GUESSED,
                                    "bottom_moved_m": correction["bottom_moved_m"], "correction": correction["basis"],
                                    "maximum_rise_m": rise_m, "maximum_drop_m": NEAREST_SUPPORT_MAX_DROP_M}}
    return support, record, low, high


def rest_where_seen(collider: Any, subject_min: Sequence[float], subject_max: Sequence[float],
                    source_geometry: Mapping[str, Any], task_masks: Mapping[str, Any],
                    subject_target: Mapping[str, Any], matrix: Any, runtime_to_sim: Any, up: int, up_sign: int,
                    meters_per_unit: float, placement_residual_m: float | None, record: dict[str, Any], /
                    ) -> tuple[list[float], list[float], dict[str, Any] | None]:
    """Compile's placement when no surface lies at the object's estimated bottom (``website_task_preparation``).

    The object's box resting on the surface it visibly rests on and that support, or its box unchanged
    and None. ``record`` is made this search's record (``support_search`` in the preparation). Compile
    places the object through this before it flattens the surfaces objects rest on, so the surface
    found is flattened under the corrected box; with none found, its nearest-surface guess
    (``nearest_support_under``) decides afterwards, as before.
    """
    support, found, low, high = visible_resting_support(
        collider=collider, subject_min=subject_min, subject_max=subject_max, track=subject_target["track"],
        frames=source_geometry["frames"], matrix=matrix, runtime_to_sim=runtime_to_sim, targets=task_masks["targets"],
        up=up, up_sign=up_sign, meters_per_unit=meters_per_unit, placement_residual_m=placement_residual_m)
    record.clear()
    record["support_search"] = found
    return low, high, support


__all__ = ["GUESS_BASIS", "SCHEMA_VERSION", "estimate_uncertainty", "rest_where_seen", "visible_rest",
           "visible_resting_support"]
