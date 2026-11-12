"""Show the site where its object will be moved, and let it pick another place: optional, never a wait.

Part 4d (2026-11-12). At the end of preparation, one picture from the site's own video: a frame where the
object and the destination are both in view, the object outlined at its start (orange) and the scored
destination area drawn (green), projected through the registration and clipped to the real surface it
rests on. The picture is private like every other frame: it is kept beside the target-choice records
(``website_task_target_choice.QUESTION_DIR``) and only the website's signed reader may deliver it.

Beside it, one optional question, "We will test moving the {object} from here to here.", with 2 to 4
places the site may pick instead when "the place is wrong": the place in use, the other supported ends
of the same surface and the farthest supported spot from the start. The question has the same fields as
the object question, so the existing answer read (``fetch_answer``, ``validate_answer``) and the
existing bounded correction replay (``MAX_CORRECTION_REPLAYS``) serve it; its own schema keeps it out
of the object corrections. The job never waits on an answer; an answer re-runs the preparation once on
the site's place, which is then the destination and the scored area.

Simulator frame (metres, Z up). Nothing here buys anything.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .website_surface_target_admission import WEBSITE_TARGET_TOLERANCE_M

QUESTION_SCHEMA = "website_task_destination_choice_question.v1"
PICTURE_SCHEMA = "website_task_destination_picture.v1"
TARGET_ID = "destination.place"
PICTURE_FILE = "destination.place.png"
PICTURE_RECORD = "destination.place.picture.json"
#: The website reads at most this many bytes (``websiteDestinationPicture.MAX_DESTINATION_PICTURE_BYTES``).
MAX_PICTURE_BYTES = 8 * 1024 * 1024
#: The picture's longest side; a frame is never sent larger.
PICTURE_LONG_SIDE = 1600
SITE_CHOICE = "site_choice"
MAX_PLACES = 4
#: Two places nearer than this are the same place to a site.
DISTINCT_PLACE_M = 0.10
_LABELS = {"named_region": "The place your job names",
           "start_surface_farthest_spot": "The spot on the same surface farthest from where it starts",
           "grounded_centre": "The place we found from your job's words",
           SITE_CHOICE: "The place you picked",
           "surface_end": "The other end of the same surface",
           "farthest_spot": "The spot on the same surface farthest from where it starts"}


def _distance_to_box(point: Sequence[float], low: Sequence[float], high: Sequence[float]) -> float:
    dx = max(low[0] - point[0], 0.0, point[0] - high[0])
    dy = max(low[1] - point[1], 0.0, point[1] - high[1])
    return math.hypot(dx, dy)


def place_candidates(*, choice: Mapping[str, Any], subject_min: Sequence[float], subject_max: Sequence[float],
                     support_min: Sequence[float], support_max: Sequence[float],
                     holds: Callable[[Sequence[float], float], bool] | None = None) -> list[dict[str, Any]]:
    """The place in use first, then up to three other supported places on the same surface.

    Another place keeps the object wholly on the support (the surface shrunk by half the object's longer
    side), off its start footprint, and is scored as today's 5 cm disc: a place the site picks is a spot,
    not a named area.
    """
    half = max(subject_max[0] - subject_min[0], subject_max[1] - subject_min[1]) / 2.0
    low = [support_min[i] + half for i in (0, 1)]
    high = [support_max[i] - half for i in (0, 1)]
    rows = [{"kind": str(choice["basis"]), "centre_xy": [float(v) for v in choice["position_world_m"][:2]],
             "radius_m": float(choice["area_radius_m"])}]
    if low[0] <= high[0] and low[1] <= high[1]:
        start = [(subject_min[i] + subject_max[i]) / 2.0 for i in (0, 1)]
        along = 0 if high[0] - low[0] >= high[1] - low[1] else 1
        across = 1 - along
        middle = (low[across] + high[across]) / 2.0
        found = []
        for end in (low[along], high[along]):
            centre = [0.0, 0.0]
            centre[along], centre[across] = end, middle
            found.append(("surface_end", centre))
        corners = [[x, y] for x in (low[0], high[0]) for y in (low[1], high[1])]
        found.append(("farthest_spot", max(corners, key=lambda corner: math.dist(corner, start))))
        for kind, centre in found:
            if (_distance_to_box(centre, subject_min, subject_max) <= WEBSITE_TARGET_TOLERANCE_M
                    or (holds is not None and not holds(centre, WEBSITE_TARGET_TOLERANCE_M))
                    or any(math.dist(centre, row["centre_xy"]) <= max(DISTINCT_PLACE_M, row["radius_m"],
                                                                      WEBSITE_TARGET_TOLERANCE_M)
                           for row in rows)):  # never a place whose centre lies in another's area
                continue
            rows.append({"kind": kind, "centre_xy": [float(v) for v in centre], "radius_m": WEBSITE_TARGET_TOLERANCE_M})
    for row in rows:
        row["candidate_id"] = "p" + canonical_digest({key: row[key] for key in ("centre_xy", "radius_m")})[7:23]
    return rows[:MAX_PLACES]


def _project(points, frame, sim_to_source):
    import numpy as np

    points = np.asarray(points, dtype=float)
    source = np.c_[points, np.ones(len(points))] @ np.asarray(sim_to_source, dtype=float).T
    camera_from_world = np.asarray(frame["camera_from_world"], dtype=float)
    camera = source[:, :3] @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
    k = np.asarray(frame["intrinsics"], dtype=float)
    depth = camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = np.c_[camera[:, 0] / depth * k[0, 0] + k[0, 2], camera[:, 1] / depth * k[1, 1] + k[1, 2]]
    return pixels, depth


def area_outline(centre: Sequence[float], radius: float, height: float, *,
                 covers: Callable[[Any], Any] | None = None, rays: int = 48) -> list[list[float]]:
    """The drawn area: the scored disc, each ray cut where the real surface ends (``covers``)."""
    import numpy as np

    outline = []
    for angle in np.linspace(0.0, 2.0 * math.pi, rays, endpoint=False):
        direction = np.array([math.cos(angle), math.sin(angle)])
        reach = float(radius)
        if covers is not None:
            steps = np.linspace(0.0, radius, 25)[1:]
            covered = covers(np.asarray(centre[:2])[None, :] + steps[:, None] * direction[None, :])
            uncovered = np.flatnonzero(~np.asarray(covered, dtype=bool))
            if len(uncovered):
                reach = float(steps[uncovered[0] - 1]) if uncovered[0] > 0 else 0.0
        outline.append([float(centre[0] + reach * direction[0]), float(centre[1] + reach * direction[1]), float(height)])
    return outline


LETTERS = "ABCD"


def draw_destination_picture(*, frames: Sequence[Mapping[str, Any]], sim_to_source: Any,
                             subject_min: Sequence[float], subject_max: Sequence[float],
                             places: Sequence[Sequence[Sequence[float]]], output: Path) -> dict[str, Any] | None:
    """Draw the object's start and EVERY place on the frame that shows the start and the first place best.

    The object is outlined in orange; each place is drawn with its letter (A, B, C, D, in the order of
    ``places``), the first, the place in use, in thick green, the others in thinner blue, so two places of
    the same kind (both ends of one surface) are told apart by their letters. A place not wholly in view
    in that frame is not drawn: ``drawn`` lists the indices that were, and only those may be offered.
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    top = float(subject_max[2])
    box = [[subject_min[0], subject_min[1], top], [subject_max[0], subject_min[1], top],
           [subject_max[0], subject_max[1], top], [subject_min[0], subject_max[1], top]]
    best = None
    for frame in frames:
        try:
            width, height = (int(v) for v in frame.get("image_size") or Image.open(frame["image_path"]).size)
            projected = [_project(points, frame, sim_to_source) for points in (box, *places)]
        except (KeyError, OSError, TypeError, ValueError):
            continue
        visible = [bool(((depth > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
                         & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)).all()) for pixels, depth in projected]
        if not visible[0] or not visible[1]:
            continue  # the object's start and the place in use must both be in view
        both = np.vstack([projected[0][0], projected[1][0]])
        spread = float(np.ptp(both[:, 0]) * np.ptp(both[:, 1])) / (width * height)
        key = (sum(visible[1:]), min(spread, 0.5))
        if best is None or key > best[0]:
            best = (key, frame, projected, visible)
    if best is None:
        return None
    _, frame, projected, visible = best
    with Image.open(frame["image_path"]) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=max(18, image.size[1] // 24))
    except TypeError:  # an older Pillow: its one default size
        font = ImageFont.load_default()
    draw.polygon([tuple(map(float, p)) for p in projected[0][0]], outline=(255, 140, 0), width=4)
    drawn = []
    for index, (pixels, _) in enumerate(projected[1:]):
        if not visible[index + 1] or len(drawn) >= len(LETTERS):
            continue
        letter = LETTERS[len(drawn)]
        colour, width_px = ((0, 200, 0), 6) if index == 0 else ((40, 120, 255), 3)
        draw.polygon([tuple(map(float, p)) for p in pixels], outline=colour, width=width_px)
        x, y = (float(v) for v in pixels.mean(axis=0))
        draw.text((x, y), letter, fill=colour, font=font, anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0))
        drawn.append(index)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.thumbnail((PICTURE_LONG_SIDE, PICTURE_LONG_SIDE))
    image.save(output, format="PNG", optimize=True)
    if output.stat().st_size > MAX_PICTURE_BYTES:
        output.unlink()
        return None
    return {"frame_id": str(frame.get("frame_id"))[:128],
            "reference_time_ms": (max(0, round(float(frame["timestamp_seconds"]) * 1000))
                                  if frame.get("timestamp_seconds") is not None else None),
            "png_sha256": "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest(), "drawn": drawn}


def build_place_question(*, preparation: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]],
                         object_label: str, scene_id: str, capture_id: str, source_geometry: Mapping[str, Any],
                         picture: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The optional question about the place; None with fewer than two places, or without its picture."""
    if len(candidates) < 2 or picture is None:
        return None
    label = (str(object_label).strip() or "object")[:60]
    shown = [{"candidate_id": row["candidate_id"], "label": _LABELS.get(row["kind"], "Another place")[:120],
              "reference_time_ms": (picture or {}).get("reference_time_ms"),
              "centre_xy": list(row["centre_xy"]), "radius_m": row["radius_m"], "place_kind": row["kind"]}
             for row in candidates]
    question = {"schema_version": QUESTION_SCHEMA, "kind": "task_destination_choice", "scene_id": scene_id,
                "capture_id": capture_id, "target_id": TARGET_ID, "semantic_label": label,
                "sentence": f"We will test moving the {label.lower()} from here to here.",
                "source_video_digest": (source_geometry.get("binding") or {}).get("source_video_digest"),
                "source_geometry_digest": source_geometry.get("digest"),
                "preparation_binding": canonical_digest((preparation.get("binding") or {})),
                "candidate_set_digest": canonical_digest({"places": [{key: row[key] for key in (
                    "candidate_id", "centre_xy", "radius_m")} for row in shown]}),
                "guessed_candidate_id": shown[0]["candidate_id"], "candidates": shown,
                **({"picture": dict(picture)} if picture is not None else {})}
    question["question_digest"] = canonical_digest(question, digest_field="question_digest")
    return question


def pipeline_of(output_root: Path) -> Path | None:
    """The capture's ``pipeline`` directory above a preparation workspace (a correction's revision too)."""
    for parent in Path(output_root).parents:
        if parent.name == "pipeline" and parent.parent.parent.name == "captures":
            return parent
    return None


def _folder(pipeline_root: Path) -> Path:
    from .website_task_target_choice import QUESTION_DIR
    return Path(pipeline_root) / QUESTION_DIR


def read_place_question(pipeline_root: Path) -> dict[str, Any] | None:
    import json

    path = _folder(pipeline_root) / f"{TARGET_ID}.question.json"
    try:
        if path.is_symlink() or not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    shown = {row.get("candidate_id") for row in value.get("candidates") or [] if isinstance(row, Mapping)}
    if (not isinstance(value, dict) or value.get("schema_version") != QUESTION_SCHEMA
            or value.get("question_digest") != canonical_digest(value, digest_field="question_digest")
            or value.get("target_id") != TARGET_ID or value.get("capture_id") != Path(pipeline_root).parent.name
            or value.get("guessed_candidate_id") not in shown):
        return None
    return value


def place_choice(pipeline_root: Path) -> dict[str, Any] | None:
    """The site's answered place, when it picked another one; else None (the job keeps its own)."""
    from .website_task_target_choice import read_answer

    question = read_place_question(pipeline_root)
    answer = read_answer(Path(pipeline_root), question) if question is not None else None
    if answer is None or answer["choice"]["kind"] != "other_candidate":
        return None
    row = next(row for row in question["candidates"] if row["candidate_id"] == answer["choice"]["candidate_id"])
    return {"question_digest": question["question_digest"], "candidate_id": row["candidate_id"],
            "centre_xy": list(row["centre_xy"]), "radius_m": float(row["radius_m"]), "place_kind": row["place_kind"]}


def place_correction(pipeline_root: Path) -> dict[str, str] | None:
    """What binds the preparation stage to the site's place, so its answer re-runs the stage once."""
    choice = place_choice(pipeline_root)
    return None if choice is None else {"target_id": TARGET_ID, "question_digest": choice["question_digest"],
                                        "candidate_id": choice["candidate_id"]}


def apply_site_place(pipeline_root: Path, destination: Mapping[str, Any], choice: Mapping[str, Any]
                     ) -> tuple[dict[str, Any], dict[str, Any]]:
    """The destination and its record with the site's picked place, when it picked one; else unchanged."""
    picked = place_choice(pipeline_root)
    if picked is None:
        return dict(destination), dict(choice)
    sealed = {key: value for key, value in destination.items() if key != "radius_m"}
    sealed["position_world_m"] = [*picked["centre_xy"], float(destination["position_world_m"][2])]
    if picked["radius_m"] > WEBSITE_TARGET_TOLERANCE_M:
        sealed["radius_m"] = picked["radius_m"]
    start = choice["start_centre_world_m"]
    distance = math.dist(start, picked["centre_xy"])
    record = {**choice, "basis": SITE_CHOICE, "position_world_m": list(sealed["position_world_m"]),
              "area_radius_m": picked["radius_m"], "distance_from_start_m": float(distance),
              "direction_from_start_xy": ([float((picked["centre_xy"][i] - start[i]) / distance) for i in (0, 1)]
                                          if distance > 0 else [0.0, 0.0]),
              # The site's own word: no longer a guess, so it carries no flag.
              "flags": [], "site_choice": {key: picked[key] for key in ("question_digest", "candidate_id", "place_kind")},
              "replaced_choice": {key: choice.get(key) for key in ("basis", "position_world_m", "area_radius_m")}}
    record.pop("area", None)
    return sealed, record


def retain_place_view(*, pipeline_root: Path, preparation: Mapping[str, Any], frames: Sequence[Mapping[str, Any]],
                      source_geometry: Mapping[str, Any], object_label: str,
                      covers: Callable[[Any], Any] | None = None) -> dict[str, Any] | None:
    """Write the picture and the optional question beside the target-choice records. Never raises.

    A question the site already answered is kept as it is, so its answer keeps binding the replay.
    """
    import numpy as np

    from .common import write_json
    try:
        choice = preparation.get("destination_choice") or {}
        candidates = choice.get("candidates") or []
        frame = preparation["coordinate_frame"]
        sim_to_source = np.linalg.inv(np.asarray(frame["runtime_to_simulator"], dtype=float)
                                      @ np.asarray(preparation["registration"]["source_to_runtime"], dtype=float))
        envelope = preparation["authoring_inputs"]["metric_envelope"]
        destination = preparation["destination"]
        height = float(destination["position_world_m"][2])
        places = [area_outline(row["centre_xy"], float(row["radius_m"]), height,
                               covers=covers if index == 0 else None)
                  for index, row in enumerate(candidates)]
        folder = _folder(pipeline_root)
        existing = read_place_question(pipeline_root)
        from .website_task_target_choice import read_answer
        if existing is not None and read_answer(Path(pipeline_root), existing) is not None:
            return existing  # answered: kept as it is, with its own picture, so the answer keeps binding
        question_path = folder / f"{TARGET_ID}.question.json"
        for stale in (question_path, folder / PICTURE_RECORD, folder / PICTURE_FILE):
            stale.unlink(missing_ok=True)  # a replaced question is no longer current, nor its picture
        from .website_preparation_outcome import website_lists_place_question
        if len(candidates) < 2 or not website_lists_place_question():
            return None  # switch (iii) off: no picture and no question are written
        picture = draw_destination_picture(frames=frames, sim_to_source=sim_to_source,
                                           subject_min=envelope["minimum_xyz_m"], subject_max=envelope["maximum_xyz_m"],
                                           places=places, output=folder / PICTURE_FILE)
        # Only the places drawn in the picture are offered, in its letters' order; no picture, no question.
        offered = [candidates[index] for index in (picture or {}).get("drawn", [])]
        if picture is None or len(offered) < 2 or offered[0] is not candidates[0]:
            (folder / PICTURE_FILE).unlink(missing_ok=True)
            return None
        record = {"schema_version": PICTURE_SCHEMA, "path": PICTURE_FILE,
                  **{key: picture[key] for key in ("frame_id", "reference_time_ms", "png_sha256")},
                  "letters": {LETTERS[position]: row["candidate_id"] for position, row in enumerate(offered)},
                  "drawn": {"object_start": "orange outline", "place_in_use": "thick green outline, letter A",
                            "other_places": "blue outlines, letters B to D"},
                  "preparation_digest": preparation.get("digest"), "private": True}
        write_json(folder / PICTURE_RECORD, record)
        question = build_place_question(preparation=preparation, candidates=offered, object_label=object_label,
                                        scene_id=str(preparation.get("scene_id") or Path(pipeline_root).parent.parent.parent.name),
                                        capture_id=Path(pipeline_root).parent.name,
                                        source_geometry=source_geometry,
                                        picture={key: record[key] for key in ("frame_id", "reference_time_ms", "png_sha256")})
        if question is not None:
            write_json(question_path, question)
        return question
    except Exception:  # noqa: BLE001 - the picture and the question are optional; the job never waits on them
        return None


def public_place_question(capture_root: Path) -> dict[str, Any] | None:
    """What the website may show: the sentence, the places' ids and labels, and the picture's reference."""
    from .website_task_target_choice import read_answer

    pipeline = Path(capture_root) / "pipeline"
    question = read_place_question(pipeline)
    if question is None or not question.get("picture") or read_answer(pipeline, question) is not None:
        return None  # a question without its picture is never published
    return {"kind": "task_destination_choice", **{key: question[key] for key in (
        "question_digest", "target_id", "semantic_label", "sentence", "candidate_set_digest", "guessed_candidate_id")},
        "candidates": [{key: row[key] for key in ("candidate_id", "label", "reference_time_ms")}
                       for row in question["candidates"]],
        **({"picture": {key: question["picture"][key] for key in ("png_sha256", "frame_id")}}
           if question.get("picture") else {})}


def settle_place(output_root: Path, destination: Mapping[str, Any], choice: Mapping[str, Any], *,
                 subject_min: Sequence[float], subject_max: Sequence[float], support: Sequence[Sequence[float]],
                 holds: Callable[[Sequence[float], float], bool] | None = None, surface_guessed: bool = False,
                 admit: Callable[[Mapping[str, Any]], bool] | None = None
                 ) -> tuple[dict[str, Any], dict[str, Any]]:
    """In preparation: the places offered (recorded on the choice), then the site's picked place if any.

    ``surface_guessed``: the surface it is put on was itself a best guess among several instances
    (``website_task_masks``); the place is then flagged ``destination_guessed`` as well.
    """
    from .website_destination_area import DESTINATION_GUESSED
    if surface_guessed and DESTINATION_GUESSED not in (choice.get("flags") or []):
        choice = {**choice, "flags": sorted([*(choice.get("flags") or []), DESTINATION_GUESSED]),
                  "surface_guess": "best_supported_guess"}
    candidates = place_candidates(choice=choice, subject_min=subject_min, subject_max=subject_max,
                                  support_min=support[0], support_max=support[1], holds=holds)
    if admit is not None:
        # Only places the sealer admits are offered (the one in use was admitted when it was chosen).
        height = float(destination["position_world_m"][2])
        candidates = candidates[:1] + [row for row in candidates[1:] if admit(
            {**{key: value for key, value in destination.items() if key != "radius_m"},
             "position_world_m": [*row["centre_xy"], height],
             **({"radius_m": row["radius_m"]} if row["radius_m"] > WEBSITE_TARGET_TOLERANCE_M else {})})]
    pipeline = pipeline_of(output_root)
    if pipeline is None:  # Not a capture's workspace: nothing to offer or apply.
        return dict(destination), {**choice, "candidates": candidates}
    question = read_place_question(pipeline)
    if question is not None and place_choice(pipeline) is not None:
        # The places the site chose among stay the ones on offer.
        candidates = [{key: row[key] for key in ("candidate_id", "centre_xy", "radius_m")} | {"kind": row["place_kind"]}
                      for row in question["candidates"]]
    sealed, record = apply_site_place(pipeline, destination, {**choice, "candidates": candidates})
    return sealed, record


class PlacePictureUnavailable(ValueError):
    """No picture to read for this capture and digest (or the website does not list the place question)."""


def _job_closed_or_ended(capture_root: Path) -> bool:
    """The job's run was closed by its owner, or its authority ended (or its ledger cannot be read)."""
    import json

    from .job_retry_policy import owner_closed
    try:
        ledger = json.loads((Path(capture_root) / "pipeline_job_ledger.json").read_text())
    except (OSError, ValueError):
        return True
    if not isinstance(ledger, dict):
        return True
    return (ledger.get("status") == "terminal_authority_ended"
            or owner_closed(ledger.get("last_error")) or owner_closed(ledger.get("terminal_code")))


def open_place_picture(*, request_id: str, capture_id: str, digest: str, storage_root: Path | None) -> bytes:
    """The place picture's bytes, for a WebApp read it already authorized for the job's owner.

    Read the way the scene preview is (``website_scene_preview.open_scene_preview``): the caller names a
    capture and a digest, never a path. Behind switch (iii); refused for a withdrawn capture, for any
    digest but the current picture's, and for a file whose bytes no longer match it.
    """
    import re

    from .website_atlas_preview_store import capture_name_matches_request, capture_withdrawn
    from .website_preparation_outcome import website_lists_place_question
    if (not website_lists_place_question() or storage_root is None
            or not capture_name_matches_request(request_id, capture_id)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(digest))):
        raise PlacePictureUnavailable("website_place_picture_unavailable")
    roots = [root for root in Path(storage_root).glob(f"*/scenes/site-{request_id}/captures/{capture_id}")
             if root.is_dir() and not root.is_symlink()]
    if len(roots) != 1 or capture_withdrawn(roots[0]) or _job_closed_or_ended(roots[0]):
        raise PlacePictureUnavailable("website_place_picture_unavailable")
    question = read_place_question(roots[0] / "pipeline")
    path = _folder(roots[0] / "pipeline") / PICTURE_FILE
    if ((question or {}).get("picture") or {}).get("png_sha256") != digest or path.is_symlink() or not path.is_file():
        raise PlacePictureUnavailable("website_place_picture_unavailable")
    if path.stat().st_size > MAX_PICTURE_BYTES:
        raise PlacePictureUnavailable("website_place_picture_unavailable")
    body = path.read_bytes()
    if "sha256:" + hashlib.sha256(body).hexdigest() != digest or not body.startswith(b"\x89PNG\r\n\x1a\n"):
        raise PlacePictureUnavailable("website_place_picture_unavailable")
    return body
