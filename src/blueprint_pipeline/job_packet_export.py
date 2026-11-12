"""Write one finished website job's job packet folder from files the operator pulled from the host.

The job packet is the unit of the product (owner direction 2026-11-12): what a deployment company
gets to evaluate a job without visiting. V1 is by hand: the operator pulls the job's files, runs this
offline (no network, no door, no provider), checks the folder and carries it. Nothing here sends it.

Two levels:

- ``summary`` (the default) is what may be shared today. Every site has agreed only that its video
  stays private (the recording statement covers building a simulation and running robot software in
  it; the Pipeline records website derivatives as ``admitted_for_internal_development``,
  ``public_redistribution_allowed: false``, ``website_native_background.construction_rights_admission``).
  So the summary carries the kind of job in generic words, the first look and the early trial's result,
  and nothing the site said or filmed: no sentence, name, address, coordinate, frame or derived file.
- ``full`` adds the site's sentence, the video, the simulation, the success rule and the trial clips.
  It is refused unless (a) the task context's ``capture_rights.statement_version`` is a recording
  statement that covers sharing (``SHARING_PERMISSION_STATEMENT_VERSIONS``, empty until the owner
  words one and a reviewed change adds it), or (b) the operator states the capture is Blueprint's own,
  names the approver and quotes the owner, and the capture id is on
  ``config/blueprint_owned_captures.v1.json``. No task context field tells Blueprint's own capture from
  a site's (the website projects every upload as ``website_site_task_context.v1``), so the allow-list
  is the only evidence.

Both levels refuse a withdrawn job (``<site>/website_withdrawal/tombstone.json``, its presence alone,
as ``screening_site_withdrawal`` reads it; or revoked ``capture_rights``), a job whose capture
authority ended (``pipeline_job_ledger.json`` status ``terminal_authority_ended``, or an
``authority_ended`` ``pipeline_job_terminal_receipt.json``; ``pubsub_handoff_listener``), and a job its
owner closed (terminal code ``website_run_closed_by_owner``, the code ``job_retry_policy`` gives the
owner-closed ending). A record that cannot be read refuses too: these are privacy and rights stops.

The tool copies only files it lists in ``manifest.json``, resolves every input inside the directory
the operator named (a symlink out of it is refused) and writes only inside its own output folder.
Where a record is missing the sheet says "not recorded"; nothing is estimated here.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "job_packet_manifest.v1"
EARLY_TRIAL_LABEL = ("Early trial of openly available robot software. It does not predict what a "
                     "deployment company's own robot would do.")
TERMS_LINE = ("This packet is shared for evaluating one job. Do not pass it on, train on it, or keep it "
              "after you are asked to delete it.")
#: Recording statement versions whose wording lets the site's video and simulation go to approved
#: deployment companies. Empty: no statement covers sharing yet (WebApp ``recordingRightsStatement``
#: v2 does not). A version is added only in a reviewed change once the owner has worded it.
SHARING_PERMISSION_STATEMENT_VERSIONS: tuple[str, ...] = ()
OWNED_CAPTURES_SCHEMA = "blueprint_owned_captures.v1"
OWNED_CAPTURES_PATH = Path(__file__).resolve().parents[2] / "config" / "blueprint_owned_captures.v1.json"
TASK_CONTEXT_SCHEMA = "website_site_task_context.v1"
OWNER_CLOSED_CODE = "website_run_closed_by_owner"
AUTHORITY_ENDED_STATUS = "terminal_authority_ended"
LEVELS = ("summary", "full")
NOT_RECORDED = "not recorded"
GENERIC_KIND = "a pick-up-and-put-down job"
VIDEO_SUFFIXES = {".mov", ".mp4", ".m4v"}
USD_SUFFIXES = {".usd", ".usda", ".usdc", ".usdz"}
RECONSTRUCTION_SUFFIXES = {".spz", ".ply", ".glb"}
#: Plain sentences for the scene estimate flags (``atlas_scene_estimates``).
ESTIMATE_SENTENCES = {
    "scene_estimate:level": "Which way is up was not observed; the reconstruction provider's up direction stands in.",
    "scene_estimate:splat_view_dependence_dropped": "The room's look keeps base colours only; shine that changes with the view was dropped.",
    "scene_estimate:floor": "A flat floor stands in for an uneven or missing scanned floor.",
    "scene_estimate:collider_registration": "Where the collision surfaces sit, or which faces were kept, is an estimate.",
    "scene_estimate:review_citations_dropped": "The scene review cited frames that could not be used; those citations were left out.",
    "scene_estimate:surrounding_views_dropped": "Some surrounding views were left out to keep within the frame budget.",
}
_KIND_BY_FAMILY = {"pouring": "pour from one container into another", "wiping": "wipe a surface",
                   "insertion": "insert one part into another", "stacking": "stack objects",
                   "button_press": "press a button", "cloth_folding": "fold cloth"}
_BODY_WORDS = {"fixed_arm": "fixed arm", "mobile_base": "mobile base with an arm",
               "mobile_manipulator": "mobile base with an arm", "humanoid": "humanoid"}
_VERDICT_WORDS = {"fits": "fits (its carry was path-checked)", "fits_spec": "fits on published figures",
                  "doesnt_fit": "does not fit", "unknown": "unknown"}


class JobPacketRefused(Exception):
    """A refusal with a typed code; nothing has been written."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(code + (f": {detail}" if detail else ""))
        self.code = code


@dataclass(frozen=True)
class JobInputs:
    capture_dir: Path
    team_scene_dir: Path | None = None
    configured_scene_dir: Path | None = None
    screening_publication: Path | None = None
    review_videos_dir: Path | None = None
    first_look_file: Path | None = None
    video: Path | None = None


@dataclass(frozen=True)
class OwnedCaptureApproval:
    capture_id: str
    approved_by: str
    approval_evidence: str


# --- reading inputs, never outside the directory named -------------------------------------------------

def _inside(root: Path, path: Path) -> Path:
    """The real file behind ``path``; refused when it (or a link on the way) leaves ``root``."""
    real_root = root.resolve()
    real = path.resolve()
    if real != real_root and not real.is_relative_to(real_root):
        raise JobPacketRefused("job_packet_input_symlink_outside_inputs", str(path.name))
    return real


def _file_argument(path: Path) -> tuple[Path, Path]:
    """A single named input file: its directory is its root; the file itself may not be a link."""
    if path.is_symlink():
        raise JobPacketRefused("job_packet_input_symlink_outside_inputs", path.name)
    if not path.is_file():
        raise JobPacketRefused("job_packet_input_missing", path.name)
    return path.parent, path


def _walk(root: Path) -> list[Path]:
    """Every regular file under ``root``, in order; any link must stay inside ``root``."""
    files: list[Path] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(directory)
        for name in sorted(dirnames):
            if (here / name).is_symlink():
                _inside(root, here / name)  # a link inside the tree is reached by its real path
        dirnames[:] = sorted(name for name in dirnames if not (here / name).is_symlink())
        for name in sorted(filenames):
            path = here / name
            real = _inside(root, path)
            if real.is_file():
                files.append(path)
    return files


def _read_json(root: Path, path: Path, *, required: bool = False) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        if required:
            raise JobPacketRefused("job_packet_input_missing", str(path.relative_to(root)))
        return None
    real = _inside(root, path)
    try:
        value = json.loads(real.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise JobPacketRefused("job_packet_input_unreadable", path.name) from exc
    if not isinstance(value, dict):
        raise JobPacketRefused("job_packet_input_unreadable", path.name)
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- the hard stops -----------------------------------------------------------------------------------

def job_state_refusal(capture_dir: Path, context: Mapping[str, Any]) -> str | None:
    """Why this job may not be packed at any level (withdrawn, authority ended, owner closed), or None."""
    root = capture_dir
    site = capture_dir.parent.parent
    if capture_dir.parent.name != "captures" or not site.name.startswith("site-"):
        # Without the site directory the withdrawal tombstone cannot be looked for.
        return "job_packet_withdrawal_state_unknown"
    tombstone = site / "website_withdrawal" / "tombstone.json"
    if tombstone.exists() or tombstone.is_symlink():
        return "job_packet_capture_withdrawn"
    rights = _mapping(context.get("capture_rights"))
    if rights.get("consent_revoked") is True or rights.get("consent_status") == "revoked":
        return "job_packet_capture_withdrawn"
    if rights.get("derived_scene_generation_allowed") is not True or rights.get("consent_revoked") is not False:
        return "job_packet_capture_rights_not_granted"
    try:
        ledger = _read_json(root, root / "pipeline_job_ledger.json")
        receipt = _read_json(root, root / "pipeline_job_terminal_receipt.json")
    except JobPacketRefused:
        return "job_packet_job_state_unreadable"
    if ledger is None:
        return "job_packet_job_ledger_missing"
    codes = {ledger.get("terminal_code"), (receipt or {}).get("code")}
    if OWNER_CLOSED_CODE in codes:
        return "job_packet_job_closed_by_owner"
    if ledger.get("status") == AUTHORITY_ENDED_STATUS or (receipt or {}).get("status") == "authority_ended":
        return "job_packet_capture_authority_ended"
    if ledger.get("status") == "corrupt":
        return "job_packet_job_state_unreadable"
    return None


def owned_capture_ids(path: Path = OWNED_CAPTURES_PATH) -> set[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != OWNED_CAPTURES_SCHEMA or not isinstance(value.get("captures"), list):
        raise JobPacketRefused("job_packet_owned_capture_list_invalid")
    return {str(row["capture_id"]) for row in value["captures"]
            if isinstance(row, Mapping) and isinstance(row.get("capture_id"), str) and row["capture_id"]}


def permission_basis(level: str, context: Mapping[str, Any], *, owned: OwnedCaptureApproval | None,
                     owned_list: Path = OWNED_CAPTURES_PATH) -> dict[str, Any]:
    """Which rule allows this level, or a refusal."""
    if level == "summary":
        return {"rule": "summary_only", "basis": "a summary carries nothing the site said or filmed"}
    version = _mapping(context.get("capture_rights")).get("statement_version")
    if isinstance(version, str) and version in SHARING_PERMISSION_STATEMENT_VERSIONS:
        return {"rule": "site_sharing_permission", "statement_version": version}
    if owned is None:
        raise JobPacketRefused("job_packet_full_level_requires_sharing_permission")
    if not owned.approved_by.strip() or not owned.approval_evidence.strip():
        raise JobPacketRefused("job_packet_owned_capture_approval_missing")
    if owned.capture_id != context.get("capture_id"):
        raise JobPacketRefused("job_packet_owned_capture_id_mismatch")
    if owned.capture_id not in owned_capture_ids(owned_list):
        raise JobPacketRefused("job_packet_capture_not_blueprint_owned")
    return {"rule": "blueprint_owned_capture", "capture_id": owned.capture_id,
            "approved_by": owned.approved_by.strip(), "approval_evidence": owned.approval_evidence.strip(),
            "allow_list": "config/" + owned_list.name, "allow_list_sha256": _sha256(owned_list)}


# --- what the records say -----------------------------------------------------------------------------

def _preparation(capture_dir: Path, handoff: Mapping[str, Any] | None) -> tuple[dict | None, Path | None]:
    """The preparation the handoff names (by digest), with the directory it sits in."""
    base = capture_dir / "pipeline" / "website_scene_preparation"
    candidates = [base / "preparation.json"]
    if (base / "revisions").is_dir():
        candidates += sorted((base / "revisions").glob("*/preparation.json"))
    found = [(path, _read_json(capture_dir, path)) for path in candidates if path.exists() or path.is_symlink()]
    wanted = _mapping(handoff).get("preparation_digest")
    chosen = [row for row in found if wanted and row[1].get("digest") == wanted]
    if not chosen and len(found) == 1:
        chosen = found
    return (chosen[0][1], chosen[0][0].parent) if len(chosen) == 1 else (None, None)


def kind_of_job(preparation: Mapping[str, Any] | None, context: Mapping[str, Any]) -> str:
    """The kind of job in generic words, from structured fields only; never the site's sentence."""
    prep = _mapping(preparation)
    strategy = _mapping(_mapping(prep.get("intake_request")).get("task")).get("strategy")
    destination = _mapping(prep.get("destination"))
    if strategy == "articulated_open_close":
        return "open or close one moving part of a fixture (a door, drawer or lid)"
    if strategy == "pick_and_place":
        if "interior_bounds_world_m" in destination:
            return "put one object into a container"
        if "destination_support" in prep or "support" in destination:
            return "move one object from one surface to another"
        if destination.get("relation") == "on":
            return "move one object to another place"
        return GENERIC_KIND
    family = context.get("task_family")
    return _KIND_BY_FAMILY.get(family, GENERIC_KIND) if isinstance(family, str) else GENERIC_KIND


def _fit_table_lines(table: Mapping[str, Any], *, full: bool) -> list[str]:
    rows = [row for row in table.get("rows") or [] if isinstance(row, Mapping)]
    if not rows:
        return ["The robot fit table has no rows."]
    lines = ["| Robot | Maker | Kind | Fit | Checks not passed |", "|---|---|---|---|---|"]
    for row in rows:
        checks = [check for check in row.get("checks") or [] if isinstance(check, Mapping)]
        open_checks = [f"{check.get('check')} ({check.get('result')})" for check in checks if check.get("result") != "pass"]
        detail = "; ".join(str(reason) for reason in row.get("reasons") or []) if full else ", ".join(open_checks)
        lines.append("| " + " | ".join(_cell(value) for value in (
            row.get("name"), row.get("manufacturer"), _BODY_WORDS.get(row.get("body_type"), row.get("body_type")),
            _VERDICT_WORDS.get(row.get("verdict"), row.get("verdict")), detail or "none")) + " |")
    counts = _mapping(table.get("counts"))
    if counts:
        lines += ["", "Counts: " + ", ".join(f"{_VERDICT_WORDS.get(k, k)}: {v}" for k, v in sorted(counts.items()))]
    lines += ["", "Checked against published specifications and the scene's estimated geometry (reach, payload, "
              "base footprint, cycle time). It is not a test of any robot."]
    return lines


def _cell(value: Any) -> str:
    return str(value if value not in (None, "") else NOT_RECORDED).replace("|", "/").replace("\n", " ")


def _trial_lines(publication: Mapping[str, Any] | None) -> list[str]:
    lines = [f"**{EARLY_TRIAL_LABEL}**", ""]
    if publication is None:
        return lines + ["No early trial result is included: none was given to the tool for this job."]
    rows = [row for row in publication.get("candidates") or [] if isinstance(row, Mapping)]
    if not rows:
        return lines + ["The trial's publication lists no robot software."]
    lines += ["| Robot software | Status | Runs scored | Runs that met the success rule |", "|---|---|---|---|"]
    for row in rows:
        scored, succeeded = row.get("episodes_scored"), row.get("episodes_succeeded")
        lines.append("| " + " | ".join(_cell(value) for value in (
            row.get("display_name") or row.get("candidate_id"), row.get("status"), scored,
            f"{succeeded} of {scored}" if isinstance(succeeded, int) and isinstance(scored, int) else succeeded)) + " |")
    lines += ["", f"Campaign status: {_cell(publication.get('status'))}. A zero is a valid result of the trial."]
    return lines


def _metres(value: Any) -> str:
    number = _number(value)
    return NOT_RECORDED if number is None else f"{number:.3f} m"


def _point(value: Any) -> str:
    if isinstance(value, (list, tuple)) and value and all(_number(v) is not None for v in value):
        return "(" + ", ".join(f"{float(v):.3f}" for v in value) + ") m"
    return NOT_RECORDED


def _range(value: Any, unit: str, scale: float = 1.0) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(_number(v) is not None for v in value):
        return f"{float(value[0]) * scale:.3g}–{float(value[1]) * scale:.3g} {unit}".strip()
    return NOT_RECORDED


def done_sentence(contract: Mapping[str, Any] | None, *, object_name: str | None, place_label: str | None,
                  assumed: bool) -> str:
    """What counts as done, in words a person can check, built only from the sealed contract's fields."""
    if not isinstance(contract, Mapping):
        return "What counts as done: not recorded (no sealed success contract was given to the tool)."
    criteria = _mapping(contract.get("criteria"))
    target = _mapping(criteria.get("surface_target"))
    thing = f"the {object_name}" if object_name else f"the object (its name: {NOT_RECORDED})"
    label = target.get("visible_label") or place_label
    where_label = f"the {label}" if label else f"a place whose name is {NOT_RECORDED}"
    parts: list[str] = []
    if target:
        radius = _number(target.get("radius_m"))
        within = f"within {radius * 100:.0f} cm" if radius is not None else f"within a distance {NOT_RECORDED}"
        relation = "inside" if target.get("placement_relation") == "inside" else "on"
        parts.append(f"{thing}'s centre ends {within} of the point {_point(target.get('surface_position_world_m'))}, "
                     f"{relation} {where_label}")
        tilt = _number(target.get("maximum_tilt_rad"))
        parts.append(f"it is tilted no more than {math.degrees(tilt):.0f}°" if tilt is not None else f"tilt: {NOT_RECORDED}")
        still = _number(target.get("stable_seconds"))
        parts.append(f"it stays still for {still:g} s" if still is not None else f"how long it stays still: {NOT_RECORDED}")
    else:
        bounds = _mapping(_mapping(criteria.get("destination_containment")).get("position_bounds_world_m"))
        low, high = bounds.get("minimum"), bounds.get("maximum")
        if _point(low) != NOT_RECORDED and _point(high) != NOT_RECORDED and len(low) == len(high) == 3:
            centre = [(float(a) + float(b)) / 2 for a, b in zip(low, high, strict=True)]
            half = [(float(b) - float(a)) / 2 * 100 for a, b in zip(low, high, strict=True)]
            parts.append(f"{thing}'s centre ends within ±{half[0]:.0f} cm (x) and ±{half[1]:.0f} cm (y) of the point "
                         f"{_point(centre)}, and within ±{half[2]:.0f} cm of that height, on {where_label}")
        else:
            parts.append(f"where {thing} must end: {NOT_RECORDED} (on {where_label})")
        orientation = _mapping(criteria.get("orientation"))
        tolerance = _number(orientation.get("tolerance_rad"))
        if orientation.get("mode") == "ignored":
            parts.append("its turn is not checked")
        else:
            parts.append(f"it is turned no more than {math.degrees(tolerance):.0f}° from its set orientation"
                         if tolerance is not None else f"turn: {NOT_RECORDED}")
        settling = _mapping(criteria.get("settling"))
        tolerance_m = _number(settling.get("position_tolerance_m"))
        parts.append(f"it settles (moves less than {tolerance_m * 100:.1f} cm at the end)" if tolerance_m is not None
                     else f"settling: {NOT_RECORDED}")
    support = _mapping(criteria.get("support"))
    if support.get("height_mode") == "ignored":
        parts.append("the height of what it rests on is not checked")
    else:
        heights = support.get("height_interval_m")
        parts.append(f"it rests on a surface with its base {_range(heights, 'm')} high" if _range(heights, "m") != NOT_RECORDED
                     else f"the height it must rest at: {NOT_RECORDED}")
    gripper = _mapping(criteria.get("gripper_state")).get("mode")
    parts.append("the gripper has let go" if gripper == "released" else "the gripper's state is not checked"
                 if gripper == "ignored" else f"the gripper's state: {NOT_RECORDED}")
    lift = _number(_mapping(criteria.get("motion")).get("minimum_lift_m"))
    parts.append(f"it was lifted at least {lift * 100:.0f} cm on the way" if lift is not None
                 else f"how high it must be lifted: {NOT_RECORDED}")
    provenance = _mapping(contract.get("provenance"))
    if assumed:
        whose = "These thresholds are Blueprint's standard ones; the site has not confirmed them."
    elif provenance:
        whose = (f"Thresholds written by: {_cell(provenance.get('author_source'))}; "
                 f"status: {_cell(provenance.get('confirmation_status'))}.")
    else:
        whose = f"Who set these thresholds: {NOT_RECORDED}."
    return "Done means: " + "; ".join(parts) + ". " + whose


def _measured_and_estimated(preparation: Mapping[str, Any] | None, base_scene: Mapping[str, Any] | None,
                            fit_table: Mapping[str, Any] | None, publication: Mapping[str, Any] | None,
                            context: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    prep = _mapping(preparation)
    region = _mapping(_mapping(prep.get("registration")).get("task_region"))
    residual = _number(region.get("trimmed_rmse_m"))
    measured = [("How closely the video's geometry and the reconstruction line up around the task (registration "
                 f"residual, trimmed RMS): {residual * 100:.1f} cm, in the scene's estimated metres."
                 if residual is not None else f"Registration residual around the task: {NOT_RECORDED}.")]
    frame = _mapping(prep.get("coordinate_frame"))
    band = _mapping(_mapping(fit_table).get("inputs")).get("scale_band")
    error = _number(_mapping(_mapping(prep.get("authoring_inputs")).get("metric_envelope")).get("maximum_dimension_relative_error"))
    estimated = [f"Scale: basis {_cell(frame.get('scale_authority'))}; physical scale measured: "
                 f"{_cell(frame.get('physical_scale_measured'))}. Scale band: "
                 + (_range(band, "×") if band else (f"sizes may be off by up to ±{error * 100:.0f} %" if error is not None else NOT_RECORDED))
                 + "."]
    subject = _mapping(prep.get("subject"))
    low, high = subject.get("aabb_min_xyz"), subject.get("aabb_max_xyz")
    if _point(low) != NOT_RECORDED and _point(high) != NOT_RECORDED:
        size = " × ".join(f"{(float(b) - float(a)) * 100:.0f}" for a, b in zip(low, high, strict=False))
        estimated.append(f"Object size: {size} cm, from the video.")
    else:
        estimated.append(f"Object size: {NOT_RECORDED}.")
    bounds = _mapping(_mapping(prep.get("physics")).get("bounds"))
    estimated.append(f"Mass: {_range(bounds.get('mass_kg'), 'kg')}, a prior from the object's size, not weighed.")
    estimated.append(f"Friction: static {_range(bounds.get('static_friction'), '')}, dynamic "
                     f"{_range(bounds.get('dynamic_friction'), '')}; prior ranges, not measured.")
    weight = _mapping(context.get("operator_task_details")).get("item_weight")
    if isinstance(weight, str) and weight.strip():
        estimated.append(f"The site stated the object's weight as \"{weight.strip()}\"; not checked.")
    codes = set()
    for row in _mapping(base_scene).get("scene_estimates") or []:
        if isinstance(row, Mapping) and isinstance(row.get("code"), str):
            codes.add(row["code"])
    for code in _mapping(_mapping(publication).get("scene_feasibility")).get("flags") or []:
        if isinstance(code, str) and code.startswith("scene_estimate:"):
            codes.add(code)
    for code in sorted(codes):
        estimated.append(f"`{code}`: {ESTIMATE_SENTENCES.get(code, 'an estimate the scene rests on (no plain sentence recorded for this code).')}")
    if not codes:
        estimated.append(f"Scene estimate flags (`scene_estimate:*`): none recorded{'' if base_scene else ' (base_scene.json not given)'}.")
    return measured, estimated


# --- collecting files ---------------------------------------------------------------------------------

def _video(inputs: JobInputs) -> tuple[Path, Path] | None:
    if inputs.video is not None:
        return _file_argument(inputs.video)
    raw = inputs.capture_dir / "raw"
    found = [path for path in (_walk(raw) if raw.is_dir() else []) if path.parent == raw
             and path.suffix.lower() in VIDEO_SUFFIXES]
    if len(found) > 1:
        raise JobPacketRefused("job_packet_video_ambiguous", "pass --video")
    return (raw, found[0]) if found else None


def _full_files(inputs: JobInputs, plan_dir: Path | None) -> list[tuple[Path, Path, str]]:
    """(root, source, packet path) for every file the full level copies."""
    rows: list[tuple[Path, Path, str]] = []
    video = _video(inputs)
    if video is not None:
        rows.append((video[0], video[1], f"files/video/{video[1].name}"))
    atlas = inputs.capture_dir / "pipeline" / "atlas"
    if atlas.is_dir():
        for path in _walk(atlas):
            if path.parent == atlas and path.suffix.lower() in RECONSTRUCTION_SUFFIXES:
                rows.append((atlas, path, f"files/simulation/reconstruction/{path.name}"))
    if inputs.configured_scene_dir is not None:
        root = inputs.configured_scene_dir
        for path in _walk(root):
            if path.name == "configured_scene_bundle.v1.zip":
                rows.append((root, path, f"files/simulation/configured-scene/{path.name}"))
            elif path.suffix.lower() == ".usdz":
                rows.append((root, path, f"files/object/{path.name}"))
    if inputs.team_scene_dir is not None:
        root = inputs.team_scene_dir
        for path in _walk(root):
            relative = path.relative_to(root).as_posix()
            if path.suffix.lower() in USD_SUFFIXES:
                rows.append((root, path, f"files/simulation/team-scene/{relative}"))
            elif path.name == "team_scene_inputs.v1.json" or (
                    plan_dir is not None and path == plan_dir / "native_task_arena_scene_plan.v1.json"):
                rows.append((root, path, f"files/task/{relative}"))
    if inputs.screening_publication is not None:
        root, path = _file_argument(inputs.screening_publication)
        rows.append((root, path, f"files/early-trial/{path.name}"))
    if inputs.review_videos_dir is not None:
        root = inputs.review_videos_dir
        for path in _walk(root):
            if path.suffix.lower() == ".mp4" or path.name == "manifest.json":
                rows.append((root, path, f"files/early-trial/clips/{path.relative_to(root).as_posix()}"))
    if inputs.first_look_file is not None:
        root, path = _file_argument(inputs.first_look_file)
        rows.append((root, path, f"files/first-look/{path.name}"))
    destinations = [row[2] for row in rows]
    if len(set(destinations)) != len(destinations):
        raise JobPacketRefused("job_packet_file_name_collision")
    return rows


def _site_strings(context: Mapping[str, Any]) -> list[str]:
    """Everything the site said, for the check that a summary repeats none of it."""
    found: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, str) and len(value.strip()) >= 4:
            found.append(value.strip())
        elif isinstance(value, Mapping):
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    for key in ("description", "operator_answers", "operator_task_details", "success_criteria", "task_items"):
        collect(context.get(key))
    return found


def _code_revision() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return NOT_RECORDED
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", revision) else NOT_RECORDED


def _readme(*, level: str, files: Iterable[str], preparation: Mapping[str, Any] | None,
            base_scene: Mapping[str, Any] | None) -> str:
    files = list(files)
    lines = ["# Job packet", "", TERMS_LINE, "",
             f"Level: **{level}**. `JOB_SHEET.md` describes the job; `manifest.json` lists every file with "
             "its size and SHA-256, the rule that allowed this level and the keys a withdrawal is matched by.", "",
             "## Opening the simulation", ""]
    simulation = [path for path in files if path.startswith("files/simulation/") or path.startswith("files/object/")]
    if not simulation:
        lines.append("This packet carries no simulation files"
                     + (" (a summary never does)." if level == "summary" else " (none were given to the tool)."))
    else:
        frame = _mapping(_mapping(preparation).get("coordinate_frame"))
        lines += [
            "- `files/simulation/team-scene/`: OpenUSD for NVIDIA Isaac Sim. Blueprint runs these in Isaac Sim 6.0.1 "
            "(NGC image). `scene_appearance.usdc` holds the room's look as a ParticleField Gaussian splat, which "
            "Isaac Sim 6 reads; other USD tools may show only the collision geometry and the object. "
            "`collision.usdc` is ordinary USD mesh geometry.",
            "- `files/simulation/configured-scene/`: the configured scene Blueprint built (scene, object, checks), zipped.",
            "- `files/object/`: the job's object as USDZ, as Blueprint authored it from the video (its size, mass and "
            "friction are estimates; see the sheet).",
            "- `files/simulation/reconstruction/`: the raw reconstruction. `.spz` / `.ply` are Gaussian splats "
            "(common splat viewers open them); `.glb` is a mesh (Blender or any glTF viewer).",
            "",
            f"Task coordinates: {_cell(frame.get('task_coordinates'))}; up axis {_cell(frame.get('declared_up_axis'))}; "
            f"physical scale measured: {_cell(frame.get('physical_scale_measured'))}. "
            f"The reconstruction's own up axis: {_cell(_mapping(base_scene).get('up_axis'))}.",
        ]
    lines += ["", "## Who received this packet", "",
              "`manifest.json` has an empty `recipients` list on purpose: the register of which deployment company "
              "received which packet (by `packet_id`) is kept on the website side, not in this folder.", ""]
    return "\n".join(lines)


def build_job_packet(inputs: JobInputs, *, output_root: Path, level: str = "summary",
                     owned: OwnedCaptureApproval | None = None, first_look_checked_by: str | None = None,
                     redact_terms: Iterable[str] = (), now: datetime | None = None,
                     owned_list: Path = OWNED_CAPTURES_PATH) -> dict[str, Any]:
    """Check everything, then write ``packet-<capture>-<date>/`` under ``output_root``; return the manifest."""
    if level not in LEVELS:
        raise JobPacketRefused("job_packet_level_invalid")
    capture_dir = inputs.capture_dir
    if capture_dir.is_symlink() or not capture_dir.is_dir():
        raise JobPacketRefused("job_packet_input_missing", "capture directory")
    context = _read_json(capture_dir, capture_dir / "pipeline" / "website_task_context.json", required=True)
    if context.get("schema_version") != TASK_CONTEXT_SCHEMA:
        raise JobPacketRefused("job_packet_task_context_unsupported")
    refusal = job_state_refusal(capture_dir, context)
    if refusal:
        raise JobPacketRefused(refusal)
    if context.get("capture_id") != capture_dir.name or context.get("scene_id") != capture_dir.parent.parent.name:
        raise JobPacketRefused("job_packet_input_identity_mismatch", "task context")
    basis = permission_basis(level, context, owned=owned, owned_list=owned_list)
    full = level == "full"

    handoff = _read_json(capture_dir, capture_dir / "pipeline" / "website_scene_preparation" / "handoff.json")
    preparation, prep_dir = _preparation(capture_dir, handoff)
    base_scene = _read_json(capture_dir, prep_dir / "base_scene.json") if prep_dir is not None else None
    fit_table = _mapping(_mapping(handoff).get("robot_fit_table")) or None
    publication = None
    if inputs.screening_publication is not None:
        root, path = _file_argument(inputs.screening_publication)
        publication = _read_json(root, path, required=True)
        if publication.get("site_id") != context["scene_id"]:
            raise JobPacketRefused("job_packet_input_identity_mismatch", "early trial publication")
    team_record = plan = None
    plan_dir = None
    if inputs.team_scene_dir is not None:
        root = inputs.team_scene_dir
        team_record = _read_json(root, root / "team_scene_inputs.v1.json")
        for name in ("native-task-packet-three-view", "native-task-packet"):
            plan = _read_json(root, root / name / "native_task_arena_scene_plan.v1.json")
            if plan is not None:
                plan_dir = root / name
                break
        if plan is not None and plan.get("scene_id") not in (None, context["scene_id"]):
            raise JobPacketRefused("job_packet_input_identity_mismatch", "team scene")
    first_look_text = None
    if inputs.first_look_file is not None:
        root, path = _file_argument(inputs.first_look_file)
        if path.suffix.lower() not in {".md", ".txt"}:
            raise JobPacketRefused("job_packet_first_look_format_unsupported", "use .md or .txt")
        first_look_text = _inside(root, path).read_text(encoding="utf-8")
        if not full:
            if not (first_look_checked_by or "").strip():
                raise JobPacketRefused("job_packet_first_look_unchecked", "pass --first-look-checked-by")
            lowered = first_look_text.casefold()
            if any(term.casefold() in lowered for term in [*_site_strings(context), *redact_terms] if term.strip()):
                raise JobPacketRefused("job_packet_first_look_repeats_site_words")
    copies = _full_files(inputs, plan_dir) if full else []

    # --- the sheet ---
    kind = kind_of_job(preparation, context)
    sheet = [f"# Job sheet ({level})", "", TERMS_LINE, ""]
    if full:
        criteria = _mapping(context.get("success_criteria"))
        details = _mapping(context.get("operator_task_details"))
        answers = _mapping(context.get("operator_answers"))
        sheet += ["## The job, in the site's words", "", f"> {context.get('description', NOT_RECORDED)}", "",
                  f"Kind of job: {kind}.", "", "## Constraints and what the site told us", ""]
        sheet += [f"- The site's own success definition: {_cell(criteria.get('successDefinition'))}",
                  f"- Success rate the site wants: {_cell(criteria.get('successRate'))} %",
                  f"- Cycle time the site wants: {_cell(criteria.get('cycleTimeSeconds'))} s",
                  f"- Object weight (stated): {_cell(details.get('item_weight'))}",
                  f"- Object make and model (stated): {_cell(details.get('item_make_model'))}"]
        sheet += [f"- {_cell(key)}: {_cell(value)}" for key, value in sorted(answers.items())]
        unresolved = context.get("unresolved")
        if isinstance(unresolved, list) and unresolved:
            sheet.append("- Still unanswered: " + ", ".join(_cell(item) for item in unresolved))
        sheet.append("- Other constraints: " + NOT_RECORDED + " (the task context has no constraints field).")
        prep = _mapping(preparation)
        subject, destination = _mapping(prep.get("subject")), _mapping(prep.get("destination"))
        support = _mapping(prep.get("support"))
        low, high = subject.get("aabb_min_xyz"), subject.get("aabb_max_xyz")
        start = ([(float(a) + float(b)) / 2 for a, b in zip(low, high, strict=False)]
                 if _point(low) != NOT_RECORDED and _point(high) != NOT_RECORDED else None)
        contract = _mapping(team_record).get("task_success_contract") or _mapping(_mapping(plan).get("task_spec")).get("task_success_contract")
        target = _mapping(_mapping(_mapping(contract).get("criteria")).get("surface_target"))
        sheet += ["", "## The object and where it goes", "",
                  f"- Object: {_cell(subject.get('description'))}; starts at {_point(start)} on "
                  f"{_cell(support.get('description'))}.",
                  f"- Goes to: {_cell(target.get('visible_label') or destination.get('visible_label'))}; centre "
                  f"{_point(target.get('surface_position_world_m') or destination.get('position_world_m'))}; radius "
                  f"{_metres(target.get('radius_m'))}.",
                  "", "## What counts as done", "",
                  done_sentence(contract, object_name=subject.get("description"),
                                place_label=destination.get("visible_label"),
                                assumed=bool(_mapping(team_record).get("owner_success_assumption"))), ""]
        measured, estimated = _measured_and_estimated(preparation, base_scene, fit_table, publication, context)
        sheet += ["## Measured", ""] + [f"- {line}" for line in measured]
        sheet += ["", "## Estimated", ""] + [f"- {line}" for line in estimated] + [""]
    else:
        sheet += ["## The job", "", f"Kind of job: {kind}.", "",
                  "Described in general words. The site's own description, its name and its location, the video, "
                  "the simulation and the success rule are not in this summary: the site has not given permission "
                  "to share them.", ""]
    sheet += ["## First look on paper", ""]
    if first_look_text is not None:
        sheet += ["Source: the website's first look, exported by the operator"
                  + (f" (checked for site details by {first_look_checked_by.strip()})." if not full and first_look_checked_by
                     else "; attached under `files/first-look/`."), "", first_look_text.strip(), ""]
    elif fit_table:
        sheet += ["Source: the pipeline's robot fit table (`website_scene_preparation/handoff.json`).", ""]
        sheet += _fit_table_lines(fit_table, full=full) + [""]
    else:
        sheet += [f"First look: {NOT_RECORDED} (no robot fit table in the handoff, and no website first look given).", ""]
    sheet += ["## Early trial", ""] + _trial_lines(publication) + [""]
    missing = []
    if full:
        present = {row[2].split("/")[1] for row in copies}
        for item, words in (("video", "the video"), ("simulation", "the simulation"), ("object", "the object asset"),
                            ("task", "the sealed task spec"), ("early-trial", "the early trial's files")):
            if item not in present:
                missing.append(f"{words}: not given to the tool, so not in this packet")
    sheet += ["## What this packet does not show", "",
              "- No test at the site: everything here comes from one video and a simulation of it.",
              "- No safety assessment of any robot at this site."] + [f"- {line.capitalize()}." for line in missing] + [""]

    # --- write, only inside the packet folder ---
    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    short = re.sub(r"[^A-Za-z0-9]", "", str(context["capture_id"]).removeprefix("walkthrough-"))[:8] or "capture"
    packet = Path(output_root) / f"packet-{short}-{moment.strftime('%Y%m%d')}"
    if not Path(output_root).is_dir():
        raise JobPacketRefused("job_packet_output_root_missing")
    if packet.exists() or packet.is_symlink():
        raise JobPacketRefused("job_packet_output_exists", packet.name)
    packet.mkdir()
    try:
        written = {"JOB_SHEET.md": "\n".join(sheet), "README.md": _readme(
            level=level, files=[row[2] for row in copies], preparation=preparation, base_scene=base_scene)}
        for name, text in written.items():
            (packet / name).write_text(text, encoding="utf-8")
        for root, source, relative in copies:
            target = packet / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_inside(root, source), target)
        listed = []
        for relative in sorted([*written, *(row[2] for row in copies)]):
            path = packet / relative
            listed.append({"path": relative, "size": path.stat().st_size, "sha256": _sha256(path)})
        packet_id = "sha256:" + hashlib.sha256("\n".join(f"{row['path']} {row['sha256']}" for row in listed)
                                                 .encode()).hexdigest()
        manifest = {
            "schema_version": SCHEMA_VERSION, "packet_id": packet_id, "level": level,
            "capture_id": context["capture_id"], "created_at": moment.isoformat().replace("+00:00", "Z"),
            "code_revision": _code_revision(), "permission_basis": basis,
            "first_look_source": ("website_export" if first_look_text is not None else
                                  "pipeline_robot_fit_table" if fit_table else NOT_RECORDED),
            **({"first_look_checked_by": first_look_checked_by.strip()} if first_look_text is not None
               and not full and first_look_checked_by else {}),
            "takedown_keys": {"request_id": context.get("request_id"), "scene_id": context["scene_id"],
                              "capture_id": context["capture_id"],
                              "withdrawal_tombstone": f"scenes/{context['scene_id']}/website_withdrawal/tombstone.json"},
            "job_state_checked": {"withdrawal_tombstone": "absent in the pulled site directory",
                                  "capture_rights": {key: _mapping(context.get("capture_rights")).get(key) for key in
                                                     ("consent_status", "statement_version", "consent_revoked")},
                                  "job_ledger_status": _mapping(_read_json(capture_dir, capture_dir / "pipeline_job_ledger.json")).get("status")},
            "files": listed, "recipients": [],
        }
        (packet / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(packet, ignore_errors=True)
        raise
    return {**manifest, "packet_dir": str(packet)}
