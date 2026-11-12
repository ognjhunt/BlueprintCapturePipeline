"""Write one finished website job's job packet folder from files the operator pulled from the host.

The job packet is the unit of the product (owner direction 2026-11-12): what a deployment company
gets to evaluate a job without visiting. V1 is by hand: the operator pulls the job's files, runs this
offline (no network, no door, no provider), checks the folder and carries it. Nothing here sends it.

Two levels:

- ``summary`` (the default) is what may be shared today. Every site has agreed only that its video
  stays private (the Pipeline records website derivatives as ``admitted_for_internal_development``,
  ``public_redistribution_allowed: false``, ``website_native_background.construction_rights_admission``).
  The summary is built so it cannot carry site text: every value on it is a sentence written here, a
  robot or software name taken from the Pipeline's own catalog (``robot_spec_catalog``,
  ``config/beta_policy_candidates.v1.json``) for an id the input names exactly, a member of an
  enumerated set (``website_robot_fit.VERDICTS``, its check names and results,
  ``beta_policy_screening_result.CANDIDATE_STATUSES``) printed in this module's own words, or a whole
  number of tries. Anything else is left out and counted. Its folder and manifest carry nothing that
  identifies the site; what ties the packet to the job goes in an operator record beside the folder.
- ``full`` adds the site's sentence, the video, the simulation, the success rule and the trial clips.
  It is refused unless (a) the task context's ``capture_rights.statement_version`` is a recording
  statement that covers sharing (``SHARING_PERMISSION_STATEMENT_VERSIONS``, empty until the owner
  words one), or (b) the operator states the capture is Blueprint's own, names the approver, the date
  and quotes the owner, and the capture id is on ``config/blueprint_owned_captures.v1.json``. No task
  context field tells Blueprint's own capture from a site's, so the allow-list is the only evidence.
  Every input must be bound to this job, and the full manifest keeps the job's ids.

Both levels refuse what the Pipeline itself would treat as withdrawn, not consented, ended or closed:

- withdrawal: ``screening_site_withdrawal.site_withdrawal_state`` over the pull root (every
  ``<root>[/<bucket>]/scenes/<site>/website_withdrawal/tombstone.json``) and
  ``website_atlas_preview_store.capture_withdrawn`` on the capture;
- consent: ``consent_takedown.read_consent_state`` (its consent files) and
  ``consent_normalization.resolve_consent_signals`` on the task context, which must be ``active``;
- authority ended: ``pubsub_handoff_listener.AUTHORITY_ENDING_CODES`` and ``authority_ending_code``
  over every text in the job ledger and its terminal receipt, and status ``terminal_authority_ended``;
- owner closed: ``website_run_closed_by_owner`` matched as a whole token, as the unreleased
  ``job_retry_policy.owner_closed`` does (not on ``main`` yet, so the literal is matched here).

A record that cannot be read, or is not a regular file, refuses too: these are privacy and rights stops.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shutil
import stat
import subprocess
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "job_packet_manifest.v2"
OPERATOR_RECORD_SCHEMA = "job_packet_operator_record.v1"
EARLY_TRIAL_LABEL = ("Early trial of openly available robot software. It does not predict what a "
                     "deployment company's own robot would do.")
TERMS_LINE = ("This packet is shared for evaluating one job. Do not pass it on, train on it, or keep it "
              "after you are asked to delete it.")
#: Recording statement versions whose wording lets the site's video and simulation go to approved
#: deployment companies. Empty: no statement covers sharing yet (WebApp ``recordingRightsStatement``
#: v2 does not). A version is added only in a reviewed change once the owner has worded it.
SHARING_PERMISSION_STATEMENT_VERSIONS: tuple[str, ...] = ()
REPO_ROOT = Path(__file__).resolve().parents[2]
OWNED_CAPTURES_SCHEMA = "blueprint_owned_captures.v1"
OWNED_CAPTURES_PATH = REPO_ROOT / "config" / "blueprint_owned_captures.v1.json"
POLICY_CATALOG_PATH = REPO_ROOT / "config" / "beta_policy_candidates.v1.json"
TASK_CONTEXT_SCHEMA = "website_site_task_context.v1"
#: ``job_retry_policy.OWNER_CLOSED_CODE`` in unreleased work; matched as its whole token, as there.
OWNER_CLOSED_CODE = "website_run_closed_by_owner"
_OWNER_CLOSED = re.compile(r"(?<![A-Za-z0-9_-])" + OWNER_CLOSED_CODE + r"(?![A-Za-z0-9_-])")
AUTHORITY_ENDED_STATUS = "terminal_authority_ended"
LEVELS = ("summary", "full")
NOT_RECORDED = "not recorded"
INVALID = "not recorded (invalid value)"
GENERIC_KIND = "a pick-up-and-put-down job"
VIDEO_SUFFIXES = {".mov", ".mp4", ".m4v"}
USD_SUFFIXES = {".usd", ".usda", ".usdc", ".usdz"}
RECONSTRUCTION_SUFFIXES = {".spz", ".ply", ".glb"}
#: The consent files ``consent_takedown.read_consent_state`` reads, checked to be regular files first.
CONSENT_FILES = ("raw/rights_consent.json", "rights_consent.json", "raw/manifest.json", "capture_descriptor.json")
#: The checks ``website_robot_fit`` writes (its ``_check`` calls) and their results, in fixed words.
CHECK_WORDS = {"stance": "a place to stand", "reach": "reach", "payload": "payload",
               "cycle_time": "cycle time", "access": "access from the entrance"}
RESULT_WORDS = {"pass": "passed", "fail": "failed", "unknown": "unknown"}
VERDICT_WORDS = {"fits": "fits (its carry was path-checked)", "fits_spec": "fits on published figures",
                 "doesnt_fit": "does not fit", "unknown": "unknown"}
BODY_WORDS = {"fixed_arm": "fixed arm", "mobile_manipulator": "arm on a moving base", "humanoid": "humanoid",
              "wheeled_humanoid": "humanoid on wheels"}
STATUS_WORDS = {"pending": "waiting to run", "waitlisted": "waiting for budget", "preparing": "being prepared",
                "launching": "starting", "running": "running", "reconciliation_required": "being reconciled",
                "completed": "completed", "blocked": "stopped", "unavailable": "could not run",
                "deferred": "not run"}
CAMPAIGN_WORDS = {"completed": "completed", "running": "still running", "queued": "waiting to run",
                  "blocked": "stopped", "reconciliation_required": "being reconciled"}
MAX_TRIES = 10_000
#: Plain sentences for the scene estimate flags (``atlas_scene_estimates``).
ESTIMATE_SENTENCES = {
    "scene_estimate:level": "Which way is up was not observed; the reconstruction provider's up direction stands in.",
    "scene_estimate:splat_view_dependence_dropped": "The room's look keeps base colours only; shine that changes with the view was dropped.",
    "scene_estimate:floor": "A flat floor stands in for an uneven or missing scanned floor.",
    "scene_estimate:collider_registration": "Where the collision surfaces sit, or which faces were kept, is an estimate.",
    "scene_estimate:review_citations_dropped": "The scene review cited frames that could not be used; those citations were left out.",
    "scene_estimate:surrounding_views_dropped": "Some surrounding views were left out to keep within the frame budget.",
}
KIND_WORDS = {"articulated": "open or close one moving part of a fixture (a door, drawer or lid)",
              "container": "put one object into a container",
              "surface_to_surface": "move one object from one surface to another",
              "new_place": "move one object to another place", "generic": GENERIC_KIND,
              "pouring": "pour from one container into another", "wiping": "wipe a surface",
              "insertion": "insert one part into another", "stacking": "stack objects",
              "button_press": "press a button", "cloth_folding": "fold cloth"}


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
    #: The directory the operator pulled into, laid out as the host's site roots
    #: (``<root>[/<bucket>]/scenes/site-<request>/captures/<capture>``). Default: the directory above ``scenes``.
    pull_root: Path | None = None


@dataclass(frozen=True)
class OwnedCaptureApproval:
    capture_id: str
    approved_by: str
    approval_evidence: str
    approved_on: str = ""


# --- reading inputs: regular files only, never outside the directory named -----------------------------

def _inside(root: Path, path: Path) -> Path:
    """The real path behind ``path``; refused when it (or a link on the way) leaves ``root``."""
    real_root, real = root.resolve(), path.resolve()
    if real != real_root and not real.is_relative_to(real_root):
        raise JobPacketRefused("job_packet_input_symlink_outside_inputs", path.name)
    return real


def _regular(real: Path, name: str) -> Path:
    try:
        info = os.stat(real)
    except OSError as exc:
        raise JobPacketRefused("job_packet_input_unreadable", name) from exc
    if not stat.S_ISREG(info.st_mode):
        raise JobPacketRefused("job_packet_input_not_a_regular_file", name)
    if info.st_nlink != 1:
        raise JobPacketRefused("job_packet_input_hard_linked", name)
    return real


def _file_argument(path: Path) -> tuple[Path, Path]:
    """A single named input file: its directory is its root; the file itself may not be a link."""
    if path.is_symlink():
        raise JobPacketRefused("job_packet_input_symlink_outside_inputs", path.name)
    if not os.path.lexists(path):
        raise JobPacketRefused("job_packet_input_missing", path.name)
    _regular(path, path.name)
    return path.parent, path


def _walk(root: Path) -> list[Path]:
    """Every regular file under ``root``, in order. Links must stay inside; anything else is refused."""
    if root.is_symlink() or not root.is_dir():
        raise JobPacketRefused("job_packet_input_missing", root.name)
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
            if path.is_symlink() and real.is_dir():
                continue
            _regular(real, name)
            files.append(path)
    return files


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate key")
    return dict(pairs)


def _read_json(root: Path, path: Path, *, required: bool = False) -> dict[str, Any] | None:
    if not os.path.lexists(path):
        if required:
            raise JobPacketRefused("job_packet_input_missing", path.name)
        return None
    real = _regular(_inside(root, path), path.name)
    try:
        value = json.loads(real.read_bytes().decode("utf-8"), object_pairs_hook=_unique_pairs)
    except (OSError, UnicodeError, ValueError) as exc:
        raise JobPacketRefused("job_packet_input_unreadable", path.name) from exc
    if not isinstance(value, dict):
        raise JobPacketRefused("job_packet_input_unreadable", path.name)
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= MAX_TRIES else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _texts(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _texts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _texts(item)


# --- the hard stops -----------------------------------------------------------------------------------

def _layout(capture_dir: Path, pull_root: Path | None) -> Path:
    """The pull root, when the capture sits where the host's site roots put it; else a refusal."""
    parents = capture_dir.absolute().parents
    root = pull_root if pull_root is not None else parents[3] if len(parents) > 3 else None
    if root is None or root.is_symlink() or not root.is_dir():
        raise JobPacketRefused("job_packet_withdrawal_state_unknown", "pull root")
    try:
        relative = capture_dir.absolute().relative_to(root.absolute()).parts
    except ValueError:
        raise JobPacketRefused("job_packet_withdrawal_state_unknown", "capture outside the pull root") from None
    shape = relative[-4:]
    if (len(relative) not in (4, 5) or shape[0] != "scenes" or not shape[1].startswith("site-")
            or shape[2] != "captures" or any(part.startswith(".") for part in relative)):
        raise JobPacketRefused("job_packet_withdrawal_state_unknown", "layout")
    return root


def job_state_refusal(capture_dir: Path, context: Mapping[str, Any], *, pull_root: Path | None = None) -> str | None:
    """Why this job may not be packed at any level (withdrawn, not consented, ended, closed), or None."""
    from .consent_normalization import resolve_consent_signals
    from .consent_takedown import read_consent_state
    from .pubsub_handoff_listener import AUTHORITY_ENDING_CODES, authority_ending_code
    from .screening_site_withdrawal import SITE_ROOTS_ENV, site_withdrawal_state
    from .website_atlas_preview_store import capture_withdrawn

    try:
        root = _layout(capture_dir, pull_root)
    except JobPacketRefused as exc:
        return exc.code
    site_state = site_withdrawal_state(capture_dir.parent.parent.name, environ={SITE_ROOTS_ENV: str(root)})
    if site_state == "withdrawn":
        return "job_packet_capture_withdrawn"
    if site_state != "not_withdrawn":
        return "job_packet_withdrawal_state_unknown"
    if os.path.lexists(capture_dir.parent.parent / "website_withdrawal" / "tombstone.json"):
        # Whatever it is (a file, a directory, a link, a pipe), capture_withdrawn would call it withdrawn;
        # answering here keeps a pipe from ever being opened.
        return "job_packet_capture_withdrawn"
    try:
        for relative in CONSENT_FILES:  # read_consent_state opens these; a FIFO or a link out must not reach it
            if os.path.lexists(capture_dir / relative):
                _regular(_inside(capture_dir, capture_dir / relative), relative)
    except JobPacketRefused:
        return "job_packet_consent_state_unreadable"
    if capture_withdrawn(capture_dir):
        return "job_packet_capture_withdrawn"
    recorded = read_consent_state(capture_dir)
    if recorded.get("source_path") and recorded.get("state") != "active":
        # A consent file the pipeline reads exists and does not say active.
        return "job_packet_capture_withdrawn" if recorded.get("state") == "revoked" else "job_packet_consent_not_active"
    signals = resolve_consent_signals(context)
    if signals["state"] == "revoked":
        return "job_packet_capture_withdrawn"
    if signals["state"] != "active":
        return "job_packet_consent_not_active"
    rights = context.get("capture_rights")
    if (not isinstance(rights, Mapping) or rights.get("derived_scene_generation_allowed") is not True
            or rights.get("consent_revoked") is not False):
        return "job_packet_capture_rights_not_granted"
    try:
        ledger = _read_json(capture_dir, capture_dir / "pipeline_job_ledger.json")
        receipt = _read_json(capture_dir, capture_dir / "pipeline_job_terminal_receipt.json")
    except JobPacketRefused:
        return "job_packet_job_state_unreadable"
    if ledger is None:
        return "job_packet_job_ledger_missing"
    texts = [*_texts(ledger), *_texts(receipt or {})]
    if any(_OWNER_CLOSED.search(text) for text in texts):
        return "job_packet_job_closed_by_owner"
    if (ledger.get("status") == AUTHORITY_ENDED_STATUS or (receipt or {}).get("status") == "authority_ended"
            or any(text.strip() in AUTHORITY_ENDING_CODES or authority_ending_code(ValueError(text)) for text in texts)):
        return "job_packet_capture_authority_ended"
    if ledger.get("status") == "corrupt" or not isinstance(ledger.get("status"), str):
        return "job_packet_job_state_unreadable"
    return None


def owned_capture_ids(path: Path | None = None) -> set[str]:
    try:
        value = json.loads((path or OWNED_CAPTURES_PATH).read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
        if value.get("schema_version") != OWNED_CAPTURES_SCHEMA or not isinstance(value.get("captures"), list):
            raise ValueError("schema")
        ids = []
        for row in value["captures"]:
            if not isinstance(row, Mapping) or not isinstance(row.get("capture_id"), str):
                raise ValueError("row")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", row["capture_id"]):
                raise ValueError("id")
            ids.append(row["capture_id"])
    except (OSError, UnicodeError, ValueError, TypeError, AttributeError) as exc:
        raise JobPacketRefused("job_packet_owned_capture_list_invalid") from exc
    return set(ids)


def permission_basis(level: str, context: Mapping[str, Any], *, owned: OwnedCaptureApproval | None,
                     owned_list: Path | None = None) -> dict[str, Any]:
    """Which rule allows this level, or a refusal."""
    if level == "summary":
        return {"rule": "summary_only"}
    version = _mapping(context.get("capture_rights")).get("statement_version")
    if isinstance(version, str) and version in SHARING_PERMISSION_STATEMENT_VERSIONS:
        return {"rule": "site_sharing_permission", "statement_version": version}
    if owned is None:
        raise JobPacketRefused("job_packet_full_level_requires_sharing_permission")
    try:
        approved_on = date.fromisoformat(owned.approved_on.strip()) if isinstance(owned.approved_on, str) else None
    except ValueError:
        approved_on = None
    if not owned.approved_by.strip() or not owned.approval_evidence.strip() or approved_on is None:
        raise JobPacketRefused("job_packet_owned_capture_approval_missing")
    if owned.capture_id != context.get("capture_id"):
        raise JobPacketRefused("job_packet_owned_capture_id_mismatch")
    path = owned_list or OWNED_CAPTURES_PATH
    if owned.capture_id not in owned_capture_ids(path):
        raise JobPacketRefused("job_packet_capture_not_blueprint_owned")
    return {"rule": "blueprint_owned_capture", "capture_id": owned.capture_id,
            "approved_by": owned.approved_by.strip(), "approved_on": approved_on.isoformat(),
            "approval_evidence": owned.approval_evidence.strip(), "allow_list_sha256": _sha256(path)}


# --- what the records say -----------------------------------------------------------------------------

def _preparation(capture_dir: Path, handoff: Mapping[str, Any] | None) -> tuple[dict | None, Path | None]:
    """The preparation the handoff names (by digest), with the directory it sits in."""
    base = capture_dir / "pipeline" / "website_scene_preparation"
    candidates = [base / "preparation.json"]
    if (base / "revisions").is_dir():
        candidates += sorted((base / "revisions").glob("*/preparation.json"))
    found = [(path, _read_json(capture_dir, path)) for path in candidates if os.path.lexists(path)]
    wanted = _mapping(handoff).get("preparation_digest")
    chosen = [row for row in found if wanted and row[1].get("digest") == wanted]
    if not chosen and len(found) == 1:
        chosen = found
    return (chosen[0][1], chosen[0][0].parent) if len(chosen) == 1 else (None, None)


def kind_of_job(preparation: Mapping[str, Any] | None, context: Mapping[str, Any]) -> str:
    """The kind of job in fixed words, chosen from structured fields only; never the site's sentence."""
    prep = _mapping(preparation)
    strategy = _mapping(_mapping(prep.get("intake_request")).get("task")).get("strategy")
    destination = _mapping(prep.get("destination"))
    if strategy == "articulated_open_close":
        return KIND_WORDS["articulated"]
    if strategy == "pick_and_place":
        if "interior_bounds_world_m" in destination:
            return KIND_WORDS["container"]
        if "destination_support" in prep or "support" in destination:
            return KIND_WORDS["surface_to_surface"]
        return KIND_WORDS["new_place"] if destination.get("relation") == "on" else GENERIC_KIND
    family = context.get("task_family")
    return KIND_WORDS[family] if family in KIND_WORDS and family not in {"articulated", "container"} else GENERIC_KIND


def _robot_catalog() -> dict[str, Mapping[str, Any]]:
    from .robot_spec_catalog import load_robot_spec_catalog
    return {row["model_id"]: row for row in load_robot_spec_catalog()["models"]}


def _policy_catalog() -> dict[str, Mapping[str, Any]]:
    # The reviewed checkout's catalog, never the host override: its names are what the summary may print.
    from .beta_policy_candidate_catalog import load_catalog
    return {row["candidate_id"]: row for row in load_catalog(POLICY_CATALOG_PATH)["candidates"]}


def _left_out(reasons: Counter) -> list[str]:
    return [f"{count} row{'s' if count != 1 else ''} left out: {reason}." for reason, count in sorted(reasons.items())]


def summary_fit_lines(table: Mapping[str, Any] | None) -> list[str]:
    """The first look from the fit table, every value from the catalog or a fixed word."""
    from .website_robot_fit import VERDICTS
    if not table:
        return [f"First look: {NOT_RECORDED} (the job's robot fit table was not among the files given)."]
    catalog, rows, dropped = _robot_catalog(), [], Counter()
    for row in table.get("rows") if isinstance(table.get("rows"), list) else []:
        model = catalog.get(row.get("model_id")) if isinstance(row, Mapping) and isinstance(row.get("model_id"), str) else None
        checks = row.get("checks") if isinstance(row, Mapping) and isinstance(row.get("checks"), list) else None
        if model is None:
            dropped["not a known robot"] += 1
        elif row.get("verdict") not in VERDICTS:
            dropped["not a known verdict"] += 1
        elif checks is None or any(not isinstance(c, Mapping) or c.get("check") not in CHECK_WORDS
                                   or c.get("result") not in RESULT_WORDS for c in checks):
            dropped["a check this tool does not know"] += 1
        else:
            open_checks = [f"{CHECK_WORDS[c['check']]} ({RESULT_WORDS[c['result']]})" for c in checks if c["result"] != "pass"]
            rows.append((model, row["verdict"], open_checks))
    lines = ["Source: the pipeline's robot fit table.", ""]
    if rows:
        lines += ["| Robot | Maker | Kind | Fit | Checks not passed |", "|---|---|---|---|---|"]
        for model, verdict, open_checks in rows:
            lines.append(f"| {model['name']} | {model['manufacturer']} | {BODY_WORDS.get(model['body_type'], 'robot')} | "
                         f"{VERDICT_WORDS[verdict]} | {', '.join(open_checks) or 'none'} |")
        counts = Counter(verdict for _, verdict, _ in rows)
        lines += ["", "Counts: " + ", ".join(f"{VERDICT_WORDS[k]}: {counts[k]}" for k in VERDICTS if counts[k])]
    else:
        lines.append("No row of the table could be shown.")
    lines += [*([""] + _left_out(dropped) if dropped else []), "",
              "Checked against published specifications and the scene's estimated geometry (reach, payload, "
              "base footprint, cycle time). It is not a test of any robot."]
    return lines


def summary_trial_lines(publication: Mapping[str, Any] | None) -> list[str]:
    from .beta_policy_screening_result import CANDIDATE_STATUSES
    lines = [f"**{EARLY_TRIAL_LABEL}**", ""]
    if publication is None:
        return lines + ["No early trial result is included: none was given to the tool for this job."]
    catalog, rows, dropped = _policy_catalog(), [], Counter()
    for row in publication.get("candidates") if isinstance(publication.get("candidates"), list) else []:
        entry = catalog.get(row.get("candidate_id")) if isinstance(row, Mapping) and isinstance(row.get("candidate_id"), str) else None
        scored = _count(row.get("episodes_scored")) if isinstance(row, Mapping) else None
        succeeded = _count(row.get("episodes_succeeded")) if isinstance(row, Mapping) else None
        if entry is None:
            dropped["not a known robot software"] += 1
        elif row.get("status") not in CANDIDATE_STATUSES or row.get("status") not in STATUS_WORDS:
            dropped["not a known status"] += 1
        elif scored is None or succeeded is None or succeeded > scored:
            dropped["counts out of range"] += 1
        else:
            rows.append((entry, row["status"], scored, succeeded))
    if rows:
        lines += ["| Robot software | Status | Runs scored | Runs that met the success rule |", "|---|---|---|---|"]
        lines += [f"| {entry['display_name']} | {STATUS_WORDS[status]} | {scored} | {succeeded} of {scored} |"
                  for entry, status, scored, succeeded in rows]
    else:
        lines.append("No row of the trial could be shown.")
    if dropped:
        lines += [""] + _left_out(dropped)
    campaign = CAMPAIGN_WORDS.get(publication.get("status")) if isinstance(publication.get("status"), str) else None
    return lines + ["", f"Trial status: {campaign or NOT_RECORDED}. A zero is a valid result of the trial."]


def _cell(value: Any) -> str:
    return str(value if value not in (None, "") else NOT_RECORDED).replace("|", "/").replace("\n", " ")


def _full_fit_lines(table: Mapping[str, Any]) -> list[str]:
    rows = [row for row in table.get("rows") or [] if isinstance(row, Mapping)]
    lines = ["Source: the pipeline's robot fit table (`website_scene_preparation/handoff.json`).", ""]
    if not rows:
        return lines + ["The robot fit table has no rows."]
    lines += ["| Robot | Maker | Kind | Fit | Reasons |", "|---|---|---|---|---|"]
    for row in rows:
        lines.append("| " + " | ".join(_cell(v) for v in (
            row.get("name"), row.get("manufacturer"), BODY_WORDS.get(row.get("body_type"), row.get("body_type")),
            VERDICT_WORDS.get(row.get("verdict"), row.get("verdict")),
            "; ".join(str(r) for r in row.get("reasons") or []) or "none")) + " |")
    return lines + ["", "Checked against published specifications and the scene's estimated geometry. It is not "
                    "a test of any robot."]


def _full_trial_lines(publication: Mapping[str, Any] | None) -> list[str]:
    lines = [f"**{EARLY_TRIAL_LABEL}**", ""]
    if publication is None:
        return lines + ["No early trial result is included: none was given to the tool for this job."]
    lines += ["| Robot software | Status | Runs scored | Runs that met the success rule |", "|---|---|---|---|"]
    for row in [row for row in publication.get("candidates") or [] if isinstance(row, Mapping)]:
        scored, succeeded = _count(row.get("episodes_scored")), _count(row.get("episodes_succeeded"))
        lines.append("| " + " | ".join(_cell(v) for v in (
            row.get("display_name") or row.get("candidate_id"), row.get("status"), scored if scored is not None else INVALID,
            f"{succeeded} of {scored}" if scored is not None and succeeded is not None and succeeded <= scored
            else INVALID)) + " |")
    return lines + ["", f"Trial status: {_cell(publication.get('status'))}. A zero is a valid result of the trial."]


def _length(value: Any) -> str:
    number = _number(value)
    if value is None:
        return NOT_RECORDED
    if number is None or number <= 0:
        return INVALID
    return f"{number * 1000:.0f} mm" if number < 0.01 else f"{number * 100:.0f} cm"


def _point(value: Any) -> str:
    if value is None:
        return NOT_RECORDED
    if isinstance(value, (list, tuple)) and len(value) in (2, 3) and all(_number(v) is not None for v in value):
        return "(" + ", ".join(f"{float(v):.3f}" for v in value) + ") m"
    return INVALID


def _range(value: Any, unit: str) -> str:
    if value is None:
        return NOT_RECORDED
    if (isinstance(value, (list, tuple)) and len(value) == 2 and all(_number(v) is not None for v in value)
            and float(value[0]) <= float(value[1])):
        return f"{float(value[0]):.3g}–{float(value[1]):.3g} {unit}".strip()
    return INVALID


def _site_answer_words(assumption: Mapping[str, Any] | None, provenance: Mapping[str, Any]) -> str:
    if assumption:
        answer = assumption.get("site_answer")
        if answer == "confirmed":
            return ("These thresholds are Blueprint's standard ones, assumed when the scene was built; the site "
                    "confirmed the same thresholds later.")
        if answer == "this_is_wrong":
            return "These thresholds are Blueprint's standard ones; the site said they are wrong."
        return ("These thresholds are Blueprint's standard ones; no answer from the site is recorded in the "
                "files given.")
    if provenance:
        return (f"Thresholds written by: {_cell(provenance.get('author_source'))}; "
                f"status: {_cell(provenance.get('confirmation_status'))}.")
    return f"Who set these thresholds: {NOT_RECORDED}."


def done_sentence(contract: Mapping[str, Any] | None, *, object_name: str | None, place_label: str | None,
                  assumption: Mapping[str, Any] | None = None) -> str:
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
        radius = _length(target.get("radius_m"))
        within = f"within {radius}" if radius not in (NOT_RECORDED, INVALID) else f"within a distance {radius}"
        relation = "inside" if target.get("placement_relation") == "inside" else "on"
        parts.append(f"{thing}'s centre ends {within} of the point {_point(target.get('surface_position_world_m'))}, "
                     f"{relation} {where_label}")
        tilt = _number(target.get("maximum_tilt_rad"))
        parts.append(f"tilt: {NOT_RECORDED}" if target.get("maximum_tilt_rad") is None else f"tilt: {INVALID}"
                     if tilt is None or tilt < 0 else f"it is tilted no more than {math.degrees(tilt):.0f}°")
        still = _number(target.get("stable_seconds"))
        parts.append(f"how long it stays still: {NOT_RECORDED}" if target.get("stable_seconds") is None else
                     f"how long it stays still: {INVALID}" if still is None or still < 0 else f"it stays still for {still:g} s")
    else:
        bounds = _mapping(_mapping(criteria.get("destination_containment")).get("position_bounds_world_m"))
        low, high = bounds.get("minimum"), bounds.get("maximum")
        valid = (isinstance(low, list) and isinstance(high, list) and len(low) == len(high) == 3
                 and all(_number(v) is not None for v in (*low, *high)) and all(a <= b for a, b in zip(low, high, strict=True)))
        if valid:
            centre = [(float(a) + float(b)) / 2 for a, b in zip(low, high, strict=True)]
            half = [(float(b) - float(a)) / 2 for a, b in zip(low, high, strict=True)]
            parts.append(f"{thing}'s centre ends within ±{_length(half[0])} (x) and ±{_length(half[1])} (y) of the "
                         f"point {_point(centre)}, and within ±{_length(half[2])} of that height, on {where_label}")
        else:
            parts.append(f"where {thing} must end: {NOT_RECORDED if not bounds else INVALID} (on {where_label})")
        orientation = _mapping(criteria.get("orientation"))
        tolerance = _number(orientation.get("tolerance_rad"))
        if orientation.get("mode") == "ignored":
            parts.append("its turn is not checked")
        else:
            parts.append(f"it is turned no more than {math.degrees(tolerance):.0f}° from its set orientation"
                         if tolerance is not None and tolerance >= 0 else f"turn: {NOT_RECORDED}")
        settle = _length(_mapping(criteria.get("settling")).get("position_tolerance_m"))
        parts.append(f"it settles (moves less than {settle} at the end)" if settle not in (NOT_RECORDED, INVALID)
                     else f"settling: {settle}")
    support = _mapping(criteria.get("support"))
    if support.get("height_mode") == "ignored":
        parts.append("the height of what it rests on is not checked")
    else:
        heights = _range(support.get("height_interval_m"), "m")
        parts.append(f"it rests on a surface with its base {heights} high" if heights not in (NOT_RECORDED, INVALID)
                     else f"the height it must rest at: {heights}")
    gripper = _mapping(criteria.get("gripper_state")).get("mode")
    parts.append("the gripper has let go" if gripper == "released" else "the gripper's state is not checked"
                 if gripper == "ignored" else f"the gripper's state: {NOT_RECORDED}")
    lift_raw = _mapping(criteria.get("motion")).get("minimum_lift_m")
    lift = _length(lift_raw)
    parts.append(f"it was lifted at least {lift} on the way" if lift not in (NOT_RECORDED, INVALID)
                 else f"how high it must be lifted: {lift}")
    return ("Done means: " + "; ".join(parts) + ". "
            + _site_answer_words(assumption, _mapping(contract.get("provenance"))))


def _measured_and_estimated(preparation: Mapping[str, Any] | None, base_scene: Mapping[str, Any] | None,
                            fit_table: Mapping[str, Any] | None, publication: Mapping[str, Any] | None,
                            context: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    prep = _mapping(preparation)
    residual = _length(_mapping(_mapping(prep.get("registration")).get("task_region")).get("trimmed_rmse_m"))
    measured = [f"How closely the video's geometry and the reconstruction line up around the task (registration "
                f"residual, trimmed RMS): {residual}, in the scene's estimated metres."]
    frame = _mapping(prep.get("coordinate_frame"))
    band = _mapping(_mapping(fit_table).get("inputs")).get("scale_band")
    error = _number(_mapping(_mapping(prep.get("authoring_inputs")).get("metric_envelope")).get("maximum_dimension_relative_error"))
    estimated = [f"Scale: basis {_cell(frame.get('scale_authority'))}; physical scale measured: "
                 f"{_cell(frame.get('physical_scale_measured'))}. Scale band: "
                 + (_range(band, "×") if band is not None else
                    f"sizes may be off by up to ±{error * 100:.0f} %" if error is not None and error >= 0 else NOT_RECORDED)
                 + "."]
    subject = _mapping(prep.get("subject"))
    low, high = subject.get("aabb_min_xyz"), subject.get("aabb_max_xyz")
    if (isinstance(low, list) and isinstance(high, list) and len(low) == len(high) == 3
            and all(_number(v) is not None for v in (*low, *high)) and all(a <= b for a, b in zip(low, high, strict=True))):
        estimated.append("Object size: " + " × ".join(_length(b - a) for a, b in zip(low, high, strict=True))
                         + ", from the video.")
    else:
        estimated.append(f"Object size: {NOT_RECORDED}.")
    bounds = _mapping(_mapping(prep.get("physics")).get("bounds"))
    estimated.append(f"Mass: {_range(bounds.get('mass_kg'), 'kg')}, a prior from the object's size, not weighed.")
    estimated.append(f"Friction: static {_range(bounds.get('static_friction'), '')}, dynamic "
                     f"{_range(bounds.get('dynamic_friction'), '')}; prior ranges, not measured.")
    weight = _mapping(context.get("operator_task_details")).get("item_weight")
    if isinstance(weight, str) and weight.strip():
        estimated.append(f"The site stated the object's weight as \"{weight.strip()}\"; not checked.")
    codes = {row["code"] for row in _mapping(base_scene).get("scene_estimates") or []
             if isinstance(row, Mapping) and isinstance(row.get("code"), str)}
    codes |= {code for code in _mapping(_mapping(publication).get("scene_feasibility")).get("flags") or []
              if isinstance(code, str) and code.startswith("scene_estimate:")}
    for code in sorted(codes):
        estimated.append(f"`{code}`: {ESTIMATE_SENTENCES.get(code, 'an estimate the scene rests on (no plain sentence recorded for this code).')}")
    if not codes:
        estimated.append(f"Scene estimate flags (`scene_estimate:*`): none recorded{'' if base_scene else ' (base_scene.json not given)'}.")
    return measured, estimated


# --- binding the full level's inputs to this job --------------------------------------------------------

def _not_this_job(what: str) -> JobPacketRefused:
    return JobPacketRefused("job_packet_input_not_this_job", what)


def _video(inputs: JobInputs) -> tuple[Path, Path] | None:
    capture = inputs.capture_dir
    if inputs.video is not None:
        if inputs.video.is_symlink() or not inputs.video.resolve().is_relative_to(capture.resolve()):
            raise _not_this_job("video")
        _regular(inputs.video, inputs.video.name)
        return capture, inputs.video
    raw = capture / "raw"
    found = [path for path in (_walk(raw) if raw.is_dir() else []) if path.parent == raw
             and path.suffix.lower() in VIDEO_SUFFIXES]
    if len(found) > 1:
        raise JobPacketRefused("job_packet_video_ambiguous", "pass --video")
    return (raw, found[0]) if found else None


def _bind_full_inputs(inputs: JobInputs, context: Mapping[str, Any], team_record: Mapping[str, Any] | None,
                      plan: Mapping[str, Any] | None) -> None:
    scene, capture = context["scene_id"], context["capture_id"]
    if inputs.team_scene_dir is not None:
        contract = _mapping(_mapping(team_record).get("task_success_contract"))
        if team_record is None or plan is None or plan.get("scene_id") != scene \
                or _mapping(contract.get("scope")).get("site_id") != scene:
            raise _not_this_job("team scene")
    if inputs.configured_scene_dir is not None:
        revision = _read_json(inputs.configured_scene_dir, inputs.configured_scene_dir / "configured_scene_revision.v1.json")
        if (revision is None or team_record is None
                or revision.get("revision_digest") != team_record.get("scene_revision_digest")):
            raise _not_this_job("configured scene (bound through the team scene's revision digest)")
    if inputs.review_videos_dir is not None:
        manifest = _read_json(inputs.review_videos_dir, inputs.review_videos_dir / "manifest.json")
        if manifest is None or manifest.get("site_id") != scene or manifest.get("capture_id") != capture:
            raise _not_this_job("trial clips")


def _full_files(inputs: JobInputs, plan_dir: Path | None) -> list[tuple[Path, Path, str]]:
    """(root, source, packet path) for every file the full level copies."""
    rows: list[tuple[Path, Path, str]] = []
    video = _video(inputs)
    if video is not None:
        rows.append((video[0], video[1], f"files/video/{video[1].name}"))
    atlas = inputs.capture_dir / "pipeline" / "atlas"
    if atlas.is_dir():
        rows += [(atlas, path, f"files/simulation/reconstruction/{path.name}") for path in _walk(atlas)
                 if path.parent == atlas and path.suffix.lower() in RECONSTRUCTION_SUFFIXES]
    if inputs.configured_scene_dir is not None:
        root = inputs.configured_scene_dir
        for path in _walk(root):
            if path.parent != root:
                continue
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
        rows += [(root, path, f"files/early-trial/clips/{path.relative_to(root).as_posix()}") for path in _walk(root)
                 if path.suffix.lower() == ".mp4" or path.name == "manifest.json"]
    if inputs.first_look_file is not None:
        root, path = _file_argument(inputs.first_look_file)
        rows.append((root, path, f"files/first-look/{path.name}"))
    unique_destinations(rows)
    return rows


def unique_destinations(rows: list[tuple[Path, Path, str]]) -> None:
    destinations = [row[2] for row in rows]
    if len(set(destinations)) != len(destinations):
        raise JobPacketRefused("job_packet_file_name_collision")


def _code_revision() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                                timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return NOT_RECORDED
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", revision) else NOT_RECORDED


def _readme(*, level: str, files: list[str], preparation: Mapping[str, Any] | None,
            base_scene: Mapping[str, Any] | None) -> str:
    lines = ["# Job packet", "", TERMS_LINE, "",
             f"Level: **{level}**. `JOB_SHEET.md` describes the job; `manifest.json` lists every file with "
             "its size and SHA-256 and the rule that allowed this level.", "", "## Opening the simulation", ""]
    has = lambda prefix: any(path.startswith(prefix) for path in files)  # noqa: E731
    if level == "summary":
        lines.append("This summary carries no simulation files.")
    elif not has("files/simulation/") and not has("files/object/"):
        lines.append("This packet carries no simulation files: none were given to the tool.")
    else:
        if has("files/simulation/team-scene/"):
            lines.append("- `files/simulation/team-scene/`: OpenUSD for NVIDIA Isaac Sim. Blueprint runs these in "
                         "Isaac Sim 6.0.1 (NGC image). A `scene_appearance.usdc` holds the room's look as a ParticleField "
                         "Gaussian splat, which Isaac Sim 6 reads; other USD tools may show only the collision geometry "
                         "and the object.")
        if has("files/simulation/configured-scene/"):
            lines.append("- `files/simulation/configured-scene/`: the configured scene Blueprint built, zipped.")
        if has("files/object/"):
            lines.append("- `files/object/`: the job's object as USDZ, as Blueprint authored it from the video (its "
                         "size, mass and friction are estimates; see the sheet).")
        if has("files/simulation/reconstruction/"):
            lines.append("- `files/simulation/reconstruction/`: the raw reconstruction. `.spz` / `.ply` are Gaussian "
                         "splats (common splat viewers open them); `.glb` is a mesh (Blender or any glTF viewer).")
        frame = _mapping(_mapping(preparation).get("coordinate_frame"))
        lines += ["", f"Task coordinates: {_cell(frame.get('task_coordinates'))}; up axis "
                  f"{_cell(frame.get('declared_up_axis'))}; physical scale measured: "
                  f"{_cell(frame.get('physical_scale_measured'))}. The reconstruction's own up axis: "
                  f"{_cell(_mapping(base_scene).get('up_axis'))}."]
    for prefix, words in (("files/video/", "- `files/video/`: the job's video as uploaded."),
                          ("files/task/", "- `files/task/`: the sealed task records (success contract, scene plan)."),
                          ("files/early-trial/", "- `files/early-trial/`: the early trial's publication and clips."),
                          ("files/first-look/", "- `files/first-look/`: the website's first look, as exported.")):
        if has(prefix):
            lines += ["", words] if not lines[-1].startswith("- `files/") else [words]
    lines += ["", "## Who received this packet", "",
              "`manifest.json` has an empty `recipients` list on purpose: the register of which deployment company "
              "received which packet (by `packet_id`) is kept on the website side, not in this folder.", ""]
    return "\n".join(lines)


def build_job_packet(inputs: JobInputs, *, output_root: Path, level: str = "summary",
                     owned: OwnedCaptureApproval | None = None, operator: str | None = None,
                     now: datetime | None = None, owned_list: Path | None = None,
                     first_look_checked_by: str | None = None, redact_terms: Iterable[str] = ()) -> dict[str, Any]:
    """Check everything, then write ``packet-<random>/`` and its operator record under ``output_root``.

    ``first_look_checked_by`` and ``redact_terms`` are refused: a summary takes no free text, so there
    is nothing to check by hand or redact.
    """
    if level not in LEVELS:
        raise JobPacketRefused("job_packet_level_invalid")
    if first_look_checked_by or list(redact_terms):
        raise JobPacketRefused("job_packet_option_removed", "a summary takes no free text")
    if level == "summary" and inputs.first_look_file is not None:
        raise JobPacketRefused("job_packet_first_look_full_level_only")
    capture_dir = inputs.capture_dir
    if capture_dir.is_symlink() or not capture_dir.is_dir():
        raise JobPacketRefused("job_packet_input_missing", "capture directory")
    context = _read_json(capture_dir, capture_dir / "pipeline" / "website_task_context.json", required=True)
    if context.get("schema_version") != TASK_CONTEXT_SCHEMA:
        raise JobPacketRefused("job_packet_task_context_unsupported")
    refusal = job_state_refusal(capture_dir, context, pull_root=inputs.pull_root)
    if refusal:
        raise JobPacketRefused(refusal)
    if context.get("capture_id") != capture_dir.name or context.get("scene_id") != capture_dir.parent.parent.name:
        raise JobPacketRefused("job_packet_input_identity_mismatch", "task context")
    basis = permission_basis(level, context, owned=owned, owned_list=owned_list)
    full = level == "full"
    output_root = Path(output_root)
    if output_root.is_symlink() or not output_root.is_dir():
        raise JobPacketRefused("job_packet_output_root_invalid")

    read: dict[str, str] = {}  # input digests, for the operator record

    def record(root: Path, path: Path, value: Any) -> Any:
        if value is not None:
            read[path.name if root == path.parent else path.relative_to(root).as_posix()] = _sha256(_inside(root, path))
        return value

    record(capture_dir, capture_dir / "pipeline" / "website_task_context.json", context)
    handoff_path = capture_dir / "pipeline" / "website_scene_preparation" / "handoff.json"
    handoff = record(capture_dir, handoff_path, _read_json(capture_dir, handoff_path))
    preparation, prep_dir = _preparation(capture_dir, handoff)
    base_scene = _read_json(capture_dir, prep_dir / "base_scene.json") if prep_dir is not None else None
    fit_table = _mapping(_mapping(handoff).get("robot_fit_table")) or None
    publication = None
    if inputs.screening_publication is not None:
        root, path = _file_argument(inputs.screening_publication)
        publication = record(root, path, _read_json(root, path, required=True))
        if (publication.get("site_id") != context["scene_id"]
                or _mapping(publication.get("source")).get("capture_id") not in (None, context["capture_id"])):
            raise _not_this_job("early trial publication")
    team_record = plan = plan_dir = None
    if inputs.team_scene_dir is not None:
        root = inputs.team_scene_dir
        team_record = record(root, root / "team_scene_inputs.v1.json", _read_json(root, root / "team_scene_inputs.v1.json"))
        for name in ("native-task-packet-three-view", "native-task-packet"):
            plan = _read_json(root, root / name / "native_task_arena_scene_plan.v1.json")
            if plan is not None:
                plan_dir = root / name
                break
    if full:
        _bind_full_inputs(inputs, context, team_record, plan)
    copies = _full_files(inputs, plan_dir) if full else []
    kind = kind_of_job(preparation, context)

    # --- the sheet ---
    sheet = [f"# Job sheet ({level})", "", TERMS_LINE, ""]
    if full:
        criteria = _mapping(context.get("success_criteria"))
        details = _mapping(context.get("operator_task_details"))
        sheet += ["## The job, in the site's words", "", f"> {_cell(context.get('description'))}", "",
                  f"Kind of job: {kind}.", "", "## Constraints and what the site told us", "",
                  f"- The site's own success definition: {_cell(criteria.get('successDefinition'))}",
                  f"- Success rate the site wants: {_cell(criteria.get('successRate'))} %",
                  f"- Cycle time the site wants: {_cell(criteria.get('cycleTimeSeconds'))} s",
                  f"- Object weight (stated): {_cell(details.get('item_weight'))}",
                  f"- Object make and model (stated): {_cell(details.get('item_make_model'))}"]
        sheet += [f"- {_cell(key)}: {_cell(value)}" for key, value in sorted(_mapping(context.get("operator_answers")).items())]
        unresolved = context.get("unresolved")
        if isinstance(unresolved, list) and unresolved:
            sheet.append("- Still unanswered: " + ", ".join(_cell(item) for item in unresolved))
        sheet.append(f"- Other constraints: {NOT_RECORDED} (the task context has no constraints field).")
        prep = _mapping(preparation)
        subject, destination, support = (_mapping(prep.get(k)) for k in ("subject", "destination", "support"))
        low, high = subject.get("aabb_min_xyz"), subject.get("aabb_max_xyz")
        start = ([(float(a) + float(b)) / 2 for a, b in zip(low, high, strict=True)]
                 if _point(low) not in (NOT_RECORDED, INVALID) and _point(high) not in (NOT_RECORDED, INVALID)
                 and len(low) == len(high) else None)
        contract = _mapping(team_record).get("task_success_contract")
        target = _mapping(_mapping(_mapping(contract).get("criteria")).get("surface_target"))
        sheet += ["", "## The object and where it goes", "",
                  f"- Object: {_cell(subject.get('description'))}; starts at {_point(start)} on "
                  f"{_cell(support.get('description'))}.",
                  f"- Goes to: {_cell(target.get('visible_label') or destination.get('visible_label'))}; centre "
                  f"{_point(target.get('surface_position_world_m') or destination.get('position_world_m'))}; radius "
                  f"{_length(target.get('radius_m'))}.",
                  "", "## What counts as done", "",
                  done_sentence(contract, object_name=subject.get("description"),
                                place_label=destination.get("visible_label"),
                                assumption=_mapping(_mapping(team_record).get("owner_success_assumption")) or None), ""]
        measured, estimated = _measured_and_estimated(preparation, base_scene, fit_table, publication, context)
        sheet += ["## Measured", ""] + [f"- {line}" for line in measured]
        sheet += ["", "## Estimated", ""] + [f"- {line}" for line in estimated] + ["", "## First look on paper", ""]
        if inputs.first_look_file is not None:
            sheet += ["Source: the website's first look, exported by the operator; attached under `files/first-look/`.", ""]
        elif fit_table:
            sheet += _full_fit_lines(fit_table) + [""]
        else:
            sheet += [f"First look: {NOT_RECORDED}.", ""]
        sheet += ["## Early trial", ""] + _full_trial_lines(publication) + [""]
    else:
        sheet += ["## The job", "", f"Kind of job: {kind}.", "",
                  "Described in general words. The site's own description, its name and its location, the video, "
                  "the simulation and the success rule are not in this summary: the site has not given permission "
                  "to share them.", "", "## First look on paper", ""]
        sheet += summary_fit_lines(fit_table) + ["", "## Early trial", ""] + summary_trial_lines(publication) + [""]
    present = {row[2].split("/")[1] for row in copies}
    missing = [words for item, words in (("video", "the video"), ("simulation", "the simulation"),
                                         ("object", "the object asset"), ("task", "the sealed task records"),
                                         ("early-trial", "the early trial's files")) if full and item not in present]
    sheet += ["## What this packet does not show", "",
              "- No test at the site: everything here comes from one video and a simulation of it.",
              "- No safety assessment of any robot at this site."]
    sheet += [f"- {words[0].upper() + words[1:]}: not given to the tool, so not in this packet." for words in missing] + [""]

    # --- write: the packet folder, then the operator record beside it ---
    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    name = f"packet-{secrets.token_hex(8)}"
    packet, operator_record = output_root / name, output_root / f"{name}.operator-record.json"
    packet.mkdir()
    try:
        written = {"JOB_SHEET.md": "\n".join(sheet), "README.md": _readme(
            level=level, files=[row[2] for row in copies], preparation=preparation, base_scene=base_scene)}
        for filename, text in written.items():
            (packet / filename).write_text(text, encoding="utf-8")
        for root, source, relative in copies:
            target = packet / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_regular(_inside(root, source), source.name), target)
        listed = []
        for relative in sorted([*written, *(row[2] for row in copies)]):
            path = packet / relative
            listed.append({"path": relative, "size": path.stat().st_size, "sha256": _sha256(path)})
        packet_id = "sha256:" + hashlib.sha256("\n".join(f"{row['path']} {row['sha256']}" for row in listed)
                                                 .encode()).hexdigest()
        keys = {"request_id": context.get("request_id"), "scene_id": context["scene_id"],
                "capture_id": context["capture_id"]}
        manifest = {"schema_version": SCHEMA_VERSION, "packet_id": packet_id, "level": level,
                    "created_on": moment.date().isoformat(), "code_revision": _code_revision(),
                    "permission_basis": basis if full else {"rule": "summary_only"},
                    "first_look_source": ("website_export" if inputs.first_look_file is not None else
                                          "pipeline_robot_fit_table" if fit_table else NOT_RECORDED),
                    **({"job": keys, "takedown_keys": keys} if full else {}),
                    "files": listed, "recipients": []}
        (packet / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        operator_value = {
            "schema_version": OPERATOR_RECORD_SCHEMA, "packet_folder": name, "packet_id": packet_id, "level": level,
            "created_at": moment.isoformat().replace("+00:00", "Z"), "run_by": operator or NOT_RECORDED,
            "code_revision": manifest["code_revision"], "job": keys, "takedown_keys": {
                **keys, "withdrawal_tombstone": f"scenes/{context['scene_id']}/website_withdrawal/tombstone.json"},
            "permission_basis": basis, "input_sha256": dict(sorted(read.items())),
            "job_state_checked": {"capture_rights": {k: _mapping(context.get("capture_rights")).get(k)
                                                     for k in ("consent_status", "statement_version", "consent_revoked")},
                                  "withdrawal": "not withdrawn in the pulled site roots"},
        }
        operator_record.write_text(json.dumps(operator_value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(packet, ignore_errors=True)
        operator_record.unlink(missing_ok=True)
        raise
    return {**manifest, "packet_dir": str(packet), "operator_record": str(operator_record)}
