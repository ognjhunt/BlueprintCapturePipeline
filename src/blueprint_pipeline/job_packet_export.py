"""Write one website job's job packet folder from files the operator pulled from the host.

The job packet is the unit of the product (owner direction 2026-11-12): what an approved deployment company reads
to judge one job without a visit. Sharing is the service (owner decision 2026-11-12): a job is shared in full, agreed
once at submission. V1 is by hand: the operator pulls the job's files, runs this offline (no network, no door, no
provider), checks the folder and carries it. Nothing here sends it.

One packet holds, when each exists: a packet number, a status line, the site's own sentence and constraints, the
video without its sound, the first look on paper, the 3D reconstruction, the simulation and the success rule, the
early trial (a scripted run kept apart as a check of the simulation) and how to answer. A packet is never held for
what does not exist yet: the status line says what is not in it.

Who may get one: (a) the task context's ``capture_rights.statement_version`` is a recording statement that covers
sharing (``sharing_statement.SHARING_STATEMENT_VERSIONS``); a job under any older version is refused until the site
accepts the current one; or (b) the operator states the capture is Blueprint's own, names the approver, the date and
quotes the owner, and the capture id is on ``config/blueprint_owned_captures.v1.json`` (read by
``sharing_statement.owned_captures``).

What never goes in: the site's name or organisation, its street address, contact details, sign-in or upload tokens,
links and storage paths. Blueprint makes the introduction by hand. So the sheet is written from named fields only,
no record that carries such a field is copied, and the job's ids go in an operator record beside the folder.

Refused, as the Pipeline itself would treat the job:

- withdrawal: ``screening_site_withdrawal.site_withdrawal_state`` over the pull root (every
  ``<root>[/<bucket>]/scenes/<site>/website_withdrawal/tombstone.json``; a hidden folder is never a bucket,
  ``site_roots.is_bucket``) and ``website_atlas_preview_store.capture_withdrawn`` on the capture;
- delivery: ``consent_takedown.evaluate_delivery_time_takedown_gate`` must say ``allowed``, and every consent file
  present must parse;
- consent: ``consent_takedown.read_consent_state`` (its consent files) and
  ``consent_normalization.resolve_consent_signals`` on the task context, which must be ``active``;
- authority ended: ``pubsub_handoff_listener.AUTHORITY_ENDING_CODES`` and ``authority_ending_code`` over every text
  in the job ledger and its terminal receipt, and status ``terminal_authority_ended``;
- owner closed: ``website_run_closed_by_owner`` matched as a whole token.

Every file the tool reads must be pulled, or recorded absent on the host in the pull root's ``pull_record.json``:
a pull cut short is a refusal naming the missing file, never "nothing recorded". A record that cannot be read, or
is not a regular file, refuses too. Every refusal names the place or file it is about.
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
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .sharing_statement import OWNED_CAPTURES_PATH, SHARING_STATEMENT_VERSIONS, covers_sharing, owned_captures
from .site_roots import is_bucket

SCHEMA_VERSION = "job_packet_manifest.v3"
OPERATOR_RECORD_SCHEMA = "job_packet_operator_record.v2"
EARLY_TRIAL_LABEL = ("Early trial of openly available robot software. It does not predict what a "
                     "deployment company's own robot would do.")
TERMS_LINE = ("This packet is shared for evaluating one job. Do not pass it on, train on it, or keep it "
              "after you are asked to delete it.")
#: The recording statement versions that share a job in full with approved deployment companies are
#: ``sharing_statement.SHARING_STATEMENT_VERSIONS``. The website records the accepted version at
#: ``request.consent_attestation.statement_version``; it reaches the task context as
#: ``capture_rights.statement_version``. The last one is the current statement.
REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_CATALOG_PATH = REPO_ROOT / "config" / "beta_policy_candidates.v1.json"
TASK_CONTEXT_SCHEMA = "website_site_task_context.v1"
FIT_TABLE_SCHEMA = "website_robot_fit_table.v1"
#: ``job_retry_policy.OWNER_CLOSED_CODE`` in unreleased work; matched as its whole token, as there.
OWNER_CLOSED_CODE = "website_run_closed_by_owner"
_OWNER_CLOSED = re.compile(r"(?<![A-Za-z0-9_-])" + OWNER_CLOSED_CODE + r"(?![A-Za-z0-9_-])")
AUTHORITY_ENDED_STATUS = "terminal_authority_ended"
NOT_RECORDED = "not recorded"
INVALID = "not recorded (invalid value)"
GENERIC_KIND = "a pick-up-and-put-down job"
VIDEO_SUFFIXES = {".mov", ".mp4", ".m4v"}
USD_SUFFIXES = {".usd", ".usda", ".usdc", ".usdz"}
TASK_CONTEXT = "pipeline/website_task_context.json"
PREPARATION = "pipeline/website_scene_preparation"
TAKEDOWN_DIRECTORY = "pipeline/consent_takedown"
#: The consent files ``consent_takedown.read_consent_state`` reads, checked to be regular files first.
CONSENT_FILES = ("raw/rights_consent.json", "rights_consent.json", "raw/manifest.json", "capture_descriptor.json")
#: The job's own records the hard stops read, beside the task context.
JOB_RECORDS = ("pipeline_job_ledger.json", "pipeline_job_terminal_receipt.json")
#: The checks ``website_robot_fit`` writes (its ``_check`` calls) and their results, in fixed words.
CHECK_WORDS = {"stance": "a place to stand", "reach": "reach", "payload": "payload",
               "cycle_time": "cycle time", "access": "access from the entrance"}
#: Every result ``website_robot_fit._check`` writes; ``not_requested`` (no cycle-time target) is not a failed check.
RESULT_WORDS = {"pass": "passed", "fail": "failed", "unknown": "unknown", "not_requested": "no target given"}
NOT_A_FAILED_CHECK = {"pass", "not_requested"}
SCRIPTED_CHECK_TITLE = "Check of the simulation (scripted, not robot software)"
SCRIPTED_CHECK_SENTENCE = ("We ran a scripted pick and place in this simulation to check the simulation itself. "
                           "It says nothing about any robot's software.")
VERDICT_WORDS = {"fits": "fits (its carry was path-checked)", "fits_spec": "fits on published figures",
                 "fits_if": "fits if the conditions below hold", "doesnt_fit": "does not fit", "unknown": "unknown"}
BODY_WORDS = {"fixed_arm": "fixed arm", "mobile_manipulator": "arm on a moving base", "humanoid": "humanoid",
              "wheeled_humanoid": "humanoid on wheels"}
STATUS_WORDS = {"pending": "waiting to run", "waitlisted": "waiting for budget", "preparing": "being prepared",
                "launching": "starting", "running": "running", "reconciliation_required": "being reconciled",
                "completed": "completed", "blocked": "stopped", "unavailable": "could not run",
                "deferred": "not run"}
CAMPAIGN_WORDS = {"completed": "completed", "running": "still running", "queued": "waiting to run",
                  "blocked": "stopped", "reconciliation_required": "being reconciled"}
MAX_TRIES = 1000
#: The most left-out rows the sheet counts out loud; above it, "more than" this.
MAX_SHOWN_LEFT_OUT = 20
MAX_APPROVAL_AGE_DAYS = 90
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
    """A refusal with a typed code and the place it is about; nothing has been written."""

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
    video: Path | None = None
    #: The directory the operator pulled into, laid out as the host's site roots
    #: (``<root>[/<bucket>]/scenes/site-<request>/captures/<capture>``) with its ``pull_record.json``.
    #: Required: a call without it is refused.
    pull_root: Path | None = None


@dataclass(frozen=True)
class OwnedCaptureApproval:
    capture_id: str
    approved_by: str
    approval_evidence: str
    approved_on: str = ""


# --- contact details in the site's own words ------------------------------------------------------------

REMOVED = "[removed]"
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
_LINK = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"']*[^\s<>\"'.,;:!?)]")
_NUMBER_RUN = re.compile(r"(?<![\w.,])(?:\+|00)?\(?\d[\d().\- ]*\d\)?(?![\w])")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _phone(run: str) -> bool:
    """A run of digits and separators that is a phone number: at least 7 digits, not a decimal, date, range or
    thousands-grouped count."""
    run = run.strip()
    digits = sum(ch.isdigit() for ch in run)
    if digits < 7 or _ISO_DATE.fullmatch(run):
        return False
    if run.startswith(("+", "00", "(")):
        return True
    if "." in run and run.count(".") < 2:
        return False  # a decimal, such as 3.1415926
    groups = re.split(r"[().\- ]+", run.strip("()"))
    if len(groups) > 1 and all(len(group) == 3 for group in groups[1:]) and len(groups[0]) <= 3 and " " in run:
        return False  # thousands grouped by spaces, such as 12 500 000
    if (len(groups) == 2 and "-" in run and len(groups[0]) == len(groups[1]) <= 4
            and int(groups[0]) < int(groups[1])):
        return False  # a range, such as 1000-2000
    return True


def redact_contacts(text: str) -> tuple[str, bool]:
    """The text with every email address, web link and phone number replaced by ``[removed]``, and whether any was."""
    found = False

    def removed(_match: re.Match[str]) -> str:
        nonlocal found
        found = True
        return REMOVED

    text = _EMAIL.sub(removed, _LINK.sub(removed, text))

    def phone(match: re.Match[str]) -> str:
        nonlocal found
        if not _phone(match.group(0)):
            return match.group(0)
        found = True
        tail = match.group(0)[len(match.group(0).rstrip()):]
        return REMOVED + tail
    return _NUMBER_RUN.sub(phone, text), found


def _redacted(value: Any, removed: list[bool]) -> Any:
    """Every text inside ``value`` passed through ``redact_contacts``; ``removed`` records whether anything was."""
    if isinstance(value, str):
        text, found = redact_contacts(value)
        removed.append(found)
        return text
    if isinstance(value, Mapping):
        return {_redacted(key, removed) if isinstance(key, str) else key: _redacted(item, removed)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_redacted(item, removed) for item in value]
    return value


# --- reading inputs: regular files only, never outside the directory named -----------------------------

def _inside(root: Path, path: Path) -> Path:
    """The real path behind ``path``; refused when it (or a link on the way) leaves ``root``."""
    real_root, real = root.resolve(), path.resolve()
    if real != real_root and not real.is_relative_to(real_root):
        raise JobPacketRefused("job_packet_input_symlink_outside_inputs", str(path))
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
        raise JobPacketRefused("job_packet_input_symlink_outside_inputs", str(path))
    if not os.path.lexists(path):
        raise JobPacketRefused("job_packet_input_missing", str(path))
    _regular(path, str(path))
    return path.parent, path


def _walk(root: Path) -> list[Path]:
    """Every regular file under ``root``, in order. Links must stay inside; anything else is refused."""
    if root.is_symlink() or not root.is_dir():
        raise JobPacketRefused("job_packet_input_missing", str(root))
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
            _regular(real, str(path.relative_to(root)))
            files.append(path)
    return files


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate key")
    return dict(pairs)


def _read_json(root: Path, path: Path, *, required: bool = False) -> dict[str, Any] | None:
    name = path.relative_to(root).as_posix() if path.is_relative_to(root) and path != root else path.name
    if not os.path.lexists(path):
        if required:
            raise JobPacketRefused("job_packet_input_missing", name)
        return None
    real = _regular(_inside(root, path), name)
    try:
        value = json.loads(real.read_bytes().decode("utf-8"), object_pairs_hook=_unique_pairs)
    except (OSError, UnicodeError, ValueError) as exc:
        raise JobPacketRefused("job_packet_input_unreadable", name) from exc
    if not isinstance(value, dict):
        raise JobPacketRefused("job_packet_input_unreadable", name)
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _count(value: Any, *, low: int = 0, high: int = MAX_TRIES) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and low <= value <= high else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _in_capture(capture_dir: Path, host_path: Any) -> str | None:
    """A host path inside this capture (``.../scenes/<site>/captures/<capture>/<rest>``) as ``<rest>``, else None."""
    marker = f"/scenes/{capture_dir.parent.parent.name}/captures/{capture_dir.name}/"
    if not isinstance(host_path, str) or host_path.count(marker) != 1:
        return None
    rest = host_path.split(marker, 1)[1]
    return None if any(part in {"", ".", ".."} for part in rest.split("/")) else rest


# --- the pull record -----------------------------------------------------------------------------------

PULL_RECORD = "pull_record.json"
PULL_RECORD_SCHEMA = "job_packet_pull_record.v1"
WITHDRAWAL_NOT_PULLED = "job_packet_withdrawal_state_not_pulled"
INPUT_NOT_PULLED = "job_packet_input_not_pulled"


@dataclass(frozen=True)
class PullRecord:
    """The operator's own statement of what was pulled and what the host lacked.

    ``buckets`` is every site-root bucket the host has (``""`` for the root itself, from ``door ls``; a hidden name
    is not a bucket and is ignored) and ``absent_on_host`` every place the door reported missing, relative to the
    pull root. A place neither pulled nor recorded absent is a refusal naming it.
    """

    root: Path
    capture: str
    buckets: tuple[str, ...]
    absent: frozenset[str]
    value: dict[str, Any]

    def pulled(self, place: str, *, code: str = INPUT_NOT_PULLED, directory: bool = False) -> bool:
        """True when ``place`` was pulled, False when it is recorded absent on the host."""
        here = self.root / place
        if place in self.absent:
            if os.path.lexists(here):
                raise JobPacketRefused(code, f"{place} is recorded absent but present")
            return False
        # A file that is there is checked by its reader (a link, a pipe or a hard link is refused there).
        if (here.is_symlink() or not here.is_dir()) if directory else not os.path.lexists(here):
            raise JobPacketRefused(code, f"pull {place}, or record it absent")
        return True

    def in_capture(self, relative: str, **kwargs: Any) -> bool:
        return self.pulled(f"{self.capture}/{relative}", **kwargs)


def _layout(capture_dir: Path, pull_root: Path | None) -> tuple[Path, str]:
    """The pull root and the capture's bucket ('' at the root), when the layout is the host's; else a refusal."""
    if pull_root is None:
        raise JobPacketRefused("job_packet_pull_root_required", "pass --pull-root")
    root = Path(pull_root)
    if root.is_symlink() or not root.is_dir():
        raise JobPacketRefused("job_packet_withdrawal_state_unknown", f"pull root {root} is not a directory")
    try:
        relative = capture_dir.absolute().relative_to(root.absolute()).parts
    except ValueError:
        raise JobPacketRefused("job_packet_withdrawal_state_unknown", f"{capture_dir} is outside the pull root") from None
    shape = relative[-4:]
    if (len(relative) not in (4, 5) or shape[0] != "scenes" or not shape[1].startswith("site-")
            or shape[2] != "captures" or any(part.startswith(".") for part in relative)):
        raise JobPacketRefused("job_packet_withdrawal_state_unknown",
                               f"{capture_dir} is not laid out as <root>[/<bucket>]/scenes/site-<request>/captures/<capture>")
    return root, relative[0] if len(relative) == 5 else ""


def _pull_record(root: Path, capture_dir: Path, bucket: str) -> PullRecord:
    try:
        value = _read_json(root, root / PULL_RECORD, required=True)
    except JobPacketRefused as exc:
        raise JobPacketRefused(WITHDRAWAL_NOT_PULLED, f"write {PULL_RECORD} at the pull root ({exc})") from exc
    buckets, absent = value.get("buckets"), value.get("absent_on_host")
    if (value.get("schema_version") != PULL_RECORD_SCHEMA or not isinstance(buckets, list) or not isinstance(absent, list)
            or not all(isinstance(item, str) and "/" not in item and item not in {".", ".."} for item in buckets)
            or not all(isinstance(item, str) for item in absent) or bucket not in buckets):
        raise JobPacketRefused(WITHDRAWAL_NOT_PULLED, f"{PULL_RECORD} is not a {PULL_RECORD_SCHEMA} record naming the "
                               f"capture's bucket {bucket!r}")
    listed = tuple(item for item in buckets if item == "" or is_bucket(item))
    local = {""} if (root / "scenes").is_dir() else set()
    local |= {entry.name for entry in root.iterdir() if is_bucket(entry.name) and entry.is_dir()
              and not entry.is_symlink() and (entry / "scenes").is_dir()}
    if not local <= set(listed):
        raise JobPacketRefused(WITHDRAWAL_NOT_PULLED, f"pulled bucket(s) {sorted(local - set(listed))} are not in {PULL_RECORD}")
    return PullRecord(root=root, capture=capture_dir.absolute().relative_to(root.absolute()).as_posix(),
                      buckets=listed, absent=frozenset(absent), value=value)


# --- the hard stops -----------------------------------------------------------------------------------

def _ended_or_closed(texts: list[str]) -> str | None:
    from .pubsub_handoff_listener import AUTHORITY_ENDING_CODES, authority_ending_code
    if any(_OWNER_CLOSED.search(text) for text in texts):
        return "job_packet_job_closed_by_owner"
    token = re.compile(r"(?<![A-Za-z0-9_-])(" + "|".join(map(re.escape, sorted(AUTHORITY_ENDING_CODES))) + r")(?![A-Za-z0-9_-])")
    if any(token.search(text) or authority_ending_code(ValueError(text)) for text in texts):
        return "job_packet_capture_authority_ended"
    return None


def _keys_and_texts(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _keys_and_texts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys_and_texts(item)
    elif isinstance(value, str):
        yield value


def _expired(value: Any, now: datetime) -> bool:
    """Any ``*expires_at*`` field at or before now, or unreadable, in the task context or its rights."""
    for source in (value, _mapping(_mapping(value).get("capture_rights"))):
        for key, item in _mapping(source).items():
            if "expires_at" not in str(key) or item is None:
                continue
            try:
                moment = datetime.fromisoformat(str(item).replace("Z", "+00:00"))
            except ValueError:
                return True
            if moment.tzinfo is None or moment <= now:
                return True
    return False


def check_job_state(capture_dir: Path, context: Mapping[str, Any], *, pull_root: Path | None = None,
                    now: datetime | None = None) -> PullRecord:
    """Refuse a job that is withdrawn, not consented, ended or closed, or whose state was not pulled; else its pull record."""
    from .consent_normalization import resolve_consent_signals
    from .consent_takedown import TAKEDOWN_MANIFEST_RELATIVE_PATH, evaluate_delivery_time_takedown_gate
    from .screening_site_withdrawal import SITE_ROOTS_ENV, site_withdrawal_state
    from .website_atlas_preview_store import capture_withdrawn

    root, bucket = _layout(capture_dir, pull_root)
    site = capture_dir.parent.parent.name
    # Whatever a tombstone is (a file, a directory, a link, a pipe), its presence is a withdrawal; answering
    # first keeps a pipe from ever being opened by the pipeline's readers below. Looked for beside every folder,
    # hidden or not: what is not a bucket need not be pulled, but a tombstone found anywhere still refuses.
    places = [root / "scenes" / site] + [entry / "scenes" / site for entry in root.iterdir() if entry.name != "scenes"]
    for place in places:
        tombstone = place / "website_withdrawal" / "tombstone.json"
        if os.path.lexists(tombstone):
            raise JobPacketRefused("job_packet_capture_withdrawn", tombstone.relative_to(root).as_posix())
    site_state = site_withdrawal_state(site, environ={SITE_ROOTS_ENV: str(root)})
    if site_state == "withdrawn":
        raise JobPacketRefused("job_packet_capture_withdrawn", f"scenes/{site}/website_withdrawal in the pulled site roots")
    if site_state != "not_withdrawn":
        raise JobPacketRefused("job_packet_withdrawal_state_unknown", f"scenes/{site}/website_withdrawal could not be read")
    pull = _pull_record(root, capture_dir, bucket)
    for name in pull.buckets:
        pull.pulled(f"{name}/scenes/{site}/website_withdrawal".lstrip("/"), code=WITHDRAWAL_NOT_PULLED, directory=True)
    pull.in_capture(TAKEDOWN_DIRECTORY, code=WITHDRAWAL_NOT_PULLED, directory=True)
    for relative in CONSENT_FILES:
        pull.in_capture(relative, code=WITHDRAWAL_NOT_PULLED)
    for relative in JOB_RECORDS:
        pull.in_capture(relative)
    # Every consent or takedown file the pipeline reads must be a regular file that parses as an object;
    # a present file that does not is a refusal, never skipped.
    consent_records = []
    for relative in (*CONSENT_FILES, TAKEDOWN_MANIFEST_RELATIVE_PATH):
        if os.path.lexists(capture_dir / relative):
            try:
                consent_records.append((relative, _read_json(capture_dir, capture_dir / relative)))
            except JobPacketRefused as exc:
                raise JobPacketRefused("job_packet_consent_state_unreadable", relative) from exc
    if capture_withdrawn(capture_dir):
        raise JobPacketRefused("job_packet_capture_withdrawn", "the capture's consent source")
    for relative, value in consent_records:  # each one, not only the first the reader stops at
        if relative == TAKEDOWN_MANIFEST_RELATIVE_PATH:
            continue
        signals = resolve_consent_signals(value)
        if signals["state"] == "revoked":
            raise JobPacketRefused("job_packet_capture_withdrawn", relative)
        if signals["has_consent_fields"] and signals["state"] != "active":
            raise JobPacketRefused("job_packet_consent_not_active", relative)
    signals = resolve_consent_signals(context)
    where = f"{TASK_CONTEXT} capture_rights"
    if signals["state"] == "revoked":
        raise JobPacketRefused("job_packet_capture_withdrawn", where)
    if signals["state"] != "active":
        raise JobPacketRefused("job_packet_consent_not_active", where)
    if _expired(context, now or datetime.now(timezone.utc)):
        raise JobPacketRefused("job_packet_consent_expired", where)
    rights = context.get("capture_rights")
    if (not isinstance(rights, Mapping) or rights.get("derived_scene_generation_allowed") is not True
            or rights.get("consent_revoked") is not False):
        raise JobPacketRefused("job_packet_capture_rights_not_granted", where)
    records = {}
    for relative in JOB_RECORDS:
        try:
            records[relative] = _read_json(capture_dir, capture_dir / relative)
        except JobPacketRefused as exc:
            raise JobPacketRefused("job_packet_job_state_unreadable", relative) from exc
    ledger, receipt = records["pipeline_job_ledger.json"], records["pipeline_job_terminal_receipt.json"]
    if ledger is None:
        raise JobPacketRefused("job_packet_job_ledger_missing", "pipeline_job_ledger.json")
    ended = _ended_or_closed([*_keys_and_texts(ledger), *_keys_and_texts(receipt or {})])
    if ended:
        raise JobPacketRefused(ended, "pipeline_job_ledger.json or pipeline_job_terminal_receipt.json")
    if ledger.get("status") == AUTHORITY_ENDED_STATUS or (receipt or {}).get("status") == "authority_ended":
        raise JobPacketRefused("job_packet_capture_authority_ended", "pipeline_job_ledger.json or pipeline_job_terminal_receipt.json")
    if ledger.get("status") == "corrupt" or not isinstance(ledger.get("status"), str):
        raise JobPacketRefused("job_packet_job_state_unreadable", "pipeline_job_ledger.json")
    # A packet is a delivery: the pipeline's own serve-time gate must allow it.
    gate = evaluate_delivery_time_takedown_gate(capture_root=capture_dir, surface="job_packet")
    if gate.get("status") != "allowed":
        source = _mapping(gate.get("consent")).get("source_path")
        raise JobPacketRefused("job_packet_delivery_gate_" + re.sub(r"[^a-z0-9_]", "_", str(gate.get("status"))),
                               f"{TAKEDOWN_MANIFEST_RELATIVE_PATH}; consent source: "
                               + (os.path.relpath(source, capture_dir) if source else "none found"))
    return pull


def recorded_video_sha256(capture_dir: Path, pull: PullRecord) -> str | None:
    """The sha256 the staging manifest recorded for the capture's raw video when it was staged, or None."""
    if not pull.in_capture("pipeline_staging_manifest.json"):
        return None
    staging = _read_json(capture_dir, capture_dir / "pipeline_staging_manifest.json")
    name = _mapping(staging.get("raw_video")).get("object_name")
    digests = [row.get("sha256") for row in staging.get("objects") or [] if isinstance(row, Mapping)
               and isinstance(name, str) and row.get("name") == name]
    digest = digests[0] if len(digests) == 1 else None
    return digest.removeprefix("sha256:") if isinstance(digest, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", digest) else None


def permission_basis(context: Mapping[str, Any], *, owned: OwnedCaptureApproval | None,
                     owned_list: Path | None = None, now: datetime | None = None,
                     video_sha256: str | None = None) -> dict[str, Any]:
    """Which rule allows this job to be shared, or a refusal."""
    version = _mapping(context.get("capture_rights")).get("statement_version")
    if covers_sharing(version):
        return {"rule": "site_sharing_permission", "statement_version": version}
    if owned is None:
        raise JobPacketRefused("job_packet_sharing_statement_not_accepted",
                               f"the site must accept the current statement ({SHARING_STATEMENT_VERSIONS[-1]}) "
                               f"first; {TASK_CONTEXT} capture_rights.statement_version is {version!r}")
    try:
        approved_on = date.fromisoformat(owned.approved_on.strip()) if isinstance(owned.approved_on, str) else None
    except ValueError:
        approved_on = None
    if not owned.approved_by.strip() or not owned.approval_evidence.strip() or approved_on is None:
        raise JobPacketRefused("job_packet_owned_capture_approval_missing",
                               "--approved-by, --approved-on (YYYY-MM-DD) and --approval-evidence")
    today = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).date()
    if approved_on > today or (today - approved_on).days > MAX_APPROVAL_AGE_DAYS:
        raise JobPacketRefused("job_packet_owned_capture_approval_out_of_date",
                               f"--approved-on must be within the last {MAX_APPROVAL_AGE_DAYS} days")
    if owned.capture_id != context.get("capture_id"):
        raise JobPacketRefused("job_packet_owned_capture_id_mismatch", f"--capture-id against {TASK_CONTEXT} capture_id")
    path = owned_list or OWNED_CAPTURES_PATH
    # The one reader of the allow-list (``sharing_statement``); a list it cannot read is a refusal here.
    rows, flag = owned_captures(path)
    if flag is not None:
        raise JobPacketRefused("job_packet_owned_capture_list_invalid", str(path))
    if owned.capture_id not in rows:
        raise JobPacketRefused("job_packet_capture_not_blueprint_owned", f"the capture id is not on {path.name}")
    # Every entry names one video (the list's rule): a different video uploaded later under the same capture id
    # is not covered.
    prefix = rows[owned.capture_id]
    if video_sha256 is None:
        raise JobPacketRefused("job_packet_owned_capture_video_not_recorded",
                               "pipeline_staging_manifest.json records no sha256 for the capture's raw video")
    if not video_sha256.startswith(prefix):
        raise JobPacketRefused("job_packet_owned_capture_video_mismatch",
                               f"the video recorded in pipeline_staging_manifest.json is not the one {path.name} names")
    return {"rule": "blueprint_owned_capture", "capture_id": owned.capture_id,
            "approved_by": owned.approved_by.strip(), "approved_on": approved_on.isoformat(),
            "approval_evidence": owned.approval_evidence.strip(), "allow_list_sha256": _sha256(path),
            "video_sha256": video_sha256}


# --- what the records say -----------------------------------------------------------------------------

def _preparation(capture_dir: Path, handoff: Mapping[str, Any] | None, pull: PullRecord) -> tuple[dict | None, Path | None]:
    """The preparation the handoff names (by digest), with the directory it sits in."""
    base = capture_dir / PREPARATION
    candidates = [base / "preparation.json"] if pull.in_capture(f"{PREPARATION}/preparation.json") else []
    if pull.in_capture(f"{PREPARATION}/revisions", directory=True):
        candidates += sorted((base / "revisions").glob("*/preparation.json"))
    found = [(path, _read_json(capture_dir, path)) for path in candidates]
    wanted = _mapping(handoff).get("preparation_digest")
    chosen = [row for row in found if wanted and row[1].get("digest") == wanted]
    if not chosen and len(found) == 1:
        chosen = found
    return (chosen[0][1], chosen[0][0].parent) if len(chosen) == 1 else (None, None)


def _fit_table(capture_dir: Path, handoff: Mapping[str, Any] | None, preparation: Mapping[str, Any] | None,
               context: Mapping[str, Any], pull: PullRecord) -> tuple[dict | None, str]:
    """The fit table the handoff points to, read and checked as ``website_robot_fit_projection`` does.

    The table and ``"shown"``; or None and why: ``"none"`` (no table for this job), ``"not_on_host"`` (recorded
    absent), or ``"not_verified"`` (not the file the handoff names, or not bound to this job's preparation).
    """
    from .decision_evidence_contracts import canonical_digest
    from .website_robot_fit import VERDICTS
    hand = _mapping(handoff)
    reference = (_mapping(hand.get("fit_check")).get("robot_fit_table") if hand.get("atlas_route") == "geometry_only"
                 else hand.get("robot_fit_table"))
    if not isinstance(reference, Mapping):
        return None, "none"
    relative = _in_capture(capture_dir, reference.get("path"))
    if relative is None:
        return None, "not_verified"
    if not pull.in_capture(relative):
        return None, "not_on_host"
    table, prep = _read_json(capture_dir, capture_dir / relative), _mapping(preparation)
    rows = table.get("rows")
    if (hand.get("digest") != canonical_digest(hand, digest_field="digest")
            or prep.get("digest") != canonical_digest(prep, digest_field="digest")
            or _mapping(prep.get("binding")).get("task_context_digest") != context.get("context_digest")
            or hand.get("preparation_digest") != prep.get("digest")
            or table.get("schema_version") != FIT_TABLE_SCHEMA or table.get("preparation_digest") != prep.get("digest")
            or table.get("table_digest") != canonical_digest(table, digest_field="table_digest")
            or table.get("table_digest") != reference.get("digest")
            or table.get("evidence_label") != "spec_bounded_estimate"
            or table.get("physical_world_truth_claimed") is not False
            or not isinstance(rows, list) or any(_mapping(row).get("verdict") not in VERDICTS for row in rows)):
        return None, "not_verified"
    return table, "shown"


def kind_of_job(preparation: Mapping[str, Any] | None, context: Mapping[str, Any]) -> str:
    """The kind of job in fixed words, chosen from structured fields."""
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
    try:
        from .robot_spec_catalog import load_robot_spec_catalog
        return {row["model_id"]: row for row in load_robot_spec_catalog()["models"]}
    except Exception as exc:  # noqa: BLE001 - any catalog failure is one clean refusal
        raise JobPacketRefused("job_packet_robot_catalog_unreadable", type(exc).__name__) from exc


def _policy_catalog() -> dict[str, Mapping[str, Any]]:
    # The reviewed checkout's catalog, never the host override: its names are what the sheet prints.
    try:
        from .beta_policy_candidate_catalog import load_catalog
        return {row["candidate_id"]: row for row in load_catalog(POLICY_CATALOG_PATH)["candidates"]}
    except Exception as exc:  # noqa: BLE001 - e.g. BetaPolicyCatalogError for a group-writable file
        raise JobPacketRefused("job_packet_policy_catalog_unreadable", f"{POLICY_CATALOG_PATH} ({type(exc).__name__})") from exc


def _left_out(dropped: int) -> list[str]:
    """One fixed line; the count is capped so a number of rows cannot carry a message."""
    if not dropped:
        return []
    shown = str(dropped) if dropped <= MAX_SHOWN_LEFT_OUT else f"more than {MAX_SHOWN_LEFT_OUT}"
    return ["", f"Some rows were left out ({shown}): not a known name, a repeat, a value outside the fixed "
            "sets this tool knows, or counts out of range."]


def _range(value: Any, unit: str) -> str:
    if value is None:
        return NOT_RECORDED
    if (isinstance(value, (list, tuple)) and len(value) == 2 and all(_number(v) is not None for v in value)
            and float(value[0]) <= float(value[1])):
        return f"{float(value[0]):.3g}–{float(value[1]):.3g} {unit}".strip()
    return INVALID


def _conditions(row: Mapping[str, Any], checks: list) -> list[tuple[str, str]] | None:
    """A ``fits_if`` row's conditions as (check, the Pipeline's sentence), or None when the row is not one this
    tool can word: every number checked, every word from ``condition_sentence``, never text from the table."""
    from .website_robot_fit_projection import condition_sentence, published_condition
    conditions = row.get("conditions")
    open_checks = {c["check"] for c in checks if c["result"] == "unknown"}
    if (row.get("verdict") != "unknown" or row.get("verdict_if_conditions_hold") not in ("fits", "fits_spec")
            or row.get("missing") or any(c["result"] == "fail" for c in checks)
            or not isinstance(conditions, list) or not conditions):
        return None
    worded = []
    try:
        for condition in conditions:
            published_condition(condition)  # a quantity, unit, origin or number outside the fixed sets: ValueError
            worded.append((condition["check"], condition_sentence(condition)))
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    return worded if {check for check, _ in worded} == open_checks else None


def fit_lines(table: Mapping[str, Any]) -> list[str]:
    """The first look from the verified fit table: names from the catalog, verdicts and checks in fixed words.

    A ``fits_if`` row (``first_look_verdict``) is shown with each condition in the Pipeline's own sentence
    (``website_robot_fit_projection.condition_sentence``), and rows are then listed fits, fits if, does not
    fit, unknown. A table without such rows reads as before, in its own order.
    """
    from .website_robot_fit import VERDICTS
    from .website_robot_fit_projection import FIRST_LOOK_VERDICTS, FITS_IF, row_order
    catalog, rows, dropped, seen = _robot_catalog(), [], 0, set()
    for row in table.get("rows") if isinstance(table.get("rows"), list) else []:
        model = catalog.get(row.get("model_id")) if isinstance(row, Mapping) and isinstance(row.get("model_id"), str) else None
        checks = row.get("checks") if isinstance(row, Mapping) and isinstance(row.get("checks"), list) else None
        verdict = row.get("first_look_verdict", row.get("verdict")) if isinstance(row, Mapping) else None
        if (model is None or model["model_id"] in seen or row.get("verdict") not in VERDICTS or checks is None
                or any(not isinstance(c, Mapping) or c.get("check") not in CHECK_WORDS
                       or c.get("result") not in RESULT_WORDS for c in checks)
                or verdict not in FIRST_LOOK_VERDICTS or (verdict != FITS_IF and verdict != row["verdict"])):
            dropped += 1
            continue
        conditions = _conditions(row, checks) if verdict == FITS_IF else []
        if conditions is None:
            dropped += 1
            continue
        seen.add(model["model_id"])
        rows.append((model, verdict, checks, conditions, row))
    lines = ["Source: the pipeline's robot fit table.", ""]
    if not rows:
        return lines + ["No row of the table could be shown.", *_left_out(dropped)]
    if any(verdict == FITS_IF for _, verdict, _, _, _ in rows):
        rows.sort(key=lambda shown: row_order(shown[4]))
    lines += ["| Robot | Maker | Kind | Fit | Checks not passed |", "|---|---|---|---|---|"]
    for model, verdict, checks, conditions, _ in rows:
        conditional = {check for check, _ in conditions}
        open_checks = [f"{CHECK_WORDS[c['check']]} (a condition below)" if c["check"] in conditional
                       else f"{CHECK_WORDS[c['check']]} ({RESULT_WORDS[c['result']]})" for c in checks
                       if c["result"] not in NOT_A_FAILED_CHECK]
        lines.append(f"| {model['name']} | {model['manufacturer']} | {BODY_WORDS.get(model['body_type'], 'robot')} | "
                     f"{VERDICT_WORDS[verdict]} | {', '.join(open_checks) or 'none'} |")
    worded = [f"- {model['name']}, {CHECK_WORDS[check]}: {sentence}." for model, _, _, conditions, _ in rows
              for check, sentence in conditions]
    if worded:
        lines += ["", "Conditions: what each robot that fits if needs, beside what the job seems to need.", *worded]
    counts = Counter(verdict for _, verdict, _, _, _ in rows)
    lines += ["", "Counts: " + ", ".join(f"{VERDICT_WORDS[k]}: {counts[k]}" for k in FIRST_LOOK_VERDICTS if counts[k]),
              *_left_out(dropped)]
    results = [(c["check"], c["result"]) for _, _, checks, _, _ in rows for c in checks]
    ran = [CHECK_WORDS[name] for name in CHECK_WORDS if any(c == name and r != "not_requested" for c, r in results)]
    lines += ["", f"Checked against published specifications and the scene's estimated geometry: {', '.join(ran)}. "
              "It is not a test of any robot."]
    if any(c == "cycle_time" for c, _ in results) and all(r == "not_requested" for c, r in results if c == "cycle_time"):
        lines.append("Cycle time was not checked: the site gave no target.")
    inputs = _mapping(table.get("inputs"))
    # A conditional payload check is worded above; this line counts only the rows left open.
    payload_unknown = sum(1 for _, verdict, checks, _, _ in rows if verdict != FITS_IF and any(
        c["check"] == "payload" and c["result"] == "unknown" for c in checks))
    if payload_unknown and inputs.get("mass_basis") == "estimated":
        mass = _range(inputs.get("mass_kg_range"), "kg")
        lines.append(f"Payload is unknown for {payload_unknown} of the {len(rows)} robots listed: the object was not "
                     "weighed, and its mass is estimated from the video only"
                     + (f" (the paper check allows {mass})." if mass not in (NOT_RECORDED, INVALID) else "."))
    return lines


def _trial_kind(entry: Mapping[str, Any], row: Mapping[str, Any]) -> str | None:
    """``learned`` or ``scripted`` when the catalog and the published row agree; None when they do not.

    A scripted run only checks the simulation: it is never robot software's result. The catalog says
    scripted by ``policy.kind``; the row by ``execution_evidence_kind`` and ``candidate_policy_queried``
    (``beta_policy_screening_result``), which older rows leave out.
    """
    from .policy_execution_evidence import LEARNED, SCRIPTED
    catalog_scripted = _mapping(entry.get("policy")).get("kind") == "scripted"
    kind = row.get("execution_evidence_kind")
    queried = row.get("candidate_policy_queried")
    if catalog_scripted:
        return "scripted" if kind in (None, SCRIPTED) and queried is not True else None
    if kind not in (None, LEARNED) or queried is False:
        return None
    return "learned"


def _trial_rows(publication: Mapping[str, Any]) -> tuple[dict[str, list], int, int]:
    """The rows of each kind (catalog name, status, tries, successes) and how many of each were left out."""
    from .beta_policy_screening_result import CANDIDATE_STATUSES
    catalog, rows, seen = _policy_catalog(), {"learned": [], "scripted": []}, set()
    dropped = {"learned": 0, "scripted": 0}
    for row in publication.get("candidates") if isinstance(publication.get("candidates"), list) else []:
        entry = catalog.get(row.get("candidate_id")) if isinstance(row, Mapping) and isinstance(row.get("candidate_id"), str) else None
        kind = _trial_kind(entry, row) if entry is not None else None
        scored = _count(row.get("episodes_scored"), low=1) if isinstance(row, Mapping) else None
        succeeded = _count(row.get("episodes_succeeded"), high=scored or 0) if scored is not None else None
        if (kind is None or entry["candidate_id"] in seen or row.get("status") not in CANDIDATE_STATUSES
                or row.get("status") not in STATUS_WORDS or scored is None or succeeded is None):
            dropped[kind or "learned"] += 1  # unknown or disagreeing rows count against the early trial
        else:
            seen.add(entry["candidate_id"])
            rows[kind].append((entry, row["status"], scored, succeeded))
    return rows, dropped["learned"], dropped["scripted"]


def _table_lines(rows: list, header: str) -> list[str]:
    return [f"| {header} | Status | Runs scored | Runs that met the success rule |", "|---|---|---|---|",
            *(f"| {entry['display_name']} | {STATUS_WORDS[status]} | {scored} | {succeeded} of {scored} |"
              for entry, status, scored, succeeded in rows)]


def trial_lines(publication: Mapping[str, Any] | None, clips: int = 0) -> list[str]:
    """The early trial of learned robot software, and apart from it any scripted check of the simulation.

    Every name is the catalog's; the two sections are never added together.
    """
    lines = [f"**{EARLY_TRIAL_LABEL}**", ""]
    if publication is None:
        return lines + ["No early trial of openly available robot software is in this packet."]
    rows, dropped, scripted_dropped = _trial_rows(publication)
    lines += _table_lines(rows["learned"], "Robot software") if rows["learned"] else ["No row of the trial could be shown."]
    lines += _left_out(dropped)
    campaign = CAMPAIGN_WORDS.get(publication.get("status")) if isinstance(publication.get("status"), str) else None
    lines += ["", f"Trial status: {campaign or NOT_RECORDED}. A zero is a valid result of the trial."]
    if clips:
        lines += ["", f"Clips of the trial's runs: {clips}, in `files/early-trial/clips/`."]
    if rows["scripted"] or scripted_dropped:
        lines += ["", f"## {SCRIPTED_CHECK_TITLE}", "", SCRIPTED_CHECK_SENTENCE, ""]
        lines += _table_lines(rows["scripted"], "Scripted run") if rows["scripted"] else ["No row of the check could be shown."]
        lines += _left_out(scripted_dropped)
    return lines


def _cell(value: Any) -> str:
    return str(value if value not in (None, "") else NOT_RECORDED).replace("|", "/").replace("\n", " ")


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
        return "No success rule is in this packet yet. It will follow with the simulation."
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


def _measured_and_estimated(preparation: Mapping[str, Any], base_scene: Mapping[str, Any] | None,
                            fit_table: Mapping[str, Any] | None, publication: Mapping[str, Any] | None
                            ) -> tuple[list[str], list[str]]:
    prep = preparation
    residual = _length(_mapping(_mapping(prep.get("registration")).get("task_region")).get("trimmed_rmse_m"))
    measured = [f"How closely the video's geometry and the reconstruction line up around the task (registration "
                f"residual, trimmed RMS): {residual}, in the scene's estimated metres."]
    frame = _mapping(prep.get("coordinate_frame"))
    inputs = _mapping(_mapping(fit_table).get("inputs"))
    band = inputs.get("scale_band")
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
    bounds = _mapping(_mapping(prep.get("physics")).get("bounds"))
    if "mass_kg" in bounds:
        estimated.append(f"Mass used in the simulation: {_range(bounds.get('mass_kg'), 'kg')}, a prior from the object's "
                         "size, not weighed.")
    if "static_friction" in bounds or "dynamic_friction" in bounds:
        estimated.append(f"Friction: static {_range(bounds.get('static_friction'), '')}, dynamic "
                         f"{_range(bounds.get('dynamic_friction'), '')}; prior ranges, not measured.")
    straight, path = _range(inputs.get("straight_line_m"), "m"), _range(inputs.get("carry_path_m"), "m")
    if straight not in (NOT_RECORDED, INVALID) and path not in (NOT_RECORDED, INVALID):
        estimated.append(f"Carry distance: {straight} in a straight line, {path} along the carry path, from the scene's "
                         "estimated geometry.")
    codes = {row["code"] for row in _mapping(base_scene).get("scene_estimates") or []
             if isinstance(row, Mapping) and isinstance(row.get("code"), str)}
    codes |= {code for code in _mapping(_mapping(publication).get("scene_feasibility")).get("flags") or []
              if isinstance(code, str) and code.startswith("scene_estimate:")}
    for code in sorted(codes):
        estimated.append(f"`{code}`: {ESTIMATE_SENTENCES.get(code, 'an estimate the scene rests on (no plain sentence recorded for this code).')}")
    if not codes:
        estimated.append("Scene estimate flags (`scene_estimate:*`): " + (
            "none recorded." if base_scene else "not known: the scene's base record is not on the host."))
    return measured, estimated


# --- binding the inputs to this job ---------------------------------------------------------------------

def _not_this_job(what: str) -> JobPacketRefused:
    return JobPacketRefused("job_packet_input_not_this_job", what)


def _video(inputs: JobInputs, pull: PullRecord, raw_manifest: Mapping[str, Any] | None) -> Path | None:
    """The job's video: ``--video``, else the file the raw manifest names, else the one video file in ``raw/``."""
    capture = inputs.capture_dir
    if inputs.video is not None:
        if inputs.video.is_symlink() or not inputs.video.resolve().is_relative_to(capture.resolve()):
            raise _not_this_job(f"video {inputs.video}")
        _regular(inputs.video, str(inputs.video))
        return inputs.video
    named = _mapping(raw_manifest).get("video_uri")
    prefix = f"scenes/{capture.parent.parent.name}/captures/{capture.name}/raw/"
    if isinstance(named, str) and named.startswith(prefix) and re.fullmatch(r"[A-Za-z0-9._-]+", named[len(prefix):]) \
            and Path(named).suffix.lower() in VIDEO_SUFFIXES:
        relative = "raw/" + named[len(prefix):]
        if not pull.in_capture(relative):
            return None
        _regular(_inside(capture, capture / relative), relative)
        return capture / relative
    raw = capture / "raw"
    found = [path for path in (_walk(raw) if raw.is_dir() else []) if path.parent == raw
             and path.suffix.lower() in VIDEO_SUFFIXES]
    if len(found) > 1:
        raise JobPacketRefused("job_packet_video_ambiguous", "raw/ holds more than one video: pass --video")
    return found[0] if found else None


def _bind_inputs(inputs: JobInputs, context: Mapping[str, Any], team_record: Mapping[str, Any] | None,
                 plan: Mapping[str, Any] | None) -> None:
    scene, capture = context["scene_id"], context["capture_id"]
    if inputs.team_scene_dir is not None:
        contract = _mapping(_mapping(team_record).get("task_success_contract"))
        if team_record is None or plan is None or plan.get("scene_id") != scene \
                or _mapping(contract.get("scope")).get("site_id") != scene:
            raise _not_this_job(f"team scene {inputs.team_scene_dir}")
    if inputs.configured_scene_dir is not None:
        revision = _read_json(inputs.configured_scene_dir, inputs.configured_scene_dir / "configured_scene_revision.v1.json")
        if (revision is None or team_record is None
                or revision.get("revision_digest") != team_record.get("scene_revision_digest")):
            raise _not_this_job(f"configured scene {inputs.configured_scene_dir} (bound through the team scene's "
                                "revision digest)")
    if inputs.review_videos_dir is not None:
        manifest = _read_json(inputs.review_videos_dir, inputs.review_videos_dir / "manifest.json")
        if manifest is None or manifest.get("site_id") != scene or manifest.get("capture_id") != capture:
            raise _not_this_job(f"trial clips {inputs.review_videos_dir}")


def _reconstruction(capture_dir: Path, base_scene: Mapping[str, Any] | None, pull: PullRecord) -> list[tuple[Path, Path, str]]:
    """The levelled splat and the collision mesh the scene is built on, as ``base_scene.json`` names them."""
    rows = []
    for key in ("splat_path", "collision_mesh_path"):
        relative = _in_capture(capture_dir, _mapping(base_scene).get(key))
        if relative is not None and pull.in_capture(relative):
            rows.append((capture_dir, capture_dir / relative, f"files/reconstruction/{Path(relative).name}"))
    return rows


def _packet_files(inputs: JobInputs, plan_dir: Path | None) -> list[tuple[Path, Path, str]]:
    """(root, source, packet path) for every simulation and trial file copied as it is. No record is copied."""
    rows: list[tuple[Path, Path, str]] = []
    if inputs.configured_scene_dir is not None:
        root = inputs.configured_scene_dir
        rows += [(root, path, f"files/object/{path.name}") for path in _walk(root)
                 if path.parent == root and path.suffix.lower() == ".usdz"]
    if inputs.team_scene_dir is not None:
        root = inputs.team_scene_dir
        rows += [(root, path, f"files/simulation/team-scene/{path.relative_to(root).as_posix()}") for path in _walk(root)
                 if path.suffix.lower() in USD_SUFFIXES and (plan_dir is None or path.is_relative_to(plan_dir))]
    if inputs.review_videos_dir is not None:
        root = inputs.review_videos_dir
        clips = [path for path in _walk(root) if path.suffix.lower() == ".mp4"]
        rows += [(root, path, f"files/early-trial/clips/clip-{index:02d}.mp4") for index, path in enumerate(clips, 1)]
    unique_destinations(rows)
    return rows


#: The composed stage of the team scene (``scene_package_stage``): the only file that says where each part sits.
SIMULATION_STAGE = "files/simulation/scene.usda"


def _stage_rows(plan: Mapping[str, Any] | None, plan_dir: Path | None,
                copies: list[tuple[Path, Path, str]]) -> tuple[list[dict[str, Any]], str] | None:
    """Each plan row placed on the copied team scene file, and the up axis; None when any row cannot be placed.

    The plan itself is a record and is never copied; without a composed stage the packet carries no simulation.
    """
    from .scene_package_stage import StageRefused, placements, up_axis
    if plan is None or plan_dir is None:
        return None
    try:
        rows, axis = placements(plan), up_axis(plan)
    except StageRefused:
        return None
    copied = {source: relative for _root, source, relative in copies}
    placed = []
    for row in rows:
        relative = copied.get(plan_dir.joinpath(*Path(row["usd_path"]).parts))
        if relative is None:
            return None
        placed.append({**row, "file": Path(relative).relative_to("files/simulation").as_posix()})
    return placed, axis


def unique_destinations(rows: list[tuple[Path, Path, str]]) -> None:
    destinations = [row[2] for row in rows]
    if len(set(destinations)) != len(destinations):
        raise JobPacketRefused("job_packet_file_name_collision", ", ".join(sorted(
            {item for item in destinations if destinations.count(item) > 1})))


def _configured_members(directory: Path | None) -> list[tuple[zipfile.ZipFile, zipfile.ZipInfo, str]]:
    """The configured scene bundle's scene files (appearance, collision, replacement); its record is left out."""
    if directory is None:
        return []
    bundle = directory / "configured_scene_bundle.v1.zip"
    if not os.path.lexists(bundle):
        return []
    _regular(_inside(directory, bundle), bundle.name)
    try:
        archive = zipfile.ZipFile(bundle)
        members = [info for info in archive.infolist() if not info.is_dir() and not info.filename.endswith(".json")]
    except (OSError, zipfile.BadZipFile) as exc:
        raise JobPacketRefused("job_packet_input_unreadable", bundle.name) from exc
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", info.filename) for info in members):
        raise JobPacketRefused("job_packet_input_unreadable", f"{bundle.name}: a member is not a plain file name")
    return [(archive, info, f"files/simulation/configured-scene/{info.filename}") for info in members]


def _require_sound_tools() -> None:
    from .capture_audio_minimization import media_tool_blockers
    if media_tool_blockers() or not shutil.which("ffprobe"):
        raise JobPacketRefused("job_packet_video_sound_tool_missing",
                               "ffmpeg and ffprobe are needed to remove the video's sound; install them and run again")


#: The only tags the copy may carry: what ffmpeg's mov/mp4 muxer writes itself once every source tag is dropped
#: (checked with ffmpeg 6.1 on .mov and .mp4). None of them comes from the source file or says where or when it was
#: filmed. ``language`` is allowed only as ``und`` (undetermined), the muxer's own value.
MUXER_TAGS = frozenset({"major_brand", "minor_version", "compatible_brands", "encoder", "handler_name", "vendor_id",
                        "language"})


def _video_details_left(path: Path) -> str | None:
    """What ffprobe still finds in the copy beyond one video stream and the muxer's own tags, or None."""
    try:
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type:stream_tags:format_tags",
                                 "-show_chapters", "-of", "json", str(path)], capture_output=True, text=True,
                                check=False, timeout=120)
        probe = json.loads(result.stdout) if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        probe = None
    if not isinstance(probe, Mapping):
        return "ffprobe could not read the copy"
    streams = probe.get("streams") if isinstance(probe.get("streams"), list) else []
    if [_mapping(stream).get("codec_type") for stream in streams] != ["video"]:
        return "the copy does not hold exactly one video stream and nothing else"
    if probe.get("chapters"):
        return "chapters survived"
    for where, tags in (("format", _mapping(probe.get("format")).get("tags")), ("stream", _mapping(streams[0]).get("tags"))):
        for key, value in _mapping(tags).items():
            if key not in MUXER_TAGS or (key == "language" and value != "und"):
                return f"a {where} tag {key!r} survived"
    return None


def _copy_video(source: Path, target: Path) -> dict[str, Any]:
    """The one video stream copied as it is, without sound, tags or chapters; refused unless ffprobe shows that."""
    from .capture_audio_minimization import write_audio_free_video
    receipt = write_audio_free_video(source, target, strip_metadata=True)
    if receipt.get("evidence") != "ffprobe_output_has_no_audio_stream" or receipt.get("output_audio_stream_count") != 0:
        target.unlink(missing_ok=True)
        raise JobPacketRefused("job_packet_video_sound_not_removed",
                               f"{source.name}: {receipt.get('reason') or receipt.get('evidence')}; the video was not copied")
    left = _video_details_left(target)
    if left:
        target.unlink(missing_ok=True)
        raise JobPacketRefused("job_packet_video_details_not_removed", f"{source.name}: {left}; the video was not copied")
    return receipt


def _secrets(capture_dir: Path, context: Mapping[str, Any], pull: PullRecord) -> list[tuple[bytes, str]]:
    """What must never be in a packet file, each with the kind of thing it is (the kind is printed, never the value)."""
    def record(relative: str) -> Mapping[str, Any]:
        path = capture_dir / relative
        return _mapping(_read_json(capture_dir, path)) if os.path.lexists(path) else {}

    raw, descriptor = record("raw/manifest.json"), record("capture_descriptor.json")
    upload, staging, ledger = (record("raw/capture_upload_complete.json"), record("pipeline_staging_manifest.json"),
                               record("pipeline_job_ledger.json"))
    authority = _mapping(_mapping(descriptor.get("metadata")).get("website_scene_execution_authority"))
    owner = _mapping(_mapping(_mapping(record(f"{PREPARATION}/preparation.json")).get("intake_request")).get("owner"))
    found = [(value, "a job id") for value in (context.get("request_id"), context.get("scene_id"), context.get("capture_id"),
                                               raw.get("site_submission_id"), descriptor.get("site_submission_id"))]
    found += [(value, "an owner or organisation id") for value in (
        *_mapping(authority.get("owner")).values(), _mapping(authority.get("consent")).get("accepted_by"), *owner.values())]
    found += [(value, "a delivery key") for value in (
        _mapping(upload.get("producer_delivery")).get("delivery_key"), staging.get("delivery_key"),
        ledger.get("producer_delivery_key"))]
    found += [(value, "the pull root's path") for value in {str(pull.root.absolute()), str(pull.root.resolve())}]
    found += [("gs://", "a storage link"), ("/var/lib/blueprint", "a host path")]
    return [(value.encode(), kind) for value, kind in found if isinstance(value, str) and len(value) >= 4]


def _scan_packet(packet: Path, secrets: list[tuple[bytes, str]], words: list[tuple[bytes, str]] | None = None) -> None:
    """Refuse when any file written into the packet, or any member of a zip there, holds an id, key, path or link.

    The scan is shared with the simulation package (``outbound_file_scan``): ids in both forms, host paths, storage,
    file and other links, bare web addresses, emails and IP addresses in text, tokens, and the text form of every
    binary USD layer. The site's identity (``words``: its name, organisation, address, contact names, email and
    phone, in any spelling) is looked for under ``files/`` only, never in the composed stage Blueprint writes. The
    site's description of its job is not looked for: the site agreed to share it, and the sheet quotes it.
    """
    from .outbound_file_scan import first_finding, literal_forms
    # Only the closed list of kinds may ship as simulation files; the video and clips are checked by their own code.
    finding = first_finding(packet, [*literal_forms(secrets), *(words or [])], words_only_under="files/",
                            words_except=(SIMULATION_STAGE,), closed=("files/simulation/", "files/object/"))
    if finding is not None:
        raise JobPacketRefused("job_packet_file_names_the_site", finding.sentence())


def _code_revision() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                                timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return NOT_RECORDED
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", revision) else NOT_RECORDED


def _readme(*, files: list[str], preparation: Mapping[str, Any] | None, base_scene: Mapping[str, Any] | None) -> str:
    lines = ["# Job packet", "", TERMS_LINE, "",
             "`JOB_SHEET.md` describes the job; `manifest.json` lists every file with its size and SHA-256.", "",
             "## Opening the 3D files", ""]
    has = lambda prefix: any(path.startswith(prefix) for path in files)  # noqa: E731
    if not any(has(prefix) for prefix in ("files/simulation/", "files/object/", "files/reconstruction/")):
        lines.append("This packet has no 3D files yet.")
    else:
        if has(SIMULATION_STAGE):
            lines.append(f"- `{SIMULATION_STAGE}`: open this one. It places the room, its collision geometry and the "
                         "job's object where the simulation puts them, in metres.")
        if has("files/simulation/team-scene/"):
            lines.append("- `files/simulation/team-scene/`: OpenUSD for NVIDIA Isaac Sim. Blueprint runs these in "
                         "Isaac Sim 6.0.1 (NGC image). A `scene_appearance.usdc` holds the room's look as a ParticleField "
                         "Gaussian splat, which Isaac Sim 6 reads; other USD tools may show only the collision geometry "
                         "and the object.")
        if has("files/simulation/configured-scene/"):
            lines.append("- `files/simulation/configured-scene/`: the configured scene Blueprint built: the room's look, "
                         "its collision geometry and the replacement object.")
        if has("files/object/"):
            lines.append("- `files/object/`: the job's object as USDZ, as Blueprint authored it from the video (its "
                         "size, mass and friction are estimates; see the sheet).")
        if has("files/reconstruction/"):
            lines.append("- `files/reconstruction/`: the 3D reconstruction of the site from the video. `.ply` / `.spz` "
                         "are Gaussian splats (common splat viewers open them); `.glb` is the collision mesh (Blender or "
                         "any glTF viewer).")
        frame = _mapping(_mapping(preparation).get("coordinate_frame"))
        lines += ["", f"Task coordinates: {_cell(frame.get('task_coordinates'))}; up axis "
                  f"{_cell(frame.get('declared_up_axis'))}; physical scale measured: "
                  f"{_cell(frame.get('physical_scale_measured'))}. The reconstruction's own up axis: "
                  f"{_cell(_mapping(base_scene).get('up_axis'))}."]
    for prefix, words in (("files/video/", "- `files/video/`: the job's video, without its sound or its recorded details "
                                           "such as place and time."),
                          ("files/task/", "- `files/task/success_rule.json`: the success rule's thresholds, as the "
                                          "simulation checks them."),
                          ("files/early-trial/", "- `files/early-trial/clips/`: clips of the early trial's runs.")):
        if has(prefix):
            lines += ["", words] if not lines[-1].startswith("- `files/") else [words]
    lines += ["", "## Who received this packet", "",
              "`manifest.json` has an empty `recipients` list on purpose: the register of which deployment company "
              "received which packet (by its packet number) is kept on the website side, not in this folder.", ""]
    return "\n".join(lines)


def _status(*, video: bool, first_look: bool, reconstruction: bool, simulation: bool, rule: bool, trial: bool) -> str:
    present = [words for flag, words in ((True, "the site's description of the job"), (video, "the video, without sound"),
                                         (first_look, "the first look on paper"),
                                         (reconstruction, "the 3D reconstruction of the site"),
                                         (simulation, "the simulation of the site"), (rule, "the success rule"),
                                         (trial, "the early trial's result")) if flag]
    missing = [words for flag, words in ((video, "the video"), (first_look, "the first look on paper")) if not flag]
    text = "In this packet: " + ", ".join(present[:-1]) + (" and " if len(present) > 1 else "") + present[-1] + "."
    if missing:
        text += " Not in it: " + " and ".join(missing) + "."
    if not simulation:
        text += (" The simulation of the site with the job's object in it is not finished; it will follow"
                 + (" with the success rule." if not rule else "."))
    elif not rule:
        text += " The success rule will follow."
    if not trial:
        text += " No early trial of openly available robot software has been added."
    return text


def build_job_packet(inputs: JobInputs, *, output_root: Path, owned: OwnedCaptureApproval | None = None,
                     operator: str | None = None, now: datetime | None = None,
                     owned_list: Path | None = None) -> dict[str, Any]:
    """Check everything, then write ``packet-<number>/`` and its operator record under ``output_root``."""
    capture_dir = inputs.capture_dir
    if capture_dir.is_symlink() or not capture_dir.is_dir():
        raise JobPacketRefused("job_packet_input_missing", f"capture directory {capture_dir}")
    context = _read_json(capture_dir, capture_dir / TASK_CONTEXT, required=True)
    if context.get("schema_version") != TASK_CONTEXT_SCHEMA:
        raise JobPacketRefused("job_packet_task_context_unsupported", f"{TASK_CONTEXT} schema_version")
    pull = check_job_state(capture_dir, context, pull_root=inputs.pull_root, now=now)
    if context.get("capture_id") != capture_dir.name or context.get("scene_id") != capture_dir.parent.parent.name:
        raise JobPacketRefused("job_packet_input_identity_mismatch", f"{TASK_CONTEXT} names another capture or site")
    basis = permission_basis(context, owned=owned, owned_list=owned_list, now=now,
                             video_sha256=recorded_video_sha256(capture_dir, pull) if owned is not None else None)
    output_root = Path(output_root)
    if output_root.is_symlink() or not output_root.is_dir():
        raise JobPacketRefused("job_packet_output_root_invalid", str(output_root))

    read: dict[str, str] = {}  # input digests, for the operator record

    def record(root: Path, path: Path, value: Any) -> Any:
        if value is not None:
            read[path.name if root == path.parent else path.relative_to(root).as_posix()] = _sha256(_inside(root, path))
        return value

    def capture_json(relative: str) -> dict[str, Any] | None:
        if not pull.in_capture(relative):
            return None
        return record(capture_dir, capture_dir / relative, _read_json(capture_dir, capture_dir / relative))

    record(capture_dir, capture_dir / TASK_CONTEXT, context)
    handoff = capture_json(f"{PREPARATION}/handoff.json")
    preparation, prep_dir = _preparation(capture_dir, handoff, pull)
    base_scene = (capture_json((prep_dir / "base_scene.json").relative_to(capture_dir).as_posix())
                  if prep_dir is not None else None)
    fit_table, first_look = _fit_table(capture_dir, handoff, preparation, context, pull)
    video = _video(inputs, pull, _read_json(capture_dir, capture_dir / "raw" / "manifest.json"))
    if video is not None:
        _require_sound_tools()
        if "video_sha256" in basis and _sha256(video) != basis["video_sha256"]:
            raise JobPacketRefused("job_packet_owned_capture_video_mismatch",
                                   f"{video.relative_to(capture_dir).as_posix()} is not the video pipeline_staging_manifest.json records")
    publication = None
    if inputs.screening_publication is not None:
        root, path = _file_argument(inputs.screening_publication)
        publication = record(root, path, _read_json(root, path, required=True))
        if (publication.get("site_id") != context["scene_id"]
                or _mapping(publication.get("source")).get("capture_id") not in (None, context["capture_id"])):
            raise _not_this_job(f"early trial publication {path}")
    team_record = plan = plan_dir = None
    if inputs.team_scene_dir is not None:
        root = inputs.team_scene_dir
        team_record = record(root, root / "team_scene_inputs.v1.json", _read_json(root, root / "team_scene_inputs.v1.json"))
        for name in ("native-task-packet-three-view", "native-task-packet"):
            plan = _read_json(root, root / name / "native_task_arena_scene_plan.v1.json")
            if plan is not None:
                plan_dir = root / name
                break
    _bind_inputs(inputs, context, team_record, plan)
    copies = _reconstruction(capture_dir, base_scene, pull) + _packet_files(inputs, plan_dir)
    stage = _stage_rows(plan, plan_dir, copies)
    members = _configured_members(inputs.configured_scene_dir)
    contract = _mapping(team_record).get("task_success_contract")
    criteria = _mapping(contract).get("criteria") if isinstance(contract, Mapping) else None
    written_files = [row[2] for row in copies] + [row[2] for row in members]
    written_files += [f"files/video/video{video.suffix.lower()}"] if video is not None else []
    written_files += ["files/task/success_rule.json"] if isinstance(criteria, Mapping) else []
    written_files += [SIMULATION_STAGE] if stage is not None else []
    clips = sum(1 for path in written_files if path.startswith("files/early-trial/clips/"))
    # The simulation is in the packet only with its composed stage: loose USD files do not say where anything sits.
    simulation = stage is not None
    reconstruction = any(path.startswith("files/reconstruction/") for path in written_files)
    prep = _mapping(preparation)
    kind = kind_of_job(preparation, context)
    number = secrets.token_hex(8)

    # --- the sheet: the site's own words go in as written, without contact details it typed ---
    removed: list[bool] = []
    own = _redacted({key: context.get(key) for key in ("description", "success_criteria", "operator_task_details",
                                                       "operator_answers", "unresolved")}, removed)
    prep = _redacted(prep, removed)
    contract = _redacted(contract, removed)
    criteria = _mapping(contract).get("criteria") if isinstance(contract, Mapping) else None
    description = str(own.get("description") or "").strip()
    sheet = ["# Job packet", "", f"Packet number: {number}", "", TERMS_LINE, "", "Status: " + _status(
        video=video is not None, first_look=first_look == "shown", reconstruction=reconstruction, simulation=simulation,
        rule=isinstance(contract, Mapping), trial=publication is not None), "",
        "## The job, in the site's words", "", *(f"> {line}" for line in description.splitlines()), "",
        f"Kind of job: {kind}.", "", "## Constraints and what the site told us", ""]
    stated = _mapping(own.get("success_criteria"))
    details = _mapping(own.get("operator_task_details"))
    constraints = [(words, value) for words, value in (
        ("The site's own success definition", stated.get("successDefinition")),
        ("Success rate the site wants (%)", stated.get("successRate")),
        ("Cycle time the site wants (s)", stated.get("cycleTimeSeconds")),
        ("Object weight, as the site stated it (not checked)", details.get("item_weight")),
        ("Object make and model, as the site stated it", details.get("item_make_model")),
        *sorted(_mapping(own.get("operator_answers")).items())) if value not in (None, "")]
    sheet += [f"- {_cell(words)}: {_cell(value)}" for words, value in constraints]
    unresolved = own.get("unresolved")
    if isinstance(unresolved, list) and unresolved:
        sheet.append("- Still unanswered: " + ", ".join(_cell(item) for item in unresolved))
    if sheet[-1] == "":
        sheet.append("The site stated nothing beyond its description.")
    if prep:
        subject, destination = _mapping(prep.get("subject")), _mapping(prep.get("destination"))
        target = _mapping(_mapping(criteria).get("surface_target"))
        place = target.get("visible_label") or destination.get("visible_label")
        centre = target.get("surface_position_world_m") or destination.get("position_world_m")
        sheet += ["", "## The object and where it goes", ""]
        if subject.get("description"):
            sheet.append(f"- Object: {_cell(subject.get('description'))}.")
        if place or centre is not None:
            sheet.append("- Goes to: " + "; ".join(item for item in (
                _cell(place) if place else "", f"centre {_point(centre)}" if centre is not None else "",
                f"within {_length(target.get('radius_m') or destination.get('radius_m'))}"
                if (target.get("radius_m") or destination.get("radius_m")) is not None else "") if item) + ".")
        measured, estimated = _measured_and_estimated(prep, base_scene, fit_table, publication)
        sheet += ["", "## Measured", ""] + [f"- {line}" for line in measured]
        sheet += ["", "## Estimated", ""] + [f"- {line}" for line in estimated]
    sheet += ["", "## The video", "", "`files/video/`: the job's video, without its sound or its recorded details such as "
              "place and time. A deployment company needs neither to judge a job." if video is not None else "The video is not in this packet."]
    sheet += ["", "## First look on paper", ""]
    sheet += fit_lines(fit_table) if fit_table is not None else [
        "The first look is not shown: its table did not match the record that names it, so it could not be checked."
        if first_look == "not_verified" else "No first look on paper is in this packet yet."]
    sheet += ["", "## The simulation", "", "The simulation files are under `files/`; `README.md` says how to open them."
              if simulation else "The simulation of the site with the job's object in it is not finished. It will follow."]
    if reconstruction and not simulation:
        sheet.append("The 3D reconstruction of the site it is built on is under `files/reconstruction/`.")
    sheet += ["", "## What counts as done", "", done_sentence(
        contract, object_name=_mapping(prep.get("subject")).get("description"),
        place_label=_mapping(prep.get("destination")).get("visible_label"),
        assumption=_mapping(_mapping(team_record).get("owner_success_assumption")) or None)]
    sheet += ["", "## Early trial", "", *trial_lines(publication, clips)]
    sheet += ["", "## How to answer", "", "Reply to the person who sent you this packet. Give the packet number and one "
              "of: interested; not a fit; would need something (say what).", "",
              "## What this packet does not show", "",
              "- No test at the site: everything here comes from one video" + (" and a simulation of it." if simulation
                                                                               else "."),
              "- No safety assessment of any robot at this site.", ""]
    if any(removed):
        sheet[6] += " Contact details the site typed into its description or answers were removed."

    # --- write: the packet folder, then the operator record beside it ---
    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    name = f"packet-{number}"
    packet, operator_record = output_root / name, output_root / f"{name}.operator-record.json"
    packet.mkdir()
    try:
        written = {"JOB_SHEET.md": "\n".join(sheet), "README.md": _readme(
            files=written_files, preparation=preparation, base_scene=base_scene)}
        if isinstance(criteria, Mapping):
            # Through the shared allow-list (``scene_package_stage.neutral_criteria``): numbers and closed mode words
            # only, so no label the site typed or path the plan holds leaves in the rule.
            from .scene_package_stage import neutral_criteria
            written["files/task/success_rule.json"] = json.dumps({"criteria": neutral_criteria(criteria)[0]}, indent=2,
                                                                 sort_keys=True) + "\n"
        for filename, text in written.items():
            (packet / filename).parent.mkdir(parents=True, exist_ok=True)
            (packet / filename).write_text(text, encoding="utf-8")
        for root, source, relative in copies:
            target = packet / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_regular(_inside(root, source), source.name), target)
        for archive, info, relative in members:
            target = packet / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as stream, target.open("wb") as handle:
                shutil.copyfileobj(stream, handle)
        if stage is not None:
            from .scene_package_stage import StageRefused, write_stage
            try:
                write_stage(packet / SIMULATION_STAGE, stage[0], axis=stage[1])
            except StageRefused as exc:
                raise JobPacketRefused("job_packet_simulation_not_composed", f"{SIMULATION_STAGE}: the {exc.role or 'scene'} "
                                       "part could not be placed") from exc
        sound = None
        if video is not None:
            target = packet / "files" / "video" / f"video{video.suffix.lower()}"
            target.parent.mkdir(parents=True, exist_ok=True)
            sound = _copy_video(video, target)
        from .outbound_file_scan import capture_site_identity, record_site_identity
        _scan_packet(packet, _secrets(capture_dir, context, pull),
                     capture_site_identity(capture_dir) + record_site_identity([context]))
        listed = []
        for relative in sorted([*written, *(row[2] for row in copies), *(row[2] for row in members),
                                *([SIMULATION_STAGE] if stage is not None else []),
                                *([f"files/video/video{video.suffix.lower()}"] if video is not None else [])]):
            path = packet / relative
            listed.append({"path": relative, "size": path.stat().st_size, "sha256": _sha256(path)})
        # The packet number is random, so the content id reads the sheet without it: the same inputs give the same id.
        unnumbered = hashlib.sha256(written["JOB_SHEET.md"].replace(number, "").encode()).hexdigest()
        packet_id = "sha256:" + hashlib.sha256("\n".join(
            f"{row['path']} {unnumbered if row['path'] == 'JOB_SHEET.md' else row['sha256']}" for row in listed
        ).encode()).hexdigest()
        keys = {"request_id": context.get("request_id"), "scene_id": context["scene_id"],
                "capture_id": context["capture_id"]}
        manifest = {"schema_version": SCHEMA_VERSION, "packet_number": number, "packet_id": packet_id,
                    "created_on": moment.date().isoformat(), "code_revision": _code_revision(),
                    "permission_basis": {key: basis[key] for key in ("rule", "statement_version") if key in basis},
                    "first_look_source": "pipeline_robot_fit_table" if first_look == "shown" else "none",
                    "files": listed, "recipients": []}
        (packet / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        operator_value = {
            "schema_version": OPERATOR_RECORD_SCHEMA, "packet_folder": name, "packet_number": number,
            "packet_id": packet_id, "created_at": moment.isoformat().replace("+00:00", "Z"),
            "run_by": operator or NOT_RECORDED, "code_revision": manifest["code_revision"], "job": keys,
            "takedown_keys": {**keys, "withdrawal_tombstone": f"scenes/{context['scene_id']}/website_withdrawal/tombstone.json"},
            "permission_basis": basis, "input_sha256": dict(sorted(read.items())),
            "first_look": first_look, "video_sound_removal": sound,
            # The operator's own statement of what was pulled and what the host lacked, kept whole, so a
            # false "absent" can be traced to who made the packet.
            "pull_record": pull.value, "pull_record_sha256": _sha256(_inside(pull.root, pull.root / PULL_RECORD)),
            "job_state_checked": {"capture_rights": {k: _mapping(context.get("capture_rights")).get(k)
                                                     for k in ("consent_status", "statement_version", "consent_revoked")},
                                  "withdrawal": "not withdrawn in the pulled site roots"},
        }
        operator_record.write_text(json.dumps(operator_value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(packet, ignore_errors=True)
        operator_record.unlink(missing_ok=True)
        raise
    finally:
        for archive in {id(row[0]): row[0] for row in members}.values():
            archive.close()
    return {**manifest, "packet_dir": str(packet), "operator_record": str(operator_record)}
