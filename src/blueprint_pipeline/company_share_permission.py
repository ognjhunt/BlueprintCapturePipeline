"""Whether a capture's video and simulation may still be shared with approved deployment companies.

One set of rules for both things a company receives from a website capture: the silent clip
(``website_company_clip``) and the simulation package (``website_company_simulation``). Both read and write
the same stored live check (``pipeline/company_clip/statement_check.json``), so a capture is read live once a
window for both, and both honour the website's recheck mark (``permission_recheck``).

- Blueprint's own list (``config/blueprint_owned_captures.v1.json``) is a local read, made every pass and on
  every route read, against the digest of the upload the shared thing was made from.
- The live rights (the WebApp's signed ``task-context`` read) are read at least once every
  ``LIVE_STATEMENT_RECHECK_SECONDS``, and at once after a recheck mark; every answer is recorded. A definite
  "no" takes the shared thing back on that pass. With no answer a live grant is kept at most
  ``LIVE_GRANT_LIFETIME_SECONDS`` from the last answer that granted it, and the route applies that bound too.
- Retraction is recorded before the stored copy is deleted, so a failed deletion is retried, never forgotten.

The reason codes stored in the check keep the clip's names (``website_company_clip_*``), as they always were;
the simulation reports them in its own names. A leaf of the two modules: nothing here builds, uploads or sends.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .permission_recheck import recheck_requested_since
from .sharing_statement import covers_sharing, owned_captures, sharing_basis
from .website_atlas_preview_store import capture_withdrawn

#: The shared live check sits in the clip's folder, where the website's recheck mark is written too.
CHECK_DIRECTORY = ("pipeline", "company_clip")
STATEMENT_CHECK_FILE = "statement_check.json"
LIVE_STATEMENT_RECHECK_SECONDS = 6 * 3600
#: With no answer from the live read, a grant is kept at most this long after the last answer that gave it.
LIVE_GRANT_LIFETIME_SECONDS = 2 * LIVE_STATEMENT_RECHECK_SECONDS
#: A WebApp answer that means the site's rights no longer allow sharing.
RIGHTS_REFUSALS = re.compile(r"(source_revoked|consent_revoked|consent_expired|website_capture_withdrawn"
                             r"|capture_revoked_or_expired|website_company_clip_not_permitted"
                             r"|website_company_simulation_not_permitted)\b")
NOT_PERMITTED = "website_company_clip_not_permitted"
RIGHTS_REFUSED = "website_company_clip_rights_refused"


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = {} if path.is_symlink() else json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def reason_code(error: BaseException) -> str:
    text = str(error)
    return text if re.fullmatch(r"[a-z][a-z0-9_.:-]{0,159}", text) else type(error).__name__


def check_path(capture_root: Path) -> Path:
    return Path(capture_root).joinpath(*CHECK_DIRECTORY) / STATEMENT_CHECK_FILE


def read_check(capture_root: Path) -> dict[str, Any]:
    return read_json(check_path(capture_root))


def capture_identity(capture_root: Path) -> dict[str, str]:
    """The request, scene and capture a website capture folder names; ``ValueError`` for any other folder."""
    from .website_atlas_preview_store import capture_name_matches_request
    root = Path(capture_root)
    scene_id, capture_id = root.parent.parent.name, root.name
    request_id = scene_id.removeprefix("site-")
    if (root.parent.name != "captures" or not scene_id.startswith("site-")
            or not capture_name_matches_request(request_id, capture_id)):
        raise ValueError("website_company_clip_capture_identity_invalid")
    return {"request_id": request_id, "scene_id": scene_id, "capture_id": capture_id}


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


# --- retraction -------------------------------------------------------------------------------------------------

def hidden(archive: Mapping[str, Any]) -> bool:
    return archive.get("retracted") is True or archive.get("retraction") == "pending"


def take_back(manifest: dict[str, Any], manifest_path: Path, retract: Callable[[str], None]) -> bool:
    """Record the retraction before deleting, so a failed deletion is retried, never forgotten."""
    from .website_scene_preview import _finish_retraction
    archive = manifest.get("archive")
    if not isinstance(archive, Mapping) or not isinstance(archive.get("uri"), str) or archive.get("retracted"):
        return True
    manifest["archive"] = {**archive, "retraction": "pending"}
    write_json(manifest_path, manifest)
    return _finish_retraction(manifest, manifest_path, retract)


def retract_object(uri: str) -> None:
    """Delete every version of one archived object (the scene preview's deletion, shared)."""
    from .website_scene_preview import _retract
    _retract(uri)


def retraction_pending(manifest: Mapping[str, Any]) -> bool:
    archive = manifest.get("archive")
    return isinstance(archive, Mapping) and archive.get("retraction") == "pending" and isinstance(archive.get("uri"), str)


# --- the basis --------------------------------------------------------------------------------------------------

def owned_basis_holds(capture_id: str, video_sha256: str | None, owned_list: Path | None = None) -> bool:
    """The capture is on Blueprint's own list and its row's prefix names this upload. A local read, made
    every pass and on every route read; an unreadable list holds for nothing."""
    return sharing_basis({"capture_id": capture_id}, video_sha256=video_sha256, owned_list=owned_list) is not None


def on_owned_list(capture_id: str, owned_list: Path | None = None) -> bool:
    return capture_id in owned_captures(owned_list)[0]


def recorded_statement_version(capture_root: Path) -> str | None:
    """The statement version the capture's files kept at submission (a snapshot)."""
    root = Path(capture_root)
    for path in (root / "pipeline" / "website_task_context.json", root / "raw" / "manifest.json"):
        rights = read_json(path).get("capture_rights")
        version = rights.get("statement_version") if isinstance(rights, Mapping) else None
        if isinstance(version, str):
            return version
    return None


def live_rights(identity: Mapping[str, str]) -> Mapping[str, Any]:
    """The WebApp's current ``capture_rights``, projected from the request's current consent."""
    from .website_task_context import load_current_website_task_context
    context = load_current_website_task_context(request_id=identity["request_id"], scene_id=identity["scene_id"],
                                                capture_id=identity["capture_id"], purpose="scene_preparation")
    rights = context.get("capture_rights")
    return rights if isinstance(rights, Mapping) else {}


def record_live(root: Path, identity: Mapping[str, str], check_rights: Callable[..., Mapping[str, Any]],
                now: float) -> dict[str, Any]:
    """Read the live rights once and record the answer, a failure included.

    ``outcome``: ``covers_sharing`` (consent granted, sharing statement), ``not_sharing`` (consent granted,
    an older statement), ``rights_refused`` (consent not granted, or the WebApp said revoked: definite) or
    ``unanswered`` (no answer: the shared thing's state does not change on it). ``granted_at_epoch`` is the
    last time the answer covered sharing; a definite answer that does not clears it.
    """
    previous = read_check(root)
    try:
        rights = check_rights(identity)
    except Exception as error:  # noqa: BLE001 - classified: a definite refusal, or no answer
        rights, failure = None, error
    if rights is None:
        outcome = "rights_refused" if RIGHTS_REFUSALS.search(str(failure)) else "unanswered"
        version, fingerprint, reason = None, None, reason_code(failure)
    else:
        version = rights.get("statement_version")
        version = version if isinstance(version, str) else None
        granted = rights.get("consent_revoked") is False and rights.get("consent_status") == "granted"
        outcome = ("rights_refused" if not granted else "covers_sharing" if covers_sharing(version)
                   else "not_sharing")
        fingerprint = canonical_digest({"statement_version": version, "consent_status": rights.get("consent_status"),
                                        "consent_revoked": rights.get("consent_revoked")})
        reason = {"rights_refused": RIGHTS_REFUSED, "not_sharing": NOT_PERMITTED}.get(outcome)
    kept = previous.get("granted_at_epoch") if outcome == "unanswered" else None
    value = {"checked_at_epoch": int(now), "outcome": outcome, "reason": reason, "statement_version": version,
             "rights_fingerprint": fingerprint,
             "granted_at_epoch": int(now) if outcome == "covers_sharing" else kept if isinstance(kept, int) else None}
    write_json(check_path(root), value)
    return value


def live_due(check: Mapping[str, Any], now: float, root: Path) -> bool:
    """A live read is due: none yet, the window has passed, or the website asked for a recheck since
    (``permission_recheck``: a job's permission changed)."""
    checked = check.get("checked_at_epoch")
    return (not isinstance(checked, int) or not 0 <= now - checked < LIVE_STATEMENT_RECHECK_SECONDS
            or recheck_requested_since(root, checked))


def live_grant_holds(check: Mapping[str, Any], now: float | None = None) -> bool:
    """The last live read granted sharing and no definite answer has said otherwise since; with ``now``,
    also that the grant is younger than ``LIVE_GRANT_LIFETIME_SECONDS``."""
    granted = check.get("granted_at_epoch")
    if check.get("outcome") not in ("covers_sharing", "unanswered") or not isinstance(granted, int):
        return False
    return now is None or 0 <= now - granted < LIVE_GRANT_LIFETIME_SECONDS


def may_be_permitted(root: Path, identity: Mapping[str, str], now: float, owned_list: Path | None = None) -> bool:
    """No network: the snapshot covers sharing, the capture is on Blueprint's own list, the last live read
    granted sharing, or a live read is due. Otherwise a capture gets no directory, file or request."""
    if covers_sharing(recorded_statement_version(root)) or on_owned_list(identity["capture_id"], owned_list):
        return True
    check = read_check(root)
    return live_due(check, now, root) or live_grant_holds(check, now)


def decided_basis(check: Mapping[str, Any], owned: bool) -> dict[str, Any] | None:
    """The rule that lets the thing be shared now: a live grant, else Blueprint's own capture, else None."""
    if check.get("outcome") == "covers_sharing":
        return {"permission_basis": "site_sharing_statement", "statement_version": check.get("statement_version")}
    if owned and check.get("outcome") == "not_sharing":  # consent definitely granted, Blueprint's own video
        return {"permission_basis": "blueprint_owned_capture", "statement_version": check.get("statement_version")}
    return None


def refused_live(check: Mapping[str, Any], manifest: dict[str, Any], manifest_path: Path,
                 retract: Callable[[str], None], *, unanswered: str) -> dict[str, Any] | None:
    """The answer to a live read that stops sharing: a definite refusal takes any archived copy back; no
    answer refuses this pass only (``unanswered`` names it). None when the answer allows nothing to be said yet."""
    if check["outcome"] == "rights_refused":
        if manifest:
            take_back(manifest, manifest_path, retract)
        return {"status": "refused", "reason": check.get("reason") or RIGHTS_REFUSED}
    if check["outcome"] == "unanswered":
        return {"status": "refused", "reason": check.get("reason") or unanswered}
    return None


def basis_holds(root: Path, capture_id: str, basis: str | None, upload_sha256: str | None, now: float,
                owned_list: Path | None = None) -> bool:
    """A basis holds now, by the stored record alone (no network). A recorded definite "no" (``rights_refused``)
    means no basis holds, whatever the basis: Blueprint's own capture too."""
    check = read_check(root)
    if check.get("outcome") == "rights_refused":
        return False
    if basis == "blueprint_owned_capture":
        return owned_basis_holds(capture_id, upload_sha256, owned_list)
    if basis == "site_sharing_statement":
        return live_grant_holds(check, now)
    return False


def basis_still_holds(root: Path, capture_id: str, manifest: Mapping[str, Any], now: float,
                      owned_list: Path | None = None) -> bool:
    """For a route read: the published thing's basis holds now (``basis_holds``)."""
    return basis_holds(root, capture_id, (manifest.get("permission") or {}).get("permission_basis"),
                       manifest.get("upload_sha256"), now, owned_list)


def delivery_gate_allows(capture_root: Path, *, surface: str) -> bool:
    """The serve-time takedown gate (consent re-read live, no open takedown) and no withdrawal."""
    from .consent_takedown import evaluate_delivery_time_takedown_gate
    root = Path(capture_root)
    if capture_withdrawn(root):
        return False
    gate = evaluate_delivery_time_takedown_gate(capture_root=root, surface=surface)
    return gate.get("status") == "allowed"


def site_withdrawn(storage_root: Path, request_id: str) -> bool:
    """A withdrawal of the site found without its capture workspace: any bucket's tombstone for the site.
    A site folder that cannot be read counts as withdrawn (nothing is served on doubt)."""
    try:
        for site in Path(storage_root).glob(f"*/scenes/site-{request_id}"):
            tombstone = site / "website_withdrawal" / "tombstone.json"
            if tombstone.exists() or tombstone.is_symlink():
                return True
    except OSError:
        return True
    return False


__all__ = ["CHECK_DIRECTORY", "basis_holds", "capture_identity", "retract_object", "sha256_file", "LIVE_GRANT_LIFETIME_SECONDS", "LIVE_STATEMENT_RECHECK_SECONDS", "NOT_PERMITTED",
           "RIGHTS_REFUSALS", "RIGHTS_REFUSED", "STATEMENT_CHECK_FILE", "basis_still_holds", "check_path",
           "decided_basis", "delivery_gate_allows", "hidden", "live_due", "live_grant_holds", "live_rights",
           "may_be_permitted", "on_owned_list", "owned_basis_holds", "read_check", "read_json", "reason_code",
           "record_live", "recorded_statement_version", "refused_live", "retraction_pending", "site_withdrawn",
           "take_back"]
