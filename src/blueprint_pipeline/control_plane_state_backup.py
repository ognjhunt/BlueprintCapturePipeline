"""Back up the control-plane host's irreplaceable state, restore it, and purge it after a withdrawal.

The whole control plane runs on one host. Its job ledgers, queues, receipts, money records, consent and
withdrawal records and the owner's standing approvals existed as one copy on one disk. This module keeps
an encrypted copy of exactly that state (class (a)) in a dedicated B2 bucket, and nothing else:

- **What.** ``ALLOWLIST`` names every backed-up root explicitly. A root not on it is not backed up, and
  each run reports the production roots it skipped and why (``skipped_roots``). Site data (class (c):
  capture video, scene workspaces, reconstructions, run evidence, the site-derived stores) is never backed
  up, even under an allowlisted root: ``_site_data`` refuses every such path, except each site's own
  withdrawal journal (tombstone, deletion receipts, purge receipts; never its quarantine). Rebuildable
  bytes (class (b): caches, scratch, releases) are skipped. Roots marked ``records_only`` keep only
  ``.json``/``.jsonl`` records up to ``RECORD_MAX_FILE_BYTES``; every other root keeps regular files up to
  ``MAX_FILE_BYTES``. What a bound excluded is counted per root in the report.
- **How.** One run every 15 minutes (``blueprint-control-plane-state-backup.timer``) packs the files into
  one gzip tar stream, cut into parts of ``PART_BYTES``, each sealed with AES-GCM under the owner's key
  (``ENCRYPTION_KEY_FILE_ENV``). The file index is encrypted too: it names paths, and paths name sites.
  Only the seal is plain: generation id, times, part digests and the key's fingerprint. A run is bounded
  in bytes and time; a run that cannot finish is ``incomplete``, uploads no seal, and is never reported as
  backed up. Seven days of generations are kept (the newest sealed one always).
- **Privacy.** After a site's deletion receipt is sealed, the next run takes a fresh complete generation,
  lists every object version in the backup prefix at that moment, deletes every version of every older
  generation, lists again to prove them absent, and writes ``backup_purge_receipts/<digest>.json`` beside
  the deletion receipt. Until that purge receipt says ``purged`` (or the backup ledger shows that no
  generation older than the deletion survives) the withdrawal stays incomplete with the reason
  ``backup_copies_not_purged`` (``backup_copies_status``, read by ``website_capture_withdrawal_automatic``).
  A bucket that cannot be listed makes the purge ``unknown``, never "nothing found".
- **Restore.** ``restore`` writes a generation back under ``--fs-root`` (a temporary root for a drill),
  verifying every file against the index, and, on a production root, writes the money hold marker
  (``control_plane_restore_money_hold``): paid admission stays closed until the spend guard's reap and the
  billing reconciler have both run after the restore, and then opens by itself.
"""

from __future__ import annotations

import argparse
import fnmatch
import grp
import hashlib
import io
import json
import os
import pwd
import re
import secrets
import stat
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .control_plane_storage_roots import STORAGE_ROOTS, classify_path
from .core.common import write_json
from .decision_evidence_contracts import canonical_digest

SEAL_SCHEMA = "control_plane_state_backup_seal.v1"
INDEX_SCHEMA = "control_plane_state_backup_index.v1"
REPORT_SCHEMA = "control_plane_state_backup_report.v1"
LEDGER_SCHEMA = "control_plane_state_backup_ledger.v1"
PURGE_SCHEMA = "control_plane_state_backup_purge_receipt.v1"
RESTORE_SCHEMA = "control_plane_state_backup_restore_receipt.v1"
#: The withdrawal deletion's sealed receipt (``website_capture_withdrawal_deletion.RECEIPT_SCHEMA``).
DELETION_RECEIPT_SCHEMA = "website_capture_withdrawal_deletion_receipt.v1"

KEY_PREFIX = "control-plane-state/generations/"
RETENTION_SECONDS = 7 * 86400
MAX_FILE_BYTES = 16 * 1024 * 1024
RECORD_MAX_FILE_BYTES = 1024 * 1024
RECORD_SUFFIXES = (".json", ".jsonl")
DEFAULT_MAX_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_MAX_SECONDS = 10 * 60
PART_BYTES = 8 * 1024 * 1024
MAX_LIST_PAGES = 10_000
_PART_MAGIC = b"BPB1"
RESTORE_ACK = "restore-control-plane-state"

DEFAULT_REPORT_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/control-plane-backup")
REPORT_ROOT_ENV = "BLUEPRINT_CONTROL_PLANE_BACKUP_REPORT_ROOT"
SITE_ROOTS_ENV = "BLUEPRINT_WEBSITE_WITHDRAWAL_SITE_ROOTS"
DEFAULT_SITE_ROOTS = ("/var/lib/blueprint/pubsub-handoffs",)
ENCRYPTION_KEY_FILE_ENV = "BLUEPRINT_CONTROL_PLANE_BACKUP_ENCRYPTION_KEY_FILE"
#: The dedicated bucket, through the same file-pair mechanism as the artifact store.
STORE_FILE_ENV = {
    "access_key": "BLUEPRINT_CONTROL_PLANE_BACKUP_ACCESS_KEY_ID_FILE",
    "secret_key": "BLUEPRINT_CONTROL_PLANE_BACKUP_SECRET_ACCESS_KEY_FILE",
    "bucket": "BLUEPRINT_CONTROL_PLANE_BACKUP_BUCKET_FILE",
    "endpoint": "BLUEPRINT_CONTROL_PLANE_BACKUP_ENDPOINT_URL_FILE",
    "region": "BLUEPRINT_CONTROL_PLANE_BACKUP_REGION_FILE",
}
#: Statuses after which no backup can still hold a withdrawn site's deleted data.
BACKUP_CLEARED_STATUSES = frozenset({"purged", "no_backup_taken", "no_backup_before_deletion"})


class StateBackupError(RuntimeError):
    """A backup, purge or restore could not be done safely; never carries file contents."""


class ListingIncomplete(StateBackupError):
    """The bucket's version listing did not finish: what it holds is unknown."""


@dataclass(frozen=True)
class BackupRoot:
    path: str  # absolute host path; a ``*`` segment matches exactly one component
    records_only: bool
    reason: str
    exclude: tuple[str, ...] = ()  # first components below the root that are never backed up


_CP = "/var/lib/blueprint/pipeline-control-plane"
_IN = "/var/lib/blueprint/task-evaluation-inputs"
_SCENES = "/var/lib/blueprint/pubsub-handoffs/*/scenes"


def _roots(records_only: bool, reason: str, *paths: str) -> tuple[BackupRoot, ...]:
    return tuple(BackupRoot(path, records_only, reason) for path in paths)


#: Every root backed up, and why. Nothing else is.
ALLOWLIST: tuple[BackupRoot, ...] = (
    BackupRoot("/etc/blueprint", False, "configuration, launch profiles and the owner's standing approvals; "
               "provider secrets and credentials are supplied by the owner at a rebuild",
               exclude=("provider-secrets", "credentials", "control-plane-backup")),
    BackupRoot("/etc/blueprint-operator-door/tokens.json", False, "the operator door's token store"),
    *_roots(False, "ledger: reservations, locks, leases and single-use spend authority",
            f"{_CP}/disk-reservations", f"{_CP}/provider-locks", f"{_CP}/release-leases", f"{_CP}/remote-cpu-jobs",
            f"{_CP}/storage-pins", f"{_CP}/submission-publication-locks", "/var/lib/blueprint/scene-retirement/coordinator",
            "/var/lib/blueprint/scene-retirement/generations", "/var/lib/blueprint/spend-authority"),
    *_roots(False, "money records: spend ledgers, billing audit, reconciliations and screening budgets",
            f"{_CP}/gpu_spend_guard", f"{_CP}/account-monthly-spend", f"{_CP}/scene-project-spend",
            f"{_CP}/beta-policy-screening", f"{_CP}/agent-execution"),
    *_roots(False, "receipts, standing authorizations and owner-bound consents",
            f"{_CP}/deploy-receipts", f"{_CP}/standing-authorizations", f"{_CP}/cleanup-receipts",
            f"{_CP}/episode-interpretation-rights", f"{_CP}/embodiment-preparation", f"{_CP}/release-retention",
            f"{_CP}/completed-replay-cache-retention", f"{_CP}/storage-gc", f"{_CP}/website-withdrawal-deletion",
            f"{_CP}/task-evaluation-launch-reconciliation", f"{_CP}/task-evaluation-launch-supervision",
            f"{_CP}/unit-failures", f"{_CP}/capacity", f"{_CP}/preflight", f"{_CP}/policy-canary-output",
            f"{_CP}/live_pipeline_control_plane_manifest.json", f"{_CP}/task-evaluation-launch-profile-catalog.json",
            "/var/lib/blueprint/scene-retirement/consents", "/var/lib/blueprint/scene-retirement/journals",
            "/var/lib/blueprint/scene-retirement/journals.metadata", f"{_IN}/g1-team-campaign-registry.json",
            f"{_IN}/g1-team-policy-approvals", f"{_IN}/sam31-profile-registry",
            "/var/lib/blueprint-operator-door/audit", "/var/lib/blueprint-operator-door/experiment-authority",
            str(DEFAULT_REPORT_ROOT)),
    *_roots(True, "materialized inputs and assets: their records only, never bulk bytes",
            f"{_CP}/launch-materialization", f"{_IN}/g1-sonic-assets", "/var/lib/blueprint/robot-models"),
    *_roots(True, "job queues and their state: records only, so jobs in flight resume where they were",
            *(f"{_CP}/{name}" for name in (
                "atlas-provider-poll", "capture-reconstruction-queue", "native-g1-site-registrations",
                "native-g1-team-campaigns", "native-g1-team-policy-runs", "native-g1-team-campaign-work",
                "native-g1-team-policy-work", "rigid-robot-free-probes", "sam31-preparation-executions",
                "scene-object-discoveries", "task-evaluation-configured-controls",
                "task-evaluation-episode-compilations", "task-evaluation-launch-activations",
                "task-evaluation-launch-preparations", "task-evaluation-launches",
                "task-evaluation-policy-canary-dispatches", "task-evaluation-scene-configuration-activation-intents",
                "task-evaluation-scene-constructions", "task-evaluation-scene-intents",
                "task-evaluation-terminal-resource-releases", "task-evaluation-terminal-resource-releases-state",
                "team-evaluations", "team-scene-packets", "website-source-bindings")),
            "/var/lib/blueprint-operator-door/requests"),
    *_roots(True, "work-volume attempts: their ledgers, receipts and result summaries only, never bundles",
            "/mnt/blueprint-work/beta-policy-screening-work", "/mnt/blueprint-work/embodiment-preparation",
            "/mnt/blueprint-work/rigid-robot-free-probe-work"),
    BackupRoot(f"{_SCENES}/site-*/website_withdrawal", False,
               "each site's consent withdrawal journal: tombstone, deletion and purge receipts",
               exclude=("quarantine",)),
    BackupRoot(f"{_SCENES}/*.retired.v1.json", True, "scene workspace retirement receipts"),
)

#: Site data (class (c)). Nothing at or below these is ever backed up, whatever the allowlist says.
SITE_DATA_ROOTS: dict[str, str] = {
    "/var/lib/blueprint/capture-intakes": "admitted capture video and intake bytes",
    "/var/lib/blueprint/capture-intake": "legacy capture video and intake bytes",
    "/var/lib/blueprint/pubsub-handoffs": "site workspaces and the handoff spool",
    f"{_IN}/team-scene-halves": "compiled site scene halves",
    f"{_IN}/completed-scene-preparation": "site source preparation work",
    f"{_IN}/completed-scene-preparation-inputs": "materialized site preparation inputs",
    f"{_IN}/sam31-preparations": "site segmentation artifacts",
    f"{_IN}/compiled-episodes": "compiled site episodes",
    f"{_IN}/prepared-references": "materialized site references",
    f"{_IN}/task-evaluation-terminal-results": "site result indexes and lineage",
    f"{_IN}/scene-object-discovery-outputs": "site object discovery outputs",
    f"{_CP}/result-artifact-cache": "site result downloads",
    f"{_CP}/capture-reconstruction-derived": "site reconstructions",
    f"{_CP}/scene-configuration-diagnostics": "site diagnostic bundles",
}
_SITE_DATA_CLASSES = frozenset({"scene_workspace", "evidence_cold"})
_REBUILDABLE_CLASSES = frozenset({"cache", "scratch", "release", "lane_scratch", "staging"})
_JOURNAL = re.compile(r"/var/lib/blueprint/pubsub-handoffs/[^/]+/scenes/site-[^/]+/website_withdrawal(/.*)?")
_RETIRED = re.compile(r"/var/lib/blueprint/pubsub-handoffs/[^/]+/scenes/[^/]+\.retired\.v1\.json")
_GENERATION = re.compile(r"g-(\d{8}T\d{12}Z)-[0-9a-f]{8}")
#: Directories above a root are recorded only inside the Blueprint trees, never /, /etc or /var themselves.
_OWNED_PREFIXES = ("/var/lib/blueprint", "/mnt/blueprint-work", "/etc/blueprint")


def _now(now: float | None) -> float:
    return time.time() if now is None else float(now)


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _iso_epoch(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.timestamp() if parsed.tzinfo is not None else None


def generation_id(now: float) -> str:
    """Sorts by start time, to the microsecond, so "latest" is the newest generation."""

    return "g-" + datetime.fromtimestamp(now, timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + secrets.token_hex(4)


def generation_started(gen: str) -> float | None:
    match = _GENERATION.fullmatch(gen)
    if match is None:
        return None
    return datetime.strptime(match[1], "%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc).timestamp()


def _site_data(host_path: str) -> bool:
    """Whether a host path is site data (class (c)). A site's withdrawal journal is not, its quarantine is."""

    journal = _JOURNAL.fullmatch(host_path)
    if journal is not None:
        return (journal[1] or "/").split("/")[1:2] == ["quarantine"]
    if _RETIRED.fullmatch(host_path):
        return False
    parts = PurePosixPath(host_path).parts
    for root in SITE_DATA_ROOTS:
        root_parts = PurePosixPath(root).parts
        if parts[:len(root_parts)] == root_parts:
            return True
    classified = classify_path(host_path)
    return classified is not None and classified.storage_class in _SITE_DATA_CLASSES


def _host(fs_root: Path, host_path: str) -> Path:
    return Path(fs_root) / host_path.lstrip("/")


def _expand(root: BackupRoot, fs_root: Path) -> list[str]:
    """Host paths matching ``root.path`` under ``fs_root``; ``*`` matches one component, links are not followed."""

    found = ["/"]
    for part in PurePosixPath(root.path).parts[1:]:
        following: list[str] = []
        for base in found:
            if "*" not in part:
                candidate = str(PurePosixPath(base) / part)
                if os.path.lexists(_host(fs_root, candidate)):
                    following.append(candidate)
                continue
            try:
                with os.scandir(_host(fs_root, base)) as entries:
                    names = sorted(entry.name for entry in entries
                                   if fnmatch.fnmatchcase(entry.name, part) and not entry.is_symlink())
            except OSError:
                continue
            following.extend(str(PurePosixPath(base) / name) for name in names)
        found = following
    return found


def skipped_roots() -> list[dict[str, str]]:
    """Every production storage root that is not backed up, and why."""

    allowed = {root.path for root in ALLOWLIST}
    rows = []
    for root in STORAGE_ROOTS:
        if root.storage_class == "container" or root.path in allowed:
            continue
        if root.path in SITE_DATA_ROOTS or root.storage_class in _SITE_DATA_CLASSES or _site_data(root.path):
            reason = "site_data_never_backed_up"
        elif root.storage_class in _REBUILDABLE_CLASSES:
            reason = "rebuildable_" + root.storage_class
        else:
            reason = "not_on_allowlist"
        rows.append({"path": root.path, "storage_class": root.storage_class, "reason": reason})
    return rows


@dataclass
class _Entry:
    path: str
    kind: str  # "file" or "dir"
    mode: int
    uid: int
    gid: int
    mtime: float
    data: bytes = b""


def _walk(root: BackupRoot, fs_root: Path, stats: dict[str, Any]) -> Iterator[_Entry]:
    """Every directory and kept file below one allowlisted root, in a stable order."""

    limit = RECORD_MAX_FILE_BYTES if root.records_only else MAX_FILE_BYTES
    counts = stats.setdefault(root.path, {"files": 0, "bytes": 0, "excluded": {}})

    def exclude(reason: str) -> None:
        counts["excluded"][reason] = counts["excluded"].get(reason, 0) + 1

    seen = stats.setdefault("_ancestors", set())
    for top in _expand(root, fs_root):
        # The directories above a root come back with their recorded mode and owner, so a restored site
        # journal or queue sits in directories the service account owns, not ones the restore made.
        for ancestor in reversed(PurePosixPath(top).parents):
            text = str(ancestor)
            if text in seen or not text.startswith(_OWNED_PREFIXES):
                continue
            seen.add(text)
            try:
                info = os.lstat(_host(fs_root, text))
            except OSError:
                continue
            if stat.S_ISDIR(info.st_mode):
                yield _Entry(text, "dir", stat.S_IMODE(info.st_mode), info.st_uid, info.st_gid, info.st_mtime)
        pending = [top]
        while pending:
            host_path = pending.pop()
            if _site_data(host_path):
                exclude("site_data")
                continue
            local = _host(fs_root, host_path)
            try:
                info = os.lstat(local)
            except OSError:
                exclude("unreadable")
                continue
            entry = _Entry(host_path, "dir", stat.S_IMODE(info.st_mode), info.st_uid, info.st_gid, info.st_mtime)
            if stat.S_ISDIR(info.st_mode):
                relative = PurePosixPath(host_path).relative_to(top).parts if host_path != top else ()
                if relative and relative[0] in root.exclude:
                    exclude("excluded_by_allowlist")
                    continue
                try:
                    names = sorted(os.listdir(local), reverse=True)
                except OSError:
                    exclude("unreadable")
                    continue
                yield entry
                pending.extend(str(PurePosixPath(host_path) / name) for name in names)
                continue
            if stat.S_ISLNK(info.st_mode):
                exclude("symlink")
                continue
            if not stat.S_ISREG(info.st_mode):
                exclude("special_file")
                continue
            if root.records_only and not host_path.endswith(RECORD_SUFFIXES):
                exclude("not_a_record")
                continue
            if info.st_size > limit:
                exclude("over_file_bound")
                continue
            try:
                with open(local, "rb") as stream:
                    entry.data = stream.read(limit + 1)
            except OSError:
                exclude("unreadable")
                continue
            if len(entry.data) > limit:
                exclude("over_file_bound")
                continue
            entry.kind = "file"
            counts["files"] += 1
            counts["bytes"] += len(entry.data)
            yield entry


# --------------------------------------------------------------------------- encryption


def load_key(path: Path) -> bytes:
    """The owner's 32-byte key: 64 hex characters in a regular file no other user can read."""

    try:
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o007:
            raise StateBackupError("control_plane_backup_key_file_unsafe")
        text = Path(path).read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise StateBackupError("control_plane_backup_key_unreadable") from error
    if re.fullmatch(r"[0-9a-fA-F]{64}", text) is None:
        raise StateBackupError("control_plane_backup_key_invalid")
    return bytes.fromhex(text)


def key_fingerprint(key: bytes) -> str:
    return "sha256:" + hashlib.sha256(b"blueprint-control-plane-backup-key:" + key).hexdigest()[:32]


def _seal(key: bytes, data: bytes, aad: str) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = secrets.token_bytes(12)
    return _PART_MAGIC + nonce + AESGCM(key).encrypt(nonce, data, aad.encode("ascii"))


def _open(key: bytes, blob: bytes, aad: str) -> bytes:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if not blob.startswith(_PART_MAGIC) or len(blob) < 4 + 12 + 16:
        raise StateBackupError("control_plane_backup_part_invalid")
    try:
        return AESGCM(key).decrypt(blob[4:16], blob[16:], aad.encode("ascii"))
    except InvalidTag as error:
        raise StateBackupError("control_plane_backup_part_authentication_failed") from error


# --------------------------------------------------------------------------- the bucket


class S3BackupStore:
    """The dedicated backup bucket through the S3 API (B2), configured by the file pair in ``STORE_FILE_ENV``."""

    def __init__(self, client: Any, bucket: str):
        self.client, self.bucket = client, bucket

    @classmethod
    def configured(cls, environ: Mapping[str, str] | None = None) -> S3BackupStore:
        from .task_evaluation_configured_scene_object_store import _client_from_file_environment

        return cls(*_client_from_file_environment(STORE_FILE_ENV, require_endpoint_and_region=True, environ=environ))

    def put(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentLength=len(data))
        head = self.client.head_object(Bucket=self.bucket, Key=key)
        if int(head.get("ContentLength", -1)) != len(data):
            raise StateBackupError("control_plane_backup_upload_readback_mismatch")

    def get(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def versions(self, prefix: str) -> list[tuple[str, str]]:
        """Every (key, version id) under ``prefix``, delete markers included; ``ListingIncomplete`` otherwise."""

        rows: list[tuple[str, str]] = []
        markers: dict[str, str] = {}
        for _page in range(MAX_LIST_PAGES):
            page = self.client.list_object_versions(Bucket=self.bucket, Prefix=prefix, **markers)
            for row in [*(page.get("Versions") or []), *(page.get("DeleteMarkers") or [])]:
                if not row.get("Key") or not row.get("VersionId"):
                    raise ListingIncomplete("control_plane_backup_listing_invalid")
                rows.append((str(row["Key"]), str(row["VersionId"])))
            if not page.get("IsTruncated"):
                return sorted(rows)
            markers = {"KeyMarker": page["NextKeyMarker"], "VersionIdMarker": page["NextVersionIdMarker"]}
        raise ListingIncomplete("control_plane_backup_listing_unbounded")

    def delete(self, key: str, version: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key, VersionId=version)


def _generation_of(key: str) -> str | None:
    if not key.startswith(KEY_PREFIX):
        return None
    gen = key[len(KEY_PREFIX):].split("/", 1)[0]
    return gen if _GENERATION.fullmatch(gen) else None


def _list(store: Any) -> list[tuple[str, str]]:
    try:
        return list(store.versions(KEY_PREFIX))
    except ListingIncomplete:
        raise
    except Exception as error:  # noqa: BLE001 - any failure to list means the bucket's contents are unknown
        raise ListingIncomplete("control_plane_backup_listing_failed:" + type(error).__name__) from error


def delete_generations(store: Any, select: Callable[[str], bool]) -> dict[str, Any]:
    """Delete every version of every object of each selected generation, from a listing taken now, and prove it
    with a second listing. ``deleted`` only when the second listing shows none of them; ``unknown`` otherwise."""

    try:
        rows = _list(store)
    except ListingIncomplete as error:
        return {"status": "unknown", "reason": str(error), "generations": [], "versions_deleted": 0}
    targets = [(key, version) for key, version in rows
               if (gen := _generation_of(key)) is not None and select(gen)]
    stray = sorted({key for key, _ in rows if _generation_of(key) is None})
    generations = sorted({str(_generation_of(key)) for key, _ in targets})
    deleted = 0
    try:
        for key, version in targets:
            store.delete(key, version)
            deleted += 1
        remaining = [key for key, _ in _list(store) if (gen := _generation_of(key)) is not None and gen in generations]
    except ListingIncomplete as error:
        return {"status": "unknown", "reason": str(error), "generations": generations, "versions_deleted": deleted}
    except Exception as error:  # noqa: BLE001 - a failed delete leaves the result unknown, retried next run
        return {"status": "unknown", "reason": "control_plane_backup_delete_failed:" + type(error).__name__,
                "generations": generations, "versions_deleted": deleted}
    if remaining:
        return {"status": "unknown", "reason": "control_plane_backup_versions_remain", "generations": generations,
                "versions_deleted": deleted, "remaining": len(remaining)}
    result: dict[str, Any] = {"status": "deleted", "generations": generations, "versions_deleted": deleted}
    if stray:
        result["unrecognized_keys"] = len(stray)
    return result


# --------------------------------------------------------------------------- the local ledger


def report_root_from_env(environ: Mapping[str, str] | None = None) -> Path:
    return Path((os.environ if environ is None else environ).get(REPORT_ROOT_ENV) or DEFAULT_REPORT_ROOT)


def ledger_path(report_root: Path) -> Path:
    return Path(report_root) / "ledger.json"


def read_ledger(path: Path) -> dict[str, Any]:
    """The generations this host uploaded. ``FileNotFoundError`` when no backup ever ran; ``ValueError`` when
    the file cannot be trusted."""

    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("schema_version") != LEDGER_SCHEMA \
            or not isinstance(value.get("generations"), list):
        raise ValueError("control_plane_backup_ledger_invalid")
    for row in value["generations"]:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) \
                or not isinstance(row.get("started_at_epoch"), (int, float)):
            raise ValueError("control_plane_backup_ledger_invalid")
    return value


def _write_ledger(path: Path, generations: Sequence[Mapping[str, Any]]) -> None:
    write_json(path, {"schema_version": LEDGER_SCHEMA, "generations": [dict(row) for row in generations]})
    os.chmod(path, 0o644)  # the withdrawal unit, as the service account, reads it


def _mark(generations: list[dict[str, Any]], ids: Sequence[str], status: str, now: float) -> None:
    for row in generations:
        if row["id"] in ids:
            row.update(status=status, **({"deleted_at_epoch": now} if status == "deleted" else {}))


STALE_CODE = "control_plane_backup_stale"
STALE_AFTER_SECONDS = 2 * 3600


def backup_freshness(ledger: Path, *, now: float) -> dict[str, Any]:
    """For the stall alarm: ``not_configured`` (no ledger, so never set up: quiet), ``unreadable``, ``fresh``, or
    ``stale`` with the age of the newest complete generation (or of the first attempt, if none ever completed)."""

    try:
        generations = read_ledger(ledger)["generations"]
    except FileNotFoundError:
        return {"status": "not_configured"}
    except (OSError, ValueError):
        return {"status": "unreadable"}
    sealed = [float(row["started_at_epoch"]) for row in generations if row.get("status") == "sealed"]
    since = max(sealed) if sealed else min((float(row["started_at_epoch"]) for row in generations), default=now)
    age = max(0.0, now - since)
    return {"status": "stale" if age > STALE_AFTER_SECONDS else "fresh", "age_seconds": age,
            "complete_generation_seen": bool(sealed)}


# --------------------------------------------------------------------------- withdrawal purge


def _purge_path(site: Path, deletion_receipt_digest: str) -> Path:
    return Path(site) / "website_withdrawal" / "backup_purge_receipts" / (deletion_receipt_digest.split(":")[-1] + ".json")


def read_purge_receipt(site: Path, deletion_receipt_digest: str) -> dict[str, Any] | None:
    path = _purge_path(site, deletion_receipt_digest)
    try:
        if path.is_symlink():
            return None
        value = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    if (isinstance(value, dict) and value.get("schema_version") == PURGE_SCHEMA
            and value.get("deletion_receipt_digest") == deletion_receipt_digest
            and value.get("digest") == canonical_digest(value, digest_field="digest")):
        return value
    return None


def backup_copies_status(site: Path, deletion_receipt_digest: str, applied_at_iso: Any, *,
                         ledger: Path) -> dict[str, Any]:
    """Whether any backup can still hold what this withdrawal deleted, read fresh from the purge receipt and the
    backup ledger. One of ``BACKUP_CLEARED_STATUSES``, or ``purge_pending`` / ``unknown`` with a reason."""

    purge = read_purge_receipt(site, deletion_receipt_digest)
    if purge is not None and purge.get("outcome") == "purged":
        return {"status": "purged", "purge_receipt_digest": purge["digest"], "purged_at_iso": purge.get("purged_at_iso")}
    try:
        generations = read_ledger(ledger)["generations"]
    except FileNotFoundError:
        return {"status": "no_backup_taken"}
    except (OSError, ValueError):
        return {"status": "unknown", "reason": "backup_ledger_unreadable"}
    applied = _iso_epoch(applied_at_iso)
    if applied is None:
        return {"status": "unknown", "reason": "deletion_applied_at_invalid"}
    older = [row["id"] for row in generations
             if row.get("status") != "deleted" and float(row["started_at_epoch"]) <= applied]
    if not older:
        return {"status": "no_backup_before_deletion"}
    if purge is not None:
        return {"status": "unknown", "reason": purge.get("reason") or "purge_not_proven", "generations": len(older)}
    return {"status": "purge_pending", "reason": "awaiting_a_fresh_backup", "generations": len(older)}


def _deletion_receipts(site_roots: Sequence[Path]) -> tuple[list[tuple[Path, dict[str, Any]]], list[dict[str, str]]]:
    """Every sealed deletion receipt under the site roots, whatever its status: any applied deletion removed bytes
    that older backups may still hold."""

    from .website_capture_withdrawal_automatic import discover_withdrawals

    sites, problems = discover_withdrawals(site_roots)
    found = []
    for site in sites:
        directory = site / "website_withdrawal" / "deletion_receipts"
        if directory.is_symlink() or not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                value = json.loads(path.read_bytes())
            except (OSError, ValueError):
                problems.append({"path": str(path), "reason": "deletion_receipt_unreadable"})
                continue
            if (isinstance(value, dict) and value.get("schema_version") == DELETION_RECEIPT_SCHEMA
                    and value.get("digest") == canonical_digest(value, digest_field="digest")):
                found.append((site, value))
    return found, problems


def _write_purge_receipt(site: Path, deletion: Mapping[str, Any], outcome: Mapping[str, Any], *,
                         fresh: str | None, now: float) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": PURGE_SCHEMA, "deletion_receipt_digest": deletion["digest"],
        "scene_id": deletion.get("scene_id"), "deletion_applied_at_iso": deletion.get("applied_at_iso"),
        "outcome": "purged" if outcome.get("status") == "deleted" else "unknown",
        "reason": outcome.get("reason"), "fresh_generation": fresh,
        "generations_deleted": list(outcome.get("generations") or []),
        "versions_deleted": int(outcome.get("versions_deleted") or 0),
        # The second listing showed none of the deleted generations' objects; physical erasure at the provider
        # is not observable, as for every other object store copy.
        "proven_absent_by_listing": outcome.get("status") == "deleted", "physical_erasure_proven": False,
        "purged_at_iso" if outcome.get("status") == "deleted" else "attempted_at_iso": _iso(now),
    }
    receipt["digest"] = canonical_digest(receipt, digest_field="digest")
    path = _purge_path(site, str(deletion["digest"]))
    if path.parent.is_symlink() or path.is_symlink():
        raise StateBackupError("control_plane_backup_purge_receipt_symlink")
    write_json(path, receipt)
    try:  # owned like the journal it sits in, so the withdrawal unit can read it
        parent = os.lstat(path.parent.parent)
        os.chown(path.parent, parent.st_uid, parent.st_gid)
        os.chown(path, parent.st_uid, parent.st_gid)
    except OSError:
        pass
    return receipt


def purge_withdrawn(store: Any, *, site_roots: Sequence[Path], ledger: Path, fresh: Mapping[str, Any] | None,
                    now: float) -> dict[str, Any]:
    """After a deletion receipt: with a fresh complete generation, delete every older generation and record it."""

    receipts, problems = _deletion_receipts(site_roots)
    pending = [(site, receipt) for site, receipt in receipts
               if backup_copies_status(site, receipt["digest"], receipt.get("applied_at_iso"), ledger=ledger)["status"]
               not in BACKUP_CLEARED_STATUSES]
    result: dict[str, Any] = {"pending": len(pending), "scan_problems": problems[:20], "purged": 0, "unknown": 0}
    if not pending:
        return result
    if fresh is None:
        result["outcome"] = "waiting_for_a_complete_backup"
        return result
    ready = [(site, receipt) for site, receipt in pending
             if (_iso_epoch(receipt.get("applied_at_iso")) or float("inf")) < float(fresh["started_at_epoch"])]
    if not ready:
        result["outcome"] = "waiting_for_a_backup_after_the_deletion"
        return result
    outcome = delete_generations(store, lambda gen: gen != fresh["id"])
    if outcome["status"] == "deleted":
        generations = read_ledger(ledger)["generations"]
        _mark(generations, [row["id"] for row in generations if row["id"] != fresh["id"]], "deleted", now)
        _write_ledger(ledger, generations)
    for site, receipt in ready:
        written = _write_purge_receipt(site, receipt, outcome, fresh=str(fresh["id"]), now=now)
        result["purged" if written["outcome"] == "purged" else "unknown"] += 1
    result.update(outcome=outcome["status"], reason=outcome.get("reason"),
                  versions_deleted=outcome.get("versions_deleted", 0))
    return result


# --------------------------------------------------------------------------- one backup run


class _PartWriter(io.RawIOBase):
    """Cuts the gzip tar stream into encrypted parts and uploads each as it fills."""

    def __init__(self, store: Any, key: bytes, gen: str, *, part_bytes: int, max_bytes: int,
                 deadline: float, clock: Callable[[], float]):
        super().__init__()
        self.store, self.key, self.gen = store, key, gen
        self.part_bytes, self.max_bytes, self.deadline, self.clock = part_bytes, max_bytes, deadline, clock
        self.buffer = bytearray()
        self.parts: list[dict[str, Any]] = []
        self.total = 0

    def writable(self) -> bool:
        return True

    def write(self, data: Any) -> int:
        view = bytes(data)
        self.total += len(view)
        if self.total > self.max_bytes:
            raise StateBackupError("control_plane_backup_byte_bound_reached")
        if self.clock() > self.deadline:
            raise StateBackupError("control_plane_backup_time_bound_reached")
        self.buffer.extend(view)
        while len(self.buffer) >= self.part_bytes:
            self._flush(bytes(self.buffer[:self.part_bytes]))
            del self.buffer[:self.part_bytes]
        return len(view)

    def _flush(self, plain: bytes) -> None:
        index = len(self.parts)
        key = f"{KEY_PREFIX}{self.gen}/part-{index:06d}.bin"
        blob = _seal(self.key, plain, f"{self.gen}:part:{index}")
        self.store.put(key, blob)
        self.parts.append({"key": key, "sha256": "sha256:" + hashlib.sha256(blob).hexdigest(), "bytes": len(blob)})

    def finish(self) -> None:
        if self.buffer or not self.parts:
            self._flush(bytes(self.buffer))
            self.buffer.clear()


def _tarinfo(entry: _Entry) -> tarfile.TarInfo:
    info = tarfile.TarInfo(entry.path.lstrip("/"))
    info.type = tarfile.DIRTYPE if entry.kind == "dir" else tarfile.REGTYPE
    info.mode, info.uid, info.gid, info.mtime = entry.mode, entry.uid, entry.gid, int(entry.mtime)
    info.size = len(entry.data) if entry.kind == "file" else 0
    # By name too: a rebuilt host's service account need not get the same numeric ids.
    info.uname, info.gname = _name(pwd.getpwuid, entry.uid), _name(grp.getgrgid, entry.gid)
    return info


def _name(lookup: Callable[[int], Any], number: int) -> str:
    try:
        return str(lookup(number)[0])
    except (KeyError, OverflowError):
        return ""


def _owner(member: tarfile.TarInfo) -> tuple[int, int]:
    """The member's owner on this host: by name where the name exists here, else the recorded number."""

    uid, gid = member.uid, member.gid
    try:
        uid = pwd.getpwnam(member.uname).pw_uid if member.uname else uid
    except KeyError:
        pass
    try:
        gid = grp.getgrnam(member.gname).gr_gid if member.gname else gid
    except KeyError:
        pass
    return uid, gid


def _enabled_units(fs_root: Path) -> list[str]:
    """The blueprint units enabled on this host, from systemd's wants links, for the rebuild to enable again."""

    names: set[str] = set()
    base = _host(fs_root, "/etc/systemd/system")
    try:
        with os.scandir(base) as entries:
            wants = [entry.path for entry in entries if entry.name.endswith(".wants") and entry.is_dir(follow_symlinks=False)]
    except OSError:
        return []
    for directory in wants:
        try:
            names.update(name for name in os.listdir(directory) if name.startswith("blueprint-"))
        except OSError:
            continue
    return sorted(names)


def run_backup(*, fs_root: Path, store: Any, key: bytes, report_root: Path, site_roots: Sequence[Path] | None = None,
               now: float | None = None, max_bytes: int = DEFAULT_MAX_BYTES, max_seconds: float = DEFAULT_MAX_SECONDS,
               part_bytes: int = PART_BYTES, clock: Callable[[], float] = time.monotonic,
               allowlist: Sequence[BackupRoot] = ALLOWLIST) -> dict[str, Any]:
    """One bounded run: a fresh generation, then retention and any withdrawal purge. Writes ``latest.json``."""

    started = _now(now)
    report_root = Path(report_root)
    # The installer creates it 0755 root; the unit's UMask keeps what this run creates readable.
    report_root.mkdir(parents=True, exist_ok=True)
    ledger = ledger_path(report_root)
    try:
        generations = read_ledger(ledger)["generations"]
    except FileNotFoundError:
        generations = []
    except (OSError, ValueError):
        # An untrusted ledger is never overwritten with a shorter one: rebuild it from the bucket, or stop.
        try:
            listed = {_generation_of(listed_key) for listed_key, _ in _list(store)}
            generations = [{"id": gen, "started_at_epoch": generation_started(gen), "status": "listed"}
                           for gen in sorted(name for name in listed if name is not None)]
        except ListingIncomplete as error:
            report: dict[str, Any] = {"schema_version": REPORT_SCHEMA, "status": "incomplete", "reason": str(error),
                      "started_at_iso": _iso(started)}
            write_json(report_root / "latest.json", report)
            return report
    gen = generation_id(started)
    row: dict[str, Any] = {"id": gen, "started_at_epoch": started, "status": "uploading"}
    generations.append(row)
    _write_ledger(ledger, generations)  # recorded before the first byte leaves the host
    # The enabled-unit list rides inside the generation, beside this report root.
    write_json(report_root / "enabled_units.json", {"units": _enabled_units(fs_root), "observed_at_iso": _iso(started)})
    deadline = clock() + max_seconds
    writer = _PartWriter(store, key, gen, part_bytes=part_bytes, max_bytes=max_bytes, deadline=deadline, clock=clock)
    stats: dict[str, Any] = {}
    files: list[dict[str, Any]] = []
    status, reason = "complete", None
    try:
        with tarfile.open(fileobj=writer, mode="w|gz") as archive:
            for root in allowlist:
                for entry in _walk(root, fs_root, stats):
                    archive.addfile(_tarinfo(entry), io.BytesIO(entry.data) if entry.kind == "file" else None)
                    if entry.kind == "file":
                        files.append({"path": entry.path, "sha256": "sha256:" + hashlib.sha256(entry.data).hexdigest(),
                                      "bytes": len(entry.data)})
        writer.finish()
        index = {"schema_version": INDEX_SCHEMA, "generation": gen, "files": files}
        index_key = f"{KEY_PREFIX}{gen}/index.bin"
        store.put(index_key, _seal(key, json.dumps(index, sort_keys=True).encode("utf-8"), f"{gen}:index"))
        seal = {"schema_version": SEAL_SCHEMA, "generation": gen, "started_at_iso": _iso(started),
                "finished_at_iso": _iso(started if now is not None else time.time()), "key_fingerprint": key_fingerprint(key),
                "parts": writer.parts, "index_key": index_key, "files": len(files),
                "plain_bytes": sum(item["bytes"] for item in files)}
        seal["digest"] = canonical_digest(seal, digest_field="digest")
        store.put(f"{KEY_PREFIX}{gen}/seal.json", json.dumps(seal, sort_keys=True).encode("utf-8"))
    except StateBackupError as error:
        status, reason = "incomplete", str(error)
    except Exception as error:  # noqa: BLE001 - an upload or read failure is an incomplete run, never "backed up"
        status, reason = "incomplete", "control_plane_backup_failed:" + type(error).__name__
    stats.pop("_ancestors", None)
    row["status"] = "sealed" if status == "complete" else "incomplete"
    if status != "complete":
        # Parts already uploaded hold data: delete them, and keep them counted until that is proven.
        cleanup = delete_generations(store, lambda name: name == gen)
        if cleanup["status"] == "deleted":
            _mark(generations, [gen], "deleted", started)
    _write_ledger(ledger, generations)
    fresh = row if status == "complete" else None
    retention: dict[str, Any] = {"status": "skipped", "reason": "no_fresh_complete_generation"}
    if fresh is not None:
        cutoff = started - RETENTION_SECONDS
        retention = delete_generations(
            store, lambda name: name != gen and (generation_started(name) or 0.0) < cutoff)
        if retention["status"] == "deleted":
            _mark(generations, retention["generations"], "deleted", started)
            _write_ledger(ledger, generations)
    roots = [Path(item) for item in site_roots] if site_roots is not None else [
        _host(fs_root, item) for item in (os.getenv(SITE_ROOTS_ENV) or ":".join(DEFAULT_SITE_ROOTS)).split(":") if item]
    try:
        purge = purge_withdrawn(store, site_roots=roots, ledger=ledger, fresh=fresh, now=started)
    except Exception as error:  # noqa: BLE001 - the purge is retried next run; the withdrawal stays incomplete
        purge = {"outcome": "failed", "reason": type(error).__name__}
    report = {
        "schema_version": REPORT_SCHEMA, "status": status, "reason": reason, "generation": gen,
        "started_at_iso": _iso(started), "files": len(files), "plain_bytes": sum(item["bytes"] for item in files),
        "parts": len(writer.parts), "roots": stats, "skipped_roots": skipped_roots(), "retention": retention,
        "withdrawal_purge": purge,
        "bounds": {"max_bytes": max_bytes, "max_seconds": max_seconds, "max_file_bytes": MAX_FILE_BYTES,
                   "record_max_file_bytes": RECORD_MAX_FILE_BYTES},
    }
    write_json(report_root / "latest.json", report)
    return report


# --------------------------------------------------------------------------- restore


def _sealed_generations(store: Any) -> list[str]:
    return sorted({gen for key, _ in _list(store)
                   if (gen := _generation_of(key)) is not None and key == f"{KEY_PREFIX}{gen}/seal.json"})


class _PartReader(io.RawIOBase):
    """Reads a generation's parts one at a time, checking each part's digest and authentication tag."""

    def __init__(self, store: Any, key: bytes, gen: str, parts: Sequence[Mapping[str, Any]]):
        super().__init__()
        self.store, self.key, self.gen, self.parts = store, key, gen, list(parts)
        self.next, self.buffer, self.offset = 0, b"", 0

    def readable(self) -> bool:
        return True

    def readinto(self, target: Any) -> int:
        while self.offset >= len(self.buffer) and self.next < len(self.parts):
            part = self.parts[self.next]
            blob = self.store.get(part["key"])
            if "sha256:" + hashlib.sha256(blob).hexdigest() != part["sha256"]:
                raise StateBackupError("control_plane_backup_part_digest_mismatch")
            self.buffer, self.offset = _open(self.key, blob, f"{self.gen}:part:{self.next}"), 0
            self.next += 1
        count = min(len(target), len(self.buffer) - self.offset)
        target[:count] = memoryview(self.buffer)[self.offset:self.offset + count]
        self.offset += count
        return count


def _write_bytes(path: Path, data: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".restore")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def restore(*, store: Any, key: bytes, fs_root: Path, generation: str = "latest", report_root: Path | None = None,
            hold_marker: Path | None = None, now: float | None = None) -> dict[str, Any]:
    """Write one sealed generation back under ``fs_root``, verify every file, and record a receipt.

    Idempotent: a generation already restored under this root is reported, not restored again. With
    ``hold_marker`` it also writes the money hold that keeps paid admission closed until the spend guard's reap
    and the billing reconciler have covered the gap (``control_plane_restore_money_hold``)."""

    observed = _now(now)
    fs_root = Path(fs_root)
    sealed = _sealed_generations(store)
    if not sealed:
        raise StateBackupError("control_plane_backup_no_sealed_generation")
    gen = sealed[-1] if generation == "latest" else generation
    if gen not in sealed:
        raise StateBackupError("control_plane_backup_generation_not_sealed")
    receipts = _host(fs_root, str(report_root or DEFAULT_REPORT_ROOT)) / "restores"
    receipt_path = receipts / f"{gen}.json"
    if receipt_path.is_file():
        return {**json.loads(receipt_path.read_bytes()), "already_restored": True}
    seal = json.loads(store.get(f"{KEY_PREFIX}{gen}/seal.json"))
    if seal.get("schema_version") != SEAL_SCHEMA or seal.get("generation") != gen \
            or seal.get("digest") != canonical_digest(seal, digest_field="digest"):
        raise StateBackupError("control_plane_backup_seal_invalid")
    if seal.get("key_fingerprint") != key_fingerprint(key):
        raise StateBackupError("control_plane_backup_key_does_not_match_generation")
    index = json.loads(_open(key, store.get(seal["index_key"]), f"{gen}:index"))
    expected = {item["path"]: item for item in index["files"]}
    restored = 0
    as_root = os.geteuid() == 0
    directories: list[tuple[Path, tarfile.TarInfo]] = []
    with tarfile.open(fileobj=_PartReader(store, key, gen, seal["parts"]), mode="r|gz") as archive:
        for member in archive:
            host_path = "/" + member.name
            parts = PurePosixPath(member.name).parts
            if member.name.startswith("/") or ".." in parts or not (member.isdir() or member.isfile()):
                raise StateBackupError("control_plane_backup_member_unsafe")
            if member.isfile() and _site_data(host_path):
                raise StateBackupError("control_plane_backup_member_is_site_data")
            target = _host(fs_root, host_path)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                directories.append((target, member))
                continue
            item = expected.get(host_path)
            source = archive.extractfile(member)
            data = source.read() if source is not None else b""
            if item is None or "sha256:" + hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise StateBackupError("control_plane_backup_file_digest_mismatch")
            _write_bytes(target, data, member.mode)
            if as_root:
                os.chown(target, *_owner(member))
            os.utime(target, (member.mtime, member.mtime))
            restored += 1
    for target, member in reversed(directories):  # after their files, so the times and modes stick
        os.chmod(target, member.mode)
        if as_root:
            os.chown(target, *_owner(member))
        os.utime(target, (member.mtime, member.mtime))
    if restored != len(expected):
        raise StateBackupError("control_plane_backup_restore_incomplete")
    started = generation_started(gen)
    receipt = {"schema_version": RESTORE_SCHEMA, "generation": gen, "generation_started_at_iso": seal["started_at_iso"],
               "restored_at_iso": _iso(observed), "files_restored_and_verified": restored, "fs_root": str(fs_root),
               "seal_digest": seal["digest"], "money_hold_marker": str(hold_marker) if hold_marker else None}
    receipt["digest"] = canonical_digest(receipt, digest_field="digest")
    if hold_marker is not None:
        from .control_plane_restore_money_hold import write_hold

        write_hold(hold_marker, generation=gen, generation_started_at_epoch=started or observed,
                   restored_at_epoch=observed)
    write_json(receipt_path, receipt)
    # The restored ledger predates this generation's seal: rebuild it from the bucket as it is now.
    ledger = ledger_path(_host(fs_root, str(report_root or DEFAULT_REPORT_ROOT)))
    _write_ledger(ledger, [{"id": name, "started_at_epoch": generation_started(name), "status": "sealed"}
                           for name in sealed])
    return receipt


# --------------------------------------------------------------------------- command line


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m blueprint_pipeline.control_plane_state_backup [run|restore|list]``."""

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run", help="one bounded backup run (the timer's)")
    run.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    run.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    back = sub.add_parser("restore", help="restore a sealed generation under --fs-root")
    back.add_argument("--fs-root", type=Path, required=True)
    back.add_argument("--generation", default="latest")
    back.add_argument("--ack", default="")
    sub.add_parser("list", help="list the sealed generations")
    args = parser.parse_args(argv)
    command = args.command or "run"
    report_root = report_root_from_env()
    store = S3BackupStore.configured()
    if command == "list":
        print(json.dumps({"sealed": _sealed_generations(store)}, sort_keys=True))
        return 0
    key_path = os.getenv(ENCRYPTION_KEY_FILE_ENV)
    if not key_path:
        raise SystemExit(f"{ENCRYPTION_KEY_FILE_ENV} is not set")
    key = load_key(Path(key_path))
    if command == "restore":
        production = Path(args.fs_root).resolve() == Path("/")
        if production and args.ack != RESTORE_ACK:
            raise SystemExit(f"refusing: restoring onto / needs --ack {RESTORE_ACK}")
        from .control_plane_restore_money_hold import DEFAULT_MARKER

        receipt = restore(store=store, key=key, fs_root=args.fs_root, generation=args.generation,
                          hold_marker=_host(args.fs_root, str(DEFAULT_MARKER)) if production else None)
        print(json.dumps(receipt, sort_keys=True))
        return 0
    report = run_backup(fs_root=Path("/"), store=store, key=key, report_root=report_root,
                        max_bytes=args.max_bytes, max_seconds=args.max_seconds)
    print(json.dumps({item: report.get(item) for item in ("status", "reason", "generation", "files", "plain_bytes",
                                                          "retention", "withdrawal_purge")}, sort_keys=True))
    # An incomplete run, or a purge that could not be proven, fails the unit so the notifier reports it.
    purge = report.get("withdrawal_purge") or {}
    return 0 if report["status"] == "complete" and purge.get("outcome") in (None, "deleted") else 2


__all__ = [
    "ALLOWLIST", "BACKUP_CLEARED_STATUSES", "BackupRoot", "ListingIncomplete", "S3BackupStore", "SITE_DATA_ROOTS",
    "StateBackupError", "backup_copies_status", "delete_generations", "key_fingerprint", "ledger_path", "load_key",
    "main", "purge_withdrawn", "read_ledger", "read_purge_receipt", "report_root_from_env", "restore", "run_backup",
    "skipped_roots",
]


if __name__ == "__main__":  # pragma: no cover - exercised through the unit
    sys.exit(main())
