"""Operator withdrawal deletion across every copy of one site the Pipeline itself controls.

Beta journey step: site withdrawal (consent revoked after a site's capture was
prepared and evaluated). Retention follows the Pipeline's real policy (raw
capture is retained, nothing is deleted automatically), but an operator-run
withdrawal deletion must be complete for what the Pipeline controls. The
earlier local cleanup covered only ``scenes/site-<id>/``; launch runs, their
evidence, the B2 offload and the scene workspace archive kept every byte.

``plan`` (the default; deletes nothing) resolves, for one acknowledged withdrawal:

- the scene directory (and ``.retiring-<scene>-*`` leftovers of a workspace
  retirement) and the workspace retirement receipt with its B2 archive;
- every run directory, offload pointer and residue pointer under the run roots
  (the ``evidence_cold`` roots and the intent-keyed scene preparation outputs by
  default) that names the site: its scene id or capture id, a scene intent whose
  website source registration names the site, or a run already bound (a team
  run naming its source launch). A run is bound only by its own records.
  Offloaded runs are judged by their pointer and retained receipts (one naming
  only another site is that site's), or, with ``scan_offloaded_archives``, by
  the records inside their archive, read through the existing client;
- every B2 object those pointers and the bound runs' registered remote result
  artifacts name, with each object version and delete marker listed through
  the existing artifact store client, and every archived Atlas preview of the
  site's walkthrough (found by its name, so also after workspace retirement);
- with a source bucket, the existing exact GCS plan for ``scenes/site-<id>/``.

The plan reads every run's JSON records once (streamed, any size), so it is an
operator tool for a quiet host, not a timer. Anything it cannot resolve fails
closed into ``unresolved`` (a link or special
file in a run, an unreadable record, a run that also names another site, an
offloaded run whose binding is unknown, an object outside the artifact bucket or
also named by an unbound run), and ``apply`` refuses while any remain. ``apply``
needs the exact plan digest and an explicit authorization; it re-resolves and
refuses if anything changed, deletes object versions first and proves their
absence by a complete listing, then removes local copies, and seals a receipt
naming what was deleted and what remains: provider copies it cannot delete,
Pipeline roots it does not cover, and the retained audit and lineage metadata.
"""
from __future__ import annotations

import base64
import fcntl
import json
import os
import re
import shutil
import stat
import tarfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

from . import website_capture_withdrawal as withdrawal
from .control_plane_storage_roots import roots_of_class
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

PLAN_SCHEMA = "website_capture_withdrawal_deletion_plan.v1"
RECEIPT_SCHEMA = "website_capture_withdrawal_deletion_receipt.v1"
_CONTROL_PLANE = Path("/var/lib/blueprint/pipeline-control-plane")
_INPUTS = Path("/var/lib/blueprint/task-evaluation-inputs")
DEFAULT_RUN_ROOTS = (*(Path(root) for root in roots_of_class("evidence_cold")),
                     _INPUTS / "completed-scene-preparation")
DEFAULT_INTENT_ROOT = _CONTROL_PLANE / "task-evaluation-scene-intents"
DEFAULT_BINDING_ROOT = _CONTROL_PLANE / "website-source-bindings"
# Pipeline-controlled roots this deletion does not scan; the receipt names each one present.
UNCOVERED_ROOTS = tuple(_INPUTS / name for name in (
    "completed-scene-preparation-inputs", "sam31-preparations", "compiled-episodes", "prepared-references",
    "task-evaluation-terminal-results", "scene-object-discovery-outputs")) + tuple(
    _CONTROL_PLANE / name for name in ("result-artifact-cache", "capture-reconstruction-derived",
                                       "scene-configuration-diagnostics"))
PROVIDER_COPIES = (
    {"holder": "gpu_and_remote_cpu_providers", "reason": "bundles reached provider hosts; teardown is recorded "
     "per run, provider-side storage erasure is not observable"},
    {"holder": "model_and_reconstruction_providers", "reason": "capture-derived uploads made during preparation "
     "follow each provider's retention terms"},
    {"holder": "object_store_provider_physical_erasure", "reason": "object versions are deleted through the API; "
     "the provider's physical erasure is not observable"},
    {"holder": "webapp_and_delivered_copies", "reason": "WebApp records and copies delivered to robot teams are "
     "outside the Pipeline"},
)
POINTER_SUFFIXES = (".offloaded.v1.json", ".residue.v1.json", ".residue-restore.v1.json")
# Each sidecar's writer seals it: (schemas, digest field, the field naming its run).
_POINTER_SEALS = {
    ".offloaded.v1.json": ({"control_plane_evidence_offload_pointer.v1"}, "pointer_digest", "directory"),
    ".residue.v1.json": ({"control_plane_result_residue_pointer.v1"}, "pointer_digest", "run"),
    ".residue-restore.v1.json": ({"control_plane_result_residue_restore_receipt.v1",
                                  "control_plane_evidence_restore_receipt.v1"}, "receipt_digest", "run"),
}
REGISTRATION_SCHEMA = "website_scene_source_registration.v1"
_SCANNED_SUFFIXES = (".json", ".jsonl")
_TOKEN = re.compile(rb"[A-Za-z0-9._-]+")
_ID = rb"[A-Za-z0-9][A-Za-z0-9._-]{0,119}"
_OTHER_SITE = re.compile(rb"scenes(?:/|\\/)+site-(" + _ID + rb")(?![A-Za-z0-9._-])"
                         rb"|\"(?:scene_id|sceneId)\"\s*:\s*\"site-(" + _ID + rb")\""
                         rb"|(?<![A-Za-z0-9._-])walkthrough-(" + _ID + rb")(?![A-Za-z0-9._-])")
_PROVIDER = re.compile(rb"\"provider(?:_name|_kind)?\"\s*:\s*\"([A-Za-z0-9][A-Za-z0-9._-]{0,63})\"")
_CHUNK, _OVERLAP = 1024 * 1024, 4096
# Run names shorter than this never bind another run: generic names would bind by accident.
_MIN_CLOSURE_NAME = 12


class WithdrawalDeletionError(ValueError):
    """A typed refusal; never carries file contents."""


def _require(value: Any, code: str) -> None:
    if not value:
        raise WithdrawalDeletionError("website_withdrawal_deletion_" + code)


class _Scan:
    """What one run's records name: vocabulary tokens, other sites and providers."""

    def __init__(self, vocabulary: set[bytes], request_id: str):
        self.vocabulary, self.request_id = vocabulary, request_id.encode()
        self.named: set[str] = set()
        self.other_sites: set[str] = set()
        self.providers: set[str] = set()

    def feed(self, stream: BinaryIO) -> None:
        tail = b""
        while True:
            chunk = stream.read(_CHUNK)
            window, final = tail + chunk, not chunk
            if window:
                self._window(window, final)
            if final:
                return
            tail = window[-_OVERLAP:]

    def _window(self, window: bytes, final: bool) -> None:
        # A match touching the window's end is incomplete; the overlap carries it whole.
        for match in _TOKEN.finditer(window):
            if (final or match.end() < len(window)) and match.group() in self.vocabulary:
                self.named.add(match.group().decode())
        for match in _OTHER_SITE.finditer(window):
            if final or match.end() < len(window):
                site = next(group for group in match.groups() if group is not None)
                # ``site-<id>.retired.v1.json`` or ``walkthrough-<id>.mov`` still name this site.
                if site != self.request_id and not site.startswith(self.request_id + b"."):
                    self.other_sites.add(site.decode())
        for match in _PROVIDER.finditer(window):
            self.providers.add(match.group(1).decode())


def _regular(path: Path) -> os.stat_result:
    info = os.lstat(path)
    _require(stat.S_ISREG(info.st_mode), "unsafe_entry")
    return info


def _json(path: Path) -> dict[str, Any]:
    _regular(path)
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError):
        raise WithdrawalDeletionError("website_withdrawal_deletion_record_unreadable") from None
    _require(isinstance(value, dict), "record_unreadable")
    return value


def _site_identity(capture_root: Path) -> tuple[Path, dict[str, str]]:
    site = withdrawal._site(capture_root)
    request_id = site.name[5:]
    try:
        receipt = withdrawal.inspect_withdrawal(capture_root=Path(capture_root))
    except FileNotFoundError:
        raise WithdrawalDeletionError("website_withdrawal_deletion_tombstone_required") from None
    return site, {"request_id": request_id, "scene_id": site.name, "capture_id": f"walkthrough-{request_id}",
                  "tombstone_digest": receipt["tombstone_digest"], "legal_hold": receipt["legal_hold"]}


def _lineage_intents(identity: Mapping[str, str], site: Path, intent_root: Path | None,
                     binding_root: Path | None, unresolved: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Scene intents whose website source registration names the site, and the registrations."""

    request_digests, registrations = set(), []
    if binding_root is not None and binding_root.is_dir():
        for path in sorted(binding_root.glob("*.json")):
            try:
                value = _json(path)
            except WithdrawalDeletionError as error:
                unresolved.append({"path": str(path), "reason": str(error)})
                continue
            # Only a sealed registration may seed deletion lineage; a stale or
            # corrupted one could name another intent and widen the deletion.
            if (value.get("schema_version") != REGISTRATION_SCHEMA
                    or value.get("registration_digest") != canonical_digest(value, digest_field="registration_digest")):
                unresolved.append({"path": str(path), "reason": "website_withdrawal_deletion_registration_unsealed"})
                continue
            source = value.get("capture_source") if isinstance(value.get("capture_source"), dict) else {}
            references = [str(ref.get("path")) for ref in (value.get("references") or {}).values()
                          if isinstance(ref, dict)]
            if (source.get("scene_id") == identity["scene_id"]
                    or any(Path(ref).is_relative_to(site) for ref in references)):
                request_digests.add(value.get("request_digest"))
                registrations.append(str(path))
    intents = []
    if intent_root is not None and intent_root.is_dir():
        for directory in sorted(intent_root.iterdir()):
            if not directory.name.startswith("scene-") or not (directory / "intent.json").exists():
                continue
            try:
                intent = _json(directory / "intent.json")
                digest = cross_runtime_canonical_digest(intent["request"])
            except (WithdrawalDeletionError, KeyError, TypeError, ValueError):
                unresolved.append({"path": str(directory / "intent.json"),
                                   "reason": "website_withdrawal_deletion_record_unreadable"})
                continue
            if digest in request_digests:
                intents.append(directory.name)
    return {"intents": intents, "registrations": registrations}


def _inventory(directory: Path) -> list[list[Any]]:
    rows = []
    for root, directories, files in os.walk(directory):
        for name in sorted((*directories, *files)):
            path = Path(root) / name
            info = os.lstat(path)
            _require(stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode), "unsafe_entry")
            if stat.S_ISREG(info.st_mode):
                rows.append([path.relative_to(directory).as_posix(), info.st_size, info.st_ino, info.st_mtime_ns])
    return sorted(rows)


def _scan_files(paths: Iterable[Path], scan: _Scan) -> None:
    for path in paths:
        _regular(path)
        with path.open("rb") as stream:
            scan.feed(stream)


def _sealed_pointer(path: Path, run: str) -> dict[str, Any]:
    """A sidecar its writer sealed, bound to this run; anything else could name another run's objects."""

    value = _json(path)
    schemas, field, name = next(seal for suffix, seal in _POINTER_SEALS.items() if path.name.endswith(suffix))
    _require(value.get("schema_version") in schemas
             and value.get(field) == canonical_digest(value, digest_field=field)
             and value.get(name) == run, "pointer_unsealed")
    return value


def _pointer_records(pointer: Mapping[str, Any]) -> list[bytes]:
    """The pointer and the receipt bytes it retains are its only local record of an offloaded run."""

    raw = [json.dumps(pointer).encode()]
    for encoded in (pointer.get("retained_receipt_bytes") or {}).values():
        try:
            raw.append(base64.b64decode(str(encoded), validate=True))
        except ValueError:
            raise WithdrawalDeletionError("website_withdrawal_deletion_record_unreadable") from None
    return raw


def _pointer_uri(pointer: Mapping[str, Any]) -> str | None:
    archive = pointer.get("archive") if isinstance(pointer.get("archive"), dict) else {}
    uri = pointer.get("uri") or archive.get("uri")
    return str(uri) if uri else None


def _remote_artifact_uris(run: Path) -> list[str]:
    directory = run / "artifacts/result_delivery/remote_artifacts"
    return sorted(str(_json(path).get("uri")) for path in directory.glob("*.json")) if directory.is_dir() else []


class _Units:
    """Every run under the run roots: its directory, sidecar pointers and dot leftovers."""

    def __init__(self, roots: Sequence[Path], unresolved: list[dict[str, Any]]):
        self.units: dict[tuple[str, str], dict[str, Any]] = {}
        for root in roots:
            if not root.exists():
                continue
            if root.is_symlink() or not root.is_dir():
                unresolved.append({"path": str(root), "reason": "website_withdrawal_deletion_root_unsafe"})
                continue
            dots = []
            for child in sorted(root.iterdir()):
                name = child.name
                pointer = next((suffix for suffix in POINTER_SUFFIXES if name.endswith(suffix)), None)
                if name.startswith("."):
                    dots.append(child)
                elif pointer is not None:
                    self._unit(root, name[: -len(pointer)])["pointers"].append(child)
                elif child.is_symlink() or not child.is_dir():
                    continue  # not a run; the roots hold only run directories and their sidecars
                else:
                    self._unit(root, name)["directory"] = child
            # Offload staging and pointer temporaries are named ``.<run>.<purpose>``.
            for child in dots:
                owners = [name for (unit_root, name) in self.units
                          if unit_root == str(root) and child.name.startswith(f".{name}.")]
                if owners:
                    self.units[(str(root), max(owners, key=len))]["leftovers"].append(child)

    def _unit(self, root: Path, name: str) -> dict[str, Any]:
        return self.units.setdefault((str(root), name), {
            "root": str(root), "name": name, "directory": None, "pointers": [], "leftovers": [],
            "pointer_values": [], "inventory": [], "remote_uris": [], "judged_by": "unjudged"})


def _scan_unit(unit: dict[str, Any], scan: _Scan, *, object_store: Any, scan_archives: bool,
               seeds: set[str]) -> str:
    """``scanned``, ``offloaded_pointer`` or ``offloaded_archive``; raises when it cannot judge the run."""

    directory = unit["directory"]
    pointers = [_sealed_pointer(path, unit["name"]) for path in unit["pointers"]]
    for pointer in pointers:
        for raw in _pointer_records(pointer):
            scan.feed(_Bytes(raw))
    unit["pointer_values"] = pointers
    for path in unit["leftovers"]:
        _require(not path.is_symlink(), "unsafe_entry")
    if directory is not None:
        unit["inventory"] = _inventory(directory)
        _scan_files((directory / row[0] for row in unit["inventory"] if row[0].endswith(_SCANNED_SUFFIXES)), scan)
        unit["remote_uris"] = _remote_artifact_uris(directory)
        return "scanned"
    if not pointers:
        return "leftovers_only"
    if not scan_archives or scan.other_sites or scan.named & seeds:
        return "offloaded_pointer"  # the local record already decides, or the archive may not be read
    for pointer in pointers:
        uri = _pointer_uri(pointer)
        _require(uri and object_store is not None, "offloaded_archive_unreachable")
        try:
            with tarfile.open(fileobj=object_store.open(uri), mode="r|") as archive:
                for member in archive:
                    if member.isfile() and member.name.endswith(_SCANNED_SUFFIXES):
                        scan.feed(archive.extractfile(member))
        except WithdrawalDeletionError:
            raise
        except Exception:  # noqa: BLE001 - provider errors vary; never echo them
            raise WithdrawalDeletionError("website_withdrawal_deletion_offloaded_archive_unreadable") from None
    return "offloaded_archive"


class _Bytes:
    def __init__(self, raw: bytes):
        self.raw, self.offset = raw, 0

    def read(self, size: int) -> bytes:
        chunk = self.raw[self.offset:self.offset + size]
        self.offset += len(chunk)
        return chunk


class B2ObjectVersions:
    """Exact object versions through the existing artifact store client (boto3 S3 API on B2)."""

    def __init__(self, client: Any, bucket: str):
        self.client, self.bucket = client, bucket

    @classmethod
    def configured(cls) -> B2ObjectVersions:
        from .task_evaluation_configured_scene_object_store import _artifact_object_store_client
        return cls(*_artifact_object_store_client())

    def key(self, uri: str) -> str:
        prefix = f"s3://{self.bucket}/"
        _require(uri.startswith(prefix) and len(uri) > len(prefix), "object_outside_artifact_bucket")
        key = uri[len(prefix):]
        _require(all(part not in {"", ".", ".."} for part in key.split("/")), "object_key_invalid")
        return key

    def versions(self, uri: str) -> list[list[str]]:
        # B2 only hides an object deleted without a version id: absence is an empty version listing.
        key, rows, markers = self.key(uri), [], {}
        for _page in range(10_000):
            page = self.client.list_object_versions(Bucket=self.bucket, Prefix=key, **markers)
            for row in [*(page.get("Versions") or []), *(page.get("DeleteMarkers") or [])]:
                if row.get("Key") == key:
                    _require(row.get("VersionId"), "object_listing_invalid")
                    rows.append([key, str(row["VersionId"])])
            if not page.get("IsTruncated"):
                return sorted(rows)
            markers = {"KeyMarker": page["NextKeyMarker"], "VersionIdMarker": page["NextVersionIdMarker"]}
        raise WithdrawalDeletionError("website_withdrawal_deletion_object_listing_unbounded")

    def named(self, prefix: str, filename: str, *, continuation_request_id: str | None = None) -> list[str]:
        """Exact original and job-scoped continuation archive names, in any version."""
        keys, markers = set(), {}
        continuation = (re.compile(re.escape(f"{continuation_request_id}-supplement-") + r"[a-f0-9]{40}\.zip")
                        if continuation_request_id is not None else None)
        for _page in range(10_000):
            page = self.client.list_object_versions(Bucket=self.bucket, Prefix=prefix, **markers)
            for row in [*(page.get("Versions") or []), *(page.get("DeleteMarkers") or [])]:
                key = str(row.get("Key") or "")
                name = key.rpartition("/")[2]
                if key.startswith(prefix) and (name == filename or continuation and continuation.fullmatch(name)):
                    keys.add(key)
            if not page.get("IsTruncated"):
                return sorted(f"s3://{self.bucket}/{key}" for key in keys)
            markers = {"KeyMarker": page["NextKeyMarker"], "VersionIdMarker": page["NextVersionIdMarker"]}
        raise WithdrawalDeletionError("website_withdrawal_deletion_object_listing_unbounded")

    def delete(self, key: str, version: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key, VersionId=version)

    def open(self, uri: str) -> BinaryIO:
        return self.client.get_object(Bucket=self.bucket, Key=self.key(uri))["Body"]


def _scene_workspace(site: Path, identity: Mapping[str, str], unresolved: list[dict[str, Any]]) -> dict[str, Any]:
    scenes = site.parent
    receipt_path = scenes / f"{identity['scene_id']}.retired.v1.json"
    leftovers = sorted(scenes.glob(f".retiring-{identity['scene_id']}-*"))
    for path in leftovers:
        if path.is_symlink() or not path.is_dir():
            unresolved.append({"path": str(path), "reason": "website_withdrawal_deletion_unsafe_entry"})
    value: dict[str, Any] = {"receipt": None, "archive_uri": None,
                             "retiring_leftovers": [str(path) for path in leftovers if path.is_dir()
                                                    and not path.is_symlink()]}
    if receipt_path.exists() or receipt_path.is_symlink():
        try:
            receipt = _json(receipt_path)
            _require(receipt.get("schema_version") == "website_scene_workspace_retired.v1"
                     and receipt.get("scene_id") == identity["scene_id"]
                     and receipt.get("receipt_digest") == canonical_digest(receipt, digest_field="receipt_digest"),
                     "scene_workspace_receipt_invalid")
        except WithdrawalDeletionError as error:
            unresolved.append({"path": str(receipt_path), "reason": str(error)})
        else:
            value["receipt"] = str(receipt_path)
            value["archive_uri"] = _pointer_uri(receipt)
    return value


def _atlas_preview_uris(object_store: Any, identity: Mapping[str, str],
                        unresolved: list[dict[str, Any]]) -> list[str]:
    """Original and same-job revision previews, including after workspace retirement."""
    from .task_evaluation_configured_scene_object_keys import LARGE_ARTIFACT_KEY_PREFIX
    from .website_atlas_preview_store import ARTIFACT_KIND, bundle_filename
    prefix = f"{LARGE_ARTIFACT_KEY_PREFIX}/{ARTIFACT_KIND}/sha256/"
    if object_store is None:
        unresolved.append({"prefix": prefix, "reason": "website_withdrawal_deletion_object_store_unconfigured"})
        return []
    try:
        return object_store.named(prefix, bundle_filename(identity["capture_id"]),
                                  continuation_request_id=identity["request_id"])
    except WithdrawalDeletionError as error:
        unresolved.append({"prefix": prefix, "reason": str(error)})
        return []


def plan_withdrawal_deletion(*, capture_root: Path, run_roots: Sequence[Path] = DEFAULT_RUN_ROOTS,
                             intent_root: Path | None = DEFAULT_INTENT_ROOT,
                             binding_root: Path | None = DEFAULT_BINDING_ROOT, object_store: Any = None,
                             cloud_storage: Any = None, scan_offloaded_archives: bool = False,
                             uncovered_roots: Sequence[Path] = UNCOVERED_ROOTS) -> dict[str, Any]:
    """Resolve every Pipeline-controlled copy of the site; reads only (the GCS plan is the existing one)."""

    site, identity = _site_identity(Path(capture_root))
    unresolved: list[dict[str, Any]] = []
    lineage = _lineage_intents(identity, site, intent_root, binding_root, unresolved)
    units = _Units([Path(root) for root in run_roots], unresolved).units
    seeds = {identity["scene_id"], identity["capture_id"], *lineage["intents"]}
    vocabulary = {token.encode() for token in seeds} | {
        name.encode() for _, name in units if len(name) >= _MIN_CLOSURE_NAME}
    facts: dict[tuple[str, str], _Scan] = {}
    for key, unit in units.items():
        scan = _Scan(vocabulary - {unit["name"].encode()}, identity["request_id"])
        try:
            unit["judged_by"] = _scan_unit(unit, scan, object_store=object_store,
                                           scan_archives=scan_offloaded_archives, seeds=seeds)
        except (WithdrawalDeletionError, OSError, tarfile.TarError) as error:
            reason = str(error) if isinstance(error, WithdrawalDeletionError) else "website_withdrawal_deletion_" \
                "record_unreadable"
            unit["judged_by"] = "unjudged"
            unresolved.append({"path": str(Path(unit["root"]) / unit["name"]), "reason": reason})
        facts[key] = scan
    # A run is bound by naming the site, a site intent, or a bound run; repeat to the fixed point.
    bound_names, bound = set(seeds), set()
    changed = True
    while changed:
        changed = False
        for key, unit in units.items():
            if key not in bound and (unit["name"] in seeds or facts[key].named & bound_names):
                bound.add(key)
                bound_names.add(unit["name"])
                changed = True
    runs, providers = [], set()
    for key in sorted(units):
        unit, scan = units[key], facts[key]
        if key not in bound:
            # An offloaded run whose local record names only another site is that site's run.
            if unit["judged_by"] == "offloaded_pointer" and not scan.other_sites:
                unresolved.append({"path": str(Path(unit["root"]) / unit["name"]),
                                   "reason": "website_withdrawal_deletion_offloaded_run_binding_undetermined"})
            continue
        if scan.other_sites:
            unresolved.append({"path": str(Path(unit["root"]) / unit["name"]),
                               "reason": "website_withdrawal_deletion_run_names_another_site"})
        providers |= scan.providers
        uris = sorted({*unit["remote_uris"], *filter(None, map(_pointer_uri, unit["pointer_values"]))})
        runs.append({"root": unit["root"], "name": unit["name"],
                     "directory": str(unit["directory"]) if unit["directory"] else None,
                     "inventory_digest": canonical_digest({"rows": unit["inventory"]}),
                     "file_count": len(unit["inventory"]), "pointers": [str(path) for path in unit["pointers"]],
                     "leftovers": [str(path) for path in unit["leftovers"]], "object_uris": uris,
                     "judged_by": unit["judged_by"]})
    workspace = _scene_workspace(site, identity, unresolved)
    uris = sorted({uri for run in runs for uri in run["object_uris"]}
                  | ({workspace["archive_uri"]} if workspace["archive_uri"] else set())
                  | set(_atlas_preview_uris(object_store, identity, unresolved)))
    # A content-addressed object an unbound run also names is not this site's to delete.
    unbound_uris = set()
    for key, unit in units.items():
        if key not in bound:
            unbound_uris.update(filter(None, map(_pointer_uri, unit["pointer_values"])))
            unbound_uris.update(unit["remote_uris"])
    objects = []
    for uri in uris:
        if uri in unbound_uris:
            unresolved.append({"uri": uri, "reason": "website_withdrawal_deletion_object_shared_with_unbound_run"})
            continue
        if object_store is None:
            unresolved.append({"uri": uri, "reason": "website_withdrawal_deletion_object_store_unconfigured"})
            continue
        try:
            objects.append({"uri": uri, "versions": object_store.versions(uri)})
        except WithdrawalDeletionError as error:
            unresolved.append({"uri": uri, "reason": str(error)})
    cloud = withdrawal.plan_cloud_cleanup(capture_root=Path(capture_root), storage=cloud_storage) \
        if cloud_storage is not None else None
    local = withdrawal._cleanup_files(site, site / "website_withdrawal")
    present_uncovered = sorted(str(root) for root in uncovered_roots if Path(root).exists())
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA, **{key: identity[key] for key in ("request_id", "scene_id", "capture_id",
                                                                       "tombstone_digest")},
        "legal_hold": identity["legal_hold"], "site_root": str(site),
        "scene_files": {"count": len(local), "digest": canonical_digest({"rows": [
            [row["path"], row["size"], row["inode"], row["mtime_ns"]] for row in local]})},
        "scene_workspace": workspace, "runs": runs, "objects": objects,
        "object_bucket": getattr(object_store, "bucket", None),
        "cloud_plan_digest": cloud["digest"] if cloud else None,
        "cloud_bucket": cloud["bucket"] if cloud else None,
        "lineage": lineage, "run_roots": [str(root) for root in run_roots],
        "providers_named_in_run_evidence": sorted(providers),
        "uncovered_pipeline_roots": present_uncovered,
        "unresolved": sorted(unresolved, key=lambda row: json.dumps(row, sort_keys=True)),
        "scan_offloaded_archives": bool(scan_offloaded_archives),
    }
    plan["digest"] = canonical_digest(plan, digest_field="digest")
    return plan


def _remaining(plan: Mapping[str, Any], cloud_deleted: bool) -> list[dict[str, Any]]:
    remaining = [dict(row) for row in PROVIDER_COPIES]
    if plan["providers_named_in_run_evidence"]:
        remaining.append({"holder": "providers_named_in_run_evidence", "names": plan["providers_named_in_run_evidence"],
                          "reason": "these providers received run inputs; the Pipeline cannot delete their copies"})
    if not cloud_deleted:
        remaining.append({"holder": "capture_source_bucket", "prefix": f"scenes/{plan['scene_id']}/",
                          "reason": "no source bucket was named for this deletion; use --bucket"})
    remaining.extend({"holder": "pipeline_root_not_covered", "path": path,
                      "reason": "not scanned by withdrawal deletion v1; may hold derived copies"}
                     for path in plan["uncovered_pipeline_roots"])
    remaining.append({"holder": "retained_audit_and_lineage_metadata",
                      "paths": [str(Path(plan["site_root"]) / "website_withdrawal"),
                                *plan["lineage"]["registrations"]],
                      "intents": plan["lineage"]["intents"],
                      "reason": "the withdrawal journal and source registrations/intents carry identifiers, "
                                "not capture bytes, and stay as audit"})
    return remaining


def apply_withdrawal_deletion(*, capture_root: Path, expected_plan_digest: str,
                              authorize_complete_deletion: bool = False, **resolution: Any) -> dict[str, Any]:
    """Delete exactly a reviewed plan's copies; refuse anything unresolved, held or changed.

    It holds every same-site Atlas preview lock throughout, so no preview step can
    publish a new copy between this plan's object listing and its deletion; a step
    in progress refuses (``atlas_preview_busy``) and the operator retries.
    """

    _require(authorize_complete_deletion is True, "explicit_authorization_required")
    from .website_atlas_preview_store import JOB_FILE, LOCK_FILE
    site = withdrawal._site(Path(capture_root))
    jobs = {path.parent for name in (JOB_FILE, LOCK_FILE)
            for path in site.glob(f"captures/*/pipeline/atlas/{name}")}
    with ExitStack() as locks:
        for job_dir in sorted(jobs):
            _require(not any(path.is_symlink() for path in (job_dir, *job_dir.parents)), "unsafe_entry")
            try:
                descriptor = os.open(job_dir / LOCK_FILE, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            except FileNotFoundError:
                _require(not (job_dir / JOB_FILE).exists(), "atlas_preview_lock_missing")
                continue
            except OSError:
                raise WithdrawalDeletionError("website_withdrawal_deletion_atlas_preview_lock_unsafe") from None
            lock = locks.enter_context(os.fdopen(descriptor, "rb"))
            _require(stat.S_ISREG(os.fstat(lock.fileno()).st_mode), "atlas_preview_lock_unsafe")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise WithdrawalDeletionError("website_withdrawal_deletion_atlas_preview_busy") from None
        return _apply_withdrawal_deletion(capture_root=capture_root, expected_plan_digest=expected_plan_digest,
                                          **resolution)


def _apply_withdrawal_deletion(*, capture_root: Path, expected_plan_digest: str, **resolution: Any) -> dict[str, Any]:
    _require(re.fullmatch(r"sha256:[0-9a-f]{64}", str(expected_plan_digest)), "plan_digest_invalid")
    plan = plan_withdrawal_deletion(capture_root=capture_root, **resolution)
    _require(plan["digest"] == expected_plan_digest, "plan_changed")
    _require(not plan["unresolved"], "unresolved_copies")
    _require(plan["legal_hold"] is False, "legal_hold")
    store, deleted = resolution.get("object_store"), {"objects": [], "runs": [], "files": []}
    for row in plan["objects"]:
        for key, version in row["versions"]:
            store.delete(key, version)
        _require(store.versions(row["uri"]) == [], "object_versions_remaining")
        deleted["objects"].append({"uri": row["uri"], "versions_deleted": len(row["versions"])})
    for run in plan["runs"]:
        if run["directory"]:
            directory = Path(run["directory"])
            _require(canonical_digest({"rows": _inventory(directory)}) == run["inventory_digest"], "run_changed")
            shutil.rmtree(directory)
            deleted["runs"].append(run["directory"])
        for path in (*run["leftovers"], *run["pointers"]):
            target = Path(path)
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
            deleted["files"].append(path)
    workspace = plan["scene_workspace"]
    for path in workspace["retiring_leftovers"]:
        shutil.rmtree(path)
        deleted["files"].append(path)
    if workspace["receipt"]:
        Path(workspace["receipt"]).unlink()
        deleted["files"].append(workspace["receipt"])
    local_plan = withdrawal.plan_local_cleanup(capture_root=Path(capture_root))
    # The local cleanup deletes only the scene files the reviewed plan covered.
    _require(len(local_plan["files"]) == plan["scene_files"]["count"] and canonical_digest({"rows": [
        [row["path"], row["size"], row["inode"], row["mtime_ns"]] for row in local_plan["files"]]})
        == plan["scene_files"]["digest"], "scene_files_changed")
    local = withdrawal.apply_local_cleanup(capture_root=Path(capture_root), expected_plan_digest=local_plan["digest"],
                                           authorize_local_deletion=True)
    cloud_storage = resolution.get("cloud_storage")
    if cloud_storage is not None:
        withdrawal.apply_cloud_cleanup(capture_root=Path(capture_root), storage=cloud_storage,
                                       expected_plan_digest=plan["cloud_plan_digest"], authorize_cloud_deletion=True)
    remaining = _remaining(plan, cloud_storage is not None)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA, "plan_digest": plan["digest"],
        **{key: plan[key] for key in ("request_id", "scene_id", "capture_id", "tombstone_digest")},
        "deleted": {**deleted, "scene_files": len(local_plan["files"]),
                    "scene_local_cleanup_receipt_digest": local["local_cleanup_receipt_digest"],
                    "capture_source_bucket_prefix": f"scenes/{plan['scene_id']}/" if cloud_storage else None},
        "remaining": remaining,
        "covered_copies_deleted": True,
        "pipeline_controlled_deletion_complete": not plan["uncovered_pipeline_roots"] and cloud_storage is not None,
        "provider_copies_deleted": False, "physical_erasure_proven": False,
        "applied_at_iso": datetime.now(timezone.utc).isoformat(),
    }
    receipt["digest"] = canonical_digest(receipt, digest_field="digest")
    withdrawal._persist(Path(plan["site_root"]) / "website_withdrawal" / "deletion_receipts"
                        / f"{plan['digest'][7:]}.json", receipt)
    return receipt


def run_cli(args: Any) -> dict[str, Any]:
    """``delete``: a dry-run plan by default; ``--apply`` executes exactly the reviewed plan."""

    store = B2ObjectVersions.configured() if args.object_store else None
    cloud = withdrawal.GoogleWebsiteCleanupStorage(args.bucket) if args.bucket else None
    resolution = {"run_roots": tuple(args.run_root or DEFAULT_RUN_ROOTS), "intent_root": args.intent_root,
                  "binding_root": args.binding_root, "object_store": store, "cloud_storage": cloud,
                  "scan_offloaded_archives": args.scan_offloaded_archives}
    if not args.apply:
        return {"mode": "dry_run", "plan": plan_withdrawal_deletion(capture_root=args.capture_root, **resolution)}
    return apply_withdrawal_deletion(capture_root=args.capture_root, expected_plan_digest=args.expected_plan_digest
                                     or "", authorize_complete_deletion=args.authorize_complete_deletion,
                                     **resolution)


def add_cli_arguments(parser: Any) -> None:
    parser.add_argument("--run-root", type=Path, action="append",
                        help="Run root to search (repeatable); defaults to the evidence_cold roots")
    parser.add_argument("--intent-root", type=Path, default=DEFAULT_INTENT_ROOT)
    parser.add_argument("--binding-root", type=Path, default=DEFAULT_BINDING_ROOT)
    parser.add_argument("--object-store", action="store_true",
                        help="Use the configured B2 artifact store client to list and delete object versions")
    parser.add_argument("--scan-offloaded-archives", action="store_true",
                        help="Read offloaded run archives whose pointer cannot show their site")
    parser.add_argument("--apply", action="store_true", help="Delete; without it the deletion is a dry run")
    parser.add_argument("--authorize-complete-deletion", action="store_true")


__all__ = [
    "DEFAULT_RUN_ROOTS",
    "PLAN_SCHEMA",
    "RECEIPT_SCHEMA",
    "B2ObjectVersions",
    "WithdrawalDeletionError",
    "apply_withdrawal_deletion",
    "plan_withdrawal_deletion",
]
