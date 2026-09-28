"""Resolve pinned publisher inputs from operator-bound, rights-admitted host packets.

No network, upload, caller filesystem path, or implicit source discovery is used.
The operator pins historical publisher receipt bytes separately from installation
receipts; source bytes remain on the production host.
"""
from __future__ import annotations

import hashlib
import json
import os
import pwd
import re
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from .decision_evidence_contracts import canonical_digest

BINDINGS_ENV = "BLUEPRINT_TASK_EVALUATION_INSTALLED_SOURCE_BINDINGS_JSON"
DEFAULT_SOURCE_ROOTS = (Path("/var/lib/blueprint/task-evaluation-inputs"),)
_SHA = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


class InstalledSourceBindingError(ValueError):
    """An operator-bound host source failed exact readback."""


def _fail(reason: str) -> None:
    raise InstalledSourceBindingError("installed_source_" + reason)


def _resident(value: Any, roots: Sequence[Path]) -> Path:
    if not isinstance(value, (str, Path)):
        _fail("path_invalid")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        _fail("path_invalid")
    if any(part.is_symlink() for part in (path, *path.parents)):
        _fail("symlink_forbidden")
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        _fail("path_unavailable")
    if not any(resolved.is_relative_to(root.resolve(strict=True)) for root in roots):
        _fail("path_outside_configured_roots")
    if not stat.S_ISREG(resolved.stat().st_mode):
        _fail("file_required")
    return resolved


def _digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            _fail("file_required")
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            count += len(chunk)
    return "sha256:" + digest.hexdigest(), count


def _read_json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 2 * 1024 * 1024:
        _fail("receipt_too_large")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        _fail("receipt_invalid")
    return value


def _identity(row: Mapping[str, Any]) -> tuple[str, int]:
    digest, size = row.get("sha256"), row.get("size_bytes")
    if (not isinstance(digest, str) or not _SHA.fullmatch(digest)
            or isinstance(size, bool) or not isinstance(size, int) or size <= 0):
        _fail("identity_invalid")
    return digest, size


@dataclass(frozen=True)
class InstalledSource:
    path: Path
    digest: str
    size_bytes: int
    installation_receipt_digest: str
    publisher_intake_sha256: str
    installed_by_commit: str = ""

    def verify(self) -> None:
        path = _resident(self.path, (self.path.parent,))
        if _digest(path) != (self.digest, self.size_bytes):
            _fail("file_readback_mismatch")

    def copy_to(self, destination: Path) -> None:
        self.verify()
        descriptor = os.open(self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as source, destination.open("xb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        if _digest(destination) != (self.digest, self.size_bytes):
            _fail("copy_readback_mismatch")


@dataclass(frozen=True)
class InstalledSourceBindings:
    sources: Mapping[str, InstalledSource]

    def resolve(self, uri: str, digest: str, size_bytes: int) -> InstalledSource | None:
        source = self.sources.get(uri)
        if source is None:
            return None
        if (source.digest, source.size_bytes) != (digest, size_bytes):
            _fail("request_identity_mismatch")
        source.verify()
        return source


def load_installed_source_bindings(
    *, expected_source_commit: str, service_account: str,
    environment: Mapping[str, str] | None = None,
    approved_roots: Sequence[Path] = DEFAULT_SOURCE_ROOTS,
    requested_uris: Sequence[str] | None = None,
) -> InstalledSourceBindings:
    """Load operator configuration, optionally selecting exact requested publisher URIs.

    BINDINGS_ENV is a JSON list of installation_receipt_path,
    publisher_intake_path, and publisher_intake_sha256 records. Both paths must
    reside beneath approved_roots. An invalid configured binding fails closed,
    even when an object-store cache already contains matching source bytes.
    A pinned publisher inventory is authenticated before filtering; unrelated
    installations are not checked against this request\'s execution commit.
    None retains full validation for operator audits and direct callers.

    An installation is identified by its content, not by the release that
    installed it: every member is bound by SHA-256 and size to the publisher's
    pinned inventory and read back before use.  The installing commit is still
    required to be a real commit and is recorded for provenance, but it need
    not equal this request's execution commit.  Requiring equality forced a
    fresh multi-gigabyte re-installation and a drop-in re-pin after every
    control-plane deploy, and added nothing the digests do not already prove.
    """
    env = os.environ if environment is None else environment
    raw = env.get(BINDINGS_ENV, "")
    if not raw:
        return InstalledSourceBindings({})
    if not _COMMIT.fullmatch(expected_source_commit):
        _fail("execution_commit_invalid")
    try:
        if pwd.getpwnam(service_account).pw_uid != os.geteuid():
            _fail("service_account_identity_mismatch")
    except KeyError:
        _fail("service_account_missing")
    try:
        bindings = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        _fail("configuration_invalid")
    if not isinstance(bindings, list) or not 1 <= len(bindings) <= 16:
        _fail("configuration_invalid")
    requested = None if requested_uris is None else frozenset(requested_uris)
    sources: dict[str, InstalledSource] = {}
    for binding in bindings:
        if not isinstance(binding, dict) or set(binding) != {
            "installation_receipt_path", "publisher_intake_path", "publisher_intake_sha256",
        }:
            _fail("configuration_invalid")
        publisher_path = _resident(binding["publisher_intake_path"], approved_roots)
        publisher_hash = binding["publisher_intake_sha256"]
        if (not isinstance(publisher_hash, str) or not _SHA.fullmatch(publisher_hash)
                or _digest(publisher_path)[0] != publisher_hash):
            _fail("publisher_receipt_readback_mismatch")
        publisher = _read_json(publisher_path)
        scene_id = publisher.get("scene_id")
        if (not isinstance(scene_id, str) or not scene_id
                or publisher.get("schema_version") not in {
                    "public_scene_publisher_source_intake.v1",
                    f"scene_{scene_id}_publisher_source_intake.v1",
                }
                or publisher.get("status") != "publisher_pinned_sources_verified_on_production"
                or publisher.get("publisher_direct_download") is not True
                or publisher.get("source_uploaded_by_blueprint") is not False
                or publisher.get("public_redistribution_allowed") is not False):
            _fail("publisher_receipt_invalid")
        if ("receipt_digest" in publisher and publisher["receipt_digest"] != canonical_digest(
                publisher, digest_field="receipt_digest")):
            _fail("publisher_receipt_digest_mismatch")
        artifacts = publisher.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            _fail("inventory_invalid")
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                _fail("inventory_invalid")
            digest, size = _identity(artifact)
            uri, revision = artifact.get("publisher_url"), artifact.get("publisher_revision")
            parsed = urlsplit(uri) if isinstance(uri, str) else None
            if (parsed is None or parsed.scheme != "https" or parsed.netloc != "huggingface.co"
                    or parsed.query or parsed.fragment or not isinstance(revision, str)
                    or not _COMMIT.fullmatch(revision)
                    or not re.fullmatch(r"/datasets/[^/]+/[^/]+/resolve/" + revision + r"/[^?#]+",
                                        parsed.path)
                    or ".." in parsed.path.split("/") or "%" in parsed.path):
                _fail("publisher_uri_not_pinned")
        if requested is not None and not requested.intersection(
                artifact["publisher_url"] for artifact in artifacts):
            continue
        installation_path = _resident(binding["installation_receipt_path"], approved_roots)
        installation = _read_json(installation_path)
        if (installation.get("schema_version") != "public_scene_host_input_installation_receipt.v1"
                or installation.get("scene_id") != scene_id
                or installation.get("status") != "installed"
                or installation.get("service_readable") is not True
                or installation.get("service_account") != service_account
                or not _COMMIT.fullmatch(str(installation.get("source_commit_sha") or ""))
                or installation.get("destination_root") != str(installation_path.parent)
                or installation.get("receipt_digest") != canonical_digest(
                    installation, digest_field="receipt_digest")):
            _fail("installation_invalid")
        files = installation.get("files")
        if not isinstance(files, list):
            _fail("inventory_invalid")
        for artifact in artifacts:
            uri = artifact["publisher_url"]
            digest, size = _identity(artifact)
            matches = [row for row in files if isinstance(row, Mapping)
                       and row.get("sha256") == digest and row.get("size_bytes") == size
                       and row.get("role") in {"appearance_3dgs", "semantic_metadata",
                           "scene_structure", "collision_usd", "publisher_scene_usdz"}]
            if len(matches) != 1:
                _fail("installed_member_not_unique")
            relative = matches[0].get("relative_path")
            if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
                    or ".." in Path(relative).parts or "\\" in relative):
                _fail("member_path_invalid")
            path = _resident(installation_path.parent / relative, (installation_path.parent,))
            if uri in sources:
                _fail("publisher_uri_duplicate")
            source = InstalledSource(
                path, digest, size, installation["receipt_digest"], publisher_hash,
                str(installation["source_commit_sha"]),
            )
            source.verify()
            sources[uri] = source
    return InstalledSourceBindings(sources)
