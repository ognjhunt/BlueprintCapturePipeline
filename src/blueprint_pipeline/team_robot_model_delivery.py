"""Beta L5: exact private robot bytes from operator-admitted company storage.

The request supplies a stable HTTPS identity and digest, never a bucket, object
generation, credential or authorization header. Only an operator-owned profile
may map that exact team's checkpoint to an existing private GCS generation.
"""
from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from .decision_evidence_contracts import cross_runtime_canonical_digest


MAX_MODEL_BYTES = 512 * 1024 * 1024
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_FIELDS = {"team_id", "checkpoint_id", "model_url", "sha256", "uri", "generation", "size_bytes"}


class RobotModelDeliveryError(ValueError):
    """Fixed refusal without private provider responses or credentials."""


def _require(value, code="configuration_invalid"):
    if not value:
        raise RobotModelDeliveryError("team_robot_model_private_" + code)


def _object(value, *, prefix=False):
    _require(isinstance(value, str) and len(value) <= 1024)
    parsed = urlsplit(value)
    name = parsed.path[1:]
    _require(parsed.scheme == "gs" and re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,220}[a-z0-9]", parsed.netloc)
             and not parsed.query and not parsed.fragment
             and re.fullmatch(r"[A-Za-z0-9._/-]+", name)
             and not any(part in ("", ".", "..") for part in name.rstrip("/").split("/")))
    _require(name.endswith("/") == prefix)
    return parsed.netloc, name


def _selection(profile, manifest, team_id, checkpoint_id):
    config = profile.get("private_robot_models")
    if config is None:
        return None
    _require(isinstance(config, Mapping) and set(config) == {"canonical_uri_prefixes", "artifacts"})
    prefixes, rows = config["canonical_uri_prefixes"], config["artifacts"]
    _require(isinstance(prefixes, list) and 1 <= len(prefixes) <= 16
             and isinstance(rows, list) and 1 <= len(rows) <= 128)
    for prefix in prefixes:
        _object(prefix, prefix=True)
    selected, seen = None, set()
    model = manifest["model"]
    for row in rows:
        _require(isinstance(row, Mapping) and set(row) == _FIELDS)
        bucket, name = _object(row["uri"])
        _require(any(row["uri"].startswith(prefix) for prefix in prefixes)
                 and row["model_url"] == f"https://storage.googleapis.com/{bucket}/{name}"
                 and isinstance(row["sha256"], str) and _DIGEST.fullmatch(row["sha256"])
                 and isinstance(row["generation"], str) and re.fullmatch(r"[1-9][0-9]{0,19}", row["generation"])
                 and type(row["size_bytes"]) is int and 0 < row["size_bytes"] <= MAX_MODEL_BYTES
                 and all(isinstance(row[key], str) and _ID.fullmatch(row[key])
                         for key in ("team_id", "checkpoint_id")))
        identity = row["team_id"], row["checkpoint_id"], row["model_url"]
        _require(identity not in seen)
        seen.add(identity)
        if identity == (team_id, checkpoint_id, model["url"]):
            _require(row["sha256"] == model["sha256"], "digest_binding_mismatch")
            selected = dict(row)
    # An admitted private URL may never fall back to anonymous HTTP because the
    # owner/checkpoint binding is missing or stale.
    if selected is None:
        private_urls = {row["model_url"] for row in rows}
        if model["url"] in private_urls or any(
                model["url"].startswith("https://storage.googleapis.com/" + prefix[5:]) for prefix in prefixes):
            _require(False, "owner_binding_missing")
    return selected


class _BoundedWriter:
    def __init__(self, stream, limit):
        self.stream, self.limit, self.count = stream, limit, 0
        self.digest = hashlib.sha256()

    def write(self, data):
        _require(self.count + len(data) <= self.limit, "size_mismatch")
        written = self.stream.write(data)
        _require(written == len(data), "write_incomplete")
        self.count += written
        self.digest.update(data)
        return written

    def tell(self):
        return self.count

    def flush(self):
        self.stream.flush()


def stage_private_robot_model(*, profile: Mapping, manifest: Mapping, team_id: str,
                              checkpoint_id: str, target: Path, storage_client=None):
    """Return retained delivery provenance, or None for the existing HTTPS path.

    All selections are checked before ADC is used. Reads are generation pinned,
    byte bounded and hash checked. Failure never retries an anonymous URL or
    replaces existing local bytes. A partial owned file remains failure evidence.
    """
    selected = _selection(profile, manifest, team_id, checkpoint_id)
    if selected is None:
        return None
    bucket, name = _object(selected["uri"])
    generation = int(selected["generation"])
    try:
        if storage_client is None:
            from google.cloud import storage
            storage_client = storage.Client()
        blob = storage_client.bucket(bucket).blob(name, generation=generation)
        blob.reload(if_generation_match=generation, timeout=30)
        _require(str(blob.generation) == selected["generation"], "generation_mismatch")
        _require(type(blob.size) is int and blob.size == selected["size_bytes"], "size_mismatch")
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            writer = _BoundedWriter(stream, selected["size_bytes"])
            blob.download_to_file(writer, if_generation_match=generation, timeout=300,
                                  checksum="auto", raw_download=True)
            _require(writer.count == selected["size_bytes"], "size_mismatch")
            _require("sha256:" + writer.digest.hexdigest() == selected["sha256"], "digest_mismatch")
            stream.flush()
            os.fsync(stream.fileno())
    except RobotModelDeliveryError:
        raise
    except Exception:
        raise RobotModelDeliveryError("team_robot_model_private_download_unavailable") from None
    return {"schema_version": "blueprint.team_robot_model_delivery.v1", **selected,
            "delivery_mode": "operator_pinned_private_gcs", "public_access_required": False}


def _walk_directory_fd(root_fd: int, parts: tuple[str, ...], *, create: bool) -> int:
    """Return an owned directory descriptor without resolving path links."""
    current = os.dup(root_fd)
    try:
        for part in parts:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=current)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def _anchored_directory_fd(path: Path, *, create: bool) -> int:
    absolute = path.absolute()
    _require(absolute.anchor == "/" and ".." not in absolute.parts, "closure_invalid")
    root_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        return _walk_directory_fd(root_fd, absolute.parts[1:], create=create)
    finally:
        os.close(root_fd)


def copy_prepared_closure(*, source: Path, destination: Path, closure: Mapping) -> None:
    """Copy a sealed relative USD closure without following filesystem links.

    Used at both operator staging and bundle assembly. The root layer and every
    dependency are verified again while read, so a changed source cannot enter
    a sealed worker bundle under an old receipt.
    """
    _require(isinstance(closure, Mapping)
             and set(closure) == {"schema_version", "model_path", "files", "closure_digest"}
             and closure["schema_version"] == "embodiment_asset_dependency_closure.v1"
             and closure["closure_digest"] == cross_runtime_canonical_digest(
                 dict(closure), digest_field="closure_digest"), "closure_invalid")
    rows = closure["files"]
    _require(isinstance(rows, list) and 1 <= len(rows) <= 10000, "closure_invalid")
    seen, total = set(), 0
    for row in rows:
        _require(isinstance(row, Mapping) and set(row) == {"path", "sha256", "size_bytes"}, "closure_invalid")
        name = row["path"]
        _require(isinstance(name, str) and name and "\\" not in name and "\x00" not in name
                 and not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts
                 and str(PurePosixPath(name)) == name and name not in seen
                 and isinstance(row["sha256"], str) and _DIGEST.fullmatch(row["sha256"])
                 and type(row["size_bytes"]) is int and 0 <= row["size_bytes"] <= MAX_MODEL_BYTES,
                 "closure_invalid")
        seen.add(name)
        total += row["size_bytes"]
    _require(isinstance(closure["model_path"], str) and closure["model_path"] in seen
             and len(PurePosixPath(closure["model_path"]).parts) == 1 and total <= 2 * 1024**3, "closure_invalid")
    try:
        with ExitStack() as roots:
            source_root = _anchored_directory_fd(source, create=False)
            roots.callback(os.close, source_root)
            destination_root = _anchored_directory_fd(destination, create=True)
            roots.callback(os.close, destination_root)
            for row in rows:
                parts = PurePosixPath(row["path"]).parts
                with ExitStack() as parents:
                    source_parent = _walk_directory_fd(source_root, parts[:-1], create=False)
                    parents.callback(os.close, source_parent)
                    destination_parent = _walk_directory_fd(destination_root, parts[:-1], create=True)
                    parents.callback(os.close, destination_parent)
                    source_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                        dir_fd=source_parent)
                    with os.fdopen(source_fd, "rb") as reader:
                        status = os.fstat(reader.fileno())
                        _require(stat.S_ISREG(status.st_mode) and status.st_size == row["size_bytes"],
                                 "closure_bytes_changed")
                        fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                     0o600, dir_fd=destination_parent)
                        with os.fdopen(fd, "wb") as writer:
                            bounded = _BoundedWriter(writer, row["size_bytes"])
                            while chunk := reader.read(1024 * 1024):
                                bounded.write(chunk)
                            _require(bounded.count == row["size_bytes"]
                                     and "sha256:" + bounded.digest.hexdigest() == row["sha256"],
                                     "closure_bytes_changed")
                            writer.flush()
                            os.fsync(writer.fileno())
    except RobotModelDeliveryError:
        raise
    except OSError:
        raise RobotModelDeliveryError("team_robot_model_private_closure_copy_unavailable") from None


def stage_prepared_robot_model(*, profile: Mapping, manifest: Mapping, team_id: str,
                               checkpoint_id: str, target_directory: Path):
    """Stage an operator-selected imported USD with its sealed dependency files.

    Requests never supply filesystem paths. Only the operator registry may map
    an exact owner/checkpoint/model identity to a retained preparation.
    """
    rows = profile.get("prepared_robot_models")
    if rows is None:
        return None
    _require(isinstance(rows, list) and 1 <= len(rows) <= 128)
    selected, seen = None, set()
    for row in rows:
        _require(isinstance(row, Mapping) and set(row) == {
            "team_id", "checkpoint_id", "model_url", "sha256", "directory", "closure_digest"})
        identity = row["team_id"], row["checkpoint_id"], row["model_url"]
        _require(all(isinstance(row[key], str) and _ID.fullmatch(row[key]) for key in ("team_id", "checkpoint_id"))
                 and isinstance(row["model_url"], str) and row["model_url"].startswith("https://")
                 and isinstance(row["directory"], str) and Path(row["directory"]).is_absolute()
                 and isinstance(row["sha256"], str) and _DIGEST.fullmatch(row["sha256"])
                 and isinstance(row["closure_digest"], str) and _DIGEST.fullmatch(row["closure_digest"])
                 and identity not in seen)
        seen.add(identity)
        if identity == (team_id, checkpoint_id, manifest["model"]["url"]):
            _require(row["sha256"] == manifest["model"]["sha256"], "digest_binding_mismatch")
            selected = row
    if selected is None:
        _require(manifest["model"]["url"] not in {row["model_url"] for row in rows}, "owner_binding_missing")
        return None
    from .deployment_embodiment_asset_import import verify_prepared_asset_directory

    directory = Path(selected["directory"])
    try:
        status = directory.lstat()
        _require(stat.S_ISDIR(status.st_mode) and not status.st_mode & 0o022
                 and status.st_uid in {0, os.getuid()}, "prepared_directory_not_operator_owned")
        receipt = verify_prepared_asset_directory(directory)
    except (OSError, ValueError):
        raise RobotModelDeliveryError("team_robot_model_private_prepared_asset_invalid") from None
    closure = receipt["dependency_closure"]
    _require(closure["closure_digest"] == selected["closure_digest"], "closure_digest_mismatch")
    roots = [row for row in closure["files"] if row["path"] == closure["model_path"]]
    _require(len(roots) == 1 and roots[0]["sha256"] == selected["sha256"], "digest_binding_mismatch")
    copy_prepared_closure(source=directory, destination=target_directory, closure=closure)
    return {"schema_version": "blueprint.team_robot_model_delivery.v1", "team_id": team_id,
            "checkpoint_id": checkpoint_id, "model_url": selected["model_url"], "sha256": selected["sha256"],
            "delivery_mode": "operator_sealed_prepared_asset", "dependency_closure": closure,
            "source_import_digest": receipt["import_digest"], "source_import_provenance": receipt,
            "public_access_required": False}
