"""Beta L5: exact private robot bytes from operator-admitted company storage.

The request supplies a stable HTTPS identity and digest, never a bucket, object
generation, credential or authorization header. Only an operator-owned profile
may map that exact team's checkpoint to an existing private GCS generation.
"""
from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit


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
