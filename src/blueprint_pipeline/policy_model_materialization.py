"""Stage one immutable private ONNX artifact on a trusted policy worker.

Only the worker host can read the private bucket. The customer container gets
the verified model and interface through a read-only bind mount, never GCP
credentials or access to the scene directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping


_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ID = re.compile(r"model_[0-9a-f-]{36}\Z")
_GENERATION = re.compile(r"[1-9][0-9]{0,24}\Z")
_MAX_BYTES = 16 * 1024 * 1024


def _object_name(artifact: Mapping[str, object], *, bucket: str, team_id: str) -> str:
    digest = artifact.get("sha256")
    artifact_id = artifact.get("artifact_id")
    generation = artifact.get("storage_generation")
    if (not isinstance(digest, str) or not _DIGEST.fullmatch(digest)
            or not isinstance(artifact_id, str) or not _ID.fullmatch(artifact_id)
            or not isinstance(generation, str) or not _GENERATION.fullmatch(generation)
            or not isinstance(artifact.get("size_bytes"), int)
            or isinstance(artifact.get("size_bytes"), bool)
            or not 0 < artifact["size_bytes"] <= _MAX_BYTES):
        raise ValueError("policy_model_artifact_identity_invalid")
    expected = (
        f"policy-models/{urllib.parse.quote(team_id, safe='')}/"
        f"{artifact_id}/{digest[7:]}.onnx"
    )
    if artifact.get("uri") != f"gs://{bucket}/{expected}":
        raise ValueError("policy_model_private_object_binding_mismatch")
    return expected


def _service_account_token() -> str:
    request = urllib.request.Request(
        "http://169.254.169.254/computeMetadata/v1/instance/"
        "service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        value = json.load(response)
    token = value.get("access_token") if isinstance(value, dict) else None
    if not isinstance(token, str) or not token:
        raise ValueError("policy_model_worker_identity_unavailable")
    return token


def _download(artifact: Mapping[str, object], *, bucket: str, object_name: str) -> bytes:
    encoded_bucket = urllib.parse.quote(bucket, safe="")
    encoded_object = urllib.parse.quote(object_name, safe="")
    generation = artifact["storage_generation"]
    url = (f"https://storage.googleapis.com/storage/v1/b/{encoded_bucket}/o/"
           f"{encoded_object}?alt=media&generation={generation}")
    request = urllib.request.Request(url, headers={
        "Authorization": "Bearer " + _service_account_token(),
        "Accept": "application/octet-stream",
    })
    with urllib.request.urlopen(request, timeout=45) as response:
        raw = response.read(_MAX_BYTES + 1)
    if (len(raw) != artifact["size_bytes"]
            or "sha256:" + hashlib.sha256(raw).hexdigest() != artifact["sha256"]):
        raise ValueError("policy_model_download_digest_mismatch")
    return raw


@contextmanager
def staged_policy_model(
    artifact: Mapping[str, object], *, bucket: str, team_id: str, directory: Path,
) -> Iterator[dict[str, str]]:
    """Yield a digest receipt after staging; erase bytes on every exit path."""
    object_name = _object_name(artifact, bucket=bucket, team_id=team_id)
    root = Path("/run/blueprint/company-policy-model")
    if directory.parent != root or directory.is_symlink() or directory.exists():
        raise ValueError("policy_model_worker_stage_path_invalid")
    raw = _download(artifact, bucket=bucket, object_name=object_name)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError("policy_model_worker_stage_root_invalid")
    directory.mkdir(mode=0o700)
    try:
        for name, payload in (
            ("policy.onnx", raw),
            ("artifact.json", json.dumps(dict(artifact), sort_keys=True).encode()),
        ):
            path = directory / name
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            path.chmod(0o444)
        directory.chmod(0o555)
        yield {"status": "digest_verified_private_model_staged",
               "sha256": str(artifact["sha256"]),
               "storage_generation": str(artifact["storage_generation"])}
    finally:
        directory.chmod(0o700)
        for name in ("policy.onnx", "artifact.json"):
            (directory / name).unlink(missing_ok=True)
        directory.rmdir()
