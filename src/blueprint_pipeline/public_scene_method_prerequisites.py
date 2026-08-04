"""Materialize exact, rights-bounded prerequisites for ADP-009A methods.

The request may pin identities, but it cannot assert readiness.  This module
re-hashes local bytes, verifies Hugging Face LFS identities at an immutable
revision, and derives license scope from a clean publisher Git checkout.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

import requests

from .decision_evidence_contracts import canonical_digest
from .model_access_env import normalize_model_access_env


REQUEST_SCHEMA_VERSION = "public_scene_method_prerequisite_request.v1"
RECEIPT_SCHEMA_VERSION = "public_scene_method_prerequisite_receipt.v1"
PROGRAM_ID = "arm-decision-proof-v1"
ADP_ITEM = "ADP-009A"

_LICENSE_POLICIES: dict[str, dict[str, Any]] = {
    "CC-BY-NC-4.0": {
        "allowed_use_ceiling": "noncommercial_internal_research",
        "commercial_use_allowed": False,
        "redistribution_allowed_with_conditions": True,
        "attribution_required_when_shared": True,
    },
    "Apache-2.0": {
        "allowed_use_ceiling": "license_terms",
        "commercial_use_allowed": True,
        "redistribution_allowed_with_conditions": True,
        "attribution_required_when_shared": True,
    },
    "BSD-3-Clause": {
        "allowed_use_ceiling": "license_terms",
        "commercial_use_allowed": True,
        "redistribution_allowed_with_conditions": True,
        "attribution_required_when_shared": True,
    },
}


class PublicSceneMethodPrerequisiteError(ValueError):
    """The prerequisite request or observed evidence failed closed."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PublicSceneMethodPrerequisiteError(f"not_json_object:{path.name}")
    return value


def _require_under(path: Path, roots: Sequence[Path]) -> Path:
    resolved = path.expanduser().resolve()
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise PublicSceneMethodPrerequisiteError(f"path_outside_approved_roots:{resolved}")
    return resolved


def _rooted(root: Path, value: str) -> Path:
    if not value or Path(value).is_absolute():
        raise PublicSceneMethodPrerequisiteError("paths_must_be_nonempty_and_relative")
    return _require_under(root / value, (root,))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _file_record(path: Path, *, root: Path, role: str) -> dict[str, Any]:
    _require_under(path, (root,))
    if not path.is_file() or path.stat().st_size <= 0:
        raise PublicSceneMethodPrerequisiteError(f"missing_or_empty:{path.name}")
    return {
        "role": role,
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _remote_head(repository: str) -> str:
    output = subprocess.run(
        ["git", "ls-remote", repository, "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    fields = output.split()
    if len(fields) != 2 or fields[1] != "HEAD" or len(fields[0]) != 40:
        raise PublicSceneMethodPrerequisiteError("rights_authority_remote_head_unresolved")
    return fields[0]


def _git_rights_authority(spec: Mapping[str, Any], *, method_root: Path) -> dict[str, Any]:
    authority_id = str(spec.get("authority_id") or "")
    repo = _rooted(method_root, str(spec.get("local_path") or ""))
    repository = str(spec.get("repository") or "")
    revision = str(spec.get("revision") or "")
    tree = str(spec.get("tree") or "")
    license_id = str(spec.get("license_id") or "")
    if license_id not in _LICENSE_POLICIES:
        raise PublicSceneMethodPrerequisiteError(f"rights_license_policy_unknown:{authority_id}")
    observed_head = _git(repo, "rev-parse", "HEAD")
    observed_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    if observed_head != revision:
        raise PublicSceneMethodPrerequisiteError(
            f"rights_authority_revision_mismatch:{authority_id}"
        )
    if tree and observed_tree != tree:
        raise PublicSceneMethodPrerequisiteError(f"rights_authority_tree_mismatch:{authority_id}")
    if _git(repo, "status", "--porcelain"):
        raise PublicSceneMethodPrerequisiteError(f"rights_authority_checkout_dirty:{authority_id}")
    origin = _git(repo, "remote", "get-url", "origin")
    if origin.rstrip("/").removesuffix(".git") != repository.rstrip("/").removesuffix(".git"):
        raise PublicSceneMethodPrerequisiteError(f"rights_authority_remote_mismatch:{authority_id}")
    remote_head = _remote_head(repository)
    if spec.get("verify_official_remote_head") is True and remote_head != revision:
        raise PublicSceneMethodPrerequisiteError(
            f"rights_authority_remote_head_advanced:{authority_id}"
        )

    evidence: list[dict[str, Any]] = []
    for item in spec.get("evidence_files") or []:
        if not isinstance(item, Mapping):
            raise PublicSceneMethodPrerequisiteError(
                f"rights_authority_evidence_invalid:{authority_id}"
            )
        relative = str(item.get("path") or "")
        path = _rooted(repo, relative)
        text = path.read_text(encoding="utf-8")
        for token in item.get("required_text") or []:
            if str(token) not in text:
                raise PublicSceneMethodPrerequisiteError(
                    f"rights_authority_text_missing:{authority_id}:{relative}"
                )
        evidence.append(_file_record(path, root=method_root, role="rights_authority"))
    if not evidence:
        raise PublicSceneMethodPrerequisiteError(
            f"rights_authority_evidence_missing:{authority_id}"
        )
    return {
        "authority_id": authority_id,
        "kind": "publisher_git_license",
        "repository": repository,
        "revision": observed_head,
        "repository_tree": observed_tree,
        "official_remote_head": remote_head,
        "license_id": license_id,
        **_LICENSE_POLICIES[license_id],
        "evidence_files": evidence,
        "established": True,
    }


def _hf_repository_info(*, repo_type: str, repository: str, revision: str) -> dict[str, Any]:
    if repo_type not in {"dataset", "model"}:
        raise PublicSceneMethodPrerequisiteError("hf_repo_type_invalid")
    normalize_model_access_env()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    namespace = "datasets" if repo_type == "dataset" else "models"
    url = (
        f"https://huggingface.co/api/{namespace}/{quote(repository, safe='/')}"
        f"/revision/{quote(revision, safe='')}?blobs=true"
    )
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        response = requests.get(url, headers=headers, timeout=30)
    except requests.RequestException as exc:
        raise PublicSceneMethodPrerequisiteError("hf_repository_probe_failed") from exc
    if response.status_code != 200:
        raise PublicSceneMethodPrerequisiteError(
            f"hf_repository_probe_http_status:{response.status_code}"
        )
    value = response.json()
    if not isinstance(value, dict) or value.get("sha") != revision:
        raise PublicSceneMethodPrerequisiteError("hf_repository_revision_mismatch")
    return value


def _hf_artifact(
    spec: Mapping[str, Any], *, data_root: Path, authorities: Mapping[str, Any]
) -> dict[str, Any]:
    artifact_id = str(spec.get("artifact_id") or "")
    path = _rooted(data_root, str(spec.get("local_path") or ""))
    publisher = spec.get("publisher")
    if not isinstance(publisher, Mapping):
        raise PublicSceneMethodPrerequisiteError(f"artifact_publisher_invalid:{artifact_id}")
    repository = str(publisher.get("repository") or "")
    revision = str(publisher.get("revision") or "")
    publisher_path = str(publisher.get("path") or "")
    repo_type = str(publisher.get("repo_type") or "")
    info = _hf_repository_info(repo_type=repo_type, repository=repository, revision=revision)
    siblings = [
        item
        for item in info.get("siblings") or []
        if isinstance(item, Mapping) and item.get("rfilename") == publisher_path
    ]
    if len(siblings) != 1:
        raise PublicSceneMethodPrerequisiteError(f"hf_artifact_identity_not_unique:{artifact_id}")
    sibling = siblings[0]
    lfs = sibling.get("lfs")
    if not isinstance(lfs, Mapping):
        raise PublicSceneMethodPrerequisiteError(f"hf_artifact_not_materialized_lfs:{artifact_id}")
    remote_size = int(lfs.get("size") or 0)
    remote_sha = "sha256:" + str(lfs.get("sha256") or "")
    expected_size = int(spec.get("expected_size_bytes") or 0)
    expected_sha = str(spec.get("expected_sha256") or "")
    if remote_size != expected_size or remote_sha != expected_sha:
        raise PublicSceneMethodPrerequisiteError(
            f"hf_artifact_pinned_identity_changed:{artifact_id}"
        )
    local = _file_record(path, root=data_root, role="method_checkpoint")
    if local["size_bytes"] != remote_size or local["sha256"] != remote_sha:
        raise PublicSceneMethodPrerequisiteError(f"hf_artifact_local_bytes_changed:{artifact_id}")
    authority_id = str(spec.get("rights_authority_id") or "")
    authority = authorities.get(authority_id)
    if not isinstance(authority, Mapping) or authority.get("established") is not True:
        raise PublicSceneMethodPrerequisiteError(f"artifact_rights_authority_missing:{artifact_id}")
    return {
        "artifact_id": artifact_id,
        **local,
        "publisher": {
            "service": "huggingface",
            "repo_type": repo_type,
            "repository": repository,
            "revision": revision,
            "path": publisher_path,
            "lfs_sha256": remote_sha,
            "size_bytes": remote_size,
            "gated": info.get("gated", False),
            "private": bool(info.get("private")),
            "fresh_access_probe_passed": True,
        },
        "rights_authority_id": authority_id,
        "rights_established": True,
    }


def _http_artifact(
    spec: Mapping[str, Any], *, data_root: Path, authorities: Mapping[str, Any]
) -> dict[str, Any]:
    artifact_id = str(spec.get("artifact_id") or "")
    path = _rooted(data_root, str(spec.get("local_path") or ""))
    publisher = spec.get("publisher")
    if not isinstance(publisher, Mapping):
        raise PublicSceneMethodPrerequisiteError(f"artifact_publisher_invalid:{artifact_id}")
    service = str(publisher.get("service") or "")
    if service == "google_drive":
        file_id = str(publisher.get("file_id") or "")
        if not file_id or any(
            character not in "-_0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            for character in file_id
        ):
            raise PublicSceneMethodPrerequisiteError(f"google_drive_file_id_invalid:{artifact_id}")
        url = (
            f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        )
    elif service == "http":
        url = str(publisher.get("url") or "")
        if not url.startswith("https://"):
            raise PublicSceneMethodPrerequisiteError(f"artifact_https_url_required:{artifact_id}")
    else:
        raise PublicSceneMethodPrerequisiteError(f"artifact_service_unsupported:{artifact_id}")
    try:
        response = requests.head(url, allow_redirects=True, timeout=30)
    except requests.RequestException as exc:
        raise PublicSceneMethodPrerequisiteError("artifact_head_probe_failed") from exc
    if response.status_code != 200:
        raise PublicSceneMethodPrerequisiteError(
            f"artifact_head_probe_http_status:{artifact_id}:{response.status_code}"
        )
    expected_size = int(spec.get("expected_size_bytes") or 0)
    expected_sha = str(spec.get("expected_sha256") or "")
    try:
        remote_size = int(response.headers.get("Content-Length") or 0)
    except ValueError as exc:
        raise PublicSceneMethodPrerequisiteError(
            f"artifact_remote_size_invalid:{artifact_id}"
        ) from exc
    if remote_size != expected_size:
        raise PublicSceneMethodPrerequisiteError(f"artifact_remote_size_changed:{artifact_id}")
    expected_filename = str(publisher.get("filename") or "")
    disposition = str(response.headers.get("Content-Disposition") or "")
    if (
        expected_filename
        and expected_filename not in disposition
        and not str(response.url).endswith("/" + expected_filename)
    ):
        raise PublicSceneMethodPrerequisiteError(f"artifact_remote_filename_changed:{artifact_id}")
    local = _file_record(path, root=data_root, role="method_checkpoint")
    if local["size_bytes"] != expected_size or local["sha256"] != expected_sha:
        raise PublicSceneMethodPrerequisiteError(f"artifact_local_bytes_changed:{artifact_id}")
    authority_id = str(spec.get("rights_authority_id") or "")
    authority = authorities.get(authority_id)
    if not isinstance(authority, Mapping) or authority.get("established") is not True:
        raise PublicSceneMethodPrerequisiteError(f"artifact_rights_authority_missing:{artifact_id}")
    observed_publisher = {
        "service": service,
        "size_bytes": remote_size,
        "filename": expected_filename,
        "final_url": str(response.url),
        "content_disposition": disposition,
        "etag": response.headers.get("ETag"),
        "last_modified": response.headers.get("Last-Modified"),
        "fresh_access_probe_passed": True,
    }
    if service == "google_drive":
        observed_publisher["file_id"] = str(publisher["file_id"])
    else:
        observed_publisher["url"] = url
    return {
        "artifact_id": artifact_id,
        **local,
        "publisher": observed_publisher,
        "rights_authority_id": authority_id,
        "rights_established": True,
    }


def _artifact(
    spec: Mapping[str, Any], *, data_root: Path, authorities: Mapping[str, Any]
) -> dict[str, Any]:
    publisher = spec.get("publisher")
    if not isinstance(publisher, Mapping):
        raise PublicSceneMethodPrerequisiteError("artifact_publisher_invalid")
    if publisher.get("service") == "huggingface":
        return _hf_artifact(spec, data_root=data_root, authorities=authorities)
    return _http_artifact(spec, data_root=data_root, authorities=authorities)


def materialize_method_prerequisites(
    *, request_path: Path, repo_root: Path, data_root: Path, method_root: Path
) -> dict[str, Any]:
    repo_root = repo_root.expanduser().resolve()
    data_root = data_root.expanduser().resolve()
    method_root = method_root.expanduser().resolve()
    request_path = _require_under(request_path, (repo_root,))
    request = _read_json(request_path)
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise PublicSceneMethodPrerequisiteError("method_prerequisite_request_schema_invalid")
    forbidden = {"status", "admitted", "rights_established"}.intersection(request)
    if forbidden:
        raise PublicSceneMethodPrerequisiteError("caller_asserted_prerequisite_status_forbidden")
    methods = request.get("methods")
    if not isinstance(methods, Mapping) or not methods:
        raise PublicSceneMethodPrerequisiteError("method_prerequisite_methods_missing")

    materialized: dict[str, Any] = {}
    for role, raw_spec in methods.items():
        if not isinstance(raw_spec, Mapping):
            raise PublicSceneMethodPrerequisiteError(f"method_prerequisite_spec_invalid:{role}")
        authorities_list = raw_spec.get("rights_authorities") or []
        authorities: dict[str, Any] = {}
        for authority_spec in authorities_list:
            if not isinstance(authority_spec, Mapping):
                raise PublicSceneMethodPrerequisiteError(f"rights_authority_spec_invalid:{role}")
            observed = _git_rights_authority(authority_spec, method_root=method_root)
            authority_id = str(observed["authority_id"])
            if not authority_id or authority_id in authorities:
                raise PublicSceneMethodPrerequisiteError(f"rights_authority_id_invalid:{role}")
            authorities[authority_id] = observed
        artifacts = [
            _artifact(item, data_root=data_root, authorities=authorities)
            for item in raw_spec.get("artifacts") or []
            if isinstance(item, Mapping)
        ]
        if len(artifacts) != len(raw_spec.get("artifacts") or []) or not artifacts:
            raise PublicSceneMethodPrerequisiteError(
                f"method_prerequisite_artifacts_missing:{role}"
            )
        materialized[str(role)] = {
            "rights_authorities": list(authorities.values()),
            "artifacts": artifacts,
            "checkpoint_rights_established": all(item["rights_established"] for item in artifacts),
            "author_data_rights_established": False,
            "unchanged_author_smoke_executed": False,
            "claim_boundary": {
                "checkpoint_availability_is_not_method_execution": True,
                "checkpoint_rights_do_not_establish_author_data_rights": True,
                "inpainting_result": False,
            },
        }

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "adp_item": ADP_ITEM,
        "evaluated_on": str(request["evaluated_on"]),
        "methods": materialized,
        "raw_secret_values_recorded": False,
        "secret_hashes_recorded": False,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--method-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = _require_under(args.output, (args.data_root.expanduser().resolve(),))
    receipt = materialize_method_prerequisites(
        request_path=args.request,
        repo_root=args.repo_root,
        data_root=args.data_root,
        method_root=args.method_root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "receipt_digest": receipt["receipt_digest"],
                "evaluated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
