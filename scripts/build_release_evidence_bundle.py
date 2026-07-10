#!/usr/bin/env python3
"""Build a deterministic, scope-complete release evidence bundle."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


POLICY_SCHEMA = "blueprint.release_evidence_retention_policy.v1"
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
IMAGE_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
GROUP_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_written_bundle(
    output_path: Path,
    *,
    manifest_bytes: bytes,
    entries: list[dict[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    expected = {str(entry["path"]): entry for entry in entries}
    try:
        with tarfile.open(output_path, mode="r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                blockers.append("written_bundle_duplicate_member")
            manifest_members = [member for member in members if member.name == "manifest.json"]
            if len(manifest_members) != 1 or not manifest_members[0].isfile():
                blockers.append("written_bundle_manifest_missing")
            else:
                handle = archive.extractfile(manifest_members[0])
                if handle is None or handle.read() != manifest_bytes:
                    blockers.append("written_bundle_manifest_mismatch")
            actual = {
                member.name: member for member in members if member.name != "manifest.json"
            }
            if set(actual) != set(expected):
                blockers.append("written_bundle_entry_set_mismatch")
            for name in sorted(set(actual) & set(expected)):
                member = actual[name]
                handle = archive.extractfile(member) if member.isfile() else None
                if handle is None:
                    blockers.append(f"written_bundle_entry_unreadable:{name}")
                    continue
                digest = hashlib.sha256()
                size = 0
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
                if digest.hexdigest() != expected[name].get("sha256"):
                    blockers.append(f"written_bundle_entry_digest_mismatch:{name}")
                if size != expected[name].get("size_bytes") or member.size != size:
                    blockers.append(f"written_bundle_entry_size_mismatch:{name}")
    except (OSError, tarfile.TarError):
        blockers.append("written_bundle_malformed")
    return sorted(set(blockers))


def _repository_sha(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip().lower() if completed.returncode == 0 else ""


def _repository_timestamp(root: Path, repository_sha: str) -> str:
    completed = subprocess.run(
        ["git", "show", "-s", "--format=%cI", repository_sha],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0 and completed.stdout.strip():
        return completed.stdout.strip()
    return datetime.now(timezone.utc).isoformat()


def _collect_group(group: str, path: Path) -> list[tuple[str, Path]]:
    if not GROUP_PATTERN.fullmatch(group):
        raise ValueError(f"invalid evidence group: {group}")
    if path.is_symlink():
        raise ValueError(f"evidence group may not be a symlink: {group}")
    resolved = path.resolve()
    if resolved.is_file():
        return [(f"evidence/{group}/{resolved.name}", resolved)]
    if not resolved.is_dir():
        raise ValueError(f"evidence path does not exist: {group}")
    files: list[tuple[str, Path]] = []
    for candidate in sorted(resolved.rglob("*")):
        if candidate.is_symlink():
            relative = candidate.relative_to(resolved).as_posix()
            raise ValueError(f"evidence tree contains symlink: {group}/{relative}")
        if candidate.is_file():
            relative = candidate.relative_to(resolved).as_posix()
            files.append((f"evidence/{group}/{relative}", candidate))
    if not files:
        raise ValueError(f"evidence directory is empty: {group}")
    return files


def build_bundle(
    *,
    scope: str,
    repository_sha: str,
    image_digest: str,
    evidence_groups: Mapping[str, Path],
    policy: Mapping[str, Any],
    output_path: Path,
    manifest_path: Path,
    generated_at: str,
) -> dict[str, Any]:
    blockers: list[str] = []
    if policy.get("schema_version") != POLICY_SCHEMA:
        blockers.append("retention_policy_schema_invalid")
    if SHA_PATTERN.fullmatch(repository_sha) is None:
        blockers.append("repository_sha_invalid")
    if IMAGE_PATTERN.fullmatch(image_digest) is None:
        blockers.append("image_digest_invalid")
    required_by_scope = policy.get("required_evidence_groups_by_scope")
    required_map = dict(required_by_scope) if isinstance(required_by_scope, Mapping) else {}
    raw_required = required_map.get(scope)
    if (
        not isinstance(raw_required, list)
        or not raw_required
        or not all(
            isinstance(group, str) and GROUP_PATTERN.fullmatch(group)
            for group in raw_required
        )
        or len(raw_required) != len(set(raw_required))
    ):
        blockers.append(f"release_scope_invalid:{scope}")
        required: list[str] = []
    else:
        required = list(raw_required)
    supplied = set(evidence_groups)
    blockers.extend(f"required_evidence_group_missing:{group}" for group in required if group not in supplied)
    blockers.extend(f"unexpected_evidence_group:{group}" for group in supplied if group not in set(required))

    files: list[tuple[str, Path]] = []
    for group, path in sorted(evidence_groups.items()):
        try:
            files.extend(_collect_group(group, path))
        except ValueError as exc:
            blockers.append(str(exc))
    archive_names = [name for name, _path in files]
    if len(archive_names) != len(set(archive_names)):
        blockers.append("evidence_archive_path_collision")
    blockers = sorted(set(blockers))
    reserved_paths = {output_path.resolve(), manifest_path.resolve()}
    validated_files: list[tuple[str, Path]] = []
    entries: list[dict[str, Any]] = []
    for name, path in files:
        if path.resolve() in reserved_paths:
            blockers.append(f"evidence_output_path_collision:{name}")
            continue
        try:
            before = path.stat()
            digest = _sha256_path(path)
            after = path.stat()
        except OSError:
            blockers.append(f"evidence_file_unreadable:{name}")
            continue
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            blockers.append(f"evidence_file_changed_during_hash:{name}")
            continue
        validated_files.append((name, path))
        entries.append(
            {
                "path": name,
                "size_bytes": after.st_size,
                "sha256": digest,
            }
        )
    blockers = sorted(set(blockers))
    manifest: dict[str, Any] = {
        "schema_version": "blueprint.release_evidence_bundle.v1",
        "generated_at": generated_at,
        "status": "ready_to_archive" if not blockers else "blocked",
        "scope": scope,
        "repository_sha": repository_sha,
        "image_digest": image_digest,
        "required_evidence_groups": required,
        "supplied_evidence_groups": sorted(supplied),
        "entries": entries,
        "entry_count": len(entries),
        "blockers": blockers,
        "claim_boundary": {
            "bundle_is_not_immutable_archive_proof": True,
            "bundle_is_not_signature_or_deployment_proof": True,
            "archive_receipt_is_required_for_retention_closure": True,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if blockers:
        return manifest

    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    entries_by_name = {str(entry["path"]): entry for entry in entries}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("wb") as raw_handle:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_handle, mtime=0
            ) as gzip_handle:
                with tarfile.open(
                    fileobj=gzip_handle, mode="w", format=tarfile.PAX_FORMAT
                ) as archive:
                    manifest_info = tarfile.TarInfo("manifest.json")
                    manifest_info.size = len(manifest_bytes)
                    manifest_info.mtime = 0
                    manifest_info.mode = 0o640
                    manifest_info.uid = manifest_info.gid = 0
                    manifest_info.uname = manifest_info.gname = "root"
                    archive.addfile(manifest_info, io.BytesIO(manifest_bytes))
                    for name, path in validated_files:
                        entry = entries_by_name[name]
                        info = tarfile.TarInfo(name)
                        info.size = int(entry["size_bytes"])
                        info.mtime = 0
                        info.mode = 0o640
                        info.uid = info.gid = 0
                        info.uname = info.gname = "root"
                        with path.open("rb") as handle:
                            archive.addfile(info, handle)
    except (OSError, tarfile.TarError):
        manifest["status"] = "blocked"
        manifest["blockers"] = ["bundle_archive_write_failed"]
        output_path.unlink(missing_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return manifest
    verification_blockers = _verify_written_bundle(
        output_path,
        manifest_bytes=manifest_bytes,
        entries=entries,
    )
    if verification_blockers:
        manifest["status"] = "blocked"
        manifest["blockers"] = verification_blockers
        output_path.unlink(missing_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return manifest
    manifest["bundle_filename"] = output_path.name
    manifest["bundle_size_bytes"] = output_path.stat().st_size
    manifest["bundle_sha256"] = _sha256_path(output_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--repository-sha")
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--evidence", action="append", default=[], metavar="GROUP=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--policy", type=Path, default=Path("docs/release_evidence_retention_policy.json")
    )
    args = parser.parse_args(argv)
    groups: dict[str, Path] = {}
    for value in args.evidence:
        if "=" not in value:
            print(f"[release-evidence-bundle] ERROR invalid --evidence {value!r}", file=sys.stderr)
            return 2
        group, raw_path = value.split("=", 1)
        if group in groups:
            print(f"[release-evidence-bundle] ERROR duplicate group {group}", file=sys.stderr)
            return 2
        groups[group] = Path(raw_path)
    try:
        policy = json.loads(args.policy.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"[release-evidence-bundle] ERROR unreadable_policy:{exc}", file=sys.stderr)
        return 1
    root = args.root.resolve()
    result = build_bundle(
        scope=args.scope.upper(),
        repository_sha=(args.repository_sha or _repository_sha(root)).lower(),
        image_digest=args.image_digest.lower(),
        evidence_groups=groups,
        policy=dict(policy) if isinstance(policy, Mapping) else {},
        output_path=args.output.resolve(),
        manifest_path=args.manifest.resolve(),
        generated_at=_repository_timestamp(
            root, (args.repository_sha or _repository_sha(root)).lower()
        ),
    )
    print(f"[release-evidence-bundle] status={result['status']} manifest={args.manifest}")
    for blocker in result["blockers"]:
        print(f"[release-evidence-bundle] blocker={blocker}", file=sys.stderr)
    return 0 if result["status"] == "ready_to_archive" else 1


if __name__ == "__main__":
    raise SystemExit(main())
