#!/usr/bin/env python3
"""Build and authenticate the privileged source inventory for a main release.

Git SHA-1 names select objects; they are never authentication evidence here.
Only a SHA256 manifest admitted by trusted keyless verification authenticates
the raw bytes, paths and modes subsequently acquired by the runtime installer.
The verifier must itself be installed from authenticated bytes before import.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import os
import re
import selectors
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "blueprint.source_sha256_manifest.v1"
PIPELINE_REPOSITORY = "ognjhunt/BlueprintCapturePipeline"
CONTRACTS_REPOSITORY = "ognjhunt/BlueprintContracts"
CONTRACTS_COMMIT = "7708a4e4c5dedeeb39cc73d3f6869304de295b81"
PIPELINE_ROOTS = ("src/blueprint_pipeline", "scripts", "deploy/systemd", "uv.lock", "pyproject.toml")
CONTRACTS_ROOTS = ("src/blueprint_contracts", "blueprint_contracts")
MAX_FILES = 32768
MAX_TOTAL_BYTES = 4 * 1024**3
MAX_MANIFEST_BYTES = 16 * 1024**2
MAX_BUNDLE_BYTES = 32 * 1024**2
MAX_LEAF_BYTES = 1024**2
MAX_LOCK_BYTES = 16 * 1024**2
MAX_SECONDS = 900
MAX_VERIFY_OUTPUT_BYTES = 64 * 1024**2
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_BATCH_FILES = 16
PREDICATE_TYPE = "https://github.com/ognjhunt/BlueprintCapturePipeline/attestations/source-sha256-manifest/v1"
_ISSUER = "https://token.actions.githubusercontent.com"
_IDENTITY = f"https://github.com/{PIPELINE_REPOSITORY}/.github/workflows/ci.yml@refs/heads/main"


class ManifestError(ValueError):
    """Source admission failed; callers must not publish or execute a release."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def _deadline(value: float | None = None) -> float:
    now = time.monotonic()
    _require(value is None or (type(value) in {int, float} and math.isfinite(value)), "invalid source verification deadline")
    result = min(now + MAX_SECONDS, value) if value is not None else now + MAX_SECONDS
    _require(result > now, "source verification deadline expired")
    return result


def canonical_manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _parse_json(raw: bytes) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ManifestError("invalid or truncated source evidence JSON") from exc


def _path(path: Any, roots: tuple[str, ...]) -> bool:
    return (isinstance(path, str) and not any(0xD800 <= ord(char) <= 0xDFFF for char in path)
            and len(path.encode("utf-8")) <= 4096
            and not any(ord(char) < 32 or ord(char) == 127 for char in path)
            and "\\" not in path and not path.startswith("/")
            and all(part not in {"", ".", ".."} for part in path.split("/"))
            and len(path.split("/")) <= 32
            and any(path == root or path.startswith(root + "/") for root in roots))


def validate_manifest_bytes(raw: bytes, expected_commit: str) -> dict[str, Any]:
    """Validate canonical shape and policy, not a signature; no admission alone."""
    _require(isinstance(expected_commit, str) and _HEX40.fullmatch(expected_commit) is not None,
             "expected release must be an exact lowercase commit selector")
    _require(isinstance(raw, bytes) and 0 < len(raw) <= MAX_MANIFEST_BYTES, "source manifest size refused")
    manifest = _parse_json(raw)
    _require(isinstance(manifest, dict) and set(manifest) == {"schema_version", "sources"}, "source manifest fields refused")
    _require(manifest["schema_version"] == SCHEMA_VERSION, "source manifest schema refused")
    sources = manifest["sources"]
    _require(isinstance(sources, list) and len(sources) == 2, "both source inventories are required")
    count = total = 0
    for source, repository, commit, roots in zip(sources,
            (PIPELINE_REPOSITORY, CONTRACTS_REPOSITORY),
            (expected_commit, CONTRACTS_COMMIT), (PIPELINE_ROOTS, CONTRACTS_ROOTS), strict=True):
        _require(isinstance(source, dict) and set(source) == {"repository", "commit", "tree", "roots", "files"}, "source inventory fields refused")
        _require(source["repository"] == repository and source["commit"] == commit, "source repository or commit binding refused")
        _require(isinstance(source["tree"], str) and _HEX40.fullmatch(source["tree"]) is not None, "source tree selector refused")
        _require(source["roots"] == list(roots), "privileged source roots refused")
        rows = source["files"]
        _require(isinstance(rows, list) and 0 < len(rows) <= MAX_FILES, "source file inventory refused")
        previous = ""
        present: set[str] = set()
        for row in rows:
            _require(isinstance(row, dict) and set(row) == {"path", "git_blob_oid", "mode", "size", "sha256"}, "source row fields refused")
            path = row["path"]
            _require(_path(path, roots) and path > previous, "source paths must be safe, unique and sorted")
            previous = path
            present.add(path)
            _require(isinstance(row["mode"], str) and row["mode"] in {"100644", "100755"}, "source mode refused")
            _require(isinstance(row["git_blob_oid"], str) and _HEX40.fullmatch(row["git_blob_oid"]) is not None, "source blob selector refused")
            _require(isinstance(row["sha256"], str) and _HEX64.fullmatch(row["sha256"]) is not None, "source SHA256 refused")
            cap = MAX_LOCK_BYTES if path == "uv.lock" else MAX_LEAF_BYTES
            _require(type(row["size"]) is int and 0 <= row["size"] <= cap, "source leaf byte budget refused")
            count += 1
            total += row["size"]
            _require(count <= MAX_FILES and total <= MAX_TOTAL_BYTES, "source inventory budget refused")
        if repository == PIPELINE_REPOSITORY:
            _require(all(any(path == root or path.startswith(root + "/") for path in present) for root in roots), "privileged root inventory missing")
            _require({"uv.lock", "pyproject.toml", "scripts/install_scene_retirement_runtime.py",
                      "scripts/deploy_control_plane_commit.py", "scripts/release_source_manifest.py"}.issubset(present), "privileged admission entrypoints missing")
        else:
            _require(any(root + "/__init__.py" in present for root in roots), "contracts package inventory missing")
    _require(canonical_manifest_bytes(manifest) == raw, "source manifest must use canonical bytes")
    return manifest


def source_inventory(manifest: dict[str, Any], repository: str, commit: str) -> dict[str, dict[str, Any]]:
    """Return rows from an already authenticated and validated manifest."""
    for source in manifest["sources"]:
        if source["repository"] == repository and source["commit"] == commit:
            return {row["path"]: dict(row) for row in source["files"]}
    raise ManifestError("requested source inventory is not bound to this release")


def _run_bounded(command: list[str], *, deadline: float, stdout_cap: int,
                 input_data: bytes = b"", cwd: Path | None = None) -> bytes:
    """Bound time and both pipes, including during batch object acquisition."""
    _require(time.monotonic() < deadline, "source command deadline expired")
    output = bytearray()
    error_size = 0
    sent = 0
    try:
        process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        raise ManifestError("source command or cryptographic verification unavailable") from exc
    try:
        assert process.stdin is not None and process.stdout is not None and process.stderr is not None
        with selectors.DefaultSelector() as selector:
            for pipe, kind in ((process.stdout, "out"), (process.stderr, "err")):
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, kind)
            if input_data:
                os.set_blocking(process.stdin.fileno(), False)
                selector.register(process.stdin, selectors.EVENT_WRITE, "in")
            else:
                process.stdin.close()
            while selector.get_map():
                remaining = deadline - time.monotonic()
                _require(remaining > 0, "source command deadline expired")
                for key, _ in selector.select(min(remaining, 0.25)):
                    if key.data == "in":
                        sent += os.write(key.fd, input_data[sent:sent + 65536])
                        if sent == len(input_data):
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                        continue
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    if key.data == "out":
                        _require(len(output) + len(chunk) <= stdout_cap, "source command output budget refused")
                        output.extend(chunk)
                    else:
                        error_size += len(chunk)
                        _require(error_size <= 65536, "source command error budget refused")
            _require(process.wait(timeout=max(0.001, deadline - time.monotonic())) == 0, "source command or cryptographic verification failed")
        return bytes(output)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ManifestError("source command or cryptographic verification unavailable") from exc
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        for pipe in (process.stdin, process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()


def _git_source(root: Path, repository: str, commit: str, roots: tuple[str, ...], deadline: float,
                budget: list[int]) -> dict[str, Any]:
    _require(_HEX40.fullmatch(commit) is not None, "Git source selector refused")
    prefix = ["git", "--no-replace-objects", "-C", str(root)]
    commit_bytes = _run_bounded(prefix + ["cat-file", "commit", commit], deadline=deadline, stdout_cap=1024**2)
    tree_line = commit_bytes.split(b"\n", 1)[0]
    _require(tree_line.startswith(b"tree ") and _HEX40.fullmatch(tree_line[5:].decode("ascii")) is not None, "Git tree selector refused")
    inventory = _run_bounded(prefix + ["ls-tree", "-r", "-z", "--full-tree", commit, "--", *roots], deadline=deadline, stdout_cap=MAX_MANIFEST_BYTES)
    _require(inventory.endswith(b"\0"), "missing Git source inventory")
    rows = []
    for raw in inventory[:-1].split(b"\0"):
        metadata, path_bytes = raw.split(b"\t", 1)
        mode, kind, oid = metadata.decode("ascii").split(" ")
        path = path_bytes.decode("utf-8")
        _require(mode in {"100644", "100755"} and kind == "blob" and _path(path, roots), "Git source leaf refused")
        _require(_HEX40.fullmatch(oid) is not None, "Git blob selector refused")
        rows.append({"path": path, "mode": mode, "git_blob_oid": oid})
        _require(len(rows) <= MAX_FILES, "Git source file budget refused")
    rows.sort(key=lambda row: row["path"])
    budget[0] += len(rows)
    _require(budget[0] <= MAX_FILES, "Git source combined file budget refused")
    for start in range(0, len(rows), _BATCH_FILES):
        batch = rows[start:start + _BATCH_FILES]
        batch_cap = sum((MAX_LOCK_BYTES if row["path"] == "uv.lock" else MAX_LEAF_BYTES) + 128 for row in batch)
        raw = _run_bounded(prefix + ["cat-file", "--batch"],
                           input_data="".join(row["git_blob_oid"] + "\n" for row in batch).encode("ascii"),
                           deadline=deadline, stdout_cap=min(batch_cap, MAX_TOTAL_BYTES - budget[1] + 128 * len(batch)))
        offset = 0
        for row in batch:
            end = raw.find(b"\n", offset)
            _require(end >= offset, "truncated Git object header")
            parts = raw[offset:end].split(b" ")
            _require(len(parts) == 3 and parts[0] == row["git_blob_oid"].encode("ascii") and parts[1] == b"blob" and parts[2].isdigit(), "Git object binding refused")
            size = int(parts[2])
            cap = MAX_LOCK_BYTES if row["path"] == "uv.lock" else MAX_LEAF_BYTES
            _require(size <= cap, "Git source leaf byte budget refused")
            budget[1] += size
            _require(budget[1] <= MAX_TOTAL_BYTES, "Git source combined byte budget refused")
            offset = end + 1
            _require(len(raw) > offset + size and raw[offset + size:offset + size + 1] == b"\n", "truncated Git object body")
            row["size"] = size
            row["sha256"] = hashlib.sha256(raw[offset:offset + size]).hexdigest()
            offset += size + 1
        _require(offset == len(raw), "unexpected Git batch bytes")
    return {"repository": repository, "commit": commit, "tree": tree_line[5:].decode("ascii"), "roots": list(roots), "files": rows}


def build_manifest(pipeline_root: Path, contracts_root: Path, expected_commit: str, *, deadline: float | None = None) -> bytes:
    """Hash complete committed source inventories with bounded batch Git reads."""
    deadline = _deadline(deadline)
    budget = [0, 0]
    manifest = {"schema_version": SCHEMA_VERSION, "sources": [
        _git_source(pipeline_root, PIPELINE_REPOSITORY, expected_commit, PIPELINE_ROOTS, deadline, budget),
        _git_source(contracts_root, CONTRACTS_REPOSITORY, CONTRACTS_COMMIT, CONTRACTS_ROOTS, deadline, budget),
    ]}
    raw = canonical_manifest_bytes(manifest)
    validate_manifest_bytes(raw, expected_commit)
    return raw


def _read_regular(path: Path, cap: int) -> tuple[bytes, tuple[int, int, int, int, int]]:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            _require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap, "source evidence file refused")
            raw = handle.read(cap + 1)
            after = os.fstat(handle.fileno())
            def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
                return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
            _require(len(raw) == before.st_size and identity(before) == identity(after), "source evidence changed during read")
            return raw, identity(before)
    except OSError as exc:
        raise ManifestError("source evidence missing or unsafe") from exc


def attestation_command(gh_executable: Path, manifest_path: Path, bundle_path: Path, expected_commit: str) -> list[str]:
    """Fixed crypto policy; exact SAN also fixes ci.yml/main signer identity.

    gh makes --cert-identity and --signer-workflow mutually exclusive. Both the
    certificate signer digest and source digest must be the exact release SHA.
    """
    return [str(gh_executable), "attestation", "verify", str(manifest_path),
            "--bundle", str(bundle_path), "--repo", PIPELINE_REPOSITORY,
            "--hostname", "github.com", "--cert-identity", _IDENTITY,
            "--cert-oidc-issuer", _ISSUER, "--source-ref", "refs/heads/main",
            "--source-digest", expected_commit, "--signer-digest", expected_commit,
            "--deny-self-hosted-runners", "--digest-alg", "sha256",
            "--predicate-type", PREDICATE_TYPE, "--limit", "1", "--format", "json"]


def _bound_statement(statement: Any, manifest_raw: bytes) -> None:
    _require(isinstance(statement, dict) and statement.get("_type") == "https://in-toto.io/Statement/v1"
             and statement.get("predicateType") == PREDICATE_TYPE, "source statement type refused")
    subjects = statement.get("subject")
    _require(isinstance(subjects, list) and len(subjects) == 1 and isinstance(subjects[0], dict)
             and subjects[0].get("digest") == {"sha256": hashlib.sha256(manifest_raw).hexdigest()}, "source manifest subject digest mismatch")
    _require(isinstance(statement.get("predicate"), dict)
             and canonical_manifest_bytes(statement["predicate"]) == manifest_raw, "source predicate and subject bytes differ")


def extract_untrusted_manifest(bundle_raw: bytes, expected_commit: str, expected_digest: str) -> bytes:
    """Extract candidate DATA for gh verification; this never grants authority.

    Public artifact names and DSSE payloads are unauthenticated locators until
    verify_manifest_attestation succeeds against a protected gh executable.
    """
    _require(0 < len(bundle_raw) <= MAX_BUNDLE_BYTES, "source bundle size refused")
    _require(isinstance(expected_digest, str) and _HEX64.fullmatch(expected_digest) is not None, "source manifest locator digest refused")
    bundle = _parse_json(bundle_raw)
    envelope = bundle.get("dsseEnvelope") if isinstance(bundle, dict) else None
    _require(isinstance(envelope, dict) and envelope.get("payloadType") == "application/vnd.in-toto+json", "source DSSE envelope refused")
    payload = envelope.get("payload")
    _require(isinstance(payload, str) and 0 < len(payload) <= MAX_BUNDLE_BYTES, "source DSSE payload refused")
    try:
        statement = _parse_json(base64.b64decode(payload, validate=True))
    except (ValueError, binascii.Error) as exc:
        raise ManifestError("source DSSE payload encoding refused") from exc
    _require(isinstance(statement, dict) and isinstance(statement.get("predicate"), dict), "source manifest predicate missing")
    raw = canonical_manifest_bytes(statement["predicate"])
    validate_manifest_bytes(raw, expected_commit)
    _bound_statement(statement, raw)
    _require(hashlib.sha256(raw).hexdigest() == expected_digest, "source manifest locator binding refused")
    return raw


def _verified_subject(raw: bytes, manifest_raw: bytes) -> None:
    results = _parse_json(raw)
    _require(isinstance(results, list) and len(results) == 1, "missing or ambiguous verified attestation")
    result = results[0].get("verificationResult") if isinstance(results[0], dict) else None
    _require(isinstance(result, dict), "missing cryptographic verification result")
    signature = result.get("signature")
    timestamps = result.get("verifiedTimestamps")
    _require(isinstance(signature, dict) and isinstance(signature.get("certificate"), dict)
             and bool(signature["certificate"]), "missing verified signing certificate")
    _require(isinstance(timestamps, list) and bool(timestamps)
             and all(isinstance(value, dict) and bool(value) for value in timestamps), "missing trusted signing timestamp")
    _bound_statement(result.get("statement"), manifest_raw)


def verify_manifest_attestation(manifest_path: Path, bundle_path: Path, expected_commit: str, *,
                                gh_executable: Path, deadline: float) -> dict[str, Any]:
    """Authenticate snapshots using a protected, preinstalled gh; fail closed.

    Callers protect the executable and all artifact directory ancestors. This
    function never accepts a caller-created 'verified JSON' receipt. Sigstore
    validates certificate validity at the trusted signing timestamp, including
    expired/unbound certificates; certificate expiry today is not artifact age.
    """
    deadline = _deadline(deadline)
    _require(gh_executable.is_absolute(), "trusted gh must have an absolute path")
    try:
        executable = gh_executable.lstat()
        _require(stat.S_ISREG(executable.st_mode) and executable.st_mode & 0o022 == 0
                 and executable.st_mode & 0o111 != 0, "trusted gh executable unsafe")
    except OSError as exc:
        raise ManifestError("trusted keyless verifier unavailable") from exc
    raw, identity = _read_regular(manifest_path, MAX_MANIFEST_BYTES)
    bundle, bundle_identity = _read_regular(bundle_path, MAX_BUNDLE_BYTES)
    manifest = validate_manifest_bytes(raw, expected_commit)
    _parse_json(bundle)
    with tempfile.TemporaryDirectory(prefix="blueprint-source-proof-") as scratch:
        root = Path(scratch)
        snapshot = root / "source-sha256-manifest.json"
        proof = root / "source-provenance.sigstore.json"
        for path, data in ((snapshot, raw), (proof, bundle)):
            with path.open("xb") as handle:
                os.fchmod(handle.fileno(), 0o400)
                handle.write(data)
        result = _run_bounded(attestation_command(gh_executable, snapshot, proof, expected_commit),
                              deadline=deadline, stdout_cap=MAX_VERIFY_OUTPUT_BYTES)
        _verified_subject(result, raw)
        _require(_read_regular(snapshot, MAX_MANIFEST_BYTES)[0] == raw
                 and _read_regular(proof, MAX_BUNDLE_BYTES)[0] == bundle, "verification snapshot changed")
    _require(_read_regular(manifest_path, MAX_MANIFEST_BYTES) == (raw, identity)
             and _read_regular(bundle_path, MAX_BUNDLE_BYTES) == (bundle, bundle_identity), "source evidence changed during verification")
    _require(time.monotonic() < deadline, "source verification deadline expired")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    builder = subparsers.add_parser("build")
    builder.add_argument("--pipeline-root", type=Path, required=True)
    builder.add_argument("--contracts-root", type=Path, required=True)
    builder.add_argument("--source-commit", required=True)
    builder.add_argument("--output", type=Path, required=True)
    verifier = subparsers.add_parser("verify")
    verifier.add_argument("--manifest", type=Path, required=True)
    verifier.add_argument("--bundle", type=Path, required=True)
    verifier.add_argument("--source-commit", required=True)
    verifier.add_argument("--gh-executable", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "build":
            raw = build_manifest(args.pipeline_root, args.contracts_root, args.source_commit)
            with args.output.open("xb") as handle:
                handle.write(raw)
            print(hashlib.sha256(raw).hexdigest())
        else:
            verify_manifest_attestation(args.manifest, args.bundle, args.source_commit,
                                        gh_executable=args.gh_executable, deadline=time.monotonic() + MAX_SECONDS)
            print("Authenticated source SHA256 manifest")
        return 0
    except (ManifestError, OSError, UnicodeError, ValueError) as exc:
        parser.exit(1, f"Source manifest refused: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
