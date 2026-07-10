#!/usr/bin/env python3
"""Build strict container-production lane evidence from CI-generated sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


EVIDENCE_SCHEMA = "blueprint.critical_capability_lane_evidence.v1"
PAYLOAD_SCHEMA = "blueprint.container_production_contract.v1"
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
IMAGE_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
SOURCE_NAMES = {
    "production_image": "production_image_id.txt",
    "development_image": "development_image_id.txt",
    "production_user": "production_user.txt",
    "development_user": "development_user.txt",
    "compose_config": "compose-config.yml",
    "compose_sentinel": "compose-config.ok",
    "systemd_security_log": "systemd-security.log",
    "systemd_security_sentinel": "systemd-security.ok",
    "production_runtime_smoke": "production-runtime-smoke.log",
    "development_runtime_smoke": "development-runtime-smoke.log",
}
MAX_SOURCE_SIZES = {
    "production_image": 256,
    "development_image": 256,
    "production_user": 256,
    "development_user": 256,
    "compose_config": 4 * 1024 * 1024,
    "compose_sentinel": 64,
    "systemd_security_log": 8 * 1024 * 1024,
    "systemd_security_sentinel": 64,
    "production_runtime_smoke": 4 * 1024,
    "development_runtime_smoke": 4 * 1024,
}
MAX_TOTAL_SOURCE_SIZE = 16 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_container_production_evidence(*, source_dir: Path, repository_sha: str) -> dict[str, Any]:
    repository_sha = repository_sha.strip().lower()
    blockers: list[str] = []
    if SHA_PATTERN.fullmatch(repository_sha) is None:
        blockers.append("container_repository_sha_invalid")
    if source_dir.is_symlink():
        blockers.append("container_source_dir_symlink")
    expected_names = set(SOURCE_NAMES.values())
    actual_names = (
        {path.name for path in source_dir.iterdir()}
        if source_dir.is_dir() and not source_dir.is_symlink()
        else set()
    )
    if actual_names != expected_names:
        blockers.append("container_source_file_set_invalid")
    artifact_digests: dict[str, str] = {}
    artifact_sizes: dict[str, int] = {}
    paths = {label: source_dir / name for label, name in SOURCE_NAMES.items()}
    total_source_size = 0
    for label, path in paths.items():
        if path.is_symlink():
            blockers.append(f"container_source_symlink:{label}")
        elif not path.is_file():
            blockers.append(f"container_source_missing:{label}")
        else:
            try:
                size = path.stat().st_size
            except OSError:
                blockers.append(f"container_source_unreadable:{label}")
                continue
            if size <= 0:
                blockers.append(f"container_source_missing:{label}")
            elif size > MAX_SOURCE_SIZES[label]:
                blockers.append(f"container_source_oversize:{label}")
            else:
                try:
                    artifact_digests[label] = _sha256(path)
                except OSError:
                    blockers.append(f"container_source_unreadable:{label}")
                    continue
                artifact_sizes[label] = size
                total_source_size += size
    if total_source_size > MAX_TOTAL_SOURCE_SIZE:
        blockers.append("container_source_total_oversize")

    def read_text(label: str) -> str:
        path = paths[label]
        if label not in artifact_sizes:
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            return ""

    production_image_id = read_text("production_image")
    development_image_id = read_text("development_image")
    production_user = read_text("production_user")
    development_user = read_text("development_user")
    compose = read_text("compose_config")
    compose_sentinel = read_text("compose_sentinel")
    systemd = read_text("systemd_security_log")
    systemd_sentinel = read_text("systemd_security_sentinel")
    production_smoke = read_text("production_runtime_smoke")
    development_smoke = read_text("development_runtime_smoke")
    if IMAGE_ID_PATTERN.fullmatch(production_image_id) is None:
        blockers.append("container_production_image_id_invalid")
    if IMAGE_ID_PATTERN.fullmatch(development_image_id) is None:
        blockers.append("container_development_image_id_invalid")
    production_nonroot_user = production_user == "blueprint:blueprint"
    development_nonroot_user = development_user == "blueprint:blueprint"
    if not production_nonroot_user:
        blockers.append("container_production_nonroot_user_invalid")
    if not development_nonroot_user:
        blockers.append("container_development_nonroot_user_invalid")
    compose_valid = bool(compose) and compose_sentinel == "compose-config-ok"
    systemd_passed = bool(systemd) and systemd_sentinel == "systemd-security-ok"
    production_runtime_smoke_passed = production_smoke == "container-smoke-ok"
    development_runtime_smoke_passed = development_smoke == "container-smoke-ok"
    if not compose_valid:
        blockers.append("container_compose_config_not_validated")
    if not systemd_passed:
        blockers.append("container_systemd_security_not_passed")
    if not production_runtime_smoke_passed:
        blockers.append("container_production_runtime_smoke_not_passed")
    if not development_runtime_smoke_passed:
        blockers.append("container_development_runtime_smoke_not_passed")
    blockers = sorted(set(blockers))
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "lane_id": "container_production",
        "evidence_schema_version": PAYLOAD_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository_sha": repository_sha,
        "status": "passed" if not blockers else "blocked",
        "executed": len(artifact_digests) == len(paths),
        "skipped_count": 0,
        "production_image_id": production_image_id or None,
        "development_image_id": development_image_id or None,
        "production_nonroot_user": production_nonroot_user,
        "development_nonroot_user": development_nonroot_user,
        "compose_config_valid": compose_valid,
        "systemd_security_passed": systemd_passed,
        "production_runtime_smoke_passed": production_runtime_smoke_passed,
        "development_runtime_smoke_passed": development_runtime_smoke_passed,
        "artifact_digests": artifact_digests,
        "artifact_sizes": artifact_sizes,
        "blockers": blockers,
        "claim_boundary": {
            "container_build_and_local_smoke_only": True,
            "container_lane_is_not_deployment_proof": True,
            "container_lane_is_not_gpu_provider_proof": True,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--repository-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_container_production_evidence(
        source_dir=args.source_dir.expanduser().absolute(),
        repository_sha=args.repository_sha,
    )
    _write_json_atomic(args.output.expanduser().absolute(), result)
    print(f"[container-production-evidence] status={result['status']}")
    for blocker in result["blockers"]:
        print(f"[container-production-evidence] blocker={blocker}", file=sys.stderr)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
