#!/usr/bin/env python3
"""Audit the frozen runtime graph and emit release-bound dependency evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blueprint_pipeline.artifact_storage import default_artifact_cache_root  # noqa: E402

PIP_AUDIT_VERSION = "2.10.1"


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _repository_sha(root: Path) -> str | None:
    configured = str(os.environ.get("GITHUB_SHA") or "").strip()
    if configured:
        return configured
    completed = _run(["git", "rev-parse", "HEAD"], cwd=root)
    return completed.stdout.strip() if completed.returncode == 0 else None


def run_gate(*, root: Path = ROOT) -> dict[str, Any]:
    blockers: list[str] = []
    lock_path = root / "uv.lock"
    if not lock_path.is_file():
        blockers.append("uv_lock_missing")
    lock_check = _run(["uv", "lock", "--check"], cwd=root)
    if lock_check.returncode != 0:
        blockers.append("uv_lock_inconsistent")
    version_check = _run(
        ["uv", "run", "--frozen", "pip-audit", "--version"],
        cwd=root,
    )
    observed_pip_audit_version = version_check.stdout.strip()
    if (
        version_check.returncode != 0
        or observed_pip_audit_version != f"pip-audit {PIP_AUDIT_VERSION}"
    ):
        blockers.append("pip_audit_version_mismatch")

    audit_payload: dict[str, Any] = {"dependencies": []}
    audit_stderr = ""
    with tempfile.TemporaryDirectory(prefix="blueprint-dependency-audit-") as temp_dir:
        temp_root = Path(temp_dir)
        requirements_path = temp_root / "runtime-requirements.txt"
        audit_path = temp_root / "pip-audit.json"
        export = _run(
            [
                "uv",
                "export",
                "--frozen",
                "--no-dev",
                "--no-emit-project",
                "--no-emit-package",
                "blueprint-contracts",
                "--no-header",
                "--quiet",
                "--format",
                "requirements-txt",
                "--output-file",
                str(requirements_path),
            ],
            cwd=root,
        )
        if export.returncode != 0 or not requirements_path.is_file():
            blockers.append("runtime_dependency_export_failed")
        else:
            audit = _run(
                [
                    "uv",
                    "run",
                    "--frozen",
                    "pip-audit",
                    "-r",
                    str(requirements_path),
                    "--format",
                    "json",
                    "--output",
                    str(audit_path),
                ],
                cwd=root,
            )
            audit_stderr = audit.stderr.strip()
            if audit_path.is_file():
                try:
                    loaded = json.loads(audit_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    blockers.append("pip_audit_output_malformed")
                else:
                    if isinstance(loaded, dict):
                        audit_payload = loaded
                    else:
                        blockers.append("pip_audit_output_malformed")
            else:
                blockers.append("pip_audit_output_missing")
            if audit.returncode not in {0, 1}:
                blockers.append("pip_audit_execution_failed")

    vulnerabilities = [
        {
            "dependency": dependency.get("name"),
            "version": dependency.get("version"),
            "id": vulnerability.get("id"),
            "fix_versions": vulnerability.get("fix_versions") or [],
        }
        for dependency in audit_payload.get("dependencies", [])
        if isinstance(dependency, dict)
        for vulnerability in dependency.get("vulns", [])
        if isinstance(vulnerability, dict)
    ]
    blockers.extend(
        f"known_runtime_vulnerability:{item['dependency']}:{item['id']}"
        for item in vulnerabilities
    )
    blockers = list(dict.fromkeys(blockers))
    return {
        "schema_version": "blueprint_dependency_security_gate.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not blockers else "blocked",
        "repository_sha": _repository_sha(root),
        "uv_lock_sha256": _sha_file(lock_path) if lock_path.is_file() else None,
        "requirements_export_source": "uv.lock runtime graph excluding local project and pinned BlueprintContracts VCS source",
        "pip_audit_version": observed_pip_audit_version or None,
        "dependencies_audited": len(audit_payload.get("dependencies", [])),
        "known_vulnerability_count": len(vulnerabilities),
        "vulnerabilities": vulnerabilities,
        "blockers": blockers,
        "audit_stderr": audit_stderr or None,
        "claim_boundary": {
            "runtime_python_dependency_scan_only": True,
            "container_image_scan_included": False,
            "sbom_digest_match_proven": False,
            "absence_of_known_advisories_is_not_absence_of_vulnerabilities": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_artifact_cache_root() / "ci" / "dependency-security-gate.json",
    )
    args = parser.parse_args(argv)
    result = run_gate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "[dependency-security] "
        f"status={result['status']} vulnerabilities={result['known_vulnerability_count']} "
        f"evidence={args.output}"
    )
    for blocker in result["blockers"]:
        print(f"[dependency-security] blocker={blocker}", file=sys.stderr)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
