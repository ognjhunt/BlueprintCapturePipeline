"""Capture digest-bound output from the pinned STEP-first CAD inspection skill."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

from .common import utc_now_iso


class PublicSceneCadInspectionCaptureError(ValueError):
    """The pinned CAD source or inspection result is not trustworthy."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _under(path: Path, root: Path, *, error: str) -> Path:
    resolved = path.expanduser().resolve()
    root = root.expanduser().resolve()
    if resolved != root and root not in resolved.parents:
        raise PublicSceneCadInspectionCaptureError(error)
    return resolved


def _git(skill_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(skill_root), *args], capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise PublicSceneCadInspectionCaptureError("cad_skill_git_inspection_failed")
    return result.stdout.strip()


def capture_cad_inspection(
    *,
    repo_root: str | Path,
    evidence_root: str | Path,
    cad_skill_root: str | Path,
    cad_python: str | Path,
    expected_commit: str,
    expected_tree: str,
    step_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve()
    evidence = Path(evidence_root).expanduser().resolve()
    skill = Path(cad_skill_root).expanduser().resolve()
    # Preserve a virtual-environment launcher symlink. Resolving it would invoke
    # the base interpreter and silently discard the pinned CAD environment.
    python = Path(os.path.abspath(Path(cad_python).expanduser()))
    step = _under(Path(step_path), repo, error="cad_step_outside_repo_root")
    output = _under(Path(output_path), evidence, error="cad_inspection_outside_evidence_root")
    if not step.is_file() or not python.is_file():
        raise PublicSceneCadInspectionCaptureError("cad_step_or_interpreter_missing")
    observed_commit = _git(skill, "rev-parse", "HEAD")
    observed_tree = _git(skill, "rev-parse", "HEAD^{tree}")
    if observed_commit != expected_commit or observed_tree != expected_tree:
        raise PublicSceneCadInspectionCaptureError("cad_skill_revision_mismatch")
    if _git(skill, "status", "--porcelain"):
        raise PublicSceneCadInspectionCaptureError("cad_skill_checkout_dirty")
    launcher = skill / "skills" / "cad" / "scripts" / "inspect" / "__main__.py"
    if not launcher.is_file():
        raise PublicSceneCadInspectionCaptureError("cad_inspection_launcher_missing")
    relative_step = step.relative_to(repo).as_posix()
    command = [
        str(python),
        str(launcher),
        "refs",
        relative_step,
        "--facts",
        "--planes",
        "--positioning",
        "--format",
        "json",
        "--quiet",
    ]
    result = subprocess.run(
        command, cwd=repo, capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise PublicSceneCadInspectionCaptureError("cad_inspection_execution_failed")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise PublicSceneCadInspectionCaptureError("cad_inspection_json_invalid") from exc
    if not isinstance(payload, dict) or payload.get("ok") is not True or payload.get("errors") != []:
        raise PublicSceneCadInspectionCaptureError("cad_inspection_not_passed")
    tokens = payload.get("tokens")
    if not isinstance(tokens, list) or len(tokens) != 1 or not isinstance(tokens[0], dict):
        raise PublicSceneCadInspectionCaptureError("cad_inspection_single_token_required")
    if tokens[0].get("stepHash") != _sha256(step):
        raise PublicSceneCadInspectionCaptureError("cad_inspection_step_digest_mismatch")
    payload["capture_provenance"] = {
        "generated_at": utc_now_iso(),
        "cad_skill_repository": "https://github.com/earthtojake/text-to-cad",
        "cad_skill_commit": observed_commit,
        "cad_skill_tree": observed_tree,
        "command": command,
        "step_relative_path": relative_step,
        "step_sha256": "sha256:" + _sha256(step),
        "exit_status": result.returncode,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--cad-skill-root", type=Path, required=True)
    parser.add_argument("--cad-python", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--step", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = capture_cad_inspection(
        repo_root=args.repo_root,
        evidence_root=args.evidence_root,
        cad_skill_root=args.cad_skill_root,
        cad_python=args.cad_python,
        expected_commit=args.expected_commit,
        expected_tree=args.expected_tree,
        step_path=args.step,
        output_path=args.output,
    )
    print(json.dumps({"ok": payload["ok"], "stepHash": payload["tokens"][0]["stepHash"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
