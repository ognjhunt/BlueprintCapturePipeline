"""Sync the canonical skill pack into Claude and Codex/OpenAI layouts."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..common import PipelineError, ensure_dir


def _repo_root(explicit_root: Optional[Path] = None) -> Path:
    if explicit_root is not None:
        return explicit_root.resolve()
    return Path(__file__).resolve().parents[3]


def _manifest_path(repo_root: Path) -> Path:
    return repo_root / "skillpacks" / "industrial_readiness" / "skillpack_manifest.json"


def load_skillpack_manifest(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    manifest_path = _manifest_path(_repo_root(repo_root))
    if not manifest_path.is_file():
        raise PipelineError(f"Missing skill pack manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def sync_skill_pack(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = _repo_root(repo_root)
    manifest = load_skillpack_manifest(root)
    source_root = root / str(manifest.get("source_root") or "")
    skill_names = [str(item) for item in manifest.get("skills", []) if str(item).strip()]
    if not source_root.is_dir():
        raise PipelineError(f"Skill source root is missing: {source_root}")

    targets = [
        root / ".claude" / "skills",
        root / ".agents" / "skills",
    ]
    copied: List[str] = []
    for target_root in targets:
        ensure_dir(target_root)
        for skill_name in skill_names:
            source_dir = source_root / skill_name
            if not source_dir.is_dir():
                raise PipelineError(f"Skill source is missing: {source_dir}")
            target_dir = target_root / skill_name
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(source_dir, target_dir)
            copied.append(str(target_dir))

    return {
        "schema_version": "v1",
        "skill_count": len(skill_names),
        "skills": skill_names,
        "targets": [str(path) for path in targets],
        "copied": copied,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sync the industrial readiness skill pack")
    parser.add_argument("--repo-root", default=None, help="Override repo root")
    args = parser.parse_args(argv)

    try:
        result = sync_skill_pack(Path(args.repo_root) if args.repo_root else None)
    except Exception as exc:
        print(f"[skill-sync] FAILED: {exc}")
        return 1

    print(
        "[skill-sync] synced "
        f"{result['skill_count']} skills into "
        + ",".join(result["targets"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
