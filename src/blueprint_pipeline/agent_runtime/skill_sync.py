"""Sync the canonical skill packs into Claude and Codex/OpenAI layouts."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..common import PipelineError, ensure_dir


def _repo_root(explicit_root: Optional[Path] = None) -> Path:
    if explicit_root is not None:
        return explicit_root.resolve()
    return Path(__file__).resolve().parents[3]


def _manifest_paths(repo_root: Path) -> List[Path]:
    manifest_paths = sorted(repo_root.glob("skillpacks/*/skillpack_manifest.json"))
    if not manifest_paths:
        raise PipelineError(f"No skill pack manifests found under {repo_root / 'skillpacks'}")
    return manifest_paths


def load_skillpack_manifests(repo_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    root = _repo_root(repo_root)
    manifests: List[Dict[str, Any]] = []
    for manifest_path in _manifest_paths(root):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["_manifest_path"] = str(manifest_path)
        manifests.append(payload)
    return manifests


def _skill_names(manifest: Dict[str, Any]) -> List[str]:
    return [str(item) for item in manifest.get("skills", []) if str(item).strip()]


def _validate_unique_skills(manifests: Sequence[Dict[str, Any]]) -> None:
    owners: Dict[str, str] = {}
    for manifest in manifests:
        manifest_name = str(manifest.get("name") or Path(str(manifest["_manifest_path"])).parent.name)
        for skill_name in _skill_names(manifest):
            existing_owner = owners.get(skill_name)
            if existing_owner is not None:
                raise PipelineError(
                    f"Duplicate skill '{skill_name}' declared in '{existing_owner}' and '{manifest_name}'"
                )
            owners[skill_name] = manifest_name


def sync_skill_pack(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = _repo_root(repo_root)
    manifests = load_skillpack_manifests(root)
    _validate_unique_skills(manifests)

    targets = [
        root / ".claude" / "skills",
        root / ".agents" / "skills",
    ]
    copied: List[str] = []
    skillpacks: List[str] = []
    all_skill_names: List[str] = []
    for target_root in targets:
        ensure_dir(target_root)
        for manifest in manifests:
            manifest_name = str(manifest.get("name") or Path(str(manifest["_manifest_path"])).parent.name)
            source_root = root / str(manifest.get("source_root") or "")
            skill_names = _skill_names(manifest)
            if not source_root.is_dir():
                raise PipelineError(f"Skill source root is missing: {source_root}")
            if manifest_name not in skillpacks:
                skillpacks.append(manifest_name)
            for skill_name in skill_names:
                source_dir = source_root / skill_name
                if not source_dir.is_dir():
                    raise PipelineError(f"Skill source is missing: {source_dir}")
                target_dir = target_root / skill_name
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.copytree(source_dir, target_dir)
                copied.append(str(target_dir))
                if skill_name not in all_skill_names:
                    all_skill_names.append(skill_name)

    return {
        "schema_version": "v1",
        "skill_count": len(all_skill_names),
        "skillpacks": skillpacks,
        "skills": all_skill_names,
        "targets": [str(path) for path in targets],
        "copied": copied,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sync the Blueprint skill packs")
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
