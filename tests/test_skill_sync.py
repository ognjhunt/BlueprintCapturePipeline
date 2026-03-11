from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.agent_runtime.skill_sync import sync_skill_pack


def _write_skill(repo_root: Path, pack_name: str, name: str, body: str) -> None:
    skill_dir = repo_root / "skillpacks" / pack_name / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")


def _write_manifest(repo_root: Path, pack_name: str, skills: list[str]) -> None:
    manifest_path = repo_root / "skillpacks" / pack_name / "skillpack_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "name": pack_name,
                "source_root": f"skillpacks/{pack_name}/skills",
                "skills": skills,
            }
        ),
        encoding="utf-8",
    )


def test_sync_skill_pack_copies_into_claude_and_agents_layouts(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _write_manifest(repo_root, "industrial_readiness", ["skill_a", "skill_b"])
    _write_skill(repo_root, "industrial_readiness", "skill_a", "# Skill A\n")
    _write_skill(repo_root, "industrial_readiness", "skill_b", "# Skill B\n")

    result = sync_skill_pack(repo_root)

    assert result["skill_count"] == 2
    assert result["skillpacks"] == ["industrial_readiness"]
    assert (repo_root / ".claude/skills/skill_a/SKILL.md").read_text(encoding="utf-8") == "# Skill A\n"
    assert (repo_root / ".agents/skills/skill_b/SKILL.md").read_text(encoding="utf-8") == "# Skill B\n"


def test_sync_skill_pack_copies_multiple_skillpacks(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _write_manifest(repo_root, "blueprint_operating_system", ["blueprint_overview"])
    _write_manifest(repo_root, "industrial_readiness", ["skill_a"])
    _write_skill(repo_root, "industrial_readiness", "skill_a", "# Skill A\n")
    _write_skill(repo_root, "blueprint_operating_system", "blueprint_overview", "# Blueprint Overview\n")

    result = sync_skill_pack(repo_root)

    assert result["skill_count"] == 2
    assert result["skillpacks"] == ["blueprint_operating_system", "industrial_readiness"]
    assert (
        repo_root / ".claude/skills/blueprint_overview/SKILL.md"
    ).read_text(encoding="utf-8") == "# Blueprint Overview\n"
    assert (repo_root / ".agents/skills/skill_a/SKILL.md").read_text(encoding="utf-8") == "# Skill A\n"
