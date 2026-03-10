from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.agent_runtime.skill_sync import sync_skill_pack


def _write_skill(repo_root: Path, name: str, body: str) -> None:
    skill_dir = repo_root / "skillpacks/industrial_readiness/skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")


def test_sync_skill_pack_copies_into_claude_and_agents_layouts(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    manifest_path = repo_root / "skillpacks/industrial_readiness/skillpack_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "source_root": "skillpacks/industrial_readiness/skills",
                "skills": ["skill_a", "skill_b"],
            }
        ),
        encoding="utf-8",
    )
    _write_skill(repo_root, "skill_a", "# Skill A\n")
    _write_skill(repo_root, "skill_b", "# Skill B\n")

    result = sync_skill_pack(repo_root)

    assert result["skill_count"] == 2
    assert (repo_root / ".claude/skills/skill_a/SKILL.md").read_text(encoding="utf-8") == "# Skill A\n"
    assert (repo_root / ".agents/skills/skill_b/SKILL.md").read_text(encoding="utf-8") == "# Skill B\n"
