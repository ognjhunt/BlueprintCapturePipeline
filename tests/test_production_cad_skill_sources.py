from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline.production_cad_skill_sources import (
    ProductionCadSkillSourcesError,
    provision_production_cad_skill_sources,
    validate_production_cad_skill_sources,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source(tmp_path: Path, source_id: str) -> tuple[Path, dict]:
    root = tmp_path / "upstream" / source_id
    root.mkdir(parents=True)
    _git(root, "init")
    _git(root, "config", "user.email", "fixture@example.com")
    _git(root, "config", "user.name", "Fixture")
    (root / "LICENSE").write_text("MIT fixture\n", encoding="utf-8")
    skills = ("cad",) if source_id == "text-to-cad" else ("multi-agent-cad",)
    if source_id == "text-to-cad":
        skill = root / "skills" / "cad" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text("# CAD\n", encoding="utf-8")
    else:
        for relative in (
            "multi_agent_cad/WORKFLOW.md",
            "multi_agent_cad/graph.py",
            "environment.yml",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "fixture")
    license_digest = "sha256:" + hashlib.sha256(
        (root / "LICENSE").read_bytes()
    ).hexdigest()
    return root, {
        "id": source_id,
        "repository": str(root),
        "commit": _git(root, "rev-parse", "HEAD"),
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "license": "MIT",
        "license_sha256": license_digest,
        "skills": skills,
    }


def test_deploy_provisions_exact_pinned_cad_skill_sources_once(tmp_path: Path) -> None:
    _text, text_spec = _source(tmp_path, "text-to-cad")
    _multi, multi_spec = _source(tmp_path, "multi-agent-cad")
    root = tmp_path / "production-sources"
    first = provision_production_cad_skill_sources(
        root, specs=(text_spec, multi_spec)
    )
    second = provision_production_cad_skill_sources(
        root, specs=(text_spec, multi_spec)
    )
    assert first == second
    assert first["skill_count"] == 2
    assert validate_production_cad_skill_sources(
        root, specs=(text_spec, multi_spec)
    )["receipt_digest"] == first["receipt_digest"]


def test_deploy_refuses_existing_drifted_cad_source(tmp_path: Path) -> None:
    _text, text_spec = _source(tmp_path, "text-to-cad")
    _multi, multi_spec = _source(tmp_path, "multi-agent-cad")
    root = tmp_path / "production-sources"
    provision_production_cad_skill_sources(root, specs=(text_spec, multi_spec))
    checkout = root / f"text-to-cad-{text_spec['commit'][:8]}"
    for path in checkout.rglob("*"):
        path.chmod(path.stat().st_mode | 0o200)
    (checkout / "LICENSE").write_text("changed\n", encoding="utf-8")
    with pytest.raises(
        ProductionCadSkillSourcesError,
        match="production_cad_skill_source_invalid:text-to-cad",
    ):
        provision_production_cad_skill_sources(
            root, specs=(text_spec, multi_spec)
        )
