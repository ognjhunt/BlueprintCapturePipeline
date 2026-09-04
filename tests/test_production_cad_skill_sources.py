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


def test_provisioned_sources_are_readable_by_the_service_account(tmp_path: Path) -> None:
    """The control-plane CAD stage runs as the service account, never as root."""

    import grp
    import os
    import pwd
    import stat

    _text, text_spec = _source(tmp_path, "text-to-cad")
    _multi, multi_spec = _source(tmp_path, "multi-agent-cad")
    root = tmp_path / "production-sources"
    account = pwd.getpwuid(os.getuid()).pw_name
    result = provision_production_cad_skill_sources(
        root, specs=(text_spec, multi_spec), service_account=account
    )
    assert result["service_account_access"] == {
        "account": account,
        "status": "readable",
    }
    expected_gid = pwd.getpwnam(account).pw_gid
    for checkout in (
        root / f"text-to-cad-{text_spec['commit'][:8]}",
        root / f"multi-agent-cad-{multi_spec['commit'][:8]}",
    ):
        for path in (checkout, *checkout.rglob("*")):
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                continue
            assert metadata.st_gid == expected_gid, path
            assert metadata.st_mode & stat.S_IRGRP, path
            if stat.S_ISDIR(metadata.st_mode):
                assert metadata.st_mode & stat.S_IXGRP, path
            assert not metadata.st_mode & 0o222, path
    assert grp.getgrgid(expected_gid).gr_gid == expected_gid


def test_validation_fails_closed_when_the_service_account_cannot_traverse(tmp_path: Path) -> None:
    import os
    import pwd

    _text, text_spec = _source(tmp_path, "text-to-cad")
    _multi, multi_spec = _source(tmp_path, "multi-agent-cad")
    root = tmp_path / "production-sources"
    # Provision without a service account, then reproduce the historical drift:
    # an owner-only 0500 checkout that another account cannot enter even though
    # every byte check still passes for the owner.
    provision_production_cad_skill_sources(
        root, specs=(text_spec, multi_spec), service_account=None
    )
    checkout = root / f"text-to-cad-{text_spec['commit'][:8]}"
    checkout.chmod(0o500)
    other_account = pwd.getpwnam("nobody").pw_name
    assert pwd.getpwnam(other_account).pw_uid != os.getuid()
    try:
        with pytest.raises(
            ProductionCadSkillSourcesError,
            match="production_cad_skill_source_unreadable_by_service_account:text-to-cad",
        ):
            validate_production_cad_skill_sources(
                root, specs=(text_spec, multi_spec), service_account=other_account
            )
    finally:
        checkout.chmod(0o550)


def test_unknown_service_account_is_reported_not_guessed(tmp_path: Path) -> None:
    _text, text_spec = _source(tmp_path, "text-to-cad")
    _multi, multi_spec = _source(tmp_path, "multi-agent-cad")
    root = tmp_path / "production-sources"
    result = provision_production_cad_skill_sources(
        root, specs=(text_spec, multi_spec), service_account="no-such-account-xyz"
    )
    assert result["service_account_access"] == {
        "account": "no-such-account-xyz",
        "status": "not_applicable_no_service_account",
    }
