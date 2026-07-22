from __future__ import annotations

import json
import subprocess
from pathlib import Path

from blueprint_pipeline import oscar_runtime_source_provenance as provenance
from blueprint_pipeline.oscar_official_release import (
    OFFICIAL_OSCAR_SOURCE_URL,
)


def _git_source(tmp_path: Path) -> Path:
    source = tmp_path / "oscar"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "config", "user.name", "Blueprint Test"],
        check=True,
    )
    (source / "inference.py").write_text("MODEL = 'oscar'\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "inference.py"], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-q", "-m", "source"], check=True)
    subprocess.run(
        ["git", "-C", str(source), "remote", "add", "origin", OFFICIAL_OSCAR_SOURCE_URL],
        check=True,
    )
    return source


def test_seal_and_verify_runtime_tree_without_git(tmp_path: Path, monkeypatch) -> None:
    source = _git_source(tmp_path)
    actual_commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(provenance, "OFFICIAL_OSCAR_SOURCE_COMMIT", actual_commit)
    monkeypatch.setattr(provenance, "source_ref_is_official", lambda value: value == actual_commit)
    seal_path = tmp_path / "seal.json"
    artifact_path = tmp_path / "artifact.json"
    provenance.seal_source_tree(
        source_root=source,
        output_path=seal_path,
        source_commit=actual_commit,
        runtime_source_root=str(source.resolve()),
    )
    (source / ".git").rename(tmp_path / "removed-git")
    monkeypatch.setattr(provenance, "DEFAULT_RUNTIME_SOURCE_ROOT", str(source.resolve()))
    result = provenance.verify_source_tree(
        source_root=source,
        seal_path=seal_path,
        artifact_path=artifact_path,
        foundation_source_url=OFFICIAL_OSCAR_SOURCE_URL,
        foundation_source_commit=actual_commit,
    )
    assert result["status"] == "passed"
    assert result["checks"]["runtime_tree_sha256_verified"] is True
    assert result["claim_boundary"]["git_executable_or_metadata_required_at_runtime"] is False
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == result


def test_runtime_tree_tampering_blocks(tmp_path: Path, monkeypatch) -> None:
    source = _git_source(tmp_path)
    actual_commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(provenance, "OFFICIAL_OSCAR_SOURCE_COMMIT", actual_commit)
    monkeypatch.setattr(provenance, "source_ref_is_official", lambda value: value == actual_commit)
    seal_path = tmp_path / "seal.json"
    provenance.seal_source_tree(
        source_root=source,
        output_path=seal_path,
        source_commit=actual_commit,
        runtime_source_root=str(source.resolve()),
    )
    (source / "inference.py").write_text("MODEL = 'forged'\n", encoding="utf-8")
    monkeypatch.setattr(provenance, "DEFAULT_RUNTIME_SOURCE_ROOT", str(source.resolve()))
    result = provenance.verify_source_tree(
        source_root=source,
        seal_path=seal_path,
        artifact_path=tmp_path / "artifact.json",
        foundation_source_url=OFFICIAL_OSCAR_SOURCE_URL,
        foundation_source_commit=actual_commit,
    )
    assert result["status"] == "blocked"
    assert result["checks"]["runtime_tree_sha256_verified"] is False
    assert result["blockers"] == ["official_oscar_runtime_provenance_mismatch"]


def test_forged_seal_and_foundation_environment_block(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "runtime-source"
    source.mkdir()
    (source / "inference.py").write_text("MODEL = 'oscar'\n", encoding="utf-8")
    monkeypatch.setattr(provenance, "DEFAULT_RUNTIME_SOURCE_ROOT", str(source.resolve()))
    seal_path = tmp_path / "seal.json"
    seal_path.write_text(
        json.dumps(
            {
                "schema_version": provenance.SEAL_SCHEMA_VERSION,
                "status": "sealed",
                "source_url": OFFICIAL_OSCAR_SOURCE_URL,
                "source_commit": provenance.OFFICIAL_OSCAR_SOURCE_COMMIT,
                "runtime_source_root": str(source.resolve()),
                "runtime_tree": provenance.source_tree_evidence(source),
                "git_metadata_required_at_runtime": False,
            }
        ),
        encoding="utf-8",
    )
    result = provenance.verify_source_tree(
        source_root=source,
        seal_path=seal_path,
        artifact_path=tmp_path / "artifact.json",
        foundation_source_url="https://example.invalid/forged.git",
        foundation_source_commit="0" * 40,
    )
    assert result["status"] == "blocked"
    assert result["checks"]["foundation_environment_binding_verified"] is False


def test_external_source_symlink_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "runtime-source"
    source.mkdir()
    external = tmp_path / "external.py"
    external.write_text("forged = True\n", encoding="utf-8")
    (source / "inference.py").symlink_to(external)
    try:
        provenance.source_tree_evidence(source)
    except ValueError as exc:
        assert str(exc) == "oscar_runtime_source_tree_external_symlink_forbidden"
    else:  # pragma: no cover
        raise AssertionError("external source symlink must fail closed")
