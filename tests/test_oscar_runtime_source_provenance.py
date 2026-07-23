from __future__ import annotations

import json
import os
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
    monkeypatch.setattr(
        provenance,
        "OFFICIAL_OSCAR_RUNTIME_TREE_SHA256",
        provenance.source_tree_evidence(source)["tree_sha256"],
    )
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
    )
    assert result["status"] == "passed"
    assert result["checks"]["runtime_tree_sha256_verified"] is True
    assert result["claim_boundary"]["git_executable_or_metadata_required_at_runtime"] is False
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == result


def test_normalize_sealed_runtime_tree_without_git(tmp_path: Path, monkeypatch) -> None:
    source = _git_source(tmp_path)
    actual_commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(provenance, "OFFICIAL_OSCAR_SOURCE_COMMIT", actual_commit)
    monkeypatch.setattr(provenance, "source_ref_is_official", lambda value: value == actual_commit)
    monkeypatch.setattr(
        provenance,
        "OFFICIAL_OSCAR_RUNTIME_TREE_SHA256",
        provenance.source_tree_evidence(source)["tree_sha256"],
    )
    existing_seal = tmp_path / "foundation-seal.json"
    normalized_seal = tmp_path / "release-seal.json"
    provenance.seal_source_tree(
        source_root=source,
        output_path=existing_seal,
        source_commit=actual_commit,
        runtime_source_root="/opt/oscar-public",
    )
    (source / ".git").rename(tmp_path / "removed-git")

    result = provenance.normalize_sealed_source_tree(
        source_root=source,
        existing_seal_path=existing_seal,
        output_path=normalized_seal,
        runtime_source_root="/opt/OSCAR",
    )

    assert result["source_commit"] == actual_commit
    assert result["runtime_source_root"] == "/opt/OSCAR"
    assert result["runtime_tree"] == provenance.source_tree_evidence(source)
    assert json.loads(normalized_seal.read_text(encoding="utf-8")) == result


def test_normalize_rejects_forged_existing_seal(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "runtime-source"
    source.mkdir()
    (source / "inference.py").write_text("MODEL = 'oscar'\n", encoding="utf-8")
    monkeypatch.setattr(
        provenance,
        "OFFICIAL_OSCAR_RUNTIME_TREE_SHA256",
        provenance.source_tree_evidence(source)["tree_sha256"],
    )
    existing_seal = tmp_path / "foundation-seal.json"
    existing_seal.write_text(
        json.dumps(
            {
                "schema_version": provenance.SEAL_SCHEMA_VERSION,
                "status": "sealed",
                "source_url": OFFICIAL_OSCAR_SOURCE_URL,
                "source_commit": provenance.OFFICIAL_OSCAR_SOURCE_COMMIT,
                "runtime_source_root": "/opt/oscar-public",
                "runtime_tree": {
                    **provenance.source_tree_evidence(source),
                    "tree_sha256": "0" * 64,
                },
                "git_metadata_required_at_runtime": False,
            }
        ),
        encoding="utf-8",
    )

    try:
        provenance.normalize_sealed_source_tree(
            source_root=source,
            existing_seal_path=existing_seal,
            output_path=tmp_path / "release-seal.json",
            runtime_source_root="/opt/OSCAR",
        )
    except ValueError as exc:
        assert str(exc) == "oscar_existing_source_seal_mismatch"
    else:  # pragma: no cover
        raise AssertionError("forged foundation seal must fail closed")


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
    monkeypatch.setattr(
        provenance,
        "OFFICIAL_OSCAR_RUNTIME_TREE_SHA256",
        provenance.source_tree_evidence(source)["tree_sha256"],
    )
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
    )
    assert result["status"] == "blocked"
    assert result["checks"]["runtime_tree_sha256_verified"] is False
    assert result["blockers"] == ["official_oscar_runtime_provenance_mismatch"]


def test_self_consistent_forged_seal_with_unreviewed_tree_blocks(
    tmp_path: Path, monkeypatch
) -> None:
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
    )
    assert result["status"] == "blocked"
    assert result["checks"]["reviewed_runtime_tree_digest_verified"] is False


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


def test_unsealed_runtime_bytecode_cache_blocks(tmp_path: Path, monkeypatch) -> None:
    source = _git_source(tmp_path)
    actual_commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(provenance, "OFFICIAL_OSCAR_SOURCE_COMMIT", actual_commit)
    monkeypatch.setattr(provenance, "source_ref_is_official", lambda value: value == actual_commit)
    monkeypatch.setattr(
        provenance,
        "OFFICIAL_OSCAR_RUNTIME_TREE_SHA256",
        provenance.source_tree_evidence(source)["tree_sha256"],
    )
    seal_path = tmp_path / "seal.json"
    provenance.seal_source_tree(
        source_root=source,
        output_path=seal_path,
        source_commit=actual_commit,
        runtime_source_root=str(source.resolve()),
    )
    (source / ".git").rename(tmp_path / "removed-git")
    bytecode_dir = source / "__pycache__"
    bytecode_dir.mkdir()
    (bytecode_dir / "inference.cpython-310.pyc").write_bytes(b"unsealed-bytecode")
    monkeypatch.setattr(provenance, "DEFAULT_RUNTIME_SOURCE_ROOT", str(source.resolve()))

    result = provenance.verify_source_tree(
        source_root=source,
        seal_path=seal_path,
        artifact_path=tmp_path / "artifact.json",
    )

    assert result["status"] == "blocked"
    assert result["checks"]["runtime_tree_contains_no_unsealed_python_bytecode"] is False
    assert result["checks"]["runtime_tree_sha256_verified"] is False
    assert result["blockers"] == ["official_oscar_runtime_provenance_mismatch"]


def test_missing_runtime_seal_reports_actionable_diagnostics(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "runtime-source"
    source.mkdir()
    (source / "inference.py").write_text("MODEL = 'oscar'\n", encoding="utf-8")
    monkeypatch.setattr(provenance, "DEFAULT_RUNTIME_SOURCE_ROOT", str(source.resolve()))
    monkeypatch.setattr(
        provenance,
        "OFFICIAL_OSCAR_RUNTIME_TREE_SHA256",
        provenance.source_tree_evidence(source)["tree_sha256"],
    )
    missing_seal = tmp_path / "missing-seal.json"

    result = provenance.verify_source_tree(
        source_root=source,
        seal_path=missing_seal,
        artifact_path=tmp_path / "artifact.json",
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["official_oscar_runtime_provenance_mismatch"]
    assert result["diagnostics"] == {
        "configured_source_root": str(source),
        "resolved_source_root": str(source.resolve()),
        "source_root_is_directory": True,
        "source_root_was_symlink": False,
        "seal_path": str(missing_seal),
        "seal_file_exists": False,
        "seal_file_is_symlink": False,
        "seal_load_error_type": "OSError",
        "runtime_tree_scan_error": None,
        "runtime_effective_uid": os.geteuid(),
    }
