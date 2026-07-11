from __future__ import annotations

from blueprint_pipeline.g1_kitchen_bundle_compatibility import (
    REQUIRED_SCHEMAS,
    build_bundle_compatibility,
    build_source_tree_identity,
    validate_bundle_compatibility,
    main,
)


def test_current_bundle_compatibility_passes_and_old_bundle_fails() -> None:
    current = build_bundle_compatibility()
    assert validate_bundle_compatibility(current)["status"] == "passed"
    old = {**current, "required_schemas": {**REQUIRED_SCHEMAS, "closure": "old.v0"}}
    result = validate_bundle_compatibility(old)
    assert result["status"] == "blocked"
    assert "g1_bundle_schema_incompatible:closure" in result["blockers"]


def test_source_tree_identity_binds_commit_and_dirty_patch() -> None:
    identity = build_source_tree_identity(".")
    assert len(identity["source_commit"]) >= 7
    assert len(identity["source_dirty_patch_sha256"]) == 64
    assert identity["identity_includes_staged_unstaged_and_untracked"] is True


def test_compatibility_cli_blocks_pre_contract_bundle(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "bundle_manifest.json"
    manifest.write_text('{"compatibility":{"schema_version":"old.v0"}}')
    monkeypatch.setattr("sys.argv", ["compat", "--manifest", str(manifest)])
    assert main() == 2
