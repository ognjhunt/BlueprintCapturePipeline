from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import build_adp009d_840313_live_profile as builder


def test_live_builder_routes_the_exact_runtime_through_the_canonical_allocator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "a" * 40
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(builder, "verify_protected_main_checkout", lambda *_args: None)
    monkeypatch.setattr(
        builder,
        "verify_materialized_source_artifacts",
        lambda *_args: [
            {"name": "appearance_3dgs", "path": str(tmp_path / "raw.ply"), "digest": "sha256:" + "1" * 64}
        ],
    )
    runtime_paths: dict[str, Path] = {}
    runtime_rows = []
    for role in (
        "approved_simready_can",
        "static_collision_geometry",
        "franka_evaluation_harness",
        "aura_nurec_appearance",
        "canonical_scenario_instance",
    ):
        path = tmp_path / role
        path.write_text(role, encoding="utf-8")
        runtime_paths[role] = path
        runtime_rows.append({"name": role, "path": str(path), "digest": "sha256:" + "2" * 64})
    monkeypatch.setattr(builder, "verify_materialized_runtime_inputs", lambda *_args, **_kwargs: runtime_rows)
    monkeypatch.setattr(builder, "_read_evidence", lambda _path: {})
    readiness = {
        "schema_version": "task_evaluation_runtime_readiness.v1",
        "status": "passed",
        "profile_id": builder.READINESS_PROFILE_ID,
        "blockers": [],
        "receipt_digest": "sha256:" + "3" * 64,
    }
    monkeypatch.setattr(builder, "build_live_readiness", lambda **_kwargs: readiness)
    monkeypatch.setattr(builder, "verify_profile_immutable_inputs", lambda _profile: [])
    monkeypatch.setattr(builder, "validate_launch_profile", lambda _profile: [])
    (tmp_path / "raw.ply").write_text("raw", encoding="utf-8")

    receipt = builder.build_live_profile_release(
        source_commit=commit,
        repo_root=repo,
        source_input_root=tmp_path / "source",
        runtime_input_root=tmp_path / "runtime",
        provider_guard_path=tmp_path / "guard.json",
        release_evidence_path=tmp_path / "release.json",
        control_bundle_receipt_path=tmp_path / "bundle.json",
        allocator_result_path=tmp_path / "allocator.json",
        control_pair_path=tmp_path / "pair.json",
        artifact_manifest_path=tmp_path / "manifest.json",
        teardown_manifest_path=tmp_path / "teardown.json",
        provider_zero_guard_path=tmp_path / "zero.json",
        readiness_uri="s3://immutable-evidence/readiness.json",
        output_dir=tmp_path / "output",
    )

    profile = json.loads(Path(receipt["profile_path"]).read_text(encoding="utf-8"))
    argv = profile["allocator"]["argv"]
    assert receipt["live_execution_enabled"] is True
    assert receipt["profile_id"] == f"{builder.LIVE_PROFILE_ID_PREFIX}-{commit}"
    assert profile["profile_id"] == receipt["profile_id"]
    assert profile["execution_admission"]["live_enabled"] is True
    assert profile["execution_admission"]["blockers"] == []
    assert profile["allocator"]["entrypoint"] == builder.CANONICAL_ALLOCATOR_ENTRYPOINT
    assert "--execute" not in argv
    assert argv[argv.index("--probe-kind") + 1] == "adp009d-franka-native-microcheck"
    assert argv[argv.index("--adp009d-aura-particlefield") + 1] == str(
        runtime_paths["aura_nurec_appearance"]
    )
    assert argv[argv.index("--adp009d-aura-particlefield-sha256") + 1] == builder.APPEARANCE_DIGEST
    assert argv[argv.index("--adp009d-policy-candidate") + 1] == "pi05_droid,groot_n17_droid"
    assert "--adp009d-controls" in argv
    assert "--adp009d-authorize-gated-backbone" in argv


def test_live_profile_identity_changes_with_allocator_source_commit() -> None:
    first = builder.live_profile_id_for_source_commit("a" * 40)
    second = builder.live_profile_id_for_source_commit("b" * 40)

    assert first == f"{builder.LIVE_PROFILE_ID_PREFIX}-{'a' * 40}"
    assert second == f"{builder.LIVE_PROFILE_ID_PREFIX}-{'b' * 40}"
    assert first != second


def test_live_builder_rejects_nonimmutable_readiness_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builder, "verify_protected_main_checkout", lambda *_args: None)
    with pytest.raises(builder.ProductionProfileBuildError, match="readiness_uri_invalid"):
        builder.build_live_profile_release(
            source_commit="a" * 40,
            repo_root=tmp_path,
            source_input_root=tmp_path,
            runtime_input_root=tmp_path,
            provider_guard_path=tmp_path / "guard",
            release_evidence_path=tmp_path / "release",
            control_bundle_receipt_path=tmp_path / "bundle",
            allocator_result_path=tmp_path / "allocator",
            control_pair_path=tmp_path / "pair",
            artifact_manifest_path=tmp_path / "manifest",
            teardown_manifest_path=tmp_path / "teardown",
            provider_zero_guard_path=tmp_path / "zero",
            readiness_uri="file:///mutable/readiness.json",
            output_dir=tmp_path / "output",
        )
