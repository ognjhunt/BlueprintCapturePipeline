from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    CANONICAL_ALLOCATOR_ENTRYPOINT,
    LAUNCH_RUN_ROOT_PLACEHOLDER,
    validate_launch_profile,
    validate_public_launch_profile_descriptor,
)
from scripts import build_adp009d_840313_controls_profile as builder
from scripts.publish_task_evaluation_launch_profiles import publish_profiles

COMMIT = "a" * 40


def _guard(
    path: Path, *, status: str = "passed", instances: list | None = None
) -> Path:
    rows = instances or []
    path.write_text(
        json.dumps(
            {"status": status, "live_instance_count": len(rows), "instances": rows}
        ),
        encoding="utf-8",
    )
    return path


def _release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Build one controls profile against stubbed source verification.

    The digest and checkout verifications belong to the dry builder and are
    covered there; this exercises the controls contract on top of them.
    """
    repo = tmp_path / "repo"
    manifests = repo / builder.MANIFEST_RELATIVE_ROOT
    manifests.mkdir(parents=True)
    (repo / builder.APPROVED_CAN_RELATIVE).parent.mkdir(parents=True, exist_ok=True)
    (repo / builder.APPROVED_CAN_RELATIVE).write_text("#usda 1.0\n", encoding="utf-8")
    for name in (
        "adp009d_840313_runtime_readiness.v1.json",
        builder.HARNESS_MANIFEST_RELATIVE.name,
        builder.SCENARIO_INSTANCE_RELATIVE.name,
    ):
        (manifests / name).write_text(json.dumps({"name": name}), encoding="utf-8")
    (manifests / builder.SOURCE_MANIFEST_NAME).write_text(
        json.dumps(
            {"bundle_id": builder.BUNDLE_ID, "bundle_digest": builder.EXPECTED_BUNDLE_DIGEST}
        ),
        encoding="utf-8",
    )
    (manifests / builder.EVALUATION_SPEC_NAME).write_text(
        json.dumps({"spec": "stub"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        builder,
        "validate_evaluation_run_spec",
        lambda value: {"status": "passed", "spec_digest": builder.EXPECTED_SPEC_DIGEST},
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / builder.SAGE_COLLISION_NAME).write_text("collision", encoding="utf-8")
    runtime_inputs = tmp_path / "runtime"
    runtime_inputs.mkdir()
    (runtime_inputs / builder.AURA_APPEARANCE_NAME).write_text("aura", encoding="utf-8")

    monkeypatch.setattr(builder, "verify_protected_main_checkout", lambda *a, **k: None)
    monkeypatch.setattr(builder, "verify_materialized_source_artifacts", lambda *a, **k: {})
    return {
        "repo": repo,
        "inputs": inputs,
        "runtime_inputs": runtime_inputs,
        "guard": _guard(tmp_path / "guard.json"),
        "out": tmp_path / "out",
    }


def test_controls_profile_is_website_triggerable_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The controls canary must be launchable through the production dispatcher
    rather than a hand-written runner, while staying inside the same spend, TTL,
    retry, and claim ceilings."""
    env = _release(tmp_path, monkeypatch)
    receipt = builder.build_controls_profile_release(
        source_commit=COMMIT,
        repo_root=env["repo"],
        production_input_root=env["inputs"],
        runtime_input_root=env["runtime_inputs"],
        provider_guard_path=env["guard"],
        output_dir=env["out"],
    )

    assert receipt["status"] == "built"
    assert receipt["provider_mutation_performed"] is False
    profile = json.loads(Path(receipt["profile_path"]).read_text(encoding="utf-8"))
    assert validate_launch_profile(profile) == []

    # Website-triggerable: the dispatcher only appends --execute when this is true.
    assert profile["execution_admission"]["live_enabled"] is True
    assert profile["execution_admission"]["blockers"] == []
    # A passing controls pair is harness evidence, never a task or policy result.
    assert profile["claim_ceiling"] == "development_only"

    allocator = profile["allocator"]
    assert allocator["entrypoint"] == CANONICAL_ALLOCATOR_ENTRYPOINT
    assert allocator["max_spend_usd"] == 6.0
    assert allocator["hard_ttl_seconds"] == 5400
    assert allocator["retry_cap"] == 0
    argv = allocator["argv"]
    # Controls only: no learned candidate may be queried by this profile.
    assert "--adp009d-controls" in argv
    assert "--adp009d-policy-candidate" not in argv
    assert "--adp009d-authorize-gated-backbone" not in argv
    # The dispatcher owns the execute flag; a profile may never carry it.
    assert "--execute" not in argv
    # Per-launch outputs must not collide across website launches.
    for flag in ("--admission-out", "--adapter-output", "--adp-job-dir"):
        assert any(
            item.startswith(LAUNCH_RUN_ROOT_PLACEHOLDER)
            for item in argv[argv.index(flag) + 1 : argv.index(flag) + 2]
        )


def test_controls_profile_requires_provider_zero_and_present_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its admission is its own, so each element of that admission must be real."""
    env = _release(tmp_path, monkeypatch)

    _guard(
        env["guard"],
        instances=[{"id": 1, "name": f"{builder.LANE_INSTANCE_PREFIX}1786"}],
    )
    with pytest.raises(builder.ProductionProfileBuildError, match="lane_instance_live"):
        builder.build_controls_profile_release(
            source_commit=COMMIT,
            repo_root=env["repo"],
            production_input_root=env["inputs"],
            runtime_input_root=env["runtime_inputs"],
            provider_guard_path=env["guard"],
            output_dir=env["out"],
        )

    _guard(env["guard"], status="blocked")
    with pytest.raises(builder.ProductionProfileBuildError, match="provider_guard_not_passed"):
        builder.build_controls_profile_release(
            source_commit=COMMIT,
            repo_root=env["repo"],
            production_input_root=env["inputs"],
            runtime_input_root=env["runtime_inputs"],
            provider_guard_path=env["guard"],
            output_dir=env["out"],
        )

    # A concurrent operator's unrelated instance must NOT block this lane.
    _guard(
        env["guard"],
        instances=[{"id": 2, "name": "blueprint-native-deformable-asset-1786"}],
    )
    receipt = builder.build_controls_profile_release(
        source_commit=COMMIT,
        repo_root=env["repo"],
        production_input_root=env["inputs"],
        runtime_input_root=env["runtime_inputs"],
        provider_guard_path=env["guard"],
        output_dir=env["out"] / "concurrent",
    )
    assert receipt["status"] == "built"

    _guard(env["guard"])
    (env["runtime_inputs"] / builder.AURA_APPEARANCE_NAME).unlink()
    with pytest.raises(builder.ProductionProfileBuildError, match="input_missing:aura_appearance"):
        builder.build_controls_profile_release(
            source_commit=COMMIT,
            repo_root=env["repo"],
            production_input_root=env["inputs"],
            runtime_input_root=env["runtime_inputs"],
            provider_guard_path=env["guard"],
            output_dir=env["out"],
        )


def test_published_controls_catalog_exposes_no_allocator_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public projection must still hide allocator arguments and secrets even
    though this profile is live-enabled."""
    env = _release(tmp_path, monkeypatch)
    receipt = builder.build_controls_profile_release(
        source_commit=COMMIT,
        repo_root=env["repo"],
        production_input_root=env["inputs"],
        runtime_input_root=env["runtime_inputs"],
        provider_guard_path=env["guard"],
        output_dir=env["out"],
    )
    publish_profiles(
        profile_paths=[receipt["profile_path"]],
        profile_dir=tmp_path / "published",
        webapp_catalog_out=tmp_path / "catalog.json",
    )
    catalog = json.loads((tmp_path / "catalog.json").read_text(encoding="utf-8"))

    assert len(catalog) == 1
    descriptor = catalog[0]
    assert validate_public_launch_profile_descriptor(descriptor) == []
    assert descriptor["execution_admission"]["live_enabled"] is True
    assert descriptor["required_authorization"]["max_spend_usd"] == 6.0
    serialized = json.dumps(catalog)
    assert "allocator" not in descriptor
    assert "--adp009d-controls" not in serialized
    assert "provider-launch-request" not in serialized
