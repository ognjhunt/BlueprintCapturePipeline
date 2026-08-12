"""A live profile must be derivable, not hand-assembled.

The first live profile for this probe was written by hand and each mistake was
caught by a different fail-closed gate, one paid round trip at a time: a
source-bundle manifest recorded at the authoring machine's path, a missing
attempt-authority argument that only matters under ``--execute``, and an
instance allowlist that has to match a value living in another file.

These pin the derivations so those three cannot recur.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_launch_dispatcher import TaskEvaluationLaunchError

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "build_retained_scene_render_live_profile",
    REPO_ROOT / "scripts" / "build_retained_scene_render_live_profile.py",
)
builder = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(builder)

COMMIT = "0" * 40
URI = f"https://raw.githubusercontent.com/example/repo/{COMMIT}/request.json"
ALLOWLIST = [47373597, 47569249]


def _fixture(tmp_path: Path, *, allowlist=None, cap: float = 12.0, commit: str = COMMIT):
    authority = tmp_path / "execution_authority.json"
    authority.write_text(
        json.dumps(
            {
                "schema_version": "third_scene_dual_task_execution_authority.v1",
                "paid_compute": {
                    "provider": "vast",
                    "external_instance_allowlist": (
                        ALLOWLIST if allowlist is None else allowlist
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"retained-scene-bundle")
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"schema_version": "request.v1"}), encoding="utf-8")
    attempt = tmp_path / "attempt_authority.json"
    attempt.write_text(json.dumps({"schema_version": "attempt.v1"}), encoding="utf-8")

    receipt = tmp_path / "bundle_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "ready",
                "probe_kind": "adp-retained-scene-gpu-render",
                "blueprint_commit": commit,
                "hard_total_spend_cap_usd": cap,
                "bundle_path": str(bundle),
                "bundle_sha256": "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest(),
                "execution_authority": {"path": str(authority)},
            }
        ),
        encoding="utf-8",
    )
    return {"receipt": receipt, "request": request, "attempt": attempt, "authority": authority}


def _build(paths, **overrides):
    return builder.build_retained_scene_render_live_profile(
        bundle_receipt_path=paths["receipt"],
        request_manifest_path=paths["request"],
        attempt_authority_path=paths["attempt"],
        source_commit=overrides.pop("source_commit", COMMIT),
        raw_manifest_uri=URI,
        **overrides,
    )


def test_builds_a_live_profile_that_passes_both_validators(tmp_path: Path) -> None:
    profile = _build(_fixture(tmp_path))

    assert profile["execution_admission"]["live_enabled"] is True
    assert profile["execution_admission"]["blockers"] == []
    assert profile["allocator"]["retry_cap"] == 0
    assert profile["required_controls"]["teardown_required"] is True
    assert profile["required_controls"]["provider_zero_required"] is True


def test_carries_the_attempt_authority_argument(tmp_path: Path) -> None:
    """Required only under --execute, so its absence surfaces as a paid blocker."""
    paths = _fixture(tmp_path)
    argv = _build(paths)["allocator"]["argv"]

    index = argv.index("--adp-retained-scene-render-attempt-authority")
    assert argv[index + 1] == str(paths["attempt"].resolve())


def test_allowlist_is_derived_from_the_bundle_authority(tmp_path: Path) -> None:
    """The allocator refuses any allowlist that differs from the authority's."""
    argv = _build(_fixture(tmp_path))["allocator"]["argv"]

    passed = [
        int(argv[index + 1])
        for index, item in enumerate(argv)
        if item == "--adp-allowed-active-vast-instance-id"
    ]
    assert sorted(passed) == ALLOWLIST


def test_allowlist_tracks_a_changed_authority(tmp_path: Path) -> None:
    argv = _build(_fixture(tmp_path, allowlist=[11, 22, 33]))["allocator"]["argv"]

    passed = [
        int(argv[index + 1])
        for index, item in enumerate(argv)
        if item == "--adp-allowed-active-vast-instance-id"
    ]
    assert sorted(passed) == [11, 22, 33]


def test_source_bundle_manifest_records_the_request_path(tmp_path: Path) -> None:
    """Recording an unreachable path fails on the control plane, not here."""
    paths = _fixture(tmp_path)
    inputs = {row["name"]: row for row in _build(paths)["immutable_inputs"]}

    assert inputs["source_bundle_manifest"]["path"] == str(paths["request"].resolve())
    assert set(inputs) >= {"source_bundle_manifest", "evaluation_run_spec"}


def test_spend_cap_comes_from_the_bundle_not_the_builder(tmp_path: Path) -> None:
    profile = _build(_fixture(tmp_path, cap=7.5), max_hourly_rate_usd=1.0, hard_ttl_seconds=3600)

    assert profile["allocator"]["max_spend_usd"] == 7.5


def test_refuses_a_worst_case_exceeding_the_bundle_cap(tmp_path: Path) -> None:
    with pytest.raises(TaskEvaluationLaunchError, match="worst_case_spend_exceeds_bundle_cap"):
        _build(_fixture(tmp_path, cap=1.0), max_hourly_rate_usd=2.0, hard_ttl_seconds=10_800)


def test_refuses_a_bundle_built_at_another_commit(tmp_path: Path) -> None:
    """The allocator would refuse this at the paid boundary instead."""
    with pytest.raises(TaskEvaluationLaunchError, match="bundle_commit_not_source_commit"):
        _build(_fixture(tmp_path, commit="f" * 40))


def test_refuses_a_ttl_outside_the_allocator_band(tmp_path: Path) -> None:
    with pytest.raises(TaskEvaluationLaunchError, match="hard_ttl_out_of_band"):
        _build(_fixture(tmp_path), hard_ttl_seconds=60)


def test_refuses_an_unready_bundle(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    receipt = json.loads(paths["receipt"].read_text())
    receipt["status"] = "blocked"
    paths["receipt"].write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(TaskEvaluationLaunchError, match="bundle_receipt_not_ready"):
        _build(paths)


def test_revision_yields_a_distinct_profile_id(tmp_path: Path) -> None:
    """Published profiles are immutable, so a changed profile needs a new id."""
    paths = _fixture(tmp_path)

    base = _build(paths)["profile_id"]
    revised = _build(paths, revision="a2")["profile_id"]

    assert revised == f"{base}-a2"
    assert base != revised
