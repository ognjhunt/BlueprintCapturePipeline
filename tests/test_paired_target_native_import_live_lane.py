"""The appearance path the program bets on has to be reachable from the website.

`paired_target_native_import_vast` arrived in #507 with a bundle, a preflight, a
render request, a Vast runner, and a closeout -- and no launch profile, which is
the one thing that carries a lane across the website boundary. Every other lane
started in exactly that state and stayed unreachable until someone noticed.

The probe itself is the gate that says whether a registered ArtiFixer3D asset
loads in a real simulator, so a mismatch caught here is one fewer consumed
attempt authority.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from blueprint_pipeline.paired_target_native_import_vast import MAX_HARD_CAP_USD
from blueprint_pipeline.task_evaluation_launch_dispatcher import TaskEvaluationLaunchError

pytestmark = pytest.mark.usefixtures(
    "_materialize_generated_manifest_publication_fixture"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMIT = "c" * 40
URI = f"https://raw.githubusercontent.com/example/repo/{COMMIT}/paired.json"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = _load("build_paired_target_native_import_live_profile")


@pytest.fixture()
def lane(tmp_path: Path) -> dict:
    bundle = tmp_path / "paired_target_bundle.zip"
    bundle.write_bytes(b"paired-target-native-import-bundle")
    digest = "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest()
    receipt = tmp_path / "paired_target_native_import_bundle_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "ready",
                "implementation_commit": COMMIT,
                "bundle_path": str(bundle),
                "bundle_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    authority = tmp_path / "attempt_authority.json"
    authority.write_text(
        json.dumps(
            {
                "hard_attempt_spend_cap_usd": MAX_HARD_CAP_USD,
                "maximum_single_resource_ttl_seconds": 7200,
                "bundle_sha256": digest,
                "excluded_vast_machine_ids": [140718],
            }
        ),
        encoding="utf-8",
    )
    return {"receipt": receipt, "authority": authority, "bundle": bundle}


def _build(lane, **overrides):
    return builder.build_paired_target_native_import_live_profile(
        bundle_receipt_path=lane["receipt"],
        attempt_authority_path=lane["authority"],
        source_commit=overrides.pop("source_commit", COMMIT),
        raw_manifest_uri=URI,
        **overrides,
    )


def test_the_profile_routes_the_paired_target_probe_through_the_allocator(lane) -> None:
    profile = _build(lane)
    argv = profile["allocator"]["argv"]

    assert argv[argv.index("--probe-kind") + 1] == "adp-paired-target-native-import"
    assert "--paired-target-native-import-bundle-receipt" in argv
    assert "--paired-target-native-import-attempt-authority" in argv
    exclusion = argv.index("--adp-excluded-vast-machine-id")
    assert argv[exclusion + 1] == "140718"
    assert profile["allocator"]["retry_cap"] == 0


def test_default_profile_budget_matches_the_allocator_authority_ceiling(lane) -> None:
    profile = _build(lane)

    assert profile["allocator"]["max_spend_usd"] == MAX_HARD_CAP_USD
    authority = json.loads(lane["authority"].read_text(encoding="utf-8"))
    assert authority["hard_attempt_spend_cap_usd"] == MAX_HARD_CAP_USD


def test_the_shared_controls_are_present(lane) -> None:
    """A zero-retry paid import gate must still prove teardown and provider zero."""

    profile = _build(lane)

    assert profile["required_controls"]["provider_zero_required"] is True
    assert profile["required_controls"]["teardown_required"] is True
    assert profile["required_controls"]["watchdog_required"] is True
    assert sorted(profile["terminal_contract"]["required_path_fields"]) == [
        "artifact_manifest_path",
        "teardown_manifest_path",
    ]


@pytest.mark.parametrize("ttl", [900, 9000], ids=["under-band", "over-band"])
def test_a_ttl_outside_the_allocator_band_is_refused_here(lane, ttl: int) -> None:
    with pytest.raises(TaskEvaluationLaunchError) as excinfo:
        _build(lane, hard_ttl_seconds=ttl)

    assert "hard_ttl_out_of_band" in str(excinfo.value)


def test_a_bundle_from_another_commit_is_refused_before_the_authority_is_spent(
    lane,
) -> None:
    """The allocator would refuse this after consuming a single-use authority."""

    with pytest.raises(TaskEvaluationLaunchError) as excinfo:
        _build(lane, source_commit="d" * 40)

    assert "bundle_commit_not_source_commit" in str(excinfo.value)


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("hard_attempt_spend_cap_usd", 9.0, "attempt_authority_spend_cap_mismatch"),
        ("maximum_single_resource_ttl_seconds", 1800, "attempt_authority_ttl_mismatch"),
        ("bundle_sha256", "sha256:" + "0" * 64, "attempt_authority_bundle_mismatch"),
    ],
    ids=["cap", "ttl", "bundle"],
)
def test_an_authority_that_disagrees_with_the_profile_is_refused(
    lane, field: str, value, expected: str
) -> None:
    authority = json.loads(lane["authority"].read_text(encoding="utf-8"))
    authority[field] = value
    lane["authority"].write_text(json.dumps(authority), encoding="utf-8")

    with pytest.raises(TaskEvaluationLaunchError) as excinfo:
        _build(lane)

    assert expected in str(excinfo.value)


def test_the_bundle_is_bound_where_it_resolved(lane) -> None:
    inputs = {row["name"]: row for row in _build(lane)["immutable_inputs"]}

    assert inputs["paired_target_native_import_bundle"]["path"] == str(lane["bundle"])
    assert inputs["paired_target_native_import_bundle"]["digest"] == "sha256:" + (
        hashlib.sha256(lane["bundle"].read_bytes()).hexdigest()
    )
