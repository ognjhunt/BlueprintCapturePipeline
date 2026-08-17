"""The head of the appearance chain, and the only lane that can go first.

Nothing downstream of ArtiFixer3D can be authorized until this lane has
completed and produced a terminal result, an object store cleanup, and a
provider-zero receipt: `paired_target_native_import_vast` validates that chain
before it will mint an attempt authority, and carries this run's spend forward
against a $12 campaign cap.

So the ordering is not a preference. Firing the import gate first is not slower,
it is impossible.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_launch_dispatcher import TaskEvaluationLaunchError

pytestmark = pytest.mark.usefixtures(
    "_materialize_generated_manifest_publication_fixture"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMIT = "7" * 40
URI = f"https://raw.githubusercontent.com/example/repo/{COMMIT}/artifixer.json"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = _load("build_artifixer3d_live_profile")


@pytest.fixture()
def lane(tmp_path: Path) -> dict:
    bundle = tmp_path / "artifixer3d_bundle.zip"
    bundle.write_bytes(b"artifixer3d-bundle")
    digest = "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest()
    receipt = tmp_path / "public_scene_artifixer3d_bundle_receipt.json"
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
                "hard_attempt_spend_cap_usd": 10.0,
                "maximum_hourly_rate_usd": 1.0,
                "maximum_single_resource_ttl_seconds": 10800,
                "bundle_sha256": digest,
                "campaign_spend_anchor_kind": "measured_campaign_start",
            }
        ),
        encoding="utf-8",
    )
    return {"receipt": receipt, "authority": authority, "bundle": bundle}


def _build(lane, **overrides):
    return builder.build_artifixer3d_live_profile(
        bundle_receipt_path=lane["receipt"],
        attempt_authority_path=lane["authority"],
        source_commit=overrides.pop("source_commit", COMMIT),
        raw_manifest_uri=URI,
        **overrides,
    )


def test_the_profile_routes_the_artifixer3d_probe_through_the_allocator(lane) -> None:
    argv = _build(lane)["allocator"]["argv"]

    assert argv[argv.index("--probe-kind") + 1] == "adp-artifixer3d-exact-support"
    assert "--adp-artifixer3d-bundle-receipt" in argv
    assert "--adp-artifixer3d-attempt-authority" in argv


def test_the_shared_controls_are_present(lane) -> None:
    profile = _build(lane)

    assert profile["required_controls"]["provider_zero_required"] is True
    assert profile["required_controls"]["teardown_required"] is True
    assert profile["allocator"]["retry_cap"] == 0
    assert sorted(profile["terminal_contract"]["required_path_fields"]) == [
        "artifact_manifest_path",
        "teardown_manifest_path",
    ]


def test_a_cap_above_the_lane_authority_is_refused(lane) -> None:
    """$10 is the ceiling, and the campaign cap above it is $12 for the chain."""

    with pytest.raises(TaskEvaluationLaunchError) as excinfo:
        _build(lane, max_spend_usd=25.0)

    assert "hard_cap_exceeds_authority" in str(excinfo.value)


@pytest.mark.parametrize("ttl", [3600, 30_000], ids=["under-band", "over-band"])
def test_a_ttl_outside_the_allocator_band_is_refused_here(lane, ttl: int) -> None:
    with pytest.raises(TaskEvaluationLaunchError) as excinfo:
        _build(lane, hard_ttl_seconds=ttl)

    assert "hard_ttl_out_of_band" in str(excinfo.value)


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("hard_attempt_spend_cap_usd", 3.0, "attempt_authority_spend_cap_mismatch"),
        ("maximum_hourly_rate_usd", 0.25, "attempt_authority_hourly_rate_mismatch"),
        ("bundle_sha256", "sha256:" + "0" * 64, "attempt_authority_bundle_mismatch"),
    ],
    ids=["cap", "rate", "bundle"],
)
def test_an_authority_that_disagrees_is_refused_before_it_is_consumed(
    lane, field: str, value, expected: str
) -> None:
    """This authority is single-use; a mismatch found later has spent it."""

    authority = json.loads(lane["authority"].read_text(encoding="utf-8"))
    authority[field] = value
    lane["authority"].write_text(json.dumps(authority), encoding="utf-8")

    with pytest.raises(TaskEvaluationLaunchError) as excinfo:
        _build(lane)

    assert expected in str(excinfo.value)


def test_retired_campaign_anchor_cannot_enter_a_new_live_profile(lane) -> None:
    authority = json.loads(lane["authority"].read_text(encoding="utf-8"))
    authority["campaign_spend_anchor_kind"] = "prior_aura_terminal_attempt"
    lane["authority"].write_text(json.dumps(authority), encoding="utf-8")

    with pytest.raises(TaskEvaluationLaunchError) as excinfo:
        _build(lane)

    assert "attempt_authority_measured_campaign_start_required" in str(excinfo.value)


def test_the_bundle_is_bound_where_it_resolved(lane) -> None:
    inputs = {row["name"]: row for row in _build(lane)["immutable_inputs"]}

    assert inputs["artifixer3d_bundle"]["path"] == str(lane["bundle"])


def test_the_import_gate_still_requires_this_lane_to_have_run() -> None:
    """Guards the ordering itself, not just this profile.

    If the downstream authority ever stopped demanding the predecessor chain,
    the campaign's spend accounting would silently lose its anchor.
    """

    from blueprint_pipeline import paired_target_native_import_vast as gate

    source = Path(gate.__file__).read_text(encoding="utf-8")
    assert "prior_terminal_artifixer" in source
    assert "validate_artifixer3d_terminal_spend_chain" in source
