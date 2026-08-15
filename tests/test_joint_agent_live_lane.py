"""The Joint Agent lane earns a narrow claim, and the profile has to keep it.

Five paid Joint Agent attempts have already returned nulls; the deterministic
source-partition CAD path is what actually built the articulated twin. So this
bundle says what it is -- optional construction enrichment, not SimReady
authority, blocking neither deterministic construction nor native simulator
qualification -- and a profile that launched a bundle which had stopped saying
so would be publishing a wider claim than the lane earns.

It also takes no attempt authority: its allocator branch does not ask for one,
so the standing authorization and the bundle's own execution authority are what
bound the spend.
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
COMMIT = "9" * 40
URI = f"https://raw.githubusercontent.com/example/repo/{COMMIT}/joint.json"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = _load("build_joint_agent_live_profile")


@pytest.fixture()
def lane(tmp_path: Path) -> dict:
    bundle = tmp_path / "joint_agent_bundle.zip"
    bundle.write_bytes(b"joint-agent-bundle")
    receipt = tmp_path / "adp_joint_agent_bundle_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "ready",
                "provider_bundle_kind": "adp_joint_agent",
                "freeze_digest": "sha256:" + "b" * 64,
                "completion_retries": 0,
                "automatic_paid_retry_allowed": False,
                "agent_output_is_simready_authority": False,
                "provider_zero_required_after_return": True,
                "bundle_path": str(bundle),
                "bundle_sha256": "sha256:"
                + hashlib.sha256(bundle.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return {"receipt": receipt, "bundle": bundle}


def _build(lane, **overrides):
    return builder.build_joint_agent_live_profile(
        bundle_receipt_path=lane["receipt"],
        source_commit=overrides.pop("source_commit", COMMIT),
        raw_manifest_uri=URI,
        **overrides,
    )


def test_builds_a_profile_that_needs_no_attempt_authority(lane) -> None:
    """This lane's allocator branch does not ask for one."""

    argv = _build(lane)["allocator"]["argv"]

    assert not any("attempt-authority" in item for item in argv)
    assert "--adp-joint-agent-bundle-receipt" in argv
    assert argv[argv.index("--probe-kind") + 1] == "adp-usd-joint-agent"


def test_the_controls_are_the_shared_ones(lane) -> None:
    profile = _build(lane)

    assert profile["required_controls"]["provider_zero_required"] is True
    assert profile["required_controls"]["teardown_required"] is True
    assert profile["allocator"]["retry_cap"] == 0
    assert sorted(profile["terminal_contract"]["required_path_fields"]) == [
        "artifact_manifest_path",
        "teardown_manifest_path",
    ]


@pytest.mark.parametrize(
    "field,value,expected",
    [
        (
            "agent_output_is_simready_authority",
            True,
            "bundle_claims_simready_authority",
        ),
        (
            "provider_zero_required_after_return",
            False,
            "bundle_does_not_require_provider_zero",
        ),
        (
            "automatic_paid_retry_allowed",
            True,
            "bundle_permits_automatic_paid_retry",
        ),
        ("completion_retries", 2, "bundle_completion_retries_not_zero"),
    ],
    ids=["simready-claim", "no-provider-zero", "auto-retry", "retries"],
)
def test_a_bundle_that_widened_its_claim_cannot_be_launched(
    lane, field: str, value, expected: str
) -> None:
    receipt = json.loads(lane["receipt"].read_text(encoding="utf-8"))
    receipt[field] = value
    lane["receipt"].write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(TaskEvaluationLaunchError) as excinfo:
        _build(lane)

    assert expected in str(excinfo.value)


@pytest.mark.parametrize(
    "ttl", [3600, 20_000], ids=["under-band", "over-band"]
)
def test_a_ttl_outside_the_allocator_band_is_refused_here(lane, ttl: int) -> None:
    with pytest.raises(TaskEvaluationLaunchError) as excinfo:
        _build(lane, hard_ttl_seconds=ttl)

    assert "hard_ttl_out_of_band" in str(excinfo.value)


def test_the_bundle_is_bound_where_it_resolved(lane) -> None:
    inputs = {row["name"]: row for row in _build(lane)["immutable_inputs"]}

    assert inputs["joint_agent_bundle"]["path"] == str(lane["bundle"])
    assert inputs["joint_agent_bundle"]["digest"] == "sha256:" + hashlib.sha256(
        lane["bundle"].read_bytes()
    ).hexdigest()
