"""The Joint Agent lane earns a narrow claim, and the profile has to keep it.

Five paid Joint Agent attempts have already returned nulls; the deterministic
source-partition CAD path is what actually built the articulated twin. So this
bundle says what it is -- optional construction enrichment, not SimReady
authority, blocking neither deterministic construction nor native simulator
qualification -- and a profile that launched a bundle which had stopped saying
so would be publishing a wider claim than the lane earns.

It reuses the website standing authorization as a one-use attempt authority.
The profile makes that requirement explicit so an execute-id fallback cannot
bypass atomic consumption before allocator invocation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from blueprint_pipeline.adp_joint_agent_vast import (
    DEFAULT_IMAGE as JOINT_AGENT_IMAGE,
    SOURCE_TREE as JOINT_AGENT_SOURCE_TREE,
)

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
                # The allocator reads these too. A fixture without them cannot
                # express the contract the launch is actually admitted against.
                "blueprint_source": {"commit": COMMIT, "dirty": False},
                "container_image": JOINT_AGENT_IMAGE,
                "source_tree": JOINT_AGENT_SOURCE_TREE,
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


def test_builds_a_profile_with_one_use_website_authority(lane) -> None:
    """The shared standing authority, not another lane-specific file, is consumed."""

    profile = _build(lane)
    argv = profile["allocator"]["argv"]

    assert not any("attempt-authority" in item for item in argv)
    assert "--adp-joint-agent-bundle-receipt" in argv
    assert argv[argv.index("--probe-kind") + 1] == "adp-usd-joint-agent"
    assert profile["standing_launch_authorization"] == {
        "schema_version": (
            "task_evaluation_standing_launch_authorization_requirement.v1"
        ),
        "required_for_live_execution": True,
        "maximum_launches": 1,
        "consumption_must_precede_allocator": True,
    }


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


def test_a_bundle_built_at_another_commit_is_refused_before_the_authority_burns(
    lane, tmp_path: Path
) -> None:
    """The one-launch authorization is consumed before the allocator runs.

    Production regression: the dispatcher consumes this lane's standing
    authorization and only then invokes the allocator, and `record_launch` is
    exclusive-create with no release path. The builder mirrored none of the
    allocator's binding checks, so a bundle built at a different commit than
    the deployed one -- the recurring case, since every merge moves main past
    the deployed release -- was published green, burned the authorization for
    zero provider work, and left the lane unlaunchable until a human
    hand-wrote a new authorization file on the control-plane host.
    """

    receipt = json.loads(lane["receipt"].read_text(encoding="utf-8"))
    receipt["blueprint_source"]["commit"] = "a" * 40
    lane["receipt"].write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(Exception) as excinfo:
        _build(lane)

    assert "bundle_commit_not_source_commit" in str(excinfo.value)


def test_a_dirty_source_tree_is_refused_before_the_authority_burns(
    lane, tmp_path: Path
) -> None:
    receipt = json.loads(lane["receipt"].read_text(encoding="utf-8"))
    receipt["blueprint_source"]["dirty"] = True
    lane["receipt"].write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(Exception) as excinfo:
        _build(lane)

    assert "bundle_source_tree_dirty" in str(excinfo.value)
