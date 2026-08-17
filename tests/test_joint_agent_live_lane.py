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

from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    TaskEvaluationLaunchError,
    validate_launch_profile,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.same_goal_spend_reconciliation import (
    materialize_same_goal_spend_reconciliation,
)

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
    result = tmp_path / "prior-launch" / "allocator" / "result.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps(
            {
                "schema_version": "adp_joint_agent_vast_run.v1",
                "status": "blocked",
                "launch_id": "adp-joint-agent-840920-task-a-r1",
                "estimated_cost_usd": 0.140048,
                "continuing_spend_from_this_run": False,
                "bundle_sha256": "sha256:" + "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    allocation = {
        "program_id": "arm-decision-proof-v1",
        "probe_kind": "adp-usd-joint-agent",
        "orchestrator_source_commit": COMMIT,
        "expected_source_commit": COMMIT,
        "bundle_sha256": "sha256:" + "b" * 64,
    }
    admission = result.with_name("admission.json")
    admission.write_text(
        json.dumps(
            {
                "schema_version": "paid_lane_admission.v1",
                "status": "admitted",
                "resource_class": "vast_provider_adapter",
                "blockers": [],
                "provider_mutations_performed": 0,
                "program_id": "arm-decision-proof-v1",
                "probe_kind": "adp-usd-joint-agent",
                "authority": "user_authorized_bounded_joint_agent_gpu_compute",
                "allocation_binding": allocation,
                "allocation_binding_digest": canonical_digest(allocation),
                "control_plane_identity": {
                    "orchestrator_source_commit": COMMIT,
                },
            }
        ),
        encoding="utf-8",
    )
    teardown = tmp_path / "prior-launch" / "teardown.json"
    teardown.write_text(
        json.dumps(
            {
                "schema_version": "vast_teardown_manifest.v1",
                "status": "completed",
                "vast_instance_ids": [47958762],
                "continuing_spend_from_this_run": False,
            }
        ),
        encoding="utf-8",
    )
    zero = tmp_path / "prior-launch" / "provider-zero.json"
    zero_value = {
        "schema_version": "task_evaluation_post_teardown_provider_zero.v1",
        "status": "provider_zero_confirmed",
        "provider_zero_verified": True,
        "continuing_spend_from_this_run": False,
    }
    zero_value["provider_zero_receipt_digest"] = canonical_digest(
        zero_value, digest_field="provider_zero_receipt_digest"
    )
    zero.write_text(json.dumps(zero_value), encoding="utf-8")
    billing = tmp_path / "prior-launch" / "response-004-vast.json"
    billing.write_text(
        json.dumps({"results": [{"source": "instance-47958762", "amount": 0.169}]}),
        encoding="utf-8",
    )
    billing_source = tmp_path / "prior-launch" / "billing-source.json"
    billing_source_value = {
        "schema_version": "blueprint.provider_billing_source_receipt.v1",
        "status": "reconciled",
        "sources": [
            {
                "provider": "vast",
                "retained_path": str(billing.resolve()),
                "response_digest": "sha256:"
                + hashlib.sha256(billing.read_bytes()).hexdigest(),
                "response_size_bytes": billing.stat().st_size,
            }
        ],
    }
    billing_source_value["receipt_digest"] = canonical_digest(
        billing_source_value, digest_field="receipt_digest"
    )
    billing_source.write_text(json.dumps(billing_source_value), encoding="utf-8")
    reconciliation = tmp_path / "prior-launch" / "same-goal-spend.json"
    materialize_same_goal_spend_reconciliation(
        lane="joint_agent",
        terminal_result_paths=[result],
        teardown_manifest_paths=[teardown],
        provider_zero_paths=[zero],
        official_billing_response_paths=[billing],
        provider_billing_source_receipt_paths=[billing_source],
        output_path=reconciliation,
    )
    return {
        "receipt": receipt,
        "bundle": bundle,
        "prior_result": result,
        "prior_reconciliation": reconciliation,
        "billing": billing,
        "zero": zero,
    }


def _build(lane, **overrides):
    return builder.build_joint_agent_live_profile(
        bundle_receipt_path=lane["receipt"],
        source_commit=overrides.pop("source_commit", COMMIT),
        raw_manifest_uri=URI,
        attempt_ordinal=overrides.pop("attempt_ordinal", 2),
        prior_result_paths=overrides.pop("prior_result_paths", (lane["prior_result"],)),
        prior_spend_reconciliation_path=overrides.pop(
            "prior_spend_reconciliation_path", lane["prior_reconciliation"]
        ),
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
    lineage = profile["same_goal_spend_lineage"]
    assert lineage["attempt_ordinal"] == 2
    assert lineage["prior_actual_provider_spend_usd"] == 0.169
    assert lineage["prior_terminal_attempts"][0]["provider_instance_id"] == 47958762
    input_names = {item["name"] for item in profile["immutable_inputs"]}
    assert {
        "joint_agent_prior_spend_reconciliation",
        "joint_agent_prior_spend_0_terminal_result",
        "joint_agent_prior_spend_0_teardown_manifest",
        "joint_agent_prior_spend_0_provider_zero",
        "joint_agent_prior_spend_0_official_billing_response",
        "joint_agent_prior_spend_0_provider_billing_source_receipt",
        "joint_agent_prior_spend_0_admission",
    }.issubset(input_names)


def test_first_attempt_requires_an_explicit_ordinal_and_no_prior_evidence(lane) -> None:
    profile = _build(
        lane,
        attempt_ordinal=1,
        prior_result_paths=(),
        prior_spend_reconciliation_path=None,
    )
    assert profile["same_goal_spend_lineage"] == {
        "schema_version": "joint_agent_same_goal_spend_lineage.v1",
        "attempt_ordinal": 1,
        "prior_terminal_attempts": [],
        "prior_spend_reconciliation": None,
        "prior_actual_provider_spend_usd": 0.0,
    }


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("omit", "joint_agent_prior_spend_lineage_missing"),
        ("branch", "joint_agent_prior_spend_lineage_invalid"),
        ("estimate", "joint_agent_prior_spend_lineage_invalid"),
    ],
)
def test_dispatcher_refuses_a_retry_profile_without_exact_linear_official_lineage(
    lane, mutation: str, expected: str
) -> None:
    profile = _build(lane)
    if mutation == "omit":
        profile.pop("same_goal_spend_lineage")
    elif mutation == "branch":
        lineage = profile["same_goal_spend_lineage"]
        lineage["prior_terminal_attempts"].append(
            dict(lineage["prior_terminal_attempts"][0])
        )
    else:
        profile["same_goal_spend_lineage"]["prior_actual_provider_spend_usd"] = (
            profile["same_goal_spend_lineage"]["prior_terminal_attempts"][0][
                "estimated_cost_usd"
            ]
        )
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    assert expected in validate_launch_profile(profile)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("omit", "prior_spend_reconciliation_required"),
        ("wrong_ordinal", "ordinal_mismatch"),
        ("wrong_instance", "same_goal_spend_source_unbound"),
        ("billing", "same_goal_spend_source_unbound"),
        ("zero", "same_goal_spend_source_unbound"),
    ],
)
def test_retry_profile_refuses_missing_or_unofficial_predecessor_evidence(
    lane, mutation: str, match: str
) -> None:
    if mutation == "omit":
        with pytest.raises(TaskEvaluationLaunchError, match=match):
            _build(lane, prior_spend_reconciliation_path=None)
        return
    if mutation == "wrong_ordinal":
        with pytest.raises(TaskEvaluationLaunchError, match=match):
            _build(lane, attempt_ordinal=3)
        return
    if mutation == "wrong_instance":
        billing = json.loads(lane["billing"].read_text(encoding="utf-8"))
        billing["results"][0]["source"] = "instance-99999999"
        lane["billing"].write_text(json.dumps(billing), encoding="utf-8")
    elif mutation == "billing":
        billing = json.loads(lane["billing"].read_text(encoding="utf-8"))
        billing["results"][0]["amount"] = 0.0
        lane["billing"].write_text(json.dumps(billing), encoding="utf-8")
    else:
        zero = json.loads(lane["zero"].read_text(encoding="utf-8"))
        zero["provider_zero_verified"] = False
        zero["provider_zero_receipt_digest"] = canonical_digest(
            zero, digest_field="provider_zero_receipt_digest"
        )
        lane["zero"].write_text(json.dumps(zero), encoding="utf-8")

    with pytest.raises(TaskEvaluationLaunchError, match=match):
        _build(lane)


def test_the_controls_are_the_shared_ones(lane) -> None:
    profile = _build(lane)

    assert profile["required_controls"]["provider_zero_required"] is True
    assert profile["required_controls"]["teardown_required"] is True
    assert profile["allocator"]["retry_cap"] == 0
    assert sorted(profile["terminal_contract"]["required_path_fields"]) == [
        "artifact_manifest_path",
        "teardown_manifest_path",
    ]


def _install_dual_task_authority_limits(
    lane: dict, *, spend_cap: float = 3.0, ttl_cap: int = 7_200
) -> None:
    receipt = json.loads(lane["receipt"].read_text(encoding="utf-8"))
    receipt.update(
        {
            "dual_task_admission_digest": "sha256:" + "d" * 64,
            "execution_authority_schema_version": (
                "joint_agent_topology_execution_authority.v1"
            ),
            "execution_authority_limits": {
                "hard_total_spend_cap_usd": spend_cap,
                "maximum_single_resource_ttl_seconds": ttl_cap,
                "model_backend": "openai",
            },
        }
    )
    lane["receipt"].write_text(json.dumps(receipt), encoding="utf-8")


def test_dual_task_profile_cannot_exceed_topology_authority_limits(lane) -> None:
    _install_dual_task_authority_limits(lane)
    assert _build(lane)["allocator"]["max_spend_usd"] == 3.0

    _install_dual_task_authority_limits(lane, spend_cap=2.99)
    with pytest.raises(TaskEvaluationLaunchError, match="authority_spend_exceeded"):
        _build(lane)

    _install_dual_task_authority_limits(lane, ttl_cap=7_199)
    with pytest.raises(TaskEvaluationLaunchError, match="authority_ttl_exceeded"):
        _build(lane)


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
