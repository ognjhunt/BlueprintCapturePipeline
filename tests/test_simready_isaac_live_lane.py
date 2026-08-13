"""The first lane reached through the shared profile skeleton.

ADP-009 names "one exact SimReady USD" in as many words; the bundle has read
`status: ready` for some time and the allocator has always had a branch for the
probe. Nothing could reach it because no launch profile existed, and a launch
profile is the one thing that carries a lane across the website boundary.

These pin the two documents that boundary needs, and the refusals that have to
happen before a provider is handed over rather than after.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_launch_dispatcher import TaskEvaluationLaunchError

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMIT = "b" * 40
URI = f"https://raw.githubusercontent.com/example/repo/{COMMIT}/simready.json"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = _load("build_simready_isaac_live_profile")
issuer = _load("issue_simready_isaac_paid_attempt_authority")


@pytest.fixture()
def lane(tmp_path: Path) -> dict:
    """A ready SimReady bundle and the authority issued against it."""

    from blueprint_pipeline.public_scene_simready_isaac_vast import DEFAULT_IMAGE

    bundle = tmp_path / "simready_bundle.zip"
    bundle.write_bytes(b"simready-isaac-bundle")
    receipt = tmp_path / "adp009b_simready_isaac_bundle_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "ready",
                "source_commit_sha": COMMIT,
                "container_image": DEFAULT_IMAGE,
                "retry_cap": 0,
                "blockers": [],
                "probe_spec_sha256": "sha256:" + "c" * 64,
                "bundle_path": str(bundle),
                "bundle_sha256": "sha256:"
                + hashlib.sha256(bundle.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    authority = issuer.issue_simready_isaac_paid_attempt_authority(
        bundle_receipt_path=receipt,
        authorized_by="nijelhunt_1",
        authority_reference="test",
        max_hourly_rate_usd=1.0,
        hard_cap_usd=3.0,
        hard_ttl_seconds=7200,
        authorized_on="2026-08-13",
    )
    authority_path = tmp_path / "attempt_authority.json"
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    return {"receipt": receipt, "authority": authority_path, "bundle": bundle}


def _build(lane, **overrides):
    return builder.build_simready_isaac_live_profile(
        bundle_receipt_path=lane["receipt"],
        attempt_authority_path=lane["authority"],
        source_commit=overrides.pop("source_commit", COMMIT),
        raw_manifest_uri=URI,
        **overrides,
    )


def test_the_issuer_derives_every_bound_value_from_the_receipt(lane) -> None:
    """A value typed by hand is a value that can disagree at the paid boundary."""

    authority = json.loads(lane["authority"].read_text(encoding="utf-8"))
    receipt = json.loads(lane["receipt"].read_text(encoding="utf-8"))

    assert authority["bundle_sha256"] == receipt["bundle_sha256"]
    assert authority["probe_spec_sha256"] == receipt["probe_spec_sha256"]
    assert authority["bundle_receipt_sha256"] == "sha256:" + hashlib.sha256(
        lane["receipt"].read_bytes()
    ).hexdigest()
    assert authority["maximum_paid_attempts"] == 1
    assert authority["automatic_paid_retry_authorized"] is False
    assert authority["zero_retry"] is True
    # No pre-existing instance is admitted by this authority.
    assert authority["active_instance_allowlist"] == []


def test_the_issuer_states_the_claim_boundary_in_the_authority(lane) -> None:
    """An import probe that passes must not later read as a physical result."""

    authority = json.loads(lane["authority"].read_text(encoding="utf-8"))

    assert authority["native_simulator_import_probe_only"] is True
    assert authority["physical_success_established"] is False
    assert authority["candidate_policy_queried"] is False


def test_the_issuer_refuses_a_bundle_this_host_cannot_resolve(tmp_path: Path) -> None:
    """Authorizing spend against unresolvable bytes authorizes nothing exact."""

    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps({"status": "ready", "bundle_path": "/nowhere/bundle.zip"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        issuer.issue_simready_isaac_paid_attempt_authority(
            bundle_receipt_path=receipt,
            authorized_by="nijelhunt_1",
            authority_reference="test",
            max_hourly_rate_usd=1.0,
            hard_cap_usd=3.0,
            hard_ttl_seconds=7200,
        )

    assert "not_host_resident" in str(excinfo.value)


@pytest.mark.parametrize(
    "field", ["authorized_by", "authority_reference"], ids=["who", "what"]
)
def test_the_issuer_will_not_invent_the_authorization(lane, field: str) -> None:
    """The one thing that cannot be derived is who approved the spend."""

    kwargs = {
        "bundle_receipt_path": lane["receipt"],
        "authorized_by": "nijelhunt_1",
        "authority_reference": "test",
        "max_hourly_rate_usd": 1.0,
        "hard_cap_usd": 3.0,
        "hard_ttl_seconds": 7200,
    }
    kwargs[field] = "   "

    with pytest.raises(ValueError):
        issuer.issue_simready_isaac_paid_attempt_authority(**kwargs)


def test_builds_a_profile_that_routes_through_the_canonical_allocator(lane) -> None:
    profile = _build(lane)

    allocator = profile["allocator"]
    assert allocator["entrypoint"].endswith("paid_resource_allocator gpu-canary")
    assert "--probe-kind" in allocator["argv"]
    assert allocator["argv"][allocator["argv"].index("--probe-kind") + 1] == (
        "adp009b-exact-simready-isaac"
    )
    assert allocator["retry_cap"] == 0
    assert profile["required_controls"]["teardown_required"] is True
    assert profile["required_controls"]["provider_zero_required"] is True
    assert sorted(profile["terminal_contract"]["required_path_fields"]) == [
        "artifact_manifest_path",
        "teardown_manifest_path",
    ]


def test_the_profile_carries_the_attempt_authority_argument(lane) -> None:
    """It only matters under ``--execute``, which is where omitting it hurts."""

    argv = _build(lane)["allocator"]["argv"]

    assert "--adp-simready-isaac-attempt-authority" in argv
    assert argv[argv.index("--adp-simready-isaac-attempt-authority") + 1] == str(
        lane["authority"]
    )


def test_the_profile_binds_the_bundle_where_it_actually_resolved(lane) -> None:
    """Not where the receipt says it was built."""

    inputs = {row["name"]: row for row in _build(lane)["immutable_inputs"]}

    assert inputs["simready_isaac_bundle"]["path"] == str(lane["bundle"])
    assert inputs["simready_isaac_bundle"]["digest"] == "sha256:" + hashlib.sha256(
        lane["bundle"].read_bytes()
    ).hexdigest()


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"hard_ttl_seconds": 900}, "hard_ttl_out_of_band"),
        ({"hard_ttl_seconds": 20_000}, "hard_ttl_out_of_band"),
        ({"source_commit": "d" * 40}, "bundle_commit_not_source_commit"),
        ({"max_spend_usd": 9.0}, "attempt_authority_spend_cap_mismatch"),
        ({"max_hourly_rate_usd": 0.5}, "attempt_authority_hourly_rate_mismatch"),
    ],
    ids=["ttl-under", "ttl-over", "wrong-commit", "cap-disagrees", "rate-disagrees"],
)
def test_refusals_happen_before_a_provider_is_handed_over(
    lane, overrides: dict, expected: str
) -> None:
    with pytest.raises(TaskEvaluationLaunchError) as excinfo:
        _build(lane, **overrides)

    assert expected in str(excinfo.value)


def test_an_unready_bundle_cannot_be_launched(lane, tmp_path: Path) -> None:
    receipt = json.loads(lane["receipt"].read_text(encoding="utf-8"))
    receipt["status"] = "blocked"
    lane["receipt"].write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(TaskEvaluationLaunchError) as excinfo:
        _build(lane)

    assert "bundle_receipt_not_ready:blocked" in str(excinfo.value)


def test_revision_yields_a_distinct_profile_id(lane) -> None:
    """Published profiles are immutable; a rebuild needs its own id."""

    assert _build(lane)["profile_id"] != _build(lane, revision="r2")["profile_id"]
    assert _build(lane, revision="r2")["profile_id"].endswith("-r2")
