"""The Gaussian ownership audit's budget is three fixed points, not a band.

The allocator refuses any hourly rate above 0.60, any spend cap that is not
exactly 1.50, and any TTL that is not exactly 3600 -- and it refuses them at the
paid boundary, after an attempt authority has been consumed. Stating them in the
profile builder moves every one of those refusals to authoring time.

Attempts here are also ordinal: a second paid attempt must name the sealed
receipt of the first. That is a deliberate anti-retry design and the profile has
to carry it through.
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
COMMIT = "e" * 40
URI = f"https://raw.githubusercontent.com/example/repo/{COMMIT}/excision.json"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = _load("build_gaussian_excision_live_profile")


@pytest.fixture()
def lane(tmp_path: Path) -> dict:
    bundle = tmp_path / "excision_bundle.zip"
    bundle.write_bytes(b"gaussian-excision-bundle")
    digest = "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest()
    freeze = "sha256:" + "f" * 64
    receipt = tmp_path / "adp_gaussian_excision_bundle_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "ready",
                "blueprint_commit": COMMIT,
                "freeze_digest": freeze,
                "hard_cap_usd": 1.50,
                "hard_ttl_seconds": 3600,
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
                "freeze_digest": freeze,
                "bundle_sha256": digest,
                "hard_attempt_spend_cap_usd": 1.50,
                "paid_attempt_ordinal": 1,
            }
        ),
        encoding="utf-8",
    )
    return {"receipt": receipt, "authority": authority, "bundle": bundle}


def _build(lane, **overrides):
    return builder.build_gaussian_excision_live_profile(
        bundle_receipt_path=lane["receipt"],
        attempt_authority_path=lane["authority"],
        source_commit=overrides.pop("source_commit", COMMIT),
        raw_manifest_uri=URI,
        **overrides,
    )


def test_the_profile_publishes_the_exact_budget_the_allocator_demands(lane) -> None:
    profile = _build(lane)
    argv = profile["allocator"]["argv"]

    assert profile["allocator"]["max_spend_usd"] == 1.50
    assert profile["allocator"]["hard_ttl_seconds"] == 3600
    assert argv[argv.index("--adp-max-spend-usd") + 1] == "1.5"
    assert argv[argv.index("--adp-hard-ttl-seconds") + 1] == "3600"
    assert profile["allocator"]["retry_cap"] == 0


def test_a_rate_above_the_ceiling_is_refused_before_the_paid_boundary(lane) -> None:
    with pytest.raises(TaskEvaluationLaunchError) as excinfo:
        _build(lane, max_hourly_rate_usd=0.75)

    assert "gaussian_excision_hourly_rate_out_of_band" in str(excinfo.value)


def test_an_authority_bound_to_another_freeze_is_refused(lane) -> None:
    """The freeze digest is what says these are the same frozen bytes."""

    authority = json.loads(lane["authority"].read_text(encoding="utf-8"))
    authority["freeze_digest"] = "sha256:" + "a" * 64
    lane["authority"].write_text(json.dumps(authority), encoding="utf-8")

    with pytest.raises(TaskEvaluationLaunchError) as excinfo:
        _build(lane)

    assert "attempt_authority_freeze_digest_mismatch" in str(excinfo.value)


def test_a_bundle_whose_ceiling_disagrees_with_the_allocator_is_refused(lane) -> None:
    receipt = json.loads(lane["receipt"].read_text(encoding="utf-8"))
    receipt["hard_cap_usd"] = 3.0
    lane["receipt"].write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(TaskEvaluationLaunchError) as excinfo:
        _build(lane)

    assert "bundle_hard_cap_mismatch" in str(excinfo.value)


def test_the_prior_attempt_receipt_is_carried_into_argv(lane, tmp_path: Path) -> None:
    """Ordinal attempts: the second is authorized against the first's evidence."""

    prior = tmp_path / "prior_attempt_receipt.json"
    prior.write_text(json.dumps({"schema_version": "adp_gaussian_excision_attempt_receipt.v1"}), encoding="utf-8")

    argv = _build(lane, previous_attempt_receipt_path=prior)["allocator"]["argv"]

    assert "--adp-gaussian-excision-previous-attempt-receipt" in argv
    assert argv[argv.index("--adp-gaussian-excision-previous-attempt-receipt") + 1] == str(prior)


def test_the_machine_avoidlist_is_carried_and_digest_bound(lane, tmp_path: Path) -> None:
    avoidlist = tmp_path / "machine_avoidlist.json"
    avoidlist.write_text(json.dumps({"machine_ids": [1234]}), encoding="utf-8")

    profile = _build(lane, machine_avoidlist_path=avoidlist)
    argv = profile["allocator"]["argv"]
    inputs = {row["name"]: row for row in profile["immutable_inputs"]}

    assert argv[argv.index("--adp-machine-avoidlist") + 1] == str(avoidlist)
    assert inputs["machine_avoidlist"]["digest"] == "sha256:" + hashlib.sha256(
        avoidlist.read_bytes()
    ).hexdigest()


def test_the_bundle_is_bound_where_it_resolved_not_where_it_was_built(lane) -> None:
    inputs = {row["name"]: row for row in _build(lane)["immutable_inputs"]}

    assert inputs["gaussian_excision_bundle"]["path"] == str(lane["bundle"])


def test_a_bundle_from_another_commit_cannot_be_launched(lane) -> None:
    with pytest.raises(TaskEvaluationLaunchError) as excinfo:
        _build(lane, source_commit="d" * 40)

    assert "bundle_commit_not_source_commit" in str(excinfo.value)
