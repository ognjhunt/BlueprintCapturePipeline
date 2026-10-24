"""A country exception is explicit, input-bound and limited to a pinned canary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_policy_canary_session import (
    PolicyCanarySessionError,
    session_geolocation_country_codes,
    validate_session_authority,
)
from tests.test_native_task_arena_policy_canary_session import _authority


def _grant(digest, country="GB"):
    return {
        "schema_version": "policy_canary_geolocation_authority.v1",
        "country_codes": [country],
        "runtime_inputs_digest": digest,
        "authorized_by": "fixture operator",
        "authorization_reference": f"Fixture: same admitted inputs; bounded private {country} compute",
        "disclosure_scope": "same_admitted_private_vast_inputs",
    }


def _pinned(tmp_path, country="GB"):
    authority, inputs = _authority(tmp_path)
    return _authority(
        tmp_path,
        allowed_machine_ids=[49863 if country == "GB" else 40508],
        geolocation_authority=_grant(inputs["runtime_inputs_digest"], country),
    )[0]


def _reseal(authority):
    authority["authority_digest"] = canonical_digest(authority, digest_field="authority_digest")


@pytest.mark.parametrize("country", ["GB", "EE"])
def test_default_country_and_explicit_input_bound_exception(tmp_path, country):
    default, _ = _authority(tmp_path)
    assert session_geolocation_country_codes(default) == ("US",)
    authority = _pinned(tmp_path, country)
    assert session_geolocation_country_codes(validate_session_authority(authority)) == (country,)
    authority["geolocation_authority"]["authorization_reference"] += " changed"
    with pytest.raises(PolicyCanarySessionError, match="authority_digest_invalid"):
        validate_session_authority(authority)


@pytest.mark.parametrize(
    "change",
    [
        "country",
        "digest",
        "author",
        "reference",
        "author_bool",
        "reference_bool",
        "disclosure",
        "machine",
        "team",
        "null",
    ],
)
@pytest.mark.parametrize("country", ["GB", "EE"])
def test_invalid_or_widened_country_authority_is_refused(tmp_path, change, country):
    authority = _pinned(tmp_path, country)
    grant = authority["geolocation_authority"]
    if change == "country":
        grant["country_codes"] = ["US", "GB"]
    elif change == "digest":
        grant["runtime_inputs_digest"] = "sha256:" + "0" * 64
    elif change == "author":
        grant["authorized_by"] = ""
    elif change == "reference":
        grant["authorization_reference"] = ""
    elif change == "author_bool":
        grant["authorized_by"] = True
    elif change == "reference_bool":
        grant["authorization_reference"] = True
    elif change == "disclosure":
        grant["disclosure_scope"] = "new_provider_or_new_inputs"
    elif change == "machine":
        authority.pop("allowed_machine_ids")
    elif change == "team":
        authority["run_kind"] = "team_candidate_evaluation"
    else:
        authority["geolocation_authority"] = None
    _reseal(authority)
    code = "machine_constraint_invalid" if change == "team" else "geolocation_authority_invalid"
    with pytest.raises(PolicyCanarySessionError, match=code):
        validate_session_authority(authority)


@pytest.mark.parametrize("country", ["GB", "EE"])
def test_invalid_exception_cannot_consume_authority_or_reach_provider(tmp_path, monkeypatch, country):
    from blueprint_pipeline import native_task_arena_vast as vast

    authority = _pinned(tmp_path, country)
    authority["geolocation_authority"]["runtime_inputs_digest"] = "sha256:" + "0" * 64
    _reseal(authority)

    def forbidden(*a, **kw):
        raise AssertionError("invalid country authority reached paid boundary")

    monkeypatch.setattr(vast, "consume_session_authority_once", forbidden)
    monkeypatch.setattr(vast, "run_arena_native_control_vast", forbidden)
    with pytest.raises(PolicyCanarySessionError, match="geolocation_authority_invalid"):
        vast.run_native_task_arena_policy_canary_session_vast(
            job_dir=tmp_path / "job",
            prepared_bundle={},
            session_authority=authority,
            paid_resource_admission_grant=None,
            execute=True,
        )


@pytest.mark.parametrize("countries", [["US"], ["FI"], ["GB", "EE"], ["EE", "GB"], [], ["EE", "EE"]])
def test_country_grant_cannot_expand_to_other_or_multiple_locations(tmp_path, countries):
    authority = _pinned(tmp_path, "EE")
    authority["geolocation_authority"]["country_codes"] = countries
    _reseal(authority)
    with pytest.raises(PolicyCanarySessionError, match="geolocation_authority_invalid"):
        validate_session_authority(authority)


@pytest.mark.parametrize("country", [None, "GB", "EE"])
def test_canonical_transport_receives_country_without_weakening_other_bounds(
    tmp_path, monkeypatch, country
):
    from blueprint_pipeline import native_task_arena_vast as vast
    from blueprint_pipeline import native_task_arena_paired_witness_staging as witness
    from blueprint_pipeline import policy_canary_output_members as members

    authority = _pinned(tmp_path, country) if country else _authority(tmp_path)[0]
    captured = {}
    monkeypatch.setattr(vast, "validate_policy_canary_provider_bundle", lambda *a, **kw: None)
    monkeypatch.setattr(
        witness, "build_paired_witness_binding", lambda *a: {"maximum_archive_bytes": 0}
    )
    monkeypatch.setattr(
        members,
        "resolve_output_delivery",
        lambda: SimpleNamespace(
            mode="archive", record=lambda: {"mode": "archive", "source": "fixture"}
        ),
    )

    def no_allocation(**kwargs):
        captured.update(kwargs)
        return {"status": "blocked", "blockers": ["fixture_no_allocation"]}

    monkeypatch.setattr(vast, "run_arena_native_control_vast", no_allocation)
    vast.run_native_task_arena_policy_canary_session_vast(
        job_dir=tmp_path / "job",
        prepared_bundle={"container_image": "fixture-image"},
        session_authority=authority,
        paid_resource_admission_grant=None,
        execute=False,
        hard_cap_usd=authority["hard_cap_usd"],
        hard_ttl_seconds=authority["hard_ttl_seconds"],
    )
    assert captured["allowed_geolocation_country_codes"] == (country or "US",)
    assert captured["require_independent_watchdog"] is True
    assert captured["stale_offer_create_retry_limit"] == 0
    assert captured["min_gpu_ram_mb"] == 46000
    assert captured["minimum_driver_version"] == vast.MINIMUM_DRIVER_VERSION
    assert captured["expected_provider_download_bytes"] > 0
    assert captured["expected_provider_upload_bytes"] > 0
    assert captured["hard_cap_usd"] == authority["hard_cap_usd"]
    assert captured["hard_ttl_seconds"] == authority["hard_ttl_seconds"]
    if country:
        assert captured["allowed_machine_ids"] == ((49863,) if country == "GB" else (40508,))
