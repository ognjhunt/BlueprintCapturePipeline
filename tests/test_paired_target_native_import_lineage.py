from __future__ import annotations

import json
from pathlib import Path

import pytest

import blueprint_pipeline.paired_target_native_import_vast as vast
from tests.test_paired_target_native_import_recovery import (
    _fixture as recovery_fixture,
    _materialize as materialize_recovery,
)
from tests.test_paired_target_native_import_vast import _bundle


def _artifixer_terminal(path: Path) -> dict:
    return {
        "aggregate_goal_spend_after_attempt_usd": 2.318914,
        "aggregate_goal_spend_cap_usd": 12.0,
        "authority_digest": "sha256:" + "d" * 64,
        "attempt_cost_usd": 1.0,
        "lineage_cost_usd": 1.318914,
        "records": {
            role: vast._record(path)
            for role in (
                "authority",
                "terminal_result",
                "object_store_cleanup",
                "provider_zero",
            )
        },
    }


def _prepared_bundle(path: Path) -> tuple[Path, dict]:
    path.mkdir()
    return _bundle(path)


def _issue(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prior_paths: tuple[Path, ...],
    output_name: str,
    reference: str = "successor",
    receipt_and_bundle: tuple[Path, dict] | None = None,
) -> dict:
    if receipt_and_bundle is None:
        bundle_root = tmp_path / output_name
        bundle_root.mkdir()
        receipt_path, bundle = _bundle(bundle_root)
    else:
        receipt_path, bundle = receipt_and_bundle
    anchor = tmp_path / "artifixer-anchor.json"
    anchor.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        vast,
        "validate_artifixer3d_terminal_spend_chain",
        lambda **_kwargs: _artifixer_terminal(anchor),
    )
    return vast.materialize_paired_target_native_import_paid_attempt_authority(
        bundle_receipt_path=receipt_path,
        prior_artifixer_authority_path=anchor,
        prior_artifixer_result_path=anchor,
        prior_artifixer_cleanup_path=anchor,
        prior_artifixer_provider_zero_path=anchor,
        prior_paired_attempt_provider_zero_paths=prior_paths,
        authorization_reference=reference,
        authorized_by="user",
        authorized_on="2026-08-17",
        blueprint_commit=bundle["implementation_commit"],
        max_hourly_rate_usd=0.5,
        hard_cap_usd=1.0,
        hard_ttl_seconds=3600,
        output_path=tmp_path / f"{output_name}.json",
    )


def test_successor_authority_requires_consumed_prior_and_uses_official_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BLUEPRINT_SPEND_AUTHORITY_ROOT", str(tmp_path / "spend"))
    shared_bundle = _prepared_bundle(tmp_path / "shared-bundle")
    recovery_paths = recovery_fixture(
        tmp_path / "recovery",
        source_request_digest=shared_bundle[1]["source_request_digest"],
    )
    prior_authority = json.loads(
        recovery_paths["authority"].read_text(encoding="utf-8")
    )
    assert vast.consume_paired_target_native_import_authority_once(
        prior_authority, blueprint_commit="d6506694"
    )["status"] == "consumed"
    zero_path = tmp_path / "prior-zero.json"
    prior_zero = materialize_recovery(recovery_paths, zero_path)
    assert prior_zero["official_cost_usd"] == 0.007
    assert prior_zero["attempt_cost_estimate_usd"] == 0.058517

    authority = _issue(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        prior_paths=(zero_path,),
        output_name="r2",
        receipt_and_bundle=shared_bundle,
    )
    assert authority["paired_attempt_ordinal"] == 2
    assert authority["aggregate_goal_spend_before_attempt_usd"] == 2.325914
    assert authority["excluded_vast_machine_ids"] == [140718]
    assert authority["prior_paired_attempts"][0]["official_cost_usd"] == 0.007


def test_successor_rejects_omitted_duplicate_and_branched_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BLUEPRINT_SPEND_AUTHORITY_ROOT", str(tmp_path / "spend"))
    shared_bundle = _prepared_bundle(tmp_path / "shared-bundle")
    recovery_paths = recovery_fixture(
        tmp_path / "recovery",
        source_request_digest=shared_bundle[1]["source_request_digest"],
    )
    prior_authority = json.loads(
        recovery_paths["authority"].read_text(encoding="utf-8")
    )
    vast.consume_paired_target_native_import_authority_once(
        prior_authority, blueprint_commit="d6506694"
    )
    zero_path = tmp_path / "prior-zero.json"
    materialize_recovery(recovery_paths, zero_path)

    with pytest.raises(ValueError, match="prior_attempt_lineage_required"):
        _issue(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            prior_paths=(),
            output_name="missing",
            receipt_and_bundle=shared_bundle,
        )
    with pytest.raises(ValueError):
        _issue(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            prior_paths=(zero_path, zero_path),
            output_name="duplicate",
            receipt_and_bundle=shared_bundle,
        )
    _issue(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        prior_paths=(zero_path,),
        output_name="first-successor",
        reference="first",
        receipt_and_bundle=shared_bundle,
    )
    with pytest.raises(ValueError, match="successor_already_claimed"):
        _issue(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            prior_paths=(zero_path,),
            output_name="branch",
            reference="different-branch",
            receipt_and_bundle=shared_bundle,
        )


def test_lineage_rejects_replacing_official_cost_with_adapter_estimate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BLUEPRINT_SPEND_AUTHORITY_ROOT", str(tmp_path / "spend"))
    shared_bundle = _prepared_bundle(tmp_path / "shared-bundle")
    recovery_paths = recovery_fixture(
        tmp_path / "recovery",
        source_request_digest=shared_bundle[1]["source_request_digest"],
    )
    prior_authority = json.loads(
        recovery_paths["authority"].read_text(encoding="utf-8")
    )
    vast.consume_paired_target_native_import_authority_once(
        prior_authority, blueprint_commit="d6506694"
    )
    zero_path = tmp_path / "prior-zero.json"
    zero = materialize_recovery(recovery_paths, zero_path)
    zero_path.chmod(0o600)
    zero["official_cost_usd"] = zero["attempt_cost_estimate_usd"]
    zero["receipt_digest"] = vast.canonical_digest(
        zero, digest_field="receipt_digest"
    )
    zero_path.write_text(json.dumps(zero), encoding="utf-8")
    with pytest.raises(ValueError):
        _issue(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            prior_paths=(zero_path,),
            output_name="estimated",
            receipt_and_bundle=shared_bundle,
        )


def test_consumed_authority_from_an_unrelated_request_does_not_join_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BLUEPRINT_SPEND_AUTHORITY_ROOT", str(tmp_path / "spend"))
    unrelated = {
        "authorization_digest": "sha256:" + "a" * 64,
        "bundle_sha256": "sha256:" + "b" * 64,
        "source_request_digest": "sha256:" + "c" * 64,
    }
    assert vast.consume_paired_target_native_import_authority_once(
        unrelated, blueprint_commit="unrelated"
    )["status"] == "consumed"
    current_bundle = _prepared_bundle(tmp_path / "current-bundle")
    authority = _issue(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        prior_paths=(),
        output_name="current",
        receipt_and_bundle=current_bundle,
    )
    assert authority["paired_attempt_ordinal"] == 1
    assert authority["prior_paired_attempts"] == []


def test_unscoped_legacy_consumption_cannot_be_silently_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "spend" / "consumed"
    root.mkdir(mode=0o700, parents=True)
    monkeypatch.setenv("BLUEPRINT_SPEND_AUTHORITY_ROOT", str(tmp_path / "spend"))
    digest = "sha256:" + "e" * 64
    legacy = {
        "schema_version": "paired_target_native_import_authority_consumption.v1",
        "authorization_digest": digest,
        "bundle_sha256": "sha256:" + "f" * 64,
        "blueprint_commit": "legacy",
        "consumed_at": "2026-08-17T00:00:00+00:00",
        "maximum_provider_allocations": 1,
    }
    (root / f"paired-target-native-import-{digest[7:]}.json").write_text(
        json.dumps(legacy), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unscoped_consumption_requires_recovery"):
        _issue(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            prior_paths=(),
            output_name="blocked",
        )
