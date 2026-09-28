"""Current protected owner consent is inspectable, without target/action authority."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_owner_consents.py
import hashlib
import json

import pytest

from blueprint_pipeline import control_plane_lane_owner_consents as c
from tests.test_owner_census_issuance import issuer as _issuer_fixture

issuer = _issuer_fixture


def issue(issuer):
    paths, store, args, _ = issuer
    value = c.issue_owner_consent(paths[0], paths[1], **args)
    return value, store, args, paths


def report(value, args, **changes):
    kw = dict(
        expected_sha256=value["expected_sha256"],
        expected_size_bytes=value["expected_size_bytes"],
        installed_config_path=args["installed_config_path"],
        now=1001,
        monotonic=lambda: 0,
    )
    kw.update(changes)
    return c.report_owner_consent(value["consent_id"], **kw)


def test_report_observes_owner_and_positive_metadata_without_any_action(issuer):
    value, store, args, _ = issue(issuer)
    before = {p.name: p.read_bytes() for p in store.iterdir()}
    result = report(value, args)
    assert result["status"] == "owner_consent_observed"
    assert result["principal"] == "operator" and result["mutations"] == 0
    assert result["execution_authorized"] is result["target_generation_bound"] is False
    assert (
        result["general_reference_inventory_complete"] is result["consumer_fence_checked"] is False
    )
    assert (
        result["candidate_bytes"]
        is result["estimated_reclaimable_bytes"]
        is result["eta_seconds"]
        is None
    )
    assert result["references_clear"] is False and result["eta_contribution_bytes"] is None
    assert result["decisions"][0]["decision"]["owner"] == "owner"
    assert "target_generation_unbound" in result["decisions"][0]["unmet_requirements"]
    assert {p.name: p.read_bytes() for p in store.iterdir()} == before


@pytest.mark.parametrize(
    "change,code",
    [
        ({"now": 1050}, "owner_consent_record_expired"),
        ({"expected_sha256": "sha256:" + "0" * 64}, "owner_consent_record_changed"),
        ({"expected_size_bytes": 1}, "owner_consent_record_changed"),
    ],
)
def test_exact_record_identity_and_expiry_refusals(issuer, change, code):
    value, _, args, _ = issue(issuer)
    with pytest.raises(c.OwnerCensusConsentError, match=code):
        report(value, args, **change)


def test_even_semantically_equal_policy_byte_replacement_invalidates_consent(issuer):
    value, _, args, paths = issue(issuer)
    paths[2].write_bytes(paths[2].read_bytes() + b"\n")
    with pytest.raises(c.OwnerCensusConsentError, match="owner_consent_policy_changed"):
        report(value, args)


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d.update(execution_authorized=True),
        lambda d: d.update(unexpected="private"),
        lambda d: d["decisions"][0]["census_row"].update(unexpected="private"),
        lambda d: d["decisions"][0]["decision"].update(owner="owner-guess"),
        lambda d: d.update(issuer_uid=1000),
        lambda d: d.update(selected_count=True),
    ],
)
def test_resealed_contradictions_and_unknown_fields_refuse(issuer, edit):
    value, store, args, _ = issue(issuer)
    p = store / (value["consent_id"] + ".json")
    data = json.loads(p.read_bytes())
    edit(data)
    data["consent_digest"] = c.canonical_digest(data, digest_field="consent_digest")
    raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    p.write_bytes(raw)
    value["expected_sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    value["expected_size_bytes"] = len(raw)
    with pytest.raises(c.OwnerCensusConsentError):
        report(value, args)


def test_report_constructs_strict_cumulative_budget_before_config(issuer, monkeypatch):
    value, _, args, _ = issue(issuer)
    previous = c._installed_config
    budgets = []

    def acquire(files, path):
        budgets.append(files.budget)
        assert files.budget.limits["values"] == 10000 and files.budget.deadline is not None
        return previous(files, path)

    monkeypatch.setattr(c, "_installed_config", acquire)
    report(value, args)
    assert len(budgets) == 1 and budgets[0].closed


def public(issuer, monkeypatch):
    value, store, args, paths = issue(issuer)
    import stat

    original_security = c._security

    def fixture_public_ancestors(info):
        # This root fixture models installed 0755 public ancestry without changing
        # pytest's private temporary ancestors. Regular file modes remain real.
        result = original_security(info)
        if stat.S_ISDIR(info.st_mode):
            result = result[:2] + (stat.S_IFDIR | 0o755,) + result[3:]
        return result

    monkeypatch.setattr(c, "_security", fixture_public_ancestors)
    cfg = c._installed_config(None, None)
    cfg.spool_root = str(store.parent / "requests")
    directory = store.parent / "requests" / "results"
    directory.mkdir(parents=True, mode=0o755)
    directory.chmod(0o755)
    request_id = "20260928T000000Z-owner-census-decision-deadbeef"
    kwargs = dict(
        expected_sha256=value["expected_sha256"],
        expected_size_bytes=value["expected_size_bytes"],
        installed_config_path=args["installed_config_path"],
        now=1001,
        monotonic=lambda: 0,
        publication=(str(directory), request_id),
    )
    return value, directory, request_id, kwargs, paths


def test_public_report_and_summary_are_readable_exact_bounded_artifacts(issuer, monkeypatch):
    value, directory, request_id, kwargs, _ = public(issuer, monkeypatch)
    summary = c._run_report(value["consent_id"], **kwargs)
    report_path = directory / (request_id + ".owner-census.json")
    summary_path = directory / (request_id + ".outcome.json")
    for p in (report_path, summary_path):
        assert p.stat().st_mode & 0o777 == 0o644
    raw = report_path.read_bytes()
    assert summary["result_sha256"] == "sha256:" + hashlib.sha256(raw).hexdigest()
    assert summary["result_size_bytes"] == len(raw)
    assert summary_path.stat().st_size <= 8192 and report_path.stat().st_size <= 524288
    assert summary["references_clear"] is False and summary["eta_contribution_bytes"] is None
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "deploy/operator-door"))
    from operator_door.config import DoorConfig
    from operator_door.fsview import FileView

    view = FileView(DoorConfig(read_roots=(str(directory),), hidden_paths=()))
    assert view.read_range(str(report_path))[0] == raw
    assert json.loads(raw)["principal_source"] == "protected_root_consent"
    assert json.loads(raw)["requestor_context_verified"] is False


def test_publication_replaced_parent_never_redirects_or_deletes_foreign_temp(issuer, monkeypatch):
    import os

    value, directory, request_id, kwargs, _ = public(issuer, monkeypatch)
    old = os.fsync
    foreign = []

    def swap(fd):
        old(fd)
        if not foreign:
            original = next(directory.glob(".consent-*.tmp"))
            previous = directory.with_name("old-results")
            directory.rename(previous)
            directory.mkdir(mode=0o755)
            p = directory / original.name
            p.write_bytes(b"foreign")
            foreign.append(p)

    monkeypatch.setattr(os, "fsync", swap)
    with pytest.raises(c.OwnerCensusConsentError, match="owner_consent_record_changed"):
        c._run_report(value["consent_id"], **kwargs)
    assert foreign[0].read_bytes() == b"foreign" and not list(directory.glob("*.json"))
    assert not list(directory.with_name("old-results").glob(".consent-*.tmp"))


def test_successful_publication_does_not_unlink_recreated_former_temp(issuer, monkeypatch):
    import os

    value, directory, request_id, kwargs, _ = public(issuer, monkeypatch)
    old = os.replace
    foreign = []

    def replace(source, target, **kw):
        old(source, target, **kw)
        os.write(
            fd := os.open(
                source, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=kw["src_dir_fd"]
            ),
            b"foreign",
        )
        os.close(fd)
        foreign.append(directory / source)

    monkeypatch.setattr(os, "replace", replace)
    c._run_report(value["consent_id"], **kwargs)
    assert all(p.read_bytes() == b"foreign" for p in foreign)


def test_report_cli_serializes_success_under_same_open_budget(issuer, monkeypatch, capsys):
    import time

    value, directory, request_id, kwargs, _ = public(issuer, monkeypatch)
    previous = c._installed_config
    budgets = []
    dump = json.dumps

    def acquire(files, path):
        budgets.append(files.budget)
        return previous(files, path)

    def encode(*a, **kw):
        if budgets:
            assert not budgets[-1].closed, "encoded success after invocation closed"
        return dump(*a, **kw)

    monkeypatch.setattr(c, "_installed_config", acquire)
    monkeypatch.setattr(json, "dumps", encode)
    monkeypatch.setattr(time, "time", lambda: 1001)
    assert (
        c.main(
            [
                "report",
                "--consent-id",
                value["consent_id"],
                "--expected-sha256",
                value["expected_sha256"],
                "--expected-size-bytes",
                str(value["expected_size_bytes"]),
                "--door-config",
                str(kwargs["installed_config_path"]),
                "--results-dir",
                str(directory),
                "--request-id",
                request_id,
            ]
        )
        == 0
    )
    assert (
        budgets[-1].closed
        and json.loads(capsys.readouterr().out)["status"] == "owner_consent_observed"
    )


def test_publication_refuses_inaccessible_intermediate_ancestor_before_output(issuer, monkeypatch):
    import stat

    value, directory, _, kwargs, _ = public(issuer, monkeypatch)
    inaccessible = directory.parent.stat().st_ino
    original_security = c._security

    def restricted(info):
        result = original_security(info)
        if info.st_ino == inaccessible:
            result = result[:2] + (stat.S_IFDIR | 0o700,) + result[3:]
        return result

    monkeypatch.setattr(c, "_security", restricted)
    with pytest.raises(c.OwnerCensusConsentError, match="owner_consent_publication_failed"):
        c._run_report(value["consent_id"], **kwargs)
    assert not list(directory.iterdir())


def test_publication_mode_drift_cannot_claim_readable_success(issuer, monkeypatch):
    import os

    value, directory, _, kwargs, _ = public(issuer, monkeypatch)
    original = os.replace

    def private_mode(source, target, **kw):
        original(source, target, **kw)
        os.chmod(target, 0o600, dir_fd=kw["dst_dir_fd"])

    monkeypatch.setattr(os, "replace", private_mode)
    with pytest.raises(c.OwnerCensusConsentError, match="owner_consent_publication_failed"):
        c._run_report(value["consent_id"], **kwargs)
    assert not list(directory.glob("*.outcome.json"))
