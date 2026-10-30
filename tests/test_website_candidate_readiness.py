"""Read-only candidate notes never become a campaign, plan or execution authority."""

# Covers: src/blueprint_pipeline/website_candidate_readiness.py
# Covers: src/blueprint_pipeline/website_candidate_readiness_http.py
# Covers: src/blueprint_pipeline/website_preparation_status_http.py
from copy import deepcopy
import json
import os

import asyncio
import httpx
import pytest

from blueprint_pipeline import website_candidate_readiness as preview
from blueprint_pipeline import live_pipeline_intake_service as service
from blueprint_pipeline import website_preparation_status as status
from blueprint_pipeline import team_candidate_quick10_registry as registry
from blueprint_pipeline.beta_policy_candidate_catalog import load_default_catalog, seal_catalog
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_live_pipeline_intake_service import _signed_intake_headers

pytest_plugins = ["tests.test_website_preparation_status"]


@pytest.fixture(autouse=True)
def development_sandbox_root_owner(monkeypatch):
    # Rootless test sandboxes expose the host root owner as nobody. Normalize
    # only that development fixture boundary; production policy and every
    # captured source/ledger/owner check below remain real and unchanged.
    from blueprint_pipeline import task_evaluation_scene_retirement_lifetime as lifetime

    original = lifetime._open_owned

    def opened(path, *args, **kwargs):
        fd, info = original(path, *args, **kwargs)
        if str(path) == "/" and info.st_uid not in {0, os.getuid()}:
            values = list(info)
            values[4] = 0
            info = os.stat_result(values)
        return fd, info

    monkeypatch.setattr(lifetime, "_open_owned", opened)
    # In this rootless sandbox thread-to-loop wakeups are blocked. ASGI still
    # runs the actual HMAC, selector, root resolver and source reader; dispatch
    # timing is not under test. No executor or production hook is changed.
    from blueprint_pipeline import website_candidate_readiness_http as http_route

    async def inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(http_route, "run_in_threadpool", inline)


def configure_registry(tmp_path, monkeypatch, *, defaults=None, profiles=None):
    value = {"schema_version": registry.REGISTRY_SCHEMA, "profiles": profiles or []}
    if defaults is not None:
        value["derived_profiles"] = {
            "root": str(tmp_path / "derived"),
            "defaults": {
                "allowed_founder_approvers": [],
                "max_hourly_rate_usd": 0.8,
                "open_loop_horizon": 1,
                **defaults,
            },
            "scored_matrix_salt_key_path": str(tmp_path / "must-never-read-salt"),
        }
    value["registry_digest"] = canonical_digest(value, digest_field="registry_digest")
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    monkeypatch.setenv(registry.REGISTRY_ENV, str(path))
    return path, value


def read(source):
    return preview.read_candidate_readiness(capture_root=source[0], selectors=source[3])


def test_effective_catalog_is_complete_without_scene_or_campaign_and_v4_is_unchanged(
    prepared_source, monkeypatch
):
    monkeypatch.delenv(registry.REGISTRY_ENV, raising=False)
    before = status.read_preparation_status(
        capture_root=prepared_source[0],
        selectors=prepared_source[3],
        schema_version=status.STATUS_V4,
    )
    files = {p: p.read_bytes() for p in prepared_source[0].rglob("*") if p.is_file()}
    result = read(prepared_source)
    expected = load_default_catalog()
    assert [r["candidate_id"] for r in result["candidates"]] == [
        r["candidate_id"] for r in expected["candidates"]
    ]
    assert result["candidate_count"] == len(expected["candidates"]) > 10
    assert result["catalog_digest"] == expected["catalog_digest"]
    assert result["registry_digest"] is None and result["profile_source"] == "not_configured"
    assert all(r["compatibility"] == "not_evaluated" for r in result["candidates"])
    assert result["preparation_status_digest"] == before["status_digest"]
    forbidden = {
        "plan",
        "campaign_id",
        "score",
        "scored",
        "planned",
        "budget",
        "quote",
        "reservation",
        "execution_spec",
        "launch_descriptor",
    }

    def visit(value):
        if isinstance(value, dict):
            assert not forbidden & set(value)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(result)
    assert files == {p: p.read_bytes() for p in prepared_source[0].rglob("*") if p.is_file()}
    assert (
        status.read_preparation_status(
            capture_root=prepared_source[0],
            selectors=prepared_source[3],
            schema_version=status.STATUS_V4,
        )
        == before
    )
    assert len(json.dumps(result).encode()) < preview.MAX_BYTES


def test_operator_defaults_show_each_real_registration_failure_without_scene_or_secret_read(
    prepared_source, tmp_path, monkeypatch
):
    candidate = "openwam_alpha_franka"
    path, configured = configure_registry(
        tmp_path,
        monkeypatch,
        defaults={
            "screening_candidates": {
                candidate: {"robot_policy": {"private_field": "PRIVATE_SHOULD_NOT_LEAK"}},
                "development_only_custom": {
                    "robot_policy": {"private_field": "PRIVATE_SHOULD_NOT_LEAK"}
                },
            }
        },
    )
    monkeypatch.setattr(
        registry, "_scored_salt", lambda *a, **k: pytest.fail("Preview must not read salt")
    )
    result = read(prepared_source)
    rows = {r["candidate_id"]: r for r in result["candidates"]}
    assert rows[candidate]["known_reasons"] == ["beta_screening_candidate_configuration_invalid"]
    assert "development_only_custom" in rows
    assert result["registry_digest"] == configured["registry_digest"]
    assert result["profile_source"] == "operator_defaults"
    assert "PRIVATE_SHOULD_NOT_LEAK" not in json.dumps(result)
    assert str(tmp_path) not in json.dumps(result)
    assert not (tmp_path / "derived").exists()
    old = result["profile_digest"]
    changed = deepcopy(configured)
    changed["derived_profiles"]["defaults"]["screening_candidates"].pop(candidate)
    changed["registry_digest"] = canonical_digest(changed, digest_field="registry_digest")
    path.write_text(json.dumps(changed))
    assert read(prepared_source)["profile_digest"] != old


def test_capture_profile_overrides_defaults_and_is_revision_bound(
    prepared_source, tmp_path, monkeypatch
):
    root, _, _, selected, *_ = prepared_source
    row = {
        "capture_root": str(root.resolve()),
        "scene_id": selected["scene_id"],
        "screening_candidates": {"capture_only_candidate": {"robot_policy": {"invalid": True}}},
    }
    configure_registry(
        tmp_path,
        monkeypatch,
        profiles=[row],
        defaults={
            "screening_candidates": {"defaults_only_candidate": {"robot_policy": {"invalid": True}}}
        },
    )
    result = read(prepared_source)
    assert result["profile_source"] == "capture_profile"
    ids = {r["candidate_id"] for r in result["candidates"]}
    assert "capture_only_candidate" in ids and "defaults_only_candidate" not in ids


@pytest.mark.parametrize("change", ["registry", "source", "consent"])
def test_changed_authority_during_catalog_read_cannot_publish_mixed_snapshot(
    prepared_source, tmp_path, monkeypatch, change
):
    from blueprint_pipeline import beta_policy_effective_catalog as effective
    from blueprint_pipeline.common import write_json

    path, configured = configure_registry(tmp_path, monkeypatch)
    original = effective.catalog_for_profile

    def raced(profile):
        result = original(profile)
        if change == "registry":
            configured["new_revision"] = True
            configured["registry_digest"] = canonical_digest(
                configured, digest_field="registry_digest"
            )
            path.write_text(json.dumps(configured))
        elif change == "source":
            ledger = json.loads((prepared_source[0] / "pipeline_job_ledger.json").read_text())
            ledger["revision"] += 1
            write_json(prepared_source[0] / "pipeline_job_ledger.json", ledger)
        else:
            from blueprint_pipeline import consent_takedown

            monkeypatch.setattr(
                consent_takedown, "read_consent_state", lambda *a, **k: {"state": "revoked"}
            )
        return result

    monkeypatch.setattr(effective, "catalog_for_profile", raced)
    with pytest.raises(ValueError):
        read(prepared_source)


@pytest.mark.parametrize("bound", ["rows", "bytes", "reasons"])
def test_bound_refusal_never_silently_truncates_catalog(prepared_source, monkeypatch, bound):
    monkeypatch.delenv(registry.REGISTRY_ENV, raising=False)
    if bound == "rows":
        monkeypatch.setattr(preview, "MAX_CANDIDATES", 1)
    elif bound == "bytes":
        monkeypatch.setattr(preview, "MAX_BYTES", 100)
    else:
        monkeypatch.setattr(preview, "MAX_REASONS", 0)
    with pytest.raises(preview.CandidateReadinessUnavailable, match="bound_exceeded"):
        read(prepared_source)


def test_names_and_malformed_reason_text_do_not_expose_endpoints(prepared_source, monkeypatch):
    from blueprint_pipeline import beta_policy_effective_catalog as effective

    catalog = load_default_catalog()
    row = catalog["candidates"][0]
    row["display_name"] = "https://PRIVATE_HOST/?token=PRIVATE_TOKEN"
    row["eligibility"]["blockers"] = ["exception /private/credential.key"]
    catalog = seal_catalog(catalog)
    monkeypatch.setattr(effective, "catalog_for_profile", lambda _: catalog)
    monkeypatch.delenv(registry.REGISTRY_ENV, raising=False)
    result = read(prepared_source)
    assert result["candidates"][0]["display_name"] == row["candidate_id"].replace("_", " ")
    assert result["candidates"][0]["known_reasons"] == ["candidate_configuration_needs_review"]
    assert "PRIVATE" not in json.dumps(result) and "/private/" not in json.dumps(result)


def http(source, *, client="blueprint-webapp", signed=True, extra=None):
    body = json.dumps({**source[3], **(extra or {})})
    headers = (
        _signed_intake_headers(
            "synthetic-secret" if client == "blueprint-webapp" else "other-secret",
            body,
            nonce="readiness-" + client,
            client_id=client,
        )
        if signed
        else {}
    )

    async def send():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=service.create_app()), base_url="http://testserver"
        ) as client:
            return await client.post(preview.PATH, content=body, headers=headers)

    return asyncio.run(send())


def test_real_signed_companion_transport_preserves_existing_v4(prepared_source, monkeypatch):
    monkeypatch.delenv(registry.REGISTRY_ENV, raising=False)
    response = http(prepared_source)
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    assert response.json()["schema_version"] == preview.SCHEMA
    assert response.json()["compatibility"] == "not_evaluated"
    assert response.json()["candidate_count"] > 10


@pytest.mark.parametrize("kind", ["unsigned", "other_issuer", "path", "source"])
def test_companion_rejects_wrong_issuer_path_or_source(prepared_source, kind):
    if kind == "unsigned":
        response = http(prepared_source, signed=False)
    elif kind == "other_issuer":
        response = http(prepared_source, client="other")
    elif kind == "path":
        response = http(prepared_source, extra={"capture_root": "/private"})
    else:
        response = http(prepared_source, extra={"task_context_digest": "sha256:" + "0" * 64})
    assert response.status_code in {401, 403, 422, 503}
    assert "candidates" not in response.json()


def test_retained_derived_profile_uses_original_seal_without_reading_scored_salt(
    prepared_source, tmp_path, monkeypatch
):
    path, value = configure_registry(tmp_path, monkeypatch, defaults={"screening_candidates": {}})
    root = value["derived_profiles"]["root"]
    from pathlib import Path

    Path(root).mkdir(mode=0o700)
    derived = {
        "schema_version": registry.DERIVED_SCHEMA,
        "capture_root": str(prepared_source[0]),
        "scene_id": prepared_source[3]["scene_id"],
        "task_id": "development-task",
        "scenario_id": "development",
        "robot_free_probe_root": str(tmp_path / "never-opened-probe"),
        "scene_revision_digest": "sha256:" + "c" * 64,
    }
    derived["profile_digest"] = canonical_digest(derived, digest_field="profile_digest")
    target = registry._derived_path(Path(root), str(prepared_source[0]))
    target.write_text(json.dumps(derived))
    target.chmod(0o600)
    monkeypatch.setattr(
        registry, "_scored_salt", lambda *a, **k: pytest.fail("salt must not be read")
    )
    assert read(prepared_source)["profile_source"] == "derived_profile"
    derived["task_id"] = "tampered"
    target.write_text(json.dumps(derived))
    with pytest.raises(preview.CandidateReadinessUnavailable):
        read(prepared_source)


def test_catalog_access_change_during_read_is_not_a_current_snapshot(prepared_source, monkeypatch):
    from blueprint_pipeline import beta_policy_effective_catalog as effective

    original = effective.catalog_for_profile
    calls = 0

    def changed(profile):
        nonlocal calls
        calls += 1
        value = original(profile)
        if calls == 2:
            value["revision"] += ":changed"
            value = seal_catalog(value)
        return value

    monkeypatch.delenv(registry.REGISTRY_ENV, raising=False)
    monkeypatch.setattr(effective, "catalog_for_profile", changed)
    with pytest.raises(preview.CandidateReadinessUnavailable):
        read(prepared_source)


def test_development_root_normalization_still_rejects_writable_policy_ancestry(
    prepared_source, tmp_path
):
    from blueprint_pipeline import task_evaluation_scene_retirement_lifetime as lifetime

    tmp_path.chmod(0o777)
    try:
        with pytest.raises(lifetime.SceneRetirementAccessError, match="access_unsafe"):
            lifetime._policy()
    finally:
        tmp_path.chmod(0o700)


def test_valid_local_registration_is_only_preparation_not_task_compatibility(
    prepared_source, tmp_path, monkeypatch
):
    from blueprint_pipeline.beta_policy_screening_launch import _record
    from blueprint_pipeline.scripted_robot_baseline_runtime import build_policy_spec
    from blueprint_pipeline.common import write_json
    from tests.test_scripted_robot_grasp_frames import development_manifest
    from tests.test_team_embodiment import write_robot_usd

    model = write_robot_usd(tmp_path / "development_only.usda")
    manifest = development_manifest(model=model)
    spec = build_policy_spec(manifest)
    manifest_path = tmp_path / "development_only_manifest.json"
    write_json(manifest_path, manifest)
    key = "development_scripted_fixture"
    rights = {
        "schema_version": "blueprint.team_policy_candidate_rights.v1",
        "status": "admitted",
        "team_id": "development-only",
        "checkpoint_id": key,
        "candidate_id": key,
        "license_to_run_granted": True,
        "policy_self_grading_permitted": False,
        "official_ranking_permitted": False,
        "secret_material_recorded": False,
        "adp_qualified_evidence_permitted": False,
    }
    rights["rights_digest"] = canonical_digest(rights, digest_field="rights_digest")
    registration = {
        "manifest": _record(manifest_path),
        "model": _record(model),
        "rights_binding": rights,
        "team_id": "development-only",
        "checkpoint_id": key,
        "policy_spec": spec,
        "display_name": "Development-only scripted configuration",
    }
    configure_registry(
        tmp_path,
        monkeypatch,
        defaults={"screening_candidates": {key: {"robot_policy": registration}}},
    )
    row = next(row for row in read(prepared_source)["candidates"] if row["candidate_id"] == key)
    assert row["policy_kind"] == "scripted" and row["known_reasons"] == []
    assert row["compatibility"] == "not_evaluated"
    model.write_text("changed development model")
    changed = next(row for row in read(prepared_source)["candidates"] if row["candidate_id"] == key)
    assert changed["known_reasons"] and changed["compatibility"] == "not_evaluated"
