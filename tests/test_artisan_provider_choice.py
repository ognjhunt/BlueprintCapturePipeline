"""ARTISAN's text provider is chosen once per grant on the host; OpenAI unless every condition holds."""
from __future__ import annotations

import json

import pytest

from blueprint_pipeline import artisan_provider_choice as choice
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

TERMS = "sha256:" + "a" * 64
PASSED = {"status": choice.MODEL_READ_PASSED, "model_metadata_digest": "sha256:" + "b" * 64}


@pytest.fixture(autouse=True)
def _no_host_folders(tmp_path, monkeypatch):
    """Nothing here may write to the host's own month journal or grant records."""
    from blueprint_pipeline import paid_spend_meter
    monkeypatch.setattr(paid_spend_meter, "DEFAULT_METER_ROOT", tmp_path / "unexpected" / "meter")
    monkeypatch.setattr(choice, "DEFAULT_SUCCESSOR_ROOT", tmp_path / "unexpected" / "choices")
    yield
    assert not (tmp_path / "unexpected").exists()


def _grant(**changes):
    value = {"schema_version": "website_scene_sponsorship.v1", "authoring_providers": ["openai", "anthropic"],
             "anthropic_provider_terms_reference": TERMS, "privacy_policy_version": choice.PRIVACY_POLICY_VERSION,
             "site_consents": [], **changes}
    value = {k: v for k, v in value.items() if v is not None}
    value["authority_digest"] = canonical_digest(value, digest_field="authority_digest")
    return value


@pytest.fixture
def host(tmp_path, monkeypatch):
    key = tmp_path / "anthropic.key"
    key.write_text("issued\n")
    key.chmod(0o600)
    monkeypatch.setenv(choice.SUCCESSOR_ROOT_ENV, str(tmp_path / "grant-choices"))
    return {choice.PREFER_ANTHROPIC_ENV: "1", choice.KEY_FILE_ENV: str(key),
            choice.SUCCESSOR_ROOT_ENV: str(tmp_path / "grant-choices")}


class _Read:
    def __init__(self, answer=PASSED):
        self.calls, self.answer = 0, answer

    def __call__(self):
        self.calls += 1
        return dict(self.answer)


def test_anthropic_only_when_every_condition_holds(host):
    read = _Read()
    record = choice.choose_authoring_provider(grant=_grant(), environ=host, read=read)
    assert record["provider"] == "anthropic" and record["reason"] == "all_conditions_met"
    assert record["anthropic_provider_terms_reference"] == TERMS and record["model_read"] == PASSED
    assert read.calls == 1 and choice.validate_choice(record) == record


@pytest.mark.parametrize(("grant", "environ", "reason"), [
    ({"authoring_providers": None}, {}, "grant_allows_openai_only"),
    ({"authoring_providers": ["openai"]}, {}, "grant_allows_openai_only"),
    ({"anthropic_provider_terms_reference": "unverified"}, {}, "grant_allows_openai_only"),
    ({}, {choice.PREFER_ANTHROPIC_ENV: "0"}, "host_prefers_openai"),
    ({}, {choice.PREFER_ANTHROPIC_ENV: None}, "host_prefers_openai"),
    ({}, {choice.KEY_FILE_ENV: None}, "anthropic_key_file_absent_or_not_owner_only"),
    ({}, {choice.KEY_FILE_ENV: "relative.key"}, "anthropic_key_file_absent_or_not_owner_only"),
    ({"privacy_policy_version": "2026-09-24.1"}, {}, "privacy_version_not_listed_and_typed_consent_absent"),
    ({"privacy_policy_version": None}, {}, "privacy_version_not_listed_and_typed_consent_absent"),
])
def test_any_failed_condition_means_openai_with_its_reason_and_no_model_read(host, grant, environ, reason):
    environ = {k: v for k, v in {**host, **environ}.items() if v is not None}
    read = _Read()
    record = choice.choose_authoring_provider(grant=_grant(**grant), environ=environ, read=read)
    assert (record["provider"], record["reason"], read.calls) == ("openai", reason, 0)


def test_a_key_file_others_can_read_is_not_used(host):
    from pathlib import Path
    Path(host[choice.KEY_FILE_ENV]).chmod(0o640)
    record = choice.choose_authoring_provider(grant=_grant(), environ=host, read=_Read())
    assert record["reason"] == "anthropic_key_file_absent_or_not_owner_only"


def test_an_older_or_unversioned_job_needs_the_typed_consent_for_that_statement_and_model(host):
    for version in ("2026-09-24.1", None):
        record = choice.choose_authoring_provider(
            grant=_grant(privacy_policy_version=version, site_consents=[dict(choice.TYPED_CONSENT)]),
            environ=host, read=_Read())
        assert record["provider"] == "anthropic"
    for wrong in ({**choice.TYPED_CONSENT, "model": "claude-opus-5"}, {**choice.TYPED_CONSENT, "statement_version": "x"},
                  {**choice.TYPED_CONSENT, "provider": "openai"}):
        record = choice.choose_authoring_provider(grant=_grant(privacy_policy_version=None, site_consents=[wrong]),
                                                  environ=host, read=_Read())
        assert record["provider"] == "openai"
    assert choice.privacy_refusal(_grant()) is None  # under the listed version no consent is needed


def test_a_key_or_model_the_unbilled_read_refuses_means_openai(host):
    record = choice.choose_authoring_provider(grant=_grant(), environ=host,
                                              read=_Read({"status": "claude_model_metadata_unavailable"}))
    assert (record["provider"], record["reason"]) == ("openai", "model_read:claude_model_metadata_unavailable")


def test_the_model_read_is_the_existing_unbilled_preflight(monkeypatch):
    from blueprint_pipeline import claude_opus_model_preflight as preflight
    from blueprint_pipeline.claude_opus_authoring_invoker import ClaudeAuthoringBlocked
    monkeypatch.setattr(preflight, "preflight", lambda: {"status": choice.MODEL_READ_PASSED,
                                                         "model_metadata_digest": PASSED["model_metadata_digest"]})
    assert choice.model_read() == PASSED

    def refused():
        raise ClaudeAuthoringBlocked("claude_key_file_missing")
    monkeypatch.setattr(preflight, "preflight", refused)
    assert choice.model_read() == {"status": "claude_key_file_missing"}


def test_the_choice_is_made_once_per_grant_and_a_new_grant_chooses_again(tmp_path, host):
    read = _Read()
    first = choice.retained_or_new_choice(root=tmp_path, grant=_grant(), environ=host, read=read)
    assert choice.retained_or_new_choice(root=tmp_path, grant=_grant(), environ=host,
                                         read=read) == first
    assert read.calls == 1 and json.loads((tmp_path / choice.CHOICE_FILE).read_text()) == first
    renewed = choice.retained_or_new_choice(root=tmp_path, grant=_grant(max_paid_attempts=2),
                                            environ=host, read=read)
    assert read.calls == 2 and renewed["grant_authority_digest"] != first["grant_authority_digest"]


def test_a_changed_or_unproven_choice_is_refused(host):
    record = choice.choose_authoring_provider(grant=_grant(), environ={}, read=_Read())
    with pytest.raises(ValueError, match="artisan_provider_choice_invalid"):
        choice.validate_choice({**record, "provider": "anthropic"})
    forged = {**record, "provider": "anthropic", "reason": "all_conditions_met", "anthropic_provider_terms_reference": TERMS}
    forged["digest"] = canonical_digest(forged, digest_field="digest")
    with pytest.raises(ValueError, match="artisan_provider_choice_invalid"):  # no passed model read
        choice.validate_choice(forged)


def test_claude_models_need_the_sealed_anthropic_choice(host):
    anthropic = choice.choose_authoring_provider(grant=_grant(), environ=host, read=_Read())
    claude = {"authoring_models": dict(choice.ARTISAN_CLAUDE_MODELS)}
    assert choice.artisan_text_provider({**claude, "artisan_provider_choice": anthropic}) == "anthropic"
    assert choice.artisan_text_provider({"authoring_models": {"provider": "openai"}}) == "openai"
    for configuration in (claude, {"authoring_models": {"provider": "openai"}, "artisan_provider_choice": anthropic}):
        with pytest.raises(ValueError):
            choice.artisan_text_provider(configuration)


def test_preparation_uses_a_choice_only_for_its_own_grant_and_an_artisan_job(host):
    grant = _grant()
    record = choice.choose_authoring_provider(grant=grant, environ=host, read=_Read())
    assert choice.choice_for_grant(None, grant=grant, backend="artisan_v1") is None
    assert choice.choice_for_grant(record, grant=grant, backend="artisan_v1") == record
    for other_grant, backend in ((_grant(max_paid_attempts=2), "artisan_v1"), (grant, "astra_cad_blender_v1"),
                                 ({**grant, "authoring_providers": ["openai"]}, "artisan_v1")):
        with pytest.raises(ValueError, match="website_artisan_provider_choice_not_for_this_grant"):
            choice.choice_for_grant(record, grant=other_grant, backend=backend)


def test_the_handoff_chooses_only_for_a_both_providers_artisan_grant_and_names_anthropic_first(
        tmp_path, host, monkeypatch):
    from blueprint_pipeline import website_capture_withdrawal_provider_copies as copies
    from blueprint_pipeline import website_scene_handoff as handoff
    for name, value in host.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(choice, "model_read", _Read())
    monkeypatch.setenv("BLUEPRINT_TASK_OBJECT_AUTHORING_BACKEND", "artisan_v1")
    capture = tmp_path / "site-req1" / "captures" / "walkthrough-req1"
    root = capture / "pipeline" / "website_scene_preparation"
    root.mkdir(parents=True)
    site = capture.parent.parent
    # Today's grant, a geometry-only job and an astra job: nothing chosen, nothing written.
    for authority, geometry_only in (({"authoring_provider": "openai"}, False), (_grant(), True),
                                     (_grant(replacement_authoring_backend="astra_cad_blender_v1"), False)):
        assert handoff.artisan_provider_choice_for(authority=authority, root=root, capture_root=capture,
                                                   geometry_only=geometry_only) is None
    assert not (root / choice.CHOICE_FILE).exists() and not (site / "website_withdrawal").exists()
    record = handoff.artisan_provider_choice_for(authority=_grant(), root=root, capture_root=capture)
    assert record["provider"] == "anthropic"
    assert json.loads((root / choice.CHOICE_FILE).read_text()) == record
    assert copies.plan_providers(site)["not_deleted_at_provider"][-1] == copies.ANTHROPIC_NOT_DELETED


def test_only_the_text_roles_move_the_design_board_stays_on_openai():
    from blueprint_pipeline.task_object_authoring_backend import ARTISAN_MODELS
    for key in ("image_model", "image_tool_model", "thinking"):
        assert choice.ARTISAN_CLAUDE_MODELS[key] == ARTISAN_MODELS[key]


def test_the_gpu_attempt_stages_only_an_owner_only_anthropic_key_for_a_declared_artisan_anthropic_run(
        tmp_path, monkeypatch):
    import inspect

    from blueprint_pipeline import task_evaluation_scene_configuration_vast as vast
    key = tmp_path / "anthropic.key"
    key.write_text("issued")
    key.chmod(0o600)
    monkeypatch.setenv("ANTHROPIC_API_KEY_FILE", str(key))
    assert vast._owner_only_anthropic_key_file() == key
    key.chmod(0o640)
    with pytest.raises(vast.TaskEvaluationSceneConfigurationVastError, match="anthropic_secret_configuration_invalid"):
        vast._owner_only_anthropic_key_file()
    for named in (None, "anthropic.key"):  # unset, or relative: main's code
        if named is None:
            monkeypatch.delenv("ANTHROPIC_API_KEY_FILE")
        else:
            monkeypatch.setenv("ANTHROPIC_API_KEY_FILE", named)
        with pytest.raises(vast.TaskEvaluationSceneConfigurationVastError,
                           match="anthropic_secret_configuration_missing"):
            vast._owner_only_anthropic_key_file()
    # Keyed on the receipt's declaration, inside the OpenAI branch (the one reservation).
    key.chmod(0o600)
    monkeypatch.setenv("ANTHROPIC_API_KEY_FILE", str(key))
    assert vast.anthropic_text_key({"replacement_authoring_text_provider": "anthropic"}) == {
        "ANTHROPIC_API_KEY_FILE": str(key)}
    assert vast.anthropic_text_key({}) == {} and vast.anthropic_text_key(None) == {}
    source = inspect.getsource(vast._provider_runtime_inputs)
    assert source.index("secret_paths.update(anthropic_text_key(receipt))") > source.index(
        'openai = authority["external_service_spend_caps"]["openai"]')


def test_the_typed_consent_must_match_exactly_and_the_key_must_be_ours(host, monkeypatch):
    extra = {**choice.TYPED_CONSENT, "scope": "anything"}
    record = choice.choose_authoring_provider(grant=_grant(privacy_policy_version=None, site_consents=[extra]),
                                              environ=host, read=_Read())
    assert record["reason"] == "privacy_version_not_listed_and_typed_consent_absent"
    monkeypatch.setattr(choice.os, "geteuid", lambda: 4321)  # a key file owned by someone else
    record = choice.choose_authoring_provider(grant=_grant(), environ=host, read=_Read())
    assert record["reason"] == "anthropic_key_file_absent_or_not_owner_only"


def test_a_stored_anthropic_choice_is_refused_for_a_grant_that_fails_the_privacy_rule(host):
    stored = choice.choose_authoring_provider(grant=_grant(), environ=host, read=_Read())
    # Same digest-bound grant identity, but the grant in hand is under an older version with no consent.
    older = {**_grant(privacy_policy_version="2026-09-24.1"), "authority_digest": stored["grant_authority_digest"]}
    with pytest.raises(ValueError, match="website_artisan_provider_choice_not_for_this_grant"):
        choice.choice_for_grant(stored, grant=older, backend="artisan_v1")
    assert choice.choice_for_grant(stored, grant=_grant(), backend="artisan_v1") == stored


def test_anthropic_means_anthropic_s_own_api_no_environment_or_configuration_can_redirect_it(host):
    from blueprint_pipeline import task_evaluation_scene_configuration_artisan_driver as driver
    # Environment: the controller inherits only its search path and proxy trust, never a base URL or headers.
    stage = {"PATH": "/usr/bin", "HTTPS_PROXY": "http://proxy", "ANTHROPIC_BASE_URL": "https://evil.example",
             "ANTHROPIC_API_URL": "https://evil.example", "ANTHROPIC_CUSTOM_HEADERS": "x: y", "ANTHROPIC_LOG": "debug"}
    environment = driver.controller_environment(stage, [], {})
    assert not any(name.startswith("ANTHROPIC") for name in environment)
    # Configuration: the frozen text models are exactly the reviewed set; one with a base URL is refused.
    anthropic = choice.choose_authoring_provider(grant=_grant(), environ=host, read=_Read())
    assert choice.artisan_text_provider({"authoring_models": dict(choice.ARTISAN_CLAUDE_MODELS),
                                         "artisan_provider_choice": anthropic}) == "anthropic"
    with pytest.raises(ValueError, match="artisan_provider_choice_models_mismatch"):
        choice.artisan_text_provider({"authoring_models": {**choice.ARTISAN_CLAUDE_MODELS, "base_url": "https://evil.example"},
                                      "artisan_provider_choice": anthropic})
    assert "base_url" not in choice.ARTISAN_CLAUDE_MODELS


def test_a_key_file_its_owner_cannot_read_is_not_used(host):
    from pathlib import Path
    Path(host[choice.KEY_FILE_ENV]).chmod(0o200)  # owner-only, but not readable
    record = choice.choose_authoring_provider(grant=_grant(), environ=host, read=_Read())
    assert record["reason"] == "anthropic_key_file_absent_or_not_owner_only"


def _handoff_case(tmp_path, monkeypatch, environ):
    from blueprint_pipeline import website_scene_handoff as handoff
    for name in (choice.PREFER_ANTHROPIC_ENV, choice.KEY_FILE_ENV):
        monkeypatch.delenv(name, raising=False)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("BLUEPRINT_TASK_OBJECT_AUTHORING_BACKEND", "artisan_v1")
    read = _Read()
    monkeypatch.setattr(choice, "model_read", read)
    capture = tmp_path / "site-req1" / "captures" / "walkthrough-req1"
    root = capture / "pipeline" / "website_scene_preparation"
    root.mkdir(parents=True)
    value = handoff.artisan_provider_choice_for(authority=_grant(), root=root, capture_root=capture)
    return value, read, root, capture.parent.parent


def test_with_no_setting_and_no_key_every_grant_runs_today_s_path_and_writes_nothing(tmp_path, monkeypatch):
    # Even a both-providers grant: no choice, no record, no ledger entry, no call; the job is today's.
    value, read, root, site = _handoff_case(tmp_path, monkeypatch, {})
    assert value is None and read.calls == 0
    assert list(root.iterdir()) == [] and not (site / "website_withdrawal").exists()
    assert choice.retained_successor(_grant()["authority_digest"]) is None
    from blueprint_pipeline import task_evaluation_scene_configuration_vast as vast
    assert vast.anthropic_text_key({"run_id": "r"}) == {}  # no key staged for a run that did not choose it


def test_with_the_setting_but_no_key_file_the_job_is_refused_before_any_call(tmp_path, monkeypatch):
    value, read, root, site = _handoff_case(tmp_path, monkeypatch, {choice.PREFER_ANTHROPIC_ENV: "1"})
    assert (value["provider"], value["reason"]) == ("openai", "anthropic_key_file_absent_or_not_owner_only")
    assert read.calls == 0 and not (site / "website_withdrawal").exists()
    assert [path.name for path in root.iterdir()] == [choice.CHOICE_FILE]
    assert not (tmp_path / "unexpected").exists()  # no month journal, no grant record


def test_a_key_file_that_is_not_utf8_reads_as_unusable_and_means_openai(tmp_path, monkeypatch, host):
    from blueprint_pipeline import claude_opus_model_preflight as preflight
    key = tmp_path / "binary.key"
    key.write_bytes(b"\xff\xfe\x00issued")
    key.chmod(0o600)
    monkeypatch.setenv(choice.KEY_FILE_ENV, str(key))

    def no_network(_key):
        raise AssertionError("no model read is sent for an unusable key")
    monkeypatch.setattr(preflight, "_get_model", no_network)
    assert choice.model_read() == {"status": "claude_key_file_invalid"}
    record = choice.choose_authoring_provider(grant=_grant(), environ={**host, choice.KEY_FILE_ENV: str(key)})
    assert (record["provider"], record["reason"]) == ("openai", "model_read:claude_key_file_invalid")


def _claude(record):
    return {"authoring_models": dict(choice.ARTISAN_CLAUDE_MODELS), "artisan_provider_choice": record}


@pytest.mark.parametrize("change, reason", [
    (lambda host, key: host.pop(choice.PREFER_ANTHROPIC_ENV), choice.SETTING_OFF_REASON),
    (lambda host, key: host.update({choice.PREFER_ANTHROPIC_ENV: "0"}), choice.SETTING_OFF_REASON),
    (lambda host, key: key.unlink(), choice.KEY_UNUSABLE_REASON),
    (lambda host, key: key.chmod(0o640), choice.KEY_UNUSABLE_REASON),
])
def test_the_owner_s_switch_or_a_lost_key_rebinds_a_chosen_job_to_openai_through_the_successor_s_rebind(
        host, change, reason):
    from pathlib import Path

    from blueprint_pipeline.task_object_authoring_backend import ARTISAN_MODELS
    from blueprint_pipeline.website_owner_authoring_cap import same_except_owner_cap
    chosen = choice.choose_authoring_provider(grant=_grant(), environ=host, read=_Read())
    frozen = _claude(chosen)
    assert choice.with_provider_successor(frozen, environ=host) == frozen  # every condition still holds
    change(host, Path(host[choice.KEY_FILE_ENV]))
    rebound = choice.with_provider_successor(frozen, environ=host)
    record = rebound["artisan_provider_choice"]
    assert rebound["authoring_models"] == ARTISAN_MODELS and choice.artisan_text_provider(rebound) == "openai"
    assert (record["provider"], record["reason"], record["supersedes"]) == (
        "openai", reason, {"digest": chosen["digest"], "provider": "anthropic"})
    # Saved as the grant's successor, exactly as the no-funds one is.
    assert choice.validate_choice(record) == record and choice.retained_successor(chosen["grant_authority_digest"],
                                                                                   environ=host) == record
    # A frozen attempt is re-checked from its own record, whatever the host reads later.
    assert same_except_owner_cap(rebound, choice.as_rebound(rebound, frozen))
    assert choice.as_rebound(frozen, frozen) == frozen
    forged = {**rebound, "artisan_provider_choice": {**record, "reason": "anything"}}
    assert choice.as_rebound(forged, frozen) == frozen


@pytest.mark.parametrize(("family", "removal", "articulated"), [
    ("navigation", (), False), ("restock_shelves", "cups", False), ("rigid_pick_place", "cups", False),
    ("wiping", (), False), ("rigid_pick_place", "cup", True)],
    ids=["navigation", "unknown_kind", "several_objects", "family_without_runtime", "profile_mechanism_mismatch"])
def test_a_room_package_job_never_chooses_a_provider_or_writes_a_choice_or_a_ledger_entry(
        tmp_path, monkeypatch, family, removal, articulated):
    # The choice is made only where an ARTISAN object will be authored: on a host set for Anthropic, a job
    # handed to the company builder as a room package (it builds no object) never reaches it.
    from blueprint_pipeline import website_scene_handoff as handoff
    from blueprint_pipeline.task_family_native_runtime import CATALOG_PATH_ENV
    from tests.test_website_scene_handoff_navigation import CUPS, room_job
    key = tmp_path / "anthropic.key"
    key.write_text("issued")
    key.chmod(0o600)
    for name, value in ((choice.PREFER_ANTHROPIC_ENV, "1"), (choice.KEY_FILE_ENV, str(key)),
                        ("BLUEPRINT_TASK_OBJECT_AUTHORING_BACKEND", "artisan_v1")):
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(CATALOG_PATH_ENV, raising=False)
    asked = []
    monkeypatch.setattr(handoff, "artisan_provider_choice_for", lambda **kw: asked.append(kw))
    job, _context, result = room_job(tmp_path, monkeypatch, family=family, description="Restock the shelves.",
                                     removal={"cups": CUPS, "cup": CUPS[:1]}.get(removal, removal),
                                     all_targets=family != "navigation", articulated=articulated)
    assert result["status"] == "room_package_written" and asked == []
    assert not list(job.capture.rglob(choice.CHOICE_FILE))
    assert not list(job.capture.parent.parent.rglob("website_withdrawal"))


def test_a_gap_compile_finds_late_returns_before_the_handoff_s_choice_is_asked_for(tmp_path, monkeypatch):
    # Compile's own gap (here a soft object only the object analysis catches) becomes a room package at the
    # handoff; the choice the handoff passes is asked for only after every such gap has returned.
    from blueprint_pipeline import website_task_preparation as preparation
    from tests.test_task_object_rigidity import DEFORMABLE, _arguments, _classified, _context
    arguments = _arguments(tmp_path, removal_manifest=_classified("plush toy", "deformable", "soft fabric squashes"))
    _context(arguments, "rigid_pick_and_place")
    monkeypatch.setattr(preparation, "register_source_to_runtime", lambda **_: pytest.fail("not authored rigid"))
    asked = []
    result = preparation.compile_website_scene_preparation(
        **arguments, artisan_provider_choice=lambda: asked.append(1) or pytest.fail("chosen for a room package"))
    assert result["blockers"] == [DEFORMABLE] and "native_task_preparation_gap" in result and asked == []


def test_a_job_moved_to_openai_stays_there_when_the_setting_and_key_return(tmp_path, monkeypatch, host):
    from pathlib import Path
    chosen = choice.choose_authoring_provider(grant=_grant(), environ=host, read=_Read())
    frozen = _claude(chosen)
    key = Path(host[choice.KEY_FILE_ENV])
    off = choice.with_provider_successor(frozen, environ={**host, choice.PREFER_ANTHROPIC_ENV: "0"})
    record = off["artisan_provider_choice"]
    assert record["reason"] == choice.SETTING_OFF_REASON
    key.unlink()
    assert choice.with_provider_successor(frozen, environ=host) == off  # the first rebind saved stays
    key.write_text("issued\n")
    key.chmod(0o600)
    # Setting on and key back: the next attempt, and a renewal (which compiles the same frozen choice), stay.
    assert choice.with_provider_successor(frozen, environ=host) == off
    assert choice.with_provider_successor(frozen) == off
    # The next handoff of the grant reads the successor: no model read, no ledger entry.
    value, read, root, site = _handoff_case(tmp_path, monkeypatch, host)
    assert value == record and read.calls == 0 and not (site / "website_withdrawal").exists()
    assert [path.name for path in root.iterdir()] == []
    # Only a new grant chooses again.
    fresh = choice.choose_authoring_provider(grant=_grant(max_paid_attempts=2), environ=host, read=_Read())
    assert choice.with_provider_successor(_claude(fresh), environ=host) == _claude(fresh)


def test_another_job_s_rebind_record_cannot_rebind_this_job(host):
    mine = _claude(choice.choose_authoring_provider(grant=_grant(), environ=host, read=_Read()))
    theirs = _claude(choice.choose_authoring_provider(grant=_grant(max_paid_attempts=2), environ=host, read=_Read()))
    moved = choice.with_provider_successor(theirs, environ={**host, choice.PREFER_ANTHROPIC_ENV: "0"})
    assert moved["artisan_provider_choice"]["provider"] == "openai"
    stored = {**mine, "authoring_models": moved["authoring_models"],
              "artisan_provider_choice": moved["artisan_provider_choice"]}
    assert choice.as_rebound(stored, mine) == mine  # theirs supersedes another choice: refused, not applied


def test_a_gap_compile_finds_late_at_the_handoff_ships_a_room_package_without_asking_for_a_choice(
        tmp_path, monkeypatch):
    # At the handoff itself: compile (stubbed) returns a gap only it finds, so the job is a room package; the
    # choice the handoff hands compile is never asked for. Asking for it before compile would fail here.
    from blueprint_pipeline import website_scene_handoff as handoff
    from tests.test_task_object_rigidity import ATLAS, _handoff, atlas_ready
    from tests.test_website_scene_handoff import room_recorder
    value = atlas_ready(tmp_path)
    monkeypatch.setattr(ATLAS, lambda **_: value)
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry",
                        lambda **kw: {**kw["task_masks"], "source_geometry_digest": "atlas-geometry"})
    kwargs, _ = _handoff(tmp_path, monkeypatch, {"target_id": "mug", "semantic_label": "mug", "rigidity": "rigid",
                                                 "rigidity_reason": "glazed ceramic"})
    packaged = room_recorder(monkeypatch)
    for name in ("website_assembly_coverage.attach_assembly_coverage", "website_object_spec_research.attach_object_specs"):
        monkeypatch.setattr("blueprint_pipeline." + name, lambda **kw: kw["task_masks"])
    monkeypatch.setattr("blueprint_pipeline.website_preparation_revision.bind_preparation", lambda value, **_: value)
    asked, handed = [], []
    monkeypatch.setattr(handoff, "artisan_provider_choice_for", lambda **kw: asked.append(kw))

    def compile_finds_a_gap(**kw):
        handed.append(callable(kw.get("artisan_provider_choice")))
        return {"status": "needs_input", "blockers": ["late_gap"], "digest": "sha256:" + "0" * 64, "thumbnail": None,
                "native_task_preparation_gap": {"code": "late_gap"}}
    monkeypatch.setattr(handoff, "compile_website_scene_preparation", compile_finds_a_gap)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["status"] == "room_package_written" and [call["flags"] for call in packaged] == [["late_gap"]]
    assert handed == [True] and asked == []
