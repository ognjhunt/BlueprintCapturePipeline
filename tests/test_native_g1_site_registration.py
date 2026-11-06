"""The service registers a pick-and-place site's G1 candidates itself; hermetic, no network or compute.

Covers: src/blueprint_pipeline/native_g1_site_registration.py, the service overlay in
src/blueprint_pipeline/team_candidate_registry_reader.py and its hook in
src/blueprint_pipeline/beta_policy_screening_worker.py.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shutil

import pytest

from tests.test_operator_door_g1_registration import (
    ASSET_DATA, DP, PI, DP_NAV, PI_NAV, builder, contract, fixture as door_fixture, installer, make_packet, ref,
    seal, write,
)
from tests.test_beta_policy_screening_g1_prepare import prepared_scene  # noqa: F401 - the real compiled task
from blueprint_pipeline import beta_policy_screening_g1 as g1
from blueprint_pipeline import beta_policy_screening_g1_prepare as prepare
from blueprint_pipeline import native_g1_site_registration as registration
from blueprint_pipeline import team_candidate_registry_reader as reader
from blueprint_pipeline.beta_policy_screening_g1_catalog import CANDIDATES
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

SHA = "sha256:" + "a" * 64
SITE = "site-development"


def shared_manifest(tmp_path, *, publisher, runtime, robot, robot_dir, **changes):
    value = {"schema_version": registration.MANIFEST_SCHEMA,
        "publisher_source": {"path": str(publisher), "receipt_digest": json.loads(publisher.read_text())["receipt_digest"]},
        "runtime_source": {"path": str(runtime), **ref(runtime)},
        "robot_asset": {"manipulation_packet_dir": str(robot_dir), **ref(robot)},
        "authorization_reference": "development_only standing approval fixture", "candidates": list(CANDIDATES)}
    value.update(changes)
    value["manifest_digest"] = canonical_digest(value, digest_field="manifest_digest")
    path = tmp_path / "pinned/native_g1_shared_inputs.v1.json"
    write(path, value)
    return path


def door_world(tmp_path, monkeypatch, **changes):
    """The door's own synthetic site, with the shared inputs pinned in a manifest instead of the ADP registry."""
    retained = door_fixture(tmp_path)
    monkeypatch.setenv(reader.REGISTRY_ENV, str(retained.registry_path))
    robot_dir = tmp_path / "shared/manipulation"
    robot = robot_dir / "assets/robot_unitree_g1.usd"
    robot.parent.mkdir(parents=True)
    robot.write_bytes(ASSET_DATA)
    write(robot_dir / "native_task_arena_scene_plan.v1.json", {"robot": {"usd_path": "assets/robot_unitree_g1.usd"}})
    pinned = shared_manifest(tmp_path, publisher=tmp_path / "shared/publisher/native_g1_publisher_source_stage.v1.json",
        runtime=tmp_path / "shared/runtime/native_task_runtime_source_packet.v1.json", robot=robot, robot_dir=robot_dir,
        **changes)
    kwargs = {"site_id": SITE, "source": dict(retained.request["source"]), "source_queue": tmp_path / "queue",
              "release": "a" * 40, "manifest_path": pinned}
    profile = {**retained.derived, **retained.registry["derived_profiles"]["defaults"]}
    # The live consent read is the WebApp's (its own tests cover it): granted unless a test says otherwise.
    monkeypatch.setattr("blueprint_pipeline.beta_policy_screening_worker.require_current_source", lambda *a, **k: {})
    return retained, kwargs, profile, robot


def overlay_path(site=SITE):
    return Path(os.environ[reader.SITE_REGISTRATION_ROOT_ENV]) / f"{site}.json"


def status_of(site=SITE):
    return json.loads(overlay_path(site).with_name(f"{site}.status.json").read_text())


def test_site_registers_exactly_what_the_door_builder_would_and_the_reader_accepts_it(tmp_path, monkeypatch):
    retained, kwargs, profile, _ = door_world(tmp_path, monkeypatch)
    result = registration.ensure_site_registration(**kwargs)
    assert result["status"] == "registered"
    assert result["candidates"] == {DP: {"status": "registered"}, PI: {"status": "registered"},
        DP_NAV: {"status": "unavailable", "reason": "g1_current_navigation_goal_authority_missing"},
        PI_NAV: {"status": "unavailable", "reason": "g1_current_navigation_goal_authority_missing"}}
    row = json.loads(overlay_path().read_text())
    assert row == {"source": kwargs["source"],
                   "screening_candidates": contract.common.read(retained.proposal)["registrations"]}
    assert overlay_path().stat().st_mode & 0o777 == 0o640
    status = status_of()
    assert status["registered_by"] == "service_automatic" and status["status"] == "registered"
    assert status["authorization_reference"] == "development_only standing approval fixture"
    assert status["execution_authority_granted"] is False and status["provider_allocations_performed"] == 0
    assert status["generated_at"] and status["source"] == kwargs["source"]
    actual = reader._site_screening_profile(retained.registry, profile)
    assert actual["native_g1_registration_source"] == kwargs["source"]
    assert actual["screening_candidates"]["existing"] == {"keep": True}
    assert actual["screening_candidates"][DP] == row["screening_candidates"][DP]
    # The root registry is never written by the service.
    assert contract.common.read(retained.registry_path) == retained.registry


def test_a_missing_packet_is_prepared_with_the_builders_own_arguments(tmp_path, monkeypatch):
    retained, kwargs, _, robot = door_world(tmp_path, monkeypatch)
    packet = retained.g1_path.parent
    kept = tmp_path / "kept-packet"
    packet.rename(kept)
    calls = []

    def prepared(**arguments):
        calls.append(arguments)
        shutil.copytree(kept, packet)

    monkeypatch.setattr(prepare, "prepare_native_g1_beta_packet", prepared)
    checked = []
    monkeypatch.setattr("blueprint_pipeline.beta_policy_screening_worker.require_current_source",
                        lambda value, *, site_id: checked.append((value, site_id)))
    assert registration.ensure_site_registration(**kwargs)["status"] == "registered"
    # Consent is read live before any new derived data, and again before the row is written.
    assert checked == [(kwargs["source"], SITE)] * 2
    site = tmp_path / "halves" / SITE
    inputs = contract.common.read(site / "team_scene_inputs.v1.json")
    assert calls == [{"base_packet_record": inputs["base_native_packet"], "source": kwargs["source"], "site_id": SITE,
        "output_root": site, "robot_asset_record": contract.file_record(robot),
        "task_success_contract": inputs["task_success_contract"]}]


@pytest.mark.parametrize("raised,reason,retry", [
    (ValueError("beta_screening_current_capture_consent_required"), "beta_screening_current_capture_consent_required",
     "on_source_pin_or_release_change"),
    (OSError("development_only unreachable"), "g1_current_consent_unverified", "next_pass"),
])
def test_withdrawn_or_unverified_consent_prepares_nothing(tmp_path, monkeypatch, raised, reason, retry):
    retained, kwargs, profile, _ = door_world(tmp_path, monkeypatch)
    shutil.rmtree(retained.g1_path.parent)

    def withdrawn(value, *, site_id):
        raise raised

    monkeypatch.setattr("blueprint_pipeline.beta_policy_screening_worker.require_current_source", withdrawn)
    monkeypatch.setattr(prepare, "prepare_native_g1_beta_packet", lambda **_: pytest.fail("prepared"))
    result = registration.ensure_site_registration(**kwargs)
    assert result["status"] == "not_registered" and result["reason"] == reason
    assert status_of()["retry"] == retry and not overlay_path().exists()
    assert not retained.g1_path.parent.exists()


def repin_runtime(tmp_path, kwargs):
    """Pin a rebuilt runtime receipt: the site's row goes stale while its packet stays where it is."""
    runtime = tmp_path / "shared/runtime/native_task_runtime_source_packet.v1.json"
    rebuilt = tmp_path / "shared/rebuilt/native_task_runtime_source_packet.v1.json"
    write(rebuilt, {**json.loads(runtime.read_text()), "development_only": "rebuilt"})
    value = json.loads(Path(kwargs["manifest_path"]).read_text())
    value["runtime_source"] = {"path": str(rebuilt), **ref(rebuilt)}
    write(kwargs["manifest_path"], seal(value, "manifest_digest"))


@pytest.mark.parametrize("refusal", ["beta_screening_current_capture_consent_required", "team_scene_capture_withdrawn"])
def test_a_refused_consent_with_an_existing_packet_writes_no_row_and_removes_the_old_one(
        tmp_path, monkeypatch, refusal):
    from blueprint_pipeline.team_scene_capture_root import TeamScenePacketRequestRefused

    retained, kwargs, profile, _ = door_world(tmp_path, monkeypatch)
    assert registration.ensure_site_registration(**kwargs)["status"] == "registered"
    old_row = overlay_path().read_bytes()
    packet = sorted(retained.g1_path.parent.rglob("*"))
    repin_runtime(tmp_path, kwargs)  # The packet exists: only the row would be written.
    checks = []

    def withdrawn(value, *, site_id):
        checks.append(site_id)
        raise TeamScenePacketRequestRefused(refusal)

    monkeypatch.setattr("blueprint_pipeline.beta_policy_screening_worker.require_current_source", withdrawn)
    monkeypatch.setattr(prepare, "prepare_native_g1_beta_packet", lambda **_: pytest.fail("prepared"))
    assert registration.ensure_site_registration(**kwargs) == {"status": "not_registered", "reason": refusal}
    assert checks == [SITE] and not overlay_path().exists()
    assert reader._site_screening_profile(retained.registry, profile) == profile
    # The status keeps the refusal reason only: no source, inputs, paths or candidates.
    status = status_of()
    assert set(status) == {"schema_version", "site_id", "registered_by", "status", "reason", "retry", "attempt_key",
                           "generated_at"}
    assert status["reason"] == refusal and status["retry"] == "on_source_pin_or_release_change"
    text = overlay_path().with_name(f"{SITE}.status.json").read_text()
    assert kwargs["source"]["capture_id"] not in text and kwargs["source"]["task_context_digest"] not in text
    # Deleting retained data is the withdrawal deletion's: the packet itself is left as it was.
    assert sorted(retained.g1_path.parent.rglob("*")) == packet
    # The refusal stands with no new live read; a row put back meanwhile is removed again, never kept.
    write(overlay_path(), json.loads(old_row))
    overlay_path().chmod(0o640)
    assert registration.ensure_site_registration(**kwargs)["reason"] == refusal
    assert checks == [SITE] and not overlay_path().exists() and status_of() == status


def test_a_refused_retained_consent_removes_the_row(tmp_path, monkeypatch):
    retained, kwargs, profile, _ = door_world(tmp_path, monkeypatch)
    assert registration.ensure_site_registration(**kwargs)["status"] == "registered"
    path = tmp_path / "capture/pipeline/website_task_context.json"
    context = json.loads(path.read_text())
    context["capture_rights"] = {**context["capture_rights"], "consent_revoked": True}
    write(path, seal(context, "context_digest"))
    result = registration.ensure_site_registration(**kwargs)
    assert result == {"status": "not_registered", "reason": "native_g1_install_retained_task_consent_invalid"}
    assert not overlay_path().exists() and "source" not in status_of()
    assert reader._site_screening_profile(retained.registry, profile) == profile


@pytest.mark.parametrize("row", ["stale", "absent"])
def test_an_unanswered_consent_check_writes_nothing_new_and_is_tried_again(tmp_path, monkeypatch, row):
    retained, kwargs, _, _ = door_world(tmp_path, monkeypatch)
    assert registration.ensure_site_registration(**kwargs)["status"] == "registered"
    old_row = overlay_path().read_bytes()
    repin_runtime(tmp_path, kwargs)
    if row == "absent":
        overlay_path().unlink()
    checks = []

    def unanswered(value, *, site_id):
        checks.append(site_id)
        raise OSError("development_only unreachable")

    monkeypatch.setattr("blueprint_pipeline.beta_policy_screening_worker.require_current_source", unanswered)
    for attempt in (1, 2):
        result = registration.ensure_site_registration(**kwargs)
        assert result["status"] == "not_registered" and result["reason"] == "g1_current_consent_unverified"
        assert status_of()["retry"] == "next_pass" and len(checks) == attempt
        # Nothing new: no row written, and a row from before is neither replaced nor removed.
        assert (overlay_path().read_bytes() == old_row) if row == "stale" else not overlay_path().exists()
    assert retained.g1_path.exists()


def test_registration_is_idempotent_for_an_unchanged_source(tmp_path, monkeypatch):
    _, kwargs, _, _ = door_world(tmp_path, monkeypatch)
    assert registration.ensure_site_registration(**kwargs)["status"] == "registered"
    row, status = overlay_path().read_bytes(), status_of()
    # The current base packet is resolved to confirm the row is still its own; nothing is prepared or written.
    monkeypatch.setattr(prepare, "prepare_native_g1_beta_packet", lambda **_: pytest.fail("prepared again"))
    for _ in range(2):
        assert registration.ensure_site_registration(**kwargs) == {"status": "already_registered"}
    assert overlay_path().read_bytes() == row and status_of() == status


def test_a_recompiled_base_packet_with_the_same_source_is_registered_again(tmp_path, monkeypatch):
    # A recompiled team scene keeps the six-field source but has a new base receipt: the old row would
    # fail g1_beta_source_scene_derivation_mismatch at planning, so the fast path must not keep it.
    _, kwargs, _, _ = door_world(tmp_path, monkeypatch)
    assert registration.ensure_site_registration(**kwargs)["status"] == "registered"
    row = json.loads(overlay_path().read_text())
    base = registration._current_base_hex(SITE, kwargs["source"], kwargs["source_queue"])
    assert base and all(f"/native-g1/{base}/" in item["native_g1"]["scene_packet"]["path"]
                        for item in row["screening_candidates"].values())
    assert registration._built_from_base(row, base) and not registration._built_from_base(row, None)
    assert not registration._built_from_base(row, "0" * 64)
    attempts = []
    monkeypatch.setattr(registration, "_current_base_hex", lambda *_a, **_k: "0" * 64)
    monkeypatch.setattr(registration, "_register", lambda **arguments: attempts.append(arguments) or (
        {"status": "registered", "candidates": {}}, False))
    assert registration.ensure_site_registration(**kwargs)["status"] == "registered"
    assert len(attempts) == 1 and attempts[0]["current"] == kwargs["source"]


def test_a_changed_record_is_registered_again(tmp_path, monkeypatch):
    retained, kwargs, _, _ = door_world(tmp_path, monkeypatch)
    registration.ensure_site_registration(**kwargs)
    first = json.loads(overlay_path().read_text())
    # A stale row (its runtime receipt record no longer verifies) is not current, so it is written anew.
    row = deepcopy(first)
    row["screening_candidates"][DP]["native_g1"]["runtime_source"]["size_bytes"] += 1
    write(overlay_path(), row)
    overlay_path().chmod(0o640)
    assert registration.ensure_site_registration(**kwargs)["status"] == "registered"
    assert json.loads(overlay_path().read_text()) == first
    # A newly pinned runtime receipt replaces a row whose old receipt still verifies.
    runtime = tmp_path / "shared/runtime/native_task_runtime_source_packet.v1.json"
    rebuilt = tmp_path / "shared/rebuilt/native_task_runtime_source_packet.v1.json"
    write(rebuilt, {**json.loads(runtime.read_text()), "development_only": "rebuilt"})
    value = json.loads(Path(kwargs["manifest_path"]).read_text())
    value["runtime_source"] = {"path": str(rebuilt), **ref(rebuilt)}
    write(kwargs["manifest_path"], seal(value, "manifest_digest"))
    assert registration.ensure_site_registration(**kwargs)["status"] == "registered"
    row = json.loads(overlay_path().read_text())
    assert row["screening_candidates"][PI]["native_g1"]["runtime_source"] == contract.file_record(rebuilt)


@pytest.mark.parametrize("fault,reason", [
    ("runtime_bytes", "g1_shared_runtime_source_changed"),
    ("runtime_size", "g1_shared_runtime_source_changed"),
    ("runtime_missing", "g1_shared_runtime_source_missing"),
    ("publisher_digest", "g1_shared_publisher_source_changed"),
    ("robot_bytes", "g1_shared_robot_asset_changed"),
])
def test_a_tampered_shared_input_is_refused_with_a_reason_and_no_row(tmp_path, monkeypatch, fault, reason):
    retained, kwargs, profile, robot = door_world(tmp_path, monkeypatch)
    runtime = tmp_path / "shared/runtime/native_task_runtime_source_packet.v1.json"
    publisher = tmp_path / "shared/publisher/native_g1_publisher_source_stage.v1.json"
    if fault == "runtime_bytes":
        runtime.write_text(runtime.read_text() + " ")
    elif fault == "runtime_size":
        value = json.loads(Path(kwargs["manifest_path"]).read_text())
        value["runtime_source"]["size_bytes"] += 1
        write(kwargs["manifest_path"], seal(value, "manifest_digest"))
    elif fault == "runtime_missing":
        runtime.unlink()
    elif fault == "publisher_digest":
        write(publisher, seal({**json.loads(publisher.read_text()), "status": "changed"}, "receipt_digest"))
    else:
        shutil.rmtree(retained.g1_path.parent)  # The robot is read only to prepare a missing packet.
        robot.write_bytes(ASSET_DATA + b"#")
    monkeypatch.setattr(prepare, "prepare_native_g1_beta_packet", lambda **_: pytest.fail("prepared"))
    result = registration.ensure_site_registration(**kwargs)
    assert result["status"] == "not_registered" and result["reason"] == reason
    assert set(result["candidates"]) == set(CANDIDATES)
    assert not overlay_path().exists()
    status = status_of()
    assert status["reason"] == reason
    assert status["retry"] == ("on_source_pin_or_release_change" if fault == "robot_bytes" else "next_pass")
    assert reader._site_screening_profile(retained.registry, profile) == profile


def rebuild_base(tmp_path, **scene_changes):
    """Re-seal the door fixture's base packet with a changed scene plan (and its team inputs)."""
    site = tmp_path / "halves" / SITE
    base = site / "base"
    receipt = contract.common.read(base / "native_task_arena_packet_receipt.v1.json")
    scene = {**contract.common.read(base / "native_task_arena_scene_plan.v1.json"), **scene_changes}
    request_value = contract.common.read(base / "native_task_arena_packet_request.v1.json")
    shutil.rmtree(base)
    path, _ = make_packet(base, scene=scene, request_value=request_value, source_bindings=receipt["source_bindings"])
    inputs = contract.common.read(site / "team_scene_inputs.v1.json")
    inputs["base_native_packet"] = contract.file_record(path)
    write(site / "team_scene_inputs.v1.json", seal(inputs, "inputs_digest"))


def test_a_non_rigid_task_is_a_flagged_reason_with_no_row_and_no_exception(tmp_path, monkeypatch):
    retained, kwargs, profile, _ = door_world(tmp_path, monkeypatch)
    rebuild_base(tmp_path, task_kind="articulated_open_close")
    resolved = []
    # The service resolves with the door builder's own loaded copy of the contract.
    site_inputs = builder.contract.site_inputs
    monkeypatch.setattr(builder.contract, "site_inputs", lambda *a, **k: resolved.append(1) or site_inputs(*a, **k))
    monkeypatch.setattr(prepare, "prepare_native_g1_beta_packet", lambda **_: pytest.fail("prepared"))
    result = registration.ensure_site_registration(**kwargs)
    assert result["status"] == "not_registered" and result["reason"] == "g1_beta_native_rigid_task_required"
    assert not overlay_path().exists()
    assert status_of()["retry"] == "on_source_pin_or_release_change"
    # Nothing it depends on changed: the next pass reads only the status.
    assert registration.ensure_site_registration(**kwargs)["reason"] == "g1_beta_native_rigid_task_required"
    assert len(resolved) == 1
    # A new release resolves the site again.
    registration.ensure_site_registration(**{**kwargs, "release": "b" * 40})
    assert len(resolved) == 2
    assert reader._site_screening_profile(retained.registry, profile) == profile


def test_a_navigation_goal_registers_only_the_navigation_rows_with_their_authority(tmp_path, monkeypatch):
    retained, kwargs, _, _ = door_world(tmp_path, monkeypatch)
    rebuild_base(tmp_path, task_spec={**contract.common.read(
        tmp_path / "halves" / SITE / "base/native_task_arena_scene_plan.v1.json")["task_spec"],
        "g1_navigation_goal": {"development_only": True}})
    base = contract.common.read(tmp_path / "halves" / SITE / "base/native_task_arena_packet_receipt.v1.json")
    packet = tmp_path / "halves" / SITE / "native-g1" / base["receipt_digest"][7:] / canonical_digest(kwargs["source"])[7:]
    shutil.copytree(retained.g1_path.parent, packet)  # Planning, not registration, verifies the packet.
    plan = contract.common.read(packet / "native_task_arena_scene_plan.v1.json")
    authority = packet.parent / "native_g1_navigation_goal_authority.v1.json"
    write(authority, {"scope": {"scene_plan_digest": plan["plan_digest"]}})
    result = registration.ensure_site_registration(**kwargs)
    assert result["candidates"] == {
        DP: {"status": "unavailable", "reason": "g1_beta_manipulation_policy_task_mismatch"},
        PI: {"status": "unavailable", "reason": "g1_beta_manipulation_policy_task_mismatch"},
        DP_NAV: {"status": "registered"}, PI_NAV: {"status": "registered"}}
    row = json.loads(overlay_path().read_text())
    assert row["screening_candidates"][DP_NAV]["native_g1"]["navigation_authority"] == contract.file_record(authority)


def test_a_current_root_registry_row_wins_over_the_service_row(tmp_path, monkeypatch):
    retained, kwargs, profile, _ = door_world(tmp_path, monkeypatch)
    installer.apply(retained.request, **retained.options)
    installed = contract.common.read(retained.registry_path)
    assert registration.ensure_site_registration(**kwargs)["status"] == "root_registry_row_current"
    assert not overlay_path().exists()
    # A service row that only names DP: the current root row (DP and PI) is the one read.
    root_row = installed["site_screening_candidates"][SITE]
    write(overlay_path(), {"source": kwargs["source"], "screening_candidates": {DP: root_row["screening_candidates"][DP]}})
    assert {DP, PI} <= set(reader._site_screening_profile(installed, profile)["screening_candidates"])
    # A root row for another source no longer applies; the service row for the current one does.
    stale = deepcopy(installed)
    stale["site_screening_candidates"][SITE]["source"]["scene_revision_digest"] = "sha256:" + "b" * 64
    actual = reader._site_screening_profile(stale, profile)
    assert DP in actual["screening_candidates"] and PI not in actual["screening_candidates"]


@pytest.mark.parametrize("fault", ["symlink", "group_writable", "foreign_owner", "symlinked_root", "other_shape"])
def test_a_symlinked_or_foreign_service_row_is_ignored(tmp_path, monkeypatch, fault):
    retained, kwargs, profile, _ = door_world(tmp_path, monkeypatch)
    registration.ensure_site_registration(**kwargs)
    path = overlay_path()
    assert reader._site_screening_profile(retained.registry, profile) != profile
    if fault == "symlink":
        target = path.with_name("elsewhere.json")
        path.rename(target)
        path.symlink_to(target)
    elif fault == "group_writable":
        path.chmod(0o660)
    elif fault == "foreign_owner":
        if os.geteuid() == 0:
            os.chown(path, 4242, 4242)
        else:
            uid = os.getuid()
            monkeypatch.setattr(reader.os, "getuid", lambda: uid + 1)
    elif fault == "symlinked_root":
        real = tmp_path / "real-root"
        path.parent.rename(real)
        path.parent.symlink_to(real, target_is_directory=True)
    else:
        write(path, {**json.loads(path.read_text()), "registered_by": "someone"})
        path.chmod(0o640)
    assert reader._site_screening_profile(retained.registry, profile) == profile
    if fault == "symlink":
        # The service writes its own regular file back, never through the link.
        before = target.read_bytes()
        assert registration.ensure_site_registration(**kwargs)["status"] == "registered"
        assert not path.is_symlink() and target.read_bytes() == before


def test_the_kill_switch_stops_registration_and_the_reading_of_service_rows(tmp_path, monkeypatch):
    retained, kwargs, profile, _ = door_world(tmp_path, monkeypatch)
    monkeypatch.setenv(reader.AUTO_REGISTRATION_ENV, "0")
    assert registration.ensure_site_registration(**kwargs) == {"status": "disabled"}
    assert not list(overlay_path().parent.iterdir())
    monkeypatch.delenv(reader.AUTO_REGISTRATION_ENV)
    assert registration.ensure_site_registration(**kwargs)["status"] == "registered"
    assert reader._site_screening_profile(retained.registry, profile) != profile
    monkeypatch.setenv(reader.AUTO_REGISTRATION_ENV, "0")
    assert reader._site_screening_profile(retained.registry, profile) == profile


def test_an_unexpected_failure_is_recorded_and_never_raised(tmp_path, monkeypatch):
    _, kwargs, _, _ = door_world(tmp_path, monkeypatch)

    def fault(*_args, **_kwargs):
        raise ZeroDivisionError("development_only injected fault")

    # A crash inside a stage is its stage's reason; one outside any stage is the generic one.
    monkeypatch.setattr(builder.contract, "site_inputs", fault)
    assert registration.ensure_site_registration(**kwargs)["reason"] == "g1_current_site_inputs_unavailable"
    monkeypatch.setattr(registration, "_shared_records", fault)
    result = registration.ensure_site_registration(**{**kwargs, "release": "b" * 40})
    assert result["status"] == "not_registered" and result["reason"] == "g1_site_registration_failed"
    assert status_of()["error"]["type"] == "builtins.ZeroDivisionError"
    monkeypatch.setattr(registration, "_register", fault)
    assert registration.ensure_site_registration(**kwargs) == {
        "status": "not_registered", "reason": "g1_site_registration_failed"}
    assert registration.ensure_site_registration(**{**kwargs, "site_id": "../escape"}) == {
        "status": "not_registered", "reason": "g1_site_identity_invalid"}


@pytest.fixture
def compiled_world(prepared_scene, tmp_path, monkeypatch):  # noqa: F811 - the imported fixture
    """A real compiled pick-and-place task, its site records and a pinned manifest; nothing mocked."""
    args, request, inputs = prepared_scene
    site = args["site_id"]
    # The compiled fixture names an interior scene; accept exactly it as a website site id.
    monkeypatch.setattr(reader, "WEBSITE_SITE_ID", re.compile(re.escape(site)))
    capture = tmp_path / "capture"
    context = seal({"schema_version": "website_site_task_context.v1", "scene_id": site,
        "capture_id": "development-capture", "capture_rights": {"consent_status": "granted", "consent_revoked": False,
            "consent_scope": ["robot_evaluation"], "derived_scene_generation_allowed": True}}, "context_digest")
    write(capture / "pipeline/website_task_context.json", context)
    source = {"capture_id": "development-capture", "task_id": request["task_id"],
        "scene_intent_id": "development-intent", "scene_intent_digest": SHA,
        "task_context_digest": context["context_digest"], "scene_revision_digest": inputs["scene_revision_digest"]}
    derived = seal({"schema_version": reader.DERIVED_SCHEMA, "capture_root": str(capture), "scene_id": site,
        "task_id": source["task_id"], "scene_revision_digest": source["scene_revision_digest"],
        "team_scene_inputs_path": str(Path(args["output_root"]) / "team_scene_inputs.v1.json")}, "profile_digest")
    write(tmp_path / "derived/site.json", derived)
    defaults = {"allowed_founder_approvers": ["development-owner"], "max_hourly_rate_usd": 0.8, "open_loop_horizon": 8}
    registry = seal({"schema_version": reader.REGISTRY_SCHEMA, "profiles": [], "derived_profiles": {
        "root": str(tmp_path / "derived"), "packet_inputs": {"output_root": str(tmp_path)}, "defaults": defaults}},
        "registry_digest")
    write(tmp_path / "etc/registry.json", registry)
    monkeypatch.setenv(reader.REGISTRY_ENV, str(tmp_path / "etc/registry.json"))
    write(tmp_path / "queue/processed/current.json", seal({"schema_version": "blueprint.team_scene_packet_request.v1",
        "intent_id": source["scene_intent_id"], "intent_digest": source["scene_intent_digest"],
        "capture_id": source["capture_id"], "task_id": source["task_id"],
        "scene_revision_digest": source["scene_revision_digest"]}, "request_digest"))
    publisher = tmp_path / "shared/publisher/native_g1_publisher_source_stage.v1.json"
    write(publisher, seal({"schema_version": "native_g1_publisher_source_stage.v1", "status": "ready"}, "receipt_digest"))
    runtime = tmp_path / "shared/runtime/native_task_runtime_source_packet.v1.json"
    write(runtime, {"schema_version": "development_only"})
    robot = Path(args["robot_asset_record"]["path"])
    write(tmp_path / "shared/manipulation/native_task_arena_scene_plan.v1.json", {"robot": {"usd_path": str(robot)}})
    pinned = shared_manifest(tmp_path, publisher=publisher, runtime=runtime, robot=robot,
                             robot_dir=tmp_path / "shared/manipulation")
    kwargs = {"site_id": site, "source": source, "source_queue": tmp_path / "queue", "release": "a" * 40,
              "manifest_path": pinned}
    # The live consent read is the WebApp's; its own tests cover it.
    monkeypatch.setattr("blueprint_pipeline.beta_policy_screening_worker.require_current_source", lambda *a, **k: {})
    return kwargs, {**derived, **defaults}, registry, args, capture


def test_a_real_pick_and_place_site_gets_a_row_planning_verifies(compiled_world, monkeypatch):
    kwargs, profile, registry, args, _ = compiled_world
    result = registration.ensure_site_registration(**kwargs)
    assert result["status"] == "registered", result
    assert {key for key, row in result["candidates"].items() if row["status"] == "registered"} == {DP, PI}
    actual = reader._site_screening_profile(registry, profile)
    assert actual["native_g1_registration_source"] == kwargs["source"]
    record = actual["screening_candidates"][DP]["native_g1"]["scene_packet"]
    assert Path(record["path"]).parent.name == canonical_digest(kwargs["source"])[7:]
    # Planning stays the verifier: the same spec the planner builds, only the CPU fixture stand-ins substituted.
    monkeypatch.setattr("blueprint_pipeline.native_g1_publisher_source_stage.verify_g1_publisher_source", lambda root: {})
    monkeypatch.setattr("blueprint_pipeline.native_task_runtime_source_packet.verify_native_task_runtime_source_packet",
        lambda path: {"runtime_profile": "unitree_g1", "redistribution_permitted": True})
    monkeypatch.setattr("blueprint_pipeline.beta_policy_screening_worker.require_current_source", lambda *a, **k: {})
    monkeypatch.setattr(g1, "ROBOT_USD_SHA256", args["robot_asset_record"]["sha256"])
    scene = json.loads((Path(args["base_packet_record"]["path"]).parent / "native_task_arena_scene_plan.v1.json").read_text())
    for candidate in (DP, PI):
        spec = g1.candidate_execution_spec(candidate_id=candidate,
            registration=actual["screening_candidates"][candidate]["native_g1"], scene_plan=scene,
            base_packet=args["base_packet_record"], source=kwargs["source"],
            contract=args["task_success_contract"], site_id=kwargs["site_id"])
        assert spec["source"] == kwargs["source"] and spec["claim_ceiling"] == "development_only"
    # The next pass finds the row current and does nothing.
    monkeypatch.setattr(prepare, "prepare_native_g1_beta_packet", lambda **_: pytest.fail("prepared again"))
    assert registration.ensure_site_registration(**kwargs) == {"status": "already_registered"}


def test_a_registered_packet_from_before_the_spawn_check_is_measured_on_the_next_pass(compiled_world, monkeypatch):
    from blueprint_pipeline import native_g1_spawn_clearance as spawn
    kwargs, _, _, _, _ = compiled_world
    assert registration.ensure_site_registration(**kwargs)["status"] == "registered"
    row = json.loads(overlay_path(kwargs["site_id"]).read_text())
    report = Path(row["screening_candidates"][DP]["native_g1"]["scene_packet"]["path"]).parent / spawn.FILENAME
    kept = json.loads(report.read_text())
    report.unlink()  # As a packet prepared before the check existed has none.
    monkeypatch.setattr(prepare, "prepare_native_g1_beta_packet", lambda **_: pytest.fail("prepared again"))
    assert registration.ensure_site_registration(**kwargs) == {"status": "already_registered"}
    measured = json.loads(report.read_text())
    assert measured["status"] == kept["status"] == "clear"
    assert measured["scene_plan_digest"] == kept["scene_plan_digest"]
    assert measured["stance_selection"]["source"] == "packet_robot_base_pose"
    # A measurement that fails leaves the row registered; planning then flags the packet as unmeasured.
    report.unlink()
    monkeypatch.setattr(spawn, "packet_spawn_clearance", lambda _packet: (_ for _ in ()).throw(OSError("injected")))
    assert registration.ensure_site_registration(**kwargs) == {"status": "already_registered"}
    assert not report.exists()



def test_a_missing_spawn_report_is_measured_only_while_consent_stands(compiled_world, monkeypatch):
    # A spawn report is new data derived from the site's scene: never written for a withdrawn site.
    from blueprint_pipeline import native_g1_spawn_clearance as spawn
    from blueprint_pipeline.team_scene_capture_root import TeamScenePacketRequestRefused
    kwargs, _, _, _, _ = compiled_world
    site = kwargs["site_id"]
    assert registration.ensure_site_registration(**kwargs)["status"] == "registered"
    row = json.loads(overlay_path(site).read_text())
    report = Path(row["screening_candidates"][DP]["native_g1"]["scene_packet"]["path"]).parent / spawn.FILENAME
    report.unlink()  # As a packet prepared before the check existed has none.
    monkeypatch.setattr(prepare, "prepare_native_g1_beta_packet", lambda **_: pytest.fail("prepared again"))
    monkeypatch.setattr(spawn, "packet_spawn_clearance", lambda _packet: pytest.fail("measured without consent"))
    # An unanswered check measures nothing and keeps the row; the next pass asks again.
    monkeypatch.setattr("blueprint_pipeline.beta_policy_screening_worker.require_current_source",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("unreachable")))
    assert registration.ensure_site_registration(**kwargs) == {"status": "already_registered"}
    assert overlay_path(site).exists() and not report.exists()
    # A refusal measures nothing, removes the row and keeps only the reason.
    refusal = "beta_screening_current_capture_consent_required"
    monkeypatch.setattr("blueprint_pipeline.beta_policy_screening_worker.require_current_source",
                        lambda *a, **k: (_ for _ in ()).throw(TeamScenePacketRequestRefused(refusal)))
    assert registration.ensure_site_registration(**kwargs) == {"status": "not_registered", "reason": refusal}
    assert not overlay_path(site).exists() and not report.exists()
    assert status_of(site)["reason"] == refusal


def test_a_current_root_row_is_measured_only_while_consent_stands(tmp_path, monkeypatch):
    from blueprint_pipeline import native_g1_spawn_clearance as spawn
    from blueprint_pipeline.team_scene_capture_root import TeamScenePacketRequestRefused
    retained, kwargs, _, _ = door_world(tmp_path, monkeypatch)
    installer.apply(retained.request, **retained.options)
    measured = []
    monkeypatch.setattr(spawn, "packet_spawn_clearance",
                        lambda packet: measured.append(packet) or (_ for _ in ()).throw(OSError("unreadable")))
    checks = []

    def consent(outcome):
        def read(value, *, site_id):
            checks.append(site_id)
            if outcome is not None:
                raise outcome
            return {}
        return read

    # Granted: the root row's packet, which has no report yet, is measured.
    monkeypatch.setattr("blueprint_pipeline.beta_policy_screening_worker.require_current_source", consent(None))
    assert registration.ensure_site_registration(**kwargs)["status"] == "root_registry_row_current"
    assert checks == [SITE] and len(measured) == 1
    # Unanswered: nothing measured and the row stays current.
    monkeypatch.setattr("blueprint_pipeline.beta_policy_screening_worker.require_current_source",
                        consent(OSError("unreachable")))
    assert registration.ensure_site_registration(**kwargs)["status"] == "root_registry_row_current"
    assert len(checks) == 2 and len(measured) == 1
    # Refused: nothing measured, the refusal is recorded and stands without another live read.
    refusal = "beta_screening_current_capture_consent_required"
    monkeypatch.setattr("blueprint_pipeline.beta_policy_screening_worker.require_current_source",
                        consent(TeamScenePacketRequestRefused(refusal)))
    assert registration.ensure_site_registration(**kwargs) == {"status": "not_registered", "reason": refusal}
    assert registration.ensure_site_registration(**kwargs)["reason"] == refusal
    assert len(checks) == 3 and len(measured) == 1

def test_a_changed_source_gets_a_new_packet_and_a_new_row(compiled_world):
    kwargs, profile, registry, _, capture = compiled_world
    registration.ensure_site_registration(**kwargs)
    first = json.loads(overlay_path(kwargs["site_id"]).read_text())
    context = json.loads((capture / "pipeline/website_task_context.json").read_text())
    context = seal({**context, "purpose": "corrected task brief"}, "context_digest")
    write(capture / "pipeline/website_task_context.json", context)
    changed = {**kwargs, "source": {**kwargs["source"], "task_context_digest": context["context_digest"]}}
    assert reader._site_screening_profile(registry, profile) == profile  # The old row is not current.
    assert registration.ensure_site_registration(**changed)["status"] == "registered"
    second = json.loads(overlay_path(kwargs["site_id"]).read_text())
    assert second["source"] == changed["source"] != first["source"]
    old, new = (row["screening_candidates"][DP]["native_g1"]["scene_packet"]["path"] for row in (first, second))
    assert old != new and Path(old).is_file() and Path(new).is_file()
    assert reader._site_screening_profile(registry, profile)["native_g1_registration_source"] == changed["source"]


def test_the_worker_registers_before_reading_the_profile_and_goes_on_when_registration_raises(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_worker as worker
    from blueprint_pipeline import team_candidate_quick10_registry as registry
    from tests.test_beta_policy_screening_worker import prepared_source_queue

    world = prepared_source_queue(tmp_path, monkeypatch)
    order = []
    read = registry.profile_for_capture
    monkeypatch.setattr(registry, "profile_for_capture", lambda path: order.append("profile") or read(path))
    calls = []

    def registers(**kwargs):
        calls.append(kwargs)
        order.append("registration")
        raise RuntimeError("development_only injected registration fault")

    monkeypatch.setattr(registration, "ensure_site_registration", registers)
    [adopted] = worker.enqueue_prepared_scenes(queue_root=world["screening_queue"],
                                               packet_queue_root=world["packet_queue"])["sources"]
    assert adopted["status"] == "enqueued"
    assert adopted["g1_registration"]["status"] == "failed"
    assert adopted["g1_registration"]["error"]["type"] == "builtins.RuntimeError"
    assert order[:2] == ["registration", "profile"]
    [call] = calls
    assert call["site_id"] == "site-job1" and call["source_queue"] == world["packet_queue"]
    plan = json.loads(next((world["screening_queue"] / "plans").glob("*.json")).read_text())
    assert call["source"] == plan["source"]
