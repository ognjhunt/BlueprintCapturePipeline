"""The guard reuses its catalog and residency phases only while nothing they read moved.

The Pub/Sub listener starts every minute and its ExecStartPre re-projected all
551 published profiles each time (12-28 s of CPU on 2026-11-02, about 860 s an
hour for that unit alone) to find the same consistent catalog. A ready verdict
from a pass that wrote nothing is reused while the profile directory, the
served catalog, every file and listing the pass consulted, the environment and
the code are unchanged; anything else runs the full pass.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import blueprint_pipeline.production_runtime_catalog_verdict as phase_verdict
import blueprint_pipeline.production_runtime_env_guard as guard
from blueprint_pipeline import host_resident_launch_inputs as residency_module
from blueprint_pipeline import task_evaluation_launch_catalog as catalog_module
from blueprint_pipeline.spend_authority_ledger_migration import SpendAuthorityLedgerError
from blueprint_pipeline.task_evaluation_launch_catalog import build_catalog_payload
from blueprint_pipeline.task_evaluation_launch_dispatcher import canonical_digest
from tests.test_task_evaluation_launch_catalog_reconciliation import _profile

ROOT = Path(__file__).resolve().parents[1]
_RESOLVE, _REALPATH = Path.resolve, os.path.realpath
PRODUCTION_FLAGS = {
    "BLUEPRINT_LAUNCH_PROOF_MODE": "production",
    "PRIVACY_PIPELINE_ENABLED": "true",
    "PRIVACY_FAIL_CLOSED": "true",
    "PIPELINE_SYNC_REQUIRED": "true",
    "RETRIEVAL_REQUIRE_PRIVACY_SAFE_VIDEO": "true",
}


class _Site:
    """Two published profiles, a consistent served catalog and a private verdict root."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.monkeypatch = monkeypatch
        self.profiles = root / "profiles"
        _profile(root, "profile-a")
        _profile(root, "profile-b")
        self.catalog = root / "catalog.json"
        self.catalog.write_bytes(build_catalog_payload(self.profiles))
        self.verdicts = root / "verdicts"
        for name, value in {
            **PRODUCTION_FLAGS,
            guard.LAUNCH_PROFILE_DIR_ENV: str(self.profiles),
            guard.LAUNCH_PUBLIC_CATALOG_PATH_ENV: str(self.catalog),
            residency_module.LAUNCH_INPUT_ROOTS_ENV: str(root),
            "BLUEPRINT_VALIDATION_VERDICT_ROOT": str(self.verdicts),
            # No provider-lock tree on this host: the slot probe creates nothing.
            "VAST_LAUNCH_LOCK_FILE": str(root / "no-lock-tree" / "vast_paid_launch.lock"),
        }.items():
            monkeypatch.setenv(name, value)
        monkeypatch.delenv("BLUEPRINT_PIPELINE_REPO", raising=False)
        self.catalog_passes = 0
        self.residency_passes = 0
        real_reconcile = catalog_module.reconcile_public_catalog
        real_residency = residency_module.published_profile_residency_report

        def counted_reconcile(**kwargs):
            self.catalog_passes += 1
            return real_reconcile(**kwargs)

        def counted_residency(*args, **kwargs):
            self.residency_passes += 1
            return real_residency(*args, **kwargs)

        monkeypatch.setattr(catalog_module, "reconcile_public_catalog", counted_reconcile)
        monkeypatch.setattr(residency_module, "published_profile_residency_report", counted_residency)
        self.ledger = lambda: {"status": "no_legacy_ledger"}

    def run(self) -> dict:
        # The unit's own start: no injected environment, catalog or residency check.
        return guard.build_production_runtime_env_guard(
            import_module=lambda _name: object(),
            reconcile_spend_authority=lambda: self.ledger(),
        )

    def entries(self) -> list[Path]:
        return sorted(self.verdicts.glob(f"{phase_verdict.VERDICT_NAME}-*.json"))

    def recorded(self) -> dict:
        report = self.run()
        assert report["status"] == "ready", report["blockers"]
        assert report["catalog_phase_verdict"]["recorded"] is True, report["catalog_phase_verdict"]
        return report


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Site:
    return _Site(tmp_path, monkeypatch)


def _reused(report: dict) -> bool:
    flags = {
        report["launch_profile_catalog"].get("verdict_reused"),
        report["launch_input_residency"].get("verdict_reused"),
    }
    assert len(flags) == 1, "the two phases are reused together or not at all"
    return flags.pop() is True


def test_an_unchanged_consistent_pass_is_reused_without_reprojecting(site: _Site) -> None:
    first = site.recorded()
    assert not _reused(first)
    assert (site.catalog_passes, site.residency_passes) == (1, 1)

    second = site.run()

    assert second["status"] == "ready" and second["blockers"] == []
    assert _reused(second)
    assert (site.catalog_passes, site.residency_passes) == (1, 1)
    assert second["catalog_phase_verdict"]["reused"] is True
    # The reused details are the ready ones the full pass produced.
    for phase in phase_verdict.PHASES:
        reused = {k: v for k, v in second[phase].items() if not k.startswith("verdict_")}
        computed = {k: v for k, v in first[phase].items() if not k.startswith("verdict_")}
        assert reused == computed
    assert second["launch_profile_catalog"]["status"] == "consistent"
    assert second["check_durations_seconds"]["launch_profile_catalog"] == 0.0
    assert second["check_durations_seconds"]["launch_input_residency"] == 0.0
    assert "catalog_phase_verdict" in second["check_durations_seconds"]


def _rewrite_identical(path: Path) -> None:
    data = path.read_bytes()
    time.sleep(0.01)
    path.unlink()
    path.write_bytes(data)


@pytest.mark.parametrize(
    "move",
    [
        "new_profile_published",
        "profile_replaced_with_identical_bytes",
        "catalog_replaced_with_identical_bytes",
        "catalog_row_dropped",
        "immutable_input_bytes_changed",
        "stray_file_in_profile_directory",
        "input_roots_environment_changed",
    ],
)
def test_any_moved_phase_input_reruns_both_phases(site: _Site, move: str) -> None:
    site.recorded()
    if move == "new_profile_published":
        _profile(site.root, "profile-c")
    elif move == "profile_replaced_with_identical_bytes":
        _rewrite_identical(site.profiles / "profile-a.json")
    elif move == "catalog_replaced_with_identical_bytes":
        _rewrite_identical(site.catalog)
    elif move == "catalog_row_dropped":
        rows = json.loads(site.catalog.read_text(encoding="utf-8"))
        site.catalog.write_text(json.dumps(rows[:1]), encoding="utf-8")
    elif move == "immutable_input_bytes_changed":
        (site.root / "profile-a-source.json").write_text('{"profile_id": "moved"}', encoding="utf-8")
    elif move == "stray_file_in_profile_directory":
        (site.profiles / ".publish-in-progress.tmp").write_text("", encoding="utf-8")
    else:
        site.monkeypatch.setenv(
            residency_module.LAUNCH_INPUT_ROOTS_ENV, f"{site.root}{os.pathsep}{site.profiles}"
        )

    report = site.run()

    assert not _reused(report)
    assert (site.catalog_passes, site.residency_passes) == (2, 2)
    if move in {"new_profile_published", "catalog_row_dropped", "immutable_input_bytes_changed"}:
        # The rerun is what keeps the served catalog equal to the directory.
        assert report["launch_profile_catalog"]["status"] == "repaired"
        assert site.catalog.read_bytes() == build_catalog_payload(site.profiles)


def test_a_probed_path_that_appears_reruns_the_phases(site: _Site) -> None:
    """Owner revocation is a stat of revoked.json: an absent path is an input too."""
    marker = site.root / "owner" / "revoked.json"
    marker.parent.mkdir()
    real_validate = catalog_module.validate_launch_profile

    def validate(profile):
        if profile["profile_id"] == "profile-a" and marker.exists():
            return ["scene_execution_owner_revoked"]
        return real_validate(profile)

    site.monkeypatch.setattr(catalog_module, "validate_launch_profile", validate)
    site.recorded()
    assert _reused(site.run())

    marker.write_text("{}", encoding="utf-8")
    report = site.run()

    assert not _reused(report)
    assert report["launch_profile_catalog"]["status"] == "repaired"
    rows = {row["profile_id"]: row for row in json.loads(site.catalog.read_text(encoding="utf-8"))}
    assert "scene_execution_owner_revoked" in rows["profile-a"]["execution_admission"]["blockers"]


def test_a_file_read_by_the_pass_that_changes_reruns_the_phases(site: _Site) -> None:
    owner_record = site.root / "owner" / "intent.json"
    owner_record.parent.mkdir()
    owner_record.write_text('{"window": "open"}', encoding="utf-8")
    real_validate = catalog_module.validate_launch_profile

    def validate(profile):
        state = json.loads(owner_record.read_text(encoding="utf-8"))
        if profile["profile_id"] == "profile-a" and state["window"] != "open":
            return ["scene_execution_owner_expired"]
        return real_validate(profile)

    site.monkeypatch.setattr(catalog_module, "validate_launch_profile", validate)
    site.recorded()
    assert _reused(site.run())

    owner_record.write_text('{"window": "lapsed"}', encoding="utf-8")
    report = site.run()

    assert not _reused(report)
    assert report["launch_profile_catalog"]["status"] == "repaired"


def _name_in_argv(site: _Site, profile_id: str, flag: str, path: Path) -> None:
    """Republish ``profile_id`` with ``flag path`` in its allocator argv."""
    published = site.profiles / f"{profile_id}.json"
    profile = json.loads(published.read_text(encoding="utf-8"))
    profile["allocator"]["argv"] += [flag, str(path)]
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    published.write_text(json.dumps(profile, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    site.catalog.write_bytes(build_catalog_payload(site.profiles))


def _replace(path: Path, data: bytes) -> None:
    """How the spend guard writes its report: a new file renamed over the old one."""
    staged = path.with_name(f".{path.name}.tmp")
    staged.write_bytes(data)
    os.replace(staged, path)


def test_a_state_file_named_in_argv_binds_where_it_resolves_not_its_bytes(
    site: _Site, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Live 2026-11-03: a profile passes the spend guard's ``latest.json`` as
    ``--preflight-bundle``, and the residency check resolves every absolute argv
    path. Bound by its full identity, each 1-2 minute rewrite moved the verdict:
    stored 23 times an hour, reused 5. Resolving reads only the entry's type and
    a link's target, so only a change to those may move it."""
    report = site.root / "gpu_spend_guard" / "latest.json"
    report.parent.mkdir()
    report.write_bytes(b'{"generated_at": "2026-11-03T03:06:00Z"}')
    _name_in_argv(site, "profile-a", "--preflight-bundle", report)
    site.recorded()

    for minute in range(3):
        time.sleep(0.01)
        _replace(report, json.dumps({"generated_at": f"2026-11-03T03:0{7 + minute}:00Z", "n": minute}).encode())
        again = site.run()
        assert _reused(again), again["catalog_phase_verdict"]
    assert (site.catalog_passes, site.residency_passes) == (1, 1)
    assert (Path.resolve, os.path.realpath) == (_RESOLVE, _REALPATH)

    # A link out of the control-plane roots is what residency refuses, so it reruns.
    elsewhere = tmp_path_factory.mktemp("authoring-machine") / "latest.json"
    elsewhere.write_bytes(report.read_bytes())
    report.unlink()
    report.symlink_to(elsewhere)

    moved = site.run()

    assert not _reused(moved)
    link = os.path.join(os.path.realpath(report.parent), report.name)
    assert moved["catalog_phase_verdict"]["miss_reason"] == f"resolution_changed:{link}"
    assert moved["launch_profile_catalog"]["status"] == "repaired"
    rows = {row["profile_id"]: row for row in json.loads(site.catalog.read_text(encoding="utf-8"))}
    assert "launch_profile_input_not_host_resident:allocator_argument:preflight-bundle" in (
        rows["profile-a"]["execution_admission"]["blockers"]
    )


def test_a_file_the_pass_resolves_and_then_reads_stays_bound_by_its_bytes(site: _Site) -> None:
    """Resolving a path first must not stand in for the read that follows it."""
    owner_record = site.root / "owner" / "intent.json"
    owner_record.parent.mkdir()
    owner_record.write_text('{"window": "open"}', encoding="utf-8")
    real_validate = catalog_module.validate_launch_profile

    def validate(profile):
        state = json.loads(owner_record.resolve().read_text(encoding="utf-8"))
        if profile["profile_id"] == "profile-a" and state["window"] != "open":
            return ["scene_execution_owner_expired"]
        return real_validate(profile)

    site.monkeypatch.setattr(catalog_module, "validate_launch_profile", validate)
    site.recorded()
    assert _reused(site.run())

    owner_record.write_text('{"window": "lapsed"}', encoding="utf-8")
    report = site.run()

    assert not _reused(report)
    assert report["catalog_phase_verdict"]["miss_reason"] == f"path_changed:{owner_record.resolve()}"
    assert report["launch_profile_catalog"]["status"] == "repaired"


def test_a_changed_package_source_reruns_the_phases(site: _Site, tmp_path: Path) -> None:
    """Code is an input: a deploy that edits a loaded module in place moves the verdict."""
    module = tmp_path / "loaded_module.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    real_sources = phase_verdict._package_sources
    site.monkeypatch.setattr(phase_verdict, "_package_sources", lambda: [*real_sources(), str(module)])
    # Each start is a fresh process that loaded the module as it is now.
    site.monkeypatch.setattr(phase_verdict, "_process_started_ns", time.time_ns)
    site.recorded()
    assert _reused(site.run())

    time.sleep(0.01)
    module.write_text("VALUE = 2\n", encoding="utf-8")
    report = site.run()

    assert not _reused(report)
    assert report["catalog_phase_verdict"]["miss_reason"] == f"code_changed:{module}"


def test_a_pass_after_the_environment_was_edited_is_not_recorded(site: _Site) -> None:
    """The import pass loads OpenCV, which edits LD_LIBRARY_PATH after the guard started."""

    def importing_edits_the_environment(_name: str) -> object:
        os.environ["LD_LIBRARY_PATH"] = "/opt/opencv/lib:"
        return object()

    site.monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    report = guard.build_production_runtime_env_guard(
        import_module=importing_edits_the_environment,
        reconcile_spend_authority=site.ledger,
    )

    assert report["status"] == "ready"
    assert report["catalog_phase_verdict"]["recorded"] is False
    assert report["catalog_phase_verdict"]["not_recorded_reason"] == "environment_changed_since_guard_start"
    assert site.entries() == []
    site.monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    assert site.recorded()["catalog_phase_verdict"]["recorded"] is True


def test_a_grant_that_lapses_is_never_served_from_the_cache(site: _Site) -> None:
    """Review of #3278: a lapsed visual-review grant blocks a full pass.

    Its blocker is not an inactive owner state, so the catalog refuses and every
    guarded unit stops. A ready verdict stored before the expiry must not let
    them start after it: the pass reports the expiry it compared with the clock.
    """
    from blueprint_pipeline.decision_deadlines import note_deadline

    clock = {"now": time.time()}
    expires = clock["now"] + 600
    grant = site.root / "owner" / "visual-review-correction.json"
    grant.parent.mkdir()
    grant.write_text(json.dumps({"expires_at_epoch": expires}), encoding="utf-8")
    real_validate = catalog_module.validate_launch_profile

    def validate(profile):
        if profile["profile_id"] == "profile-a":
            value = json.loads(grant.read_text(encoding="utf-8"))
            if not clock["now"] < note_deadline(value["expires_at_epoch"]):
                return ["scene_execution_visual_review_correction_invalid"]
        return real_validate(profile)

    site.monkeypatch.setattr(catalog_module, "validate_launch_profile", validate)
    site.monkeypatch.setattr(phase_verdict, "_now", lambda: clock["now"])
    site.recorded()

    clock["now"] = expires - phase_verdict.DEADLINE_MARGIN_SECONDS - 1
    before = site.run()
    assert before["status"] == "ready" and _reused(before)

    clock["now"] = expires - phase_verdict.DEADLINE_MARGIN_SECONDS
    margin = site.run()
    assert not _reused(margin) and margin["status"] == "ready"
    assert margin["catalog_phase_verdict"]["miss_reason"] == "decision_deadline_reached"

    clock["now"] = expires
    after = site.run()
    assert not _reused(after)
    assert after["status"] == "blocked"
    assert any(b.startswith("launch_profile_catalog_not_reconciled") for b in after["blockers"])


def test_a_pass_that_reads_the_clock_without_reporting_is_never_recorded(site: _Site) -> None:
    """A time check that reports no instant could flip unseen, so its pass is not cached."""
    unreported = {"__name__": "blueprint_pipeline.unreported_expiry_check", "time": time}
    exec("def still_valid():\n    return time.time() < 4e9\n", unreported)
    real_validate = catalog_module.validate_launch_profile

    def validate(profile):
        assert unreported["still_valid"]()
        return real_validate(profile)

    site.monkeypatch.setattr(catalog_module, "validate_launch_profile", validate)

    report = site.run()

    assert report["status"] == "ready"
    assert report["catalog_phase_verdict"]["recorded"] is False
    assert report["catalog_phase_verdict"]["not_recorded_reason"] == (
        "unrecordable:clock:blueprint_pipeline.unreported_expiry_check.still_valid"
    )
    assert time.time is phase_verdict._TIME and time.time_ns is phase_verdict._TIME_NS


def test_every_permitted_clock_reader_is_real_and_reports_or_only_stamps() -> None:
    """The permitted readers are named by function: a rename must not silently widen them."""
    import importlib
    import inspect

    for site_name in phase_verdict._CLOCK_SITES:
        module_name, function_name = site_name.rsplit(".", 1)
        source = inspect.getsource(getattr(importlib.import_module(module_name), function_name))
        assert "time.time()" in source, site_name
        assert "note_deadline(" in source or "created_at_epoch" in source, site_name


def test_a_repaired_catalog_is_recorded_only_once_a_pass_finds_it_consistent(site: _Site) -> None:
    site.catalog.unlink()

    repaired = site.run()
    assert repaired["launch_profile_catalog"]["status"] == "repaired"
    assert repaired["catalog_phase_verdict"]["recorded"] is False
    assert "catalog_repaired" in repaired["catalog_phase_verdict"]["not_recorded_reason"]
    assert site.entries() == []

    consistent = site.run()
    assert consistent["launch_profile_catalog"]["status"] == "consistent"
    assert consistent["catalog_phase_verdict"]["recorded"] is True

    assert _reused(site.run())


def test_a_blocked_guard_never_records_or_reuses_a_verdict(site: _Site) -> None:
    def stranded():
        raise SpendAuthorityLedgerError("legacy_ledger_unreadable:/old/root")

    site.ledger = stranded
    blocked = site.run()
    assert blocked["status"] == "blocked"
    assert blocked["catalog_phase_verdict"]["recorded"] is False
    assert site.entries() == []

    # A refusal was never stored, so the next start proves the phases again.
    site.ledger = lambda: {"status": "no_legacy_ledger"}
    ready = site.run()
    assert not _reused(ready)
    assert ready["catalog_phase_verdict"]["recorded"] is True


def test_a_catalog_that_cannot_be_projected_is_never_recorded(site: _Site) -> None:
    (site.profiles / "broken.json").write_text("[]", encoding="utf-8")

    for _ in range(2):
        report = site.run()
        assert report["status"] == "blocked"
        assert any(b.startswith("launch_profile_catalog_not_reconciled") for b in report["blockers"])
        assert report["catalog_phase_verdict"]["recorded"] is False
    assert site.entries() == []


def test_an_input_that_moves_during_the_pass_is_not_recorded(site: _Site) -> None:
    """The bytes the pass judged are not the bytes a later start would find."""
    source = site.root / "profile-b-source.json"
    original = source.read_bytes()
    moved = original.replace(b"profile-b", b"profile-x")
    assert len(moved) == len(original)
    # Opened before the pass, so the late write raises no audit event of its own:
    # only the identity re-check at record time can see it.
    handle = source.open("r+b")
    real_reconcile = catalog_module.reconcile_public_catalog

    def reconcile_then_move(**kwargs):
        result = real_reconcile(**kwargs)
        time.sleep(0.01)
        handle.seek(0)
        handle.write(moved)
        handle.flush()
        return result

    site.monkeypatch.setattr(catalog_module, "reconcile_public_catalog", reconcile_then_move)
    try:
        report = site.run()
    finally:
        handle.close()

    assert report["catalog_phase_verdict"]["recorded"] is False
    assert "changed_during_pass" in report["catalog_phase_verdict"]["not_recorded_reason"]
    assert site.entries() == []


@pytest.mark.parametrize("effect", ["write_outside_the_verdict_store", "spawn_a_process"])
def test_a_pass_with_effects_no_fingerprint_holds_is_not_recorded(site: _Site, effect: str) -> None:
    real_validate = catalog_module.validate_launch_profile

    def validate(profile):
        if effect == "write_outside_the_verdict_store":
            (site.root / "side-effect.json").write_text("{}", encoding="utf-8")
        else:
            subprocess.run(["/bin/sh", "-c", "exit 0"], check=True)
        return real_validate(profile)

    site.monkeypatch.setattr(catalog_module, "validate_launch_profile", validate)

    report = site.run()

    assert report["status"] == "ready"
    assert report["catalog_phase_verdict"]["recorded"] is False
    assert site.entries() == []


def _tamper(site: _Site, how: str) -> None:
    (entry,) = site.entries()
    if how == "symlinked":
        copy = site.root / "copied-entry.json"
        copy.write_bytes(entry.read_bytes())
        copy.chmod(0o640)
        entry.unlink()
        entry.symlink_to(copy)
    elif how == "group_writable":
        entry.chmod(0o660)
    elif how == "world_writable":
        entry.chmod(0o646)
    elif how == "foreign_owned":
        owner = entry.stat().st_uid
        site.monkeypatch.setattr(phase_verdict, "_euid", lambda: owner + 1)
    elif how == "malformed":
        entry.write_text("{not json", encoding="utf-8")
    elif how == "older_than_the_bound":
        later = time.time() + phase_verdict.MAX_VERDICT_AGE_SECONDS + 1
        site.monkeypatch.setattr(phase_verdict, "_now", lambda: later)
    elif how == "stored_in_the_future":
        earlier = time.time() - 120
        site.monkeypatch.setattr(phase_verdict, "_now", lambda: earlier)
    elif how == "unreadable":
        entry.chmod(0o000)
    else:
        raise AssertionError(how)


@pytest.mark.parametrize(
    "how",
    [
        "symlinked",
        "group_writable",
        "world_writable",
        "foreign_owned",
        "malformed",
        "older_than_the_bound",
        "stored_in_the_future",
        "unreadable",
    ],
)
def test_an_untrustworthy_verdict_entry_runs_the_full_pass(site: _Site, how: str) -> None:
    if how == "unreadable" and os.geteuid() == 0:
        pytest.skip("root reads a mode-0000 file")
    site.recorded()
    _tamper(site, how)

    report = site.run()

    assert report["status"] == "ready"
    assert not _reused(report)
    assert (site.catalog_passes, site.residency_passes) == (2, 2)
    assert report["catalog_phase_verdict"]["miss_reason"]


def test_the_bound_is_a_quarter_hour() -> None:
    # Owner windows lapse with no file changing; the dispatcher re-checks them at
    # dispatch, and the served catalog may lag a lapse by at most this bound.
    assert phase_verdict.MAX_VERDICT_AGE_SECONDS == 15 * 60


@pytest.mark.parametrize("broken", ["_reuse", "_record", "_identity", "_resolution_identity"])
def test_a_broken_verdict_layer_falls_back_to_the_full_pass(
    site: _Site, broken: str
) -> None:
    """A cache can only make the guard slower, never stop a unit from starting."""
    real = getattr(phase_verdict, broken)
    calls: list[str] = []

    def explode(*_args, **_kwargs):
        calls.append(broken)
        raise RuntimeError("verdict layer broken")

    if broken == "_reuse":
        site.recorded()  # a usable verdict exists; only reading it is broken
    site.monkeypatch.setattr(phase_verdict, broken, explode)

    for _ in range(2):
        report = site.run()
        assert report["status"] == "ready" and report["blockers"] == []
        assert not _reused(report)
        assert report["launch_profile_catalog"]["status"] == "consistent"
    assert calls and callable(real)


def test_injected_checks_never_read_or_write_the_verdict(site: _Site) -> None:
    site.recorded()
    stored = site.entries()

    report = guard.build_production_runtime_env_guard(
        import_module=lambda _name: object(),
        reconcile_spend_authority=site.ledger,
        reconcile_launch_catalog=lambda: {"status": "consistent"},
    )

    assert report["launch_profile_catalog"] == {"status": "consistent"}
    assert "catalog_phase_verdict" not in report
    assert site.entries() == stored

    explicit = guard.build_production_runtime_env_guard(
        env=dict(os.environ), import_module=lambda _name: object(),
        reconcile_spend_authority=site.ledger,
    )
    assert "catalog_phase_verdict" not in explicit
    assert explicit["launch_profile_catalog"]["status"] == "consistent"


def test_retained_verdicts_stay_bounded_to_the_newest(site: _Site) -> None:
    for index in range(phase_verdict.RETAINED_VERDICTS + 3):
        site.monkeypatch.setenv("BLUEPRINT_PROBE_UNIT_ENVIRONMENT", str(index))
        site.recorded()

    assert len(site.entries()) == phase_verdict.RETAINED_VERDICTS
    assert _reused(site.run())


def test_importing_the_guard_leaves_the_catalog_projection_unloaded() -> None:
    """A reused verdict must not pay for importing the dispatcher's module graph."""
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "import blueprint_pipeline.production_runtime_env_guard;"
        "loaded = sorted(m for m in ('blueprint_pipeline.task_evaluation_launch_catalog',"
        " 'blueprint_pipeline.task_evaluation_launch_dispatcher',"
        " 'blueprint_pipeline.host_resident_launch_inputs') if m in sys.modules);"
        "assert not loaded, loaded"
    )
    child = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(ROOT / "src")],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert child.returncode == 0, child.stderr
