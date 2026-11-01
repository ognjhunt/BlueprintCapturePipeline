"""A catalog written by older code is never repaired by fixing the writer.

PR #454 made the publisher build the WebApp catalog from the profile directory
instead of from one invocation's arguments. That fixes every catalog written
*after* it deploys and none of the ones already on disk: the catalog is a
derived artifact, and nothing recomputed it.

Observed on the live control plane after #454 deployed -- seven profiles in the
directory, four in the served catalog, and the three missing ones invisible to
any launch. The repair was a manual publish, which is precisely the kind of
remembered step that does not survive a host rebuild.

So drift is reconciled at start-up, from the directory, every time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_launch_catalog import (
    LaunchCatalogError,
    build_catalog_payload,
    reconcile_public_catalog,
)
from blueprint_pipeline.task_evaluation_launch_dispatcher import canonical_digest

COMMIT = "0" * 40


def _profile(tmp_path: Path, profile_id: str) -> Path:
    """A minimal profile that passes both fail-closed validators."""
    manifest = tmp_path / f"{profile_id}-source.json"
    manifest.write_text(json.dumps({"profile_id": profile_id}), encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()
    uri = f"https://raw.githubusercontent.com/example/repo/{COMMIT}/{profile_id}.json"
    run_root = "{launch_run_root}"
    allocator = "python -m blueprint_pipeline.paid_resource_allocator gpu-canary"
    profile = {
        "schema_version": "task_evaluation_launch_profile.v1",
        "profile_id": profile_id,
        "program_id": "arm-decision-proof-v1",
        "claim_ceiling": "development_only",
        "allocator": {
            "entrypoint": allocator,
            "subcommand": "gpu-canary",
            "argv": ["--adapter-output", f"{run_root}/allocator/result.json"],
            "max_spend_usd": 6.0,
            "hard_ttl_seconds": 5400,
            "retry_cap": 0,
        },
        "execution_admission": {
            "live_enabled": False,
            "blockers": ["development_only"],
            "readiness_receipt": {"uri": uri, "digest": digest},
        },
        "evaluation_run_spec": {"uri": uri, "digest": digest},
        "source_bundle": {
            "bundle_id": profile_id,
            "source_kind": "interiorgs_sage",
            "uri": uri,
            "digest": digest,
        },
        "immutable_inputs": [
            {"name": "source_bundle_manifest", "path": str(manifest), "digest": digest},
            {"name": "evaluation_run_spec", "path": str(manifest), "digest": digest},
        ],
        "reconciliation": {"max_guard_age_seconds": 300, "required_providers": ["vast"]},
        "required_controls": {
            "canonical_allocator": allocator,
            "secret_profile_id": "canonical-vast-adp",
            "watchdog_required": True,
            "artifact_storage_required": True,
            "teardown_required": True,
            "provider_zero_required": True,
            "webapp_status_sync_required": True,
            "retry_cap": 0,
        },
        "runtime_environment": {},
        "terminal_contract": {
            "result_path": f"{run_root}/allocator/result.json",
            "success_statuses": ["completed"],
            "required_values": {"continuing_spend_from_this_run": False, "retry_cap": 0},
            "required_path_fields": ["teardown_manifest_path", "artifact_manifest_path"],
        },
        "webapp_sync": {"max_attempts": 20},
    }
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    path = profile_dir / f"{profile_id}.json"
    path.write_text(
        json.dumps(profile, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return path


def _catalog_ids(catalog: Path) -> set[str]:
    return {str(row["profile_id"]) for row in json.loads(catalog.read_text(encoding="utf-8"))}


def test_repairs_a_catalog_that_omits_published_profiles(tmp_path: Path) -> None:
    """The exact production condition: 7 in the directory, 4 in the catalog."""
    for index in range(7):
        _profile(tmp_path, f"profile-{index}")
    catalog = tmp_path / "catalog.json"
    stale = json.loads(build_catalog_payload(tmp_path / "profiles").decode())[:4]
    catalog.write_text(json.dumps(stale, separators=(",", ":")), encoding="utf-8")

    receipt = reconcile_public_catalog(
        profile_dir=tmp_path / "profiles", catalog_path=catalog
    )

    assert receipt["status"] == "repaired"
    assert receipt["profile_ids_added"] == ["profile-4", "profile-5", "profile-6"]
    assert len(_catalog_ids(catalog)) == 7


def test_a_catalog_already_matching_the_directory_is_left_untouched(
    tmp_path: Path,
) -> None:
    _profile(tmp_path, "profile-alpha")
    catalog = tmp_path / "catalog.json"
    catalog.write_bytes(build_catalog_payload(tmp_path / "profiles"))
    before = catalog.stat().st_mtime_ns

    receipt = reconcile_public_catalog(
        profile_dir=tmp_path / "profiles", catalog_path=catalog
    )

    assert receipt["status"] == "consistent"
    assert receipt["profile_ids_added"] == []
    assert catalog.stat().st_mtime_ns == before


def test_creates_a_missing_catalog_from_the_directory(tmp_path: Path) -> None:
    """A host rebuilt with a restored profile directory has no catalog yet."""
    _profile(tmp_path, "profile-alpha")
    catalog = tmp_path / "nested" / "catalog.json"

    receipt = reconcile_public_catalog(
        profile_dir=tmp_path / "profiles", catalog_path=catalog
    )

    assert receipt["status"] == "repaired"
    assert _catalog_ids(catalog) == {"profile-alpha"}


def test_drops_a_catalog_entry_with_no_profile_behind_it(tmp_path: Path) -> None:
    """A descriptor the WebApp can select but no profile can serve is worse."""
    _profile(tmp_path, "profile-alpha")
    catalog = tmp_path / "catalog.json"
    rows = json.loads(build_catalog_payload(tmp_path / "profiles").decode())
    rows.append({**rows[0], "profile_id": "profile-vanished"})
    catalog.write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")

    receipt = reconcile_public_catalog(
        profile_dir=tmp_path / "profiles", catalog_path=catalog
    )

    assert receipt["profile_ids_removed"] == ["profile-vanished"]
    assert _catalog_ids(catalog) == {"profile-alpha"}


def test_an_invalid_profile_in_the_directory_fails_closed(tmp_path: Path) -> None:
    """Refusing beats emitting a catalog that silently omits a published file."""
    _profile(tmp_path, "profile-alpha")
    (tmp_path / "profiles" / "broken.json").write_text("{}", encoding="utf-8")

    with pytest.raises(LaunchCatalogError, match="published_profile_invalid"):
        reconcile_public_catalog(
            profile_dir=tmp_path / "profiles", catalog_path=tmp_path / "catalog.json"
        )


def test_an_unparseable_catalog_is_rebuilt_rather_than_trusted(tmp_path: Path) -> None:
    _profile(tmp_path, "profile-alpha")
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{not json", encoding="utf-8")

    receipt = reconcile_public_catalog(
        profile_dir=tmp_path / "profiles", catalog_path=catalog
    )

    assert receipt["status"] == "repaired"
    assert _catalog_ids(catalog) == {"profile-alpha"}


def test_a_missing_profile_directory_fails_closed(tmp_path: Path) -> None:
    """An empty catalog is a valid document, so it cannot mean 'no directory'."""
    with pytest.raises(LaunchCatalogError, match="profile_dir_missing"):
        reconcile_public_catalog(
            profile_dir=tmp_path / "absent", catalog_path=tmp_path / "catalog.json"
        )


def test_reconciliation_is_idempotent(tmp_path: Path) -> None:
    _profile(tmp_path, "profile-alpha")
    catalog = tmp_path / "catalog.json"

    first = reconcile_public_catalog(
        profile_dir=tmp_path / "profiles", catalog_path=catalog
    )
    second = reconcile_public_catalog(
        profile_dir=tmp_path / "profiles", catalog_path=catalog
    )

    assert first["status"] == "repaired"
    assert second["status"] == "consistent"


def test_the_catalog_still_carries_no_allocator_arguments(tmp_path: Path) -> None:
    """Widening the catalog must not widen what it discloses."""
    _profile(tmp_path, "profile-alpha")
    catalog = tmp_path / "catalog.json"

    reconcile_public_catalog(profile_dir=tmp_path / "profiles", catalog_path=catalog)

    text = catalog.read_text(encoding="utf-8")
    assert "argv" not in text
    assert "--adapter-output" not in text


def test_a_symlinked_profile_fails_closed(tmp_path: Path) -> None:
    """Published profiles are immutable; a symlink defeats that."""
    real = _profile(tmp_path, "profile-alpha")
    (tmp_path / "profiles" / "profile-link.json").symlink_to(real)

    with pytest.raises(LaunchCatalogError, match="launch_profile_source_invalid"):
        reconcile_public_catalog(
            profile_dir=tmp_path / "profiles", catalog_path=tmp_path / "catalog.json"
        )


@pytest.mark.parametrize("inactive_blocker", ["scene_execution_owner_revoked",
                                              "scene_execution_owner_expired",
                                              "scene_execution_owner_attempt_cancelled_before_execution",
                                              "scene_execution_owner_store_missing"])
@pytest.mark.parametrize("extra_blocker", [None, "scene_execution_owner_record_mismatch"])
def test_revoked_owner_profile_is_retained_as_non_runnable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra_blocker: str | None, inactive_blocker: str
) -> None:
    from blueprint_pipeline import task_evaluation_launch_catalog as catalog_module

    revoked_path = _profile(tmp_path, "revoked-profile")
    _profile(tmp_path, "other-profile")
    original = revoked_path.read_bytes()
    real_validate = catalog_module.validate_launch_profile

    def validate(profile):
        if profile["profile_id"] == "revoked-profile":
            return [inactive_blocker, *([extra_blocker] if extra_blocker else [])]
        return real_validate(profile)

    monkeypatch.setattr(catalog_module, "validate_launch_profile", validate)
    if extra_blocker:
        with pytest.raises(LaunchCatalogError, match=extra_blocker):
            build_catalog_payload(tmp_path / "profiles")
    else:
        rows = {row["profile_id"]: row for row in json.loads(build_catalog_payload(tmp_path / "profiles"))}
        assert set(rows) == {"revoked-profile", "other-profile"}
        admission = rows["revoked-profile"]["execution_admission"]
        assert admission["live_enabled"] is False
        assert inactive_blocker in admission["blockers"]
    assert revoked_path.read_bytes() == original


def test_one_unreadable_owner_store_cannot_starve_the_whole_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scene 840938, 2026-09-16: one retired source blocked every launch.

    A single stale profile carried scene_execution_owner_store_missing. The
    builder raised, the catalog never reconciled, and launch activation
    starved -- a healthy retargeted run sat at awaiting_execution for hours
    with nothing wrong with it. An owner record this host cannot read is
    unavailability, not corrupted evidence, so it demotes like a missing input.
    """
    from blueprint_pipeline import task_evaluation_launch_catalog as catalog_module

    _profile(tmp_path, "retired-source-profile")
    _profile(tmp_path, "live-profile")
    real_validate = catalog_module.validate_launch_profile

    def validate(profile):
        if profile["profile_id"] == "retired-source-profile":
            return ["scene_execution_owner_store_missing"]
        return real_validate(profile)

    monkeypatch.setattr(catalog_module, "validate_launch_profile", validate)
    rows = {row["profile_id"]: row for row in json.loads(build_catalog_payload(tmp_path / "profiles"))}

    # The catalog builds at all -- that is the whole point. Before this change
    # the single stale profile raised and no catalog was produced, so every
    # other profile became unreachable and activation had nothing to read.
    assert set(rows) == {"retired-source-profile", "live-profile"}
    assert "scene_execution_owner_store_missing" not in (
        rows["live-profile"]["execution_admission"]["blockers"]
    )
    # The stale one is retained as evidence but cannot start a run.
    demoted = rows["retired-source-profile"]["execution_admission"]
    assert demoted["live_enabled"] is False
    assert "scene_execution_owner_store_missing" in demoted["blockers"]


def test_a_profile_whose_inputs_are_missing_is_demoted_not_fatal(tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    """Raising here is a total outage on exactly the host that most needs to
    keep accepting launches. A host restored from an image has none of the
    inputs a previous host accumulated, and the catalog reconciler is an
    ExecStartPre for the intake service."""

    import json as _json

    from blueprint_pipeline.task_evaluation_launch_catalog import build_catalog_payload
    from test_task_evaluation_launch_dispatcher import _profile, _write

    profile_dir = tmp_path / "profiles"
    intact = _profile(tmp_path)
    stale = _json.loads(_json.dumps(intact))
    stale["profile_id"] = "inputs-since-deleted"
    # Same declared inputs, one of them no longer on this host -- the shape a
    # rebuilt host produces, not a malformed document.
    stale["immutable_inputs"] = [dict(row) for row in intact["immutable_inputs"]]
    stale["immutable_inputs"][0]["path"] = str(tmp_path / "deleted-by-a-host-rebuild.json")
    from blueprint_pipeline.task_evaluation_launch_dispatcher import canonical_digest

    stale["profile_digest"] = canonical_digest(stale, digest_field="profile_digest")
    _write(profile_dir / f"{intact['profile_id']}.json", intact)
    _write(profile_dir / f"{stale['profile_id']}.json", stale)

    rows = {
        row["profile_id"]: row
        for row in _json.loads(build_catalog_payload(profile_dir).decode("utf-8"))
    }

    # The whole catalog still projects.
    assert set(rows) == {intact["profile_id"], "inputs-since-deleted"}
    demoted = rows["inputs-since-deleted"]["execution_admission"]
    assert demoted["live_enabled"] is False
    assert any("immutable_input_missing" in b for b in demoted["blockers"])
    assert rows[intact["profile_id"]]["execution_admission"]["live_enabled"] is True


def test_a_malformed_profile_still_fails_the_whole_catalog(tmp_path):
    """A document that cannot be projected is a different problem from a host
    that does not have the bytes it names."""

    import pytest as _pytest

    from blueprint_pipeline.task_evaluation_launch_catalog import (
        LaunchCatalogError,
        build_catalog_payload,
    )
    from test_task_evaluation_launch_dispatcher import _profile, _write

    profile_dir = tmp_path / "profiles"
    broken = _profile(tmp_path)
    broken.pop("required_controls")
    _write(profile_dir / "broken.json", broken)

    with _pytest.raises(LaunchCatalogError):
        build_catalog_payload(profile_dir)


def test_repair_works_on_the_read_only_catalog_publication_installs(
    tmp_path: Path,
) -> None:
    """Publication installs the catalog 0440 and the reconciler must still fix drift.

    ``publish_task_evaluation_launch_profiles`` seals the catalog read-only so
    no consumer can edit it in place. Its owner cannot reopen 0440 for writing
    either, so an in-place rewrite raises ``catalog_unwritable`` -- and since
    this reconciler runs from the intake unit's ExecStartPre, that stops intake
    from starting at all. Repair must replace the inode and preserve the mode.
    """

    import os
    import stat

    _profile(tmp_path, "profile-a")
    profile_dir = tmp_path / "profiles"
    catalog = tmp_path / "catalog.json"
    catalog.write_text("[]\n", encoding="utf-8")
    os.chmod(catalog, 0o440)

    result = reconcile_public_catalog(
        profile_dir=profile_dir, catalog_path=catalog
    )

    assert result["status"] == "repaired"
    assert "profile-a" in result["profile_ids_added"]
    assert stat.S_IMODE(catalog.stat().st_mode) == 0o440
    assert json.loads(catalog.read_text(encoding="utf-8"))
    assert not list(catalog.parent.glob(".catalog.json.*.tmp"))


def test_each_distinct_input_is_checked_once_per_pass(tmp_path, monkeypatch):
    """2026-10-12: the 539 published profiles list 26,015 inputs that are 3,784 distinct files.
    Checking every reference made one guard start digest the same file up to 186 times and
    resolve every bound path once per profile naming it."""
    from blueprint_pipeline import host_resident_launch_inputs as residency
    from blueprint_pipeline import launch_profile_immutable_inputs as inputs

    monkeypatch.setenv(residency.LAUNCH_INPUT_ROOTS_ENV, str(tmp_path))
    for index in range(3):
        _profile(tmp_path, f"profile-{index}")  # one manifest, named by two input roles
    digested: list[str] = []
    resolved: list[str] = []
    real_digest = inputs.immutable_input_digest
    real_resident = residency.path_is_host_resident
    monkeypatch.setattr(
        inputs, "immutable_input_digest", lambda path: digested.append(path.name) or real_digest(path)
    )
    monkeypatch.setattr(
        residency,
        "path_is_host_resident",
        lambda path, roots: resolved.append(Path(path).name) or real_resident(path, roots),
    )
    manifests = [f"profile-{index}-source.json" for index in range(3)]

    first = build_catalog_payload(tmp_path / "profiles")

    assert sorted(digested) == manifests
    assert sorted(resolved) == manifests
    digested.clear()
    assert build_catalog_payload(tmp_path / "profiles") == first
    assert sorted(digested) == manifests  # nothing carries over to the next pass


def test_each_profile_is_validated_once_per_pass(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_launch_dispatcher as dispatcher

    _profile(tmp_path, "profile-0")
    _profile(tmp_path, "profile-1")
    (tmp_path / "profile-1-source.json").unlink()  # inputs gone: demoted after the structural pass
    passes: list[tuple[str, bool]] = []
    real = dispatcher._validate_launch_profile

    def counted(value, *, reopen_external_lineage):
        passes.append((value["profile_id"], reopen_external_lineage))
        return real(value, reopen_external_lineage=reopen_external_lineage)

    monkeypatch.setattr(dispatcher, "_validate_launch_profile", counted)

    rows = {row["profile_id"]: row for row in json.loads(build_catalog_payload(tmp_path / "profiles"))}

    assert passes == [("profile-0", False), ("profile-0", True), ("profile-1", False)]
    assert "launch_profile_immutable_input_missing:source_bundle_manifest" in (
        rows["profile-1"]["execution_admission"]["blockers"]
    )


# A unit whose environment names the scene-intake owner store reaches the real owner
# state; one without BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT can only report that
# it has no store to read.
FULL_VIEW: list[str] = []
NO_OWNER_STORE = ["scene_execution_owner_store_missing"]


def _owner_view(monkeypatch: pytest.MonkeyPatch, blockers: list[str]) -> None:
    from blueprint_pipeline import task_evaluation_launch_catalog as catalog_module
    from blueprint_pipeline.task_evaluation_launch_dispatcher import validate_launch_profile

    monkeypatch.setattr(
        catalog_module,
        "validate_launch_profile",
        lambda profile: list(blockers) if profile["profile_id"] == "owner-profile" else validate_launch_profile(profile),
    )


def _rows(catalog: Path) -> dict[str, dict]:
    return {row["profile_id"]: row for row in json.loads(catalog.read_text(encoding="utf-8"))}


def test_a_unit_without_the_owner_store_never_overwrites_an_owner_judgement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2026-10-12: the served catalog flipped between two versions every few minutes.

    Every guarded unit reconciles the catalog. The control-plane one-shot does not
    load task-evaluation-scene-progression.env, so its guard demoted each owner-bound
    profile as scene_execution_owner_store_missing; the intake and listener guards
    then wrote the real owner state back.
    """
    _profile(tmp_path, "owner-profile")
    _profile(tmp_path, "plain-profile")
    catalog = tmp_path / "catalog.json"
    _owner_view(monkeypatch, FULL_VIEW)
    reconcile_public_catalog(profile_dir=tmp_path / "profiles", catalog_path=catalog)
    judged = catalog.read_bytes()

    _owner_view(monkeypatch, NO_OWNER_STORE)
    receipt = reconcile_public_catalog(profile_dir=tmp_path / "profiles", catalog_path=catalog)

    assert receipt["status"] == "consistent"
    assert receipt["owner_state_carried_forward"] == 1
    assert catalog.read_bytes() == judged


def test_a_unit_without_the_owner_store_still_repairs_what_it_can_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _profile(tmp_path, "owner-profile")
    _profile(tmp_path, "plain-profile")
    catalog = tmp_path / "catalog.json"
    _owner_view(monkeypatch, FULL_VIEW)
    reconcile_public_catalog(profile_dir=tmp_path / "profiles", catalog_path=catalog)
    judged = _rows(catalog)["owner-profile"]
    _profile(tmp_path, "new-profile")  # published after the last full-view reconcile

    _owner_view(monkeypatch, NO_OWNER_STORE)
    receipt = reconcile_public_catalog(profile_dir=tmp_path / "profiles", catalog_path=catalog)

    assert receipt["status"] == "repaired"
    assert receipt["profile_ids_added"] == ["new-profile"]
    assert _rows(catalog)["owner-profile"] == judged


def test_an_owner_profile_no_full_view_has_judged_is_served_demoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing to carry forward: a missing catalog, or a profile republished with new bytes."""
    path = _profile(tmp_path, "owner-profile")
    catalog = tmp_path / "catalog.json"
    _owner_view(monkeypatch, NO_OWNER_STORE)
    assert reconcile_public_catalog(profile_dir=tmp_path / "profiles", catalog_path=catalog)["status"] == "repaired"
    assert "scene_execution_owner_store_missing" in _rows(catalog)["owner-profile"]["execution_admission"]["blockers"]

    _owner_view(monkeypatch, FULL_VIEW)
    reconcile_public_catalog(profile_dir=tmp_path / "profiles", catalog_path=catalog)
    assert "scene_execution_owner_store_missing" not in (
        _rows(catalog)["owner-profile"]["execution_admission"]["blockers"]
    )
    profile = json.loads(path.read_text(encoding="utf-8"))
    profile["allocator"]["max_spend_usd"] = 5.0  # republished: a judgement of the old bytes no longer applies
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    path.write_text(json.dumps(profile, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

    _owner_view(monkeypatch, NO_OWNER_STORE)
    receipt = reconcile_public_catalog(profile_dir=tmp_path / "profiles", catalog_path=catalog)

    assert receipt["owner_state_carried_forward"] == 0
    row = _rows(catalog)["owner-profile"]
    assert row["profile_digest"] == profile["profile_digest"]
    assert row["execution_admission"]["live_enabled"] is False
    assert "scene_execution_owner_store_missing" in row["execution_admission"]["blockers"]


def test_an_invalid_served_row_is_never_carried_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _profile(tmp_path, "owner-profile")
    catalog = tmp_path / "catalog.json"
    _owner_view(monkeypatch, FULL_VIEW)
    reconcile_public_catalog(profile_dir=tmp_path / "profiles", catalog_path=catalog)
    rows = json.loads(catalog.read_text(encoding="utf-8"))
    rows[0]["execution_admission"]["live_enabled"] = True  # live while still carrying blockers
    catalog.write_text(json.dumps(rows), encoding="utf-8")

    _owner_view(monkeypatch, NO_OWNER_STORE)
    receipt = reconcile_public_catalog(profile_dir=tmp_path / "profiles", catalog_path=catalog)

    assert receipt["status"] == "repaired"
    assert receipt["owner_state_carried_forward"] == 0
    admission = _rows(catalog)["owner-profile"]["execution_admission"]
    assert admission["live_enabled"] is False
    assert "scene_execution_owner_store_missing" in admission["blockers"]



def test_receipt_reports_where_the_pass_spent_its_time(tmp_path: Path) -> None:
    """The guard logs this so the next slow catalog pass names its phase."""
    for index in range(3):
        _profile(tmp_path, f"profile-{index}")
    receipt = reconcile_public_catalog(profile_dir=tmp_path / "profiles", catalog_path=tmp_path / "catalog.json")
    timing = receipt["pass_timing_seconds"]
    assert set(timing) == {"profile_parse", "immutable_inputs", "profile_validation", "input_residency",
                           "profiles", "distinct_inputs"}
    assert timing["profiles"] == 3
    assert all(isinstance(timing[name], float) and timing[name] >= 0 for name in
               ("profile_parse", "immutable_inputs", "profile_validation", "input_residency"))
    second = reconcile_public_catalog(profile_dir=tmp_path / "profiles", catalog_path=tmp_path / "catalog.json")
    assert second["status"] == "consistent" and "pass_timing_seconds" in second
