# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_release_leases.py
"""Release protection comes from typed, expiring leases, never from a grep.

On 2026-09-26 deploy-time retirement retired 0 of 95 release trees because it
protected every 40-hex token in about twenty JSON roots: git tree ids, commits
embedded in profile ids and the consumption records of expired
authorizations.  These tests pin the typed sources that replaced the grep.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

import blueprint_pipeline.control_plane_release_leases as leases
from blueprint_pipeline.control_plane_release_leases import (
    CONFIG_KIND,
    DEFAULT_MAX_LIFETIME_SECONDS,
    LEASE_SCHEMA,
    LIVE_QUEUE_STATES,
    PROTECTIONS_SCHEMA,
    ProtectionSources,
    collect_release_protections,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json


DAY = 86_400.0
NOW = 1_800_000_000.0
B, C, D, E, F = ("b" * 40, "c" * 40, "d" * 40, "e" * 40, "f" * 40)
TREE = "9" * 40  # a git tree id: 40 hex, but never a commit
RUNTIMES = "/var/lib/blueprint/task-evaluation-inputs/system-runtimes"


def _sources(tmp_path: Path, **overrides: object) -> ProtectionSources:
    control_plane = tmp_path / "pipeline-control-plane"
    for queue, states in LIVE_QUEUE_STATES.items():
        for state in states:
            (control_plane / queue / state).mkdir(parents=True)
    (control_plane / "sam31-preparation-executions" / "wake-pending").mkdir()
    profiles = tmp_path / "task-evaluation-launch-profiles"
    profiles.mkdir()
    standing = control_plane / "standing-authorizations"
    standing.mkdir()
    bindings = control_plane / "task-evaluation-release-retention-bindings"
    bindings.mkdir()
    values: dict[str, object] = {
        "control_plane_root": control_plane,
        "profile_dir": profiles,
        "standing_authorization_dir": standing,
        "binding_root": bindings,
        "lease_root": control_plane / "release-leases",
        "config_files": (),
        "intent_root": control_plane / "task-evaluation-scene-intents",
        "launch_run_root": control_plane / "task-evaluation-launch-runs",
    }
    values.update(overrides)
    return ProtectionSources(**values)  # type: ignore[arg-type]


def _write(path: Path, value: object, *, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _protected(result: dict) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for row in result["leases"]:
        rows.setdefault(row["commit"], []).append(f"{row['kind']}:{row['source']}")
    return {commit: sorted(sources) for commit, sources in rows.items()}


def test_queue_envelopes_protect_only_through_typed_fields(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    queues = sources.control_plane_root
    _write(
        sources.profile_dir / "scene-launch.json",
        {"profile_id": "scene-launch", "source_commit": B, "allocator": {"argv": []}},
    )
    _write(
        queues / "task-evaluation-launches" / "pending" / "launch.json",
        {"launch_profile_id": "scene-launch", "git_tree": TREE, "note": f"built from {TREE}",
         "run_id": f"run-{TREE}"},
        mtime=NOW - DAY,
    )
    _write(
        queues / "task-evaluation-launch-activations" / "processing" / "activation.json",
        {"request": {"expected_production_commit": C, "scene": {"tree": TREE}}},
        mtime=NOW - DAY,
    )
    # A wake-pending marker holds only a job digest; even a commit field there
    # is not a live queue envelope.
    _write(
        queues / "sam31-preparation-executions" / "wake-pending" / "child.json",
        {"job_digest": "sha256:" + "0" * 64, "expected_source_commit": D},
        mtime=NOW - DAY,
    )
    _write(
        queues / "task-evaluation-launch-preparations" / "awaiting_capacity" / "prep.json",
        {"request": {"expected_production_commit": E}},
        mtime=NOW - DAY,
    )
    _write(
        queues / "task-evaluation-episode-compilations" / "pending" / "compile.json",
        {"inputs": [{"path": f"{RUNTIMES}/splat-render/{F}/bin/node"}]},
        mtime=NOW - DAY,
    )

    result = collect_release_protections(sources, now=NOW, migrate=False)

    assert result["schema_version"] == PROTECTIONS_SCHEMA
    assert result["blockers"] == []
    assert _protected(result) == {
        B: ["live_queue:task-evaluation-launches/pending/launch.json"],
        C: ["live_queue:task-evaluation-launch-activations/processing/activation.json"],
        E: ["live_queue:task-evaluation-launch-preparations/awaiting_capacity/prep.json"],
        F: ["live_queue:task-evaluation-episode-compilations/pending/compile.json"],
    }
    launch = next(row for row in result["leases"] if row["commit"] == B)
    assert launch == {
        "commit": B,
        "kind": "live_queue",
        "owner": "task-evaluation-launches",
        "reason": "live_queue:task-evaluation-launches/pending",
        "run_ref": {
            "kind": "queue_envelope",
            "queue": "task-evaluation-launches",
            "state": "pending",
            "name": "launch.json",
        },
        "expires_at_epoch": NOW - DAY + DEFAULT_MAX_LIFETIME_SECONDS,
        "source": "task-evaluation-launches/pending/launch.json",
    }
    assert result["lapsed"] == [] and result["migrated"] == []


def test_unreadable_queue_envelope_blocks(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    queues = sources.control_plane_root
    (queues / "task-evaluation-launches" / "pending" / "broken.json").write_text(
        "{not json", encoding="utf-8"
    )
    elsewhere = _write(tmp_path / "elsewhere.json", {"source_commit": B})
    (queues / "task-evaluation-policy-canary-dispatches" / "processing" / "linked.json").symlink_to(
        elsewhere
    )
    _write(
        queues / "task-evaluation-launch-activations" / "pending" / "unpublished.json",
        {"launch_profile_id": "never-published"},
    )

    result = collect_release_protections(sources, now=NOW, migrate=False)

    assert result["blockers"] == [
        "release_protection_profile_missing:never-published",
        "release_protection_queue_unreadable:task-evaluation-launches/pending/broken.json",
        "release_protection_queue_unreadable:"
        "task-evaluation-policy-canary-dispatches/processing/linked.json",
    ]
    assert B not in _protected(result)

    absent = collect_release_protections(
        _sources(tmp_path / "second", control_plane_root=tmp_path / "absent"),
        now=NOW,
        migrate=False,
    )
    assert absent["blockers"] == ["release_protection_control_plane_root_missing"]


def test_queue_envelope_lapses_after_max_lifetime(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    queues = sources.control_plane_root
    stale = _write(
        queues / "task-evaluation-launches" / "processing" / "stale.json",
        {"source_commit": B},
        mtime=NOW - 31 * DAY,
    )
    _write(
        queues / "task-evaluation-launches" / "pending" / "fresh.json",
        {"source_commit": C},
        mtime=NOW - 29 * DAY,
    )

    result = collect_release_protections(sources, now=NOW, migrate=False)

    assert _protected(result) == {C: ["live_queue:task-evaluation-launches/pending/fresh.json"]}
    assert result["lapsed"] == [
        {
            "commit": B,
            "kind": "live_queue",
            "owner": "task-evaluation-launches",
            "reason": "live_queue:task-evaluation-launches/processing",
            "run_ref": {
                "kind": "queue_envelope",
                "queue": "task-evaluation-launches",
                "state": "processing",
                "name": "stale.json",
            },
            "expires_at_epoch": NOW - 31 * DAY + DEFAULT_MAX_LIFETIME_SECONDS,
            "source": "task-evaluation-launches/processing/stale.json",
            "why": "max_lifetime",
        }
    ]
    assert result["warnings"] == [
        "release_protection_queue_envelope_past_max_lifetime:"
        "task-evaluation-launches/processing/stale.json"
    ]
    assert result["blockers"] == []
    # Lapsing is a retention decision only; the envelope itself is untouched.
    assert json.loads(stale.read_text(encoding="utf-8")) == {"source_commit": B}


def test_configured_runtime_paths_protect_and_bootstrap_machinery_path_is_followed(
    tmp_path: Path,
) -> None:
    etc = tmp_path / "etc"
    machinery = _write(
        etc / "task-evaluation-public-scene-machinery.json",
        {
            "preparation": {"runtime_root": f"{RUNTIMES}/splat-render/{B}"},
            # A bare commit in configuration is not a runtime this host runs.
            "provider": {"expected_production_commit": C},
        },
    )
    rotated = _write(
        etc / "rotated-public-scene-machinery.json",
        {"preparation": {"toolchain": f"{RUNTIMES}/scene-configuration/{D}/bin/node"}},
    )
    bootstrap = _write(
        etc / "task-evaluation-scene-preparation-bootstrap.json",
        {"public_scene_machinery_path": str(rotated), "running_repo_root": E},
    )
    sources = _sources(tmp_path, config_files=(machinery, bootstrap, etc / "absent.json"))

    result = collect_release_protections(sources, now=NOW, migrate=False)

    assert result["blockers"] == []
    assert _protected(result) == {
        B: ["configured_runtime:task-evaluation-public-scene-machinery.json"],
        D: ["configured_runtime:rotated-public-scene-machinery.json"],
    }
    assert next(row for row in result["leases"] if row["commit"] == B) == {
        "commit": B,
        "kind": CONFIG_KIND,
        "owner": "task-evaluation-public-scene-machinery.json",
        "reason": "configured_runtime:task-evaluation-public-scene-machinery.json",
        "run_ref": None,
        "expires_at_epoch": None,
        "source": "task-evaluation-public-scene-machinery.json",
    }

    machinery.write_text("{", encoding="utf-8")
    blocked = collect_release_protections(sources, now=NOW, migrate=False)
    assert blocked["blockers"] == [
        "release_protection_config_unreadable:task-evaluation-public-scene-machinery.json"
    ]


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _profile(sources: ProtectionSources, profile_id: str, commit: str) -> dict:
    profile = {
        "profile_id": profile_id,
        "profile_digest": f"sha256:{profile_id}",
        "source_commit": commit,
        "allocator": {"max_spend_usd": 4.0},
    }
    _write(sources.profile_dir / f"{profile_id}.json", profile)
    return profile


def _authorize(
    sources: ProtectionSources,
    profile: dict,
    *,
    expires_at: float,
    max_launches: int = 3,
    max_total_spend_usd: float = 20.0,
    consumed_usd: tuple[float, ...] = (),
    **fields: object,
) -> None:
    value = {
        "schema_version": "task_evaluation_standing_launch_authorization.v1",
        "profile_id": profile["profile_id"],
        "profile_digest": profile["profile_digest"],
        "authorized_by": "ops-lead@blueprint",
        "authorization_reference": "OPS-4242",
        "issued_at": _iso(NOW - 2 * DAY),
        "expires_at": _iso(expires_at),
        "max_launches": max_launches,
        "max_total_spend_usd": max_total_spend_usd,
        **fields,
    }
    directory = sources.standing_authorization_dir
    _write(directory / f"{profile['profile_id']}.json", value)
    for index, amount in enumerate(consumed_usd):
        _write(
            directory / "consumed" / profile["profile_id"] / f"launch-{index}.json",
            {"profile_id": profile["profile_id"], "max_spend_usd": amount},
        )


def test_standing_authorization_protects_only_while_valid(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    _authorize(sources, _profile(sources, "valid", B), expires_at=NOW + 3 * DAY)
    _authorize(
        sources,
        _profile(sources, "anonymous", "1" * 40),
        expires_at=NOW + DAY,
        authorized_by="",
        authorization_reference=None,
    )
    _authorize(sources, _profile(sources, "expired", C), expires_at=NOW - 1)
    _authorize(
        sources,
        _profile(sources, "exhausted", D),
        expires_at=NOW + DAY,
        max_launches=1,
        consumed_usd=(1.0,),
    )
    _authorize(
        sources,
        _profile(sources, "ceiling", E),
        expires_at=NOW + DAY,
        max_total_spend_usd=5.0,
        consumed_usd=(2.0,),
    )
    _authorize(sources, _profile(sources, "malformed", F), expires_at=NOW + DAY)
    malformed = sources.standing_authorization_dir / "malformed.json"
    value = json.loads(malformed.read_text(encoding="utf-8"))
    malformed.write_text(json.dumps({**value, "expires_at": "next tuesday"}), encoding="utf-8")

    result = collect_release_protections(sources, now=NOW, migrate=False)

    assert result["blockers"] == ["release_protection_standing_authorization_invalid:malformed"]
    assert [row for row in result["leases"] if row["commit"] == B] == [
        {
            "commit": B,
            "kind": "standing_authorization",
            "owner": "ops-lead@blueprint",
            "reason": "OPS-4242",
            "run_ref": {"kind": "standing_authorization", "profile_id": "valid"},
            "expires_at_epoch": NOW + 3 * DAY,
            "source": "standing-authorizations/valid.json",
        }
    ]
    anonymous = next(row for row in result["leases"] if row["commit"] == "1" * 40)
    assert anonymous["owner"] == "unknown"
    assert anonymous["reason"] == "unconsumed_standing_authorization:anonymous"
    assert _protected(result) == {
        B: ["standing_authorization:standing-authorizations/valid.json"],
        "1" * 40: ["standing_authorization:standing-authorizations/anonymous.json"],
    }
    assert {(row["commit"], row["why"]) for row in result["lapsed"]} == {
        (C, "run_terminal"),
        (D, "run_terminal"),
        (E, "run_terminal"),
    }


def test_consumption_records_and_profiles_alone_do_not_protect(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    # A published profile with no live launch and no authorization.
    _write(
        sources.profile_dir / f"scene-unauthorized-{B}.json",
        {
            "profile_id": f"scene-unauthorized-{B}",
            "allocator": {"argv": ["--expected-source-commit", C, "--tree", TREE]},
        },
    )
    # Consumption records and step logs of an authorization that is gone.
    standing = sources.standing_authorization_dir
    _write(
        standing / "consumed" / "retired-profile" / "launch-1.json",
        {"profile_id": "retired-profile", "source_commit": D, "max_spend_usd": 1.0},
    )
    (standing / "retired-profile.json.standing_authorization.stdout.log").write_text(
        f"launched {E}\n", encoding="utf-8"
    )

    result = collect_release_protections(sources, now=NOW, migrate=False)

    assert result["leases"] == []
    assert result["lapsed"] == []
    assert result["blockers"] == []

    # An unlistable profile directory is only a warning; profiles never protect
    # alone.  (Root ignores directory permissions, so only a user can see it.)
    if os.geteuid() != 0:
        sources.profile_dir.chmod(0)
        try:
            unlistable = collect_release_protections(sources, now=NOW, migrate=False)
        finally:
            sources.profile_dir.chmod(0o755)
        assert unlistable["blockers"] == []
        assert "release_protection_profile_dir_unreadable" in unlistable["warnings"]


BINDING_REASON = "Completed SAM prefix replay requires its original immutable renderer release"


def _intent(
    sources: ProtectionSources,
    intent_id: str,
    *,
    expires_at: float,
    status: str | None = None,
    revoked: bool = False,
    attempts: tuple[str, ...] = (),
) -> Path:
    from blueprint_pipeline import task_evaluation_scene_intake as intake

    assert sources.intent_root is not None
    directory = sources.intent_root / intent_id
    directory.mkdir(parents=True)
    intent = intake._seal(
        {
            "schema_version": intake.INTENT_SCHEMA,
            "intent_id": intent_id,
            "request": {"owner": {"tenant_id": "tenant"}, "execution": {"expires_at_epoch": expires_at}},
            "authenticated_issuer": "blueprint-webapp",
            "accepted_at_epoch": NOW - 2 * DAY,
        },
        "intent_digest",
    )
    _write(directory / "intent.json", intent)
    if status is not None:
        projection = intake._seal(
            {
                "schema_version": "task_evaluation_scene_progression.v1",
                "intent_id": intent_id,
                "intent_digest": intent["intent_digest"],
                "status": status,
                "phase": "fixture",
            },
            "progression_digest",
        )
        _write(directory / "progression.json", projection)
    if revoked:
        _write(directory / "revoked.json", {"status": "revoked"})
    for attempt_id in attempts:
        # A reserved paid attempt with no cancellation or settlement: in flight.
        row = intake._seal(
            {
                "schema_version": intake.ATTEMPT_SCHEMA,
                "intent_id": intent_id,
                "intent_digest": intent["intent_digest"],
                "attempt_id": attempt_id,
                "source_commit": B,
                "provider": "vast",
                "maximum_spend_usd": 1.0,
                "status": "reserved",
            },
            "attempt_digest",
        )
        _write(directory / "attempts" / f"{attempt_id}.json", row)
    return directory


def _binding(
    sources: ProtectionSources,
    name: str,
    commit: str,
    *,
    intent_id: str | None = None,
    retained: dict | None = None,
) -> Path:
    value: dict[str, object] = {
        "schema_version": "task_evaluation_release_retention_binding.v1",
        "status": "required",
        "source_commit": commit,
        "reason": BINDING_REASON,
    }
    if intent_id is not None:
        evidence = (
            sources.control_plane_root.parent
            / "task-evaluation-inputs"
            / "completed-scene-preparation"
            / intent_id
            / "attempt-1"
            / "evidence.json"
        )
        # Evidence that is not a prefix adoption: the binding has no ancestors.
        _write(evidence, {"schema_version": "fixture_terminal_evidence"})
        value["evidence"] = {"path": str(evidence), "sha256": "sha256:" + "0" * 64, "size_bytes": 1}
    if retained is not None:
        value["retained_release"] = retained
    path = sources.binding_root / name
    path.write_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    return path


def _sidecar(sources: ProtectionSources, name: str) -> Path:
    return sources.lease_root / "bindings" / f"{name}.lease.v1.json"


def _lapsed(result: dict) -> list[tuple[str, str]]:
    return sorted((row["commit"], row["why"]) for row in result["lapsed"])


def test_legacy_binding_is_migrated_to_a_sidecar_lease_and_reported(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    _intent(sources, "scene-live", expires_at=NOW + 10 * DAY)
    binding = _binding(
        sources,
        "sam31-prefix-1.json",
        B,
        intent_id="scene-live",
        retained={
            "path": f"/opt/blueprint/task-evaluation-control-plane-releases/{C}",
            "source_commit": C,
            "tree": TREE,
        },
    )
    before = binding.read_bytes()

    result = collect_release_protections(sources, now=NOW, migrate=True)

    assert binding.read_bytes() == before
    sidecar = _sidecar(sources, "sam31-prefix-1.json")
    lease = json.loads(sidecar.read_text(encoding="utf-8"))
    assert lease == {
        "schema_version": LEASE_SCHEMA,
        "binding": "sam31-prefix-1.json",
        "binding_sha256": "sha256:" + hashlib.sha256(before).hexdigest(),
        "commits": [B, C],
        "owner": "legacy-migration",
        "reason": BINDING_REASON,
        "run_ref": {"kind": "scene_intent", "intent_id": "scene-live"},
        "created_at_epoch": NOW,
        "expires_at_epoch": NOW + 14 * DAY,
        "max_expires_at_epoch": NOW + 30 * DAY,
        "migrated": True,
        "lease_digest": canonical_digest(lease, digest_field="lease_digest"),
    }
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o640
    assert result["migrated"] == ["sam31-prefix-1.json"]
    assert result["blockers"] == []
    source = "task-evaluation-release-retention-bindings/sam31-prefix-1.json"
    assert _protected(result) == {
        B: [f"retention_binding:{source}"],
        C: [f"retention_binding:{source}"],
    }
    assert next(row for row in result["leases"] if row["commit"] == B) == {
        "commit": B,
        "kind": "retention_binding",
        "owner": "legacy-migration",
        "reason": BINDING_REASON,
        "run_ref": {"kind": "scene_intent", "intent_id": "scene-live"},
        "expires_at_epoch": NOW + 14 * DAY,
        "source": source,
    }

    # A dry run treats an unmigrated binding as that would-be lease and writes nothing.
    _binding(sources, "legacy-dry-run.json", D)
    dry_run = collect_release_protections(sources, now=NOW + DAY, migrate=False)
    assert dry_run["migrated"] == []
    assert not _sidecar(sources, "legacy-dry-run.json").exists()
    assert D in _protected(dry_run)

    # A second migration reuses the first sidecar and migrates only what is new.
    lease_bytes = sidecar.read_bytes()
    again = collect_release_protections(sources, now=NOW + DAY, migrate=True)
    assert again["migrated"] == ["legacy-dry-run.json"]
    assert sidecar.read_bytes() == lease_bytes


def test_expired_binding_no_longer_protects(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    _binding(sources, "unowned.json", B)  # no evidence path: the run is unknown

    first = collect_release_protections(sources, now=NOW, migrate=True)
    assert B in _protected(first)

    later = collect_release_protections(sources, now=NOW + 15 * DAY, migrate=True)
    assert _protected(later) == {}
    assert _lapsed(later) == [(B, "expired")]
    assert later["migrated"] == [] and later["renewed"] == [] and later["blockers"] == []


def test_live_run_ref_still_protects_and_renews_within_max_lifetime(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    _intent(sources, "scene-running", expires_at=NOW + 60 * DAY, status="running")
    binding = _binding(sources, "running.json", B, intent_id="scene-running")
    before = binding.read_bytes()
    collect_release_protections(sources, now=NOW, migrate=True)
    sidecar = _sidecar(sources, "running.json")

    def lease() -> dict:
        return json.loads(sidecar.read_text(encoding="utf-8"))

    quiet = collect_release_protections(sources, now=NOW + 6 * DAY, migrate=True)
    assert quiet["renewed"] == [] and lease()["expires_at_epoch"] == NOW + 14 * DAY

    # Past its own expiry the lease still protects a live run; a dry run never renews.
    dry_run = collect_release_protections(sources, now=NOW + 15 * DAY, migrate=False)
    assert B in _protected(dry_run) and dry_run["renewed"] == []
    assert lease()["expires_at_epoch"] == NOW + 14 * DAY

    renewed = collect_release_protections(sources, now=NOW + 15 * DAY, migrate=True)
    assert B in _protected(renewed) and renewed["renewed"] == ["running.json"]
    assert lease()["expires_at_epoch"] == NOW + 29 * DAY
    assert lease()["max_expires_at_epoch"] == NOW + 30 * DAY
    assert lease()["lease_digest"] == canonical_digest(lease(), digest_field="lease_digest")

    capped = collect_release_protections(sources, now=NOW + 23 * DAY, migrate=True)
    assert capped["renewed"] == ["running.json"]
    assert lease()["expires_at_epoch"] == NOW + 30 * DAY

    ended = collect_release_protections(sources, now=NOW + 30 * DAY, migrate=True)
    assert _protected(ended) == {}
    assert _lapsed(ended) == [(B, "max_lifetime")]
    assert ended["warnings"] == ["release_protection_lease_past_max_lifetime:running.json"]
    assert binding.read_bytes() == before


def test_binding_for_a_terminal_run_lapses(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    _intent(sources, "scene-completed", expires_at=NOW + 10 * DAY, status="completed")
    _intent(sources, "scene-revoked", expires_at=NOW + 10 * DAY, revoked=True)
    _intent(sources, "scene-running", expires_at=NOW + 10 * DAY, status="running")
    _binding(sources, "completed.json", B, intent_id="scene-completed")
    _binding(sources, "revoked.json", C, intent_id="scene-revoked")
    _binding(sources, "running.json", E, intent_id="scene-running")

    result = collect_release_protections(sources, now=NOW, migrate=True)

    assert _protected(result) == {
        E: ["retention_binding:task-evaluation-release-retention-bindings/running.json"]
    }
    assert sorted(
        (row["commit"], row["why"], row["run_ref"]["intent_id"]) for row in result["lapsed"]
    ) == [
        (B, "run_terminal", "scene-completed"),
        (C, "run_terminal", "scene-revoked"),
    ]
    # Every binding still gets its lease on record, lapsed or not.
    assert result["migrated"] == ["completed.json", "revoked.json", "running.json"]
    assert result["blockers"] == []


def test_an_expired_intent_keeps_its_binding_until_the_lease_ends(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    _intent(sources, "scene-expired", expires_at=NOW - 1)
    _binding(sources, "expired.json", D, intent_id="scene-expired")

    # Expiry closes new execution, but the owner may still extend the window,
    # so the run is unknown rather than over: the lease runs its TTL.
    first = collect_release_protections(sources, now=NOW, migrate=True)
    assert _protected(first) == {
        D: ["retention_binding:task-evaluation-release-retention-bindings/expired.json"]
    }

    later = collect_release_protections(sources, now=NOW + 15 * DAY, migrate=True)
    assert _protected(later) == {}
    assert _lapsed(later) == [(D, "expired")]


def test_an_attempt_in_flight_keeps_a_revoked_or_expired_intent_live(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    _intent(sources, "scene-revoked-running", expires_at=NOW + 90 * DAY, revoked=True,
            attempts=("attempt-1",))
    _intent(sources, "scene-expired-running", expires_at=NOW - 1, attempts=("attempt-1",))
    unprovable = _intent(sources, "scene-revoked-unprovable", expires_at=NOW + 90 * DAY,
                         revoked=True, attempts=("attempt-1",))
    # A cancellation record that does not validate proves nothing about the attempt.
    _write(unprovable / "cancelled-unstarted-controls" / "attempt-1.json", {"schema_version": "bogus"})
    _binding(sources, "revoked-running.json", B, intent_id="scene-revoked-running")
    _binding(sources, "expired-running.json", C, intent_id="scene-expired-running")
    _binding(sources, "unprovable.json", D, intent_id="scene-revoked-unprovable")
    collect_release_protections(sources, now=NOW, migrate=True)

    # Past the 14-day TTL: live runs still protect (and renew); unknown ones lapse.
    result = collect_release_protections(sources, now=NOW + 15 * DAY, migrate=True)

    assert sorted(_protected(result)) == [B, C]
    assert result["renewed"] == ["expired-running.json", "revoked-running.json"]
    assert _lapsed(result) == [(D, "expired")]
    assert result["blockers"] == []


def test_an_owner_can_extend_an_expired_intent_and_keep_its_release(tmp_path: Path) -> None:
    from blueprint_pipeline.task_evaluation_scene_execution_window import (
        ACK,
        extend_scene_execution_window,
    )

    sources = _sources(tmp_path)
    directory = _intent(sources, "scene-extended", expires_at=NOW - 1)
    _binding(sources, "extended.json", D, intent_id="scene-extended")
    # Between expiry and the owner's extension, the release must still be there.
    expired = collect_release_protections(sources, now=NOW, migrate=True)
    assert D in _protected(expired) and expired["lapsed"] == []
    intent = json.loads((directory / "intent.json").read_text(encoding="utf-8"))

    # Ten days after expiry the owner extends the window; nothing had lapsed yet.
    extend_scene_execution_window(
        queue_root=sources.intent_root,
        intent_id="scene-extended",
        intent_digest=intent["intent_digest"],
        owner=intent["request"]["owner"],
        authenticated_client="blueprint-webapp",
        trusted_clients={"blueprint-webapp"},
        expires_at_epoch=NOW + 12 * DAY,
        authorization_reference="OWNER-7",
        ack=ACK,
        now=NOW + 10 * DAY,
    )
    extended = collect_release_protections(sources, now=NOW + 10 * DAY, migrate=True)

    assert D in _protected(extended)
    assert extended["renewed"] == ["extended.json"]
    lease = json.loads(_sidecar(sources, "extended.json").read_text(encoding="utf-8"))
    assert lease["expires_at_epoch"] == NOW + 24 * DAY


def test_misplaced_plan_is_a_warning_not_protection(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    _write(
        sources.binding_root / "plan-20260828T234417Z.json",
        {
            "schema_version": "task_evaluation_release_retention_plan.v1",
            "status": "dry_run",
            "protected_commits": {B: ["active_release"], C: ["keep_last"]},
            "eligible_commits": [{"source_commit": D}],
        },
    )

    result = collect_release_protections(sources, now=NOW, migrate=True)

    assert result["leases"] == [] and result["lapsed"] == []
    assert result["blockers"] == [] and result["migrated"] == []
    assert result["warnings"] == ["misplaced_retention_plan:plan-20260828T234417Z.json"]
    assert not _sidecar(sources, "plan-20260828T234417Z.json").exists()


def test_changed_binding_bytes_block(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    binding = _binding(sources, "sam31-prefix-2.json", B)
    collect_release_protections(sources, now=NOW, migrate=True)
    value = json.loads(binding.read_text(encoding="utf-8"))
    binding.write_text(json.dumps({**value, "reason": "rewritten in place"}), encoding="utf-8")
    _write(
        sources.binding_root / "optional.json",
        {
            "schema_version": "task_evaluation_release_retention_binding.v1",
            "status": "optional",
            "source_commit": C,
            "reason": "not a required binding",
        },
    )

    result = collect_release_protections(sources, now=NOW + DAY, migrate=True)

    assert result["blockers"] == [
        "release_protection_binding_changed:sam31-prefix-2.json",
        "release_protection_binding_invalid:optional.json",
    ]
    assert result["migrated"] == []


def test_republishing_a_sam_prefix_binding_still_matches(tmp_path: Path) -> None:
    from blueprint_pipeline import task_evaluation_sam31_prefix_adoption as adoption

    sources = _sources(tmp_path)
    release = tmp_path / "task-evaluation-control-plane-releases" / B
    release.mkdir(parents=True)
    profile = {"schema_version": "fixture_profile", "source_commit": B}
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    profile_path = tmp_path / "source-profile.json"
    profile_path.write_text(canonical_json(profile), encoding="utf-8")
    value = {
        "schema_version": adoption.SCHEMA,
        "status": "verified_completed_prefix",
        "original_execution_commit": B,
        "source_profile": {"path": str(profile_path)},
        "retained_release_pin": {"source_commit": B, "path": str(release), "tree": TREE},
    }
    value["adoption_digest"] = canonical_digest(value, digest_field="adoption_digest")
    adoption_path = tmp_path / "adoption.json"
    adoption_path.write_text(canonical_json(value), encoding="utf-8")

    pin = adoption.publish_adoption_release_binding(adoption_path, binding_root=sources.binding_root)
    before = Path(pin["path"]).read_bytes()
    result = collect_release_protections(sources, now=NOW, migrate=True)

    assert result["migrated"] == [Path(pin["path"]).name]
    assert _protected(result) == {
        B: [f"retention_binding:task-evaluation-release-retention-bindings/{Path(pin['path']).name}"]
    }
    # The writer compares whole documents on republish: still no conflict.
    assert adoption.publish_adoption_release_binding(
        adoption_path, binding_root=sources.binding_root
    ) == pin
    assert Path(pin["path"]).read_bytes() == before


def test_permission_errors_block_instead_of_reading_as_absent(tmp_path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    sources = _sources(tmp_path)
    queue_root = sources.control_plane_root / "task-evaluation-launches"
    queue_root.chmod(0)
    try:
        result = collect_release_protections(sources, now=NOW, migrate=False)
    finally:
        queue_root.chmod(0o755)

    # An unsearchable queue is not an empty one.
    assert result["blockers"] == [
        "release_protection_queue_unreadable:task-evaluation-launches/pending",
        "release_protection_queue_unreadable:task-evaluation-launches/processing",
    ]


def test_a_fifo_where_a_document_belongs_blocks_without_hanging(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    os.mkfifo(sources.control_plane_root / "task-evaluation-launches" / "pending" / "stuck.json")
    # Documents read on the deploy's behalf by other modules are guarded too:
    # a consumption record of a live authorization, and a binding's intent.
    _authorize(sources, _profile(sources, "consuming", C), expires_at=NOW + DAY)
    consumed = sources.standing_authorization_dir / "consumed" / "consuming"
    consumed.mkdir(parents=True)
    os.mkfifo(consumed / "launch-1.json")
    intent = _intent(sources, "scene-fifo", expires_at=NOW + DAY)
    (intent / "intent.json").unlink()
    os.mkfifo(intent / "intent.json")
    _binding(sources, "fifo-intent.json", D, intent_id="scene-fifo")
    outcome: dict = {}

    def collect() -> None:
        outcome["result"] = collect_release_protections(sources, now=NOW, migrate=False)

    worker = threading.Thread(target=collect, daemon=True)
    worker.start()
    worker.join(timeout=10)

    assert not worker.is_alive(), "opening a FIFO with no writer must not block the deploy"
    result = outcome["result"]
    assert result["blockers"] == [
        "release_protection_queue_unreadable:task-evaluation-launches/pending/stuck.json",
        "release_protection_standing_authorization_invalid:consuming",
    ]
    # An intent that cannot be read leaves its binding's run unknown: protected.
    assert D in _protected(result)


def test_a_short_sidecar_write_leaves_no_partial_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _sources(tmp_path)
    _binding(sources, "short.json", B)
    complete_write = leases._write_all

    def short_write(descriptor: int, payload: bytes) -> None:
        os.write(descriptor, payload[: len(payload) // 2])
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(leases, "_write_all", short_write)
    failed = collect_release_protections(sources, now=NOW, migrate=True)

    assert failed["blockers"] == ["release_protection_lease_write_failed:short.json"]
    assert not _sidecar(sources, "short.json").exists()
    assert list((sources.lease_root / "bindings").iterdir()) == []

    # With space again the next deploy migrates it; nothing was wedged.
    monkeypatch.setattr(leases, "_write_all", complete_write)
    retried = collect_release_protections(sources, now=NOW, migrate=True)
    assert retried["blockers"] == [] and retried["migrated"] == ["short.json"]
    assert json.loads(_sidecar(sources, "short.json").read_text(encoding="utf-8"))["binding"] == "short.json"


def test_standing_authorization_commit_comes_from_the_id_prefix_or_blocks(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    directory = sources.standing_authorization_dir

    def authorize(profile_id: str) -> None:
        _write(
            directory / f"{profile_id}.json",
            {
                "schema_version": "task_evaluation_standing_launch_authorization.v1",
                "profile_id": profile_id,
                "profile_digest": f"sha256:{profile_id}",
                "max_launches": 2,
                "max_total_spend_usd": 5.0,
                "expires_at": _iso(NOW + DAY),
            },
        )

    # No readable profile: the commit is the 40-hex segment after the prefix,
    # never a later one (a revision or digest suffix).
    authorize(f"lane-{B}-{C}")
    # No readable profile and nothing in the id: which release it needs is unknown.
    authorize("lane-without-commit")
    # A readable profile that pins no release runs from the active one.
    _write(sources.profile_dir / "unpinned.json", {"profile_id": "unpinned", "profile_digest": "sha256:unpinned",
                                                   "allocator": {"max_spend_usd": 1.0}})
    authorize("unpinned")

    result = collect_release_protections(sources, now=NOW, migrate=False)

    assert _protected(result) == {B: [f"standing_authorization:standing-authorizations/lane-{B}-{C}.json"]}
    assert result["blockers"] == [
        "release_protection_standing_authorization_commit_unknown:lane-without-commit"
    ]
    assert result["warnings"] == ["release_protection_profile_commit_unpinned:unpinned"]


def test_blocker_codes_never_carry_raw_identities(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    pending = sources.control_plane_root / "task-evaluation-launches" / "pending"
    _write(pending / "launch.json", {"launch_profile_id": "../../etc/shadow"})
    (pending / "odd name\n.json").write_text("{", encoding="utf-8")

    result = collect_release_protections(sources, now=NOW, migrate=False)

    assert len(result["blockers"]) == 2
    for code in result["blockers"]:
        identity = code.rpartition(":")[2].rpartition("/")[2]
        assert identity.startswith("invalid-") and len(identity) == len("invalid-") + 12, code
        assert "/etc" not in code and "\n" not in code and " " not in code
    assert {code.rpartition(":")[0] for code in result["blockers"]} == {
        "release_protection_profile_missing",
        "release_protection_queue_unreadable",
    }


def test_missing_protection_sources_block_or_warn(tmp_path: Path) -> None:
    import shutil

    sources = _sources(tmp_path)
    queues = sources.control_plane_root
    # A queue state never used on this host holds nothing: no row, no warning.
    (queues / "task-evaluation-launches" / "processing").rmdir()
    # A queue that is missing altogether is worth a look.
    shutil.rmtree(queues / "task-evaluation-scene-constructions")
    # With the control-plane root present, a missing authorization or binding
    # root is not "none exist": it is a source that cannot be read.
    sources.standing_authorization_dir.rmdir()
    sources.binding_root.rmdir()

    result = collect_release_protections(sources, now=NOW, migrate=True)

    assert result["blockers"] == [
        "release_protection_source_missing:standing-authorizations",
        "release_protection_source_missing:task-evaluation-release-retention-bindings",
    ]
    assert result["warnings"] == [
        "release_protection_queue_root_missing:task-evaluation-scene-constructions"
    ]
    assert not sources.binding_root.exists()  # the collector never invents a source


def test_queue_scan_rescans_when_an_envelope_moves_mid_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _sources(tmp_path)
    preparations = sources.control_plane_root / "task-evaluation-launch-preparations"
    _write(
        preparations / "processing" / "sentinel.json",
        {"request": {"expected_production_commit": B}},
        mtime=NOW - DAY,
    )
    waiting = _write(
        preparations / "awaiting_capacity" / "waiting.json",
        {"request": {"expected_production_commit": C}},
        mtime=NOW - DAY,
    )
    read_document = leases._read_document
    moved: list[bool] = []

    def capacity_frees_up(path: Path):
        # After pending was scanned and before awaiting_capacity is listed, the
        # waiting envelope moves back to pending: one pass never sees it.
        if path.name == "sentinel.json" and not moved:
            os.replace(waiting, preparations / "pending" / "waiting.json")
            moved.append(True)
        return read_document(path)

    monkeypatch.setattr(leases, "_read_document", capacity_frees_up)
    result = collect_release_protections(sources, now=NOW, migrate=False)

    assert moved == [True]
    assert result["blockers"] == []
    assert _protected(result) == {
        B: ["live_queue:task-evaluation-launch-preparations/processing/sentinel.json"],
        C: ["live_queue:task-evaluation-launch-preparations/pending/waiting.json"],
    }


def test_a_queue_that_never_settles_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sources = _sources(tmp_path)
    _write(
        sources.control_plane_root / "task-evaluation-launches" / "pending" / "moving.json",
        {"source_commit": B},
        mtime=NOW - DAY,
    )
    read_document = leases._read_document
    reads: list[str] = []

    def always_moving(path: Path):
        if path.name == "moving.json":
            reads.append(path.name)
            raise FileNotFoundError(path)  # moved between listing and reading
        return read_document(path)

    monkeypatch.setattr(leases, "_read_document", always_moving)
    result = collect_release_protections(sources, now=NOW, migrate=False)

    # A moved envelope is not an unreadable one; it is retried, then refused.
    assert result["blockers"] == ["release_protection_queue_unstable"]
    assert reads == ["moving.json"] * 3


def _complete(sources: ProtectionSources, intent_id: str) -> None:
    from blueprint_pipeline import task_evaluation_scene_intake as intake

    assert sources.intent_root is not None
    directory = sources.intent_root / intent_id
    intent = json.loads((directory / "intent.json").read_text(encoding="utf-8"))
    _write(
        directory / "progression.json",
        intake._seal(
            {
                "schema_version": "task_evaluation_scene_progression.v1",
                "intent_id": intent_id,
                "intent_digest": intent["intent_digest"],
                "status": "completed",
                "phase": "fixture",
            },
            "progression_digest",
        ),
    )


def test_a_live_descendant_keeps_its_completed_ancestors_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from blueprint_pipeline import task_evaluation_sam31_prefix_adoption as adoption

    # The collector recognizes adoption records without importing their writer.
    assert leases._SAM_PREFIX_ADOPTION_SCHEMA == adoption.SCHEMA
    sources = _sources(tmp_path)
    factory = tmp_path / "task-evaluation-inputs" / "completed-scene-preparation"
    releases = tmp_path / "task-evaluation-control-plane-releases"
    for commit in (B, C):
        (releases / commit).mkdir(parents=True)
    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_inpainting_inputs._git_identity",
        lambda root: {"commit": Path(root).name, "tree": TREE},
    )

    def adopt(intent_id: str, commit: str, parent: Path | None = None) -> Path:
        attempt = factory / intent_id / "attempt-1"
        attempt.mkdir(parents=True)
        profile: dict = {"schema_version": "fixture_profile", "repo_root": str(releases / commit)}
        if parent is not None:
            profile["completed_prefix_adoption"] = adoption.record(parent)
        profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
        (attempt / "profile.json").write_text(canonical_json(profile), encoding="utf-8")
        value = {
            "schema_version": adoption.SCHEMA,
            "status": "verified_completed_prefix",
            "original_execution_commit": commit,
            "source_profile": adoption.record(attempt / "profile.json"),
            "retained_release_pin": {"path": str(releases / commit), "source_commit": commit, "tree": TREE},
        }
        value["adoption_digest"] = canonical_digest(value, digest_field="adoption_digest")
        (attempt / "adoption.json").write_text(canonical_json(value), encoding="utf-8")
        return attempt / "adoption.json"

    # The ancestor's scene is finished; the descendant's replay reuses its prefix.
    ancestor = adopt("scene-ancestor", B)
    descendant = adopt("scene-descendant", C, parent=ancestor)
    _intent(sources, "scene-ancestor", expires_at=NOW + 10 * DAY, status="completed")
    _intent(sources, "scene-descendant", expires_at=NOW + 10 * DAY, status="running")
    pin = adoption.publish_adoption_release_binding(descendant, binding_root=sources.binding_root)
    descendant_name = Path(pin["path"]).name
    ancestor_digest = json.loads(ancestor.read_text(encoding="utf-8"))["adoption_digest"]
    ancestor_name = "sam31-prefix-" + ancestor_digest.removeprefix("sha256:") + ".json"
    assert sorted(path.name for path in sources.binding_root.iterdir()) == sorted(
        [ancestor_name, descendant_name]
    )

    result = collect_release_protections(sources, now=NOW, migrate=True)

    source = "task-evaluation-release-retention-bindings/"
    assert result["blockers"] == [] and result["lapsed"] == []
    assert _protected(result) == {
        B: [f"retention_binding:{source}{ancestor_name}"],
        C: [f"retention_binding:{source}{descendant_name}"],
    }
    assert next(row for row in result["leases"] if row["commit"] == B)["reason"] == (
        f"ancestor_of:{descendant_name}"
    )

    # Once the descendant's run ends too, neither release is needed.
    _complete(sources, "scene-descendant")
    ended = collect_release_protections(sources, now=NOW + DAY, migrate=True)
    assert _protected(ended) == {}
    assert _lapsed(ended) == [(B, "run_terminal"), (C, "run_terminal")]


def test_an_unreadable_adoption_chain_never_lapses_as_terminal(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    _intent(sources, "scene-done", expires_at=NOW + 10 * DAY, status="completed")
    binding = _binding(sources, "sam31-prefix-unreadable.json", B, intent_id="scene-done")
    # The evidence record is gone, so any ancestor it built on cannot be found.
    Path(json.loads(binding.read_text(encoding="utf-8"))["evidence"]["path"]).unlink()

    first = collect_release_protections(sources, now=NOW, migrate=True)
    assert B in _protected(first) and first["lapsed"] == []

    later = collect_release_protections(sources, now=NOW + 15 * DAY, migrate=True)
    assert _lapsed(later) == [(B, "expired")]
