# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_release_leases.py
"""Release protection comes from typed, expiring leases, never from a grep.

On 2026-09-26 deploy-time retirement retired 0 of 95 release trees because it
protected every 40-hex token in about twenty JSON roots: git tree ids, commits
embedded in profile ids and the consumption records of expired
authorizations.  These tests pin the typed sources that replaced the grep.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from blueprint_pipeline.control_plane_release_leases import (
    CONFIG_KIND,
    DEFAULT_MAX_LIFETIME_SECONDS,
    LIVE_QUEUE_STATES,
    PROTECTIONS_SCHEMA,
    ProtectionSources,
    collect_release_protections,
)


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
