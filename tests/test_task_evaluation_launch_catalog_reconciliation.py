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
