"""The WebApp catalog must describe every published profile.

The catalog was built from the profiles named on the command line, so a profile
could be published into the directory and still be absent from the catalog the
WebApp reads. Observed live: a launch was rejected at profile lookup while the
profile directory looked correct and the catalog listed one stale entry. The
catalog has to be a function of published state, not of one invocation's
arguments.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pwd
import stat
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    TaskEvaluationLaunchError,
    canonical_digest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "publish_task_evaluation_launch_profiles",
    REPO_ROOT / "scripts" / "publish_task_evaluation_launch_profiles.py",
)
publisher = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(publisher)

COMMIT = "0" * 40


def _profile(tmp_path: Path, profile_id: str) -> dict:
    """A minimal profile that passes both fail-closed validators."""
    manifest = tmp_path / f"{profile_id}-source.json"
    manifest.write_text(json.dumps({"profile_id": profile_id}), encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()
    uri = f"https://raw.githubusercontent.com/example/repo/{COMMIT}/{profile_id}.json"
    run_root = "{launch_run_root}"
    profile = {
        "schema_version": "task_evaluation_launch_profile.v1",
        "profile_id": profile_id,
        "program_id": "arm-decision-proof-v1",
        "claim_ceiling": "development_only",
        "allocator": {
            "entrypoint": "python -m blueprint_pipeline.paid_resource_allocator gpu-canary",
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
            "canonical_allocator": (
                "python -m blueprint_pipeline.paid_resource_allocator gpu-canary"
            ),
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
    source = tmp_path / f"{profile_id}.json"
    source.write_text(json.dumps(profile, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": source, "profile": profile}


def _catalog_ids(catalog: Path) -> set[str]:
    return {str(row["profile_id"]) for row in json.loads(catalog.read_text(encoding="utf-8"))}


def test_catalog_describes_profiles_published_by_earlier_invocations(tmp_path: Path) -> None:
    """The regression: publishing B must not drop A from the catalog."""
    profile_dir = tmp_path / "profiles"
    catalog = tmp_path / "catalog.json"
    first = _profile(tmp_path, "profile-alpha")
    second = _profile(tmp_path, "profile-beta")

    publisher.publish_profiles(
        profile_paths=[str(first["path"])],
        profile_dir=str(profile_dir),
        webapp_catalog_out=str(catalog),
    )
    assert _catalog_ids(catalog) == {"profile-alpha"}

    publisher.publish_profiles(
        profile_paths=[str(second["path"])],
        profile_dir=str(profile_dir),
        webapp_catalog_out=str(catalog),
    )

    assert _catalog_ids(catalog) == {"profile-alpha", "profile-beta"}


def test_every_published_profile_reaches_the_catalog(tmp_path: Path) -> None:
    profile_dir = tmp_path / "profiles"
    catalog = tmp_path / "catalog.json"
    ids = [f"profile-{index}" for index in range(3)]
    for profile_id in ids:
        publisher.publish_profiles(
            profile_paths=[str(_profile(tmp_path, profile_id)["path"])],
            profile_dir=str(profile_dir),
            webapp_catalog_out=str(catalog),
        )

    published = {path.stem for path in profile_dir.glob("*.json")}
    assert _catalog_ids(catalog) == published == set(ids)


def test_invalid_profile_in_the_directory_fails_closed(tmp_path: Path) -> None:
    """Refuse rather than emit a catalog that silently omits a published file."""
    profile_dir = tmp_path / "profiles"
    catalog = tmp_path / "catalog.json"
    first = _profile(tmp_path, "profile-alpha")
    publisher.publish_profiles(
        profile_paths=[str(first["path"])],
        profile_dir=str(profile_dir),
        webapp_catalog_out=str(catalog),
    )

    (profile_dir / "corrupt.json").write_text(json.dumps({"schema_version": "nope"}), "utf-8")
    second = _profile(tmp_path, "profile-beta")

    with pytest.raises(TaskEvaluationLaunchError, match="published_profile_invalid:corrupt.json"):
        publisher.publish_profiles(
            profile_paths=[str(second["path"])],
            profile_dir=str(profile_dir),
            webapp_catalog_out=str(catalog),
        )


def test_catalog_never_carries_allocator_arguments(tmp_path: Path) -> None:
    """Widening the catalog must not widen what the public descriptor exposes.

    Allocator argv carries host paths and spend flags, so it stays out of a
    document the WebApp serves. The secret *profile id* is deliberately present
    -- it names which credential profile the allocator must use and is not
    itself a credential.
    """
    profile_dir = tmp_path / "profiles"
    catalog = tmp_path / "catalog.json"
    publisher.publish_profiles(
        profile_paths=[str(_profile(tmp_path, "profile-alpha")["path"])],
        profile_dir=str(profile_dir),
        webapp_catalog_out=str(catalog),
    )

    text = catalog.read_text(encoding="utf-8")
    assert "argv" not in text
    assert "--adapter-output" not in text


def test_publication_makes_root_style_0600_inputs_service_group_read_only(
    tmp_path: Path,
) -> None:
    """Regression: root-created inputs must be readable by the dispatcher.

    The production failure presented three correct, digest-bound JSON files as
    ``root:root 0600``. Root could publish them, but the ``blueprint`` unit
    could not reopen them. Use this test process as the service identity while
    preserving the exact restrictive starting mode.
    """

    profile_dir = tmp_path / "profiles"
    catalog = tmp_path / "catalog.json"
    fixture = _profile(tmp_path, "profile-private-inputs")
    immutable = Path(fixture["profile"]["immutable_inputs"][0]["path"])
    immutable_bytes = immutable.read_bytes()
    immutable.chmod(0o600)
    account = pwd.getpwuid(os.geteuid()).pw_name

    publisher.publish_profiles(
        profile_paths=[fixture["path"]],
        profile_dir=profile_dir,
        webapp_catalog_out=catalog,
        service_account=account,
    )

    metadata = immutable.stat()
    assert immutable.read_bytes() == immutable_bytes
    assert stat.S_IMODE(metadata.st_mode) == 0o440
    assert metadata.st_gid == pwd.getpwnam(account).pw_gid
    assert (profile_dir / "profile-private-inputs.json").is_file()


def test_publication_fails_closed_when_service_cannot_read_an_immutable_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_dir = tmp_path / "profiles"
    catalog = tmp_path / "catalog.json"
    fixture = _profile(tmp_path, "profile-unreadable-inputs")
    account = pwd.getpwuid(os.geteuid()).pw_name
    monkeypatch.setattr(publisher, "_digest_as_account", lambda *_args, **_kwargs: "")

    with pytest.raises(
        TaskEvaluationLaunchError,
        match="launch_profile_immutable_input_consumer_unreadable",
    ):
        publisher.publish_profiles(
            profile_paths=[fixture["path"]],
            profile_dir=profile_dir,
            webapp_catalog_out=catalog,
            service_account=account,
        )

    assert not (profile_dir / "profile-unreadable-inputs.json").exists()
    assert not catalog.exists()


def test_cli_uses_the_invoking_account_outside_production_roots(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _profile(tmp_path, "profile-local-cli")
    profile_dir = tmp_path / "profiles"
    catalog = tmp_path / "catalog.json"

    assert (
        publisher.main(
            [
                "--profile",
                str(fixture["path"]),
                "--profile-dir",
                str(profile_dir),
                "--webapp-catalog-out",
                str(catalog),
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "published"
    assert (profile_dir / "profile-local-cli.json").is_file()
