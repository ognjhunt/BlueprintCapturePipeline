"""A launch input that lives on another machine must not read as ready."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.adp_retained_scene_render_vast import (
    validate_retained_scene_render_bundle,
)
from blueprint_pipeline.host_resident_launch_inputs import (
    LAUNCH_INPUT_ROOTS_ENV,
    HostResidentInputError,
    configured_launch_input_roots,
    launch_profile_residency_blockers,
    path_is_host_resident,
    published_profile_residency_report,
    resolve_host_resident_bundle_receipt,
)


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _job(root: Path, *, bundle_bytes: bytes = b"bundle-archive-bytes") -> Path:
    """A staged job directory: archive, authority copy, and receipt together."""

    job = root / "retained-scene-render-job"
    runtime = job / "provider_runtime"
    runtime.mkdir(parents=True)
    archive = job / "adp_retained_scene_gpu_render_bundle.zip"
    archive.write_bytes(bundle_bytes)
    authority = runtime / "execution_authority.json"
    authority.write_bytes(b'{"schema_version": "third_scene_dual_task_execution_authority.v1"}')
    request = runtime / "source_request_manifest.json"
    request.write_bytes(b'{"schema_version": "adp009d_retained_scene_gpu_render_request.v1"}')
    receipt = {
        "schema_version": "adp009d_retained_scene_gpu_render_bundle.v1",
        "status": "ready",
        # Recorded on the machine that built the bundle; absent everywhere else.
        "bundle_path": "/Users/author/workspace/validation/job/bundle.zip",
        "bundle_relative_path": "adp_retained_scene_gpu_render_bundle.zip",
        "bundle_sha256": _digest(bundle_bytes),
        "execution_authority": {
            "path": "/private/tmp/authoring-checkout/execution_authority.json",
            "relative_path": "provider_runtime/execution_authority.json",
            "sha256": _digest(authority.read_bytes()),
        },
        "request": {
            "path": "/private/tmp/authoring-checkout/request.v1.json",
            "relative_path": "provider_runtime/source_request_manifest.json",
            "sha256": _digest(request.read_bytes()),
        },
    }
    (job / "adp_retained_scene_gpu_render_bundle_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    return job


def _receipt_path(job: Path) -> Path:
    return job / "adp_retained_scene_gpu_render_bundle_receipt.json"


def _artifixer_job(root: Path, *, pipeline_mode: str = "dual_target_artifixer3d_only") -> Path:
    job = root / "artifixer3d-job"
    job.mkdir(parents=True)
    archive = job / "public_scene_artifixer3d_provider_bundle.zip"
    archive.write_bytes(b"paired-artifixer3d-bundle")
    receipt = {
        "schema_version": "public_scene_artifixer3d_bundle.v1",
        "status": "sealed_rehearsal_passed_no_upload_no_execution",
        "blueprint_source_identity": {
            "commit": "a" * 40,
            "tracked_files_clean": True,
        },
        "bundle": {
            "path": "/Users/author/job/public_scene_artifixer3d_provider_bundle.zip",
            "sha256": _digest(archive.read_bytes()),
            "size_bytes": archive.stat().st_size,
        },
        "pipeline_mode": pipeline_mode,
        "direct_editor_backend": "none",
        "semantic_editor_only": False,
        "provider_mutations_performed": 0,
        "local_rehearsal": {
            "status": "passed",
            "pipeline_mode": pipeline_mode,
            "provider_mutations_performed": 0,
            "paid_inference_performed": False,
            "gpu_runtime_started": False,
        },
    }
    receipt_path = job / "public_scene_artifixer3d_bundle_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return receipt_path


def test_receipt_resolves_against_its_own_directory_not_its_recorded_paths(tmp_path):
    job = _job(tmp_path)

    resolution = resolve_host_resident_bundle_receipt(_receipt_path(job), roots=[tmp_path])

    assert resolution["status"] == "ready"
    assert resolution["blockers"] == []
    assert resolution["receipt"]["status"] == "ready"
    # The recorded authoring path is replaced by the one that resolved here.
    assert (
        resolution["receipt"]["bundle_path"]
        == str((job / "adp_retained_scene_gpu_render_bundle.zip").resolve())
    )
    assert resolution["receipt"]["execution_authority"]["path"].endswith(
        "provider_runtime/execution_authority.json"
    )


def test_paired_artifixer_native_receipt_projects_onto_the_live_launch_contract(tmp_path):
    receipt_path = _artifixer_job(tmp_path)

    resolution = resolve_host_resident_bundle_receipt(receipt_path, roots=[tmp_path])

    assert resolution["status"] == "ready"
    assert resolution["blockers"] == []
    projected = resolution["receipt"]
    assert projected["status"] == "ready"
    assert projected["native_receipt_status"] == (
        "sealed_rehearsal_passed_no_upload_no_execution"
    )
    assert projected["implementation_commit"] == "a" * 40
    assert projected["bundle_sha256"] == projected["bundle"]["sha256"]
    assert projected["bundle_path"] == str(
        (receipt_path.parent / "public_scene_artifixer3d_provider_bundle.zip").resolve()
    )


def test_nonpaired_artifixer_receipt_does_not_become_live_ready(tmp_path):
    receipt_path = _artifixer_job(tmp_path, pipeline_mode="broad_repair")

    resolution = resolve_host_resident_bundle_receipt(receipt_path, roots=[tmp_path])

    assert resolution["status"] == "ready"
    assert resolution["receipt"]["status"] == (
        "sealed_rehearsal_passed_no_upload_no_execution"
    )


def test_receipt_without_its_bundle_cannot_report_ready(tmp_path):
    job = _job(tmp_path)
    (job / "adp_retained_scene_gpu_render_bundle.zip").unlink()

    resolution = resolve_host_resident_bundle_receipt(_receipt_path(job), roots=[tmp_path])

    assert resolution["status"] == "blocked"
    assert "host_resident_input_unresolved:bundle" in resolution["blockers"]
    assert resolution["receipt"]["status"] == "not_host_resident"


def test_bundle_bytes_outside_the_control_plane_roots_are_refused(tmp_path):
    """The exact 2026-08-12 arrangement: right bytes, foreign directory."""

    control_plane = tmp_path / "var-lib-blueprint"
    control_plane.mkdir()
    authoring_tree = tmp_path / "Users" / "author"
    authoring_tree.mkdir(parents=True)
    job = _job(authoring_tree)

    resolution = resolve_host_resident_bundle_receipt(
        _receipt_path(job), roots=[control_plane]
    )

    assert resolution["status"] == "blocked"
    assert (
        "host_resident_input_outside_control_plane_roots:receipt" in resolution["blockers"]
    )
    assert (
        "host_resident_input_outside_control_plane_roots:bundle" in resolution["blockers"]
    )
    assert resolution["receipt"]["status"] == "not_host_resident"


def test_bundle_bytes_that_do_not_match_the_recorded_digest_do_not_resolve(tmp_path):
    job = _job(tmp_path)
    (job / "adp_retained_scene_gpu_render_bundle.zip").write_bytes(b"different-bytes")

    resolution = resolve_host_resident_bundle_receipt(_receipt_path(job), roots=[tmp_path])

    assert "host_resident_input_unresolved:bundle" in resolution["blockers"]
    assert resolution["receipt"]["status"] == "not_host_resident"


def test_a_symlinked_bundle_does_not_resolve(tmp_path):
    job = _job(tmp_path)
    archive = job / "adp_retained_scene_gpu_render_bundle.zip"
    payload = archive.read_bytes()
    archive.unlink()
    elsewhere = tmp_path / "elsewhere.zip"
    elsewhere.write_bytes(payload)
    archive.symlink_to(elsewhere)

    resolution = resolve_host_resident_bundle_receipt(_receipt_path(job), roots=[tmp_path])

    assert "host_resident_input_unresolved:bundle" in resolution["blockers"]


def test_a_missing_receipt_raises_rather_than_returning_a_verdict(tmp_path):
    with pytest.raises(HostResidentInputError):
        resolve_host_resident_bundle_receipt(tmp_path / "absent.json", roots=[tmp_path])


def test_a_not_host_resident_receipt_is_refused_by_the_paid_bundle_validator(tmp_path):
    """The link that matters: the allocator admits only ``ready``."""

    job = _job(tmp_path)
    (job / "adp_retained_scene_gpu_render_bundle.zip").unlink()
    resolution = resolve_host_resident_bundle_receipt(_receipt_path(job), roots=[tmp_path])

    with pytest.raises(ValueError) as excinfo:
        validate_retained_scene_render_bundle(resolution["receipt"])

    assert "schema_or_status_invalid" in str(excinfo.value)


def _profile(*, bundle_path: str, receipt_path: str) -> dict:
    return {
        "profile_id": "adp-retained-scene-render-live-test",
        "execution_admission": {"live_enabled": True, "blockers": []},
        "immutable_inputs": [
            {"name": "retained_scene_render_bundle", "path": bundle_path, "digest": "sha256:x"}
        ],
        "allocator": {
            "argv": [
                "--adapter-output",
                "{launch_run_root}/allocator/result.json",
                "--adp-retained-scene-render-bundle-receipt",
                receipt_path,
                "--adp-retained-scene-render-max-hourly-rate-usd",
                "2.0",
            ]
        },
    }


def test_a_profile_binding_an_authoring_path_is_refused(tmp_path):
    control_plane = tmp_path / "var-lib-blueprint"
    control_plane.mkdir()
    profile = _profile(
        bundle_path="/Users/author/workspace/validation/job/bundle.zip",
        receipt_path="/Users/author/workspace/validation/job/receipt.json",
    )

    blockers = launch_profile_residency_blockers(profile, roots=[control_plane])

    assert blockers == [
        "launch_profile_input_not_host_resident:allocator_argument:"
        "adp-retained-scene-render-bundle-receipt",
        "launch_profile_input_not_host_resident:immutable_input:retained_scene_render_bundle",
    ]


def test_a_profile_bound_under_a_control_plane_root_passes(tmp_path):
    control_plane = tmp_path / "var-lib-blueprint"
    (control_plane / "job").mkdir(parents=True)

    profile = _profile(
        bundle_path=str(control_plane / "job" / "bundle.zip"),
        receipt_path=str(control_plane / "job" / "receipt.json"),
    )

    assert launch_profile_residency_blockers(profile, roots=[control_plane]) == []


def test_run_root_placeholders_are_outputs_and_are_not_checked(tmp_path):
    control_plane = tmp_path / "var-lib-blueprint"
    control_plane.mkdir()
    profile = {
        "allocator": {"argv": ["--adapter-output", "{launch_run_root}/allocator/result.json"]}
    }

    assert launch_profile_residency_blockers(profile, roots=[control_plane]) == []


def test_without_control_plane_roots_residency_is_not_claimed(tmp_path):
    profile = _profile(
        bundle_path="/Users/author/workspace/validation/job/bundle.zip",
        receipt_path="/Users/author/workspace/validation/job/receipt.json",
    )

    assert launch_profile_residency_blockers(profile, roots=[]) == []
    assert path_is_host_resident("/anywhere/at/all", [])


def test_configured_roots_come_from_the_environment_when_set(tmp_path, monkeypatch):
    present = tmp_path / "present"
    present.mkdir()
    monkeypatch.setenv(
        LAUNCH_INPUT_ROOTS_ENV, f"{present}:{tmp_path / 'absent'}"
    )

    assert configured_launch_input_roots() == (present.resolve(),)


def test_only_a_live_enabled_profile_blocks_the_startup_report(tmp_path):
    control_plane = tmp_path / "var-lib-blueprint"
    control_plane.mkdir()
    profile_dir = control_plane / "profiles"
    profile_dir.mkdir()
    foreign = _profile(
        bundle_path="/Users/author/job/bundle.zip",
        receipt_path="/Users/author/job/receipt.json",
    )
    (profile_dir / "live.json").write_text(json.dumps(foreign), encoding="utf-8")
    dry = {**foreign, "profile_id": "dry", "execution_admission": {"live_enabled": False}}
    (profile_dir / "dry.json").write_text(json.dumps(dry), encoding="utf-8")

    report = published_profile_residency_report(profile_dir, roots=[control_plane])

    assert report["status"] == "blocked"
    assert [row["status"] for row in report["profiles"]] == [
        "not_host_resident",
        "not_host_resident",
    ]
    assert all("adp-retained-scene-render-live-test" in row for row in report["blockers"])
    assert not any(":dry" in row for row in report["blockers"])


def test_the_startup_guard_reports_residency_without_taking_intake_down(monkeypatch):
    """The guard is an ExecStartPre for the intake service. One unusable
    published profile must not answer "this profile cannot launch" with "no
    launch of any kind can be accepted"."""

    from blueprint_pipeline import production_runtime_env_guard as guard

    monkeypatch.setenv(guard.LAUNCH_PROFILE_DIR_ENV, "/etc/blueprint/profiles")
    report = guard.build_production_runtime_env_guard(
        env={
            guard.LAUNCH_PROFILE_DIR_ENV: "/etc/blueprint/profiles",
            "BLUEPRINT_LAUNCH_PROOF_MODE": "production",
            "PRIVACY_PIPELINE_ENABLED": "1",
            "PRIVACY_FAIL_CLOSED": "1",
            "PIPELINE_SYNC_REQUIRED": "1",
            "RETRIEVAL_REQUIRE_PRIVACY_SAFE_VIDEO": "1",
        },
        import_module=lambda name: object(),
        reconcile_spend_authority=lambda: {"status": "consistent"},
        reconcile_launch_catalog=lambda: {"status": "consistent"},
        residency_report=lambda _: {
            "status": "blocked",
            "blockers": ["launch_profile_input_not_host_resident:x:live-profile"],
        },
    )

    assert report["status"] == "ready"
    assert report["blockers"] == []
    # Reported, so a rebuilt host surfaces it at every start.
    assert report["launch_input_residency"]["status"] == "blocked"


def test_the_profile_builder_refuses_a_receipt_from_an_authoring_tree(
    tmp_path, monkeypatch
):
    import importlib.util

    control_plane = tmp_path / "var-lib-blueprint"
    control_plane.mkdir()
    authoring_tree = tmp_path / "Users" / "author"
    authoring_tree.mkdir(parents=True)
    job = _job(authoring_tree)
    monkeypatch.setenv(LAUNCH_INPUT_ROOTS_ENV, str(control_plane))

    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_residency_profile_builder",
        repo_root / "scripts" / "build_retained_scene_render_live_profile.py",
    )
    builder = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(builder)

    with pytest.raises(Exception) as excinfo:
        builder.build_retained_scene_render_live_profile(
            bundle_receipt_path=_receipt_path(job),
            request_manifest_path=job / "provider_runtime/source_request_manifest.json",
            attempt_authority_path=job / "provider_runtime/execution_authority.json",
            source_commit="0" * 40,
            raw_manifest_uri="https://raw.githubusercontent.com/example/repo/x/r.json",
        )

    assert "host_resident_input_outside_control_plane_roots" in str(excinfo.value)


def test_a_fully_resident_profile_directory_reports_ready(tmp_path):
    control_plane = tmp_path / "var-lib-blueprint"
    job = control_plane / "job"
    job.mkdir(parents=True)
    profile_dir = control_plane / "profiles"
    profile_dir.mkdir()
    (profile_dir / "live.json").write_text(
        json.dumps(
            _profile(
                bundle_path=str(job / "bundle.zip"), receipt_path=str(job / "receipt.json")
            )
        ),
        encoding="utf-8",
    )

    report = published_profile_residency_report(profile_dir, roots=[control_plane])

    assert report["status"] == "ready"
    assert report["blockers"] == []


def test_the_served_catalog_demotes_a_profile_this_host_cannot_run(tmp_path, monkeypatch):
    """Enforcement where the consequence is proportionate: the profile stops
    being selectable, every other profile stays reachable."""

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_task_evaluation_launch_dispatcher import _profile, _write

    from blueprint_pipeline.task_evaluation_launch_catalog import build_catalog_payload

    # The resident profile writes its inputs under tmp_path, so tmp_path is
    # this fixture's control plane; the foreign one points outside it.
    monkeypatch.setenv(LAUNCH_INPUT_ROOTS_ENV, str(tmp_path))
    profile_dir = tmp_path / "profiles"
    resident = _profile(tmp_path)
    foreign = json.loads(json.dumps(resident))
    foreign["profile_id"] = "foreign-inputs-profile"
    foreign["allocator"]["argv"] = [
        *foreign["allocator"]["argv"],
        "--adp-retained-scene-render-bundle-receipt",
        "/Users/author/workspace/validation/job/receipt.json",
    ]
    from blueprint_pipeline.task_evaluation_launch_dispatcher import canonical_digest

    foreign["profile_digest"] = canonical_digest(foreign, digest_field="profile_digest")
    _write(profile_dir / f"{resident['profile_id']}.json", resident)
    _write(profile_dir / f"{foreign['profile_id']}.json", foreign)

    rows = {
        row["profile_id"]: row
        for row in json.loads(build_catalog_payload(profile_dir).decode("utf-8"))
    }

    # The whole catalog still projects, and the unusable one is not selectable.
    assert set(rows) == {resident["profile_id"], "foreign-inputs-profile"}
    assert rows["foreign-inputs-profile"]["execution_admission"]["live_enabled"] is False
    assert any(
        "not_host_resident" in blocker
        for blocker in rows["foreign-inputs-profile"]["execution_admission"]["blockers"]
    )
    assert rows[resident["profile_id"]]["execution_admission"]["live_enabled"] is True
