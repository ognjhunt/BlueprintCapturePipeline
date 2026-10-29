"""Offline bundle-byte evidence; no engine or qualification is exercised."""
from __future__ import annotations

import hashlib
import json
import socket
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import team_candidate_quick10_bundle as team_bundle
from blueprint_pipeline import team_candidate_quick10_launch as launch
from blueprint_pipeline.native_task_arena_construction_bundle import (
    build_native_task_arena_construction_bundle,
)
from blueprint_pipeline.native_task_arena_controls_bundle import build_native_task_arena_controls_bundle
from blueprint_pipeline.native_task_arena_execution_contract import (
    NATIVE_METROLOGY_RUNTIME_MODULE_NAMES,
    NATIVE_METROLOGY_RUNTIME_PACKAGE_FILES,
)
from tests.test_native_task_arena_bundle import (
    _articulated_packet, _native_bundle_preflight, _packet, _qualified_construction, _runtime_source_packet,
)
from tests.test_provider_runtime_import_closure import _build_real_canary_bundle
from tests.test_rigid_robot_free_probe_lane import build_world, staged_attempt
from tests.test_site_task_simulation_qualification import _qualified_world
from tests.test_team_candidate_quick10_launch import _staged_run_reported


def _bundle(tmp_path, monkeypatch, kind):
    if kind == "construction":
        receipt = build_native_task_arena_construction_bundle(
            job_dir=tmp_path / "bundle", packet_dir=_packet(tmp_path, scene_id="metrology-fixture"),
            runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
            implementation_commit="a" * 40, generated_at="fixed",
        )
    elif kind == "controls":
        packet, scene = _articulated_packet(tmp_path)
        receipt = build_native_task_arena_controls_bundle(
            job_dir=tmp_path / "bundle", packet_dir=packet,
            construction_result_path=_qualified_construction(tmp_path, scene),
            runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
            implementation_commit="a" * 40, generated_at="fixed",
        )
    elif kind == "rigid_probe":
        attempt = staged_attempt(build_world(tmp_path, monkeypatch))
        receipt = json.loads((attempt["root"] / "bundle/native_task_arena_provider_bundle_receipt.v1.json").read_text())
    else:
        receipt = _build_real_canary_bundle(tmp_path)
    return receipt


@pytest.mark.parametrize("kind", ["construction", "controls", "policy_canary", "rigid_probe"])
def test_real_archive_preserves_metrology_closure_bytes_and_manifest(tmp_path, monkeypatch, kind):
    receipt = _bundle(tmp_path, monkeypatch, kind)
    package = Path(team_bundle.__file__).resolve().parent
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        manifest = json.loads(archive.read("provider_runtime/adp_arena_provider_manifest.json"))
        rows = {row["relative_path"]: row for row in
                [*manifest["runtime_modules"], *manifest["runtime_root_modules"]]}
        for name in (*NATIVE_METROLOGY_RUNTIME_MODULE_NAMES, *NATIVE_METROLOGY_RUNTIME_PACKAGE_FILES):
            expected = (package / name).read_bytes()
            member = "blueprint_pipeline/" + name
            actual = archive.read("provider_runtime/" + member)
            assert actual == expected
            assert rows[member] == {
                "relative_path": member, "size_bytes": len(actual),
                "sha256": "sha256:" + hashlib.sha256(actual).hexdigest(),
            }


@pytest.mark.parametrize("change", ["missing_security", "missing_package_init", "extra_package_module"])
def test_real_preflight_retains_exact_package_membership_refusals(tmp_path, monkeypatch, change):
    receipt = _bundle(tmp_path, monkeypatch, "controls")
    path = Path(receipt["bundle_path"])
    member = "provider_runtime/blueprint_pipeline/core/" + {
        "missing_security": "security_controls.py", "missing_package_init": "__init__.py",
        "extra_package_module": "unexpected.py",
    }[change]
    with zipfile.ZipFile(path) as archive:
        rows = [(info, archive.read(info.filename)) for info in archive.infolist()]
    with zipfile.ZipFile(path, "w") as archive:
        for info, data in rows:
            if info.filename != member:
                archive.writestr(info, data)
        if change == "extra_package_module":
            archive.writestr(member, "unexpected = True\n")
    network_calls = []

    def no_network(*_args, **_kwargs):
        network_calls.append(True)
        raise AssertionError("offline preflight reached network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    result = _native_bundle_preflight(tmp_path, receipt, name=change)
    assert result["status"] == "blocked"
    if change == "extra_package_module":
        assert "native_task_arena_provider_manifest_invalid" in result["blockers"]
    else:
        assert "provider_runtime_bundle_required_entries_missing" in result["blockers"]
        assert member in result["missing_zip_entries"]
    assert not network_calls


def test_missing_metrology_leaf_refuses_rehearsal_before_any_allocation(tmp_path, monkeypatch):
    world, _receipt = _qualified_world(tmp_path, monkeypatch)
    monkeypatch.setattr(team_bundle, "POLICY_RUNTIME_MODULE_NAMES", tuple(
        name for name in team_bundle.POLICY_RUNTIME_MODULE_NAMES
        if name != "site_task_native_metrology.py"
    ))
    calls = {"network": 0, "allocator": 0, "provider": 0}

    def forbid(boundary):
        def blocked(*_args, **_kwargs):
            calls[boundary] += 1
            raise AssertionError("unexpected offline " + boundary)
        return blocked

    monkeypatch.setattr(socket.socket, "connect", forbid("network"))
    world["runner"]["allocator_starter"] = forbid("allocator")
    from blueprint_pipeline import team_candidate_quick10_vast
    monkeypatch.setattr(team_candidate_quick10_vast, "run_arena_native_control_vast", forbid("provider"))
    failures = []
    original = launch._stage

    def observed_stage(**kwargs):
        try:
            return original(**kwargs)
        except ValueError as exc:
            failures.append(str(exc))
            raise

    monkeypatch.setattr(launch, "_stage", observed_stage)
    staged, reported = _staged_run_reported(world)
    assert staged["pending"] == reported["pending"] == 1
    assert reported["completed"] == 0 and not world["client"].posted
    assert len(failures) == 2
    assert all(reason.startswith("team_candidate_bundle_import_closure_incomplete:")
               and "native_task_arena_runtime.py->site_task_native_metrology:lazy" in reason
               and "native_task_robot_registry.py->site_task_native_metrology:lazy" in reason
               for reason in failures)
    attempts = sorted(world["job_dir"].glob("preflight_attempt_*.json"))
    assert len(attempts) == 2
    assert all(json.loads(path.read_text())["status"] == "refused_before_allocation"
               for path in attempts)
    assert not (world["job_dir"] / launch.INTENT_FILENAME).exists()
    assert calls == {"network": 0, "allocator": 0, "provider": 0}
    assert not world["allocator_calls"] and not world["provider"].calls
