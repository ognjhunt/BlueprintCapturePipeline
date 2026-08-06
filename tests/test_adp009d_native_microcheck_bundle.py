from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import adp009d_isaac_runtime as isaac_runtime
from blueprint_pipeline.adp009d_native_microcheck_bundle import (
    DEFAULT_IMAGE,
    PROBE_KIND,
    TARGET_COLLIDER_PRIM,
    build_native_microcheck_bundle,
)
from blueprint_pipeline import adp009d_franka_vast as franka_vast
from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionGrant
from blueprint_pipeline.vast_provider_adapter import _blueprint_bundle_preflight


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    approved = tmp_path / "approved.usda"
    approved.write_text("#usda 1.0\ndef Xform \"Can\" {}\n", encoding="utf-8")
    sage = tmp_path / "sage.usda"
    target_name = TARGET_COLLIDER_PRIM.rsplit("/", 1)[-1]
    sage.write_text(
        (
            '#usda 1.0\n(defaultPrim = "Root")\n\n'
            'def Xform "Root"\n{\n'
            f'    def Xform "{target_name}"\n    {{\n    }}\n'
            '}\n'
        ),
        encoding="utf-8",
    )
    harness = tmp_path / "harness.json"
    harness.write_text('{"schema_version":"test"}\n', encoding="utf-8")
    bindings = {
        "approved_can.usda": _digest(approved),
        "sage_collision.usd": _digest(sage),
    }
    return approved, sage, harness, bindings


def test_bundle_is_deterministic_and_keeps_sealed_sources_unchanged(tmp_path: Path) -> None:
    approved, sage, harness, bindings = _inputs(tmp_path)
    approved_before = approved.read_bytes()
    sage_before = sage.read_bytes()
    kwargs = {
        "approved_can_path": approved,
        "sage_collision_path": sage,
        "harness_manifest_path": harness,
        "implementation_commit": "a" * 40,
        "generated_at": "fixed",
        "expected_asset_bindings": bindings,
    }
    first = build_native_microcheck_bundle(job_dir=tmp_path / "first", **kwargs)
    second = build_native_microcheck_bundle(job_dir=tmp_path / "second", **kwargs)

    assert first["probe_kind"] == PROBE_KIND
    assert first["container_image"] == DEFAULT_IMAGE
    assert first["bundle_sha256"] == second["bundle_sha256"]
    assert first["candidate_policy_queried"] is False
    assert approved.read_bytes() == approved_before
    assert sage.read_bytes() == sage_before
    Usd = pytest.importorskip("pxr.Usd")
    source_stage = Usd.Stage.Open(str(sage))
    overlay_stage = Usd.Stage.Open(
        str(Path(first["bundle_path"]).parent / "provider_runtime/assets/sage_collision_overlay.usda")
    )
    assert source_stage.GetPrimAtPath(TARGET_COLLIDER_PRIM).IsActive()
    assert not overlay_stage.GetPrimAtPath(TARGET_COLLIDER_PRIM).IsActive()
    with zipfile.ZipFile(first["bundle_path"]) as archive:
        names = set(archive.namelist())
        overlay = archive.read("provider_runtime/assets/sage_collision_overlay.usda").decode()
        entrypoint = archive.read("provider_runtime/run_adp_arena_provider_runtime.sh").decode()
    assert TARGET_COLLIDER_PRIM.rsplit("/", 1)[-1] in overlay
    assert "active = false" in overlay
    assert "adp009d_native_microcheck.json" in entrypoint
    assert "provider_runtime/assets/approved_can.usda" in names
    assert "provider_runtime/assets/sage_collision.usd" in names


def test_bundle_passes_existing_arena_transport_preflight(tmp_path: Path) -> None:
    approved, sage, harness, bindings = _inputs(tmp_path)
    receipt = build_native_microcheck_bundle(
        job_dir=tmp_path / "bundle",
        approved_can_path=approved,
        sage_collision_path=sage,
        harness_manifest_path=harness,
        implementation_commit="b" * 40,
        generated_at="fixed",
        expected_asset_bindings=bindings,
    )

    preflight = _blueprint_bundle_preflight(
        job_dir=tmp_path / "preflight",
        generated_at="fixed",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="adp009d_isaac",
        bundle_path=Path(receipt["bundle_path"]),
        provider_bundle_url="https://example.com/bundle.zip?sig=redacted",
        provider_output_put_url="https://example.com/output.zip?sig=redacted",
    )

    assert preflight["status"] == "passed"
    assert preflight["blockers"] == []


def test_bundle_rejects_unbound_asset_bytes(tmp_path: Path) -> None:
    approved, sage, harness, bindings = _inputs(tmp_path)
    bindings["sage_collision.usd"] = "sha256:" + "0" * 64

    with pytest.raises(ValueError, match="adp009d_bound_asset_digest_mismatch:sage_collision.usd"):
        build_native_microcheck_bundle(
            job_dir=tmp_path / "bundle",
            approved_can_path=approved,
            sage_collision_path=sage,
            harness_manifest_path=harness,
            implementation_commit="c" * 40,
            expected_asset_bindings=bindings,
        )


def test_runtime_binds_official_droid_franka_and_sealed_anchor() -> None:
    assert isaac_runtime.ARENA_REVISION == "8b4a3a47fc53de23e8205089d71109a2e2348acd"
    assert isaac_runtime.ISAAC_LAB_REVISION == "e57379c634b42db5a0fe9f754341be6e2a7c7c43"
    assert isaac_runtime.EXPECTED_ASSETS == {
        "approved_can.usda": "sha256:61c2a03bef425803d82cc5ef24ced5b2ccb4160923c53bb10c6ad0e3f52532ec",
        "sage_collision.usd": "sha256:b265706c24f6a8ace3ee6743fd138583c4e21d83f61b99a06fd435e6ac2d6b41",
    }
    assert isaac_runtime.ROBOT_BASE_POSITION_M == (3.4681748, -2.9100837, 0.2766791)
    assert isaac_runtime.CAN_START_POSITION_M == (
        3.4681748,
        -3.3100837,
        0.5264650138348479,
    )


def test_native_transport_prefers_one_48gb_class_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict = {}

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "dry_run_ready"}

    monkeypatch.setattr(franka_vast, "run_arena_native_control_vast", fake_run)

    result = franka_vast.run_adp009d_native_microcheck_vast(
        job_dir="job",
        prepared_bundle={"status": "ready"},
        paid_resource_admission_grant=None,
        execute=False,
    )

    assert result["status"] == "dry_run_ready"
    assert observed["min_gpu_ram_mb"] == 46_000
    assert observed["preferred_gpu_keywords"] == ("L40S", "RTX 6000 Ada", "RTX A6000")
    assert observed["provider_bundle_kind"] == "adp009d_isaac"


def _allocator_args(tmp_path: Path, *, execute: bool) -> list[str]:
    values = [
        "gpu-canary",
        "--probe-kind",
        PROBE_KIND,
        "--provider",
        "vast",
        "--provider-launch-request",
        str(tmp_path / "unused-request.json"),
        "--release-evidence",
        str(tmp_path / "unused-release.json"),
        "--model-cache-evidence",
        str(tmp_path / "unused-model.json"),
        "--preflight-bundle",
        str(tmp_path / "unused-preflight.json"),
        "--admission-out",
        str(tmp_path / "admission.json"),
        "--bound-request-out",
        str(tmp_path / "unused-bound.json"),
        "--adapter-output",
        str(tmp_path / "adapter.json"),
        "--pod-name",
        "adp009d-native",
        "--adp009d-approved-can",
        str(tmp_path / "can.usda"),
        "--adp009d-sage-collision",
        str(tmp_path / "sage.usd"),
        "--adp009d-harness-manifest",
        str(tmp_path / "harness.json"),
        "--adp-job-dir",
        str(tmp_path / "job"),
        "--adp-max-hourly-rate-usd",
        "1.0",
        "--adp-max-spend-usd",
        "4.0",
        "--adp-hard-ttl-seconds",
        "14400",
    ]
    if execute:
        values.append("--execute")
    return values


@pytest.mark.parametrize("execute", [False, True])
def test_allocator_routes_microcheck_only_through_canonical_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execute: bool
) -> None:
    observed: dict = {}
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )
    monkeypatch.setattr(
        allocator,
        "build_native_microcheck_bundle",
        lambda **kwargs: {
            "status": "ready",
            "bundle_sha256": "sha256:" + "b" * 64,
            "input_digest": "sha256:" + "c" * 64,
        },
    )

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "completed" if kwargs["execute"] else "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_adp009d_native_microcheck_vast", fake_run)

    assert allocator.main(_allocator_args(tmp_path, execute=execute)) == 0
    assert observed["execute"] is execute
    assert isinstance(observed["paid_resource_admission_grant"], PaidResourceAdmissionGrant) is execute
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["probe_kind"] == PROBE_KIND
    assert admission["retry_cap"] == 0
    assert admission["candidate_policy_queried"] is False
