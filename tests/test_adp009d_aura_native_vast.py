from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from PIL import Image

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import adp009d_aura_native_vast as subject
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionGrant
from blueprint_pipeline.provider_runtime_bundle_contract import (
    PROVIDER_RUNTIME_BUNDLE_KINDS,
    provider_runtime_contract_blockers,
)
from blueprint_pipeline.vast_provider_adapter import (
    _probe_shell_script,
    _resolve_launch_mode,
)


def _exact_cameras() -> list[dict]:
    rows = []
    for index, camera_id in enumerate(subject.DEFAULT_CAMERA_IDS):
        rows.append(
            {
                "id": camera_id,
                "spec": {
                    "intrinsics": {
                        "width": 64,
                        "height": 48,
                        "fx": 50.0,
                        "fy": 50.0,
                        "cx": 32.0,
                        "cy": 24.0,
                    },
                    "pose": {
                        "T_world_camera_opencv": [
                            [1.0, 0.0, 0.0, float(index)],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    },
                },
            }
        )
    return rows


def _native_manifest(root: Path) -> dict:
    renders = []
    for index, camera_id in enumerate(subject.DEFAULT_CAMERA_IDS):
        path = root / f"{camera_id}.png"
        Image.new("RGB", (64, 48), color=(index * 40 + 1, 2, 3)).save(path)
        renders.append(
            {
                "camera_id": camera_id,
                "relative_path": path.name,
                "digest": f"sha256:reference-{camera_id}",
                "width": 64,
                "height": 48,
            }
        )
    value = {
        "schema_version": "sealed_camera_render_manifest.v1",
        "status": "rendered_exact_cameras",
        "rendered_by": "aurafusion360_native_2d_gaussian_rasterizer",
        "renders": renders,
    }
    value["sealed_camera_render_manifest_digest"] = canonical_digest(
        value, digest_field="sealed_camera_render_manifest_digest"
    )
    return value


def test_materializer_binds_metric_pose_reference_and_sealed_ply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exact_path = tmp_path / "exact.json"
    exact_path.write_text(json.dumps(_exact_cameras()), encoding="utf-8")
    native = _native_manifest(tmp_path)
    native_path = tmp_path / "native.json"
    native_path.write_text(json.dumps(native), encoding="utf-8")
    ply = tmp_path / "aura.ply"
    ply.write_bytes(b"sealed-aura")

    real_sha256 = subject._sha256

    def fake_sha256(path: Path) -> str:
        if path == ply:
            return subject.EXPECTED_AURA_PLY_SHA256
        for row in native["renders"]:
            if path.name == row["relative_path"]:
                return row["digest"]
        return real_sha256(path)

    monkeypatch.setattr(subject, "_sha256", fake_sha256)
    output = tmp_path / "probe.json"
    result = subject.materialize_aura_native_exact_camera_probe(
        exact_camera_path=exact_path,
        aura_native_manifest_path=native_path,
        aura_ply_path=ply,
        output_path=output,
    )

    assert result["status"] == "materialized_unexecuted"
    assert result["depth_output"] == "surf_depth_expected_camera_z_m"
    assert result["depth_ratio"] == 0.0
    assert result["conformance_thresholds"] == subject.FROZEN_THRESHOLDS
    assert (
        result["threshold_definition_commit"]
        == subject.THRESHOLD_DEFINITION_COMMIT
    )
    assert result["aura_ply_sha256"] == subject.EXPECTED_AURA_PLY_SHA256
    assert len(result["camera_configs"]) == 2
    assert result["candidate_policy_queried"] is False
    assert result["manifest_digest"] == canonical_digest(
        result, digest_field="manifest_digest"
    )
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_materializer_rejects_offcenter_camera_or_changed_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exact = _exact_cameras()
    exact[0]["spec"]["intrinsics"]["cx"] = 31.5
    exact_path = tmp_path / "exact.json"
    exact_path.write_text(json.dumps(exact), encoding="utf-8")
    native = _native_manifest(tmp_path)
    native_path = tmp_path / "native.json"
    native_path.write_text(json.dumps(native), encoding="utf-8")
    ply = tmp_path / "aura.ply"
    ply.write_bytes(b"sealed-aura")
    monkeypatch.setattr(
        subject,
        "_sha256",
        lambda path: (
            subject.EXPECTED_AURA_PLY_SHA256
            if path == ply
            else next(
                (
                    row["digest"]
                    for row in native["renders"]
                    if path.name == row["relative_path"]
                ),
                "sha256:" + "0" * 64,
            )
        ),
    )

    with pytest.raises(ValueError, match="offcenter_pinhole_unsupported"):
        subject.materialize_aura_native_exact_camera_probe(
            exact_camera_path=exact_path,
            aura_native_manifest_path=native_path,
            aura_ply_path=ply,
            output_path=tmp_path / "probe.json",
        )

    exact[0]["spec"]["intrinsics"]["cx"] = 32.0
    exact_path.write_text(json.dumps(exact), encoding="utf-8")
    changed = copy.deepcopy(native)
    changed["renders"][0]["digest"] = "sha256:changed"
    changed["sealed_camera_render_manifest_digest"] = canonical_digest(
        changed, digest_field="sealed_camera_render_manifest_digest"
    )
    native_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="reference_digest_mismatch"):
        subject.materialize_aura_native_exact_camera_probe(
            exact_camera_path=exact_path,
            aura_native_manifest_path=native_path,
            aura_ply_path=ply,
            output_path=tmp_path / "probe.json",
        )


def _allocator_args(tmp_path: Path, *, execute: bool) -> list[str]:
    values = [
        "gpu-canary",
        "--probe-kind",
        subject.PROBE_KIND,
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
        "adp009d-aura-native",
        "--adp009d-aura-native-probe-manifest",
        str(tmp_path / "probe.json"),
        "--adp009d-aura-source-root",
        str(tmp_path / "aura-source"),
        "--adp-job-dir",
        str(tmp_path / "job"),
        "--adp-max-hourly-rate-usd",
        "0.8",
        "--adp-max-spend-usd",
        "1.25",
        "--adp-hard-ttl-seconds",
        "2700",
    ]
    if execute:
        values.append("--execute")
    return values


@pytest.mark.parametrize("execute", [False, True])
def test_allocator_routes_native_aura_only_through_canonical_grant(
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
        "build_aura_native_live_camera_bundle",
        lambda **kwargs: {
            "status": "ready",
            "bundle_sha256": "sha256:" + "b" * 64,
            "input_digest": "sha256:" + "c" * 64,
            "source_commit": subject.SOURCE_COMMIT,
            "source_tree": subject.SOURCE_TREE,
            "aura_ply_sha256": subject.EXPECTED_AURA_PLY_SHA256,
        },
    )

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "completed" if kwargs["execute"] else "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_aura_native_live_camera_vast", fake_run)

    assert allocator.main(_allocator_args(tmp_path, execute=execute)) == 0
    assert observed["execute"] is execute
    assert (
        isinstance(observed["paid_resource_admission_grant"], PaidResourceAdmissionGrant)
        is execute
    )
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["probe_kind"] == subject.PROBE_KIND
    assert admission["retry_cap"] == 0
    assert admission["candidate_policy_queried"] is False
    assert admission["hard_cap_usd"] == 1.25


def test_native_aura_transport_has_dedicated_contract_and_ssh_launch() -> None:
    root = Path(__file__).resolve().parents[1]
    entrypoint = (
        root / "scripts/run_adp009d_aura_native_provider_runtime.sh"
    ).read_text(encoding="utf-8")
    runner = (root / "scripts/adp009d_aura_native_provider_runner.py").read_text(
        encoding="utf-8"
    )

    assert "adp009d_aura_native" in PROVIDER_RUNTIME_BUNDLE_KINDS
    assert (
        provider_runtime_contract_blockers(
            provider_bundle_kind="adp009d_aura_native",
            entrypoint_text=entrypoint,
            runner_text=runner,
        )
        == []
    )
    assert (
        _resolve_launch_mode(
            requested="auto",
            enable_isaac_smoke=True,
            enable_blueprint_bundle=True,
            provider_bundle_kind="adp009d_aura_native",
        )
        == "ssh_direct"
    )
    launch = _probe_shell_script(
        "https://example.com",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        provider_bundle_kind="adp009d_aura_native",
    )
    assert "run_adp009d_aura_native_provider_runtime.sh" in launch
    assert "adp009d_aura_native_provider_runtime_output.zip" in launch


def test_native_aura_uses_cuda_driver_floor_without_omniverse_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict = {}

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "dry_run_ready"}

    monkeypatch.setattr(subject, "run_arena_native_control_vast", fake_run)

    result = subject.run_aura_native_live_camera_vast(
        job_dir=tmp_path,
        prepared_bundle={"status": "ready"},
        paid_resource_admission_grant=None,
        execute=False,
    )

    assert result["status"] == "dry_run_ready"
    assert observed["minimum_driver_version"] == "550.54.14"
    assert observed["require_known_supported_isaac_driver"] is False
    assert observed["min_gpu_ram_mb"] == 46_000
    assert observed["min_compute_cap"] == 860
