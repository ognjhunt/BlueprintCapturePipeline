from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline.adp009d_ovrtx_vast import build_ovrtx_live_camera_bundle
from blueprint_pipeline.common import sha256_file
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.provider_runtime_bundle_contract import (
    provider_runtime_contract_blockers,
)
from blueprint_pipeline.vast_provider_adapter import (
    _blueprint_bundle_preflight,
    _probe_shell_script,
)


def _sha(path: Path) -> str:
    return f"sha256:{sha256_file(path)}"


def _probe(tmp_path: Path) -> Path:
    particlefield = tmp_path / "aura.usdc"
    particlefield.write_bytes(b"particlefield")
    rows = []
    for camera_id in ("external", "wrist"):
        config = tmp_path / f"{camera_id}.json"
        config.write_text(
            json.dumps(
                {
                    "camera_id": camera_id,
                    "metric_depth_aov": "DistanceToCameraSD",
                    "warmup_frames": 40,
                }
            ),
            encoding="utf-8",
        )
        rows.append(
            {
                "camera_id": camera_id,
                "configuration_path": str(config),
                "configuration_sha256": _sha(config),
            }
        )
    probe = {
        "schema_version": "adp009d_ovrtx_live_camera_probe.v1",
        "status": "materialized_unexecuted",
        "particlefield_path": str(particlefield),
        "particlefield_sha256": _sha(particlefield),
        "camera_configs": rows,
        "metric_depth_aov": "DistanceToCameraSD",
        "unitless_depth_sd_used": False,
    }
    probe["manifest_digest"] = canonical_digest(probe, digest_field="manifest_digest")
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(probe), encoding="utf-8")
    return path


def test_ovrtx_bundle_is_deterministic_and_contract_complete(tmp_path: Path) -> None:
    probe = _probe(tmp_path)
    kwargs = {
        "probe_manifest_path": probe,
        "implementation_commit": "a" * 40,
        "generated_at": "2026-08-06T00:00:00+00:00",
    }
    first = build_ovrtx_live_camera_bundle(job_dir=tmp_path / "first", **kwargs)
    second = build_ovrtx_live_camera_bundle(job_dir=tmp_path / "second", **kwargs)
    assert first["bundle_sha256"] == second["bundle_sha256"]
    assert first["particlefield_sha256"] == _sha(tmp_path / "aura.usdc")
    with zipfile.ZipFile(first["bundle_path"]) as archive:
        names = set(archive.namelist())
        assert {
            "provider_runtime/assets/aura_gaussian_surflets.usdc",
            "provider_runtime/configs/external.ovrtx.json",
            "provider_runtime/configs/wrist.ovrtx.json",
            "provider_runtime/run_adp009d_ovrtx_provider_runtime.sh",
        } <= names
        entrypoint = archive.read(
            "provider_runtime/run_adp009d_ovrtx_provider_runtime.sh"
        ).decode()
        runner = archive.read("provider_runtime/adp009d_ovrtx_provider_runner.py").decode()
    assert provider_runtime_contract_blockers(
        provider_bundle_kind="adp009d_ovrtx",
        entrypoint_text=entrypoint,
        runner_text=runner,
    ) == []
    script = tmp_path / "entrypoint.sh"
    script.write_text(entrypoint, encoding="utf-8")
    assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0
    preflight = _blueprint_bundle_preflight(
        job_dir=tmp_path / "preflight",
        generated_at="fixed",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp009d_ovrtx",
        bundle_path=Path(first["bundle_path"]),
        provider_bundle_url="https://example.com/bundle.zip?signature=redacted",
        provider_output_put_url="https://example.com/output.zip?signature=redacted",
    )
    assert preflight["status"] == "passed"
    launch = _probe_shell_script(
        "https://example.com/heartbeat",
        enable_blueprint_bundle=True,
        provider_bundle_kind="adp009d_ovrtx",
    )
    assert "run_adp009d_ovrtx_provider_runtime.sh" in launch
    assert "adp009d_ovrtx_provider_runtime_output.zip" in launch


def test_ovrtx_bundle_rejects_changed_camera_config(tmp_path: Path) -> None:
    probe_path = _probe(tmp_path)
    probe = json.loads(probe_path.read_text())
    Path(probe["camera_configs"][0]["configuration_path"]).write_text(
        "changed", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="camera_config_digest_mismatch"):
        build_ovrtx_live_camera_bundle(
            job_dir=tmp_path / "job",
            probe_manifest_path=probe_path,
            implementation_commit="a" * 40,
            generated_at="2026-08-06T00:00:00+00:00",
        )
