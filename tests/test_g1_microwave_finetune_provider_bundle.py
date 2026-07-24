from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tarfile
import urllib.request
import zipfile

import pytest

from blueprint_pipeline import g1_microwave_finetune_provider_bundle as bundle


def _dataset_archive(tmp_path: Path, *, include_appledouble: bool = False) -> Path:
    root = tmp_path / "microwave_owned_lerobot_v21_20260717"
    root.mkdir()
    (root / "groot_n17_finetune_preflight.json").write_text(
        json.dumps(
            {
                "status": "qualified_exact_groot_n1_7_training_data_preflight",
                "bounded_finetune_plan": {
                    "warm_starts_from_sealed_sonic_checkpoint": True,
                    "max_steps": 500,
                },
            }
        ),
        encoding="utf-8",
    )
    if include_appledouble:
        data = root / "data/chunk-000"
        data.mkdir(parents=True)
        (data / "._episode_000000.parquet").write_bytes(b"not parquet")
    archive = tmp_path / "dataset.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(root, arcname=root.name)
    return archive


def test_provider_bundle_is_exact_and_bootstrap_is_valid_bash(tmp_path: Path) -> None:
    output = tmp_path / "provider_bundle.zip"
    result = bundle.build_provider_bundle(
        dataset_archive=_dataset_archive(tmp_path),
        output_path=output,
    )

    assert result["status"] == "qualified_provider_bundle"
    assert result["image_ref"].endswith(
        "sha256:ab8fbccb714242b55811aa5142933001dfba76d56b5cc29dead4d0bdf1346e88"
    )
    assert result["training"]["warm_start_path"] == "/opt/blueprint/ckpts/sonic"
    assert result["training"]["max_steps"] == 500
    assert (
        result["claim_boundary"][
            "worker_requires_open_loop_improvement_over_sealed_warm_start"
        ]
        is True
    )
    assert result["claim_boundary"]["open_loop_exact_owned_training_trajectory_only"] is True
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {
            bundle.DATASET_ARCHIVE_NAME,
            bundle.WORKER_NAME,
            bundle.MANIFEST_NAME,
        }
        manifest = json.loads(archive.read(bundle.MANIFEST_NAME))
    assert manifest["dataset"]["sha256"] == result["dataset"]["sha256"]
    assert manifest["worker"]["sha256"] == result["worker"]["sha256"]

    script = bundle.render_provider_bootstrap(
        expected_bundle_sha256=result["bundle"]["sha256"]
    )
    syntax = subprocess.run(
        ["bash", "-n"], input=script, check=False, capture_output=True, text=True
    )
    assert syntax.returncode == 0, syntax.stderr
    assert bundle.BUNDLE_URL_ENV in script
    assert bundle.OUTPUT_PUT_URL_ENV in script
    assert "upload_bootstrap_failure" in script
    assert "g1_microwave_finetune_provider_bootstrap_exit_" in script
    assert 'write_bootstrap_stage bundle_download_started' in script
    assert 'write_bootstrap_stage bundle_verified' in script
    assert 'write_bootstrap_stage dataset_manifest_verified' in script
    assert 'write_bootstrap_stage worker_invocation_started' in script
    assert 'payload["provider_bootstrap_phase"] = bootstrap_phase' in script
    assert result["bundle"]["sha256"] in script
    assert "urllib.request.build_opener(NoRedirect()).open" in script
    assert "NoRedirect().build_opener()" not in script
    embedded_python = script.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    compile(embedded_python, "provider_bundle_downloader", "exec")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    assert hasattr(urllib.request.build_opener(NoRedirect()), "open")


def test_provider_bundle_is_deterministic(tmp_path: Path) -> None:
    dataset = _dataset_archive(tmp_path)
    first = bundle.build_provider_bundle(
        dataset_archive=dataset,
        output_path=tmp_path / "first.zip",
    )
    second = bundle.build_provider_bundle(
        dataset_archive=dataset,
        output_path=tmp_path / "second.zip",
    )

    assert first["bundle"]["sha256"] == second["bundle"]["sha256"]


def test_provider_bundle_rejects_macos_appledouble_members(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match="g1_microwave_provider_bundle_appledouble_members_forbidden",
    ):
        bundle.build_provider_bundle(
            dataset_archive=_dataset_archive(tmp_path, include_appledouble=True),
            output_path=tmp_path / "provider_bundle.zip",
        )
