import importlib.util
import hashlib
import json
import os
import sys
from types import SimpleNamespace
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import adp_aura_interiorgs_vast as bundle
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    build_paid_lane_admission,
    require_paid_resource_admission,
)


def _load_runner():
    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "test_adp_aura_interiorgs_provider_runner",
            scripts / "adp_aura_interiorgs_provider_runner.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_aura_lama_runtime_isolates_legacy_opencv_distribution() -> None:
    entrypoint = (
        Path(__file__).resolve().parents[1] / "scripts/run_adp_aura_interiorgs_provider_runtime.sh"
    ).read_text(encoding="utf-8")

    assert 'pip uninstall --python "${LAMA_PY}" opencv-python' in entrypoint
    assert "--reinstall --no-deps opencv-python-headless==4.5.5.64" in entrypoint
    assert 'metadata.version("opencv-python-headless") == "4.5.5.64"' in entrypoint
    assert 'numpy.__version__ == "1.21.6"' in entrypoint
    assert "aurafusion360_interiorgs_lama_opencv_validation_failed" in entrypoint


def test_aura_bundle_preflights_all_released_runtime_sources(tmp_path: Path) -> None:
    wonderworld = tmp_path / "WonderWorld"
    present = wonderworld / "marigold_module/LICENSE.txt"
    present.parent.mkdir(parents=True)
    present.write_text("Apache-2.0\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="adp_aura_interiorgs_wonderworld_runtime_source_missing",
    ):
        bundle._validated_runtime_source_rows(
            wonderworld,
            {
                "utils/LICENSE.txt": "marigold_module/LICENSE.txt",
                "utils/batchsize.py": "marigold_lcm/util/batchsize.py",
            },
            error="adp_aura_interiorgs_wonderworld_runtime_source_missing",
        )

    assert not (tmp_path / "provider_runtime.zip").exists()


def test_aura_live_run_arms_watchdog_before_provider_create_and_closes_after_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_path = tmp_path / "bundle.zip"
    bundle_path.write_bytes(b"immutable-bundle")
    prepared = {
        "status": "ready",
        "bundle_path": str(bundle_path),
        "bundle_sha256": "sha256:" + hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
    }
    grant = require_paid_resource_admission(
        build_paid_lane_admission(resource_class="vast_provider_adapter"),
        resource_class="vast_provider_adapter",
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )
    observed: dict[str, object] = {}
    started_id = tmp_path / "watchdog/started_vast_instance_id.txt"
    handle = SimpleNamespace(started_instance_id_path=started_id)
    monkeypatch.setattr(bundle, "_remaining_minutes", lambda **_kwargs: 240)
    def fake_stage(**kwargs: object) -> dict[str, object]:
        staging = Path(str(kwargs["job_dir"]))
        staging.mkdir(parents=True, exist_ok=True)
        for name in (
            "provider_bundle_url.txt",
            "provider_output_put_url.txt",
            "provider_output_get_url.txt",
        ):
            (staging / name).write_text("https://example.test/object\n", encoding="utf-8")
        return {"status": "completed"}

    monkeypatch.setattr(bundle, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(
        bundle,
        "cleanup_staged_wam_provider_objects",
        lambda _path: {"all_objects_absent": True},
    )
    monkeypatch.setattr(
        bundle,
        "arm_independent_vast_watchdog",
        lambda **_kwargs: (
            {"status": "armed", "watchdog_armed_before_allocation": True},
            handle,
        ),
    )

    def fake_provider(**kwargs: object) -> dict[str, object]:
        observed["started_instance_id_path"] = kwargs["started_instance_id_path"]
        provider_job = Path(str(kwargs["job_dir"]))
        provider_job.mkdir(parents=True, exist_ok=True)
        (provider_job / "vast_teardown_manifest.json").write_text(
            json.dumps(
                {
                    "vast_instance_ids": [7],
                    "continuing_spend_from_this_run": False,
                }
            ),
            encoding="utf-8",
        )
        return {
            "status": "completed",
            "vast_instance_ids": [7],
            "blockers": [],
            "estimated_cost_usd": 0.1,
        }

    monkeypatch.setattr(bundle, "run_vast_provider_adapter", fake_provider)
    monkeypatch.setattr(
        bundle,
        "_extract_provider_output",
        lambda _path, _destination: {
            "execution": {"status": "completed", "blockers": []},
            "blockers": [],
            "result_path": "result.json",
        },
    )

    def fake_close(**kwargs: object) -> dict[str, object]:
        observed["close"] = kwargs
        return {"status": "provider_terminal", "provider_absence_confirmed": True}

    monkeypatch.setattr(bundle, "close_independent_vast_watchdog", fake_close)

    result = bundle.run_aura_interiorgs_vast(
        job_dir=tmp_path / "run",
        paid_resource_admission_grant=grant,
        execute=True,
        prepared_bundle=prepared,
    )

    assert result["status"] == "completed", result
    assert observed["started_instance_id_path"] == started_id
    assert observed["close"]["instance_ids"] == [7]  # type: ignore[index]
    assert observed["close"]["provider_teardown_completed"] is True  # type: ignore[index]


def test_aura_adapter_overlay_preserves_publisher_tree(tmp_path: Path) -> None:
    runner = _load_runner()
    source = tmp_path / "adapter"
    destination = tmp_path / "publisher"
    (source / "Other-360/new-scene").mkdir(parents=True)
    (source / "Other-360/new-scene/train.config").write_text("new\n")
    (destination / "Other-360/publisher-scene").mkdir(parents=True)
    publisher = destination / "Other-360/publisher-scene/train.config"
    publisher.write_text("publisher\n")

    runner._copy_tree_additions(source, destination)

    assert publisher.read_text() == "publisher\n"
    assert (destination / "Other-360/new-scene/train.config").read_text() == "new\n"


def test_aura_adapter_overlay_rejects_overwrite(tmp_path: Path) -> None:
    runner = _load_runner()
    source = tmp_path / "adapter"
    destination = tmp_path / "publisher"
    source.mkdir()
    destination.mkdir()
    (source / "train.config").write_text("adapter\n")
    (destination / "train.config").write_text("publisher\n")

    try:
        runner._copy_tree_additions(source, destination)
    except ValueError as exc:
        assert str(exc) == "aurafusion360_interiorgs_adapter_path_conflict"
    else:
        raise AssertionError("adapter overwrite was not rejected")


def test_aura_subprocess_receives_explicit_environment(tmp_path: Path) -> None:
    runner = _load_runner()
    log_path = tmp_path / "environment.log"

    result = runner.shared._run(
        ["/usr/bin/env"],
        cwd=tmp_path,
        log_path=log_path,
        env={"AURA_LAMA_SOURCE_BOUND": "yes"},
    )

    assert result["returncode"] == 0
    assert "AURA_LAMA_SOURCE_BOUND=yes" in log_path.read_text()


def test_aura_lama_environment_binds_source_before_runtime_dependencies(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _load_runner()
    dependencies = tmp_path / "dependencies"
    lama_source = tmp_path / "LaMa"
    monkeypatch.setenv("PYTHONPATH", "inherited")

    aura_env, lama_env = runner._runtime_environments(
        dependencies=dependencies,
        lama_source=lama_source,
    )

    assert aura_env["PYTHONPATH"] == f"{dependencies}{os.pathsep}inherited"
    assert lama_env["PYTHONPATH"] == (
        f"{lama_source}{os.pathsep}{dependencies}{os.pathsep}inherited"
    )
    runner_source = (
        Path(__file__).resolve().parents[1] / "scripts/adp_aura_interiorgs_provider_runner.py"
    ).read_text(encoding="utf-8")
    assert "env=env" in runner_source


def test_aura_lama_checkpoint_is_rematerialized_immediately_before_execution(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    runtime = tmp_path / "runtime"
    source = runtime / "AuraFusion360_official"
    runtime.mkdir()
    archive_path = runtime / "big-lama.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("big-lama/config.yaml", "training_model:\n  kind: default\n")
        archive.writestr("big-lama/models/best.ckpt", b"checkpoint-bytes")
    archive_sha256 = "sha256:" + hashlib.sha256(archive_path.read_bytes()).hexdigest()
    spec = {
        "lama": {
            "checkpoint_archive": "big-lama.zip",
            "checkpoint_sha256": archive_sha256,
        }
    }

    first = runner._materialize_lama_checkpoint(
        runtime=runtime, source=source, spec=spec
    )
    (source / "LaMa/big-lama/config.yaml").unlink()
    receipt_path = tmp_path / "evidence/materialization.json"
    second = runner._materialize_lama_checkpoint(
        runtime=runtime,
        source=source,
        spec=spec,
        receipt_path=receipt_path,
    )

    assert first["status"] == "accepted"
    assert second["status"] == "accepted"
    assert second["files"]["config"]["size_bytes"] > 0
    assert second["files"]["checkpoint"]["size_bytes"] > 0
    assert receipt_path.is_file()


def test_aura_lama_checkpoint_rejects_archive_without_config(tmp_path: Path) -> None:
    runner = _load_runner()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    archive_path = runtime / "big-lama.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("big-lama/models/best.ckpt", b"checkpoint-bytes")
    archive_sha256 = "sha256:" + hashlib.sha256(archive_path.read_bytes()).hexdigest()

    receipt = runner._materialize_lama_checkpoint(
        runtime=runtime,
        source=runtime / "AuraFusion360_official",
        spec={
            "lama": {
                "checkpoint_archive": "big-lama.zip",
                "checkpoint_sha256": archive_sha256,
            }
        },
    )

    assert receipt["status"] == "blocked"
    assert "aurafusion360_interiorgs_lama_checkpoint_members_missing" in receipt["blockers"]


def test_aura_reference_lama_command_uses_absolute_verified_checkpoint_path(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    source = tmp_path / "AuraFusion360_official"
    checkpoint_root = source / "LaMa/big-lama"
    (checkpoint_root / "models").mkdir(parents=True)
    (checkpoint_root / "config.yaml").write_text("training_model: {}\n")
    (checkpoint_root / "models/best.ckpt").write_bytes(b"checkpoint")

    command = runner._reference_lama_command(
        source=source,
        runtime=tmp_path / "runtime",
        lama_python=str(source / "LaMa/.venv/bin/python"),
    )

    assert command[2] == f"model.path={checkpoint_root.resolve()}"
    assert command[3] == f"indir={(tmp_path / 'runtime/reference_lama_input').resolve()}"
    assert command[4] == f"outdir={(tmp_path / 'runtime/reference_lama_output').resolve()}"


def test_aura_reference_lama_command_rejects_missing_checkpoint_bytes(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    source = tmp_path / "AuraFusion360_official"
    checkpoint_root = source / "LaMa/big-lama"
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / "config.yaml").write_text("training_model: {}\n")

    try:
        runner._reference_lama_command(
            source=source,
            runtime=tmp_path / "runtime",
            lama_python=str(source / "LaMa/.venv/bin/python"),
        )
    except ValueError as exc:
        assert str(exc) == "aurafusion360_interiorgs_lama_checkpoint_path_unresolved"
    else:
        raise AssertionError("missing LaMa checkpoint bytes were not rejected")


@pytest.mark.parametrize(
    ("scene_id", "target_id", "reference_camera"),
    [
        ("840313", "ins160", "low_approach"),
        ("840796", "ins123", "front_medium"),
    ],
)
def test_aura_runtime_scene_binding_is_not_fixture_specific(
    scene_id: str, target_id: str, reference_camera: str
) -> None:
    runner = _load_runner()

    binding = runner._scene_binding(
        {
            "scene": {
                "publisher_scene_id": scene_id,
                "target_instance_id": target_id,
                "scene_slug": f"{scene_id}_{target_id}",
                "reference_camera_id": reference_camera,
            }
        }
    )

    assert binding["scene_slug"] == f"{scene_id}_{target_id}"
    runner_source = (
        Path(__file__).resolve().parents[1]
        / "scripts/adp_aura_interiorgs_provider_runner.py"
    ).read_text(encoding="utf-8")
    assert 'SCENE = "840313_ins160"' not in runner_source
    assert '"scene_id": "840313"' not in runner_source


def test_aura_openclip_cache_is_resolved_offline_and_exact(tmp_path: Path) -> None:
    runner = _load_runner()
    checkpoint = tmp_path / "open_clip_pytorch_model.bin"
    checkpoint.write_bytes(b"exact-openclip")
    sha256 = "sha256:" + hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    calls: list[dict[str, object]] = []

    def fake_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(checkpoint)

    receipt = runner._verify_openclip_offline_cache(
        spec={
            "runtime_models": [
                {
                    "repository": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                    "revision": "1" * 40,
                    "materialized_files": [
                        {
                            "path": checkpoint.name,
                            "size_bytes": checkpoint.stat().st_size,
                            "sha256": sha256,
                        }
                    ],
                }
            ]
        },
        hf_hub_download=fake_download,
    )

    assert receipt["resolved_without_network"] is True
    assert calls == [
        {
            "repo_id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            "filename": "open_clip_pytorch_model.bin",
            "local_files_only": True,
        }
    ]


def test_aura_openclip_cache_rejects_changed_bytes(tmp_path: Path) -> None:
    runner = _load_runner()
    checkpoint = tmp_path / "open_clip_pytorch_model.bin"
    checkpoint.write_bytes(b"changed")

    try:
        runner._verify_openclip_offline_cache(
            spec={
                "runtime_models": [
                    {
                        "repository": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                        "revision": "1" * 40,
                        "materialized_files": [
                            {
                                "path": checkpoint.name,
                                "size_bytes": checkpoint.stat().st_size,
                                "sha256": "sha256:" + "0" * 64,
                            }
                        ],
                    }
                ]
            },
            hf_hub_download=lambda **_kwargs: str(checkpoint),
        )
    except ValueError as exc:
        assert str(exc) == "aurafusion360_openclip_offline_cache_missing_or_changed"
    else:
        raise AssertionError("changed OpenCLIP bytes were not rejected")


def test_aura_runtime_prerequisite_binds_exact_public_openclip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt: dict[str, object] = {
        "methods": {
            "aurafusion360_interiorgs_runtime": {
                "checkpoint_rights_established": True,
                "remote_snapshots": [
                    {
                        "artifact_id": "aurafusion360_openclip_vit_h_14",
                        "rights_established": True,
                        "rights": {"license_id": "MIT"},
                        "publisher": {
                            "repository": bundle.OPENCLIP_REPOSITORY,
                            "revision": bundle.OPENCLIP_REVISION,
                            "path_prefix": bundle.OPENCLIP_PATH,
                            "snapshot_digest": bundle.OPENCLIP_SNAPSHOT_DIGEST,
                            "gated": False,
                            "private": False,
                            "single_file_identity": {
                                "path": bundle.OPENCLIP_PATH,
                                "size_bytes": bundle.OPENCLIP_SIZE_BYTES,
                                "lfs_sha256": bundle.OPENCLIP_SHA256,
                            },
                        },
                    }
                ],
            }
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    monkeypatch.setattr(
        bundle,
        "AURA_RUNTIME_PREREQUISITE_RECEIPT_DIGEST",
        receipt["receipt_digest"],
    )

    model = bundle._validated_runtime_prerequisite(receipt)

    assert model["repository"] == bundle.OPENCLIP_REPOSITORY
    assert model["materialized_files"][0]["sha256"] == bundle.OPENCLIP_SHA256
    mutated = dict(receipt)
    mutated["receipt_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="runtime_prerequisite_digest_mismatch"):
        bundle._validated_runtime_prerequisite(mutated)


def test_aura_interiorgs_entrypoint_emits_long_stage_liveness() -> None:
    entrypoint = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_adp_aura_interiorgs_provider_runtime.sh"
    ).read_text(encoding="utf-8")
    assert "BLUEPRINT_ADP_AURA_INTERIORGS_RUNTIME_PROGRESS" in entrypoint
    assert "BLUEPRINT_ADP_AURA_INTERIORGS_STAGE_STARTED" in entrypoint
    assert "run_with_progress prepare" in entrypoint
    assert "run_with_progress execute" in entrypoint
