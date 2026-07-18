from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import tarfile
import zipfile

import pytest

from blueprint_pipeline import g1_microwave_finetune_worker as worker


def test_training_command_is_bounded_and_uses_sealed_checkpoint(tmp_path: Path) -> None:
    argv = worker.training_argv(
        dataset=tmp_path / "dataset",
        output=tmp_path / "output",
    )
    pairs = dict(zip(argv[2:-1:2], argv[3:-1:2]))

    assert argv[:2] == [
        "/opt/gr00t-venv/bin/python",
        "/opt/gr00t/gr00t/experiment/launch_finetune.py",
    ]
    assert pairs["--base-model-path"] == "/opt/blueprint/ckpts/sonic"
    assert pairs["--embodiment-tag"] == "UNITREE_G1_SONIC"
    assert pairs["--num-gpus"] == "1"
    assert pairs["--max-steps"] == "500"
    assert pairs["--save-steps"] == "500"
    assert pairs["--save-total-limit"] == "1"
    assert pairs["--state-dropout-prob"] == "0.0"
    assert argv[-1] == "--save-only-model"
    assert worker.HARD_TIMEOUT_SECONDS == 7_200
    assert worker.LOG_STALL_TIMEOUT_SECONDS == 900


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo("../escape")
        payload = b"unsafe"
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="path_traversal"):
        worker.safe_extract_dataset(archive, tmp_path / "out")


def test_safe_extract_requires_native_dataset_members(tmp_path: Path) -> None:
    archive = tmp_path / "incomplete.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo(f"{worker.EXPECTED_DATASET_DIR}/meta/info.json")
        payload = b"{}"
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="members_missing"):
        worker.safe_extract_dataset(archive, tmp_path / "out")


def test_resolve_trained_checkpoint_requires_one_complete_model(tmp_path: Path) -> None:
    output = tmp_path / "output"
    checkpoint = output / "checkpoint-500"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model-00001-of-00003.safetensors").write_bytes(b"weights-1")
    (checkpoint / "model-00002-of-00003.safetensors").write_bytes(b"weights-2")
    (checkpoint / "model-00003-of-00003.safetensors").write_bytes(b"weights-3")

    # GR00T also exports a mirror of the final model at the output root.  That
    # is not a second trainer checkpoint and must not make checkpoint-500 look
    # ambiguous.
    (output / "config.json").write_text("{}", encoding="utf-8")
    (output / "model-00001-of-00003.safetensors").write_bytes(b"weights-1")

    assert worker._resolve_trained_checkpoint(output) == checkpoint

    other = output / "checkpoint-499"
    other.mkdir()
    (other / "config.json").write_text("{}", encoding="utf-8")
    (other / "model.safetensors").write_bytes(b"other")
    assert worker._resolve_trained_checkpoint(output) is None


def test_open_loop_qualification_requires_measured_20_percent_improvement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []

    def fake_measurement(**kwargs):
        model = kwargs["model"]
        calls.append(model)
        tuned = model == tmp_path / "checkpoint-500"
        return {
            "exit_code": 0,
            "measurement": {
                "finite": True,
                "mse": 0.4 if tuned else 1.0,
                "mae": 0.5 if tuned else 1.0,
            },
            "log_path": str(kwargs["log"]),
        }

    monkeypatch.setattr(worker, "_run_open_loop_measurement", fake_measurement)
    monkeypatch.setattr(
        worker,
        "prepare_warm_start_eval_checkpoint",
        lambda **_kwargs: (
            worker.SEALED_SONIC_CHECKPOINT,
            {"status": "passed", "blockers": []},
        ),
    )
    result = worker.qualify_checkpoint_open_loop(
        dataset=tmp_path / "dataset",
        trained_checkpoint=tmp_path / "checkpoint-500",
        workspace=tmp_path,
    )

    assert result["status"] == "passed"
    assert result["mse_ratio"] == pytest.approx(0.4)
    assert result["mae_ratio"] == pytest.approx(0.5)
    assert result["exact_owned_training_trajectory_only"] is True
    assert calls == [worker.SEALED_SONIC_CHECKPOINT, tmp_path / "checkpoint-500"]
    compile((tmp_path / "run_microwave_open_loop.py").read_text(), "open_loop", "exec")


def test_open_loop_qualification_fails_closed_on_missing_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        worker,
        "_run_open_loop_measurement",
        lambda **_kwargs: {"exit_code": 1, "measurement": None, "log_path": "log"},
    )
    monkeypatch.setattr(
        worker,
        "prepare_warm_start_eval_checkpoint",
        lambda **_kwargs: (
            worker.SEALED_SONIC_CHECKPOINT,
            {"status": "passed", "blockers": []},
        ),
    )

    result = worker.qualify_checkpoint_open_loop(
        dataset=tmp_path / "dataset",
        trained_checkpoint=tmp_path / "checkpoint-500",
        workspace=tmp_path,
    )

    assert result["status"] == "blocked"
    assert result["mse_ratio"] is None
    assert result["blockers"] == ["g1_microwave_groot_open_loop_not_improved"]


def test_open_loop_qualification_reuses_pretraining_warm_start_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []

    def fake_measurement(**kwargs):
        calls.append(kwargs["model"])
        return {
            "exit_code": 0,
            "measurement": {"finite": True, "mse": 0.4, "mae": 0.5},
            "log_path": str(kwargs["log"]),
        }

    monkeypatch.setattr(worker, "_run_open_loop_measurement", fake_measurement)
    monkeypatch.setattr(
        worker,
        "prepare_warm_start_eval_checkpoint",
        lambda **_kwargs: pytest.fail("warm start must not be measured twice"),
    )
    preflight = {
        "status": "passed",
        "warm_start": {
            "exit_code": 0,
            "measurement": {"finite": True, "mse": 1.0, "mae": 1.0},
            "log_path": "warm.log",
        },
        "warm_start_model_resolution": {"status": "passed", "blockers": []},
        "blockers": [],
    }

    result = worker.qualify_checkpoint_open_loop(
        dataset=tmp_path / "dataset",
        trained_checkpoint=tmp_path / "checkpoint-500",
        workspace=tmp_path,
        warm_start_preflight=preflight,
    )

    assert result["status"] == "passed"
    assert result["mse_ratio"] == pytest.approx(0.4)
    assert result["mae_ratio"] == pytest.approx(0.5)
    assert calls == [tmp_path / "checkpoint-500"]


def test_warm_start_eval_checkpoint_uses_baked_model_without_copying_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "hf_home"
    baked = cache / "hub/models--nvidia--Cosmos-Reason2-2B/snapshots" / worker.PINNED_COSMOS_REVISION
    baked.mkdir(parents=True)
    alias = tmp_path / "cosmos-reason2-2b"
    alias.symlink_to(baked, target_is_directory=True)
    monkeypatch.setattr(worker, "LOCAL_COSMOS_MODEL_ROOT", alias)
    sealed = tmp_path / "sealed-sonic"
    processor = sealed / "processor"
    processor.mkdir(parents=True)
    original_config = {
        "model_name": "/workspace/.blueprint-model-aliases/cosmos/missing",
    }
    (sealed / "config.json").write_text(json.dumps(original_config), encoding="utf-8")
    (sealed / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    (sealed / "model.safetensors").write_bytes(b"sealed-weights")
    original_processor_config = {
        "processor_class": "Gr00tN1D7Processor",
        "processor_kwargs": {
            "model_name": "nvidia/Cosmos-Reason2-2B",
            "model_type": "qwen",
        },
    }
    (processor / "processor_config.json").write_text(
        json.dumps(original_processor_config), encoding="utf-8"
    )

    resolved, report = worker.prepare_warm_start_eval_checkpoint(
        workspace=tmp_path / "workspace",
        sealed_checkpoint=sealed,
        trusted_cache_root=cache,
    )

    assert report["status"] == "passed"
    assert report["sealed_weight_files_modified"] is False
    assert resolved is not None
    assert (resolved / "model.safetensors").is_symlink()
    assert not (resolved / "processor_config.json").is_symlink()
    assert json.loads((resolved / "config.json").read_text())["model_name"] == str(baked)
    assert json.loads((resolved / "processor_config.json").read_text()) == {
        "processor_class": "Gr00tN1D7Processor",
        "processor_kwargs": {
            "model_name": str(baked),
            "model_type": "qwen",
        },
    }
    assert json.loads((sealed / "config.json").read_text()) == original_config
    assert json.loads((processor / "processor_config.json").read_text()) == (
        original_processor_config
    )


def test_checkpoint_archive_upload_splits_into_ordered_hash_bound_parts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "checkpoint.zip"
    archive.write_bytes(b"abcdefghij")
    uploaded: list[bytes] = []

    def fake_upload(path: Path, url: str) -> dict[str, object]:
        assert url == f"https://parts.invalid/{len(uploaded) + 1}"
        uploaded.append(path.read_bytes())
        return {
            "status": "passed",
            "uploaded_size_bytes": path.stat().st_size,
            "uploaded_sha256": worker._sha256(path),
            "raw_signed_url_recorded": False,
            "blockers": [],
        }

    monkeypatch.setattr(worker, "_upload", fake_upload)
    result = worker._upload_checkpoint_archive(
        archive,
        "https://single.invalid/checkpoint",
        part_urls=[f"https://parts.invalid/{index}" for index in range(1, 4)],
        part_bytes=4,
    )

    assert result["status"] == "passed"
    assert result["transport"] == "ordered_parts"
    assert uploaded == [b"abcd", b"efgh", b"ij"]
    assert [part["part_number"] for part in result["parts"]] == [1, 2, 3]
    assert sum(part["size_bytes"] for part in result["parts"]) == 10
    assert not list(tmp_path.glob("*.part-*"))


def test_proof_archive_excludes_checkpoint_weights_and_input_dataset(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "checkpoint").mkdir(parents=True)
    (workspace / "input").mkdir()
    (workspace / "warm_start_eval_checkpoint").mkdir()
    (workspace / "checkpoint" / "model.safetensors").write_bytes(b"weights")
    (workspace / "input" / "episode.parquet").write_bytes(b"dataset")
    (workspace / "warm_start_eval_checkpoint" / "model.safetensors").write_bytes(
        b"sealed-alias"
    )
    (workspace / "g1_microwave_finetune_worker_report.json").write_text(
        "{}", encoding="utf-8"
    )
    archive = tmp_path / "proof.zip"

    worker._archive_outputs(
        workspace,
        archive,
        excluded_top_level=("checkpoint", "input", "warm_start_eval_checkpoint"),
    )

    with zipfile.ZipFile(archive) as handle:
        assert handle.namelist() == ["g1_microwave_finetune_worker_report.json"]
    assert worker.CHECKPOINT_PUT_URL_ENV == (
        "BLUEPRINT_G1_MICROWAVE_FINETUNE_CHECKPOINT_PUT_URL"
    )


def test_checkpoint_archive_excludes_duplicate_output_root_mirror(tmp_path: Path) -> None:
    output = tmp_path / "output"
    checkpoint = output / "checkpoint-500"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"qualified")
    (output / "config.json").write_text("{}", encoding="utf-8")
    (output / "model.safetensors").write_bytes(b"mirror")
    archive = tmp_path / "checkpoint.zip"

    worker._archive_trained_checkpoint(checkpoint, archive)

    with zipfile.ZipFile(archive) as handle:
        assert handle.namelist() == ["config.json", "model.safetensors"]
        assert handle.read("model.safetensors") == b"qualified"


def test_training_patch_accepts_sealed_local_cosmos_path_without_modifying_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "gr00t/model/gr00t_n1d7/gr00t_n1d7.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        'def get_backbone_cls(config):\n'
        '    if "nvidia/Cosmos-Reason2" in config.model_name '
        'or "Qwen/Qwen3-VL" in config.model_name:\n'
        '        return Qwen3Backbone\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(worker, "GROOT_ROOT", tmp_path)

    result = worker.patch_local_cosmos_backbone_classifier(source)

    assert result["status"] == "passed"
    assert result["sealed_checkpoint_files_modified"] is False
    assert result["before_sha256"] != result["after_sha256"]
    assert '"cosmos-reason2" in str(config.model_name).lower()' in source.read_text()


def test_writable_groot_overlay_preserves_sealed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sealed_root = tmp_path / "sealed"
    source = sealed_root / "gr00t/model/gr00t_n1d7/gr00t_n1d7.py"
    setup = sealed_root / "gr00t/model/gr00t_n1d7/setup.py"
    launch = sealed_root / "gr00t/experiment/launch_finetune.py"
    source.parent.mkdir(parents=True)
    launch.parent.mkdir(parents=True)
    original = (
        'if "nvidia/Cosmos-Reason2" in config.model_name '
        'or "Qwen/Qwen3-VL" in config.model_name:\n'
    )
    source.write_text(original, encoding="utf-8")
    setup_original = (
        "from pathlib import Path\n"
        "class Setup:\n"
        "    def create(self):\n"
        "        if self.config.training.start_from_checkpoint is not None:\n"
        "            processor = AutoProcessor.from_pretrained(\n"
        "                self.config.training.start_from_checkpoint,\n"
        "            )\n"
        "        else:\n"
        "            processor = self.processor_class()\n"
        "        return processor\n"
    )
    setup.write_text(setup_original, encoding="utf-8")
    source.chmod(0o444)
    setup.chmod(0o444)
    original_mode = source.stat().st_mode
    setup_original_mode = setup.stat().st_mode
    launch.write_text(
        '    config.model.model_name = "nvidia/Cosmos-Reason2-2B"\n',
        encoding="utf-8",
    )
    unreadable_cache = sealed_root / "gr00t/model/__pycache__"
    unreadable_cache.mkdir()
    unreadable_cache.chmod(0o000)
    monkeypatch.setattr(worker, "GROOT_ROOT", sealed_root)
    runtime_root = tmp_path / "g1_microwave_groot_runtime"

    overlay = worker.prepare_writable_groot_runtime(destination_root=runtime_root)
    patch = worker.patch_local_cosmos_backbone_classifier(
        runtime_root / "gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
        trusted_root=runtime_root,
    )
    processor_patch = worker.patch_missing_checkpoint_processor_fallback(
        runtime_root / "gr00t/model/gr00t_n1d7/setup.py",
        trusted_root=runtime_root,
    )
    model_path_patch = worker.patch_offline_cosmos_model_path(
        runtime_root / "gr00t/experiment/launch_finetune.py",
        trusted_root=runtime_root,
        model_root=tmp_path / "local-cosmos",
    )

    assert overlay["status"] == "passed"
    assert overlay["sealed_source_files_modified"] is False
    assert patch["status"] == "passed"
    assert processor_patch["status"] == "passed"
    assert model_path_patch["status"] == "passed"
    assert processor_patch["warm_start_weights_preserved"] is True
    assert source.read_text(encoding="utf-8") == original
    assert source.stat().st_mode == original_mode
    assert setup.read_text(encoding="utf-8") == setup_original
    assert setup.stat().st_mode == setup_original_mode
    assert "cosmos-reason2" in (
        runtime_root / "gr00t/model/gr00t_n1d7/gr00t_n1d7.py"
    ).read_text(encoding="utf-8")
    patched_setup = runtime_root / "gr00t/model/gr00t_n1d7/setup.py"
    assert "nested_checkpoint_processor_root" in patched_setup.read_text(
        encoding="utf-8"
    )
    assert "checkpoint_processor_root / \"processor_config.json\"" in patched_setup.read_text(
        encoding="utf-8"
    )
    compile(patched_setup.read_text(encoding="utf-8"), str(patched_setup), "exec")
    assert str(tmp_path / "local-cosmos") in (
        runtime_root / "gr00t/experiment/launch_finetune.py"
    ).read_text(encoding="utf-8")
    assert not (runtime_root / "gr00t/model/__pycache__").exists()
    unreadable_cache.chmod(0o700)


def test_local_cosmos_processor_inventory_requires_sealed_cache_targets(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "sealed-cache"
    snapshot = cache / "hub/model/snapshots" / worker.PINNED_COSMOS_REVISION
    snapshot.mkdir(parents=True)
    alias = tmp_path / "cosmos"
    for name in worker.LOCAL_COSMOS_REQUIRED_FILES:
        target = snapshot / name
        target.write_text(name, encoding="utf-8")
    alias.symlink_to(snapshot, target_is_directory=True)

    result = worker.local_cosmos_processor_inventory(
        alias, trusted_cache_root=cache
    )

    assert result["status"] == "passed"
    assert result["offline_only"] is True
    assert result["resolved_model_root"] == str(snapshot.resolve())
    assert len(result["files"]) == len(worker.LOCAL_COSMOS_REQUIRED_FILES)


def test_training_command_can_use_writable_runtime_overlay(tmp_path: Path) -> None:
    runtime_root = tmp_path / "g1_microwave_groot_runtime"
    argv = worker.training_argv(
        dataset=tmp_path / "dataset",
        output=tmp_path / "output",
        groot_root=runtime_root,
    )

    assert argv[1] == str(runtime_root / "gr00t/experiment/launch_finetune.py")
    assert worker._runtime_env(runtime_root)["PYTHONPATH"].split(
        worker.os.pathsep
    )[0] == str(runtime_root)


def test_signed_zip_upload_sends_required_content_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "proof.zip"
    artifact.write_bytes(b"zip")
    captured: dict[str, str] = {}

    def fake_run(argv, *, input, **kwargs):
        captured["config"] = input
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(worker.subprocess, "run", fake_run)
    result = worker._upload(artifact, "https://objects.example/proof?signed=test")

    assert result["status"] == "passed"
    assert 'header = "Content-Type: application/zip"' in captured["config"]


def test_main_persists_secret_safe_fatal_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"

    def fail_worker(**_kwargs):
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "g1_microwave_finetune_progress.json").write_text(
            json.dumps({"phase": "backbone_classifier_patch"}), encoding="utf-8"
        )
        raise PermissionError("sensitive provider path")

    monkeypatch.setattr(worker, "run_worker", fail_worker)
    exit_code = worker.main(
        [
            "--dataset-archive",
            str(tmp_path / "dataset.tar.gz"),
            "--expected-dataset-sha256",
            "0" * 64,
            "--workspace",
            str(workspace),
        ]
    )

    report = json.loads(
        (workspace / "g1_microwave_finetune_worker_report.json").read_text()
    )
    assert exit_code == 1
    assert report["fatal_exception"] == {
        "message_recorded": False,
        "phase": "backbone_classifier_patch",
        "type": "PermissionError",
    }
    assert report["blockers"] == [
        "g1_microwave_finetune_worker_fatal_exception:PermissionError"
    ]
