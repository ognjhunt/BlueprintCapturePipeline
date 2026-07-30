import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from blueprint_pipeline import openpi_policy_ranking_gpu_bootstrap as bootstrap_module
from blueprint_pipeline.openpi_policy_ranking_gpu_bootstrap import (
    EXECUTION_MODE_NEW_SITE_CANARY,
    _download_signed_input,
    _upload_output,
    build_multi_scene_private_input_bundle,
    build_private_input_bundle,
    extract_private_input_bundle,
    run_signed_gpu_bootstrap,
)
from blueprint_pipeline.new_site_diagnostic_canary_gpu import build_canary_input_bundle
from blueprint_pipeline.new_site_diagnostic_smoke import build_protocol


ROOT = Path(__file__).resolve().parents[1]


def test_private_bundle_roundtrip_and_hash_binding(tmp_path: Path) -> None:
    background = tmp_path / "background.png"
    Image.new("RGB", (224, 224), color=(1, 2, 3)).save(background)
    bundle = tmp_path / "input.zip"
    receipt = build_private_input_bundle(
        background_path=background,
        output_zip=bundle,
        source_scene_id="scene",
        source_revision="a" * 40,
        source_asset_sha256="b" * 64,
    )
    extracted = extract_private_input_bundle(
        bundle_path=bundle,
        expected_bundle_sha256=receipt["bundle_sha256"],
        output_dir=tmp_path / "extracted",
    )
    assert Path(extracted["background_path"]).read_bytes() == background.read_bytes()
    assert extracted["manifest"]["raw_3dgs_included"] is False

    with pytest.raises(ValueError, match="bundle_sha256_mismatch"):
        extract_private_input_bundle(
            bundle_path=bundle,
            expected_bundle_sha256="0" * 64,
            output_dir=tmp_path / "wrong",
        )


def test_signed_transport_rejects_non_https_before_network(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="gpu_input_url_not_safe_https"):
        _download_signed_input("http://storage.example/input", tmp_path / "input.zip")
    archive = tmp_path / "output.zip"
    archive.write_bytes(b"zip")
    with pytest.raises(ValueError, match="gpu_output_url_not_safe_https"):
        _upload_output("file:///tmp/output.zip", archive)


def test_multi_scene_bundle_keeps_warehouse_and_captured_claims_separate(
    tmp_path: Path,
) -> None:
    captured = tmp_path / "captured.png"
    warehouse = tmp_path / "warehouse.png"
    Image.new("RGB", (224, 224), color=(1, 2, 3)).save(captured)
    Image.new("RGB", (224, 224), color=(4, 5, 6)).save(warehouse)
    bundle = tmp_path / "multi.zip"
    receipt = build_multi_scene_private_input_bundle(
        scenes=[
            {
                "background_path": captured,
                "source_scene_id": "captured",
                "source_scene_kind": "captured_3dgs",
                "source_revision": "a" * 40,
                "source_asset_sha256": "b" * 64,
            },
            {
                "background_path": warehouse,
                "source_scene_id": "warehouse",
                "source_scene_kind": "controlled_nvidia_usd",
                "source_revision": "c" * 40,
                "source_asset_sha256": "d" * 64,
            },
        ],
        output_zip=bundle,
    )
    extracted = extract_private_input_bundle(
        bundle_path=bundle,
        expected_bundle_sha256=receipt["bundle_sha256"],
        output_dir=tmp_path / "extracted-multi",
    )
    assert receipt["manifest"]["scene_count"] == 2
    assert [row["scene_kind"] for row in extracted["scene_backgrounds"]] == [
        "captured_3dgs",
        "controlled_nvidia_usd",
    ]
    assert Path(extracted["scene_backgrounds"][1]["background_path"]).read_bytes() == (
        warehouse.read_bytes()
    )


def test_bootstrap_uploads_terminal_failure_envelope_for_early_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "BLUEPRINT_OPENPI_POLICY_RANKING_INPUT_SECRET_URL",
        "https://storage.example/input?signature=secret",
    )
    monkeypatch.setenv("BLUEPRINT_OPENPI_POLICY_RANKING_INPUT_SHA256", "a" * 64)
    monkeypatch.setenv(
        "BLUEPRINT_OPENPI_POLICY_RANKING_OUTPUT_SECRET_PUT_URL",
        "https://storage.example/output?signature=secret",
    )
    monkeypatch.setattr(
        bootstrap_module,
        "_download_signed_input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("secret detail")),
    )
    observed = {}

    def upload(_url: str, archive_path: Path) -> int:
        with zipfile.ZipFile(archive_path) as archive:
            observed.update(
                json.loads(archive.read("openpi_policy_ranking_gpu_job.json").decode("utf-8"))
            )
        return 200

    monkeypatch.setattr(bootstrap_module, "_upload_output", upload)

    result = run_signed_gpu_bootstrap(workspace=tmp_path)

    assert result["status"] == "blocked"
    assert result["failure_type"] == "RuntimeError"
    assert observed["status"] == "blocked"
    assert observed["blockers"] == ["openpi_gpu_bootstrap_failed:RuntimeError"]
    assert "secret detail" not in json.dumps(observed)


def test_bootstrap_routes_canary_bundle_to_one_arm_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = build_protocol(ROOT, experiment_id="bootstrap_canary_test_v1")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    background = tmp_path / "background.png"
    Image.new("RGB", (224, 224), color=(8, 9, 10)).save(background)
    bundle = tmp_path / "canary-input.zip"
    receipt = build_canary_input_bundle(
        protocol_path=protocol_path,
        background_path=background,
        output_zip=bundle,
        arm_id="skeleton_only",
    )
    monkeypatch.setenv(
        "BLUEPRINT_OPENPI_POLICY_RANKING_INPUT_SECRET_URL",
        "https://storage.example/input?signature=secret",
    )
    monkeypatch.setenv("BLUEPRINT_OPENPI_POLICY_RANKING_INPUT_SHA256", receipt["bundle_sha256"])
    monkeypatch.setenv(
        "BLUEPRINT_OPENPI_POLICY_RANKING_OUTPUT_SECRET_PUT_URL",
        "https://storage.example/output?signature=secret",
    )
    monkeypatch.setenv("BLUEPRINT_OPENPI_EXECUTION_MODE", EXECUTION_MODE_NEW_SITE_CANARY)
    monkeypatch.setattr(
        bootstrap_module,
        "_download_signed_input",
        lambda _url, destination, **_kwargs: destination.write_bytes(bundle.read_bytes()),
    )
    observed = {}

    def run_canary(**kwargs):
        observed.update(kwargs)
        manifest = {
            "schema_version": "new_site_diagnostic_canary_gpu.v1",
            "status": "completed",
            "canary": {"status": "passed"},
            "manifest_sha256": "c" * 64,
        }
        Path(kwargs["output_dir"]).mkdir(parents=True)
        (Path(kwargs["output_dir"]) / "new_site_diagnostic_canary_gpu.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return manifest

    archive_names = []

    def upload(_url: str, archive_path: Path) -> int:
        with zipfile.ZipFile(archive_path) as archive:
            archive_names.extend(archive.namelist())
        return 200

    monkeypatch.setattr(bootstrap_module, "run_skeleton_only_canary", run_canary)
    monkeypatch.setattr(bootstrap_module, "_upload_output", upload)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    result = run_signed_gpu_bootstrap(workspace=workspace)

    assert result["status"] == "completed"
    assert result["execution_mode"] == EXECUTION_MODE_NEW_SITE_CANARY
    assert Path(observed["protocol_path"]).read_bytes() == protocol_path.read_bytes()
    assert "new_site_diagnostic_canary_gpu.json" in archive_names


def test_bootstrap_fails_closed_before_routing_unconfigured_ctrl_world_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "BLUEPRINT_OPENPI_POLICY_RANKING_INPUT_SECRET_URL", "https://storage.example/input"
    )
    monkeypatch.setenv("BLUEPRINT_OPENPI_POLICY_RANKING_INPUT_SHA256", "a" * 64)
    monkeypatch.setenv(
        "BLUEPRINT_OPENPI_POLICY_RANKING_OUTPUT_SECRET_PUT_URL",
        "https://storage.example/output",
    )
    monkeypatch.setenv("BLUEPRINT_OPENPI_EXECUTION_MODE", EXECUTION_MODE_NEW_SITE_CANARY)
    monkeypatch.setattr(
        bootstrap_module,
        "_download_signed_input",
        lambda _url, destination, **_kwargs: destination.write_bytes(b"placeholder"),
    )
    monkeypatch.setattr(
        bootstrap_module,
        "extract_canary_input_bundle",
        lambda **_kwargs: {
            "manifest": {
                "arm_id": "ctrl_world",
                "protocol_sha256": "b" * 64,
                "wam_seed": 23,
            },
            "protocol_path": str(tmp_path / "protocol.json"),
            "background_path": str(tmp_path / "background.png"),
            "initial_camera_paths": {
                "image": str(tmp_path / "external.png"),
                "image2": str(tmp_path / "external-2.png"),
                "wrist_image": str(tmp_path / "wrist.png"),
            },
        },
    )
    monkeypatch.setattr(
        bootstrap_module,
        "run_skeleton_only_canary",
        lambda **_kwargs: pytest.fail("must not route Ctrl-World through skeleton-only"),
    )
    monkeypatch.setattr(bootstrap_module, "_upload_output", lambda *_args: 200)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    result = run_signed_gpu_bootstrap(workspace=workspace)

    assert result["status"] == "blocked"
    assert result["failure_type"] == "ValueError"
    assert result["input_manifest"]["arm_id"] == "ctrl_world"


def test_bootstrap_routes_ctrl_world_to_isolated_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "BLUEPRINT_OPENPI_POLICY_RANKING_INPUT_SECRET_URL", "https://storage.example/input"
    )
    monkeypatch.setenv("BLUEPRINT_OPENPI_POLICY_RANKING_INPUT_SHA256", "a" * 64)
    monkeypatch.setenv(
        "BLUEPRINT_OPENPI_POLICY_RANKING_OUTPUT_SECRET_PUT_URL",
        "https://storage.example/output",
    )
    monkeypatch.setenv("BLUEPRINT_OPENPI_EXECUTION_MODE", EXECUTION_MODE_NEW_SITE_CANARY)
    monkeypatch.setattr(
        bootstrap_module,
        "_download_signed_input",
        lambda _url, destination, **_kwargs: destination.write_bytes(b"placeholder"),
    )
    cameras = {
        "image": str(tmp_path / "external.png"),
        "image2": str(tmp_path / "external-2.png"),
        "wrist_image": str(tmp_path / "wrist.png"),
    }
    monkeypatch.setattr(
        bootstrap_module,
        "extract_canary_input_bundle",
        lambda **_kwargs: {
            "manifest": {"arm_id": "ctrl_world", "wam_seed": 23},
            "protocol_path": str(tmp_path / "protocol.json"),
            "background_path": str(tmp_path / "background.png"),
            "initial_camera_paths": cameras,
        },
    )
    model_root = tmp_path / "models"
    monkeypatch.setenv("BLUEPRINT_CTRL_WORLD_MODEL_ROOT", str(model_root))
    staged_paths = {
        "world_model_checkpoint": str(model_root / "checkpoint.pt"),
        "svd_model_root": str(model_root / "svd"),
        "clip_model_root": str(model_root / "clip"),
    }
    runner = SimpleNamespace(
        world_model_checkpoint=Path(staged_paths["world_model_checkpoint"]),
        svd_model_root=Path(staged_paths["svd_model_root"]),
        clip_model_root=Path(staged_paths["clip_model_root"]),
    )
    monkeypatch.setattr(
        bootstrap_module,
        "stage_ctrl_world_runtime_assets",
        lambda **_kwargs: {"status": "completed", "paths": staged_paths},
    )

    class FakeRuntimeFactory:
        @classmethod
        def from_environment(cls):
            return runner

    monkeypatch.setattr(
        bootstrap_module, "CtrlWorldJointPositionSubprocessRuntime", FakeRuntimeFactory
    )
    observed: dict = {}

    def run_canary(**kwargs):
        observed.update(kwargs)
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True)
        manifest = {
            "schema_version": "new_site_diagnostic_canary_gpu.v1",
            "status": "completed",
            "manifest_sha256": "c" * 64,
        }
        (output / "new_site_diagnostic_canary_gpu.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return manifest

    monkeypatch.setattr(bootstrap_module, "run_ctrl_world_canary", run_canary)
    monkeypatch.setattr(bootstrap_module, "_upload_output", lambda *_args: 200)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = run_signed_gpu_bootstrap(workspace=workspace)

    assert result["status"] == "completed"
    assert observed["ctrl_world_runner"] is runner
    assert observed["seed"] == 23
    assert observed["initial_camera_paths"] == cameras


def test_bootstrap_routes_oscar_to_resident_multiview_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "BLUEPRINT_OPENPI_POLICY_RANKING_INPUT_SECRET_URL", "https://storage.example/input"
    )
    monkeypatch.setenv("BLUEPRINT_OPENPI_POLICY_RANKING_INPUT_SHA256", "a" * 64)
    monkeypatch.setenv(
        "BLUEPRINT_OPENPI_POLICY_RANKING_OUTPUT_SECRET_PUT_URL",
        "https://storage.example/output",
    )
    monkeypatch.setenv("BLUEPRINT_OPENPI_EXECUTION_MODE", EXECUTION_MODE_NEW_SITE_CANARY)
    monkeypatch.setattr(
        bootstrap_module,
        "_download_signed_input",
        lambda _url, destination, **_kwargs: destination.write_bytes(b"placeholder"),
    )
    cameras = {
        "image": str(tmp_path / "external.png"),
        "wrist_image": str(tmp_path / "wrist.png"),
    }
    monkeypatch.setattr(
        bootstrap_module,
        "extract_canary_input_bundle",
        lambda **_kwargs: {
            "manifest": {"arm_id": "oscar", "wam_seed": 42},
            "protocol_path": str(tmp_path / "protocol.json"),
            "background_path": str(tmp_path / "background.png"),
            "initial_camera_paths": cameras,
        },
    )

    class FakeRuntime:
        entered = False
        exited = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, *_exc):
            self.exited = True

    runtime = FakeRuntime()

    class FakeRuntimeFactory:
        @classmethod
        def from_environment(cls, *, evidence_dir):
            assert evidence_dir.name == "oscar_runtime"
            return runtime

    monkeypatch.setattr(
        bootstrap_module, "OscarMultiViewReferenceRuntime", FakeRuntimeFactory
    )
    observed: dict = {}

    def run_canary(**kwargs):
        observed.update(kwargs)
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True)
        manifest = {
            "schema_version": "new_site_diagnostic_canary_gpu.v1",
            "status": "completed",
            "manifest_sha256": "c" * 64,
        }
        (output / "new_site_diagnostic_canary_gpu.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return manifest

    monkeypatch.setattr(bootstrap_module, "run_oscar_canary", run_canary)
    monkeypatch.setattr(bootstrap_module, "_upload_output", lambda *_args: 200)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = run_signed_gpu_bootstrap(workspace=workspace)

    assert result["status"] == "completed"
    assert observed["oscar_generator"] is runtime
    assert observed["seed"] == 42
    assert observed["initial_camera_paths"] == cameras
    assert runtime.entered is True
    assert runtime.exited is True
