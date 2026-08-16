from __future__ import annotations

import gzip
import importlib.util
import io
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace
import zipfile

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_artifixer3d_native_exports import (
    ArtiFixerNativeExportError,
    materialize_artifixer3d_native_appearance_exports,
    validate_artifixer3d_native_appearance_export,
)
from blueprint_pipeline.public_scene_artifixer3d_vast import _materialize_raw_result
from blueprint_pipeline.public_scene_artifixer3d_bundle import DUAL_TARGET_PIPELINE_MODE


class _Tensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def detach(self) -> _Tensor:
        return self


def _load_real_provider_runner():
    path = Path(__file__).parents[1] / "scripts/public_scene_artifixer3d_runner.py"
    spec = importlib.util.spec_from_file_location("_exact_artifixer_provider_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_exporter_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = ModuleType("torch")

    def load(_path: Path, **_kwargs):
        return {
            "config": SimpleNamespace(
                export_usdz=SimpleNamespace(apply_normalizing_transform=True)
            ),
            "positions": _Tensor((1, 3)),
            "rotation": _Tensor((1, 4)),
            "scale": _Tensor((1, 3)),
            "density": _Tensor((1, 1)),
            "features_albedo": _Tensor((1, 3)),
            "features_specular": _Tensor((1, 0)),
            "max_n_features": 0,
            "n_active_features": 0,
        }

    torch.load = load  # type: ignore[attr-defined]

    class PLYExporter:
        def export(self, _model, path: Path, **_kwargs) -> None:
            path.write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\n")

    class USDZExporter:
        def export(self, _model, path: Path, **_kwargs) -> None:
            buffer = io.BytesIO()
            with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=123) as stream:
                stream.write(b"exact-provider-nurec")
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("default.usda", b"#usda 1.0")
                archive.writestr("repaired_scene.nurec", buffer.getvalue())
                archive.writestr("gauss.usda", b"#usda 1.0")

    modules = {
        "torch": torch,
        "threedgrut": ModuleType("threedgrut"),
        "threedgrut.export": ModuleType("threedgrut.export"),
        "threedgrut.export.ply_exporter": ModuleType(
            "threedgrut.export.ply_exporter"
        ),
        "threedgrut.export.usdz_exporter": ModuleType(
            "threedgrut.export.usdz_exporter"
        ),
    }
    modules["threedgrut.export.ply_exporter"].PLYExporter = PLYExporter  # type: ignore[attr-defined]
    modules["threedgrut.export.usdz_exporter"].USDZExporter = USDZExporter  # type: ignore[attr-defined]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _exact_provider_to_host_raw_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict, dict, Path]:
    runner = _load_real_provider_runner()
    _install_exporter_seam(monkeypatch)
    provider_output = tmp_path / "provider/runtime_output"
    task_root = provider_output / "tasks/task_a"
    checkpoint = task_root / "artifixer3d/checkpoints/ckpt_30000.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"exact-provider-checkpoint")
    native = runner._export_checkpoint_native_appearance(
        checkpoint=checkpoint,
        task_output=task_root,
    )
    frames = []
    for index in range(8):
        frame = task_root / "artifixer3d_review_frames" / f"{index:05d}.png"
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(f"provider-frame-{index}".encode())
        frames.append(
            {
                "frame_index": index,
                "camera_id": f"camera_{index}",
                **runner._file_record(frame),
            }
        )
    execution = {
        "tasks": [
            {
                "task_id": "task_a",
                "pipeline_mode": DUAL_TARGET_PIPELINE_MODE,
                "training_record_count": 16,
                "artifixer3d_review_frames": frames,
                "artifixer3d_checkpoint": runner._file_record(checkpoint),
                "native_appearance": native,
                "outside_support_invariance_status": (
                    "deferred_until_final_soft_composite"
                ),
                "outside_support_changed_pixels_total": None,
            }
        ]
    }
    execution_root = tmp_path / "host/immutable_execution"
    shutil.copytree(provider_output, execution_root)
    raw = _materialize_raw_result(
        execution=execution,
        execution_root=execution_root,
        bundle={
            "pipeline_mode": DUAL_TARGET_PIPELINE_MODE,
            "phases": ["native_appearance_export"],
            "task_ids": ["task_a"],
            "task_camera_counts": {"task_a": 8},
            "task_training_record_counts": {"task_a": 16},
            "bundle_sha256": "sha256:bundle",
            "manifest_digest": "sha256:manifest",
            "runtime_request_digest": "sha256:request",
            "replacement_object_count": 1,
        },
        closeout={"provider_zero_confirmed": True},
    )
    raw_path = tmp_path / "host/public_scene_artifixer3d_raw_result.json"
    write_json(raw_path, raw)
    return native, raw, raw_path


def test_real_provider_export_is_rebased_and_sealed_for_native_consumer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider_native, raw, raw_path = _exact_provider_to_host_raw_result(
        tmp_path, monkeypatch
    )

    host_native = raw["tasks"][0]["native_appearance"]
    assert host_native["source_export_digest"] == provider_native["export_digest"]
    assert "export_digest" not in host_native
    assert "/host/immutable_execution/" in host_native["isaac_nurec_usdz"]["path"]

    outputs = materialize_artifixer3d_native_appearance_exports(
        raw_result_path=raw_path,
        output_root=tmp_path / "host/native_appearance_exports",
    )
    assert [row["task_id"] for row in outputs] == ["task_a"]
    receipt = validate_artifixer3d_native_appearance_export(outputs[0]["path"])
    assert receipt["source_export_digest"] == provider_native["export_digest"]
    assert receipt["export_digest"] == canonical_digest(
        receipt, digest_field="export_digest"
    )
    assert receipt["host_path_rebased_from_provider_runtime_output"] is True
    assert Path(receipt["isaac_nurec_usdz"]["path"]).is_file()


def test_native_export_handoff_rejects_mutated_rebased_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _provider_native, raw, raw_path = _exact_provider_to_host_raw_result(
        tmp_path, monkeypatch
    )
    Path(raw["tasks"][0]["native_appearance"]["isaac_nurec_usdz"]["path"]).write_bytes(
        b"mutated-after-host-closeout"
    )

    with pytest.raises(ArtiFixerNativeExportError, match="native_export_file_invalid"):
        materialize_artifixer3d_native_appearance_exports(
            raw_result_path=raw_path,
            output_root=tmp_path / "host/native_appearance_exports",
        )
