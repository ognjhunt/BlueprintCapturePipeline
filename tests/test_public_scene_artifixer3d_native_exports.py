from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import struct
import sys
from types import ModuleType, SimpleNamespace
import zipfile

import numpy
import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply
from blueprint_pipeline.public_scene_artifixer3d_native_exports import (
    ArtiFixerNativeExportError,
    materialize_artifixer3d_native_appearance_exports,
    validate_artifixer3d_native_appearance_export,
)
from blueprint_pipeline.public_scene_artifixer3d_vast import _materialize_raw_result
from blueprint_pipeline.public_scene_artifixer3d_bundle import DUAL_TARGET_PIPELINE_MODE
from tests.test_native_task_appearance_frame_alignment import (
    EXPORTER_AXIS_MATRIX,
    room_positions,
    write_appearance_usdz,
)


class _Tensor:
    """A checkpoint tensor stub that carries values, not just a shape.

    The export adapter measures position representability before handing the
    model to the pinned NuRec exporter, so a shape-only fake no longer stands
    in for a real checkpoint tensor.
    """

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        # Origin-centred and finite: representable room-scale coordinates.
        self._array = numpy.zeros(shape, dtype=numpy.float32)

    def detach(self) -> _Tensor:
        return self

    def __array__(self, dtype=None):
        return self._array.astype(dtype) if dtype is not None else self._array

    def __getitem__(self, key) -> _Tensor:
        selected = self._array[key]
        tensor = _Tensor(tuple(selected.shape))
        tensor._array = selected
        return tensor


def _load_real_provider_runner():
    path = Path(__file__).parents[1] / "scripts/public_scene_artifixer3d_runner.py"
    spec = importlib.util.spec_from_file_location("_exact_artifixer_provider_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reference_ply(path: Path, count: int) -> Path:
    return write_standard_3dgs_ply(
        SplatData(
            count=count,
            xyz=numpy.zeros((count, 3), dtype=numpy.float32),
            opacity=numpy.zeros(count, dtype=numpy.float32),
            f_dc=numpy.zeros((count, 3), dtype=numpy.float32),
            scales=numpy.zeros((count, 3), dtype=numpy.float32),
            quats=numpy.tile(
                numpy.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=numpy.float32),
                (count, 1),
            ),
            properties=(),
        ),
        path,
    )


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
        """A real NuRec package carrying the upstream exporter's axis matrix.

        Placeholder layers cannot exercise the export's own contract: the
        packaged volume has to compose at identity, and that is only checkable
        against a package a USD runtime can actually open.
        """

        def export(self, _model, path: Path, **_kwargs) -> None:
            write_appearance_usdz(
                Path(path),
                room_positions(count=256),
                matrix=EXPORTER_AXIS_MATRIX,
                payload_name="repaired_scene.nurec",
                payload_first=True,
            )
            members = []
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    body = archive.read(name)
                    if name.endswith(".nurec"):
                        # Non-zero gzip mtime, so the alignment pass's
                        # normalization stays exercised rather than trivial.
                        body = body[:4] + struct.pack("<I", 123) + body[8:]
                    members.append((name, body))
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
                for name, body in members:
                    archive.writestr(name, body)

    modules = {
        "torch": torch,
        "threedgrut": ModuleType("threedgrut"),
        "threedgrut.export": ModuleType("threedgrut.export"),
        "threedgrut.export.ply_exporter": ModuleType("threedgrut.export.ply_exporter"),
        "threedgrut.export.usdz_exporter": ModuleType("threedgrut.export.usdz_exporter"),
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
        reference_gaussian_ply=_reference_ply(tmp_path / "reference.ply", 1),
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
                "outside_support_invariance_status": ("deferred_until_final_soft_composite"),
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
    provider_native, raw, raw_path = _exact_provider_to_host_raw_result(tmp_path, monkeypatch)

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
    assert receipt["export_digest"] == canonical_digest(receipt, digest_field="export_digest")
    assert receipt["host_path_rebased_from_provider_runtime_output"] is True
    assert Path(receipt["isaac_nurec_usdz"]["path"]).is_file()


def test_native_export_handoff_rejects_mutated_rebased_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _provider_native, raw, raw_path = _exact_provider_to_host_raw_result(tmp_path, monkeypatch)
    Path(raw["tasks"][0]["native_appearance"]["isaac_nurec_usdz"]["path"]).write_bytes(
        b"mutated-after-host-closeout"
    )

    with pytest.raises(ArtiFixerNativeExportError, match="native_export_file_invalid"):
        materialize_artifixer3d_native_appearance_exports(
            raw_result_path=raw_path,
            output_root=tmp_path / "host/native_appearance_exports",
        )
