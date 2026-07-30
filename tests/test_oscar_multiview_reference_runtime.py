from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline.droid_oscar_closed_loop_adapter import EXTERIOR_VIEW
from blueprint_pipeline.oscar_multiview_reference_runtime import (
    OSCAR_NUM_FRAMES,
    OscarMultiViewReferenceRuntime,
)
from blueprint_pipeline.policy_ranking_thesis import file_sha256


class FakeWorker:
    instances: list["FakeWorker"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.requests: list[dict[str, Any]] = []
        self.closed = False
        self.__class__.instances.append(self)

    def start(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "cuda_device_name": "NVIDIA L40S",
            "checkpoint_sha256": "a" * 64,
        }

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(dict(request))
        output = Path(request["output_video"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"generated-video")
        return {
            "status": "ok",
            "blockers": [],
            "output_video": str(output),
            "runtime_result_id": f"result-{len(self.requests)}",
        }

    def close(self) -> None:
        self.closed = True

    def close_and_report(self, output_dir: Path) -> dict[str, Any]:
        self.closed = True
        return {"status": "completed", "output_dir": str(output_dir)}


def _runtime(tmp_path: Path, **kwargs: Any) -> OscarMultiViewReferenceRuntime:
    python = tmp_path / "python"
    python.write_text("python", encoding="utf-8")
    repo = tmp_path / "OSCAR"
    repo.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    seal = tmp_path / "seal.json"
    seal.write_text("{}", encoding="utf-8")
    assets = tmp_path / "assets"
    assets.mkdir()
    return OscarMultiViewReferenceRuntime(
        python=python,
        oscar_repo=repo,
        checkpoint=checkpoint,
        source_seal=seal,
        asset_cache=assets,
        evidence_dir=tmp_path / "evidence",
        worker_factory=kwargs.get("worker_factory", FakeWorker),
        provenance_verifier=kwargs.get(
            "provenance_verifier",
            lambda **_values: {"status": "passed", "blockers": []},
        ),
        asset_preflight=kwargs.get(
            "asset_preflight",
            lambda *_args, **_values: {"status": "passed", "blockers": []},
        ),
    )


def _view_request(tmp_path: Path) -> dict[str, Any]:
    first = tmp_path / "first.png"
    first.write_bytes(b"first-frame")
    skeleton = tmp_path / "skeleton.mp4"
    skeleton.write_bytes(b"skeleton-video")
    return {
        "first_frame_path": str(first),
        "first_frame_sha256": file_sha256(first),
        "skeleton_video_path": str(skeleton),
        "skeleton_video_sha256": file_sha256(skeleton),
        "camera_calibration_sha256": "c" * 64,
    }


def test_runtime_loads_once_and_dispatches_exact_view_bytes(tmp_path: Path) -> None:
    FakeWorker.instances.clear()
    runtime = _runtime(tmp_path)
    ready = runtime.start()

    receipt = runtime(
        view_id=EXTERIOR_VIEW,
        view_request=_view_request(tmp_path),
        task_prompt="Pick up the spray can.",
        negative_prompt="ignored by official runtime",
        output_dir=tmp_path / "view-output",
        seed=42,
    )
    report = runtime.close()

    assert ready["status"] == "ready"
    assert len(FakeWorker.instances) == 1
    worker = FakeWorker.instances[0]
    assert worker.requests[0]["num_frames"] == OSCAR_NUM_FRAMES
    assert worker.requests[0]["seed"] == 42
    assert worker.requests[0]["negative_prompt"] == "ignored by official runtime"
    assert worker.requests[0]["reference_frame_path"] == str(
        Path(_view_request(tmp_path)["first_frame_path"]).resolve()
    )
    assert Path(receipt["generated_video_path"]).read_bytes() == b"generated-video"
    assert receipt["provider"] == "resident_official_oscar_multiview"
    assert receipt["official_negative_prompt_parameter_supported"] is True
    assert worker.closed is True
    assert report is not None


def test_runtime_fails_closed_on_conditioning_hash_mismatch(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.start()
    request = _view_request(tmp_path)
    request["skeleton_video_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="skeleton_video_sha256_mismatch"):
        runtime(
            view_id=EXTERIOR_VIEW,
            view_request=request,
            task_prompt="task",
            negative_prompt="negative",
            output_dir=tmp_path / "view-output",
            seed=42,
        )
    runtime.close()


@pytest.mark.parametrize("gate", ["provenance", "assets"])
def test_runtime_fails_before_worker_start_when_identity_gate_blocks(
    tmp_path: Path, gate: str
) -> None:
    FakeWorker.instances.clear()
    runtime = _runtime(
        tmp_path,
        provenance_verifier=(
            (lambda **_values: {"status": "blocked", "blockers": ["bad-source"]})
            if gate == "provenance"
            else (lambda **_values: {"status": "passed", "blockers": []})
        ),
        asset_preflight=(
            (lambda *_args, **_values: {"status": "blocked", "blockers": ["bad-assets"]})
            if gate == "assets"
            else (lambda *_args, **_values: {"status": "passed", "blockers": []})
        ),
    )

    expected = "runtime_source_provenance" if gate == "provenance" else "runtime_asset_preflight"
    with pytest.raises(ValueError, match=expected):
        runtime.start()
    assert FakeWorker.instances == []


def test_runtime_forbids_output_path_reuse(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.start()
    request = _view_request(tmp_path)
    output = tmp_path / "view-output"
    runtime(
        view_id=EXTERIOR_VIEW,
        view_request=request,
        task_prompt="task",
        negative_prompt="negative",
        output_dir=output,
        seed=42,
    )
    with pytest.raises(ValueError, match="output_reuse_forbidden"):
        runtime(
            view_id=EXTERIOR_VIEW,
            view_request=request,
            task_prompt="task",
            negative_prompt="negative",
            output_dir=output,
            seed=42,
        )
    runtime.close()
