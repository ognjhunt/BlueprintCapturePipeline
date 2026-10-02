"""The enrolled handoff downloader must consume the selected object version."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest


def test_selected_generation_is_the_actual_download_precondition(tmp_path, monkeypatch):
    from blueprint_pipeline import pubsub_handoff_disk_admission as admission

    calls = []

    class Blob:
        name = "scenes/scene-1/captures/capture-1/raw/video.mov"
        generation = "17000000000000000001"

        def download_to_file(self, stream, **kwargs):
            calls.append(kwargs)
            stream.write(b"v1")

    class Reservation:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(admission, "reserve_control_plane_disk", lambda *_, **__: Reservation())
    monkeypatch.setattr(admission, "keep_reservation_live",
                        lambda _: nullcontext(SimpleNamespace(check=lambda: None)))
    blob = Blob()
    target = tmp_path / "capture" / "raw" / "video.mov"
    admission.download_with_reservation(
        downloads=[(blob, target)],
        manifest_rows=[{"name": blob.name, "size": 2, "generation": blob.generation}],
        selected_generations={blob.name: blob.generation},
        storage_root=tmp_path, capture_root=tmp_path / "capture",
    )
    assert target.read_bytes() == b"v1"
    assert calls == [{"if_generation_match": int(blob.generation), "timeout": 60, "retry": None}]


def test_selected_generation_mismatch_refuses_before_disk_reservation(tmp_path, monkeypatch):
    from blueprint_pipeline import pubsub_handoff_disk_admission as admission
    from blueprint_pipeline.common import PipelineError

    monkeypatch.setattr(admission, "reserve_control_plane_disk",
                        lambda *_, **__: pytest.fail("unproved version reserved disk"))
    blob = SimpleNamespace(name="scenes/scene-1/captures/capture-1/raw/video.mov",
                           generation="17000000000000000002")
    target = tmp_path / "capture" / "raw" / "video.mov"
    with pytest.raises(PipelineError, match="handoff_source_generation_unproven"):
        admission.download_with_reservation(
            downloads=[(blob, target)],
            manifest_rows=[{"name": blob.name, "size": 2, "generation": blob.generation}],
            selected_generations={blob.name: "17000000000000000001"},
            storage_root=tmp_path, capture_root=tmp_path / "capture",
        )
    assert not target.exists()
