"""Legacy optional detector cannot write into the original capture bundle."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace


def test_extraction_writes_only_under_requested_output(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("sam3_detect_test", Path(__file__).parents[1] / "scripts/sam3_detect.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = tmp_path / "raw"
    raw.mkdir()
    video = raw / "walkthrough.mov"
    video.write_bytes(b"original video")
    def ffmpeg(argv, **kwargs):
        Path(argv[-1]).write_bytes(b"frame")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, "run", ffmpeg)
    output = tmp_path / "pipeline/object-index-run"
    frames = module._extract_frames(video, n_frames=2, output_root=output)
    assert len(frames) == 2
    assert all(p.is_relative_to(output) for p in frames)
    assert list(raw.iterdir()) == [video]
    assert video.read_bytes() == b"original video"
