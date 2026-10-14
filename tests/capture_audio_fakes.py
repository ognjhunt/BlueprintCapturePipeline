"""Hermetic ffmpeg/ffprobe stand-ins for the capture audio-minimization contract.

No real media tool runs. The fake ``ffmpeg`` writes its input bytes behind a
marker to the output path, and the fake ``ffprobe`` reports a configured number
of audio streams, so a test can pin exactly which command ran, which bytes left
the process, and what the receipt concluded.

Only ``blueprint_pipeline.capture_audio_minimization`` sees the fakes; other
modules keep their own ``subprocess`` and ``shutil``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from blueprint_pipeline import capture_audio_minimization

VIDEO_ONLY_MARKER = b"video-only:"


class FakeMediaTools:
    def __init__(
        self,
        *,
        output_audio_streams: int = 0,
        ffmpeg_returncode: int = 0,
        ffmpeg: bool = True,
        ffprobe: bool = True,
    ) -> None:
        self.output_audio_streams = output_audio_streams
        self.ffmpeg_returncode = ffmpeg_returncode
        self.available = {"ffmpeg": ffmpeg, "ffprobe": ffprobe}
        self.commands: list[list[str]] = []

    def which(self, name: str) -> str | None:
        return f"/usr/bin/{name}" if self.available.get(name) else None

    def run(self, command: Any, **_kwargs: Any) -> subprocess.CompletedProcess:
        argv = [str(item) for item in command]
        self.commands.append(argv)
        tool = Path(argv[0]).name
        if tool == "ffprobe":
            streams = "".join(f"{index}\n" for index in range(1, self.output_audio_streams + 1))
            return subprocess.CompletedProcess(argv, 0, stdout=streams, stderr="")
        if tool == "ffmpeg":
            if self.ffmpeg_returncode == 0:
                source = Path(argv[argv.index("-i") + 1])
                Path(argv[-1]).write_bytes(VIDEO_ONLY_MARKER + source.read_bytes())
                return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
            return subprocess.CompletedProcess(argv, self.ffmpeg_returncode, stdout="", stderr="ffmpeg failed")
        raise AssertionError(f"unexpected media command: {argv}")

    def ffmpeg_commands(self) -> list[list[str]]:
        return [command for command in self.commands if Path(command[0]).name == "ffmpeg"]

    def install(self, monkeypatch: Any) -> "FakeMediaTools":
        monkeypatch.setattr(
            capture_audio_minimization,
            "subprocess",
            SimpleNamespace(run=self.run, SubprocessError=subprocess.SubprocessError),
        )
        monkeypatch.setattr(capture_audio_minimization, "shutil", SimpleNamespace(which=self.which))
        return self
