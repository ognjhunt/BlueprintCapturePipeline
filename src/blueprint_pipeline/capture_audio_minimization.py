"""Keep capture audio out of derived video and provider uploads.

A phone walkthrough of a workplace can record people talking. No analysis,
reconstruction, or evaluation stage uses sound, so none of the copies made from
a capture keep it:

- the raw capture stays exactly as received (raw truth is never rewritten);
- every privacy-processed walkthrough is video-only;
- every upload of capture video bytes to a model or reconstruction provider is
  made from a video-only copy, never from the source file.

A receipt says audio was removed only when the produced bytes show it: an
ffprobe count of audio streams in the output when ffprobe is available,
otherwise the exact encoder command (ffmpeg with ``-an`` that exited zero) or an
encoder that has no audio input at all. It never asserts a removal that did not
happen. When removal cannot be done or shown, callers fail closed rather than
publish or upload the file.

Backlog: website free-beta privacy minimization. Site-submitted videos are the
ADP-010 demand-discovery intake (partner-phase day-7 gate), and the same
capture rights/privacy contract carries into the ADP-021 fresh capture.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from .common import ensure_dir, sha256_file

AUDIO_REMOVAL_SCHEMA_VERSION = "capture_audio_removal.v1"

# Video-only stream copy: pixels and timing are untouched, sound is dropped.
STREAM_COPY_METHOD = "ffmpeg_video_stream_copy"

Runner = Callable[..., Any]
Which = Callable[[str], Optional[str]]


class AudioRemovalError(RuntimeError):
    """Raised when a video-only copy could not be produced or shown to be audio-free."""

    def __init__(self, receipt: Dict[str, Any]) -> None:
        self.receipt = dict(receipt)
        super().__init__(str(receipt.get("reason") or "capture_audio_removal_failed"))


def _runner(runner: Optional[Runner]) -> Runner:
    # Resolved at call time so callers and tests that patch subprocess.run see it.
    return runner or subprocess.run


def _which(which: Optional[Which]) -> Which:
    return which or shutil.which


def audio_free_remux_command(ffmpeg: str, source: Path, destination: Path) -> List[str]:
    """The exact ffmpeg command that copies the first video stream and nothing else."""

    return [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
        "-an",
        "-sn",
        "-dn",
        str(destination),
    ]


def video_only_reencode_command(
    ffmpeg: str,
    source: Path,
    destination: Path,
    attempt: Mapping[str, Any],
) -> List[str]:
    """A size-bounded H.264 clip of ``source``'s first video stream and nothing else.

    ``attempt`` carries ``clip_start_seconds``, ``actual_clip_seconds``,
    ``max_width``, ``crf``, ``maxrate`` and ``bufsize``. Used for provider
    inputs that must be re-encoded (World Labs), so the clip never carries the
    walkthrough's sound.
    """

    return [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{float(attempt['clip_start_seconds']):.3f}",
        "-i",
        str(source),
        "-t",
        f"{float(attempt['actual_clip_seconds']):.3f}",
        "-map",
        "0:v:0",
        "-vf",
        f"scale='min({int(attempt['max_width'])},iw)':-2",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(int(attempt["crf"])),
        "-maxrate",
        str(attempt["maxrate"]),
        "-bufsize",
        str(attempt["bufsize"]),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(destination),
    ]


def probe_audio_stream_count(
    path: Path,
    *,
    runner: Optional[Runner] = None,
    which: Optional[Which] = None,
) -> Optional[int]:
    """Count audio streams in ``path`` with ffprobe; ``None`` when that cannot be measured."""

    ffprobe = _which(which)("ffprobe")
    if not ffprobe:
        return None
    try:
        proc = _runner(runner)(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if getattr(proc, "returncode", 1) != 0:
        return None
    stdout = getattr(proc, "stdout", "") or ""
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="ignore")
    return sum(1 for line in str(stdout).splitlines() if line.strip())


def _public_command(command: Sequence[Any], *, source: Optional[Path], output: Path) -> List[str]:
    """The command as run, with local paths replaced so receipts carry no scratch paths."""

    replacements = {str(output): "{output}"}
    if source is not None:
        replacements[str(source)] = "{source}"
    rendered = [replacements.get(str(item), str(item)) for item in command]
    if rendered:
        rendered[0] = Path(rendered[0]).name or rendered[0]
    return rendered


def _base_receipt(*, method: str, command: Optional[List[str]]) -> Dict[str, Any]:
    return {
        "schema_version": AUDIO_REMOVAL_SCHEMA_VERSION,
        "method": method,
        "command": command,
        "audio_removed": False,
        "evidence": None,
        "output_audio_stream_count": None,
        "reason": None,
    }


def audio_removal_receipt(
    *,
    command: Sequence[Any],
    returncode: int,
    output: Path,
    method: str,
    source: Optional[Path] = None,
    runner: Optional[Runner] = None,
    which: Optional[Which] = None,
) -> Dict[str, Any]:
    """Describe what an ffmpeg run actually did to the audio of ``output``."""

    receipt = _base_receipt(method=method, command=_public_command(command, source=source, output=output))
    receipt["returncode"] = int(returncode)
    if returncode != 0 or not output.is_file():
        receipt.update(
            status="failed",
            evidence="encoder_failed",
            reason=f"audio_removal_encoder_failed:{returncode}",
        )
        return receipt
    count = probe_audio_stream_count(output, runner=runner, which=which)
    if count is not None:
        receipt["output_audio_stream_count"] = count
        if count == 0:
            receipt.update(status="removed", audio_removed=True, evidence="ffprobe_output_has_no_audio_stream")
        else:
            receipt.update(
                status="failed",
                evidence="ffprobe_output_has_audio_stream",
                reason="audio_stream_present_after_encode",
            )
        return receipt
    if "-an" in [str(item) for item in command]:
        receipt.update(status="removed", audio_removed=True, evidence="encoder_command_disables_audio")
    else:
        receipt.update(
            status="failed",
            evidence="encoder_command_keeps_audio",
            reason="audio_removal_not_requested_by_encoder_command",
        )
    return receipt


def video_only_encoder_receipt(
    output: Path,
    *,
    encoder: str,
    runner: Optional[Runner] = None,
    which: Optional[Which] = None,
) -> Dict[str, Any]:
    """Receipt for frames written by an encoder that has no audio input (e.g. OpenCV)."""

    receipt = _base_receipt(method=encoder, command=None)
    if not output.is_file():
        receipt.update(status="failed", evidence="output_missing", reason="video_only_output_missing")
        return receipt
    count = probe_audio_stream_count(output, runner=runner, which=which)
    if count is not None:
        receipt["output_audio_stream_count"] = count
        if count == 0:
            receipt.update(status="removed", audio_removed=True, evidence="ffprobe_output_has_no_audio_stream")
        else:
            receipt.update(
                status="failed",
                evidence="ffprobe_output_has_audio_stream",
                reason="audio_stream_present_after_encode",
            )
        return receipt
    receipt.update(status="removed", audio_removed=True, evidence="encoder_has_no_audio_input")
    return receipt


def write_audio_free_video(
    source: Path,
    destination: Path,
    *,
    runner: Optional[Runner] = None,
    which: Optional[Which] = None,
    timeout_seconds: int = 600,
) -> Dict[str, Any]:
    """Write a video-only copy of ``source`` to ``destination`` and return its receipt.

    ``source`` is only read. On any failure ``destination`` is removed so no
    audio-bearing derivative is left behind for a later stage to pick up.
    """

    ffmpeg = _which(which)("ffmpeg")
    if not ffmpeg:
        receipt = _base_receipt(method=STREAM_COPY_METHOD, command=None)
        receipt.update(status="failed", evidence="not_attempted", reason="ffmpeg_not_found")
        destination.unlink(missing_ok=True)
        return receipt
    ensure_dir(destination.parent)
    command = audio_free_remux_command(ffmpeg, source, destination)
    try:
        proc = _runner(runner)(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        returncode = int(getattr(proc, "returncode", 1))
        stderr = str(getattr(proc, "stderr", "") or "")
    except (OSError, subprocess.SubprocessError) as exc:
        returncode, stderr = -1, f"{type(exc).__name__}: {exc}"
    receipt = audio_removal_receipt(
        command=command,
        returncode=returncode,
        output=destination,
        method=STREAM_COPY_METHOD,
        source=source,
        runner=runner,
        which=which,
    )
    if not receipt["audio_removed"]:
        if stderr:
            receipt["stderr"] = stderr[-2000:]
        destination.unlink(missing_ok=True)
        return receipt
    receipt["output_sha256"] = sha256_file(destination)
    return receipt


@contextmanager
def audio_free_upload_copy(
    source: Path,
    *,
    runner: Optional[Runner] = None,
    which: Optional[Which] = None,
) -> Iterator[Tuple[Path, Dict[str, Any]]]:
    """Yield a temporary video-only copy of ``source`` for a provider upload.

    The copy keeps the source file name and container so the provider sees the
    same media type. Raises :class:`AudioRemovalError` instead of yielding when
    the copy cannot be shown to be audio-free; nothing should be uploaded then.
    """

    with tempfile.TemporaryDirectory(prefix="audio-free-upload-") as temp_dir:
        copy = Path(temp_dir) / (source.name or "capture-video.mp4")
        receipt = write_audio_free_video(source, copy, runner=runner, which=which)
        if not receipt["audio_removed"]:
            raise AudioRemovalError(receipt)
        yield copy, receipt


def audio_free_video_bytes(
    data: bytes,
    *,
    suffix: str,
    runner: Optional[Runner] = None,
    which: Optional[Which] = None,
) -> Tuple[bytes, Dict[str, Any]]:
    """Video-only bytes for an in-memory upload; raises :class:`AudioRemovalError`."""

    extension = suffix if suffix.startswith(".") else f".{suffix}"
    with tempfile.TemporaryDirectory(prefix="audio-free-bytes-") as temp_dir:
        source = Path(temp_dir) / f"source{extension}"
        source.write_bytes(data)
        with audio_free_upload_copy(source, runner=runner, which=which) as (copy, receipt):
            return copy.read_bytes(), receipt
