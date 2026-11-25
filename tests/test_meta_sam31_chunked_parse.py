"""A SAM answer is parsed a run of whole frame lines at a time, each run kept as it ends: a pass cut part way resumes.

The parser read a whole answer under one time limit, at most an hour. The 2026-11-24 sweep put a
prompt's answer for five minutes of 4K at about 37 minutes, and the one-hour cap past about nine
minutes: every retry then parsed the same kept answer from its start and was cut at the same
place. Each frame is one line (``<Nf>`` and its masks), read on its own; a run's measure is at most
``PARSE_CHUNK_SECONDS`` and its masks are kept as it ends.
"""
from __future__ import annotations

import json
import subprocess
from io import BytesIO

import numpy as np
import pytest
from meta_sam_parser._mask_codec import _encode_raster

from blueprint_pipeline import meta_sam31 as sam
from tests.test_meta_sam31 import PROMPT, inputs, response

WIDTH, HEIGHT = 64, 48


def _line(frame: int, objects: dict[str, tuple[int, int, int, int]], salt: int = 0) -> str:
    """A frame line of the provider's grammar: each object's box and its one-bit mask."""
    parts = []
    for object_id, (left, top, right, bottom) in objects.items():
        width, height = right - left, bottom - top
        rng = np.random.default_rng(frame * 31 + int(object_id) * 7 + salt)
        raster = (rng.random(width * height) < 0.6).astype(np.uint8).tolist()
        parts.append(f"{object_id}<|box;x1={left};y1={top};x2={right - 1};y2={bottom - 1};w={WIDTH};h={HEIGHT}|>"
                     f"<|mask;x=0;y=0;data={height},{width},{_encode_raster(raster, width, height)}|>")
    return f"<{frame}f>" + "".join(parts)


def _answer(frames: int) -> str:
    lines = []
    for frame in range(frames):
        objects = {"1": (2 + frame % 5, 3, 30, 40)}
        if frame % 3:  # a second object, seen in two frames of three
            objects["2"] = (32, 10 + frame % 4, 60, 44)
        lines.append(_line(frame, objects))
    lines.insert(7, _line(2, {"1": (5, 5, 20, 20)}, salt=9))  # frame 2 again, later: its mask replaces the first
    return "\n".join(lines) + "\n"


def _registry(frames: int) -> list[dict]:
    return [{"source_frame_id": f"decoded-{index:09d}", "width": WIDTH, "height": HEIGHT} for index in range(frames)]


def test_an_answer_is_cut_at_line_ends_into_runs_within_the_bound():
    text = _answer(40)
    chunks = sam.answer_chunks(text)
    assert "\n".join(chunks) == text
    assert all(chunk.count("\n") + 1 == len(chunk.split("\n")) for chunk in chunks)
    # A long 4K answer: each run's measure stays within the bound, however long the whole text's.
    box = "<|box;x1=0;y1=0;x2=3839;y2=2159;w=3840;h=2160|><|mask;x=0;y=0;data=2,2,!!!!!&T:u9`|>"
    long = "\n".join(f"<{frame}f>1{box}2{box}" for frame in range(2000))
    assert sam.parser_timeout_seconds(long) == sam.PARSER_TIMEOUT_BOUNDS_S[1]  # the whole text: cut at the cap
    pieces = sam.answer_chunks(long)
    assert len(pieces) > 20 and "\n".join(pieces) == long
    assert max(sam._parse_measure(len(piece.encode()), sam._box_area(piece)) for piece in pieces) \
        <= sam.PARSE_CHUNK_SECONDS


@pytest.mark.parametrize("bound", [1e-9, 0.002, 10**9])
def test_tracks_parsed_in_runs_are_the_tracks_of_the_whole_text(monkeypatch, bound):
    text, registry = _answer(40), _registry(40)
    whole = sam.parse_tracks(response(text), prompt=PROMPT, registry=registry)
    monkeypatch.setattr(sam, "PARSE_CHUNK_SECONDS", bound)
    assert (len(sam.answer_chunks(text)) > 1) is (bound < 1)
    chunked = sam.parse_tracks(response(text), prompt=PROMPT, registry=registry)
    assert chunked == whole
    assert [track["track_id"] for track in chunked] == ["meta-sam31-blue-container-1", "meta-sam31-blue-container-2"]
    # Frame 2's second line replaced its first mask, as in one text.
    two = next(row for row in chunked[0]["observations"] if row["source_frame_id"] == "decoded-000000002")
    assert sum(run["length"] for run in two["runs"]) <= 15 * 15


#: The module's parser, before any test wraps it.
REAL_PARSE = sam._parse_js_rasters


def _counted(monkeypatch, *, cut_at: int | None = None) -> list[int]:
    """Every run the parser is asked; with ``cut_at``, the parser is cut by its time limit at that run (from 1)."""
    asked: list[int] = []

    def parse(chunk, accept):
        asked.append(len(asked) + 1)
        if len(asked) == cut_at:
            raise subprocess.TimeoutExpired(cmd="node", timeout=1)
        return REAL_PARSE(chunk, accept)

    monkeypatch.setattr(sam, "_parse_js_rasters", parse)
    return asked


def test_a_cut_parse_resumes_at_the_first_run_not_kept(tmp_path, monkeypatch):
    text, registry = _answer(40), _registry(40)
    whole = sam.parse_tracks(response(text), prompt=PROMPT, registry=registry)
    monkeypatch.setattr(sam, "PARSE_CHUNK_SECONDS", 0.002)
    count = len(sam.answer_chunks(text))
    assert count >= 5
    binding = {"response_digest": "sha256:" + "1" * 64}
    _counted(monkeypatch, cut_at=4)
    with pytest.raises(subprocess.TimeoutExpired):
        sam.parse_tracks(response(text), prompt=PROMPT, registry=registry, resume_root=tmp_path, resume_binding=binding)
    assert sorted(path.name for path in tmp_path.iterdir()) == [f"chunk-{n:05d}.json" for n in range(3)]
    asked = _counted(monkeypatch)
    again = sam.parse_tracks(response(text), prompt=PROMPT, registry=registry, resume_root=tmp_path,
                             resume_binding=binding)
    assert again == whole and len(asked) == count - 3


def test_a_kept_run_of_another_answer_or_release_is_parsed_again_never_refused(tmp_path, monkeypatch):
    text, registry = _answer(30), _registry(30)
    monkeypatch.setattr(sam, "PARSE_CHUNK_SECONDS", 0.002)
    binding = {"response_digest": "sha256:" + "1" * 64}
    whole = sam.parse_tracks(response(text), prompt=PROMPT, registry=registry, resume_root=tmp_path,
                             resume_binding=binding)
    kept = tmp_path / "chunk-00001.json"
    record = json.loads(kept.read_text())
    record["observations"] = []  # a release that wrote it differently, or a damaged file: its digest no longer holds
    kept.write_text(json.dumps(record))
    (tmp_path / "chunk-00002.json").write_text("not json")
    asked = _counted(monkeypatch)
    assert sam.parse_tracks(response(text), prompt=PROMPT, registry=registry, resume_root=tmp_path,
                            resume_binding=binding) == whole
    assert len(asked) == 2
    # Another answer's runs are not this answer's.
    asked = _counted(monkeypatch)
    other = {"response_digest": "sha256:" + "2" * 64}
    assert sam.parse_tracks(response(text), prompt=PROMPT, registry=registry, resume_root=tmp_path,
                            resume_binding=other) == whole
    assert len(asked) == len(sam.answer_chunks(text))


def test_a_kept_answer_cut_while_parsing_resumes_on_the_next_run_and_keeps_its_bytes(tmp_path, monkeypatch):
    """Through ``run_meta_sam31``: the bought answer is kept, the cut parse resumes at its kept runs, nothing is
    bought again, and the parsed tracks are the bytes one whole parse writes."""
    lines = [_line(0, {"1": (0, 0, 2 + n % 2, 2 + (n // 2) % 2)}, salt=n).replace(f"w={WIDTH};h={HEIGHT}", "w=4;h=4")
             for n in range(6)]
    text = "\n".join(lines)
    monkeypatch.setenv("META_MODEL_API_KEY", "fixture-meta-secret")
    bought = []

    def opener(request, **kwargs):
        bought.append(request.full_url)
        return BytesIO(json.dumps(response(text)).encode())

    for name in ("whole", "cut"):
        (tmp_path / name).mkdir()
    whole_args = inputs(tmp_path / "whole")
    sam.run_meta_sam31(**whole_args, opener=opener)
    whole = next(whole_args["output_root"].rglob("parsed-response-0.json")).read_bytes()
    monkeypatch.setattr(sam, "PARSE_CHUNK_SECONDS", 1e-9)
    args = inputs(tmp_path / "cut")
    bought.clear()
    _counted(monkeypatch, cut_at=3)
    with pytest.raises(subprocess.TimeoutExpired):
        sam.run_meta_sam31(**args, opener=opener)
    assert len(bought) == 1 and len(list(args["output_root"].rglob("chunk-*.json"))) == 2
    asked = _counted(monkeypatch)
    result = sam.run_meta_sam31(**args, opener=lambda *a, **k: pytest.fail("the kept answer is not bought again"))
    assert result["status"] == "completed" and len(asked) == len(lines) - 2
    assert next(args["output_root"].rglob("parsed-response-0.json")).read_bytes() == whole
    assert not list(args["output_root"].rglob("chunk-*.json"))
