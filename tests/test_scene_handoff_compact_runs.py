"""The scene handoff keeps kept surfaces' full-video tracks compact, and every persisted byte stays what it was.

``complete_retained_static_task_masks`` selects the kept surfaces (shelves, the floor) from the
clean plate's first paid SAM response. Their source tracks span every frame at full resolution:
on a 4K, 437-frame job with four large surfaces, 2.57 million runs. The handoff used to receive
them as one Python dict per run (about 760 MiB) and then hash, copy and write them with the
stdlib. It now receives ``RunLengths`` and every consumer hashes, detaches and writes them with
``website_track_mask_codec``. The pinned values below were produced by the code before this
change (9d77246) on the same synthetic job (``tests/meta_sam31_synthetic.py``); no customer data
is involved.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline import website_track_mask_codec as codec
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_track_mask_codec import (RunLengths, canonical_track_digest, detached_track_json,
                                                         load_track_json, plain_runs)
from tests import meta_sam31_synthetic as synthetic
from tests.test_meta_sam31_memory import _compacted, _payloads, stdlib_digest, written

REPOSITORY = Path(__file__).resolve().parents[1]


def _same(expected, got):
    """Equal and of the same types all the way down (``==`` alone lets 1 equal 1.0 and True)."""
    if type(expected) is not type(got):
        return False
    if type(expected) is dict:
        return list(expected) == list(got) and all(_same(expected[key], got[key]) for key in expected)
    if type(expected) is list:
        return len(expected) == len(got) and all(_same(a, b) for a, b in zip(expected, got))
    if type(expected) is RunLengths:
        return expected == got and expected.starts.dtype == got.starts.dtype
    if type(expected) is float:
        return (math.isnan(expected) and math.isnan(got)) or (
            expected == got and math.copysign(1, expected) == math.copysign(1, got))
    return expected == got


def _documents(seed):
    rng = random.Random(seed)
    scalars = [None, True, False, 0, -1, 2 ** 63, 2 ** 80, 1.5, -0.0, 1e-300, float("nan"), float("inf"),
               -float("inf"), "", "runs", "a]b", 'q"uote', "back\\slash", "ünï ☃ \U0001f600", " ", "\x01"]

    def runs():
        draw, count = rng.random(), rng.randrange(0, 6)
        if draw < 0.6:
            return [{"start": rng.randrange(-5, 10 ** 12), "length": rng.randrange(-3, 5000)} for _ in range(count)]
        if draw < 0.7:
            return [{"length": 1, "start": 2}] * count                      # keys out of order
        if draw < 0.8:
            return [{"start": rng.choice([1.0, 2 ** 70]), "length": 2}] * max(1, count)  # float, beyond int64
        if draw < 0.9:
            return [{"start": 1, "length": 2, "x": [1]}] * max(1, count)    # another key
        return [[1, 2], "]"]

    def value(depth=0):
        draw = rng.random()
        if depth > 4 or draw < 0.3:
            return rng.choice(scalars)
        if draw < 0.5:
            return [value(depth + 1) for _ in range(rng.randrange(4))]
        if draw < 0.7:
            return {"runs": runs(), "width": 3, **{f"k{i}": value(depth + 1) for i in range(rng.randrange(3))}}
        return {rng.choice(["a", "runs", "é"]) + rng.choice(["", "1"]): value(depth + 1) for _ in range(rng.randrange(4))}

    styles = [dict(indent=2), dict(separators=(",", ":")), dict(ensure_ascii=False),
              dict(indent="\t", separators=(" ,\r\n", " : "))]
    for _ in range(300):
        yield json.dumps(value(), **rng.choice(styles))
    # Repeated keys, malformed and truncated text, a BOM, odd numbers and literals.
    yield from ['{"runs": [{"start": 1, "length": 2}], "runs": [{"start": 3, "length": 4}]}',
                '{"a": 1, "a": {"runs": []}}', '[1, 2,]', '{"a" 1}', '{"runs": [{"start": 01, "length": 1}]}',
                '{"runs": [{"start": 1, "length": 1}', '[1] x', '﻿[1]', '"\\ud800"',
                '[NaN, Infinity, -Infinity, -0, 1E5, 1e-2, 0.5]', '[1.]', '[-]', '{"runs": [ ]}',
                '{"runs": [{"start": 1, "length": 1},]}', '{"runs": [{"start": 1, "length": 1e2}]}',
                '[' * 40 + ']' * 40, '"\t"', '[tru]', 'nul', '', '   ']


def _outcome(read):
    try:
        return "value", read()
    except Exception as exc:  # noqa: BLE001 - the error itself is what is compared
        return "error", (type(exc), str(exc))


@pytest.mark.parametrize("characters", [1, 3, 64, 1 << 20])
def test_streamed_track_json_reads_what_json_loads_reads(tmp_path, monkeypatch, characters):
    # Whatever the read size, every value and every refusal is json.loads's own.
    monkeypatch.setattr(codec, "_READ_CHARACTERS", characters)
    path = tmp_path / "x.json"
    for index, text in enumerate(_documents(31)):
        path.write_text(text)
        expected = _outcome(lambda: json.loads(text, object_hook=codec._compact_observation))
        got = _outcome(lambda: load_track_json(path))
        assert expected[0] == got[0], (index, text[:120])
        assert _same(*[row[1] for row in (expected, got)]) if expected[0] == "value" else expected == got, index
        if expected[0] == "value" and not text.startswith("﻿"):
            # Well-formed text is read by the streaming reader itself, never by the fallback.
            assert _same(expected[1], codec._StreamingTrackReader(io.StringIO(text)).document()), index


def test_streamed_track_json_never_holds_the_text_or_a_dict_per_run(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    observations = [{"source_frame_id": f"decoded-{index:09d}", "width": 3840, "height": 2160,
                     "runs": [{"start": int(start), "length": int(length)} for start, length in
                              zip(np.sort(rng.integers(0, 8_000_000, 2000)), rng.integers(1, 4000, 2000))]}
                    for index in range(150)]
    path = tmp_path / "tracks.json"
    synthetic_tracks = {"tracks": [{"track_id": "t", "observations": observations}]}
    path.write_text(json.dumps(synthetic_tracks, indent=2))
    expected = json.loads(path.read_text(), object_hook=codec._compact_observation)
    size = path.stat().st_size
    assert size > 20_000_000
    monkeypatch.setattr(Path, "read_text", lambda *_, **__: pytest.fail("the fallback read the text whole"))
    tracemalloc.start()
    try:
        loaded = load_track_json(path)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert _same(expected, loaded) and {type(row["runs"]) for row in observations} == {list}
    compact = sum(row["runs"].nbytes for row in loaded["tracks"][0]["observations"])
    # A few reads of about a MB and the compact arrays; never the 25 MB of text, nor a dict per run.
    assert peak < compact + (8 << 20) and peak < size / 3, (peak, compact, size)


def test_detached_track_json_is_the_json_round_trip_with_runs_kept_compact():
    for payload in _payloads(9):
        expected = _outcome(lambda: json.loads(json.dumps(payload, allow_nan=False)))
        for form in (payload, _compacted(payload)):
            got = _outcome(lambda: detached_track_json(form))
            assert expected[0] == got[0]
            if expected[0] == "error":
                assert expected == got
                continue
            assert _same(expected[1], plain_runs(got[1]))
            kept = [value for value in _walk(got[1]) if type(value) is RunLengths]
            assert all(any(value is source for source in _walk(form)) for value in kept)  # shared, not copied
    loop = {"a": []}
    loop["a"].append(loop)
    for value in (loop, {"x": float("nan")}, {"x": object()}, {"x": 10 ** 5000}):
        assert _outcome(lambda: detached_track_json(value)) == _outcome(
            lambda: json.loads(json.dumps(value, allow_nan=False)))


def _walk(value):
    yield value
    if type(value) is dict:
        for item in value.values():
            yield from _walk(item)
    elif type(value) in (list, tuple):
        for item in value:
            yield from _walk(item)


# The handoff's files on the small synthetic job (72 frames, 96x54, shelves, floor, a box, people),
# produced by the code before compact handoff runs (9d77246) through the plain route
# (``run_scene_handoff(compact=False)``). The run's root path appears in the paid receipts, so the
# digests the root path reaches are blanked (they are checked against the stdlib directly).
SMALL_HANDOFF_BLANKED = {
    "task_masks.json": "e9ce3a6f3a0043ec184ef9a82fe68dfe21f9d5b6aaadb3677687d48ca5c150a0",
    "task_masks.geometry.json": "d19a5eb93a2642e1293c1f8584a128848f7d565dcc468ff67bfbfd8ebe1cd4f0",
    "preparation.json": "a5240bd6f05de68fb4e03c89f2a0f39478d21deea01ecf0cd37476e0b85d36fb",
}
#: Values the run's root path reaches: through the paid receipts' paths and every digest over them.
ROOT_BOUND = {("digest",), ("retained_static_source", "provider_result_digest"),
              ("retained_static_source", "retained_video_receipt_digest"), ("binding", "task_masks_digest"),
              ("task_evaluation_inputs", "source_bindings", "task_masks_digest")}


def blanked_digest(value):
    """sha256 of ``json.dumps(plain_runs(value), indent=2)`` with the root-bound values written ``<root-bound>``.

    Hashed from the codec's chunks (the same text; ``test_meta_sam31_memory`` proves it), so a
    large job is never rendered whole; the pre-change codec has the same renderer.
    """
    def blank(node, path):
        if type(node) is not dict:
            return node
        out = dict(node)
        for key in out:
            here = (*path, key)
            if here in ROOT_BOUND:
                out[key] = "<root-bound>"
            elif any(bound[:len(here)] == here for bound in ROOT_BOUND):
                out[key] = blank(out[key], here)
        return out

    digest = hashlib.sha256()
    for chunk in codec._indent_chunks(blank(dict(value), ()), 0, set()):
        digest.update(chunk.encode())
    return digest.hexdigest()


def test_small_synthetic_handoff_persists_exactly_the_pre_change_bytes(tmp_path, monkeypatch):
    frames, width, height = 72, 96, 54
    monkeypatch.setenv("META_MODEL_API_KEY", "synthetic-key")
    monkeypatch.setenv("BLUEPRINT_WEBSITE_SAM31_PROVIDER", "meta")
    job = synthetic.run_clean_plate(monkeypatch.setattr, tmp_path, frames=frames, width=width, height=height,
                                    targets=synthetic.MIXED)
    monkeypatch.chdir(tmp_path)
    result = synthetic.run_scene_handoff(tmp_path, job, width=width, height=height)
    completed, bound, preparation = result["completed"], result["bound"], result["preparation"]
    kept = set(job["task_masks"]["deferred_target_ids"])
    # Kept surfaces arrive compact; the first pass's targets (the box, the people) stay as they came.
    for target in bound["targets"]:
        kinds = {type(row["runs"]) for row in target["source_track"]["observations"] if len(row["runs"])}
        assert kinds == ({RunLengths} if target["target_id"] in kept else {list}), target["target_id"]
    assert {type(row["runs"]) for target in bound["targets"] for row in target["track"]["observations"]} == {list}
    workspace = tmp_path / "website_scene_preparation"
    files = {"task_masks.json": (next(tmp_path.rglob("task_masks.json")), completed),
             "task_masks.geometry.json": (workspace / "task_masks.geometry.json", bound),
             "preparation.json": (workspace / "family" / "preparation.json", preparation)}
    for name, (path, value) in files.items():
        plain = plain_runs(value)
        # The bytes and digest the stdlib gives the plain lists: write_json, canonical_digest.
        assert path.read_text() == written(plain), name
        assert value["digest"] == stdlib_digest(plain, "digest") == canonical_track_digest(value, digest_field="digest")
        assert blanked_digest(value) == SMALL_HANDOFF_BLANKED[name], name
    # The family preparation leaves with plain lists (its record goes on to the handoff and status
    # readers); ``source_targets`` carry every full source track, and equal the bound targets.
    assert all(type(row["runs"]) is list for target in preparation["task_evaluation_inputs"]["source_targets"]
               for track in (target["source_track"], target["track"]) for row in track["observations"])
    assert preparation["task_evaluation_inputs"]["source_targets"] == bound["targets"]
    assert preparation["binding"]["task_masks_digest"] == bound["digest"]
    # A rerun reads its own retained bytes back and changes nothing.
    before = synthetic.persisted_digests(tmp_path)
    again = synthetic.run_scene_handoff(tmp_path, job, width=width, height=height)
    assert again["bound"] == bound and synthetic.persisted_digests(tmp_path) == before


def _with_source_tracks(masks):
    """``masks`` with a full-resolution source track of exact runs beside each target's track."""
    value = copy.deepcopy(masks)
    for index, target in enumerate(value["targets"]):
        observations = [{"source_frame_id": row["source_frame_id"], "width": 640, "height": 480,
                         "runs": [{"start": (200 + index * 40 + line) * 640 + 100, "length": 50 + index}
                                  for line in range(30)]}
                        for row in target["track"]["observations"]]
        target["source_track"] = {"track_id": target["track"]["track_id"], "label": target["target_id"],
                                  "observations": observations}
    value["digest"] = canonical_digest(value, digest_field="digest")
    return value


def _snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def test_rigid_compile_reads_compact_masks_as_it_reads_plain_ones(tmp_path):
    from tests.test_website_task_preparation import _arguments
    from blueprint_pipeline import website_task_preparation as preparation

    outputs = []
    for form in (lambda masks: masks, _compacted):
        arguments = _arguments(tmp_path)
        arguments["task_masks"] = form(_with_source_tracks(arguments["task_masks"]))
        shutil.rmtree(tmp_path / "out", ignore_errors=True)
        value = preparation.compile_website_scene_preparation(**arguments)
        outputs.append((value, _snapshot(tmp_path / "out")))
    assert any(type(target["source_track"]["observations"][0]["runs"]) is RunLengths
               for target in arguments["task_masks"]["targets"])
    assert outputs[0][0]["status"] == outputs[1][0]["status"]
    assert outputs[0] == outputs[1]


def test_target_correction_derives_the_same_request_from_compact_source_masks():
    from blueprint_pipeline.website_target_background_correction import derive_review
    from tests.test_website_target_background_correction import case

    plain = derive_review(**case())
    inputs = case()
    inputs["targets"] = _compacted(inputs["targets"])
    assert any(type(row["runs"]) is RunLengths for target in inputs["targets"]
               for row in target["source_track"]["observations"])
    compact = derive_review(**inputs)
    assert compact == plain and compact["digest"] == plain["digest"] == canonical_digest(compact, digest_field="digest")
    # Only the geometry frames' source masks are retained, as plain lists.
    assert {type(row["runs"]) for track in compact["selected_tracks"].values() for row in track["observations"]} <= {list}


def test_furniture_segmentation_asks_for_compact_runs_and_lifts_them_as_plain_ones(tmp_path, monkeypatch):
    from blueprint_pipeline import meta_sam31, robot_access_measurement as measurement
    from blueprint_pipeline import website_task_masks as masks

    seen = {}

    def run(**kwargs):
        seen.update(kwargs)
        return {"tracks": []}

    with monkeypatch.context() as patch:
        patch.setattr(meta_sam31, "prepare_continuous_video", lambda **_: ([{"source_frame_id": "a"}], {}))
        patch.setattr(meta_sam31, "run_meta_sam31", run)
        measurement._segment(video_path=str(tmp_path / "v.mp4"), source_video_digest="sha256:v",
                             prompts=[{"text": "chair"}], output_root=tmp_path, task_context={}, maximum_cost_usd=1.0)
    assert seen["compact_runs"] is True
    # ``lift`` gives the same pieces from compact tracks as from plain ones.
    frames, width, height = 40, 96, 54
    monkeypatch.setenv("META_MODEL_API_KEY", "synthetic-key")
    monkeypatch.setenv("BLUEPRINT_WEBSITE_SAM31_PROVIDER", "meta")
    job = synthetic.run_clean_plate(monkeypatch.setattr, tmp_path / "job", frames=frames, width=width, height=height)
    paid_root = next((tmp_path / "job").rglob("tracks.json")).parent
    binding = json.loads((paid_root / "binding.json").read_text())
    replay = dict(frame_registry=binding["frame_registry"], frame_artifacts=[], prompts=binding["prompts"],
                  output_root=paid_root.parent, admission={},
                  video_artifact={**binding["continuous_video"], "path": str(paid_root.parent / "continuous-upright.mp4")})
    monkeypatch.chdir(tmp_path)
    geometry = synthetic.flat_depth_geometry(job["source_geometry"], width=width, height=height)
    lifted = [measurement.lift(job["plan"]["targets"], registry=binding["frame_registry"],
                               tracks=masks.run_meta_sam31(**replay, compact_runs=compact)["tracks"],
                               prompts=binding["prompts"], geometry=geometry) for compact in (False, True)]
    assert lifted[0] == lifted[1] and len(lifted[0][0]) == len(synthetic.SHELVES)


# 3840x2160, the first 110 of the job's 437 frames, four large surfaces (647,000 runs): the clean
# plate, then the handoff's stages on its retained output, in a fresh interpreter. Blanked digests
# produced by the pre-change code (9d77246) through the plain route on the same job.
SCALED_HANDOFF_BLANKED = {
    "task_masks.json": "3fca55ce1d703dae090a504b2be44cbab9eead1be137b3474e27f8052ba88683",
    "task_masks.geometry.json": "d967fea577242ba76d550a7b923f02983906d2a4e42076b35e67c7a68f942e92",
    "preparation.json": "ac86af02db5c4f889e92c905e6f8725a9240b8536fef9f54b4204552abaf25ae",
}
#: Growth of peak resident memory over the handoff's own stages (completion, binding and the
#: geometry file, the input digest checks), after the clean plate. The pre-change plain route
#: grew by about 213 MiB on this case (835 MiB at all 437 frames); the compact route by about
#: 58 MiB (70 MiB at 437 frames). The bound leaves room for other hosts and allocators and
#: still fails below the old behaviour.
HANDOFF_RSS_GROWTH_BOUND_BYTES = 128 << 20
#: The address-space cap on that interpreter: a regression fails there with MemoryError.
HANDOFF_ADDRESS_SPACE_CAP_MIB = 1536


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="peak RSS and the address-space cap are Linux")
def test_scaled_production_shaped_handoff_stays_bounded_and_byte_identical(tmp_path):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPOSITORY / "src"), str(REPOSITORY)]),
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "META_MODEL_API_KEY": "synthetic-key", "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-m", "tests.meta_sam31_synthetic", str(tmp_path),
                             str(110), "3840", "2160", str(HANDOFF_ADDRESS_SPACE_CAP_MIB), "handoff"],
                            cwd=REPOSITORY, env=env, capture_output=True, text=True, timeout=900, check=False)
    assert result.returncode == 0, result.stderr[-4000:]
    measured = json.loads(result.stdout.strip().splitlines()[-1])
    # The handoff's own path: completion, binding and the geometry file, the input digest checks.
    growth = measured["stage_peak_rss_bytes"]["checked"] - measured["baseline_peak_rss_bytes"]
    assert growth < HANDOFF_RSS_GROWTH_BOUND_BYTES, measured
    workspace = tmp_path / "website_scene_preparation"
    for name, path in {"task_masks.json": next(tmp_path.rglob("task_masks.json")),
                       "task_masks.geometry.json": workspace / "task_masks.geometry.json",
                       "preparation.json": workspace / "family" / "preparation.json"}.items():
        value = load_track_json(path)
        assert value["digest"] == canonical_track_digest(value, digest_field="digest"), name
        assert blanked_digest(value) == SCALED_HANDOFF_BLANKED[name], name
