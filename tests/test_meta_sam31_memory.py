"""Hosted SAM 3.1 tracks stay compact in memory, and every persisted byte stays what it was.

A 4K, 437-frame website job with four large surfaces (two side shelves, an end shelf, the floor)
was killed by the host's OOM killer in the clean-plate stage: each mask of each frame built a
full-frame array and then one Python dict per pixel run (about one per image row), and
``json.dumps(..., indent=2)`` then held several string pieces per run before writing. Runs are now
computed inside each mask's box, held as one integer array per observation, and hashed and written
chunk by chunk. The pinned digests below were produced by the code before that change on the same
synthetic inputs (``tests/meta_sam31_synthetic.py``); no customer data is involved.
"""
from __future__ import annotations

import collections
import copy
import enum
import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline import common, meta_sam31 as sam
from blueprint_pipeline.core import common as core_common
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_track_mask_codec import (RunLengths, canonical_track_digest, decode_track_mask,
                                                         load_track_json, plain_runs, raster_runs,
                                                         write_track_json)
from tests import meta_sam31_synthetic as synthetic

REPOSITORY = Path(__file__).resolve().parents[1]


def full_frame_runs(raster, *, left, top, width, height):
    """The pre-change computation: paste into a frame-sized array, difference it flattened as int8."""
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top:top + raster.shape[0], left:left + raster.shape[1]] = raster
    edges = np.diff(np.pad(mask.reshape(-1).astype(np.int8), (1, 1)))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [{"start": int(start), "length": int(end - start)} for start, end in zip(starts, ends)]


def written(payload):
    """The pre-change ``write_json`` text."""
    text = json.dumps(payload, indent=2)
    return text.replace("<redacted:secret-field>", "REDACTED_SECRET_FIELD").replace("<redacted:secret>",
                                                                                    "REDACTED_SECRET")


def stdlib_digest(payload, digest_field=None):
    value = {key: item for key, item in payload.items() if key != digest_field}
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_box_runs_equal_the_runs_of_the_mask_pasted_into_a_full_frame():
    generator = np.random.default_rng(31)
    for case in range(3000):
        width, height = int(generator.integers(1, 48)), int(generator.integers(1, 32))
        left, top = int(generator.integers(0, width)), int(generator.integers(0, height))
        right, bottom = int(generator.integers(left + 1, width + 1)), int(generator.integers(top + 1, height + 1))
        if case % 7 == 0:
            left, right = 0, width  # a box as wide as the frame: its runs continue across rows
        shape = (bottom - top, right - left)
        raster = [lambda: generator.random(shape) < 0.5, lambda: np.ones(shape, bool),
                  lambda: np.zeros(shape, bool),
                  lambda: generator.integers(0, 256, shape)][case % 4]().astype(np.uint8)
        compact = raster_runs(raster, left=left, top=top, width=width)
        expected = full_frame_runs(raster, left=left, top=top, width=width, height=height)
        assert compact == expected and expected == compact and plain_runs(compact) == expected, case


def test_parsed_tracks_hold_compact_runs_equal_to_the_full_frame_computation(monkeypatch):
    width, height = 40, 24
    masks = [("7", 0, [3, 2, 19, 11], np.ones((9, 16), np.uint8)),
             ("7", 1, [0, 20, 40, 24], np.ones((4, 40), np.uint8)),              # full width, last rows
             ("8", 0, [30, 0, 40, 24], np.eye(24, 10, dtype=np.uint8)),          # touches the right edge
             ("8", 2, [5, 5, 35, 21], np.ones((4, 6), np.uint8))]                # resized up to its box
    registry = [{"source_frame_id": f"f{i}", "width": width, "height": height} for i in range(3)]

    def parse(text, accept):
        for object_id, index, box, raster in masks:
            accept(object_id, index, box, raster.shape[1], raster.shape[0], raster.tobytes())
        return True

    monkeypatch.setattr(sam, "_parse_js_rasters", parse)
    response = {"status": "completed", "output": []}
    tracks = sam.parse_tracks(response, prompt={"prompt_id": "p", "output_label": "shelf"}, registry=registry)
    assert [track["track_id"] for track in tracks] == ["meta-sam31-p-7", "meta-sam31-p-8"]
    observed = {(track["track_id"][-1], row["source_frame_id"]): row for track in tracks for row in track["observations"]}
    for object_id, index, (left, top, right, bottom), raster in masks:
        if raster.shape != (bottom - top, right - left):
            raster = np.asarray(Image.fromarray(raster).resize((right - left, bottom - top), Image.Resampling.NEAREST))
        row = observed[(object_id, f"f{index}")]
        assert isinstance(row["runs"], RunLengths)
        assert row["runs"] == full_frame_runs(raster, left=left, top=top, width=width, height=height)
        assert decode_track_mask(row).sum() == int(raster.astype(bool).sum())


class _Count(enum.IntEnum):
    TWO = 2


def _payloads(seed):
    rng = random.Random(seed)
    scalars = [None, True, False, 0, -1, 2 ** 70, 1.5, -0.0, float("nan"), float("inf"), -float("inf"), 1e300,
               np.float64(0.25), _Count.TWO, "", "plain", "ünïcode ☃", "<redacted:secret>",
               "a<redacted:secret-field>b", "\n\t\"\\ "]

    def value(depth=0):
        draw = rng.random()
        if depth > 4 or draw < 0.3:
            return rng.choice(scalars)
        if draw < 0.45:
            return [value(depth + 1) for _ in range(rng.randrange(4))]
        if draw < 0.5:
            return tuple(value(depth + 1) for _ in range(rng.randrange(3)))
        if draw < 0.55:
            return {rng.choice([1, 2.5, None, True]): value(depth + 1) for _ in range(rng.randrange(3))}
        if draw < 0.6:
            return collections.OrderedDict((f"o{index}", value(depth + 1)) for index in range(rng.randrange(3)))
        if draw < 0.75:
            return [{"start": rng.randrange(10 ** 7), "length": rng.randrange(1, 4000)}
                    for _ in range(rng.randrange(6))]
        return {rng.choice(["b", "a", "runs", "é", "<redacted:secret>"]) + str(index): value(depth + 1)
                for index in range(rng.randrange(5))}

    for _ in range(400):
        yield {"p": value(), "q": value()}


def _compacted(value):
    if type(value) is list and value and RunLengths.from_runs(value) is not None:
        return RunLengths.from_runs(value)
    if type(value) is dict:
        return {key: _compacted(item) for key, item in value.items()}
    if type(value) is list:
        return [_compacted(item) for item in value]
    return value


def test_track_writers_give_write_json_and_canonical_digest_bytes(tmp_path, monkeypatch):
    # Batch boundaries every few characters must not change a byte, redaction included.
    monkeypatch.setattr(core_common, "_JSON_WRITE_BATCH", 7)
    for payload in _payloads(5):
        expected = written(payload)
        common.write_json(tmp_path / "common.json", payload)
        assert (tmp_path / "common.json").read_text() == expected
        try:
            digest = stdlib_digest(payload)
        except TypeError:  # keys of mixed types cannot be sorted, before or after
            digest = None
        assert digest is None or canonical_digest(payload) == digest
        for form in (payload, _compacted(payload)):
            write_track_json(tmp_path / "track.json", form)
            assert (tmp_path / "track.json").read_text() == expected
            if digest is None:
                with pytest.raises(TypeError):
                    canonical_track_digest(form)
            else:
                assert canonical_track_digest(form) == digest


def test_writers_refuse_what_json_refuses(tmp_path):
    loop = {"a": []}
    loop["a"].append(loop)
    for writer in (write_track_json, common.write_json):
        with pytest.raises(TypeError, match="Object of type object is not JSON serializable"):
            writer(tmp_path / "new" / "x.json", {"runs": object()})
        with pytest.raises(ValueError, match="Circular reference detected"):
            writer(tmp_path / "x.json", loop)
        assert list(tmp_path.iterdir()) == []  # no file, no temporary, no directory left behind
    with pytest.raises(TypeError, match="Object of type RunLengths is not JSON serializable"):
        common.write_json(tmp_path / "x.json", {"runs": RunLengths([1], [2])})  # compact runs stay in the codec
    with pytest.raises(ValueError, match="Circular reference detected"):
        canonical_track_digest(loop)


def test_compact_runs_read_compare_and_copy_like_the_list_of_dicts(tmp_path):
    plain = [{"start": 3, "length": 2}, {"start": 9, "length": 4}, {"start": 30, "length": 1}]
    runs = RunLengths.from_runs(plain)
    assert len(runs) == 3 and runs and not RunLengths() and list(runs) == plain and runs[-1] == plain[-1]
    assert runs[1:] == plain[1:] and runs == plain and plain == runs and not runs != plain
    assert runs != plain[:2] and runs != tuple(plain) and runs != [{**plain[0], "x": 1}, *plain[1:]]
    assert pickle.loads(pickle.dumps(runs)) == runs and copy.deepcopy({"runs": runs})["runs"] == plain
    assert sum(run["length"] for run in runs) == 7 and runs.starts.dtype == np.int32
    observation = {"width": 8, "height": 5, "runs": runs}
    assert np.array_equal(decode_track_mask(observation), decode_track_mask({**observation, "runs": plain}))
    with pytest.raises(ValueError, match="run_invalid"):
        decode_track_mask({**observation, "runs": RunLengths([38], [3])})
    # Only runs that re-encode to the same bytes are compacted when read back.
    rows = {"exact": plain, "extra_key": [{"start": 1, "length": 2, "probability": 0.9}],
            "reordered": [{"length": 2, "start": 1}], "float": [{"start": 1.0, "length": 2}],
            "boolean": [{"start": True, "length": 2}], "huge": [{"start": 2 ** 70, "length": 1}], "empty": []}
    (tmp_path / "rows.json").write_text(json.dumps([{"runs": value} for value in rows.values()]))
    loaded = load_track_json(tmp_path / "rows.json")
    assert [type(row["runs"]) is RunLengths for row in loaded] == [True] + [False] * 6
    assert loaded == [{"runs": value} for value in rows.values()]


def _reference_tracks(prompt, *, frames, width, height):
    """``parse_tracks`` output computed the pre-change way, for one synthetic prompt."""
    observations = {}
    for index in range(frames):
        for object_id, (left, top, right, bottom), raster in synthetic.synthetic_masks(prompt["text"], index,
                                                                                       width, height):
            observations.setdefault(object_id, {})[index] = {
                "source_frame_id": f"decoded-{index:09d}", "width": width, "height": height,
                "runs": full_frame_runs(raster, left=left, top=top, width=width, height=height)}
    return [{"track_id": f"meta-sam31-{prompt['prompt_id']}-{object_id}", "label": prompt["output_label"],
             "label_source": "model_inferred", "observations": [rows[i] for i in sorted(rows)]}
            for object_id, rows in observations.items()]


# Produced by the pre-change code (69e48f8c5) on the same synthetic job; the root path reads <root>.
SMALL_JOB_DIGESTS = {
    "binding.json": "8270f461a87069870e49c8c20bf0701f10ecf37d2d81ea769b3c97bac920ef57",
    "parsed-response-0.json": "deeaf3a77de009639406d07c4fc94ea61d02746be549d6de24b857eeb2231013",
    "parsed-response-1.json": "81431ad57eab5c4f3c1c4bb903a34c7ed677ba6fcc691d35c30cc6b87388c8bc",
    "parsed-response-2.json": "0786e42ae4a8391c36caa176151ed0304340616873b133eda896179a1871074e",
    "parsed-response-3.json": "3ea44a6a0ecbd29ebd8f9e155d49fd6a3660b465985b1f231672e60f085b4298",
    "parsed-response-4.json": "0babee2c11c1a131488d864070cb5ead57d30ebb71665d253eec7c9d96eb2bf9",
    "parsed-response-5.json": "c51c39ea8a199249dfee555f37ca0e8019a25a60c0e0da6f3f2cec3aaa10580a",
    "tracks.json": "61cdd0f5fda43a6df525ad394628c47f957af97aafca6a2e036b46f95ce594ef",
    "task_masks.object_removal.json": "b97c30af9e40175426d65e2aec6f018afb547cf7145446cfe5ee866caecead93",
    "clean_plate_stage_manifest.json": "a42ad4b081e7d28901fe17e7e55fe21bab86782df6348e0df93ef4a26b7ab8c7",
}
# The completed manifest with its three path-bound digests blanked (they are checked directly).
SMALL_JOB_COMPLETED_BLANKED = "75b3d982787fa24e58aeb09a77cbf6bb064c62d336ab5a68aac544f517b33b8d"


def _by_name(digests):
    return {Path(path).name: value for path, value in digests.items()}


def test_small_synthetic_job_persists_exactly_the_pre_change_bytes(tmp_path, monkeypatch):
    from blueprint_pipeline import website_task_masks as masks

    frames, width, height = 72, 96, 54
    monkeypatch.setenv("META_MODEL_API_KEY", "synthetic-key")
    monkeypatch.setenv("BLUEPRINT_WEBSITE_SAM31_PROVIDER", "meta")
    job = synthetic.run_clean_plate(monkeypatch.setattr, tmp_path, frames=frames, width=width, height=height,
                                    targets=synthetic.MIXED)
    paid_root = next(tmp_path.rglob("tracks.json")).parent
    # Full text: the provider receipts against the pre-change computation and ``json.dumps``.
    prompts = json.loads((paid_root / "binding.json").read_text())["prompts"]
    every = []
    for index, prompt in enumerate(prompts):
        text = (paid_root / f"parsed-response-{index}.json").read_text()
        stored = json.loads(text)
        expected = {"binding": stored["binding"],
                    "tracks": _reference_tracks(prompt, frames=frames, width=width, height=height),
                    "parser_runtime": stored["parser_runtime"]}
        expected["digest"] = stdlib_digest(expected, "digest")
        assert text == written(expected)
        every.extend(expected["tracks"])
    text = (paid_root / "tracks.json").read_text()
    stored = json.loads(text)
    assert text == written({"schema_version": "website_meta_sam31_tracks.v1", "status": "completed",
                            "binding_digest": stored["binding_digest"], "profile": sam.PROFILE, "tracks": every,
                            "responses": stored["responses"], "claim_ceiling": "development_only"})
    # The returned manifest is the plain form, written and hashed exactly as before.
    manifest = job["task_masks"]
    assert all(type(row["runs"]) is list for target in manifest["targets"]
               for track in (target["track"], target["source_track"]) for row in track["observations"])
    assert manifest["digest"] == stdlib_digest(manifest, "digest")
    assert next(tmp_path.rglob("task_masks.object_removal.json")).read_text() == written(manifest)
    # Every public result carries plain runs, as callers outside the task-mask modules expect;
    # only an explicit ``compact_runs=True`` hands out ``RunLengths``. The replay buys nothing.
    binding = json.loads((paid_root / "binding.json").read_text())
    replay = dict(frame_registry=binding["frame_registry"], frame_artifacts=[], prompts=binding["prompts"],
                  output_root=paid_root.parent, admission={},
                  video_artifact={**binding["continuous_video"], "path": str(paid_root.parent / "continuous-upright.mp4")})
    plain, compact = masks.run_meta_sam31(**replay), masks.run_meta_sam31(**replay, compact_runs=True)
    assert plain == compact and plain["tracks"] == stored["tracks"]
    assert {type(row["runs"]) for track in plain["tracks"] for row in track["observations"]} == {list}
    assert {type(row["runs"]) for track in compact["tracks"] for row in track["observations"]} == {RunLengths}
    people = [track for track in compact["tracks"] if track["label"] == "person"]
    union = masks.union_person_tracks(people, label="person")
    assert {type(row["runs"]) for row in union["observations"]} == {list}
    assert union == next(row["source_track"] for row in manifest["targets"] if row["target_id"] == "person")

    completed = masks.complete_retained_static_task_masks(
        plan=job["plan"], source_geometry=job["source_geometry"], task_masks=manifest,
        output_root=tmp_path / "clean_plate" / "task_masks", view_plan_root=tmp_path / "clean_plate" / "mask_view_plan")
    path = next(tmp_path.rglob("task_masks.json"))
    assert path.read_text() == written(completed) and completed["digest"] == stdlib_digest(completed, "digest")
    source = completed["retained_static_source"]
    assert source["provider_result_digest"] == "sha256:" + hashlib.sha256((paid_root / "tracks.json").read_bytes()).hexdigest()
    blank = {**completed, "digest": "<digest>", "retained_static_source": {
        **source, "provider_result_digest": "<provider_result_digest>",
        "retained_video_receipt_digest": "<retained_video_receipt_digest>"}}
    assert hashlib.sha256(json.dumps(blank, indent=2).encode()).hexdigest() == SMALL_JOB_COMPLETED_BLANKED
    assert {name: value for name, value in _by_name(synthetic.persisted_digests(tmp_path)).items()
            if name in SMALL_JOB_DIGESTS} == SMALL_JOB_DIGESTS


# 3840x2160, the first 110 of the job's 437 frames (a quarter), four large surfaces: 647,000 runs.
# Produced by the pre-change code (69e48f8c5) on the same synthetic job; the root path reads <root>.
SCALED_JOB_FRAMES = 110
SCALED_JOB_DIGESTS = {
    "binding.json": "8c55039ad4caf74aa61aecd502fa940d13fff6ebf671f56cbbf6eb2f91478e4b",
    "parsed-response-0.json": "e8f4a0ffd92325fd83d05e053ea2eaf6a4beadf168a0aaec6dda39c79816a2ef",
    "parsed-response-1.json": "a3f149b28a4e0d5d0ade2e09fc2d8c1497ae90b82c978341fc1f2e5a551770a9",
    "parsed-response-2.json": "cd1593fe951a656f90f88e21b25029add960c001091c3f8dad0a9cbb55b3f2ce",
    "parsed-response-3.json": "db1a33c4046af9a95d5c2ac6c3e55c304d84634a1fb29c4f8b48206167575b06",
    "tracks.json": "4b1171eafce8f4471cf33122452f321f059ed1ed9f5baffcedebfb64b91965cf",
    "task_masks.object_removal.json": "1730543d5c79589960553cbc27f4589ede90387948cc2debba9dc03f4f1bcecf",
    "clean_plate_stage_manifest.json": "6a550f5d71c4af004b35d436d7b7460612e3517d797105b96c54ce0bd6010147",
}
#: Growth of peak resident memory while the clean-plate SAM step runs in a fresh interpreter
#: (after imports and synthetic setup). The pre-change code grew by about 570 MiB on this case
#: (and 2.1 GiB at all 437 frames); the compact path by about 25 MiB (49 MiB at 437 frames). The
#: bound leaves room for other hosts and allocators and still fails far below the old behaviour.
PEAK_RSS_GROWTH_BOUND_BYTES = 192 << 20
#: Address-space cap on that interpreter: a regression fails there with MemoryError, so this test
#: never needs more than about 1 GiB however the code behaves.
ADDRESS_SPACE_CAP_MIB = 1024


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="peak RSS and the address-space cap are Linux")
def test_scaled_production_shaped_job_stays_bounded_and_byte_identical(tmp_path):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPOSITORY / "src"), str(REPOSITORY)]),
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "META_MODEL_API_KEY": "synthetic-key", "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-m", "tests.meta_sam31_synthetic", str(tmp_path),
                             str(SCALED_JOB_FRAMES), "3840", "2160", str(ADDRESS_SPACE_CAP_MIB)],
                            cwd=REPOSITORY, env=env, capture_output=True, text=True, timeout=600, check=False)
    assert result.returncode == 0, result.stderr[-4000:]
    measured = json.loads(result.stdout.strip().splitlines()[-1])
    assert {name: value for name, value in _by_name(measured["persisted"]).items()
            if name in SCALED_JOB_DIGESTS} == SCALED_JOB_DIGESTS
    growth = measured["peak_rss_bytes"] - measured["baseline_peak_rss_bytes"]
    assert growth < PEAK_RSS_GROWTH_BOUND_BYTES, measured
