"""The clean plate keeps its tracks compact through to the scene handoff, and every persisted byte stays what it was.

The clean-plate stage asked ``run_website_task_masks`` for plain runs, so every run of the tracks it
selects (the job's object and the people, each spanning the whole video at full resolution) became
a Python dict of about 256 bytes, against 8 bytes compact, and was then written with the
pure-Python JSON encoder. On a 21-second 1920x1080 site job the stage manifest is 222 MB, about
two million runs; a five-minute 4K clip with a half-frame object reaches millions of runs a track,
gigabytes on a lane bounded at 5 GiB. The stage now asks for compact runs and writes the stage
manifest with ``website_track_mask_codec``; the scene handoff, which already takes compact runs,
receives them. The pinned values below were produced by the code before this change (aedc281c5)
on the same synthetic jobs (``tests/meta_sam31_synthetic.py``); no customer data is involved.
"""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline import website_task_masks as masks
from blueprint_pipeline import website_track_mask_codec as codec
from blueprint_pipeline.website_task_target_choice import TaskTargetAmbiguous, _run_length
from blueprint_pipeline.website_track_mask_codec import (RunLengths, canonical_track_digest, load_track_json,
                                                         plain_runs)
from tests import meta_sam31_synthetic as synthetic
from tests.test_meta_sam31_memory import stdlib_digest, written
from tests.test_scene_handoff_compact_runs import blanked_digest

REPOSITORY = Path(__file__).resolve().parents[1]
FRAMES, WIDTH, HEIGHT = 72, 96, 54
#: Files whose bytes reach the run's root path through a digest over a path: checked blanked, and against the stdlib.
ROOT_BOUND_FILES = ("continuous-video.json", "task_masks.json", "task_masks.geometry.json", "preparation.json")


def pinned_files(root: Path) -> dict[str, str]:
    """sha256 of each persisted JSON file the root path does not reach, keyed by its path with digest folders as ``*``."""
    digests = synthetic.persisted_digests(synthetic.capture_root(root))
    return {re.sub(r"(?<=/)[0-9a-f]{16,64}(?=/)", "*", name): value for name, value in digests.items()
            if Path(name).name not in ROOT_BOUND_FILES}


def stage_and_handoff(set_attribute, root: Path, targets: tuple[str, ...]) -> dict:
    """The clean-plate stage on the small synthetic job, then the scene handoff's stages on what it returned."""
    job = synthetic.run_clean_plate_stage(set_attribute, root, frames=FRAMES, width=WIDTH, height=HEIGHT,
                                          targets=targets)
    pipeline = synthetic.capture_root(root) / "pipeline"
    os.chdir(pipeline)
    handed = synthetic.run_scene_handoff(pipeline, job, width=WIDTH, height=HEIGHT)
    return {"job": job, "handed": handed, "pipeline": pipeline}


def blanked(run: dict) -> dict[str, str]:
    job, handed = run["job"], run["handed"]
    return {"task_masks": blanked_digest(job["task_masks"]), "completed": blanked_digest(handed["completed"]),
            "bound": blanked_digest(handed["bound"]), "preparation": blanked_digest(handed["preparation"])}


# Produced by the code before this change (aedc281c5) through ``stage_and_handoff``.
SMALL_STAGE = {
    "large": {
        "files": {
            "pipeline/clean_plate/clean_plate_stage_manifest.json":
                "208549263a2b3ec9b3b4f524cf0e1c844b1cada97b681b93c3f74477a17ddd4c",
            "pipeline/clean_plate/current.json":
                "47282019dd89182815d1424771a44e8ae92647bfe80d3fac0851ab2098320092",
            "pipeline/clean_plate/removal_manifest.json":
                "4bfd3cda0c8412d744c50598c9a29c0557f03ce69ab1c1b32314073207797eab",
            "pipeline/clean_plate/removal_plan.json":
                "82449481b81c8926ecd64dafa308e5d1f827613914477625069beb1079f972d9",
            "pipeline/clean_plate/task_masks/*/*/binding.json":
                "4ecdfb45d49e44adbc997a726f3950a06be1efe5f5d0757ce861804750d93e98",
            "pipeline/clean_plate/task_masks/*/*/intent-0.json":
                "41d3289394eebc81122426d58f355a2169e0d31984d128444e133f44ff3f87ee",
            "pipeline/clean_plate/task_masks/*/*/intent-1.json":
                "41d3289394eebc81122426d58f355a2169e0d31984d128444e133f44ff3f87ee",
            "pipeline/clean_plate/task_masks/*/*/intent-2.json":
                "41d3289394eebc81122426d58f355a2169e0d31984d128444e133f44ff3f87ee",
            "pipeline/clean_plate/task_masks/*/*/intent-3.json":
                "41d3289394eebc81122426d58f355a2169e0d31984d128444e133f44ff3f87ee",
            "pipeline/clean_plate/task_masks/*/*/intent-4.json":
                "41d3289394eebc81122426d58f355a2169e0d31984d128444e133f44ff3f87ee",
            "pipeline/clean_plate/task_masks/*/*/intent-5.json":
                "41d3289394eebc81122426d58f355a2169e0d31984d128444e133f44ff3f87ee",
            "pipeline/clean_plate/task_masks/*/*/parsed-response-0.json":
                "fd29eb259aff3e6fc726c0baafb3fbad2aa0d3c388aac77069c20457b78a446a",
            "pipeline/clean_plate/task_masks/*/*/parsed-response-1.json":
                "6ef97366680ac85142000e3e19c4c0599c6229ed923269d8c7d0fd2044cef4ab",
            "pipeline/clean_plate/task_masks/*/*/parsed-response-2.json":
                "47b3d44bb9cd780498782bbb908129fb2a9fe5af76ee2675ec4a157b105e1272",
            "pipeline/clean_plate/task_masks/*/*/parsed-response-3.json":
                "3667d4c6f037a7193091d3801af5da9c2b7a07a040ddb477f6c6c69114f86efc",
            "pipeline/clean_plate/task_masks/*/*/parsed-response-4.json":
                "2b6d4263ec85987ac1707dec43b3cfc603e029fb48e222b0912d1786a53f42ac",
            "pipeline/clean_plate/task_masks/*/*/parsed-response-5.json":
                "bacc8e441bf6d5f5e641ac95304f2df0e5cd429579b6be45f69e659630a34423",
            "pipeline/clean_plate/task_masks/*/*/response-0.json":
                "78f4be00da142f6dae27e8298e54da9e2d94f26fd6ae00f36ccb9bd3e90ca890",
            "pipeline/clean_plate/task_masks/*/*/response-1.json":
                "ccb37562f7a9427c9e6e1104eabe7ac85db37d2aea09768ac67df8a7a080631c",
            "pipeline/clean_plate/task_masks/*/*/response-2.json":
                "f20b2158e5120a7cf2e750e9ccfb381ff8981fec41cf50a222174e11ab886e7f",
            "pipeline/clean_plate/task_masks/*/*/response-3.json":
                "5a190e51b3afb696c7dbfea3d9802ac12ab34f3a08721346af435a1b31cb8fc4",
            "pipeline/clean_plate/task_masks/*/*/response-4.json":
                "e82f3955b8a6b7dc89ec1acf734b4306bd534244116361673f4cc43d9dc92e88",
            "pipeline/clean_plate/task_masks/*/*/response-5.json":
                "6bb6643c954483e0bfa24308b137016b2e5cb04c97d7cd928ebd645ae1ba9024",
            "pipeline/clean_plate/task_masks/*/*/tracks.json":
                "a96c61d2319ea96515a65cd554cc66f473db29f06893d1aa4bcd24afa74c9feb",
            "pipeline/clean_plate/task_masks/*/*/uploaded-video.json":
                "fab363944c5a161c8682a9857286ac78291384d3a7e88030388116addb9de7b2",
            "pipeline/clean_plate/task_masks/*/*/video-cleanup.json":
                "e81a7b932c1df445e880a937f5ac6cf9f97606d6cfe86e451829208d30e73a76",
            "pipeline/clean_plate/task_masks/*/*/video-upload-intent.json":
                "feb89a6db9a7ffe274df73a72f69fbe268851f9799f0b2f5a38313560a7a176a",
            "pipeline/clean_plate/task_masks/*/task_masks.object_removal.json":
                "fcb55eaddc2528a9cad34758c18398a65e94ab195a2f05e616299e80e503a609",
            "pipeline/website_task_target_choices/current.json":
                "4d897712d88a2c59f28d0dc5c09a4f22c13c3e46b9a7ffeb0c4eb684b92d8a89",
        },
        "blanked": {
            "task_masks": "d694353e59ed7d0229af335940ba02a8bb0c7dbedcb1bc969ccc1d1d5ea91ca1",
            "completed": "db6741030efd594452e829cc6bb27efd4d32e56a4a83c6cd56ba2737f26f9f45",
            "bound": "31fafe86c516a78a17daffc37cde9d7641e4e9dbca8e9deb39de837f8c77d85e",
            "preparation": "0e404f779489a381264d25d21461b99327c5a50238f5def54791e3b475d819b3",
        },
        "returned_digest": "sha256:48d5556df8ae1102eec4338de3f8440c83f70ddfabe182058bc0f3fa3c6edf65",
    },
    "large_ambiguous": {
        "files": {
            "pipeline/clean_plate/clean_plate_stage_manifest.json":
                "4b1c68580372a68aa37d7b81b0e8e055ba3562adc5908e18228c05fea92e017a",
            "pipeline/clean_plate/current.json":
                "47282019dd89182815d1424771a44e8ae92647bfe80d3fac0851ab2098320092",
            "pipeline/clean_plate/removal_manifest.json":
                "aaadf101634add7c7a0c8565aa85dbb02d139a581dac0543f6c8384a03d3cfaf",
            "pipeline/clean_plate/removal_plan.json":
                "12160b1ec9ae1d2738a2d930f1cd5bd5d0a518b36d99ad49dab15601fc8861f6",
            "pipeline/clean_plate/task_masks/*/*/binding.json":
                "1c11ea997ac4b8877e61882c0f7790835b583c854e5f49a81af5cbec2fcd3850",
            "pipeline/clean_plate/task_masks/*/*/intent-0.json":
                "2b92f748d6eb9bd531ed5d666e11617513195e1cf6f641998661f9175277f4a7",
            "pipeline/clean_plate/task_masks/*/*/intent-1.json":
                "2b92f748d6eb9bd531ed5d666e11617513195e1cf6f641998661f9175277f4a7",
            "pipeline/clean_plate/task_masks/*/*/intent-2.json":
                "2b92f748d6eb9bd531ed5d666e11617513195e1cf6f641998661f9175277f4a7",
            "pipeline/clean_plate/task_masks/*/*/intent-3.json":
                "2b92f748d6eb9bd531ed5d666e11617513195e1cf6f641998661f9175277f4a7",
            "pipeline/clean_plate/task_masks/*/*/intent-4.json":
                "2b92f748d6eb9bd531ed5d666e11617513195e1cf6f641998661f9175277f4a7",
            "pipeline/clean_plate/task_masks/*/*/intent-5.json":
                "2b92f748d6eb9bd531ed5d666e11617513195e1cf6f641998661f9175277f4a7",
            "pipeline/clean_plate/task_masks/*/*/parsed-response-0.json":
                "d5c912e08b79e2eb1296773a5355d630fce83f0991e2c8d2fef92585ff267a61",
            "pipeline/clean_plate/task_masks/*/*/parsed-response-1.json":
                "48b89d5c8a695093eefb29b42b6956d6932cf66f2108192e678863d34b4c0c3c",
            "pipeline/clean_plate/task_masks/*/*/parsed-response-2.json":
                "6b0c137dd9d90a177a877e0b71078eeb39d2b9e73de67ea37d0161e1f5393767",
            "pipeline/clean_plate/task_masks/*/*/parsed-response-3.json":
                "ab42ff1b90f5dcb1ffc6fc490e36ae37a9e1bc97a493282304c655bbe7cb7cf9",
            "pipeline/clean_plate/task_masks/*/*/parsed-response-4.json":
                "9f3189dd9921f0b138c6b23e37cbb373f6cf2190d04b991e79de1611ede5bdc7",
            "pipeline/clean_plate/task_masks/*/*/parsed-response-5.json":
                "878867dd850ecc8e0be54b5e66035059ea1de1bb0abee74df5f4e93375963d0c",
            "pipeline/clean_plate/task_masks/*/*/response-0.json":
                "0efd0cbb4abfd510d59061bc8acd62be800786194d71b3c869d9689e6aa5c2cb",
            "pipeline/clean_plate/task_masks/*/*/response-1.json":
                "94d57b6bd19f4fa7af514319740dc9c27e48f3ba25c471d799f9aaba24ad55ae",
            "pipeline/clean_plate/task_masks/*/*/response-2.json":
                "0c725a7d27614a6045cf9351a424711038cf6498a9f0df3664a69e9bbc8b986e",
            "pipeline/clean_plate/task_masks/*/*/response-3.json":
                "849190060bcb1edac9036024f7ce61318008623c073aca47be3672cb8182b216",
            "pipeline/clean_plate/task_masks/*/*/response-4.json":
                "7ecf9fadfba0b008ae08a45fd0f0f9d28f989966385dd4139c35c452db3d25d8",
            "pipeline/clean_plate/task_masks/*/*/response-5.json":
                "cb7331190d7b2cd42f6a24c81d88fea3d2c2fa9dd0c8296f5200c1fc6df4ab52",
            "pipeline/clean_plate/task_masks/*/*/tracks.json":
                "e5dd1fa7c61d61cfc52526ef25f20c20d0d347c74b0179f97f6a8558021d6034",
            "pipeline/clean_plate/task_masks/*/*/uploaded-video.json":
                "fab363944c5a161c8682a9857286ac78291384d3a7e88030388116addb9de7b2",
            "pipeline/clean_plate/task_masks/*/*/video-cleanup.json":
                "e81a7b932c1df445e880a937f5ac6cf9f97606d6cfe86e451829208d30e73a76",
            "pipeline/clean_plate/task_masks/*/*/video-upload-intent.json":
                "feb89a6db9a7ffe274df73a72f69fbe268851f9799f0b2f5a38313560a7a176a",
            "pipeline/clean_plate/task_masks/*/task_masks.object_removal.json":
                "b30c6d320df4273cb8fd9cbe43f32da5d4bb08080ac4d4880aa1e5dee906373d",
            "pipeline/website_task_target_choices/current.json":
                "e998dc05b82d535c6262561f08c74643db189fec0a566838ba8f982830771aec",
            "pipeline/website_task_target_choices/twin_crate.question.json":
                "b268d443cd43964d3a0214cf223a52742cfb268fe47d1e379aff6e9e58337971",
        },
        "blanked": {
            "task_masks": "557d11367395cb0ad325603a00788624c8d806609672aafc3348a4daf322605a",
            "completed": "8ca7ae6fc1981584438117159e163dae7cb8cbf47d32bbf66e2f55123aeca266",
            "bound": "fb5207e78ccf12e4b3e675e32d78e6949761454cfcf5bd3f1ae27c5f9f505026",
            "preparation": "1f43113df57fa359a5569fb67b4d1ffa3ae26183bba5ee84ebc0f0f8d1dcc837",
        },
        "returned_digest": "sha256:8fb73539919d7aae58ccb0e92d12009011a2625137f53781f00dd5894e367432",
    },
}


@pytest.mark.parametrize("name", sorted(SMALL_STAGE))
def test_small_clean_plate_stage_returns_compact_runs_and_persists_the_pre_change_bytes(tmp_path, monkeypatch, name):
    monkeypatch.setenv("META_MODEL_API_KEY", "synthetic-key")
    monkeypatch.setenv("BLUEPRINT_WEBSITE_SAM31_PROVIDER", "meta")
    monkeypatch.chdir(tmp_path)
    run = stage_and_handoff(monkeypatch.setattr, tmp_path, synthetic.TARGET_SETS[name])
    job, pinned = run["job"], SMALL_STAGE[name]
    task_masks = job["task_masks"]
    # The selected targets' whole-video source tracks reach the handoff compact.
    assert {type(row["runs"]) for target in task_masks["targets"]
            for row in target["source_track"]["observations"]} == {RunLengths}
    assert task_masks["digest"] == pinned["returned_digest"]
    assert task_masks["digest"] == stdlib_digest(plain_runs(task_masks), "digest")
    # The stage manifest is the bytes ``write_json`` gives the plain lists, and carries the returned masks.
    manifest_path = run["pipeline"] / "clean_plate" / "clean_plate_stage_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest_path.read_text() == written(manifest)
    assert manifest["task_masks"] == plain_runs(task_masks)
    assert pinned_files(tmp_path) == pinned["files"]
    assert blanked(run) == pinned["blanked"]
    # Each file the root path reaches is the bytes the stdlib gives the plain value it was written from.
    handed = run["handed"]
    workspace = run["pipeline"] / "website_scene_preparation"
    for path, value in ((workspace / "task_masks.geometry.json", handed["bound"]),
                        (workspace / "family" / "preparation.json", handed["preparation"])):
        assert path.read_text() == written(plain_runs(value))
        assert value["digest"] == stdlib_digest(plain_runs(value), "digest")
    if name == "large_ambiguous":
        # The guessed object's question names the same candidates, bound to the same masks.
        question = run["pipeline"] / "website_task_target_choices" / "twin_crate.question.json"
        assert json.loads(question.read_text())["candidate_set_digest"]
    # A retry reads its retained SAM answers back and changes no byte.
    before = synthetic.persisted_digests(synthetic.capture_root(tmp_path))
    again = synthetic.run_clean_plate_stage(monkeypatch.setattr, tmp_path, frames=FRAMES, width=WIDTH, height=HEIGHT,
                                            targets=synthetic.TARGET_SETS[name])
    assert again["task_masks"] == task_masks
    assert synthetic.persisted_digests(synthetic.capture_root(tmp_path)) == before


def test_a_guess_and_an_ambiguity_keep_compact_runs_only_for_a_caller_that_asks(monkeypatch):
    runs = RunLengths([5, 40], [3, 7])
    candidates = [{"track_id": "a", "observations": [{"source_frame_id": "f", "width": 8, "height": 8,
                                                      "runs": runs}]}]

    @masks._plain_ambiguity
    def refuse(**_):
        raise TaskTargetAmbiguous("crate", candidates=candidates)

    for compact in (False, True):
        with pytest.raises(TaskTargetAmbiguous) as caught:
            refuse(compact_runs=compact)
        kept = caught.value.candidates[0]["observations"][0]["runs"]
        assert kept == runs.to_list() and (type(kept) is RunLengths) is compact
    # The guess's ranking reads a compact track's lengths as one array, to the value the dicts give.
    rng = np.random.default_rng(3)
    for lengths in (rng.integers(1, 4000, 5000), np.array([], dtype=np.int64), np.array([2 ** 40, 3])):
        compact = RunLengths(np.arange(len(lengths)) * 5000, lengths)
        assert _run_length({"runs": compact}) == _run_length({"runs": compact.to_list()}) == int(sum(lengths.tolist()))


@pytest.mark.parametrize("text", [
    '[{"start": 0, "length": 1}, {"start": -0, "length": -3}]',
    '[\r\n\t{"start":12,"length":3},{\n"start" : 9223372036854775807 , "length":1}]',
    '[{"start": 9223372036854775808, "length": 1}]',            # beyond int64: kept as its dicts
    '[{"start": 1, "length": 2}, {"start": 3, "length": 4, "x": 5}]',
])
def test_streamed_run_lists_read_the_integers_json_reads(tmp_path, text):
    document = '{"observations": [{"width": 4, "runs": ' + text + '}]}'
    path = tmp_path / "x.json"
    path.write_text(document)
    expected = json.loads(document)
    for loaded in (load_track_json(path), codec._StreamingTrackReader(io.StringIO(document)).document()):
        runs = loaded["observations"][0]["runs"]
        assert loaded == expected and canonical_track_digest(loaded) == stdlib_digest(expected)
        assert type(runs) is type(codec._compact_observation(dict(expected["observations"][0]))["runs"])


# 1920x1080, 450 frames (15 s at 30 fps): the clean-plate stage itself, in a fresh interpreter. Its
# job's object fills half the frame's rows and two people stay in view; the run goes on with its
# best supported guess for a twin object (``LARGE_AMBIGUOUS``). Produced by the code before this
# change (aedc281c5) on the same synthetic job.
SCALED_STAGE_FRAMES, SCALED_STAGE_WIDTH, SCALED_STAGE_HEIGHT = 450, 1920, 1080
SCALED_STAGE_FILES = {
    "pipeline/clean_plate/clean_plate_stage_manifest.json":
        "9ddb529e80b8b37ddbe0f678d142e90e9c077fe341510dc5c8209aaff58ccc59",
    "pipeline/clean_plate/current.json":
        "47282019dd89182815d1424771a44e8ae92647bfe80d3fac0851ab2098320092",
    "pipeline/clean_plate/removal_manifest.json":
        "70f78876785100d9b8f64ecf4885cc83cd926b34504104dde871399e573ac4a4",
    "pipeline/clean_plate/removal_plan.json":
        "e3696ad0e8bb2d8bea5bf8af46da011b7ff6e3ada58369e151421dc647e1aa30",
    "pipeline/clean_plate/task_masks/*/*/binding.json":
        "4d3dfdd6a8701c8e42d7f9ac4173443b743ad6acf8878da8dc94263956362f2d",
    "pipeline/clean_plate/task_masks/*/*/intent-0.json":
        "92bbd00100d87ade85f93985766be6ef3b2824aa61aeb4061e7b1206bf07a831",
    "pipeline/clean_plate/task_masks/*/*/intent-1.json":
        "92bbd00100d87ade85f93985766be6ef3b2824aa61aeb4061e7b1206bf07a831",
    "pipeline/clean_plate/task_masks/*/*/intent-2.json":
        "92bbd00100d87ade85f93985766be6ef3b2824aa61aeb4061e7b1206bf07a831",
    "pipeline/clean_plate/task_masks/*/*/intent-3.json":
        "92bbd00100d87ade85f93985766be6ef3b2824aa61aeb4061e7b1206bf07a831",
    "pipeline/clean_plate/task_masks/*/*/intent-4.json":
        "92bbd00100d87ade85f93985766be6ef3b2824aa61aeb4061e7b1206bf07a831",
    "pipeline/clean_plate/task_masks/*/*/intent-5.json":
        "92bbd00100d87ade85f93985766be6ef3b2824aa61aeb4061e7b1206bf07a831",
    "pipeline/clean_plate/task_masks/*/*/parsed-response-0.json":
        "7a0c97f47411b0ca759a979fade424a48c31fe31f4d0bd503bc3c1f147f586d3",
    "pipeline/clean_plate/task_masks/*/*/parsed-response-1.json":
        "083b60c53547cdcca338b92951913070ce6a35b583e1f2029363805ab72bd1fb",
    "pipeline/clean_plate/task_masks/*/*/parsed-response-2.json":
        "8b06d9270895d9f85ee7f3ee14688196c02f2e19dbcfcbf805dc037cda9a36bc",
    "pipeline/clean_plate/task_masks/*/*/parsed-response-3.json":
        "998e746ccec5a70d5207f4fdb090c8ea702ac3c6eeb659bd1eba1b77b512ea12",
    "pipeline/clean_plate/task_masks/*/*/parsed-response-4.json":
        "fc6915a43c1c362ad6d1fc036cbb3097f2ceef3ae494833a64565317f5c4e19e",
    "pipeline/clean_plate/task_masks/*/*/parsed-response-5.json":
        "bdf7bf7e1d5ac61549aee4d6ab623e0057e01a4da262c47e41704edd06ba5466",
    "pipeline/clean_plate/task_masks/*/*/response-0.json":
        "40033a2296b9e1018234be4b4ed31dd15abdbd3f4aff04e39056a3d355c0abc1",
    "pipeline/clean_plate/task_masks/*/*/response-1.json":
        "d7cc4aed93a44e3e8478ea74bee0b3069b88273d3dadc3404eab3c9287f7dd08",
    "pipeline/clean_plate/task_masks/*/*/response-2.json":
        "c13d29ef18886b99fc91d0f62f01370c0919acb066f1e5a368436bc30be56d9b",
    "pipeline/clean_plate/task_masks/*/*/response-3.json":
        "778d7d572f8c2c61ff8b620f956d7c30b5f2aafcd576e537fb6c3558e8bf0049",
    "pipeline/clean_plate/task_masks/*/*/response-4.json":
        "b6dfa55bde1e321f0bdf8a5a68e501bc8a221c14f3c870dee94a900d962daa60",
    "pipeline/clean_plate/task_masks/*/*/response-5.json":
        "3333d3f9c1fdcdd19241a88146bdbae4d4ee4b8af9760332e6788743612ab67d",
    "pipeline/clean_plate/task_masks/*/*/tracks.json":
        "f73cebfddb7757c11ca1505657514ca0275c3b64478311572470381f1dd9b2fa",
    "pipeline/clean_plate/task_masks/*/*/uploaded-video.json":
        "fab363944c5a161c8682a9857286ac78291384d3a7e88030388116addb9de7b2",
    "pipeline/clean_plate/task_masks/*/*/video-cleanup.json":
        "e81a7b932c1df445e880a937f5ac6cf9f97606d6cfe86e451829208d30e73a76",
    "pipeline/clean_plate/task_masks/*/*/video-upload-intent.json":
        "feb89a6db9a7ffe274df73a72f69fbe268851f9799f0b2f5a38313560a7a176a",
    "pipeline/clean_plate/task_masks/*/task_masks.object_removal.json":
        "2317c81534ca6f24ee19c14656d10721be918d52dd2729f72f266114e183dde3",
    "pipeline/website_task_target_choices/current.json":
        "132222e37cae22df94b775698c0ca5c90d09a0ba95e64f5b364d66e03f4159d1",
    "pipeline/website_task_target_choices/twin_crate.question.json":
        "1bbc49699b7bcd0a84b0a6e4e560315f94cd77712508f0bc0f9568998936937a",
}
#: Growth of peak resident memory while the clean-plate stage runs (after imports and synthetic
#: setup). The code before this change grew by about 316 MiB on this case (1.5 GiB on 60 s of 4K
#: with one such object, ``LARGE``); the compact path by about 34 MiB. The bound leaves room for other hosts
#: and allocators and still fails far below the old behaviour.
SCALED_STAGE_RSS_GROWTH_BOUND_BYTES = 128 << 20
SCALED_STAGE_ADDRESS_SPACE_CAP_MIB = 1536


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="peak RSS and the address-space cap are Linux")
def test_scaled_clean_plate_stage_stays_bounded_and_byte_identical(tmp_path):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPOSITORY / "src"), str(REPOSITORY)]),
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "META_MODEL_API_KEY": "synthetic-key", "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-m", "tests.meta_sam31_synthetic", str(tmp_path),
                             str(SCALED_STAGE_FRAMES), str(SCALED_STAGE_WIDTH), str(SCALED_STAGE_HEIGHT),
                             str(SCALED_STAGE_ADDRESS_SPACE_CAP_MIB), "stage", "large_ambiguous"],
                            cwd=REPOSITORY, env=env, capture_output=True, text=True, timeout=900, check=False)
    assert result.returncode == 0, result.stderr[-4000:]
    measured = json.loads(result.stdout.strip().splitlines()[-1])
    assert pinned_files(tmp_path) == SCALED_STAGE_FILES
    growth = measured["peak_rss_bytes"] - measured["baseline_peak_rss_bytes"]
    assert growth < SCALED_STAGE_RSS_GROWTH_BOUND_BYTES, measured
