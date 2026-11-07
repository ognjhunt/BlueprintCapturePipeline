"""``write_json`` streams its text, and every byte it writes is what ``json.dumps`` wrote before.

``write_json`` now feeds the encoder ``json.dumps(payload, indent=2)`` uses straight to the file
in batches instead of joining all of its pieces first; for millions of track runs those pieces
were several times the payload. The digests below were computed with the pre-change
implementation (``json.dumps(dict(payload), indent=2)`` and its two redactions). The SAM track
codec's own writer and digest must give the same bytes and the same ``canonical_digest``.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.core import common as core_common
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_track_mask_codec import (RunLengths, canonical_track_digest, load_track_json,
                                                         write_track_json)


def _payloads():
    return {
        "nested_dicts": {"a": {"b": {"c": [1, 2, {"d": None, "e": {}}], "f": []}}, "z": {"y": {"x": {"w": "v"}}}},
        "floats": {"f": [0.1, -0.0, 1e300, 1e-300, 3.141592653589793, 2.5, 1 / 3, float("nan"), float("inf"),
                         -float("inf")]},
        "unicode": {"ünïcödé": "☃ snow   \x00 \t \"q\" \\ é 中文 😀", "keys/é": ["Ωmega", "퟿"]},
        "lists_of_dicts": {"observations": [{"source_frame_id": f"decoded-{i:09d}", "width": 3840, "height": 2160,
                                             "runs": [{"start": i * 3840 + j * 7, "length": 1100 + j}
                                                      for j in range(5)]} for i in range(20)]},
        "big_integers": {"big": [2 ** 63, -(2 ** 63) - 1, 2 ** 200, 10 ** 30, 0, -1], "flags": [True, False, None]},
        "redaction": {"token": "<redacted:secret>", "field": "x<redacted:secret-field>y",
                      "<redacted:secret>": ["<redacted:secret-field>", "<redacted:secret"]},
        "beyond_one_batch": {"many": ["<redacted:secret>" + "x" * 100 + str(i) for i in range(20000)]},
        # Keys of mixed types: written by write_json, refused by canonical_digest (unsortable).
        "mixed_keys_and_tuples": {1: "one", 2.5: "two and a half", None: "none", False: "false",
                                  "tuple": (1, "a", (2, [3]))},
    }


# name: (sha256 of the written file, canonical_digest) from the pre-change implementation.
PINNED = {
    "nested_dicts": ("fe79cae8092499dc3303a504b0c6575d506e73a11bf410d6b0f3dd11a2d425e8",
                     "sha256:1b5165cb204703a70c0784e81d97a802319d4058a2794966144899b07c6b0a23"),
    "floats": ("61573f511477849ec3a6d422af2bd20b4fb2c52aa0f339689ae3c149f366e72b",
               "sha256:0b2b2c18d333211b05ddf30a38ee5ed51511911d26948f233652d348707bda40"),
    "unicode": ("c50db0d9cf64721703abe6e8489a112b32c8fc397bbc3f0f4477e0ba609aa7a9",
                "sha256:20306d3778be6668dec122c3fe0335758d04079eb0333ed3fa07e0c7c0032941"),
    "lists_of_dicts": ("73b0c03ef64efef182c5fda5539ae936ba48fe0268f03d26a5e948b3d2b4f30b",
                       "sha256:eb9291c32b1072e6f4d0ff81d99cccd2781c8353b724fb0f61303f4eb3c12fb8"),
    "big_integers": ("42bcd6d2dbb80a94ef3b869e7507820585b75c75292826c4ef731784da897e2b",
                     "sha256:c02aaa4a51ae674a8f712f7475592644400cb12cd6b471429be47e044a533141"),
    "redaction": ("63522a00e3e27b399e9a8d55134020597f521cf7b79dc714df51f7f5f78e1a3c",
                  "sha256:75efef2e62f7d8c034ac8a5989c5883cd539be69a64086a83516096176a8cfc7"),
    "beyond_one_batch": ("6fd51e687dce5b300504220a97efebc81c3a06ef014ffedf2d366f47540b7947",
                         "sha256:02dab1b3c6d7d63d3acaf2bb1736fc56a64bbde560a27d699e8ab277e3fc09fd"),
    "mixed_keys_and_tuples": ("84e7a3cd009e8b85bb76268af6bea067eefc89a9732f024539261b128fcb4ccd", None),
}


def _compact(value):
    """The same payload with every exact run list held as ``RunLengths``."""
    if type(value) is list and value and RunLengths.from_runs(value) is not None:
        return RunLengths.from_runs(value)
    if type(value) is dict:
        return {key: _compact(item) for key, item in value.items()}
    if type(value) is list:
        return [_compact(item) for item in value]
    return value


@pytest.mark.parametrize("batch", [core_common._JSON_WRITE_BATCH, 1, 64])
@pytest.mark.parametrize("name", sorted(PINNED))
def test_write_json_and_canonical_digest_keep_their_pinned_bytes(tmp_path, monkeypatch, name, batch):
    monkeypatch.setattr(core_common, "_JSON_WRITE_BATCH", batch)
    payload = _payloads()[name]
    written, digest = PINNED[name]
    write_json(tmp_path / "plain.json", payload)
    assert hashlib.sha256((tmp_path / "plain.json").read_bytes()).hexdigest() == written
    for form in (payload, _compact(payload)):
        write_track_json(tmp_path / "track.json", form)
        assert hashlib.sha256((tmp_path / "track.json").read_bytes()).hexdigest() == written
        if digest is None:
            with pytest.raises(TypeError):
                canonical_digest(payload)
            with pytest.raises(TypeError):
                canonical_track_digest(form)
        else:
            assert canonical_digest(payload) == digest == canonical_track_digest(form)
    if name == "lists_of_dicts":
        loaded = load_track_json(tmp_path / "track.json")
        assert type(loaded["observations"][0]["runs"]) is RunLengths and loaded == json.loads(
            (tmp_path / "plain.json").read_text())


def test_a_refused_payload_leaves_the_filesystem_as_json_dumps_did(tmp_path):
    (tmp_path / "kept").mkdir()
    (tmp_path / "kept" / "x.json").write_text("previous")
    for target in (tmp_path / "kept" / "x.json", tmp_path / "new" / "deeper" / "x.json"):
        with pytest.raises(TypeError, match="Object of type set is not JSON serializable"):
            write_json(target, {"first": "x" * 100, "bad": {1, 2}})
    assert sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")) == ["kept", "kept/x.json"]
    assert (tmp_path / "kept" / "x.json").read_text() == "previous"
