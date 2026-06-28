"""Hermetic tests for the VLM visual sanity-QC gate (no google-genai, no network, no GPU).

The Gemini call is injected, so every path is exercised with canned model replies."""
from __future__ import annotations

import json

from blueprint_pipeline import render_visual_qc as qc

_FAUCET_TASK = "a humanoid robot at a kitchen sink about to turn on the faucet"

# the exact anomaly the user caught by eye: a city visible through a kitchen window
_CITY_REPLY = json.dumps({
    "coherent": False,
    "robot_visible": True,
    "background_consistent": False,
    "dark_region_fraction": 0.1,
    "overall_severity": "high",
    "anomalies": [
        {"category": "incongruous_background", "severity": "high",
         "description": "A city skyline is visible through the window; the scene should be an enclosed kitchen."}
    ],
    "summary": "Kitchen sink POV but an outdoor cityscape shows through the window.",
})

_CLEAN_REPLY = json.dumps({
    "coherent": True, "robot_visible": True, "background_consistent": True,
    "dark_region_fraction": 0.05, "overall_severity": "none", "anomalies": [],
    "summary": "Coherent kitchen sink POV.",
})

_DARK_REPLY = json.dumps({
    "coherent": True, "robot_visible": True, "background_consistent": True,
    "dark_region_fraction": 0.42, "overall_severity": "low", "anomalies": [],
    "summary": "Sink POV but a large lower region is very dark.",
})


def test_prompt_targets_the_human_obvious_anomalies() -> None:
    p = qc.build_qc_prompt(_FAUCET_TASK, scene_context="enclosed kitchen")
    assert _FAUCET_TASK in p
    assert "enclosed kitchen" in p
    assert "window" in p.lower() and "cityscape" in p.lower()  # the incongruous-background check
    assert "dark" in p.lower() and "robot" in p.lower()
    assert "STRICT JSON" in p and "anomalies" in p


def test_parse_normalizes_and_is_robust_to_junk() -> None:
    v = qc.parse_qc_verdict("here you go:\n" + _CITY_REPLY + "\nthanks")  # JSON embedded in prose
    assert v["parsed"] is True
    assert v["coherent"] is False and v["background_consistent"] is False
    assert v["anomalies"][0]["category"] == "incongruous_background"
    assert v["anomalies"][0]["severity"] == "high"
    # totally malformed -> parsed False (inconclusive, NOT silently 'clean')
    bad = qc.parse_qc_verdict("the model is thinking...")
    assert bad["parsed"] is False


def test_flagging_rules() -> None:
    assert qc.verdict_is_flagged(qc.parse_qc_verdict(_CITY_REPLY)) is True       # incoherent + high anomaly
    assert qc.verdict_is_flagged(qc.parse_qc_verdict(_CLEAN_REPLY)) is False      # clean passes
    assert qc.verdict_is_flagged(qc.parse_qc_verdict(_DARK_REPLY)) is True        # dark-region floor
    assert qc.verdict_is_flagged(qc.parse_qc_verdict("garbage")) is True          # unparsed -> flagged
    # a low-severity anomaly alone does not trip the medium floor
    low = json.dumps({"coherent": True, "robot_visible": True, "background_consistent": True,
                      "dark_region_fraction": 0.0, "overall_severity": "low",
                      "anomalies": [{"category": "other", "severity": "low", "description": "tiny speck"}]})
    assert qc.verdict_is_flagged(qc.parse_qc_verdict(low)) is False


def test_sample_frames_evenly_with_first_and_last() -> None:
    frames = [f"f{i}.png" for i in range(10)]
    s = qc.sample_frame_paths(frames, 3)
    assert s[0] == "f0.png" and s[-1] == "f9.png" and len(s) == 3
    assert qc.sample_frame_paths(frames, 0) == frames          # 0 -> all
    assert qc.sample_frame_paths(["a", "b"], 5) == ["a", "b"]   # fewer than sample_n -> all
    assert qc.sample_frame_paths(frames, 1) == ["f0.png"]


def test_review_frame_with_injected_generate_flags_city(tmp_path) -> None:
    img = tmp_path / "robot_pov_0000.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n fake pixels")
    seen = {}

    def fake_generate(image_bytes: bytes, prompt: str) -> str:
        seen["bytes"] = image_bytes
        seen["prompt"] = prompt
        return _CITY_REPLY

    v = qc.review_render_frame(img, _FAUCET_TASK, generate=fake_generate)
    assert seen["bytes"].startswith(b"\x89PNG")          # it actually read + sent the image bytes
    assert _FAUCET_TASK in seen["prompt"]
    assert v["frame"] == "robot_pov_0000.png"
    assert v["flagged"] is True and v["error"] is None


def test_review_frame_accepts_raw_bytes_and_handles_generate_errors() -> None:
    ok = qc.review_render_frame(b"rawbytes", _FAUCET_TASK, generate=lambda b, p: _CLEAN_REPLY,
                                frame_label="frame_x")
    assert ok["frame"] == "frame_x" and ok["flagged"] is False

    def boom(_b, _p):
        raise RuntimeError("gemini exploded")

    err = qc.review_render_frame(b"rawbytes", _FAUCET_TASK, generate=boom)
    assert err["flagged"] is True and "gemini exploded" in err["error"]   # failed review -> flagged


def test_qc_render_frames_aggregates_and_would_have_caught_the_city(tmp_path) -> None:
    # 3 frames: clean, clean, city-anomaly
    paths = []
    for i in range(3):
        p = tmp_path / f"robot_pov_000{i}.png"
        p.write_bytes(b"\x89PNG fake")
        paths.append(p)
    replies = {paths[0].name: _CLEAN_REPLY, paths[1].name: _CLEAN_REPLY, paths[2].name: _CITY_REPLY}

    def fake_generate_by_unused(_b, _p):  # all frames sampled (sample_n>=3) -> need per-frame replies
        return _CLEAN_REPLY

    # route reply by frame: use review per-frame via a closure keyed on call order
    order = iter([_CLEAN_REPLY, _CLEAN_REPLY, _CITY_REPLY])
    report = qc.qc_render_frames(paths, _FAUCET_TASK, sample_n=3,
                                 generate=lambda b, p: next(order))
    d = report.to_dict()
    assert d["frames_reviewed"] == 3
    assert d["flagged"] is True                       # the city frame trips the gate
    assert d["worst_severity"] == "high"
    assert any(a["category"] == "incongruous_background" for a in d["anomalies"])
    assert d["schema_version"] == "render_visual_qc.v1"


def test_qc_output_dir_finds_robot_pov_frames(tmp_path) -> None:
    fdir = tmp_path / "render_output" / "at_sink_turn_on_faucet" / "frames"
    fdir.mkdir(parents=True)
    for i in range(2):
        (fdir / f"robot_pov_000{i}.png").write_bytes(b"\x89PNG fake")
    (fdir / "overview_0000.png").write_bytes(b"\x89PNG fake")  # not robot_pov -> ignored by default glob
    report = qc.qc_render_output_dir(tmp_path / "render_output", _FAUCET_TASK, sample_n=5,
                                     generate=lambda b, p: _CLEAN_REPLY)
    assert report.frames_reviewed == 2 and report.flagged is False


def test_extract_model_text_skips_thinking_parts() -> None:
    class P:
        def __init__(self, text, thought=False):
            self.text = text
            self.thought = thought

    class C:
        def __init__(self, parts):
            self.content = type("X", (), {"parts": parts})

    class R:
        text = ""
        def __init__(self, parts):
            self.candidates = [C(parts)]

    r = R([P("internal reasoning...", thought=True), P(_CLEAN_REPLY)])
    assert qc.extract_model_text(r) == _CLEAN_REPLY  # the thinking part is skipped
