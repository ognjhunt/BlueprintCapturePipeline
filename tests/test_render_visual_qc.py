"""Hermetic tests for the VLM visual sanity-QC gate (no google-genai, no network, no GPU).

The Gemini call is injected, so every path is exercised with canned model replies."""
from __future__ import annotations

import json
import sys
import types

import pytest

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

_PLACEMENT_PASS_REPLY = json.dumps({
    "pass": True,
    "robot_on_open_floor": True,
    "facing_target": True,
    "not_clipping_counter_cabinets_sink": True,
    "reason": "Robot is on open floor in front of the sink and facing it.",
})

_PLACEMENT_FAIL_REPLY = json.dumps({
    "pass": False,
    "robot_on_open_floor": False,
    "facing_target": True,
    "not_clipping_counter_cabinets_sink": False,
    "reason": "Robot legs appear merged into the sink cabinet.",
})

_MANIPULATION_POV_PASS_REPLY = json.dumps({
    "pass": True,
    "target_visible": True,
    "gripper_or_hand_visible": True,
    "robot_arm_visible_beyond_gripper": True,
    "arm_reaching_target": True,
    "not_mostly_dark_or_occluded": True,
    "reason": "The robot forearm and gripper are visible reaching the refrigerator handle.",
})

_MANIPULATION_POV_FAIL_REPLY = json.dumps({
    "pass": False,
    "target_visible": True,
    "gripper_or_hand_visible": True,
    "robot_arm_visible_beyond_gripper": False,
    "arm_reaching_target": True,
    "not_mostly_dark_or_occluded": True,
    "reason": "The frame shows only an isolated gripper with no forearm or arm context.",
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


def test_parse_qc_verdict_missing_safety_booleans_fails_closed() -> None:
    missing = qc.parse_qc_verdict(json.dumps({"summary": "looks ok"}))

    assert missing["parsed"] is True
    assert missing["coherent"] is None
    assert missing["robot_visible"] is None
    assert missing["background_consistent"] is None
    assert qc.verdict_is_flagged(missing) is True

    clean = qc.parse_qc_verdict(_CLEAN_REPLY)
    assert clean["coherent"] is True
    assert clean["robot_visible"] is True
    assert clean["background_consistent"] is True
    assert qc.verdict_is_flagged(clean) is False


def test_unknown_nonempty_severity_clamps_high() -> None:
    reply = json.dumps({
        "coherent": True,
        "robot_visible": True,
        "background_consistent": True,
        "dark_region_fraction": 0.0,
        "overall_severity": "critical",
        "anomalies": [{"category": "other", "severity": "severe", "description": "bad"}],
    })

    verdict = qc.parse_qc_verdict(reply)

    assert verdict["overall_severity"] == "high"
    assert verdict["anomalies"][0]["severity"] == "high"
    assert qc.verdict_is_flagged(verdict) is True


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


def test_parse_qc_verdict_flags_missing_critical_booleans() -> None:
    verdict = qc.parse_qc_verdict(json.dumps({"summary": "partial model reply"}))

    assert verdict["parsed"] is True
    assert verdict["coherent"] is None
    assert verdict["robot_visible"] is None
    assert verdict["background_consistent"] is None
    assert qc.verdict_is_flagged(verdict) is True

    clean = qc.parse_qc_verdict(json.dumps({
        "coherent": True,
        "robot_visible": True,
        "background_consistent": True,
        "dark_region_fraction": 0.0,
        "overall_severity": "none",
        "anomalies": [],
    }))
    assert qc.verdict_is_flagged(clean) is False


def test_unknown_nonempty_severity_clamps_high_and_flags() -> None:
    verdict = qc.parse_qc_verdict(json.dumps({
        "coherent": True,
        "robot_visible": True,
        "background_consistent": True,
        "overall_severity": "critical",
        "anomalies": [],
    }))

    assert verdict["overall_severity"] == "high"
    assert qc.verdict_is_flagged(verdict) is True

    anomaly = qc.parse_qc_verdict(json.dumps({
        "coherent": True,
        "robot_visible": True,
        "background_consistent": True,
        "overall_severity": "none",
        "anomalies": [{"category": "other", "severity": "critical", "description": "bad"}],
    }))
    assert anomaly["anomalies"][0]["severity"] == "high"
    assert qc.verdict_is_flagged(anomaly) is True


def test_sample_frames_evenly_with_first_and_last() -> None:
    frames = [f"f{i}.png" for i in range(10)]
    s = qc.sample_frame_paths(frames, 3)
    assert s[0] == "f0.png" and s[-1] == "f9.png" and len(s) == 3
    assert qc.sample_frame_paths(frames, 0) == frames          # 0 -> all
    assert qc.sample_frame_paths(["a", "b"], 5) == ["a", "b"]   # fewer than sample_n -> all
    assert qc.sample_frame_paths(frames, 1) == ["f0.png"]


@pytest.mark.parametrize("n", [2, 3, 5, 7])
@pytest.mark.parametrize("sample_n", [2, 3, 4])
def test_sample_frames_preserves_first_last_and_dedups(n: int, sample_n: int) -> None:
    frames = [f"f{i}.png" for i in range(n)]
    sampled = qc.sample_frame_paths(frames, sample_n)

    if n <= sample_n:
        assert sampled == frames
        return
    assert sampled[0] == frames[0]
    assert sampled[-1] == frames[-1]
    assert len(sampled) <= sample_n
    assert len(sampled) == len(set(sampled))
    if sample_n == 2:
        assert sampled == [frames[0], frames[-1]]


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


def test_gemini_review_image_ignores_non_flash_override(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class FakePart:
        @staticmethod
        def from_bytes(*, data, mime_type):
            assert data == b"image"
            assert mime_type == "image/png"
            return {"data": data, "mime_type": mime_type}

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            captured["model"] = model
            assert contents[0]["data"] == b"image"
            assert config["response_mime_type"] == "application/json"
            return types.SimpleNamespace(text=_CLEAN_REPLY)

    class FakeClient:
        def __init__(self, *, api_key):
            assert api_key == "google-key"
            self.models = FakeModels()

    google_module = types.ModuleType("google")
    google_module.genai = types.SimpleNamespace(
        Client=FakeClient,
        types=types.SimpleNamespace(Part=FakePart),
    )
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setenv("GOOGLE_GENAI_API_KEY", "google-key")
    monkeypatch.setenv("BLUEPRINT_RENDER_QC_GEMINI_MODEL", "gemini-2.5-pro")

    assert qc._gemini_review_image(b"image", "prompt") == _CLEAN_REPLY
    assert captured["model"] == "gemini-3-flash-preview"


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


def test_extract_model_text_all_thinking_parts_flags_unparsed() -> None:
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

    raw = qc.extract_model_text(R([P("internal reasoning only", thought=True)]))
    verdict = qc.parse_qc_verdict(raw)

    assert raw == ""
    assert verdict["parsed"] is False
    assert qc.verdict_is_flagged(verdict) is True


def test_extract_model_text_uses_candidate_when_response_text_blank() -> None:
    class P:
        text = _CLEAN_REPLY
        thought = False

    class C:
        content = type("X", (), {"parts": [P()]})

    response = type("R", (), {"text": "   ", "candidates": [C()]})()

    assert qc.extract_model_text(response) == _CLEAN_REPLY


def test_extract_json_object_ignores_reasoning_braces_before_qc_payload() -> None:
    raw = 'thinking {not json} more notes {"coherent": true, "robot_visible": true, '\
        '"background_consistent": true, "dark_region_fraction": 0, '\
        '"overall_severity": "none", "anomalies": []}'

    verdict = qc.parse_qc_verdict(raw)

    assert verdict["parsed"] is True
    assert verdict["coherent"] is True
    assert verdict["overall_severity"] == "none"


def test_robot_placement_prompt_asks_the_exact_gate_question() -> None:
    prompt = qc.build_robot_placement_qc_prompt(
        "sink",
        task_description="turn on the faucet",
    )
    assert "open floor in front of the sink" in prompt
    assert "NOT inside/clipping the counter/cabinets/sink" in prompt
    assert '"pass"' in prompt and "STRICT JSON" in prompt


def test_manipulation_pov_prompt_asks_for_arm_and_affordance() -> None:
    prompt = qc.build_manipulation_pov_qc_prompt(
        "refrigerator",
        task_description="open the refrigerator",
    )
    assert "refrigerator" in prompt
    assert "gripper/hand AND a visible forearm" in prompt
    assert "handle/affordance" in prompt
    assert "too dark or occluded" in prompt


def test_parse_robot_placement_verdict_fails_closed() -> None:
    ok = qc.parse_robot_placement_verdict(_PLACEMENT_PASS_REPLY)
    assert ok["parsed"] is True
    assert ok["passed"] is True
    assert ok["not_clipping_counter_cabinets_sink"] is True

    bad = qc.parse_robot_placement_verdict(_PLACEMENT_FAIL_REPLY)
    assert bad["parsed"] is True
    assert bad["passed"] is False
    assert "cabinet" in bad["reason"]

    junk = qc.parse_robot_placement_verdict("not json")
    assert junk["parsed"] is False
    assert junk["passed"] is False


def test_parse_manipulation_pov_verdict_fails_closed() -> None:
    ok = qc.parse_manipulation_pov_verdict(_MANIPULATION_POV_PASS_REPLY)
    assert ok["parsed"] is True
    assert ok["passed"] is True
    assert ok["gripper_or_hand_visible"] is True
    assert ok["robot_arm_visible_beyond_gripper"] is True

    bad = qc.parse_manipulation_pov_verdict(_MANIPULATION_POV_FAIL_REPLY)
    assert bad["parsed"] is True
    assert bad["passed"] is False
    assert bad["target_visible"] is True
    assert bad["gripper_or_hand_visible"] is True
    assert bad["robot_arm_visible_beyond_gripper"] is False

    junk = qc.parse_manipulation_pov_verdict("not json")
    assert junk["parsed"] is False
    assert junk["passed"] is False


def test_qc_robot_placement_frames_blocks_on_any_failed_frame(tmp_path) -> None:
    paths = []
    for name in ("verify_0000.png", "robot_pov_0000.png"):
        p = tmp_path / name
        p.write_bytes(b"\x89PNG fake")
        paths.append(p)
    replies = iter([_PLACEMENT_PASS_REPLY, _PLACEMENT_FAIL_REPLY])

    report = qc.qc_robot_placement_frames(
        paths,
        "sink",
        task_description="turn on the faucet",
        generate=lambda _b, _p: next(replies),
    )

    assert report["schema_version"] == "robot_placement_visual_qc.v1"
    assert report["status"] == "blocked"
    assert report["frames_reviewed"] == 2
    assert "placement_visual_qc_failed" in report["blockers"]


def test_qc_robot_placement_frames_passes_when_all_frames_pass(tmp_path) -> None:
    p = tmp_path / "verify_0000.png"
    p.write_bytes(b"\x89PNG fake")

    report = qc.qc_robot_placement_frames(
        [p],
        "sink",
        task_description="turn on the faucet",
        generate=lambda _b, _p: _PLACEMENT_PASS_REPLY,
    )

    assert report["status"] == "passed"
    assert report["blockers"] == []


def test_qc_manipulation_pov_frames_blocks_without_visible_arm(tmp_path) -> None:
    paths = []
    for name in ("robot_pov_0000.png", "robot_pov_0001.png"):
        p = tmp_path / name
        p.write_bytes(b"\x89PNG fake")
        paths.append(p)
    replies = iter([_MANIPULATION_POV_PASS_REPLY, _MANIPULATION_POV_FAIL_REPLY])

    report = qc.qc_manipulation_pov_frames(
        paths,
        "refrigerator",
        task_description="open the refrigerator",
        generate=lambda _b, _p: next(replies),
    )

    assert report["schema_version"] == "manipulation_pov_visual_qc.v1"
    assert report["status"] == "blocked"
    assert "manipulation_pov_visual_qc_failed" in report["blockers"]


def test_qc_manipulation_pov_frames_passes_when_arm_reaches_target(tmp_path) -> None:
    p = tmp_path / "robot_pov_0000.png"
    p.write_bytes(b"\x89PNG fake")

    report = qc.qc_manipulation_pov_frames(
        [p],
        "refrigerator",
        task_description="open the refrigerator",
        generate=lambda _b, _p: _MANIPULATION_POV_PASS_REPLY,
    )

    assert report["status"] == "passed"
    assert report["blockers"] == []
