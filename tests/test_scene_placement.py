"""Hermetic tests for the scene_placement package.

NO GPU, NO VLM, NO network, NO isaacsim/torch/google-genai. The VLM call is
injected as a fake ``generate``; the shared types are pure stdlib. These tests pin
the foundation (``types``) and the target resolver (``target_resolver``): the VLM
path with a fake model, the pure label fallback + synonyms, and the degrade-to-
label behavior when the VLM returns junk, an unknown id, or raises.
"""
from __future__ import annotations

import math

import pytest

from blueprint_pipeline.scene_placement import target_resolver as tr
from blueprint_pipeline.scene_placement.types import SceneObject, StandPose


# ----------------------------- fixtures / helpers -----------------------------

def _obj(id_, label, *, cx=0.0, cy=0.0, cz=0.0, size=1.0, category="") -> SceneObject:
    """A unit-ish AABB centered at (cx, cy, cz) with a given label."""
    h = size / 2.0
    return SceneObject(
        id=id_,
        label=label,
        bbox_min=(cx - h, cy - h, cz - h),
        bbox_max=(cx + h, cy + h, cz + h),
        centroid=(cx, cy, cz),
        category=category,
    )


@pytest.fixture
def kitchen_objects():
    return [
        _obj("faucet_1", "kitchen_faucet_01", cx=1.0, cy=2.0, cz=0.9),
        _obj("sink_1", "sink", cx=1.0, cy=2.0, cz=0.85),
        _obj("stove_1", "stove", cx=3.0, cy=2.0, cz=0.9),
        _obj("fridge_1", "refrigerator", cx=5.0, cy=0.0, cz=1.0, size=2.0),
    ]


# ----------------------------- types helpers -----------------------------

def test_sceneobject_helpers():
    o = SceneObject(
        id="x", label="counter",
        bbox_min=(0.0, 0.0, 0.0), bbox_max=(2.0, 4.0, 1.0),
        centroid=(1.0, 2.0, 0.5),
    )
    assert o.size() == (2.0, 4.0, 1.0)
    assert o.footprint_center() == (1.0, 2.0)
    assert o.min_z() == 0.0
    assert o.max_z() == 1.0
    # defaults from the shared contract
    assert o.category == ""
    assert o.source == ""
    assert o.confidence == 1.0
    assert o.extra == {}


def test_standpose_shape():
    p = StandPose(
        position=(1.0, 1.5, 0.79), yaw=0.0, target_id="faucet_1",
        clear=True, standoff_m=0.55, notes="ok",
    )
    assert p.position == (1.0, 1.5, 0.79)
    assert p.target_id == "faucet_1"
    assert p.clear is True
    assert p.standoff_m == 0.55


# ----------------------------- prompt -----------------------------

def test_build_target_prompt_lists_objects_and_task(kitchen_objects):
    prompt = tr.build_target_prompt("turn on the faucet", kitchen_objects)
    assert "turn on the faucet" in prompt
    # every object id + label is surfaced for the model to choose from
    for obj in kitchen_objects:
        assert obj.id in prompt
        assert obj.label in prompt
    # strict-JSON contract is spelled out
    assert "target_id" in prompt


def test_build_target_prompt_handles_empty_objects():
    prompt = tr.build_target_prompt("do something", [])
    assert "do something" in prompt
    assert "target_id" in prompt


# ----------------------------- resolve_target (VLM path, fake generate) -----------------------------

def test_resolve_target_uses_fake_vlm(kitchen_objects):
    captured = {}

    def fake_generate(prompt: str) -> str:
        captured["prompt"] = prompt
        return '{"target_id": "faucet_1"}'

    target = tr.resolve_target("turn on the faucet", kitchen_objects, generate=fake_generate)
    assert target is not None
    assert target.id == "faucet_1"
    # the prompt the fake saw actually contained the task + objects
    assert "turn on the faucet" in captured["prompt"]


def test_resolve_target_parses_messy_vlm_reply(kitchen_objects):
    # model wraps JSON in prose / code fence — robust parse must still find the id
    messy = 'Sure! Here is the answer:\n```json\n{"target_id": "stove_1"}\n```\nDone.'
    target = tr.resolve_target("turn on the stove", kitchen_objects, generate=lambda p: messy)
    assert target is not None
    assert target.id == "stove_1"


def test_resolve_target_empty_objects_returns_none():
    called = {"n": 0}

    def fake_generate(prompt: str) -> str:
        called["n"] += 1
        return '{"target_id": "anything"}'

    assert tr.resolve_target("turn on the faucet", [], generate=fake_generate) is None
    # no point asking the VLM when there is nothing to choose from
    assert called["n"] == 0


def test_resolve_target_unknown_id_falls_back_to_label(kitchen_objects):
    # VLM hallucinates an id not in the scene -> must not return it; fall back to label
    target = tr.resolve_target(
        "turn on the faucet", kitchen_objects, generate=lambda p: '{"target_id": "ghost_99"}'
    )
    assert target is not None
    assert target.id == "faucet_1"  # label fallback finds the faucet


def test_resolve_target_null_id_falls_back_to_label(kitchen_objects):
    target = tr.resolve_target(
        "use the stove", kitchen_objects, generate=lambda p: '{"target_id": null}'
    )
    assert target is not None
    assert target.id == "stove_1"


def test_resolve_target_junk_reply_falls_back_to_label(kitchen_objects):
    target = tr.resolve_target(
        "turn on the faucet", kitchen_objects, generate=lambda p: "not json at all"
    )
    assert target is not None
    assert target.id == "faucet_1"


def test_resolve_target_generate_raises_falls_back_to_label(kitchen_objects):
    def boom(prompt: str) -> str:
        raise RuntimeError("vlm down")

    target = tr.resolve_target("turn on the faucet", kitchen_objects, generate=boom)
    assert target is not None
    assert target.id == "faucet_1"


def test_resolve_target_no_match_anywhere_returns_none():
    objs = [_obj("a", "lamp"), _obj("b", "rug")]
    # VLM returns nothing usable AND the label fallback finds no faucet-like object
    target = tr.resolve_target("turn on the faucet", objs, generate=lambda p: '{"target_id": ""}')
    assert target is None


# ----------------------------- resolve_target_by_label (pure fallback) -----------------------------

def test_label_fallback_direct_substring(kitchen_objects):
    target = tr.resolve_target_by_label("turn on the faucet", kitchen_objects)
    assert target is not None
    assert target.id == "faucet_1"  # "faucet" is a substring of "kitchen_faucet_01"


def test_label_fallback_ignores_room_qualifier_word():
    # Regression: a real GPU render resolved "Stand at the kitchen sink and turn on the
    # faucet" to a "kitchen_box" prim because "kitchen" (longest token) was tried first and
    # direct-matched. Room words must be stopped so the actual target noun wins.
    objs = [_obj("box", "kitchen_box"), _obj("sink", "sink"), _obj("faucet", "faucet")]
    target = tr.resolve_target_by_label("Stand at the kitchen sink and turn on the faucet.", objs)
    assert target is not None
    assert target.id in {"faucet", "sink"}        # the action target, NOT the kitchen_box
    # even with only a sink present (no clean "faucet" prim), it must not pick kitchen_box
    objs2 = [_obj("box", "kitchen_box"), _obj("sink", "sink")]
    assert tr.resolve_target_by_label("turn on the kitchen faucet", objs2).id == "sink"


def test_label_fallback_synonym_tap_to_faucet():
    objs = [_obj("f", "faucet"), _obj("s", "stove")]
    # "tap" is a synonym mapped onto faucet-like labels
    target = tr.resolve_target_by_label("turn the tap", objs)
    assert target is not None
    assert target.id == "f"


def test_label_fallback_synonym_fridge_to_refrigerator():
    objs = [_obj("r", "refrigerator"), _obj("c", "cabinet")]
    target = tr.resolve_target_by_label("open the fridge", objs)
    assert target is not None
    assert target.id == "r"


def test_label_fallback_prefers_shortest_label():
    # two faucet-like objects: the most direct name wins the tie
    objs = [
        _obj("long", "kitchen_faucet_handle_left"),
        _obj("short", "faucet"),
    ]
    target = tr.resolve_target_by_label("turn on the faucet", objs)
    assert target is not None
    assert target.id == "short"


def test_label_fallback_stopwords_do_not_match():
    # "on"/"the"/"turn" are stopwords; only the noun should drive the match
    objs = [_obj("o", "oven"), _obj("t", "table")]
    target = tr.resolve_target_by_label("turn on the oven", objs)
    assert target is not None
    assert target.id == "o"


def test_label_fallback_resolves_switch_toggle_pull_object_nouns():
    # Regression: "switch"/"toggle"/"pull" were BOTH stopwords and synonym-group
    # members, so they were stripped before they could be the intent noun and the
    # task resolved to None. They are object nouns here and must resolve.
    assert tr.resolve_target_by_label("flip the switch", [_obj("s", "switch")]).id == "s"
    assert tr.resolve_target_by_label("toggle the toggle", [_obj("t", "toggle")]).id == "t"
    assert tr.resolve_target_by_label("pull the drawer pull", [_obj("p", "pull")]).id == "p"
    # synonym still works: "switch" task should reach a "lightswitch" label
    assert tr.resolve_target_by_label("hit the switch", [_obj("l", "lightswitch")]).id == "l"


def test_label_fallback_no_match_returns_none():
    objs = [_obj("a", "lamp"), _obj("b", "rug")]
    assert tr.resolve_target_by_label("turn on the faucet", objs) is None


def test_label_fallback_empty_objects_returns_none():
    assert tr.resolve_target_by_label("turn on the faucet", []) is None


# ----------------------------- json/text extraction internals -----------------------------

def test_extract_json_object_plain_and_embedded():
    assert tr._extract_json_object('{"target_id": "x"}') == {"target_id": "x"}
    assert tr._extract_json_object('junk {"target_id": "y"} more') == {"target_id": "y"}
    assert tr._extract_json_object("no braces here") == {}


def test_extract_response_text_skips_thinking_parts():
    class _Part:
        def __init__(self, text, thought=False):
            self.text = text
            self.thought = thought

    class _Content:
        def __init__(self, parts):
            self.parts = parts

    class _Candidate:
        def __init__(self, parts):
            self.content = _Content(parts)

    class _Resp:
        text = ""  # force the candidate/parts path

        def __init__(self, parts):
            self.candidates = [_Candidate(parts)]

    resp = _Resp([_Part("reasoning...", thought=True), _Part('{"target_id": "z"}')])
    assert tr._extract_response_text(resp) == '{"target_id": "z"}'


# ----------------------------- default gemini call (no network) -----------------------------

def test_default_gemini_requires_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="missing_GOOGLE_GENAI_API_KEY"):
        tr._gemini_resolve_text("any prompt")


def test_resolve_target_default_generate_degrades_when_no_key(monkeypatch, kitchen_objects):
    # With no key, the default Gemini call raises -> resolve_target must degrade to
    # the label fallback rather than crash (exercises the default-generate wiring).
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    target = tr.resolve_target("turn on the faucet", kitchen_objects)
    assert target is not None
    assert target.id == "faucet_1"


# =========================================================================== #
# usd_index — PURE logic only (synthetic bounds, NO pxr / NO USD stage / NO GPU)
#
# The pxr walk in UsdSceneSpatialIndex._walk_stage is the only non-pure code and
# is NOT exercised here (it needs USD); every decision it feeds — exclude the
# shell, clean the label, build the SceneObject — is tested through the pure
# helpers below, plus a monkeypatched walk to prove objects() just glues them.
# =========================================================================== #

from blueprint_pipeline.scene_placement import types as sp_types  # noqa: E402
from blueprint_pipeline.scene_placement import usd_index  # noqa: E402
from blueprint_pipeline.scene_placement.usd_index import (  # noqa: E402
    UsdSceneSpatialIndex,
    _clean_label,
    _is_excluded,
    _objects_from_bounds,
)


def _named_bounds(name, bmin, bmax):
    return (name, (bmin, bmax))


# ----------------------------- _clean_label -----------------------------

def test_clean_label_strips_instance_indices():
    assert _clean_label("Faucet_01") == "faucet"
    assert _clean_label("Sink001") == "sink"
    assert _clean_label("Stove2") == "stove"
    assert _clean_label("Cabinet_12") == "cabinet"


def test_clean_label_strips_structural_suffixes():
    assert _clean_label("Stove_link") == "stove"
    assert _clean_label("Faucet_geo") == "faucet"
    assert _clean_label("Sink_mesh") == "sink"
    assert _clean_label("Island_xform") == "island"


def test_clean_label_strips_stacked_decorations():
    # iterative stripping reduces an index + a structural suffix together
    assert _clean_label("Faucet_01_geo") == "faucet"
    assert _clean_label("Sink_link_001") == "sink"


def test_clean_label_normalizes_separators_and_case():
    assert _clean_label("Kitchen-Island") == "kitchen_island"
    assert _clean_label("Coffee.Maker") == "coffee_maker"
    assert _clean_label("UPPER") == "upper"


def test_clean_label_handles_blank_and_pure_index():
    assert _clean_label("") == ""
    assert _clean_label("   ") == ""
    # a name that is ONLY an index has nothing left after stripping
    assert _clean_label("01") == ""
    assert _clean_label("_001") == ""


# ----------------------------- _is_excluded -----------------------------

def test_is_excluded_matches_shell_substrings():
    subs = usd_index.DEFAULT_EXCLUDE_SUBSTRINGS
    assert _is_excluded("EastWall", subs)
    assert _is_excluded("kitchen_floor", subs)
    assert _is_excluded("Ceiling_01", subs)
    assert _is_excluded("GroundPlane", subs)
    assert _is_excluded("KeyLight", subs)
    assert _is_excluded("OverviewCamera", subs)
    assert _is_excluded("DomeLight", subs)
    assert _is_excluded("Room_Envelope", subs)


def test_is_excluded_keeps_real_objects():
    subs = usd_index.DEFAULT_EXCLUDE_SUBSTRINGS
    assert not _is_excluded("Faucet", subs)
    assert not _is_excluded("Sink", subs)
    assert not _is_excluded("Stove", subs)
    assert not _is_excluded("Refrigerator", subs)


def test_is_excluded_keeps_objects_with_buried_shell_substrings():
    # Regression: a raw substring test wrongly dropped real manipulable objects whose
    # names merely CONTAIN a shell word inside a token ("room" inside "mushroom",
    # "broom", "bedroom"). Token-boundary matching must keep these.
    subs = usd_index.DEFAULT_EXCLUDE_SUBSTRINGS
    assert not _is_excluded("Mushroom", subs)
    assert not _is_excluded("Broom", subs)
    assert not _is_excluded("Bedroom_Door", subs)
    assert not _is_excluded("RoombaVacuum", subs)


def test_is_excluded_matches_shell_only_on_whole_token():
    # The shell word must appear as a WHOLE token, not buried in a longer word.
    subs = usd_index.DEFAULT_EXCLUDE_SUBSTRINGS
    # whole-token shell -> excluded
    assert _is_excluded("Wall_North", subs)
    assert _is_excluded("EastWall", subs)  # camelCase boundary splits east|wall
    assert _is_excluded("kitchen_floor", subs)
    # buried (sub-token) -> kept. "wall_clock"/"floor_lamp" still carry a literal
    # shell TOKEN, so they remain excluded by design; the camelCase forms below do
    # NOT (the shell letters are inside a single token) and must be kept.
    assert not _is_excluded("Wallflower", subs)   # one token "wallflower"
    assert not _is_excluded("Floored", subs)      # one token "floored"


def test_is_excluded_drops_structural_scaffolding_wrappers():
    # /World and /World/Scene grouping Xforms wrap the whole authored scene; their
    # names must be treated as scaffolding so the walk never emits them as one
    # scene-spanning "world" object.
    subs = usd_index.DEFAULT_EXCLUDE_SUBSTRINGS
    assert _is_excluded("World", subs)
    assert _is_excluded("Scene", subs)
    assert _is_excluded("Root", subs)
    assert _is_excluded("Environment", subs)
    # but a real object is still kept
    assert not _is_excluded("Faucet", subs)
    assert not _is_excluded("Stove", subs)


def test_is_excluded_blank_is_excluded():
    assert _is_excluded("", usd_index.DEFAULT_EXCLUDE_SUBSTRINGS)
    assert _is_excluded("   ", usd_index.DEFAULT_EXCLUDE_SUBSTRINGS)


def test_is_excluded_custom_substrings_override_default():
    # caller-supplied list replaces the default shell list
    assert _is_excluded("Decoration", ("decoration",))
    assert not _is_excluded("Wall", ("decoration",))


# ----------------------------- _objects_from_bounds (the testable crux) ------

def test_objects_from_bounds_basic_object():
    objs = _objects_from_bounds(
        [_named_bounds("Faucet_01", (1.0, 2.0, 0.8), (1.4, 3.0, 1.2))]
    )
    assert len(objs) == 1
    o = objs[0]
    assert o.label == "faucet"
    assert o.id == "faucet"
    assert o.source == "usd"
    assert o.bbox_min == (1.0, 2.0, 0.8)
    assert o.bbox_max == (1.4, 3.0, 1.2)
    assert o.centroid == (1.2, 2.5, 1.0)
    assert o.extra["usd_prim_name"] == "Faucet_01"


def test_objects_from_bounds_excludes_shell():
    named = [
        _named_bounds("EastWall", (0, 0, 0), (10, 0.2, 3)),
        _named_bounds("Floor", (0, 0, -0.1), (10, 10, 0)),
        _named_bounds("KeyLight", (5, 5, 5), (5.1, 5.1, 5.1)),
        _named_bounds("Sink", (2, 2, 0.8), (2.6, 2.8, 1.0)),
        _named_bounds("Stove_01", (4, 2, 0.0), (5, 3, 1.0)),
    ]
    objs = _objects_from_bounds(named)
    assert sorted(o.label for o in objs) == ["sink", "stove"]


def test_objects_from_bounds_disambiguates_duplicate_ids():
    named = [
        _named_bounds("Knob_01", (0, 0, 0), (0.1, 0.1, 0.1)),
        _named_bounds("Knob_02", (1, 0, 0), (1.1, 0.1, 0.1)),
        _named_bounds("Knob_03", (2, 0, 0), (2.1, 0.1, 0.1)),
    ]
    objs = _objects_from_bounds(named)
    assert [o.id for o in objs] == ["knob", "knob_1", "knob_2"]
    assert all(o.label == "knob" for o in objs)


def test_objects_from_bounds_skips_unlabelable_names():
    named = [
        _named_bounds("001", (0, 0, 0), (1, 1, 1)),
        _named_bounds("Faucet", (1, 1, 0.8), (1.4, 1.4, 1.2)),
    ]
    objs = _objects_from_bounds(named)
    assert [o.label for o in objs] == ["faucet"]


def test_objects_from_bounds_coerces_to_float_tuples():
    objs = _objects_from_bounds([_named_bounds("Sink", [0, 0, 0], [2, 2, 2])])
    o = objs[0]
    assert o.bbox_min == (0.0, 0.0, 0.0)
    assert o.bbox_max == (2.0, 2.0, 2.0)
    assert isinstance(o.centroid[0], float)
    assert o.centroid == (1.0, 1.0, 1.0)


def test_objects_from_bounds_respects_custom_exclude():
    named = [
        _named_bounds("Wall", (0, 0, 0), (1, 1, 1)),
        _named_bounds("Faucet", (1, 1, 0), (1.4, 1.4, 1)),
    ]
    # with a non-default exclude list "Wall" is kept and "faucet" becomes shell
    objs = _objects_from_bounds(named, exclude_substrings=("faucet",))
    assert [o.label for o in objs] == ["wall"]


# ----------------------------- UsdSceneSpatialIndex (no pxr touched) ---------

def test_usd_index_requires_stage_or_path():
    with pytest.raises(ValueError):
        UsdSceneSpatialIndex()


def test_usd_index_objects_glues_walk_to_pure_builder(monkeypatch):
    # stub the only pxr-touching method; prove objects() delegates to the pure
    # builder (excludes the wall, cleans the label, computes the centroid)
    idx = UsdSceneSpatialIndex(stage=object())
    monkeypatch.setattr(
        idx,
        "_walk_stage",
        lambda: [
            ("Wall_North", ((0, 0, 0), (5, 0.2, 3))),
            ("Faucet_01", ((1.0, 2.0, 0.8), (1.4, 3.0, 1.2))),
        ],
    )
    objs = idx.objects()
    assert [o.label for o in objs] == ["faucet"]
    assert objs[0].source == "usd"
    assert objs[0].centroid == (1.2, 2.5, 1.0)


def test_usd_index_satisfies_spatial_index_protocol():
    idx = UsdSceneSpatialIndex(usd_path="/nonexistent/scene.usda")
    assert isinstance(idx, sp_types.SceneSpatialIndex)


def test_usd_walk_collapses_submeshes_and_excludes_shell_and_wrapper():
    """pxr-gated end-to-end walk: ONE SceneObject per real object, NOT per sub-mesh,
    and the structural shell + /World wrapper are excluded.

    This is the test that exercises ``_walk_stage`` (the only pxr-touching path, and
    where the wrapper-Xform bug lived). It builds a tiny in-memory Usd stage:

      /World                (Xform scaffolding -- must NOT become an object)
        /World/Sink         (Xform assembly -> ONE "sink", not basin+spout)
          /World/Sink/Basin (Mesh)
          /World/Sink/Spout (Mesh)
        /World/Stove        (Mesh -> "stove")
        /World/Wall         (Mesh -> shell, excluded)
        /World/Floor        (Mesh -> shell, excluded)

    and asserts objects() yields exactly {"sink", "stove"}: the multi-mesh sink is a
    single object, the wall/floor shell is dropped, and the /World wrapper does NOT
    collapse the whole scene into one bogus "world" object.
    """
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")

    def _cube(path, lo, hi):
        cube = UsdGeom.Mesh.Define(stage, path)
        # author an explicit world extent so BBoxCache returns a real range
        cube.CreatePointsAttr(
            [
                (lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]),
                (hi[0], hi[1], lo[2]), (lo[0], hi[1], lo[2]),
                (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]),
                (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2]),
            ]
        )
        cube.CreateExtentAttr([(lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2])])
        return cube

    # Multi-mesh "sink" assembly under a named Xform.
    UsdGeom.Xform.Define(stage, "/World/Sink")
    _cube("/World/Sink/Basin", (2.0, 2.0, 0.8), (2.6, 2.8, 1.0))
    _cube("/World/Sink/Spout", (2.2, 2.3, 1.0), (2.4, 2.5, 1.3))
    # A standalone object mesh.
    _cube("/World/Stove", (4.0, 2.0, 0.0), (5.0, 3.0, 1.0))
    # Shell prims.
    _cube("/World/Wall", (0.0, 0.0, 0.0), (10.0, 0.2, 3.0))
    _cube("/World/Floor", (0.0, 0.0, -0.1), (10.0, 10.0, 0.0))

    idx = UsdSceneSpatialIndex(stage=stage)
    objs = idx.objects()
    labels = sorted(o.label for o in objs)
    # exactly the two real objects; sink collapsed to one; shell + /World excluded
    assert labels == ["sink", "stove"]
    sink = next(o for o in objs if o.label == "sink")
    # the single "sink" bound spans BOTH its child meshes (basin + spout)
    assert sink.bbox_min[2] <= 0.8 + 1e-6
    assert sink.bbox_max[2] >= 1.3 - 1e-6


def test_drop_degenerate_boxes_filters_instancer_and_thin_planes():
    from blueprint_pipeline.scene_placement.usd_index import _drop_degenerate_boxes

    def _b(oid, lo, hi):
        return SceneObject(id=oid, label=oid, bbox_min=lo, bbox_max=hi,
                           centroid=tuple(0.5*(lo[i]+hi[i]) for i in range(3)), source="usd_leaf")
    boxes = [
        _b("cabinet", (0, 0, 0), (4.0, 0.6, 0.85)),        # long but real counter run -> KEEP
        _b("instancer", (-50, -50, -50), (50, 50, 50)),    # 100m degenerate -> DROP
        _b("knob", (1.0, 1.0, 0.8), (1.1, 1.1, 0.9)),      # normal -> KEEP
        _b("wall_plane", (0, 2.48, 0), (4.0, 2.48, 2.4)),  # zero-thickness sheet
    ]
    kept = _drop_degenerate_boxes(boxes, max_box_size=6.0)
    assert sorted(o.id for o in kept) == ["cabinet", "knob", "wall_plane"]   # only the 100m box dropped
    kept2 = _drop_degenerate_boxes(boxes, max_box_size=6.0, min_box_thickness=0.01)
    assert sorted(o.id for o in kept2) == ["cabinet", "knob"]                # thin plane also dropped


def test_obstacle_boxes_splits_aggregate_and_drops_degenerate():
    """pxr-gated: obstacle_boxes() emits one TIGHT box per leaf Gprim (so a multi-mesh assembly is
    several boxes for the clip test) while objects() keeps it grouped, and a 100m instancer box is
    dropped. This is the fix for an L-counter collapsing into one aisle-covering AABB.
    """
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")

    def _cube(path, lo, hi):
        m = UsdGeom.Mesh.Define(stage, path)
        m.CreatePointsAttr([(lo[0],lo[1],lo[2]),(hi[0],lo[1],lo[2]),(hi[0],hi[1],lo[2]),(lo[0],hi[1],lo[2]),
                            (lo[0],lo[1],hi[2]),(hi[0],lo[1],hi[2]),(hi[0],hi[1],hi[2]),(lo[0],hi[1],hi[2])])
        m.CreateExtentAttr([(lo[0],lo[1],lo[2]),(hi[0],hi[1],hi[2])])
        return m

    # one named "Cabinet" assembly of two SEPARATED segments (an L) + a degenerate huge leaf
    UsdGeom.Xform.Define(stage, "/World/Cabinet")
    _cube("/World/Cabinet/SegA", (0.0, 0.0, 0.0), (0.6, 0.6, 0.85))
    _cube("/World/Cabinet/SegB", (3.0, 3.0, 0.0), (3.6, 3.6, 0.85))
    _cube("/World/Cabinet/Instancer", (-50, -50, -50), (50, 50, 50))   # degenerate

    idx = UsdSceneSpatialIndex(stage=stage)
    grouped = idx.objects()
    assert [o.label for o in grouped] == ["cabinet"]            # ONE grouped object
    g = grouped[0]
    assert g.size()[0] >= 3.0 and g.size()[1] >= 3.0            # grouped AABB spans the whole L (coarse)

    fine = idx.obstacle_boxes()
    # two real segment boxes survive; the 100m instancer leaf is dropped; each box is TIGHT (~0.6m)
    assert len(fine) == 2
    assert all(o.label == "cabinet" for o in fine)
    assert all(max(o.size()) < 1.0 for o in fine)              # tight, not the coarse 3m+ aggregate


def test_obstacle_boxes_splits_disconnected_components_inside_one_mesh():
    """A single mesh can contain multiple disconnected cabinet pieces; split those too."""
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/PackedCabinet")

    def cube_points(lo, hi):
        return [
            (lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]), (hi[0], hi[1], lo[2]), (lo[0], hi[1], lo[2]),
            (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]), (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2]),
        ]

    faces = [
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
        [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
    ]
    points = cube_points((0.0, 0.0, 0.0), (0.5, 0.5, 0.8))
    points += cube_points((3.0, 3.0, 0.0), (3.5, 3.5, 0.8))
    counts = [4] * 12
    indices = [idx for face in faces for idx in face]
    indices += [idx + 8 for face in faces for idx in face]
    mesh = UsdGeom.Mesh.Define(stage, "/World/PackedCabinet/PackedCabinet")
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateExtentAttr([(0.0, 0.0, 0.0), (3.5, 3.5, 0.8)])

    fine = UsdSceneSpatialIndex(stage=stage).obstacle_boxes()

    assert len(fine) == 2
    assert [o.id for o in fine] == ["packedcabinet", "packedcabinet_1"]
    assert all(o.size()[0] == pytest.approx(0.5) for o in fine)
    assert all(o.size()[1] == pytest.approx(0.5) for o in fine)


# ===========================================================================
# perception_index — the camera math (the crux) + the detection->AABB index.
# Self-contained: depends only on types + perception_index, so it runs even
# while sibling scene_placement modules are still landing.
# ===========================================================================
from blueprint_pipeline.scene_placement.perception_index import (  # noqa: E402
    PerceptionSceneSpatialIndex,
    camera_basis,
    pixel_ray,
    resolve_intrinsics,
    unproject,
)


def _approx_vec(a, b, tol=1e-6):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x == pytest.approx(y, abs=tol)


def test_camera_basis_is_orthonormal_right_handed():
    """forward=eye->target; right/up_cam orthonormal; up_cam points world-up.

    A pixel ray expressed in (right, up_cam, forward) only maps to a clean world
    direction if the frame is orthonormal with forward as the optical axis, so we
    pin unit length, mutual orthogonality, and forward == normalized eye->target.
    """
    eye = (0.0, 0.0, 1.0)
    target = (2.0, 0.0, 1.0)  # looking down +x
    up = (0.0, 0.0, 1.0)
    right, up_cam, forward = camera_basis(eye, target, up)

    _approx_vec(forward, (1.0, 0.0, 0.0))
    for v in (right, up_cam, forward):
        assert math.sqrt(sum(c * c for c in v)) == pytest.approx(1.0)
    assert sum(a * b for a, b in zip(right, up_cam)) == pytest.approx(0.0, abs=1e-9)
    assert sum(a * b for a, b in zip(right, forward)) == pytest.approx(0.0, abs=1e-9)
    assert sum(a * b for a, b in zip(up_cam, forward)) == pytest.approx(0.0, abs=1e-9)
    assert up_cam[2] == pytest.approx(1.0)  # level camera -> up_cam is world +z


def test_camera_basis_handles_up_parallel_to_forward():
    """Looking straight down (up parallel to forward) must still yield a clean basis."""
    eye = (0.0, 0.0, 3.0)
    target = (0.0, 0.0, 0.0)  # straight down -z
    up = (0.0, 0.0, 1.0)  # parallel to view axis -> fallback path
    right, up_cam, forward = camera_basis(eye, target, up)
    _approx_vec(forward, (0.0, 0.0, -1.0))
    for v in (right, up_cam):
        assert math.sqrt(sum(c * c for c in v)) == pytest.approx(1.0)
    assert sum(a * b for a, b in zip(right, forward)) == pytest.approx(0.0, abs=1e-9)
    assert sum(a * b for a, b in zip(up_cam, forward)) == pytest.approx(0.0, abs=1e-9)


def test_unproject_on_axis_is_eye_plus_forward_times_depth():
    """The defining property: an on-axis pixel at depth d lands at eye + forward*d."""
    eye = (1.0, 2.0, 1.5)
    target = (5.0, 2.0, 1.5)  # +x
    up = (0.0, 0.0, 1.0)
    basis = camera_basis(eye, target, up)
    _, _, forward = basis
    intr = (200.0, 200.0, 64.0, 48.0)  # cx, cy == principal point

    for d in (0.5, 1.0, 3.7):
        p = unproject(64.0, 48.0, d, eye, intr, basis)
        expected = tuple(eye[k] + forward[k] * d for k in range(3))
        _approx_vec(p, expected)


def test_unproject_off_axis_sign_and_z_depth():
    """Off-axis pixels move along the correct basis axes; forward-extent == z-depth.

    px > cx must displace toward +right; py < cy (image-y grows downward) must
    displace toward +up_cam; and the displacement's forward component equals the
    requested z-depth (z-depth semantics, not Euclidean range).
    """
    eye = (0.0, 0.0, 0.0)
    target = (1.0, 0.0, 0.0)  # +x; right=(0,-1,0), up_cam=(0,0,1)
    up = (0.0, 0.0, 1.0)
    basis = camera_basis(eye, target, up)
    right, up_cam, forward = basis
    intr = (100.0, 100.0, 50.0, 50.0)
    d = 4.0

    p_right = unproject(80.0, 50.0, d, eye, intr, basis)
    disp = tuple(p_right[k] - eye[k] for k in range(3))
    assert sum(disp[k] * forward[k] for k in range(3)) == pytest.approx(d)
    assert sum(disp[k] * right[k] for k in range(3)) > 0.0
    assert sum(disp[k] * up_cam[k] for k in range(3)) == pytest.approx(0.0, abs=1e-9)

    p_up = unproject(50.0, 20.0, d, eye, intr, basis)  # above center
    disp_up = tuple(p_up[k] - eye[k] for k in range(3))
    assert sum(disp_up[k] * up_cam[k] for k in range(3)) > 0.0
    assert sum(disp_up[k] * forward[k] for k in range(3)) == pytest.approx(d)


def test_resolve_intrinsics_explicit_and_vfov_aspect():
    """vfov form yields square-pixel intrinsics; a wider image gets a wider H-FOV."""
    fx, fy, cx, cy = resolve_intrinsics({"fx": 10.0, "fy": 11.0, "cx": 5.0, "cy": 6.0})
    assert (fx, fy, cx, cy) == (10.0, 11.0, 5.0, 6.0)

    vfov = math.radians(60.0)
    fx, fy, cx, cy = resolve_intrinsics({"vfov": vfov, "width": 1600, "height": 900})
    assert fx == pytest.approx(fy)  # square pixels
    assert cx == pytest.approx(800.0)
    assert cy == pytest.approx(450.0)
    assert fy == pytest.approx((900 / 2.0) / math.tan(vfov / 2.0))

    # aspect handled: the right-edge horizontal half-angle is wider on 16:9 than 1:1
    eye, target, up = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    basis = camera_basis(eye, target, up)
    forward = basis[2]

    def edge_h_angle(width, height):
        intr = resolve_intrinsics({"vfov": vfov, "width": width, "height": height})
        ray = pixel_ray(width - 1.0, intr[3], intr, basis)  # right edge, mid height
        dot = sum(ray[i] * forward[i] for i in range(3))
        nr = math.sqrt(sum(c * c for c in ray))
        return math.degrees(math.acos(dot / nr))

    assert edge_h_angle(1600, 900) > edge_h_angle(900, 900)
    # square image: horizontal half-angle ~ vertical half (vfov/2 = 30deg)
    assert edge_h_angle(900, 900) == pytest.approx(30.0, abs=0.5)


def test_resolve_intrinsics_missing_raises():
    with pytest.raises(ValueError):
        resolve_intrinsics({"width": 100, "height": 100})  # no vfov, no fx/fy


def test_perception_index_unprojects_centered_detection():
    """A frame-centered box at constant depth -> world AABB straddling eye+forward*d."""
    width, height = 128.0, 96.0
    camera = {
        "vfov": math.radians(60.0),
        "width": width,
        "height": height,
        "eye": (0.0, 0.0, 1.0),
        "target": (4.0, 0.0, 1.0),  # +x
        "up": (0.0, 0.0, 1.0),
    }
    depth = 2.0
    depth_provider = lambda px, py: depth  # noqa: E731  (constant-depth slab)

    cx, cy = width / 2.0, height / 2.0
    bbox = (cx - 20.0, cy - 15.0, cx + 20.0, cy + 15.0)
    detections = [{"label": "faucet", "bbox_px": bbox, "confidence": 0.91}]

    index = PerceptionSceneSpatialIndex(detections, depth_provider, camera)
    objs = index.objects()
    assert len(objs) == 1
    obj = objs[0]
    assert isinstance(obj, SceneObject)
    assert obj.label == "faucet"
    assert obj.source == "perception"
    assert obj.confidence == pytest.approx(0.91)

    # box center is on the optical axis at depth d -> eye + forward*d = (2, 0, 1)
    _approx_vec(obj.centroid, (2.0, 0.0, 1.0))
    assert obj.bbox_min[1] < obj.centroid[1] < obj.bbox_max[1]
    assert obj.bbox_min[2] < obj.centroid[2] < obj.bbox_max[2]
    # constant-depth slab is flat in the forward (x) direction
    assert obj.bbox_min[0] == pytest.approx(2.0)
    assert obj.bbox_max[0] == pytest.approx(2.0)
    assert obj.extra["depth_m"] == pytest.approx(depth)


def test_perception_index_uses_median_depth_robust_to_outliers():
    """A single far-outlier depth sample inside the box must not move the AABB.

    2D boxes clip background, so median (not mean) sampling is the WHY: one stray
    large depth should be ignored and the resolved depth stays at the dominant value.
    """
    camera = {
        "fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0,
        "eye": (0.0, 0.0, 0.0), "target": (1.0, 0.0, 0.0), "up": (0.0, 0.0, 1.0),
    }
    bbox = (40.0, 40.0, 60.0, 60.0)

    def depth_provider(px, py):
        if px <= 40.5 and py <= 40.5:  # exactly one grid corner is a wild outlier
            return 99.0
        return 3.0

    index = PerceptionSceneSpatialIndex(
        [{"label": "sink", "bbox_px": bbox}], depth_provider, camera, samples_per_axis=3
    )
    obj = index.objects()[0]
    assert obj.extra["depth_m"] == pytest.approx(3.0)  # median of 8x3.0 + 1x99.0
    assert obj.centroid[0] == pytest.approx(3.0)


def test_perception_index_skips_detection_with_no_valid_depth():
    """A box over a depth-map hole (all invalid samples) is dropped, not crashed."""
    camera = {
        "fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0,
        "eye": (0.0, 0.0, 0.0), "target": (1.0, 0.0, 0.0), "up": (0.0, 0.0, 1.0),
    }
    detections = [
        {"label": "good", "bbox_px": (45.0, 45.0, 55.0, 55.0)},
        {"label": "hole", "bbox_px": (10.0, 10.0, 20.0, 20.0)},
    ]

    def depth_provider(px, py):
        if px < 30.0 and py < 30.0:  # the "hole" box reads NaN everywhere
            return float("nan")
        return 2.5

    index = PerceptionSceneSpatialIndex(detections, depth_provider, camera)
    assert [o.label for o in index.objects()] == ["good"]


def test_perception_index_skips_malformed_bbox_without_crashing():
    """A non-iterable / wrong-length bbox_px is SKIPPED, not allowed to crash the
    whole scene build — same robustness contract as the no-valid-depth skip.

    Regression: a scalar bbox_px (e.g. an int from upstream) made ``tuple(bbox_px)``
    raise TypeError, which propagated out of objects() and lost every other object.
    """
    camera = {
        "fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0,
        "eye": (0.0, 0.0, 0.0), "target": (1.0, 0.0, 0.0), "up": (0.0, 0.0, 1.0),
    }
    detections = [
        {"label": "scalar_box", "bbox_px": 5},          # not iterable at all
        {"label": "short_box", "bbox_px": (1.0, 2.0)},  # iterable but wrong length
        {"label": "good", "bbox_px": (45.0, 45.0, 55.0, 55.0)},
    ]
    index = PerceptionSceneSpatialIndex(detections, lambda px, py: 2.0, camera)
    # The two malformed detections are dropped; the valid one survives.
    assert [o.label for o in index.objects()] == ["good"]


def test_perception_index_satisfies_spatial_index_protocol():
    """Duck-typed conformance: objects() returns SceneObjects, isinstance-checkable."""
    from blueprint_pipeline.scene_placement.types import SceneSpatialIndex

    camera = {
        "fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0,
        "eye": (0.0, 0.0, 0.0), "target": (1.0, 0.0, 0.0), "up": (0.0, 0.0, 1.0),
    }
    index = PerceptionSceneSpatialIndex(
        [{"label": "stove", "bbox_px": (45.0, 45.0, 55.0, 55.0)}],
        lambda px, py: 2.0,
        camera,
    )
    assert isinstance(index, SceneSpatialIndex)
    objs = index.objects()
    assert objs and all(isinstance(o, SceneObject) for o in objs)


# =========================================================================== #
# placement — the open-side solver (PURE given an injected probe; NO PhysX/GPU)
#
# A mock probe marks occupied floor cells exactly like a PhysX footprint overlap
# (0 == clear). These pin the contract that GENERALIZES the runner's old -y-only
# find_clear_stand: pick the open side of an L-counter, face the target, respect
# the standoff, expose a single open side when boxed by two walls, and fall back
# (clear=False) when nothing is clear.
# =========================================================================== #

from blueprint_pipeline.scene_placement.placement import compute_stand_pose  # noqa: E402


def _target(cx=0.0, cy=0.0, cz=0.9, hx=0.2, hy=0.2, hz=0.1, id_="faucet", label="faucet"):
    """A target AABB centered at (cx, cy, cz) with half-extents (hx, hy, hz)."""
    return SceneObject(
        id=id_, label=label,
        bbox_min=(cx - hx, cy - hy, cz - hz),
        bbox_max=(cx + hx, cy + hy, cz + hz),
        centroid=(cx, cy, cz),
    )


def test_compute_stand_pose_picks_open_side_of_l_counter():
    # L-shaped counter: the counter occupies the +x leg and the +y leg, so the
    # robot must NOT stand on +x or +y. The open sides are -x and -y; the solver
    # should land on a clear spot off the target, never inside the occupied legs.
    tgt = _target()

    def probe(pose, yaw):
        x, y, _z = pose
        # occupied wherever the L's legs are (anything at +x or +y of the corner)
        if x > 0.15 or y > 0.15:
            return 3  # PhysX-style hit count > 0
        return 0

    pose = compute_stand_pose(tgt, probe=probe)
    assert pose.clear is True
    assert pose.target_id == "faucet"
    # stood on an open side (negative x or negative y), not inside the legs
    px, py, pz = pose.position
    assert px <= 0.16 and py <= 0.16
    assert (px < 0) or (py < 0)
    # pelvis lifted to floor_z + pelvis_height
    assert abs(pz - 0.79) < 1e-9


def test_compute_stand_pose_preferred_direction_beats_closest_clear():
    # Every side reads as clear (e.g. walls with no collision). Without a preference the solver
    # would pick whatever direction sorts first; with preferred_direction it must stand on the
    # side aligned with that hint (the open room / approach side), not against a wall.
    tgt = _target(cx=2.28, cy=1.33)

    pose = compute_stand_pose(
        tgt, probe=lambda p, y: 0, include_diagonals=True,
        preferred_direction=(0.0, -1.0),   # prefer the -y (room/approach) side
    )
    assert pose.clear is True
    assert pose.position[1] < 1.33          # stood on the -y side, in front of the target
    assert pose.position[0] == pytest.approx(2.28, abs=1e-6)  # aligned, not off on +x/-x
    assert math.sin(pose.yaw) > 0.5         # faces +y toward the target
    # the opposite hint flips it to the +y side
    pose2 = compute_stand_pose(tgt, probe=lambda p, y: 0, include_diagonals=True,
                               preferred_direction=(0.0, 1.0))
    assert pose2.position[1] > 1.33


def test_compute_stand_pose_faces_the_target():
    # target offset from origin; whatever open side is chosen, yaw must point the
    # pelvis forward axis at the target centroid (atan2 of the delta).
    tgt = _target(cx=2.0, cy=3.0)

    def probe(pose, yaw):
        # only -y side open (force a known facing direction)
        x, y, _z = pose
        return 0 if (y < tgt.centroid[1] - 0.1 and abs(x - tgt.centroid[0]) < 0.3) else 1

    pose = compute_stand_pose(tgt, probe=probe)
    assert pose.clear is True
    expected_yaw = math.atan2(
        tgt.centroid[1] - pose.position[1], tgt.centroid[0] - pose.position[0]
    )
    assert abs(pose.yaw - expected_yaw) < 1e-9
    # stood on the -y side as forced
    assert pose.position[1] < tgt.centroid[1]


def test_compute_stand_pose_respects_standoff():
    # all four sides open -> robot stands at the nearest clear spot, which is
    # half_extent + standing_distance from the footprint center.
    tgt = _target(hx=0.3, hy=0.3)
    pose = compute_stand_pose(
        tgt, probe=lambda p, y: 0, standing_distance=0.55, step=0.1
    )
    assert pose.clear is True
    # standoff is measured from the target surface, ~ standing_distance
    assert abs(pose.standoff_m - 0.55) < 1e-6
    # distance from footprint center == half_extent (0.3) + standoff (0.55)
    cx, cy = tgt.footprint_center()
    dist = math.hypot(pose.position[0] - cx, pose.position[1] - cy)
    assert abs(dist - (0.3 + 0.55)) < 1e-6


def test_compute_stand_pose_two_walls_single_open_side():
    # boxed by walls on +x, -x, +y; only -y is open -> must pick -y.
    tgt = _target()

    def probe(pose, yaw):
        x, y, _z = pose
        return 0 if (y < -0.1 and abs(x) < 0.3) else 1

    pose = compute_stand_pose(tgt, probe=probe)
    assert pose.clear is True
    assert pose.position[1] < 0.0  # -y side
    assert abs(pose.position[0]) < 0.3


def test_compute_stand_pose_nothing_clear_falls_back_unclear():
    # fully boxed in: every probe returns a hit -> best-effort pose at max_out,
    # flagged clear=False so the caller knows the floor was never verified.
    tgt = _target()
    pose = compute_stand_pose(tgt, probe=lambda p, y: 5, max_out=2.5, step=0.1)
    assert pose.clear is False
    assert "no clear side" in pose.notes
    # fell back to the farthest probed spot: within one probe step of the max_out
    # search ceiling, and an honest read of where the probe actually reached (not a
    # synthetic max_out constant).
    assert 2.5 - 0.1 - 1e-6 <= pose.standoff_m <= 2.5 + 1e-6
    # still lifts the pelvis to floor_z + pelvis_height
    assert abs(pose.position[2] - 0.79) < 1e-9


def test_compute_stand_pose_probes_standoff_even_when_distance_exceeds_max_out():
    """When standing_distance >= max_out the standoff spot must STILL be probed.

    Regression: with start = half + standing_distance and ceiling = half + max_out,
    a standing_distance > max_out made start > ceiling, so the probe loop never ran,
    the probe was never called, and the pose was reported clear=False even though
    the floor was clear. The ceiling must dominate the start so the loop runs at
    least once at the standoff spot.
    """
    tgt = _target()
    calls = {"n": 0}

    def probe(pose, yaw):
        calls["n"] += 1
        return 0  # clear floor everywhere

    pose = compute_stand_pose(
        tgt, probe=probe, standing_distance=3.0, max_out=2.5, step=0.1
    )
    assert calls["n"] > 0  # the standoff spot was actually probed
    assert pose.clear is True  # and the clear floor was recognized
    # stood at half_extent + standing_distance from the footprint center
    cx, cy = tgt.footprint_center()
    dist = math.hypot(pose.position[0] - cx, pose.position[1] - cy)
    assert abs(dist - (0.2 + 3.0)) < 1e-6


def test_compute_stand_pose_fallback_reflects_most_open_not_first_cardinal():
    """The boxed-in fallback's openness signal varies by direction (it is no longer a
    constant that always makes +x win).

    Regression: every non-clear direction returned the SAME constant standoff, so the
    fallback tiebreak (``score > best``) never fired and the first cardinal (+x) was
    always chosen regardless of which side was genuinely more open. We now score each
    blocked direction by how far it reached minus how many steps were blocked, so a
    target whose half-extents differ per axis yields DIFFERENT fallback scores per
    direction — proving the selection has real signal rather than a dead constant.

    Here the target is much wider in x than y, so the +x/-x probes start (and reach)
    farther out than +y/-y; with everything blocked, the chosen fallback is therefore
    the deeper-reaching x axis, and crucially the reported standoff is the honest
    farthest-probed distance, not a synthetic constant identical across directions.
    """
    # Wide-in-x target: half-extent along x (1.0) >> along y (0.1).
    tgt = _target(hx=1.0, hy=0.1)

    pose = compute_stand_pose(tgt, probe=lambda p, y: 7, max_out=1.0, step=0.1)
    assert pose.clear is False
    assert "most-open direction" in pose.notes
    cx, cy = tgt.footprint_center()
    # The fallback landed on an x-axis direction (the one that reaches farthest given
    # the wide footprint), NOT pinned to +y/-y, and the standoff is the real reached
    # distance off the surface (honest, not a constant max_out).
    assert abs(pose.position[1] - cy) < 1e-6  # on the x axis (y ~ centered)
    assert pose.standoff_m > 0.0


def test_compute_stand_pose_prefers_closest_clear_side():
    # -x is open but only far out (clutter close in); -y is open right away. The
    # solver minimizes distance to target, so it must pick the closer -y spot.
    tgt = _target()

    def probe(pose, yaw):
        x, y, _z = pose
        # +x, +y always blocked
        if x > 0.15 or y > 0.15:
            return 1
        # -x blocked until far out (> 1.5 from center)
        if x < 0 and abs(x) < 1.5:
            return 1
        # -y open immediately past the standoff
        return 0

    pose = compute_stand_pose(tgt, probe=probe)
    assert pose.clear is True
    # the closer open side is -y
    assert pose.position[1] < 0.0


def test_compute_stand_pose_diagonals_when_cardinals_blocked():
    # corner target: all four cardinals blocked, but a diagonal approach is open.
    # include_diagonals lets the solver find it (off-axis approach into a corner).
    tgt = _target()

    def probe(pose, yaw):
        x, y, _z = pose
        # block pure-axis approaches (one coord ~ 0 => a cardinal direction)
        on_axis = abs(x) < 0.05 or abs(y) < 0.05
        if on_axis:
            return 1
        # the -x,-y diagonal quadrant is open
        return 0 if (x < 0 and y < 0) else 1

    # without diagonals -> no clear cardinal -> unclear fallback
    no_diag = compute_stand_pose(tgt, probe=probe, include_diagonals=False)
    assert no_diag.clear is False
    # with diagonals -> finds the open -x,-y corner
    with_diag = compute_stand_pose(tgt, probe=probe, include_diagonals=True)
    assert with_diag.clear is True
    assert with_diag.position[0] < 0.0 and with_diag.position[1] < 0.0


def test_compute_stand_pose_returns_standpose_type():
    pose = compute_stand_pose(_target(), probe=lambda p, y: 0)
    assert isinstance(pose, StandPose)


# =========================================================================== #
# place_robot_for_task + build_scene_index — the package orchestrator/factory
# (fake index + fake probe + fake generate; NO VLM / NO USD / NO GPU)
# =========================================================================== #

from blueprint_pipeline import scene_placement as sp  # noqa: E402


class _FakeIndex:
    """Minimal SceneSpatialIndex: returns a fixed object list."""

    def __init__(self, objects):
        self._objects = objects

    def objects(self):
        return list(self._objects)


def _scene_for_orchestrator():
    return [
        _target(cx=1.0, cy=2.0, id_="faucet_1", label="kitchen_faucet"),
        _target(cx=3.0, cy=2.0, id_="stove_1", label="stove"),
    ]


def test_place_robot_for_task_end_to_end_with_fake_vlm():
    index = _FakeIndex(_scene_for_orchestrator())
    # fake VLM picks the faucet; fake probe says all floor is clear
    pose = sp.place_robot_for_task(
        index,
        "turn on the faucet",
        probe=lambda p, y: 0,
        generate=lambda prompt: '{"target_id": "faucet_1"}',
    )
    assert isinstance(pose, StandPose)
    assert pose.target_id == "faucet_1"
    assert pose.clear is True
    # stands near the faucet (centroid at (1,2)), facing it
    cx, cy = 1.0, 2.0
    expected_yaw = math.atan2(cy - pose.position[1], cx - pose.position[0])
    assert abs(pose.yaw - expected_yaw) < 1e-9


def test_place_robot_for_task_vlm_hallucination_degrades_to_label_end_to_end():
    """A VLM that names a ghost id must NOT break placement: the orchestrator degrades
    to the label fallback and still stands the robot at a real target.

    Covers the VLM->label degrade end-to-end (previously only unit-tested on
    resolve_target, never through place_robot_for_task).
    """
    index = _FakeIndex(_scene_for_orchestrator())
    pose = sp.place_robot_for_task(
        index,
        "turn on the faucet",
        probe=lambda p, y: 0,
        generate=lambda prompt: '{"target_id": "ghost_does_not_exist"}',
    )
    assert isinstance(pose, StandPose)
    # label fallback found the faucet despite the hallucinated id
    assert pose.target_id == "faucet_1"
    assert pose.clear is True


def test_place_robot_for_task_faucet_against_wall_stands_on_open_side():
    """Contract end-to-end: faucet against a back wall on a counter, robot stands
    on the OPEN side and faces the faucet.

    This is the realistic placement the whole package exists for. The scene has a
    few SceneObjects (faucet, sink, stove) with the faucet pushed up against a wall
    that runs along +y, sitting on a counter that occupies the +y half-plane. A
    fake probe marks that counter/wall band as occupied (PhysX-style hit count > 0),
    and a fake VLM returns the faucet's id. The orchestrator must therefore resolve
    the faucet, refuse the blocked +y/wall side, and land the pelvis on the open -y
    side facing back toward the faucet centroid — with NO hardcoded coordinates.
    """
    faucet = _target(cx=1.0, cy=2.0, cz=1.0, id_="faucet_1", label="kitchen_faucet")
    scene = [
        faucet,
        _target(cx=1.0, cy=2.0, cz=0.85, id_="sink_1", label="sink"),
        _target(cx=3.5, cy=2.0, cz=0.9, hx=0.5, hy=0.4, id_="stove_1", label="stove"),
    ]
    index = _FakeIndex(scene)

    fcx, fcy = faucet.footprint_center()

    def probe(pose, yaw):
        # Counter + back wall occupy the +y half-plane at the faucet's x band; any
        # candidate that lands at/behind the faucet (y >= its footprint center) is a
        # collision. Everything on the open -y side in front of the counter is clear.
        x, y, _z = pose
        if y >= fcy - 1e-6 and abs(x - fcx) < 1.2:
            return 4  # occupied: counter/back wall
        return 0  # open floor in front

    captured = {}

    def fake_generate(prompt: str) -> str:
        captured["prompt"] = prompt
        return '{"target_id": "faucet_1"}'

    pose = sp.place_robot_for_task(
        index, "turn on the faucet", probe=probe, generate=fake_generate
    )

    assert isinstance(pose, StandPose)
    # VLM target was honored and the scene was actually described to it.
    assert pose.target_id == "faucet_1"
    assert "turn on the faucet" in captured["prompt"]
    assert "faucet_1" in captured["prompt"]
    # Probe verified a clear spot (did NOT fall back to the unclear best-effort).
    assert pose.clear is True
    # Stood on the OPEN side: in front of the counter (-y of the faucet), never in
    # the occupied +y band behind it.
    assert pose.position[1] < fcy
    # Faces the faucet centroid (yaw points from the pelvis back at the target).
    expected_yaw = math.atan2(
        faucet.centroid[1] - pose.position[1],
        faucet.centroid[0] - pose.position[0],
    )
    assert abs(pose.yaw - expected_yaw) < 1e-9
    # And it is genuinely standing off the surface, not on top of the faucet.
    assert pose.standoff_m > 0.0
    # Pelvis lifted to the default standing height.
    assert abs(pose.position[2] - 0.79) < 1e-9


def test_place_and_validate_robot_for_task_returns_pose_and_verdict():
    # The self-validating orchestrator: place the robot AND geometrically validate the pose against
    # the scene catalog in one call, so a caller knows it's clean without rendering.
    faucet = _target(cx=1.0, cy=2.0, cz=1.0, id_="faucet_1", label="kitchen_faucet")
    counter = _target(cx=1.0, cy=2.0, cz=0.45, hx=0.6, hy=0.3, id_="counter", label="counter")
    scene = [faucet, _target(cx=1.0, cy=2.0, cz=0.85, id_="sink_1", label="sink"), counter]
    index = _FakeIndex(scene)
    fcx, fcy = faucet.footprint_center()

    def probe(pose, yaw):
        x, y, _z = pose
        return 4 if (y >= fcy - 1e-6 and abs(x - fcx) < 1.2) else 0   # counter/wall band occupied

    pose, verdict = sp.place_and_validate_robot_for_task(
        index, "turn on the faucet", probe=probe, floor_z=0.0, standing_distance=0.85,
    )
    assert isinstance(pose, StandPose) and isinstance(verdict, sp.PlacementVerdict)
    assert pose.target_id == "faucet_1"
    assert pose.position[1] < fcy            # stood on the open side, in front
    assert verdict.ok is True                # self-validated: on floor, facing, no clip, in standoff
    assert verdict.clipping == [] and verdict.on_floor is True
    assert 0.30 <= verdict.standoff_m <= 1.30 and verdict.facing_error_deg < 5


def test_place_robot_for_task_uses_label_fallback_when_no_generate():
    index = _FakeIndex(_scene_for_orchestrator())
    # generate=None -> pure label resolver finds the faucet by label
    pose = sp.place_robot_for_task(index, "turn on the faucet", probe=lambda p, y: 0)
    assert pose.target_id == "faucet_1"
    assert pose.clear is True


def test_place_robot_for_task_no_target_raises():
    index = _FakeIndex([_target(label="lamp", id_="lamp_1")])
    with pytest.raises(LookupError):
        sp.place_robot_for_task(index, "turn on the faucet", probe=lambda p, y: 0)


def test_place_robot_for_task_forwards_place_kwargs():
    index = _FakeIndex(_scene_for_orchestrator())
    pose = sp.place_robot_for_task(
        index,
        "use the stove",
        probe=lambda p, y: 0,
        pelvis_height=1.0,
        floor_z=0.5,
    )
    assert pose.target_id == "stove_1"
    # pelvis z = floor_z + pelvis_height = 0.5 + 1.0
    assert abs(pose.position[2] - 1.5) < 1e-9


def test_build_scene_index_factory_dispatch():
    # 'perception' backend builds a PerceptionSceneSpatialIndex
    camera = {
        "fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0,
        "width": 640, "height": 480,
        "eye": (0.0, 0.0, 1.0), "target": (1.0, 0.0, 1.0),
    }
    idx = sp.build_scene_index(
        "perception", detections=[], depth_provider=lambda px, py: 1.0, camera=camera
    )
    assert isinstance(idx, PerceptionSceneSpatialIndex)
    # 'usd' backend builds a UsdSceneSpatialIndex
    usd = sp.build_scene_index("usd", usd_path="/nonexistent/scene.usda")
    assert isinstance(usd, UsdSceneSpatialIndex)
    # unknown backend is rejected
    with pytest.raises(ValueError):
        sp.build_scene_index("nonsense")
