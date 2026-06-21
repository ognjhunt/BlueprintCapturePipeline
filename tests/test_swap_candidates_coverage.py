from types import SimpleNamespace

import pytest

from blueprint_pipeline.capture_bridge import CaptureDescriptor
from blueprint_pipeline import swap_candidates as sc


def _descriptor(**overrides):
    data = {
        "schema_version": "v1",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "capture_source": "iphone",
        "capture_tier": "alpha",
        "raw_prefix_uri": "file:///raw",
        "frames_index_uri": "file:///frames.json",
        "environment_type_hint": "warehouse",
        "swap_focus": ["Factory", "warehouse", ""],
        "manipulation_candidates": [],
        "articulation_hints": [],
    }
    data.update(overrides)
    return CaptureDescriptor(**data)


def test_swap_policy_helpers_and_custom_policy_loading(tmp_path, monkeypatch):
    config = sc.SwapPolicyConfig(
        name="policy",
        source="unit",
        articulated_appliance_keywords=frozenset({"fridge", "oven"}),
        articulated_furniture_keywords=frozenset({"cabinet"}),
        manipulable_keywords=frozenset({"tote", "mug"}),
        exclude_keywords=frozenset({"floor"}),
        min_volume_m3={"manipulable_object": 0.2},
    )
    assert config.to_dict() == {
        "name": "policy",
        "source": "unit",
        "articulated_appliance_keywords": ["fridge", "oven"],
        "articulated_furniture_keywords": ["cabinet"],
        "manipulable_keywords": ["mug", "tote"],
        "exclude_keywords": ["floor"],
        "min_volume_m3": {"manipulable_object": 0.2},
    }

    candidate = sc.SwapCandidate(
        object_id="obj-1",
        label="Tote",
        sim_role="manipulable_object",
        articulation_required=False,
        articulation_reason="keyword",
        must_be_separate_asset=True,
        asset_dir="obj_obj-1",
        point_cloud_file=None,
        obb={"center": [0, 0, 0]},
        dimensions_est={"width": 1.0},
        physics_hints={"dynamic": True},
        reference_crop="crop.png",
        all_crops=["crop.png", "crop2.png"],
    )
    assert candidate.to_dict()["reference_crop"] == "crop.png"
    assert candidate.to_dict()["all_crops"] == ["crop.png", "crop2.png"]
    assert sc.SwapCandidate(
        object_id="obj-2",
        label="Shelf",
        sim_role="articulated_furniture",
        articulation_required=True,
        articulation_reason="descriptor",
        must_be_separate_asset=True,
        asset_dir="obj_obj-2",
        point_cloud_file="cloud.ply",
        obb={},
        dimensions_est={},
        physics_hints={},
    ).to_dict()["point_cloud_file"] == "cloud.ply"

    assert sc._normalized_tokens([" A ", "a", "", None, "B"]) == ["a", "none", "b"]
    assert sc._merge_keywords(["cup"], None) == ["cup"]
    assert sc._merge_keywords(["cup"], "Mug") == ["cup", "mug"]
    assert sc._merge_keywords(None, ["Mug", "mug", "Bin"]) == ["mug", "bin"]
    assert sc._safe_float("1.25") == 1.25
    assert sc._safe_float("bad", 4.0) == 4.0
    assert sc._merge_min_volume({"a": "0.5"}, {"b": "bad"}) == {"a": 0.5, "b": 0.0}

    copied = sc._deep_copy_policy({"defaults": [], "environments": []})
    assert copied == {
        "defaults": {
            "articulated_appliance_keywords": [],
            "articulated_furniture_keywords": [],
            "manipulable_keywords": [],
            "exclude_keywords": [],
            "min_volume_m3": {},
        },
        "environments": {},
    }
    assert sc._deep_copy_policy({"defaults": {}, "environments": {"skip": "not-a-mapping"}})[
        "environments"
    ] == {}

    defaults, name, source = sc._load_policy_payload(None)
    assert name == "auto_by_signals_default"
    assert source == "builtin_default"
    assert "warehouse" in defaults["environments"]

    missing = tmp_path / "missing.yaml"
    with pytest.raises(ValueError, match="not found"):
        sc._load_policy_payload(str(missing))

    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text("ignored: true", encoding="utf-8")

    monkeypatch.setattr(sc, "yaml", None)
    with pytest.raises(ValueError, match="PyYAML"):
        sc._load_policy_payload(str(policy_path))

    monkeypatch.setattr(sc, "yaml", SimpleNamespace(safe_load=lambda _: ["not", "a", "mapping"]))
    with pytest.raises(ValueError, match="invalid swap policy payload type"):
        sc._load_policy_payload(str(policy_path))

    monkeypatch.setattr(
        sc,
        "yaml",
        SimpleNamespace(safe_load=lambda _: {"schema_version": "v9"}),
    )
    with pytest.raises(ValueError, match="unsupported swap policy schema_version"):
        sc._load_policy_payload(str(policy_path))

    loaded = {
        "schema_version": "v1",
        "policy_name": "custom-policy",
        "defaults": {
            "manipulable_keywords": "custom_bin",
            "exclude_keywords": ["custom_floor"],
            "min_volume_m3": {"manipulable_object": "0.4", "bad": "nan"},
        },
        "environments": {
            "Factory": {
                "articulated_appliance_keywords": ["servo_door"],
                "articulated_furniture_keywords": ["tool_chest"],
                "manipulable_keywords": ["fixture"],
                "exclude_keywords": ["weld_cell"],
                "min_volume_m3": {"articulated_furniture": "0.12"},
            },
            "ignored": "not-a-mapping",
        },
    }
    monkeypatch.setattr(sc, "yaml", SimpleNamespace(safe_load=lambda _: loaded))

    payload, policy_name, loaded_source = sc._load_policy_payload(str(policy_path))
    assert policy_name == "custom-policy"
    assert loaded_source == str(policy_path)
    assert "custom_bin" in payload["defaults"]["manipulable_keywords"]
    assert "factory" in payload["environments"]

    resolved = sc.resolve_policy_config(descriptor=_descriptor(), policy_path=str(policy_path))
    assert resolved.name == "custom-policy"
    assert resolved.source == str(policy_path)
    assert "fixture" in resolved.manipulable_keywords
    assert "servo_door" in resolved.articulated_appliance_keywords
    assert resolved.min_volume_m3["articulated_furniture"] == 0.12

    patched_payload = {
        "defaults": {
            "articulated_appliance_keywords": [],
            "articulated_furniture_keywords": [],
            "manipulable_keywords": ["box"],
            "exclude_keywords": [],
            "min_volume_m3": {},
        },
        "environments": {"warehouse": "not-a-mapping"},
    }
    monkeypatch.setattr(
        sc,
        "_load_policy_payload",
        lambda _policy_path: (patched_payload, "patched-policy", "patched-source"),
    )
    patched = sc.resolve_policy_config(descriptor=_descriptor(environment_type_hint="warehouse"))
    assert patched.name == "patched-policy"
    assert patched.manipulable_keywords == frozenset({"box"})


def test_swap_object_helpers_and_role_classification():
    assert sc._normalized_text(" Mug ", None, "Blue") == "mug blue"
    assert sc._object_id({"id": " primary "}) == "primary"
    assert sc._object_id({"object_id": " object "}) == "object"
    assert sc._object_id({"uuid": " uuid "}) == "uuid"
    assert sc._object_id({"identifier": " identifier "}) == "identifier"
    with pytest.raises(ValueError, match="missing id"):
        sc._object_id({"label": "missing"})

    assert sc._label({"label": " Mug "}) == "Mug"
    assert sc._label({"name": "Box"}) == "Box"
    assert sc._label({"class_name": "Cabinet"}) == "Cabinet"
    assert sc._label({"category": "Appliance"}) == "Appliance"
    assert sc._label({}) == "object"

    obb = sc._bounding_box(
        {
            "obb": {
                "center": ["1", "bad", 3],
                "extents": [0, "0.5"],
                "axes": [["1", 0], "bad", [0, 0, "1"]],
                "orientationQuaternion": [0, 0, 0, 0],
            }
        }
    )
    assert obb["center"] == [1.0, 0.0, 3.0]
    assert obb["extents"] == [0.02, 0.5, 0.25]
    assert obb["axes"][0] == [1.0, 0.0, 0.0]
    assert obb["axes"][1] == [0.0, 0.0, 0.0]
    assert obb["orientationQuaternion"] == [1.0, 0.0, 0.0, 0.0]
    assert sc._bounding_box({"boundingBox": {"extents": [1, 2, 3]}})["extents"] == [1.0, 2.0, 3.0]
    assert sc._dimensions_from_obb({"extents": ["0", "2", "bad"]}) == {
        "width": 0.02,
        "height": 2.0,
        "depth": 0.25,
    }
    assert sc._dimensions_from_obb({}) == {"width": 0.25, "height": 0.25, "depth": 0.25}

    descriptor = _descriptor(
        manipulation_candidates=[
            {"instance_id": "m1", "label": "Special Tote"},
            {"id": "m2", "name": "Loose Bin"},
            {"label": ""},
        ],
        articulation_hints=[
            {"instance_id": "a1", "label": "Fridge Door"},
            {"id": "a2", "name": "Locker Door"},
            {"name": ""},
        ],
    )
    manip_ids, manip_labels, articulated_ids, articulated_labels = sc._manipulation_lookup(descriptor)
    assert manip_ids == {"m1", "m2"}
    assert manip_labels == ["special tote", "loose bin"]
    assert articulated_ids == {"a1", "a2"}
    assert articulated_labels == ["fridge door", "locker door"]

    policy = sc.resolve_policy_config(descriptor=_descriptor(environment_type_hint="kitchen", swap_focus=[]))
    assert sc._contains_any("large blue mug", ["mug"])
    assert sc._classify_role("floor tile", policy=policy, force_manipulable=False, force_articulated=False) == (
        None,
        False,
        "policy_excluded",
    )
    assert sc._classify_role("fridge door", policy=policy, force_manipulable=False, force_articulated=True) == (
        "articulated_appliance",
        True,
        "descriptor_articulation_hint",
    )
    assert sc._classify_role("unknown panel", policy=policy, force_manipulable=False, force_articulated=True) == (
        "articulated_furniture",
        True,
        "descriptor_articulation_hint",
    )
    assert sc._classify_role("oven handle", policy=policy, force_manipulable=False, force_articulated=False) == (
        "articulated_appliance",
        True,
        "keyword",
    )
    assert sc._classify_role("cabinet drawer", policy=policy, force_manipulable=False, force_articulated=False) == (
        "articulated_furniture",
        True,
        "keyword",
    )
    assert sc._classify_role("floor mug", policy=policy, force_manipulable=True, force_articulated=False) == (
        "manipulable_object",
        False,
        "descriptor_manipulation_candidate",
    )
    assert sc._classify_role("cup", policy=policy, force_manipulable=False, force_articulated=False) == (
        "manipulable_object",
        False,
        "keyword",
    )
    assert sc._classify_role("unmatched thing", policy=policy, force_manipulable=False, force_articulated=False) == (
        None,
        False,
        "not_selected",
    )
    assert sc._physics_hints("manipulable_object") == {"dynamic": True, "mass_kg": 1.0}
    assert sc._physics_hints("articulated_furniture") == {"dynamic": False, "kinematic": True}
    assert sc._physics_hints("articulated_appliance") == {"dynamic": False, "kinematic": True}
    assert sc._physics_hints("unknown") == {"dynamic": False}


def test_select_swap_candidates_and_payload(monkeypatch):
    descriptor = _descriptor(
        scene_id="scene-swaps",
        capture_id="capture-swaps",
        environment_type_hint="warehouse",
        swap_focus=["industrial_unknown"],
        manipulation_candidates=[
            {"instance_id": "tiny-forced", "label": "custom tiny object"},
        ],
        articulation_hints=[
            {"instance_id": "forced-panel", "name": "service panel"},
        ],
    )
    entries = [
        {"id": "floor", "label": "floor slab", "boundingBox": {"extents": [10, 1, 10]}},
        {"id": "tiny", "label": "tote", "boundingBox": {"extents": [0.01, 0.01, 0.01]}},
        {
            "id": "tiny-forced",
            "label": "tiny part",
            "obb": {"center": [1, 2, 3], "extents": [0.01, 0.01, 0.01]},
            "pointCloudFile": "tiny.ply",
            "reference_crop": "tiny.png",
            "all_crops": ["tiny.png", "", "tiny-2.png"],
        },
        {"id": "cabinet-1", "label": "cabinet drawer", "boundingBox": {"extents": [1, 1, 1]}},
        {"id": "fridge-1", "label": "fridge door", "boundingBox": {"extents": [1, 1, 1]}},
        {"id": "forced-panel", "label": "service panel", "boundingBox": {"extents": [1, 1, 1]}},
        {"id": "unknown", "label": "featureless plane", "boundingBox": {"extents": [1, 1, 1]}},
    ]

    candidates = sc.select_swap_candidates(descriptor=descriptor, object_index_entries=entries)
    by_id = {candidate.object_id: candidate for candidate in candidates}
    assert set(by_id) == {"tiny-forced", "cabinet-1", "fridge-1", "forced-panel"}
    assert by_id["tiny-forced"].sim_role == "manipulable_object"
    assert by_id["tiny-forced"].articulation_reason == "descriptor_manipulation_candidate"
    assert by_id["tiny-forced"].reference_crop == "tiny.png"
    assert by_id["tiny-forced"].all_crops == ["tiny.png", "tiny-2.png"]
    assert by_id["tiny-forced"].point_cloud_file == "tiny.ply"
    assert by_id["cabinet-1"].sim_role == "articulated_furniture"
    assert by_id["fridge-1"].sim_role == "articulated_appliance"
    assert by_id["forced-panel"].sim_role == "articulated_furniture"

    resolved_policy = sc.resolve_policy_config(descriptor=descriptor)
    direct_candidates = sc.select_swap_candidates(
        descriptor=descriptor,
        object_index_entries=entries,
        policy_path="/unused/when/resolved",
        resolved_policy=resolved_policy,
    )
    assert [candidate.object_id for candidate in direct_candidates] == list(by_id)

    monkeypatch.setattr(sc, "utc_now_iso", lambda: "2026-06-20T00:00:00Z")
    payload = sc.build_swap_candidates_payload(descriptor=descriptor, object_index_entries=entries)
    assert payload["schema_version"] == "v1"
    assert payload["scene_id"] == "scene-swaps"
    assert payload["capture_id"] == "capture-swaps"
    assert payload["policy"] == "auto_by_signals"
    assert payload["generated_at"] == "2026-06-20T00:00:00Z"
    assert payload["environment_hints"] == ["warehouse", "industrial_unknown"]
    assert payload["policy_details"]["name"] == "auto_by_signals_default"
    assert [candidate["object_id"] for candidate in payload["candidates"]] == list(by_id)
