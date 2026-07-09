from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline.capture_bridge import CaptureDescriptor
from blueprint_pipeline.ios_manifest import IOSManifest
from blueprint_pipeline import task_targets as tt


def _descriptor(**overrides) -> CaptureDescriptor:
    data = {
        "schema_version": "v1",
        "scene_id": "scene",
        "capture_id": "capture",
        "capture_source": "iphone",
        "capture_tier": "alpha",
        "raw_prefix_uri": "gs://bucket/scenes/scene/captures/capture/raw",
        "frames_index_uri": "gs://bucket/scenes/scene/captures/capture/raw/frames.jsonl",
        "environment_type_hint": "warehouse",
        "swap_focus": [],
        "manipulation_candidates": [],
        "articulation_hints": [],
    }
    data.update(overrides)
    return CaptureDescriptor(**data)


def _manifest(video_uri: str = "") -> IOSManifest:
    return IOSManifest.from_dict({"scene_id": "scene", "video_uri": video_uri})


def _entry(
    object_id: str,
    label: str,
    center: list[float],
    *,
    extents: list[float] | None = None,
    confidence: float = 0.8,
    detections: int = 3,
    crops: list[str] | None = None,
) -> dict[str, object]:
    return {
        "id": object_id,
        "label": label,
        "boundingBox": {
            "center": center,
            "extents": extents or [1.0, 1.0, 1.0],
        },
        "mean_confidence": confidence,
        "confidence": confidence,
        "n_total_detections": detections,
        "n_frame_detections": max(1, detections - 1),
        "reference_crop": crops[0] if crops else "",
        "all_crops": crops or [],
        "mean_box_px": {"width": 100, "height": 50},
    }


def test_task_targets_helper_ranking_filtering_and_grounding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert tt._safe_int("4") == 4
    assert tt._safe_int(object(), 9) == 9
    assert tt._safe_float("1.5") == 1.5
    assert tt._safe_float(object(), 2.5) == 2.5
    assert tt._clamp01(-1) == 0.0
    assert tt._clamp01(2) == 1.0
    assert tt._clamp01(0.25) == 0.25
    assert tt._normalized_text(" Door ", "", None, "Shelf") == "door none shelf"
    assert tt._candidate_id({"uuid": " u "}) == "u"
    assert tt._candidate_id({}) == ""
    assert tt._candidate_label({"category": "Cabinet"}) == "Cabinet"
    assert tt._candidate_label({}) == "object"
    assert tt._semantic_label_bucket("rolling door") == "door"
    assert tt._semantic_label_bucket("") == "object"
    assert tt._mean_box_area_px({"mean_box_px": 12}) == 12.0
    assert tt._mean_box_area_px({"mean_box_px": {"area": 18}}) == 18.0
    assert tt._mean_box_area_px({"mean_box_px": {"width": 3, "height": 4}}) == 12.0
    assert tt._mean_box_area_px({"mean_box_px": [5, 6]}) == 30.0
    assert tt._mean_box_area_px({}) == 0.0

    door = _entry("door-1", "Door", [0, 0, 0], crops=["a.png", "b.png"])
    door_dup = _entry("door-2", "cabinet door", [0.05, 0, 0], confidence=0.7, detections=2, crops=["b.png", "c.png"])
    far_box = _entry("box-1", "Box", [5, 0, 0], extents=[0.5, 0.5, 0.5], detections=1)
    assert tt._object_salience_score(door) > 0
    assert tt._entry_center_extents({"obb": {"center": [1], "extents": [0, 2]}})[1] == [0.02, 2.0, 0.25]
    assert tt._entry_bounds(door)[0] == [-0.5, -0.5, -0.5]
    assert tt._obb_iou3d(door, door_dup) > 0
    assert tt._obb_iou3d(door, far_box) == 0.0
    assert tt._center_distance(door, door_dup) < 0.1
    assert tt._diag_extent(far_box) == pytest.approx(0.8660254)
    assert tt._labels_match(door, door_dup) is True
    assert tt._labels_match(door, far_box) is False

    merged = tt._merge_cluster_entries([door, door_dup])
    assert merged["merged_object_ids"] == ["door-1", "door-2"]
    assert merged["all_crops"] == ["a.png", "b.png", "c.png"]
    assert merged["merge_count"] == 2
    deduped, dedupe_report = tt._dedupe_object_index_entries(
        [door, door_dup, far_box, "bad"],
        iou_threshold=0.1,
        center_ratio=0.3,
    )
    assert dedupe_report["original_count"] == 3
    assert dedupe_report["merged_clusters"] == 1
    assert len(deduped) == 2

    assert tt._parse_caps_mapping({"Door": "2", "bad": "0"}) == {"door": 2}
    assert tt._parse_per_class_caps({"Cabinet": 3}) == ({"cabinet": 3}, "argument_override")
    monkeypatch.setenv("SWAP_PER_CLASS_MAX_COUNTS_JSON", '{"door": 2}')
    assert tt._parse_per_class_caps() == ({"door": 2}, "env_json_override")
    monkeypatch.setenv("SWAP_PER_CLASS_MAX_COUNTS_JSON", "{bad")
    monkeypatch.setenv("SWAP_PER_CLASS_MAX_COUNTS", "drawer:4,box=2,bad")
    assert tt._parse_per_class_caps() == ({"drawer": 4, "box": 2}, "env_kv_override")
    monkeypatch.delenv("SWAP_PER_CLASS_MAX_COUNTS_JSON", raising=False)
    monkeypatch.delenv("SWAP_PER_CLASS_MAX_COUNTS", raising=False)
    assert tt._parse_per_class_caps()[1] == "default"

    residential = _descriptor(environment_type_hint="kitchen", swap_focus=["warehouse"])
    assert tt._descriptor_environment_hints(residential) == ["kitchen", "warehouse"]
    assert tt._resolve_default_class_caps_for_descriptor(residential)[0]["drawer"] == 8
    industrial_caps, industrial_source = tt._resolve_default_class_caps_for_descriptor(_descriptor())
    assert industrial_source == "warehouse"
    assert industrial_caps["box"] == 32
    assert industrial_caps["rack"] == 24
    caps, diagnostics = tt._resolve_per_class_caps(
        descriptor=residential,
        override_caps=None,
        explicit_object_ids={"door-1"},
    )
    assert caps["door"] == 4
    assert diagnostics["explicit_object_id_count"] == 1

    assert tt._entry_has_explicit_object_id({"id": "x", "merged_object_ids": ["door-1"]}, {"door-1"}) is True
    assert tt._entry_has_explicit_object_id({"id": "x"}, set()) is False
    assert tt._entry_detection_counts({"n_frame_detections": "-1", "n_total_detections": 5}) == (0, 5)
    support_disabled, support_disabled_report = tt._apply_detection_support_filter(
        [dict(door)],
        min_frame_detections=0,
        min_total_detections=0,
        explicit_object_ids=set(),
    )
    assert support_disabled == [door]
    assert support_disabled_report["enabled"] is False
    support, support_report = tt._apply_detection_support_filter(
        [
            {"id": "keep", "n_frame_detections": 2, "n_total_detections": 2},
            {"id": "explicit", "n_frame_detections": 0, "n_total_detections": 0},
            {"id": "drop", "n_frame_detections": 0, "n_total_detections": 0},
            {"id": "unknown"},
        ],
        min_frame_detections=1,
        min_total_detections=1,
        explicit_object_ids={"explicit"},
    )
    assert [item["id"] for item in support] == ["keep", "explicit", "unknown"]
    assert support_report["explicit_override_kept_count"] == 1
    capped_disabled, cap_disabled_report = tt._apply_per_class_caps([dict(door)], class_caps={}, explicit_object_ids=set())
    assert capped_disabled == [door]
    assert cap_disabled_report["enabled"] is False
    capped, cap_report = tt._apply_per_class_caps(
        [
            dict(door),
            dict(door_dup),
            dict(far_box),
            {**_entry("door-3", "Door", [1, 0, 0]), "merged_object_ids": ["explicit-door"]},
            {**_entry("door-4", "Door", [2, 0, 0]), "merged_object_ids": ["explicit-door-2"]},
        ],
        class_caps={"door": 1},
        explicit_object_ids={"explicit-door", "explicit-door-2"},
    )
    assert len(capped) == 3
    assert cap_report["dropped_by_label"]["door"] == 2
    assert cap_report["explicit_bypass_kept_count"] == 1

    hint = tt._normalize_hint_entry({"id": "door-1", "label": "Door", "confidence": 1.5, "rationale": "owner", "obb": {"center": [0], "extents": [1]}}, source="s", role="r", default_confidence=0.5)
    assert hint["confidence"] == 1.0
    assert hint["boundingBox"] is not None
    hints = tt._normalize_hint_list([{"id": "x"}, " y ", "", 5], source="s", role="r", default_confidence=0.5)
    assert [item["instance_id"] for item in hints] == ["x", "y"]
    deduped_hints = tt._dedupe_hint_entries(
        [
            {"instance_id": "x", "label": "Door", "confidence": 0.2, "source": "grounding_backend", "reason": "low"},
            {"instance_id": "x", "label": "Door", "confidence": 1.0, "source": "descriptor", "reason": "explicit"},
            {"label": "Box", "confidence": 0.7, "source": "video_inference"},
        ]
    )
    assert deduped_hints[0]["instance_id"] == "x"
    assert deduped_hints[0]["reason"] == "low; explicit"
    desc_manip, desc_artic = tt._descriptor_target_entries(
        _descriptor(
            manipulation_candidates=[{"instance_id": "m", "label": "Door"}],
            articulation_hints=["a"],
        )
    )
    assert desc_manip[0]["instance_id"] == "m"
    assert desc_artic[0]["instance_id"] == "a"

    grounding_payload = {
        "grounded_objects": [
            {
                "object_id": "door-1",
                "label": "Door",
                "confidence": 0.95,
                "boundingBox": {"center": [0, 0, 0], "extents": [1, 1, 1]},
            }
        ]
    }
    assert tt._grounding_objects(grounding_payload, [far_box])[0]["label"] == "Door"
    assert tt._grounding_objects({}, [far_box])[0]["id"] == "box-1"
    by_id, by_label = tt._grounding_lookup(grounding_payload, [])
    assert by_id["door-1"]["label"] == "Door"
    assert by_label["door"][0]["label"] == "Door"
    assert tt._normalize_bbox({"boundingBox": {"center": [1], "extents": [0], "orientationQuaternion": [1]}})["extents"] == [0.02, 0.25, 0.25]
    assert tt._normalize_bbox({}) is None
    enriched = tt._enrich_hints_with_grounding(
        [{"instance_id": "", "label": "Door", "confidence": 0.1}],
        grounding_payload=grounding_payload,
        fallback_entries=[],
    )
    assert enriched[0]["instance_id"] == "door-1"
    label_enriched = tt._enrich_hints_with_grounding(
        [{"instance_id": "door-1", "label": "object", "confidence": 0.1}],
        grounding_payload=grounding_payload,
        fallback_entries=[],
    )
    assert label_enriched[0]["label"] == "Door"

    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(tt, "resolve_gs_uri_to_path", lambda _uri, _root: video)
    video_uri, video_path = tt._resolve_video_uri_and_path(
        descriptor=_descriptor(world_model_video_uri="gs://bucket/video.mp4"),
        manifest=_manifest(),
        storage_root=tmp_path,
    )
    assert video_uri == "gs://bucket/video.mp4"
    assert video_path == video
    local_video = tmp_path / "local.mp4"
    local_video.write_bytes(b"local")
    assert tt._resolve_video_uri_and_path(
        descriptor=_descriptor(world_model_video_uri="", raw_video_uri=str(local_video)),
        manifest=_manifest(),
        storage_root=tmp_path,
    )[1] == local_video


def test_task_targets_external_inference_and_payload_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _descriptor(
        manipulation_candidates=[{"instance_id": "door-1", "label": "Door"}],
        articulation_hints=[{"instance_id": "drawer-1", "label": "Drawer"}],
        raw_video_uri=str(tmp_path / "raw.mp4"),
    )
    video = tmp_path / "raw.mp4"
    video.write_bytes(b"video")
    object_index = tmp_path / "object_index.json"
    object_index.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(tt, "resolve_gs_uri_to_path", lambda _uri, _root: object_index)
    monkeypatch.delenv("TASK_INFERENCE_COMMAND", raising=False)
    assert tt._run_external_video_task_inference(
        descriptor=descriptor,
        object_index_uri="gs://bucket/object_index.json",
        storage_root=tmp_path,
        video_uri=str(video),
        video_path=video,
    )[1]["status"] == "skipped"
    monkeypatch.setenv("TASK_INFERENCE_COMMAND", "runner {OUTPUT_JSON}")
    assert tt._run_external_video_task_inference(
        descriptor=descriptor,
        object_index_uri="gs://bucket/object_index.json",
        storage_root=tmp_path,
        video_uri="missing",
        video_path=None,
    )[1]["reason"] == "video file not found"
    monkeypatch.setattr(tt, "resolve_gs_uri_to_path", lambda _uri, _root: tmp_path / "missing.json")
    assert tt._run_external_video_task_inference(
        descriptor=descriptor,
        object_index_uri="gs://bucket/missing.json",
        storage_root=tmp_path,
        video_uri=str(video),
        video_path=video,
    )[1]["reason"] == "object index path not found"
    monkeypatch.setattr(tt, "resolve_gs_uri_to_path", lambda _uri, _root: object_index)
    monkeypatch.setenv("TASK_INFERENCE_COMMAND", '"unterminated')
    assert tt._run_external_video_task_inference(
        descriptor=descriptor,
        object_index_uri="gs://bucket/object_index.json",
        storage_root=tmp_path,
        video_uri=str(video),
        video_path=video,
    )[1]["status"] == "failed"
    monkeypatch.setenv("TASK_INFERENCE_COMMAND", "   ")
    assert tt._run_external_video_task_inference(
        descriptor=descriptor,
        object_index_uri="gs://bucket/object_index.json",
        storage_root=tmp_path,
        video_uri=str(video),
        video_path=video,
    )[1]["status"] == "skipped"

    monkeypatch.setenv("TASK_INFERENCE_COMMAND", "runner {OUTPUT_JSON}")
    monkeypatch.setenv("TASK_INFERENCE_TIMEOUT_SECONDS", "0")
    monkeypatch.setattr(tt.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("runner", 240)))
    assert "timeout" in tt._run_external_video_task_inference(
        descriptor=descriptor,
        object_index_uri="gs://bucket/object_index.json",
        storage_root=tmp_path,
        video_uri=str(video),
        video_path=video,
    )[1]["reason"]
    monkeypatch.setattr(tt.subprocess, "run", lambda command, **_kwargs: subprocess.CompletedProcess(command, 1, stdout="out", stderr="err"))
    assert tt._run_external_video_task_inference(
        descriptor=descriptor,
        object_index_uri="gs://bucket/object_index.json",
        storage_root=tmp_path,
        video_uri=str(video),
        video_path=video,
    )[1]["return_code"] == 1
    monkeypatch.setattr(tt.subprocess, "run", lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""))
    assert tt._run_external_video_task_inference(
        descriptor=descriptor,
        object_index_uri="gs://bucket/object_index.json",
        storage_root=tmp_path,
        video_uri=str(video),
        video_path=video,
    )[1]["reason"] == "output JSON missing"

    def run_writes_list(command, **_kwargs):
        Path(command[-1]).write_text("[]", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(tt.subprocess, "run", run_writes_list)
    assert tt._run_external_video_task_inference(
        descriptor=descriptor,
        object_index_uri="gs://bucket/object_index.json",
        storage_root=tmp_path,
        video_uri=str(video),
        video_path=video,
    )[1]["reason"] == "unexpected output type: list"

    def run_writes_mapping(command, **_kwargs):
        Path(command[-1]).write_text('{"manipulation_candidates": [{"instance_id": "door-1"}]}', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(tt.subprocess, "run", run_writes_mapping)
    payload, report = tt._run_external_video_task_inference(
        descriptor=descriptor,
        object_index_uri="gs://bucket/object_index.json",
        storage_root=tmp_path,
        video_uri=str(video),
        video_path=video,
    )
    assert payload["manipulation_candidates"][0]["instance_id"] == "door-1"
    assert report["status"] == "ok"

    grounding_payload = {
        "backend": "object_index_stage",
        "backend_status": "ok",
        "backend_report": {"status": "ok", "backend": "object_index_stage"},
        "grounded_objects": [
            {"object_id": "door-1", "label": "Door", "confidence": 0.9, "boundingBox": {"center": [0, 0, 0], "extents": [1, 1, 1]}},
            {"object_id": "nav-1", "label": "Shelf", "confidence": 0.7, "boundingBox": {"center": [2, 0, 0], "extents": [1, 1, 1]}},
        ],
        "manipulation_candidates": [{"instance_id": "door-1", "label": "Door", "confidence": 0.8}],
        "articulation_hints": [{"instance_id": "drawer-1", "label": "Drawer", "confidence": 0.9}],
        "navigation_hints": [{"instance_id": "nav-1", "label": "Shelf"}],
        "tasks": [{"task_id": "open-door"}],
    }
    inferred = tt.infer_task_targets(
        descriptor=descriptor,
        manifest=_manifest(str(video)),
        object_index_entries=[
            _entry("door-1", "Door", [0, 0, 0]),
            _entry("door-dup", "cabinet door", [0.05, 0, 0]),
        ],
        object_index_uri="gs://bucket/object_index.json",
        storage_root=tmp_path,
        grounding_payload=grounding_payload,
        max_targets=2,
    )
    assert inferred["inference_mode"] == "descriptor+external"
    assert inferred["target_object_ids"] == ["door-1"]
    assert inferred["articulation_required_ids"] == ["drawer-1"]
    assert inferred["navigation_hints"][0]["instance_id"] == "nav-1"
    assert inferred["tasks"][0]["task_id"] == "open-door"
    empty = tt.infer_task_targets(
        descriptor=_descriptor(),
        manifest=_manifest(),
        object_index_entries=[],
        object_index_uri="gs://bucket/object_index.json",
        storage_root=tmp_path,
        grounding_payload=None,
        max_targets=0,
    )
    assert empty["inference_mode"] == "empty"
    assert empty["video_analysis"]["external_inference"]["reason"] == "no_grounding_backend"


def test_task_aware_swap_candidate_selection_and_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _descriptor(
        manipulation_candidates=[{"instance_id": "door-1", "label": "Door"}],
        articulation_hints=[{"instance_id": "cabinet-1", "label": "Cabinet"}],
        environment_type_hint="kitchen",
    )
    task_targets = {
        "inference_mode": "external",
        "manipulation_candidates": [{"instance_id": "box-1", "label": "Box"}],
        "articulation_hints": ["drawer-1"],
        "explicit_target_object_ids": ["box-1"],
        "explicit_articulation_required_ids": ["drawer-1"],
        "explicit_target_labels": ["box"],
        "explicit_articulation_labels": ["drawer"],
    }
    merged_descriptor = tt._merge_descriptor_with_task_targets(descriptor, task_targets)
    assert {item["instance_id"] for item in merged_descriptor.manipulation_candidates} >= {"door-1", "box-1"}
    assert tt._merge_descriptor_with_task_targets(descriptor, None) is descriptor
    assert tt._normalize_selection_mode("bad") == "hybrid"
    assert tt._normalize_selection_mode("explicit_only") == "explicit_only"
    explicit_sets = tt._extract_explicit_sets(descriptor, task_targets)
    assert explicit_sets["task_obj_ids"] >= {"box-1", "drawer-1"}
    fallback_sets = tt._extract_explicit_sets(
        _descriptor(),
        {
            "inference_mode": "external",
            "target_object_ids": ["target-1"],
            "articulation_required_ids": ["artic-1"],
            "manipulation_candidates": ["string-id"],
        },
    )
    assert fallback_sets["task_obj_ids"] >= {"target-1", "artic-1", "string-id"}
    assert tt._candidate_rank_score({"articulation": {"required": True}}, _entry("x", "Door", [0, 0, 0]), explicit=True) > 3
    assert tt._candidate_rank_score({}, None, explicit=False) == 0.5
    no_cap, no_cap_summary = tt._apply_ranked_cap([{"articulation": {"required": True}}], max_candidates=0)
    assert no_cap == [{"articulation": {"required": True}}]
    assert no_cap_summary["max_candidates"] == 0
    selected, cap_summary = tt._apply_ranked_cap(
        [
            {"object_id": "required", "articulation": {"required": True}},
            {"object_id": "optional"},
        ],
        max_candidates=1,
    )
    assert selected[0]["object_id"] == "required"
    assert cap_summary["dropped_count"] == 1

    entries = [
        _entry("door-1", "Door", [0, 0, 0], detections=3),
        _entry("box-1", "Box", [2, 0, 0], detections=1),
        _entry("policy-1", "Shelf", [4, 0, 0], detections=3),
        _entry("drawer-1", "Drawer", [6, 0, 0], detections=1),
    ]

    def fake_swap_payload(*, descriptor: CaptureDescriptor, object_index_entries: list[dict[str, object]], policy_path: str | None = None):
        del descriptor, policy_path
        return {
            "candidates": [
                {
                    "object_id": str(entry["id"]),
                    "label": str(entry["label"]),
                    "articulation": {"required": str(entry["id"]) in {"drawer-1", "door-1"}},
                }
                for entry in object_index_entries
            ]
        }

    monkeypatch.setattr(tt, "build_swap_candidates_payload", fake_swap_payload)
    monkeypatch.setenv("SWAP_MIN_TOTAL_DETECTIONS", "2")
    payload = tt.build_task_aware_swap_candidates_payload(
        descriptor=descriptor,
        object_index_entries=entries,
        task_targets=task_targets,
        selection_mode="hybrid",
        max_candidates=3,
        per_class_caps={"door": 1, "box": 1, "drawer": 1},
    )
    assert payload["selection_mode"] == "hybrid"
    assert payload["task_targets_attached"] is True
    assert "box-1" in payload["task_target_object_ids"]
    assert payload["index_preprocessing"]["detection_support"]["explicit_override_kept_count"] >= 1
    assert payload["selection_summary"]["explicit_count"] >= 1
    assert payload["industrial_task_target_context_summary"]["status"] == "detected"
    assert any(
        "industrial_task_target_context" in candidate
        for candidate in payload["candidates"]
    )

    high_repetition_entries = [
        _entry(f"tote-{index}", "Tote", [float(index) * 2.0, 0, 0], detections=3)
        for index in range(40)
    ]
    industrial_payload = tt.build_task_aware_swap_candidates_payload(
        descriptor=_descriptor(environment_type_hint="warehouse"),
        object_index_entries=high_repetition_entries,
        task_targets={},
        selection_mode="policy_only",
        max_candidates=100,
        per_class_caps=None,
    )
    assert industrial_payload["index_preprocessing"]["class_caps"]["caps"]["box"] == 32
    assert industrial_payload["index_preprocessing"]["class_caps"]["kept_by_label"]["box"] == 32
    assert len(industrial_payload["candidates"]) == 32

    explicit_only = tt.build_task_aware_swap_candidates_payload(
        descriptor=descriptor,
        object_index_entries=entries,
        task_targets=task_targets,
        selection_mode="explicit_only",
        max_candidates=10,
        per_class_caps={},
    )
    assert all(item["selection"]["explicit"] for item in explicit_only["candidates"])
    policy_only = tt.build_task_aware_swap_candidates_payload(
        descriptor=descriptor,
        object_index_entries=entries,
        task_targets=task_targets,
        selection_mode="policy_only",
        max_candidates=2,
        per_class_caps={},
    )
    assert all(not item["selection"]["explicit"] for item in policy_only["candidates"])
    hybrid_zero = tt.build_task_aware_swap_candidates_payload(
        descriptor=descriptor,
        object_index_entries=entries,
        task_targets=task_targets,
        selection_mode="hybrid",
        max_candidates=0,
        per_class_caps={},
    )
    assert all(item["selection"]["explicit"] for item in hybrid_zero["candidates"])

    monkeypatch.setattr(tt, "build_swap_candidates_payload", lambda **_kwargs: {"candidates": "bad"})
    with pytest.raises(ValueError, match="invalid swap candidates payload"):
        tt.build_task_aware_swap_candidates_payload(
            descriptor=descriptor,
            object_index_entries=entries,
            task_targets=task_targets,
        )

    output = tmp_path / "task_targets.json"
    tt.write_task_targets(output, {"status": "ok"})
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "ok"


def test_task_targets_remaining_branch_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert tt._labels_match({"label": "Door"}, {"name": "Door"}) is True
    no_reference = tt._merge_cluster_entries(
        [
            {
                "id": "crop-only",
                "label": "Box",
                "boundingBox": {"center": [0, 0, 0], "extents": [1, 1, 1]},
                "all_crops": ["crop-only.png"],
            }
        ]
    )
    assert no_reference["reference_crop"] == "crop-only.png"

    original_salience_score = tt._object_salience_score
    salience_values = iter([1.0, 1.0, 0.1, 0.9, 0.1, 0.9, 0.9])
    monkeypatch.setattr(tt, "_object_salience_score", lambda _entry: next(salience_values))
    updated_rep, _report = tt._dedupe_object_index_entries(
        [
            _entry("first", "Door", [0, 0, 0]),
            _entry("second", "Door", [0.01, 0, 0]),
        ],
        iou_threshold=0.1,
        center_ratio=1.0,
    )
    assert updated_rep[0]["id"] == "second"
    monkeypatch.setattr(tt, "_object_salience_score", original_salience_score)

    monkeypatch.setenv("SWAP_PER_CLASS_MAX_COUNTS_JSON", "{bad")
    monkeypatch.setenv("SWAP_PER_CLASS_MAX_COUNTS", " , door:2")
    assert tt._parse_per_class_caps() == ({"door": 2}, "env_kv_override")
    monkeypatch.delenv("SWAP_PER_CLASS_MAX_COUNTS_JSON", raising=False)
    monkeypatch.delenv("SWAP_PER_CLASS_MAX_COUNTS", raising=False)

    assert tt._normalize_bbox({"obb": {"center": [1, 2, 3], "extents": [1, 1, 1]}})["center"] == [1.0, 2.0, 3.0]
    assert tt._normalize_bbox({"boundingBox": {"center": [1, 2, 3]}}) is None
    missing_uri, missing_path = tt._resolve_video_uri_and_path(
        descriptor=_descriptor(world_model_video_uri="", privacy_processed_video_uri="", raw_video_uri=""),
        manifest=_manifest(str(tmp_path / "missing.mp4")),
        storage_root=tmp_path,
    )
    assert missing_uri == str(tmp_path / "missing.mp4")
    assert missing_path is None

    object_index = tmp_path / "object_index.json"
    object_index.write_text("{}", encoding="utf-8")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(tt, "resolve_gs_uri_to_path", lambda _uri, _root: object_index)
    monkeypatch.setenv("TASK_INFERENCE_COMMAND", "#")
    original_shlex_split = tt.shlex.split
    monkeypatch.setattr(tt.shlex, "split", lambda _rendered: [])
    assert tt._run_external_video_task_inference(
        descriptor=_descriptor(),
        object_index_uri="gs://bucket/object_index.json",
        storage_root=tmp_path,
        video_uri=str(video),
        video_path=video,
    )[1]["reason"] == "empty command"
    monkeypatch.setattr(tt.shlex, "split", original_shlex_split)

    monkeypatch.setenv("TASK_INFERENCE_COMMAND", "runner {OUTPUT_JSON}")
    monkeypatch.setattr(tt.subprocess, "run", lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""))
    monkeypatch.setattr(tt.Path, "unlink", lambda self, missing_ok=False: (_ for _ in ()).throw(RuntimeError("unlink failed")))
    assert tt._run_external_video_task_inference(
        descriptor=_descriptor(),
        object_index_uri="gs://bucket/object_index.json",
        storage_root=tmp_path,
        video_uri=str(video),
        video_path=video,
    )[1]["reason"] == "output JSON missing"

    descriptor_only = tt.infer_task_targets(
        descriptor=_descriptor(manipulation_candidates=[{"instance_id": "only", "label": "Door"}]),
        manifest=_manifest(),
        object_index_entries=[],
        object_index_uri="gs://bucket/object_index.json",
        storage_root=tmp_path,
        grounding_payload=None,
    )
    assert descriptor_only["inference_mode"] == "descriptor_only"

    merged = tt._merge_descriptor_with_task_targets(
        _descriptor(),
        {
            "explicit_target_object_ids": ["new-target"],
            "explicit_articulation_required_ids": ["new-artic"],
        },
    )
    assert {"new-target"} == {item["instance_id"] for item in merged.manipulation_candidates}
    assert {"new-artic"} == {item["instance_id"] for item in merged.articulation_hints}
    explicit_sets = tt._extract_explicit_sets(
        _descriptor(manipulation_candidates=["bad", {"label": "Door"}]),
        {"inference_mode": "heuristic", "target_object_ids": ["ignored"]},
    )
    assert "ignored" not in explicit_sets["task_obj_ids"]
    assert "door" in explicit_sets["descriptor_labels"]

    original_dedupe = tt._dedupe_object_index_entries
    monkeypatch.setattr(
        tt,
        "_dedupe_object_index_entries",
        lambda entries, **_kwargs: (list(entries), {"deduped_count": len(entries)}),
    )
    monkeypatch.setattr(
        tt,
        "_apply_detection_support_filter",
        lambda entries, **_kwargs: (list(entries), {"enabled": True}),
    )
    monkeypatch.setattr(
        tt,
        "_apply_per_class_caps",
        lambda entries, **_kwargs: ([*entries, "bad-capped-entry"], {"enabled": True}),
    )
    monkeypatch.setattr(
        tt,
        "build_swap_candidates_payload",
        lambda **_kwargs: {
            "candidates": [
                "bad-candidate",
                {"object_id": "merged-source", "label": "Valve"},
            ]
        },
    )
    explicit_payload = tt.build_task_aware_swap_candidates_payload(
        descriptor=_descriptor(),
        object_index_entries=[
            {**_entry("merged-source", "Valve", [0, 0, 0]), "merged_object_ids": ["external-merged"]},
        ],
        task_targets={"explicit_target_object_ids": ["external-merged"]},
        selection_mode="explicit_only",
        max_candidates=5,
        per_class_caps={},
    )
    assert explicit_payload["candidates"][0]["selection"]["selected_by"] == "explicit"
    monkeypatch.setattr(tt, "_dedupe_object_index_entries", original_dedupe)
