"""A retry retains the last attempt's reviewer-accepted raw edits for byte-identical inputs."""

import hashlib
import json
import os
from pathlib import Path

from blueprint_pipeline import semantic_teacher_candidate_discovery as discovery
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

BACKEND = {"execution": {"model_snapshot": "gpt-image-2.5-sunburst-2026-09-08"},
           "registry_entry": {"backend_id": "openai_gpt_image_2_5_sunburst_2026_09_08_semantic_teacher"}}
CAMERAS = ("source-01", "source-02", "source-03", "source-04")


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _record(path: Path, relative: str) -> dict:
    data = path.read_bytes()
    return {"relative_path": relative, "size_bytes": len(data), "sha256": _digest(data)}


def _seal(value: dict, field: str) -> dict:
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _request(cameras, inputs, masks, *, backend=BACKEND):
    frames = [{"frame_index": i, "camera_id": c, "frame_role": "semantic_edit",
               "input_rgb": {"relative_path": f"in/{c}.png", "sha256": inputs[c], "size_bytes": 10},
               "edit_mask": {"relative_path": f"mask/{c}.png", "sha256": masks[c], "size_bytes": 10}}
              for i, c in enumerate(cameras)]
    return _seal({"schema_version": "semantic_teacher_image_edit_runtime_request.v1", "backend": backend,
                  "tasks": [{"task_id": "remove", "frames": frames}]}, "request_digest")


def _run(root: Path, name: str, cameras, inputs, masks, request, *, colors):
    """Write a sealed edit result plus raw candidate PNG-ish bytes under ``root/name``."""
    frames = []
    for i, c in enumerate(cameras):
        raw = f"{name}:{c}:{colors[c]}".encode() * 8
        path = root / name / "tasks/remove" / f"{i:05d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        frames.append({"frame_index": i, "camera_id": c, "terminal_state": "completed_unreviewed_candidate",
                       "source_rgb_sha256": inputs[c], "edit_mask_sha256": masks[c],
                       "semantic_teacher_frame": _record(path, f"tasks/remove/{i:05d}.png"),
                       "computed_editor_cost_usd": 0.07, "provider_usage": {"output_image_tokens": 1000}})
    result = _seal({"schema_version": "semantic_teacher_image_edit_runtime_result.v1",
                    "status": "completed_unreviewed_semantic_teacher_candidates",
                    "source_runtime_request_digest": request["request_digest"],
                    "model_snapshot": BACKEND["execution"]["model_snapshot"],
                    "backend_id": BACKEND["registry_entry"]["backend_id"],
                    "tasks": [{"task_id": "remove", "frames": frames}]}, "result_digest")
    return result, {c: frames[i]["semantic_teacher_frame"]["sha256"] for i, c in enumerate(cameras)}


def _review(sealed: dict, verdicts: dict) -> dict:
    rows = []
    for camera, sha in sealed.items():
        accepted = verdicts.get(camera, False)
        rows.append({"camera_id": camera, "frame_sha256": sha, "decision": "accepted" if accepted else "rejected",
                     "source_object_absent": True, "repair_is_locally_plausible": accepted,
                     "preserves_non_target_content": accepted, "orientation_is_upright": True})
    return _seal({"schema_version": "task_evaluation_artifixer_ai_visual_review_execution.v1", "status": "completed",
                  "decision": "accepted" if all(verdicts.get(c) for c in sealed) else "rejected",
                  "frames": rows}, "execution_digest")


def _workspace(root: Path, name: str, cameras, inputs, masks, *, review1, repair=None, review2=None, backend=BACKEND, age=0):
    runtime = root / name / discovery.RUNTIME
    request = _request(cameras, inputs, masks, backend=backend)
    _write(runtime / discovery.REQUEST, request)
    result, raw = _run(runtime, "semantic_teacher_output", cameras, inputs, masks, request, colors={c: "base" for c in cameras})
    _write(runtime / discovery.RESULT, result)
    # The locality seal composites each raw edit; sealed digests differ from the raw ones.
    sealed = {c: _digest(f"sealed:{name}:{c}".encode()) for c in cameras}
    _write(runtime / discovery.SEAL, _seal({"schema_version": "task_evaluation_artifixer_semantic_locality_seal.v1",
                                            "frames": [{"camera_id": c, "sealed_semantic_teacher": {"sha256": sealed[c]}} for c in cameras]},
                                           "receipt_digest"))
    _write(runtime / discovery.REVIEW_1, _review(sealed, review1))
    if repair:
        repair_request = _request(repair, inputs, masks, backend=backend)
        _write(runtime / discovery.REPAIR_REQUEST, repair_request)
        repair_result, _ = _run(runtime / "semantic_target_recovery", "selective_semantic_repair_output", repair, inputs, masks,
                                repair_request, colors={c: "repaired" for c in repair})
        _write(runtime / discovery.REPAIR_RESULT, repair_result)
        merged_sealed = {c: (_digest(f"merged:{name}:{c}".encode()) if c in repair else sealed[c]) for c in cameras}
        _write(runtime / discovery.MERGE, _seal({"schema_version": "task_evaluation_artifixer_selective_repair_merge.v1",
            "frame_inventory": [{"camera_id": c, "sha256": merged_sealed[c],
                                 "role": "selectively_repaired_semantic_frame" if c in repair else "reused_first_pass_semantic_frame"}
                                for c in cameras]}, "merge_digest"))
        _write(runtime / discovery.REVIEW_2, _review(merged_sealed, review2 or {}))
    stamp = 5_000_000 - age
    os.utime(root / name, (stamp, stamp))
    return request


def _render(inputs):
    return {"derived_frames": [{"camera_id": c, "digest": d} for c, d in inputs.items()]}


def test_retains_only_frames_the_last_review_accepted_with_a_bound_digest_chain(tmp_path):
    inputs = {c: _digest(f"rgb:{c}".encode()) for c in CAMERAS}
    masks = {c: _digest(f"mask:{c}".encode()) for c in CAMERAS}
    root = tmp_path / "semantic-pretraining"
    # Older workspace: everything accepted, but it is older than the newest match.
    _workspace(root, "a" * 64, CAMERAS, inputs, masks, review1={c: True for c in CAMERAS}, age=7200)
    # Newest matching workspace: review 1 rejected 01 and 03; 03 was repaired and accepted in review 2,
    # 01 stayed rejected. Review 2 is the authority.
    _workspace(root, "b" * 64, CAMERAS, inputs, masks,
               review1={"source-01": False, "source-02": True, "source-03": False, "source-04": True},
               repair=("source-01", "source-03"),
               review2={"source-01": False, "source-02": True, "source-03": True, "source-04": True}, age=60)
    # A workspace with different inputs (another scene) must never be used.
    other = {c: _digest(f"other:{c}".encode()) for c in CAMERAS}
    _workspace(root, "c" * 64, CAMERAS, other, masks, review1={c: True for c in CAMERAS}, age=10)
    current_request = _write(tmp_path / "current" / "request.json", _request(CAMERAS, inputs, masks))

    outcome = discovery.discover_retained_candidates(
        runtime_request_path=current_request, render=_render(inputs), workspace_root=root, output_root=tmp_path / "disc")

    assert outcome["status"] == "retained_from_previous_attempt"
    assert outcome["source_workspace"].endswith("b" * 64)
    assert outcome["digest_chain"] == "repair_merge"
    assert outcome["retained_camera_ids"] == ["source-02", "source-03", "source-04"]
    assert [row["camera_id"] for row in outcome["candidates"]] == ["source-02", "source-04", "source-03"]
    assert outcome["skipped"] == [{"camera_id": "source-01", "reason": "not_accepted_by_last_review"}]
    receipt = json.loads((tmp_path / "disc" / "discovery.json").read_text())
    assert receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")
    assert [e["status"] for e in receipt["examined"]] == ["not_applicable", "selected"]
    # The repaired camera's candidate bytes come from the repair output, the others from the first pass.
    by_camera = {row["camera_id"]: Path(row["candidate"]["path"]).read_bytes() for row in outcome["candidates"]}
    assert by_camera["source-03"].startswith(b"selective_semantic_repair_output:source-03:repaired")
    assert by_camera["source-02"].startswith(b"semantic_teacher_output:source-02:base")


def test_nothing_is_retained_without_a_matching_backend_review_or_digest(tmp_path):
    inputs = {c: _digest(f"rgb:{c}".encode()) for c in CAMERAS}
    masks = {c: _digest(f"mask:{c}".encode()) for c in CAMERAS}
    root = tmp_path / "semantic-pretraining"
    other_backend = {"execution": {"model_snapshot": "gpt-image-2-2026-04-21"}, "registry_entry": {"backend_id": "old"}}
    _workspace(root, "d" * 64, CAMERAS, inputs, masks, review1={c: True for c in CAMERAS}, backend=other_backend, age=10)
    tampered = _workspace(root, "e" * 64, CAMERAS, inputs, masks, review1={c: True for c in CAMERAS}, age=20)
    review_path = root / ("e" * 64) / discovery.RUNTIME / discovery.REVIEW_1
    review = json.loads(review_path.read_text())
    review["frames"][0]["frame_sha256"] = _digest(b"tampered")  # breaks the seal
    review_path.write_text(json.dumps(review) + "\n")
    current_request = _write(tmp_path / "current" / "request.json", _request(CAMERAS, inputs, masks))
    outcome = discovery.discover_retained_candidates(
        runtime_request_path=current_request, render=_render(inputs), workspace_root=root, output_root=tmp_path / "disc")
    assert outcome["status"] == "no_matching_workspace" and outcome["candidates"] == []
    assert tampered["request_digest"]
    assert discovery.discovery_enabled({}) is True
    assert discovery.discovery_enabled({discovery.DISCOVERY_ENV: "0"}) is False
    missing = discovery.discover_retained_candidates(
        runtime_request_path=current_request, render=_render(inputs), workspace_root=tmp_path / "absent", output_root=tmp_path / "disc2")
    assert missing["status"] == "discovery_root_unavailable" and missing["candidates"] == []
