"""Verify native per-cell controls and expose their retained evidence privately."""
from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json

CONTROL_IDS = ("zero_action_negative", "deterministic_scripted_positive")
WARNINGS = {
    "configured_controls_pending": "Controls pending — results are unqualified.",
    "controls_failed": "Required controls failed — results are unqualified.",
    "controls_verified_development_only": "Controls verified for this simulation matrix — development-only results remain unqualified.",
}


def control_summary(controls: list[Mapping[str, Any]], episodes: list[Mapping[str, Any]]) -> dict[str, int]:
    pairs: dict[str, list[Mapping[str, Any]]] = {}
    for row in controls:
        pairs.setdefault(str(row["cell_id"]), []).append(row)
    learned: dict[str, set[tuple[str, int]]] = {}
    for row in episodes:
        if isinstance(row.get("seed"), int) and not isinstance(row["seed"], bool):
            learned.setdefault(str(row.get("candidate_id")), set()).add((str(row.get("cell_id")), row["seed"]))
    aligned = learned.get("pi05_droid", set()) & learned.get("groot_n17_droid", set())
    verified = sum(len(rows) == 2 and {row["control_id"] for row in rows} == set(CONTROL_IDS)
                   and len({row["seed"] for row in rows}) == 1
                   and (cell, rows[0]["seed"]) in aligned
                   and all(row["terminal_state"] == "completed" and row["control_passed"]
                           and not row["evidence_gaps"] for row in rows)
                   for cell, rows in pairs.items())
    return {"expected_count": 20, "recorded_count": len(controls),
            "completed_count": sum(row["terminal_state"] == "completed" for row in controls),
            "passed_count": sum(row["control_passed"] for row in controls),
            "verified_cell_count": verified}


def materialize_control_delivery(*, result, evidence_root: Path, delivery_root: Path,
                                 public_artifacts, add_artifact, write_immutable, write_zip, error_factory) -> dict[str, Any]:
    rows = result.get("controls")
    if rows is None:
        return {}
    if (not isinstance(rows, list) or len(rows) > 20
            or result.get("control_episode_count") != len(rows)):
        raise error_factory("policy_canary_control_inventory_invalid")

    def artifact(record, *, prefix=""):
        if not isinstance(record, Mapping):
            return None
        relative = (Path(prefix) / str(record.get("relative_path") or "")).as_posix()
        matches = [row for row in public_artifacts if row.get("relative_path") == relative
                   and row.get("digest") == record.get("sha256")
                   and row.get("size_bytes") == record.get("size_bytes")]
        if len(matches) != 1:
            return None
        return {key: matches[0][key] for key in ("artifact_id", "digest", "size_bytes", "role")}

    def compact(value):
        return {key: value[key] for key in ("artifact_id", "digest", "size_bytes")} if value else None

    def reopened(record, *, prefix=""):
        bound = artifact(record, prefix=prefix)
        if bound is None:
            raise error_factory("policy_canary_control_artifact_not_delivered")
        relative = Path(prefix) / record["relative_path"]
        if relative.is_absolute() or ".." in relative.parts:
            raise error_factory("policy_canary_control_artifact_path_invalid")
        return evidence_root / relative, bound

    controls, seen, archives = [], set(), {}
    from .episode_visual_evidence import validate_multicamera_frame_manifest
    for row in rows:
        if not isinstance(row, Mapping):
            raise error_factory("policy_canary_control_envelope_invalid")
        cell, control, seed = row.get("cell_id"), row.get("control_id"), row.get("seed")
        if (not isinstance(cell, str) or not cell or control not in CONTROL_IDS
                or type(seed) is not int or (cell, control) in seen):
            raise error_factory("policy_canary_control_identity_invalid")
        seen.add((cell, control))
        receipt = row.get("receipt")
        if (not isinstance(receipt, Mapping) or receipt.get("schema_version") != "adp_task_control_episode.v1"
                or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
                or receipt.get("control_id") != control or receipt.get("episode_id") != f"{cell}-{control}"
                or receipt.get("candidate_policy_queried") is not False
                or receipt.get("grader_authority") != "deterministic_simulator_state"
                or receipt.get("caller_asserted_success_accepted") is not False):
            raise error_factory("policy_canary_control_receipt_invalid")
        parent_path, parent_artifact = reopened(row.get("cell_receipt_artifact"))
        parent = json.loads(parent_path.read_text())
        prefix = str(row.get("evidence_root") or "")
        if (parent.get("schema_version") != "policy_canary_cell_controls.v1"
                or parent.get("receipt_digest") != canonical_digest(parent, digest_field="receipt_digest")
                or parent.get("cell_id") != cell or parent.get("seed") != seed
                or parent.get("candidate_policy_queried") is not False
                or parent.get("task_spec_digest") != receipt.get("task_spec_digest")
                or any(re.fullmatch(r"sha256:[0-9a-f]{64}", str(parent.get(name) or "")) is None
                       for name in ("scene_plan_digest", "task_spec_digest", "camera_binding_digest"))
                or parent.get("files") != row.get("evidence_files")
                or parent_path.parent != evidence_root / prefix
                or sum(candidate == receipt for candidate in parent.get("controls", [])) != 1):
            raise error_factory("policy_canary_control_cell_binding_invalid")
        files = row.get("evidence_files")
        if not isinstance(files, list) or not files:
            raise error_factory("policy_canary_control_files_missing")
        bound_files = [reopened(record, prefix=prefix) for record in files]
        native_receipts = [(path, bound) for path, bound in bound_files
                           if path.name == f"adp_task_control_episode.{control}.json"]
        if len(native_receipts) != 1 or json.loads(native_receipts[0][0].read_text()) != receipt:
            raise error_factory("policy_canary_control_original_receipt_missing")
        receipt_artifact = compact(native_receipts[0][1])
        if prefix not in archives:
            archive_path = delivery_root / "controls" / f"cell-{len(archives):02d}.zip"
            write_zip(archive_path, [(str(path.relative_to(parent_path.parent)), path) for path, _ in bound_files]
                      + [(parent_path.name, parent_path)])
            archives[prefix] = add_artifact(role="control_cell_archive", path=archive_path, artifact_root=delivery_root)
        videos, gaps = {}, []
        for camera, record in ((receipt.get("visual_evidence") or {}).get("videos") or {}).items():
            video_path = native_receipts[0][0].parent / str(record.get("relative_path") or "")
            matches = [bound for path, bound in bound_files if path == video_path and bound["digest"] == record.get("sha256", record.get("digest"))
                       and bound["size_bytes"] == record.get("size_bytes")]
            if len(matches) == 1:
                videos[str(camera)] = compact(matches[0])
            else:
                gaps.append("video:" + str(camera))
        if (receipt.get("visual_evidence") or {}).get("status") != "complete":
            gaps.append("visual_evidence_incomplete")
        if set(videos) != {"external", "wrist", "overview"}:
            gaps.append("review_videos_missing")
        manifests = [record for record in receipt.get("media_artifacts", [])
                     if record.get("role") == "multicamera_observation_frame_manifest"]
        manifest_artifact = None
        try:
            if len(manifests) != 1:
                raise ValueError("manifest_count")
            manifest_path = native_receipts[0][0].parent / manifests[0]["relative_path"]
            manifest_matches = [bound for path, bound in bound_files if path == manifest_path
                                and bound["digest"] == manifests[0].get("sha256")
                                and bound["size_bytes"] == manifests[0].get("size_bytes")]
            if len(manifest_matches) != 1:
                raise ValueError("manifest_binding")
            manifest = json.loads(manifest_path.read_text())
            validate_multicamera_frame_manifest(manifest, output_dir=native_receipts[0][0].parent, verify_files=True)
            if (manifest.get("frame_manifest_digest") != receipt["visual_evidence"].get("frame_manifest_digest")
                    or set(manifest.get("required_camera_ids", [])) != {"external", "wrist", "overview"}
                    or manifest.get("review_only_camera_ids") != ["overview"]):
                raise ValueError("manifest_camera_binding")
            manifest_artifact = compact(manifest_matches[0])
        except (OSError, ValueError, KeyError, TypeError):
            gaps.append("lossless_frame_manifest")
        for trace, field, key in (("state_trace", "state_trace_digest", "samples"),
                                  ("action_trace", "action_trace_digest", "actions")):
            if (not isinstance(receipt.get(trace), list)
                    or receipt.get(field) != canonical_digest({key: receipt.get(trace)})):
                gaps.append(trace)
        score = dict(receipt.get("score") or {})
        complete = score.get("status") == "scored" and not gaps
        expected = score.get("task_succeeded") is (control == "deterministic_scripted_positive")
        if control == "zero_action_negative":
            expected = expected and score.get("outcome") == "never_moved"
        passed = (complete and expected and receipt.get("control_passed") is True
                  and row.get("control_passed") is True and not receipt.get("blockers"))
        downloads = [{**compact(archives[prefix]), "role": "control_cell_archive"}]
        if manifest_artifact:
            downloads.append({**manifest_artifact, "role": "frame_manifest"})
        for trace in ("state_trace", "action_trace"):
            trace_path = delivery_root / "controls" / f"control-{len(controls):02d}-{trace}.json"
            write_immutable(trace_path, (canonical_json({"receipt_digest": receipt["receipt_digest"], trace: receipt.get(trace)}) + "\n").encode())
            downloads.append({**add_artifact(role="control_"+trace, path=trace_path, artifact_root=delivery_root), "role": trace})
        controls.append({"episode_id": receipt["episode_id"], "control_id": control, "cell_id": cell,
            "seed": seed, "terminal_state": "completed" if complete else "blocked", "control_passed": passed,
            "receipt_digest": receipt["receipt_digest"], "receipt": receipt_artifact,
            "cell_receipt": compact(parent_artifact), "videos": videos,
            "artifacts": downloads,
            "score": {key: score[key] for key in ("status", "task_succeeded", "outcome", "failed_criteria", "measurements", "blockers") if key in score},
            "evidence_gaps": sorted(set(gaps))})
    summary = control_summary(controls, result.get("episodes") or [])
    gate = result.get("controls_gate") or {}
    verified = (summary["recorded_count"] == summary["passed_count"] == 20 and summary["verified_cell_count"] == 10
                and gate.get("status") == "passed" and gate.get("required_control_episode_count") == 20
                and gate.get("candidate_policies_loaded_during_controls") is False)
    status = "controls_verified_development_only" if verified else "controls_failed"
    output = {"controls": controls, "controls_summary": summary, "scene_controls_status": status,
              "warning": WARNINGS[status], "strict_gate_blockers": list(result.get("strict_gate_blockers") or [])}
    for name, fields in {
        "controls_gate": ("status", "required_control_episode_count", "candidate_policies_loaded_during_controls"),
        "strict_paired_gate": ("schema_version", "status", "blockers", "deterministic_success_required"),
        "paired_delivery": ("schema_version", "status", "archive_sha256", "archive_size_bytes", "witness_manifest_digest",
                            "authority_digest", "runtime_inputs_digest", "run_id", "result_digest"),
    }.items():
        if isinstance(result.get(name), Mapping):
            output[name] = {key: result[name][key] for key in fields if key in result[name]}
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("cell_id", "seed", "control_id", "terminal_state", "control_passed", "task_succeeded", "outcome", "receipt_digest"))
    for row in controls:
        writer.writerow((row["cell_id"], row["seed"], row["control_id"], row["terminal_state"], row["control_passed"],
                         row["score"].get("task_succeeded"), row["score"].get("outcome"), row["receipt_digest"]))
    csv_path = delivery_root / "policy_canary_controls.csv"
    write_immutable(csv_path, stream.getvalue().encode())
    output["controls_csv"] = add_artifact(role="controls_csv", path=csv_path, artifact_root=delivery_root)
    return output
