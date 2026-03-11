"""Generate consolidated summaries across scene-wide and task-run Phase 2 results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Mapping, Optional

from .common import ensure_dir, write_text


_DEFAULT_BUCKET = "localbucket"
_DEFAULT_OUTPUT_ROOT = Path("/tmp/blueprint_interiorgs_phase2")
_SCENE_DASHBOARD_SCHEMA_VERSION = "v1"
_READY_ACTION = "advance to human signoff"
_RECAPTURE_ACTION = "recapture"
_REDESIGN_ACTION = "redesign"
_DEFER_ACTION = "defer"

_BLOCKER_CATEGORY_THEME_MAP = {
    "capture_coverage": "hidden-zone coverage",
    "capture_quality": "uncertainty / route viability",
    "geometry_clearance": "route / clearance",
    "geometry_floor": "route / clearance",
    "geometry_reach": "reach",
    "machine_interface": "articulation complexity",
    "platform_limitation": "reach",
    "traffic_pedestrian": "shared-space coexistence",
    "traffic_shared": "shared-space coexistence",
    "workflow_ambiguity": "workcell span",
}

_RESOLUTION_PATH_ACTION_MAP = {
    "recapture": _RECAPTURE_ACTION,
    "scope_change": _REDESIGN_ACTION,
    "site_modification": _REDESIGN_ACTION,
    "platform_change": _DEFER_ACTION,
    "not_resolvable": _DEFER_ACTION,
    "oem_consultation": _DEFER_ACTION,
}

_THEME_ACTION_BUCKETS = {
    "hidden-zone coverage": "recapture",
    "uncertainty / route viability": "recapture",
    "human review only": "human review / policy",
    "shared-space coexistence": "human review / policy",
    "route / clearance": "task redesign",
    "reach": "robot capability mismatch",
    "workcell span": "task redesign",
    "articulation complexity": "robot capability mismatch",
    "other": "human review / policy",
}


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scene_capture_roots(base_root: Path) -> List[Path]:
    roots: List[Path] = []
    for manifest_path in sorted(base_root.glob("*/captures/*/pipeline/task_run_manifest.json")):
        roots.append(manifest_path.parent.parent)
    return roots


def _status_counter(items: List[Mapping[str, object]]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for item in items:
        status = str(item.get("status") or "unknown")
        counter[status] += 1
    return {
        "ready": counter.get("ready", 0),
        "risky": counter.get("risky", 0),
        "not_ready_yet": counter.get("not_ready_yet", 0),
    }


def _safe_number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalized_blockers(readiness: Mapping[str, object]) -> List[Mapping[str, object]]:
    blockers = readiness.get("blockers", [])
    if not isinstance(blockers, list):
        return []
    return [blocker for blocker in blockers if isinstance(blocker, Mapping)]


def _normalized_checks(readiness: Mapping[str, object]) -> List[Mapping[str, object]]:
    checks = readiness.get("capability_checks", [])
    if not isinstance(checks, list):
        return []
    return [check for check in checks if isinstance(check, Mapping)]


def _task_memo_uri(*, bucket: str, scene_id: str, capture_id: str) -> str:
    return f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/pipeline/agent_readiness_memo.md"


def _theme_from_detail(detail: str) -> Optional[str]:
    if "reach" in detail:
        return "reach"
    if "workcell span" in detail or "occupancy" in detail:
        return "workcell span"
    if "articulated targets" in detail or "articulation" in detail:
        return "articulation complexity"
    if (
        "route width" in detail
        or "clearance" in detail
        or "choke-point" in detail
        or "choke point" in detail
    ):
        return "route / clearance"
    if "hidden-zone" in detail or "hidden zone" in detail or "occlusion" in detail:
        return "hidden-zone coverage"
    if "uncertainty" in detail or "route viability" in detail:
        return "uncertainty / route viability"
    if "coexistence" in detail or "traffic visibility" in detail:
        return "shared-space coexistence"
    return None


def _theme_for_blocker(blocker: Mapping[str, object]) -> Optional[str]:
    category = str(blocker.get("category") or "").strip().lower()
    if category in _BLOCKER_CATEGORY_THEME_MAP:
        return _BLOCKER_CATEGORY_THEME_MAP[category]
    detail = str(blocker.get("detail") or "").strip().lower()
    resolution_path = str(blocker.get("resolution_path") or "").strip().lower()
    if resolution_path == "recapture" and ("hidden" in detail or "occlusion" in detail):
        return "hidden-zone coverage"
    theme = _theme_from_detail(detail)
    if theme:
        return theme
    if resolution_path == "recapture":
        return "uncertainty / route viability"
    return None


def _themes_from_checks(readiness: Mapping[str, object]) -> List[str]:
    themes = set()
    for check in _normalized_checks(readiness):
        status = str(check.get("status") or "").strip().lower()
        if status in {"pass", "passed", "ok"}:
            continue
        detail = str(check.get("detail") or "").strip().lower()
        theme = _theme_from_detail(detail)
        if theme:
            themes.add(theme)
    return sorted(themes)


def _blocker_themes(readiness: Mapping[str, object]) -> List[str]:
    themes = set()
    for blocker in _normalized_blockers(readiness):
        theme = _theme_for_blocker(blocker)
        if theme:
            themes.add(theme)
    for theme in _themes_from_checks(readiness):
        themes.add(theme)

    hidden_zone_bound = _safe_number(readiness.get("hidden_zone_bound"), default=0.0)
    if hidden_zone_bound > 0.35:
        themes.add("hidden-zone coverage")

    if not themes and str(readiness.get("status") or "") in {"ready", "risky"} and bool(
        readiness.get("human_review_required")
    ):
        themes.add("human review only")
    if not themes:
        themes.add("other")
    return sorted(themes)


def _theme_action_bucket(theme: str) -> str:
    return _THEME_ACTION_BUCKETS.get(theme, "human review / policy")


def _recommended_next_action(readiness: Mapping[str, object]) -> str:
    status = str(readiness.get("status") or "unknown").strip().lower()
    blockers = _normalized_blockers(readiness)

    for blocker in blockers:
        resolution_path = str(blocker.get("resolution_path") or "").strip().lower()
        mapped = _RESOLUTION_PATH_ACTION_MAP.get(resolution_path)
        if mapped:
            return mapped

    themes = _blocker_themes(readiness)

    if "hidden-zone coverage" in themes or "uncertainty / route viability" in themes:
        return _RECAPTURE_ACTION
    if "route / clearance" in themes or "workcell span" in themes:
        return _REDESIGN_ACTION
    if "reach" in themes or "articulation complexity" in themes:
        return _DEFER_ACTION
    if "shared-space coexistence" in themes and status not in {"ready", "risky"}:
        return _DEFER_ACTION
    if status in {"ready", "risky"}:
        return _READY_ACTION
    if "human review only" in themes:
        return _READY_ACTION
    return _DEFER_ACTION


def _task_entry(
    *,
    entry: Mapping[str, object],
    readiness: Mapping[str, object],
    bucket: str,
    scene_id: str,
) -> Dict[str, object]:
    capture_id = str(entry.get("capture_id") or "")
    return {
        "task_text": str(entry.get("task_text") or ""),
        "capture_id": capture_id,
        "status": str(readiness.get("status") or "unknown"),
        "next_action": _recommended_next_action(readiness),
        "themes": _blocker_themes(readiness),
        "memo_path": str(entry.get("final_memo_path") or ""),
        "memo_uri": _task_memo_uri(bucket=bucket, scene_id=scene_id, capture_id=capture_id),
    }


def _deployment_summary(category_summaries: Mapping[str, Mapping[str, object]]) -> Dict[str, int]:
    total_tasks = 0
    ready_now = 0
    needs_redesign = 0
    outside_robot_envelope = 0
    for category in category_summaries.values():
        tasks = category.get("tasks", []) if isinstance(category, Mapping) else []
        if not isinstance(tasks, list):
            continue
        for task in tasks:
            if not isinstance(task, Mapping):
                continue
            total_tasks += 1
            action = str(task.get("next_action") or "")
            if action == _READY_ACTION:
                ready_now += 1
            elif action == _REDESIGN_ACTION:
                needs_redesign += 1
            elif action == _DEFER_ACTION:
                outside_robot_envelope += 1
    return {
        "total_tasks": total_tasks,
        "ready_now": ready_now,
        "needs_redesign": needs_redesign,
        "outside_robot_envelope": outside_robot_envelope,
    }


def _scene_dashboard_entry(scene_capture_root: Path, *, bucket: str) -> Dict[str, object]:
    scene_id = scene_capture_root.parts[-3]
    capture_id = scene_capture_root.parts[-1]
    pipeline_dir = scene_capture_root / "pipeline"
    whole_ready = _read_json(pipeline_dir / "readiness_decision.json")
    manifest = _read_json(pipeline_dir / "task_run_manifest.json")
    groups = manifest.get("groups", {}) if isinstance(manifest, Mapping) else {}
    category_summaries: Dict[str, Dict[str, object]] = {}
    blocker_theme_rollup: Counter[str] = Counter()
    action_rollup: Counter[str] = Counter()
    for category in ("pick", "open_close", "navigate"):
        entries = (
            groups.get(category, [])
            if isinstance(groups, Mapping) and isinstance(groups.get(category), list)
            else []
        )
        expanded: List[Dict[str, object]] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            capture_root = Path(str(entry.get("capture_root") or ""))
            readiness_path = capture_root / "pipeline" / "readiness_decision.json"
            readiness = _read_json(readiness_path) if readiness_path.is_file() else {}
            task = _task_entry(entry=entry, readiness=readiness, bucket=bucket, scene_id=scene_id)
            expanded.append(task)
            for theme in task["themes"]:
                blocker_theme_rollup[str(theme)] += 1
            action_rollup[str(task["next_action"])] += 1
        category_summaries[category] = {
            "counts": _status_counter(expanded),
            "tasks": expanded,
        }
    return {
        "schema_version": _SCENE_DASHBOARD_SCHEMA_VERSION,
        "scene": scene_id,
        "whole_home": {
            "capture_id": capture_id,
            "status": str(whole_ready.get("status") or "unknown"),
            "confidence": whole_ready.get("confidence"),
            "memo_path": str(pipeline_dir / "agent_readiness_memo.md"),
            "memo_uri": _task_memo_uri(bucket=bucket, scene_id=scene_id, capture_id=capture_id),
        },
        "categories": category_summaries,
        "theme_counts": dict(blocker_theme_rollup),
        "action_counts": dict(action_rollup),
        "deployment_summary": _deployment_summary(category_summaries),
    }


def _dashboard_payload(*, output_root: Path, bucket: str) -> Dict[str, object]:
    scenes_root = output_root / bucket / "scenes"
    scenes: List[Dict[str, object]] = []
    for scene_capture_root in _scene_capture_roots(scenes_root):
        scenes.append(_scene_dashboard_entry(scene_capture_root, bucket=bucket))
    return {
        "schema_version": _SCENE_DASHBOARD_SCHEMA_VERSION,
        "bucket": bucket,
        "output_root": str(output_root),
        "scenes": scenes,
    }


def _task_group_summary(entries: List[Mapping[str, object]]) -> Dict[str, object]:
    expanded: List[Dict[str, object]] = []
    for entry in entries:
        capture_root = Path(str(entry.get("capture_root") or ""))
        readiness_path = capture_root / "pipeline" / "readiness_decision.json"
        readiness = _read_json(readiness_path) if readiness_path.is_file() else {}
        expanded.append(
            {
                "task_text": str(entry.get("task_text") or ""),
                "capture_id": str(entry.get("capture_id") or ""),
                "status": str(readiness.get("status") or "unknown"),
                "memo": str(entry.get("final_memo_path") or ""),
            }
        )
    return {
        "counts": _status_counter(expanded),
        "tasks": expanded,
    }


def _iter_task_rows(*, output_root: Path, bucket: str) -> List[Dict[str, str]]:
    scenes_root = output_root / bucket / "scenes"
    rows: List[Dict[str, str]] = []
    for scene_capture_root in _scene_capture_roots(scenes_root):
        scene_id = scene_capture_root.parts[-3]
        whole_capture_id = scene_capture_root.parts[-1]
        pipeline_dir = scene_capture_root / "pipeline"
        manifest = _read_json(pipeline_dir / "task_run_manifest.json")
        groups = manifest.get("groups", {}) if isinstance(manifest, Mapping) else {}
        for category in ("pick", "open_close", "navigate"):
            entries = (
                groups.get(category, [])
                if isinstance(groups, Mapping) and isinstance(groups.get(category), list)
                else []
            )
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                capture_root = Path(str(entry.get("capture_root") or ""))
                readiness_path = capture_root / "pipeline" / "readiness_decision.json"
                readiness = _read_json(readiness_path) if readiness_path.is_file() else {}
                rows.append(
                    {
                        "scene": scene_id,
                        "whole_home_capture_id": whole_capture_id,
                        "category": category,
                        "task_text": str(entry.get("task_text") or ""),
                        "capture_id": str(entry.get("capture_id") or ""),
                        "status": str(readiness.get("status") or "unknown"),
                        "next_action": _recommended_next_action(readiness),
                        "memo_path": str(entry.get("final_memo_path") or ""),
                    }
                )
    return rows


def build_consolidated_summary(*, output_root: Path, bucket: str = _DEFAULT_BUCKET) -> str:
    scenes_root = output_root / bucket / "scenes"
    lines = [
        "# Consolidated InteriorGS Phase 2 Summary",
        "",
        f"- Output root: `{output_root}`",
        f"- Bucket: `{bucket}`",
        "",
    ]
    for scene_capture_root in _scene_capture_roots(scenes_root):
        scene_id = scene_capture_root.parts[-3]
        capture_id = scene_capture_root.parts[-1]
        pipeline_dir = scene_capture_root / "pipeline"
        whole_ready = _read_json(pipeline_dir / "readiness_decision.json")
        whole_status = str(whole_ready.get("status") or "unknown")
        whole_confidence = whole_ready.get("confidence")
        manifest = _read_json(pipeline_dir / "task_run_manifest.json")
        lines.extend(
            [
                f"## Scene `{scene_id}`",
                "",
                f"- Whole-home capture: `{capture_id}`",
                f"- Whole-home status: `{whole_status}`",
                f"- Whole-home confidence: `{whole_confidence}`",
                f"- Whole-home memo: `{pipeline_dir / 'agent_readiness_memo.md'}`",
                "",
            ]
        )
        groups = manifest.get("groups", {}) if isinstance(manifest, Mapping) else {}
        for category in ("pick", "open_close", "navigate"):
            entries = (
                groups.get(category, [])
                if isinstance(groups, Mapping) and isinstance(groups.get(category), list)
                else []
            )
            summary = _task_group_summary(entries)
            counts = summary["counts"]
            lines.extend(
                [
                    f"### {category}",
                    "",
                    f"- Counts: `ready={counts['ready']}`, `risky={counts['risky']}`, `not_ready_yet={counts['not_ready_yet']}`",
                ]
            )
            for task in summary["tasks"]:
                lines.append(f"- `{task['status']}` {task['task_text']} ([memo]({task['memo']}))")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_blocker_theme_summary(*, output_root: Path, bucket: str = _DEFAULT_BUCKET) -> str:
    scenes_root = output_root / bucket / "scenes"
    lines = [
        "# InteriorGS Phase 2 Blocker Theme Summary",
        "",
        f"- Output root: `{output_root}`",
        f"- Bucket: `{bucket}`",
        "",
    ]
    for scene_capture_root in _scene_capture_roots(scenes_root):
        scene_id = scene_capture_root.parts[-3]
        pipeline_dir = scene_capture_root / "pipeline"
        manifest = _read_json(pipeline_dir / "task_run_manifest.json")
        groups = manifest.get("groups", {}) if isinstance(manifest, Mapping) else {}
        lines.extend([f"## Scene `{scene_id}`", ""])
        for category in ("pick", "open_close", "navigate"):
            entries = (
                groups.get(category, [])
                if isinstance(groups, Mapping) and isinstance(groups.get(category), list)
                else []
            )
            theme_counts: Counter[str] = Counter()
            examples: DefaultDict[str, List[str]] = defaultdict(list)
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                capture_root = Path(str(entry.get("capture_root") or ""))
                readiness_path = capture_root / "pipeline" / "readiness_decision.json"
                readiness = _read_json(readiness_path) if readiness_path.is_file() else {}
                task_text = str(entry.get("task_text") or "")
                for theme in _blocker_themes(readiness):
                    theme_counts[theme] += 1
                    if len(examples[theme]) < 3:
                        examples[theme].append(task_text)
            lines.append(f"### {category}")
            lines.append("")
            if not theme_counts:
                lines.append("- none")
                lines.append("")
                continue
            for theme, count in theme_counts.most_common():
                sample_text = "; ".join(examples[theme])
                lines.append(f"- `{theme}`: `{count}` tasks")
                if sample_text:
                    lines.append(f"  Examples: {sample_text}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_actionability_summary(*, output_root: Path, bucket: str = _DEFAULT_BUCKET) -> str:
    scenes_root = output_root / bucket / "scenes"
    lines = [
        "# InteriorGS Phase 2 Actionability Summary",
        "",
        f"- Output root: `{output_root}`",
        f"- Bucket: `{bucket}`",
        "",
        "Action buckets:",
        "- `recapture`: more or better evidence should plausibly improve the decision",
        "- `task redesign`: narrow or restructure the task / route / workcell scope",
        "- `robot capability mismatch`: the task appears outside the current bounded envelope",
        "- `human review / policy`: requires signoff, traffic policy, or non-geometric operational review",
        "",
    ]
    for scene_capture_root in _scene_capture_roots(scenes_root):
        scene_id = scene_capture_root.parts[-3]
        pipeline_dir = scene_capture_root / "pipeline"
        manifest = _read_json(pipeline_dir / "task_run_manifest.json")
        groups = manifest.get("groups", {}) if isinstance(manifest, Mapping) else {}
        lines.extend([f"## Scene `{scene_id}`", ""])
        for category in ("pick", "open_close", "navigate"):
            entries = (
                groups.get(category, [])
                if isinstance(groups, Mapping) and isinstance(groups.get(category), list)
                else []
            )
            bucket_counts: Counter[str] = Counter()
            bucket_examples: DefaultDict[str, List[str]] = defaultdict(list)
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                capture_root = Path(str(entry.get("capture_root") or ""))
                readiness_path = capture_root / "pipeline" / "readiness_decision.json"
                readiness = _read_json(readiness_path) if readiness_path.is_file() else {}
                task_text = str(entry.get("task_text") or "")
                for theme in _blocker_themes(readiness):
                    action_bucket = _theme_action_bucket(theme)
                    bucket_counts[action_bucket] += 1
                    if len(bucket_examples[action_bucket]) < 3:
                        bucket_examples[action_bucket].append(f"{task_text} ({theme})")
            lines.append(f"### {category}")
            lines.append("")
            if not bucket_counts:
                lines.append("- none")
                lines.append("")
                continue
            for bucket_name, count in bucket_counts.most_common():
                sample_text = "; ".join(bucket_examples[bucket_name])
                lines.append(f"- `{bucket_name}`: `{count}` theme hits")
                if sample_text:
                    lines.append(f"  Examples: {sample_text}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_recommended_next_actions_summary(*, output_root: Path, bucket: str = _DEFAULT_BUCKET) -> str:
    scenes_root = output_root / bucket / "scenes"
    lines = [
        "# InteriorGS Phase 2 Recommended Next Actions",
        "",
        f"- Output root: `{output_root}`",
        f"- Bucket: `{bucket}`",
        "",
        "Actions:",
        f"- `{_RECAPTURE_ACTION}`: gather better evidence before judging the task",
        f"- `{_REDESIGN_ACTION}`: narrow or restructure the task / route / workcell",
        f"- `{_DEFER_ACTION}`: likely outside the current bounded robot envelope",
        f"- `{_READY_ACTION}`: ready enough to move to explicit human approval",
        "",
    ]
    for scene_capture_root in _scene_capture_roots(scenes_root):
        scene_id = scene_capture_root.parts[-3]
        pipeline_dir = scene_capture_root / "pipeline"
        manifest = _read_json(pipeline_dir / "task_run_manifest.json")
        groups = manifest.get("groups", {}) if isinstance(manifest, Mapping) else {}
        lines.extend([f"## Scene `{scene_id}`", ""])
        for category in ("pick", "open_close", "navigate"):
            entries = (
                groups.get(category, [])
                if isinstance(groups, Mapping) and isinstance(groups.get(category), list)
                else []
            )
            lines.append(f"### {category}")
            lines.append("")
            if not entries:
                lines.append("- none")
                lines.append("")
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                capture_root = Path(str(entry.get("capture_root") or ""))
                readiness_path = capture_root / "pipeline" / "readiness_decision.json"
                readiness = _read_json(readiness_path) if readiness_path.is_file() else {}
                action = _recommended_next_action(readiness)
                lines.append(f"- `{action}` {entry.get('task_text')} ([memo]({entry.get('final_memo_path')}))")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_scene_deployment_summary(scene_dashboard: Mapping[str, object]) -> str:
    scene_id = str(scene_dashboard.get("scene") or "unknown")
    whole_home = scene_dashboard.get("whole_home", {})
    deployment_summary = scene_dashboard.get("deployment_summary", {})
    categories = scene_dashboard.get("categories", {})
    lines = [
        "# Scene Deployment Summary",
        "",
        f"- Scene: `{scene_id}`",
        f"- Whole-home capture: `{whole_home.get('capture_id', '')}`",
        f"- Whole-home status: `{whole_home.get('status', 'unknown')}`",
        f"- Ready now: `{deployment_summary.get('ready_now', 0)}`",
        f"- Need redesign: `{deployment_summary.get('needs_redesign', 0)}`",
        f"- Outside robot envelope: `{deployment_summary.get('outside_robot_envelope', 0)}`",
        "",
    ]
    sections = [
        ("Ready Now", _READY_ACTION),
        ("Need Redesign", _REDESIGN_ACTION),
        ("Outside Robot Envelope", _DEFER_ACTION),
    ]
    for heading, action in sections:
        lines.extend([f"## {heading}", ""])
        wrote_any = False
        if isinstance(categories, Mapping):
            for category_name in ("pick", "open_close", "navigate"):
                category = categories.get(category_name, {})
                tasks = category.get("tasks", []) if isinstance(category, Mapping) else []
                if not isinstance(tasks, list):
                    continue
                for task in tasks:
                    if not isinstance(task, Mapping):
                        continue
                    if str(task.get("next_action") or "") != action:
                        continue
                    wrote_any = True
                    lines.append(f"- `{category_name}` {task.get('task_text')} (`{task.get('capture_id')}`)")
        if not wrote_any:
            lines.append("- none")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_consolidated_summary(
    *,
    output_root: Path,
    bucket: str = _DEFAULT_BUCKET,
    destination: Optional[Path] = None,
) -> Path:
    output_path = destination or (output_root / bucket / "phase2_consolidated_summary.md")
    ensure_dir(output_path.parent)
    write_text(output_path, build_consolidated_summary(output_root=output_root, bucket=bucket))
    return output_path


def write_blocker_theme_summary(
    *,
    output_root: Path,
    bucket: str = _DEFAULT_BUCKET,
    destination: Optional[Path] = None,
) -> Path:
    output_path = destination or (output_root / bucket / "phase2_blocker_theme_summary.md")
    ensure_dir(output_path.parent)
    write_text(output_path, build_blocker_theme_summary(output_root=output_root, bucket=bucket))
    return output_path


def write_actionability_summary(
    *,
    output_root: Path,
    bucket: str = _DEFAULT_BUCKET,
    destination: Optional[Path] = None,
) -> Path:
    output_path = destination or (output_root / bucket / "phase2_actionability_summary.md")
    ensure_dir(output_path.parent)
    write_text(output_path, build_actionability_summary(output_root=output_root, bucket=bucket))
    return output_path


def write_recommended_next_actions_summary(
    *,
    output_root: Path,
    bucket: str = _DEFAULT_BUCKET,
    destination: Optional[Path] = None,
) -> Path:
    output_path = destination or (output_root / bucket / "phase2_recommended_next_actions.md")
    ensure_dir(output_path.parent)
    write_text(output_path, build_recommended_next_actions_summary(output_root=output_root, bucket=bucket))
    return output_path


def write_recommended_next_actions_csv(
    *,
    output_root: Path,
    bucket: str = _DEFAULT_BUCKET,
    destination: Optional[Path] = None,
) -> Path:
    output_path = destination or (output_root / bucket / "phase2_recommended_next_actions.csv")
    ensure_dir(output_path.parent)
    rows = _iter_task_rows(output_root=output_root, bucket=bucket)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scene",
                "whole_home_capture_id",
                "category",
                "task_text",
                "capture_id",
                "status",
                "next_action",
                "memo_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_dashboard_summary_json(
    *,
    output_root: Path,
    bucket: str = _DEFAULT_BUCKET,
    destination: Optional[Path] = None,
) -> Path:
    output_path = destination or (output_root / bucket / "phase2_dashboard_summary.json")
    ensure_dir(output_path.parent)
    output_path.write_text(
        json.dumps(_dashboard_payload(output_root=output_root, bucket=bucket), indent=2),
        encoding="utf-8",
    )
    return output_path


def write_scene_dashboard_summaries(
    *,
    output_root: Path,
    bucket: str = _DEFAULT_BUCKET,
) -> List[Path]:
    written: List[Path] = []
    scenes_root = output_root / bucket / "scenes"
    for scene_capture_root in _scene_capture_roots(scenes_root):
        written.append(write_scene_dashboard_summary(scene_capture_root=scene_capture_root, bucket=bucket))
    return written


def write_scene_dashboard_summary(*, scene_capture_root: Path, bucket: str = _DEFAULT_BUCKET) -> Path:
    pipeline_dir = scene_capture_root / "pipeline"
    ensure_dir(pipeline_dir)
    output_path = pipeline_dir / "dashboard_summary.json"
    output_path.write_text(
        json.dumps(_scene_dashboard_entry(scene_capture_root, bucket=bucket), indent=2),
        encoding="utf-8",
    )
    return output_path


def write_scene_deployment_summaries(
    *,
    output_root: Path,
    bucket: str = _DEFAULT_BUCKET,
) -> List[Path]:
    written: List[Path] = []
    scenes_root = output_root / bucket / "scenes"
    for scene_capture_root in _scene_capture_roots(scenes_root):
        written.append(write_scene_deployment_summary(scene_capture_root=scene_capture_root, bucket=bucket))
    return written


def write_scene_deployment_summary(
    *,
    scene_capture_root: Path,
    bucket: str = _DEFAULT_BUCKET,
) -> Path:
    pipeline_dir = scene_capture_root / "pipeline"
    ensure_dir(pipeline_dir)
    payload = _scene_dashboard_entry(scene_capture_root, bucket=bucket)
    output_path = pipeline_dir / "scene_deployment_summary.md"
    write_text(output_path, build_scene_deployment_summary(payload))
    return output_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a consolidated InteriorGS Phase 2 summary")
    parser.add_argument("--output-root", default=str(_DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--bucket", default=_DEFAULT_BUCKET)
    parser.add_argument("--destination", default=None)
    args = parser.parse_args(argv)

    destination = Path(args.destination).resolve() if args.destination else None
    resolved_root = Path(args.output_root).resolve()
    path = write_consolidated_summary(
        output_root=resolved_root,
        bucket=args.bucket,
        destination=destination,
    )
    theme_path = write_blocker_theme_summary(output_root=resolved_root, bucket=args.bucket)
    actionability_path = write_actionability_summary(output_root=resolved_root, bucket=args.bucket)
    next_actions_path = write_recommended_next_actions_summary(output_root=resolved_root, bucket=args.bucket)
    next_actions_csv_path = write_recommended_next_actions_csv(output_root=resolved_root, bucket=args.bucket)
    dashboard_path = write_dashboard_summary_json(output_root=resolved_root, bucket=args.bucket)
    scene_dashboard_paths = write_scene_dashboard_summaries(output_root=resolved_root, bucket=args.bucket)
    scene_deployment_paths = write_scene_deployment_summaries(output_root=resolved_root, bucket=args.bucket)
    print(f"[interiorgs-phase2-summary] wrote {path}")
    print(f"[interiorgs-phase2-summary] wrote {theme_path}")
    print(f"[interiorgs-phase2-summary] wrote {actionability_path}")
    print(f"[interiorgs-phase2-summary] wrote {next_actions_path}")
    print(f"[interiorgs-phase2-summary] wrote {next_actions_csv_path}")
    print(f"[interiorgs-phase2-summary] wrote {dashboard_path}")
    for scene_path in scene_dashboard_paths:
        print(f"[interiorgs-phase2-summary] wrote {scene_path}")
    for scene_path in scene_deployment_paths:
        print(f"[interiorgs-phase2-summary] wrote {scene_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
