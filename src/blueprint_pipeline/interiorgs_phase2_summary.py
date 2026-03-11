"""Generate a consolidated summary across scene-wide and task-run Phase 2 results."""

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
            entries = groups.get(category, []) if isinstance(groups, Mapping) and isinstance(groups.get(category), list) else []
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                capture_root = Path(str(entry.get("capture_root") or ""))
                readiness_path = capture_root / "pipeline" / "readiness_decision.json"
                readiness = _read_json(readiness_path) if readiness_path.is_file() else {}
                status = str(readiness.get("status") or "unknown")
                next_action = _recommended_next_action(readiness)
                rows.append(
                    {
                        "scene": scene_id,
                        "whole_home_capture_id": whole_capture_id,
                        "category": category,
                        "task_text": str(entry.get("task_text") or ""),
                        "capture_id": str(entry.get("capture_id") or ""),
                        "status": status,
                        "next_action": next_action,
                        "memo_path": str(entry.get("final_memo_path") or ""),
                    }
                )
    return rows


def _dashboard_payload(*, output_root: Path, bucket: str) -> Dict[str, object]:
    scenes_root = output_root / bucket / "scenes"
    scenes: List[Dict[str, object]] = []
    for scene_capture_root in _scene_capture_roots(scenes_root):
        scenes.append(_scene_dashboard_entry(scene_capture_root))
    return {
        "schema_version": "v1",
        "bucket": bucket,
        "output_root": str(output_root),
        "scenes": scenes,
    }


def _scene_dashboard_entry(scene_capture_root: Path) -> Dict[str, object]:
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
        entries = groups.get(category, []) if isinstance(groups, Mapping) and isinstance(groups.get(category), list) else []
        expanded: List[Dict[str, object]] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            capture_root = Path(str(entry.get("capture_root") or ""))
            readiness_path = capture_root / "pipeline" / "readiness_decision.json"
            readiness = _read_json(readiness_path) if readiness_path.is_file() else {}
            status = str(readiness.get("status") or "unknown")
            themes = _blocker_themes(readiness)
            action = _recommended_next_action(readiness)
            expanded.append(
                {
                    "task_text": str(entry.get("task_text") or ""),
                    "capture_id": str(entry.get("capture_id") or ""),
                    "status": status,
                    "next_action": action,
                    "themes": themes,
                    "memo_path": str(entry.get("final_memo_path") or ""),
                }
            )
            for theme in themes:
                blocker_theme_rollup[theme] += 1
            action_rollup[action] += 1
        category_summaries[category] = {
            "counts": _status_counter(expanded),
            "tasks": expanded,
        }
    return {
        "scene": scene_id,
        "whole_home": {
            "capture_id": capture_id,
            "status": str(whole_ready.get("status") or "unknown"),
            "confidence": whole_ready.get("confidence"),
            "memo_path": str(pipeline_dir / "agent_readiness_memo.md"),
        },
        "categories": category_summaries,
        "theme_counts": dict(blocker_theme_rollup),
        "action_counts": dict(action_rollup),
    }


def _blocker_themes(readiness: Mapping[str, object]) -> List[str]:
    blockers = readiness.get("blockers", [])
    themes = set()
    if isinstance(blockers, list):
        for blocker in blockers:
            if not isinstance(blocker, Mapping):
                continue
            detail = str(blocker.get("detail") or "").strip().lower()
            if "reach" in detail:
                themes.add("reach")
            if "workcell span" in detail or "occupancy" in detail:
                themes.add("workcell span")
            if "articulated targets" in detail or "articulation" in detail:
                themes.add("articulation complexity")
            if "route width" in detail or "clearance" in detail or "choke-point" in detail or "choke point" in detail:
                themes.add("route / clearance")
            if "hidden-zone" in detail or "hidden zone" in detail or "occlusion" in detail:
                themes.add("hidden-zone coverage")
            if "uncertainty" in detail or "route viability" in detail:
                themes.add("uncertainty / route viability")
            if "coexistence" in detail or "traffic visibility" in detail:
                themes.add("shared-space coexistence")
    if not themes and str(readiness.get("status") or "") == "ready" and bool(readiness.get("human_review_required")):
        themes.add("human review only")
    if not themes:
        themes.add("other")
    return sorted(themes)


def _theme_action_bucket(theme: str) -> str:
    mapping = {
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
    return mapping.get(theme, "human review / policy")


def _recommended_next_action(readiness: Mapping[str, object]) -> str:
    status = str(readiness.get("status") or "unknown")
    themes = _blocker_themes(readiness)
    if status == "ready":
        return "advance to human signoff"
    if "hidden-zone coverage" in themes or "uncertainty / route viability" in themes:
        return "recapture"
    if "route / clearance" in themes or "workcell span" in themes:
        return "redesign"
    if "reach" in themes or "articulation complexity" in themes or "shared-space coexistence" in themes:
        return "defer"
    if "human review only" in themes:
        return "advance to human signoff"
    return "defer"


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
            entries = groups.get(category, []) if isinstance(groups, Mapping) and isinstance(groups.get(category), list) else []
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
                lines.append(
                    f"- `{task['status']}` {task['task_text']} "
                    f"([memo]({task['memo']}))"
                )
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
            entries = groups.get(category, []) if isinstance(groups, Mapping) and isinstance(groups.get(category), list) else []
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
            entries = groups.get(category, []) if isinstance(groups, Mapping) and isinstance(groups.get(category), list) else []
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
        "- `recapture`: gather better evidence before judging the task",
        "- `redesign`: narrow or restructure the task / route / workcell",
        "- `defer`: likely outside the current bounded robot envelope",
        "- `advance to human signoff`: ready enough to move to explicit human approval",
        "",
    ]
    for scene_capture_root in _scene_capture_roots(scenes_root):
        scene_id = scene_capture_root.parts[-3]
        pipeline_dir = scene_capture_root / "pipeline"
        manifest = _read_json(pipeline_dir / "task_run_manifest.json")
        groups = manifest.get("groups", {}) if isinstance(manifest, Mapping) else {}
        lines.extend([f"## Scene `{scene_id}`", ""])
        for category in ("pick", "open_close", "navigate"):
            entries = groups.get(category, []) if isinstance(groups, Mapping) and isinstance(groups.get(category), list) else []
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
                lines.append(
                    f"- `{action}` {entry.get('task_text')} "
                    f"([memo]({entry.get('final_memo_path')}))"
                )
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
        pipeline_dir = scene_capture_root / "pipeline"
        ensure_dir(pipeline_dir)
        output_path = pipeline_dir / "dashboard_summary.json"
        output_path.write_text(
            json.dumps(_scene_dashboard_entry(scene_capture_root), indent=2),
            encoding="utf-8",
        )
        written.append(output_path)
    return written


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a consolidated InteriorGS Phase 2 summary")
    parser.add_argument("--output-root", default=str(_DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--bucket", default=_DEFAULT_BUCKET)
    parser.add_argument("--destination", default=None)
    args = parser.parse_args(argv)

    destination = Path(args.destination).resolve() if args.destination else None
    path = write_consolidated_summary(
        output_root=Path(args.output_root).resolve(),
        bucket=args.bucket,
        destination=destination,
    )
    theme_path = write_blocker_theme_summary(
        output_root=Path(args.output_root).resolve(),
        bucket=args.bucket,
    )
    actionability_path = write_actionability_summary(
        output_root=Path(args.output_root).resolve(),
        bucket=args.bucket,
    )
    next_actions_path = write_recommended_next_actions_summary(
        output_root=Path(args.output_root).resolve(),
        bucket=args.bucket,
    )
    next_actions_csv_path = write_recommended_next_actions_csv(
        output_root=Path(args.output_root).resolve(),
        bucket=args.bucket,
    )
    dashboard_path = write_dashboard_summary_json(
        output_root=Path(args.output_root).resolve(),
        bucket=args.bucket,
    )
    scene_dashboard_paths = write_scene_dashboard_summaries(
        output_root=Path(args.output_root).resolve(),
        bucket=args.bucket,
    )
    print(f"[interiorgs-phase2-summary] wrote {path}")
    print(f"[interiorgs-phase2-summary] wrote {theme_path}")
    print(f"[interiorgs-phase2-summary] wrote {actionability_path}")
    print(f"[interiorgs-phase2-summary] wrote {next_actions_path}")
    print(f"[interiorgs-phase2-summary] wrote {next_actions_csv_path}")
    print(f"[interiorgs-phase2-summary] wrote {dashboard_path}")
    for scene_path in scene_dashboard_paths:
        print(f"[interiorgs-phase2-summary] wrote {scene_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
