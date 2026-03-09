"""Log summarization helpers for long-running pipeline executions."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping


_PHASE_PATTERN = re.compile(r"\bPHASE\s+(?P<phase>\d+)\s*:\s*(?P<label>.+)$", re.IGNORECASE)
_TIMING_PATTERNS = [
    re.compile(
        r"(?i)\b(?P<label>[a-z0-9 _/\-\.]+?)\s+"
        r"(?:completed|finished|done)\s+in\s+(?P<seconds>\d+(?:\.\d+)?)\s*s(?:ec(?:onds?)?)?\b"
    ),
    re.compile(
        r"(?i)\b(?P<label>[a-z0-9 _/\-\.]+?)\s+duration[:=]\s*(?P<seconds>\d+(?:\.\d+)?)\s*s\b"
    ),
    re.compile(r"(?i)\belapsed(?:\s+time)?[:=]\s*(?P<seconds>\d+(?:\.\d+)?)\s*s\b"),
]


@dataclass(frozen=True)
class ParsedTiming:
    stage: str
    seconds: float
    source: str
    line: int

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "seconds": round(self.seconds, 3),
            "source": self.source,
            "line": self.line,
        }


@dataclass(frozen=True)
class ParsedIssue:
    severity: str
    message: str
    source: str
    line: int

    def to_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "message": self.message,
            "source": self.source,
            "line": self.line,
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _normalize_stage_label(raw: str, *, fallback: str) -> str:
    cleaned = " ".join(raw.strip().split())
    if not cleaned:
        return fallback
    if len(cleaned) > 96:
        return f"{cleaned[:93]}..."
    return cleaned


def _extract_timing(line: str, *, source: str, line_number: int) -> ParsedTiming | None:
    stripped = line.strip()
    if not stripped:
        return None
    for pattern in _TIMING_PATTERNS:
        match = pattern.search(stripped)
        if not match:
            continue
        seconds = float(match.group("seconds"))
        label = match.groupdict().get("label") or "elapsed"
        stage = _normalize_stage_label(label, fallback="elapsed")
        return ParsedTiming(stage=stage, seconds=seconds, source=source, line=line_number)
    return None


def _extract_phase(line: str) -> str | None:
    match = _PHASE_PATTERN.search(line.strip())
    if not match:
        return None
    phase = match.group("phase")
    label = _normalize_stage_label(match.group("label"), fallback=f"Phase {phase}")
    return f"Phase {phase}: {label}"


def _looks_like_warning(line: str) -> bool:
    lowered = line.lower()
    return "warning" in lowered or "warn:" in lowered


def _looks_like_error(line: str) -> bool:
    lowered = line.lower()
    if "0 failed" in lowered:
        return False
    if "failed in 0." in lowered:
        return False
    return any(
        token in lowered
        for token in (
            "traceback",
            "error:",
            "exception",
            "fatal",
            " failed",
            "failed ",
        )
    )


def _collect_issues(lines: Iterable[str], *, source: str) -> tuple[list[ParsedIssue], list[ParsedIssue]]:
    errors: list[ParsedIssue] = []
    warnings: list[ParsedIssue] = []
    for line_number, line in enumerate(lines, start=1):
        message = line.strip()
        if not message:
            continue
        if _looks_like_error(message):
            errors.append(
                ParsedIssue(severity="error", message=message, source=source, line=line_number)
            )
            continue
        if _looks_like_warning(message):
            warnings.append(
                ParsedIssue(severity="warning", message=message, source=source, line=line_number)
            )
    return errors, warnings


def summarize_logs(log_paths: Mapping[str, Path]) -> dict[str, object]:
    timings: list[ParsedTiming] = []
    errors: list[ParsedIssue] = []
    warnings: list[ParsedIssue] = []
    stages_seen: list[dict[str, object]] = []
    logs_meta: list[dict[str, object]] = []

    for source, path in log_paths.items():
        lines = _read_lines(path)
        logs_meta.append(
            {
                "source": source,
                "path": str(path),
                "exists": path.is_file(),
                "line_count": len(lines),
            }
        )
        for line_number, line in enumerate(lines, start=1):
            phase = _extract_phase(line)
            if phase:
                stages_seen.append({"stage": phase, "source": source, "line": line_number})
            timing = _extract_timing(line, source=source, line_number=line_number)
            if timing:
                timings.append(timing)
        source_errors, source_warnings = _collect_issues(lines, source=source)
        errors.extend(source_errors)
        warnings.extend(source_warnings)

    timings.sort(key=lambda item: item.seconds, reverse=True)
    errors = errors[:50]
    warnings = warnings[:50]

    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "logs": logs_meta,
        "stages_seen": stages_seen,
        "stage_timings_seconds": [item.to_dict() for item in timings],
        "errors": [item.to_dict() for item in errors],
        "warnings": [item.to_dict() for item in warnings],
        "stats": {
            "timing_count": len(timings),
            "error_count": len(errors),
            "warning_count": len(warnings),
            "stage_count": len(stages_seen),
        },
    }


def render_markdown(summary: Mapping[str, object]) -> str:
    stages_seen = summary.get("stages_seen") if isinstance(summary.get("stages_seen"), list) else []
    timings = summary.get("stage_timings_seconds") if isinstance(summary.get("stage_timings_seconds"), list) else []
    errors = summary.get("errors") if isinstance(summary.get("errors"), list) else []
    warnings = summary.get("warnings") if isinstance(summary.get("warnings"), list) else []
    logs = summary.get("logs") if isinstance(summary.get("logs"), list) else []

    lines: list[str] = [
        "# Pipeline Log Summary",
        "",
        f"- Generated: {summary.get('generated_at', 'unknown')}",
        "",
        "## Log Inputs",
    ]
    if not logs:
        lines.append("- No logs were provided.")
    else:
        for item in logs:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or "log")
            path = str(item.get("path") or "")
            exists = bool(item.get("exists"))
            line_count = int(item.get("line_count") or 0)
            lines.append(f"- `{source}`: `{path}` ({'present' if exists else 'missing'}, {line_count} lines)")

    lines.extend(["", "## Stages Seen"])
    if not stages_seen:
        lines.append("- No explicit phase markers found.")
    else:
        for item in stages_seen[:20]:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- {item.get('stage', 'unknown')} "
                f"(source: `{item.get('source', 'log')}`, line {item.get('line', '?')})"
            )

    lines.extend(["", "## Timings"])
    if not timings:
        lines.append("- No duration lines found.")
    else:
        lines.append("| Stage | Seconds | Source | Line |")
        lines.append("| --- | ---: | --- | ---: |")
        for item in timings[:25]:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"| {item.get('stage', 'unknown')} | "
                f"{float(item.get('seconds') or 0.0):.3f} | "
                f"{item.get('source', 'log')} | "
                f"{int(item.get('line') or 0)} |"
            )

    lines.extend(["", "## Errors"])
    if not errors:
        lines.append("- No error lines detected.")
    else:
        for item in errors[:25]:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- `{item.get('source', 'log')}:{item.get('line', '?')}` {item.get('message', '')}"
            )

    lines.extend(["", "## Warnings"])
    if not warnings:
        lines.append("- No warning lines detected.")
    else:
        for item in warnings[:25]:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- `{item.get('source', 'log')}:{item.get('line', '?')}` {item.get('message', '')}"
            )

    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing nurec.log and orchestrator.log (default: cwd)",
    )
    parser.add_argument("--nurec-log", type=Path, default=None, help="Optional explicit NuRec log path")
    parser.add_argument(
        "--orchestrator-log",
        type=Path,
        default=None,
        help="Optional explicit orchestrator log path",
    )
    parser.add_argument("--out-json", type=Path, default=None, help="Summary JSON output path")
    parser.add_argument("--out-md", type=Path, default=None, help="Summary Markdown output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    pipeline_dir = args.pipeline_dir
    nurec_log = args.nurec_log or (pipeline_dir / "nurec.log")
    orchestrator_log = args.orchestrator_log or (pipeline_dir / "orchestrator.log")
    out_json = args.out_json or (pipeline_dir / "log_summary.json")
    out_md = args.out_md or (pipeline_dir / "log_summary.md")

    summary = summarize_logs({"nurec": nurec_log, "orchestrator": orchestrator_log})
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote log summary JSON: {out_json}")
    print(f"Wrote log summary Markdown: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
