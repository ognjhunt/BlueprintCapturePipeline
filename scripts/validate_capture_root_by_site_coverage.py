#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "capture_root_by_site_beta_coverage.v1"
PIPELINE_ENV = "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _load_json_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(payload)


def _parse_mapping_json(raw: str, *, label: str) -> dict[str, str]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return {
        _string(site_slug): _string(capture_root)
        for site_slug, capture_root in payload.items()
        if _string(site_slug)
    }


def _webapp_preflight_site_slugs(path: Path | None) -> tuple[set[str], dict[str, Any], list[str]]:
    if path is None:
        return set(), {}, []
    if not path.is_file():
        return set(), {}, [f"webapp_forwarding_preflight_missing:{path}"]
    try:
        payload = _load_json_mapping(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return set(), {}, [f"webapp_forwarding_preflight_read_failed:{type(exc).__name__}"]
    configured_env = _mapping(payload.get("configured_env"))
    by_site = _mapping(configured_env.get("capture_root_by_site_json"))
    blockers: list[str] = []
    if by_site.get("configured") is not True:
        blockers.append("webapp_capture_root_by_site_not_configured")
    if by_site.get("valid") is not True:
        blockers.append("webapp_capture_root_by_site_invalid")
    return {
        _string(site_slug)
        for site_slug in by_site.get("site_slugs") or []
        if _string(site_slug)
    }, payload, blockers


def validate_coverage(
    *,
    expected_site_roots: Mapping[str, str],
    pipeline_site_roots: Mapping[str, str],
    webapp_forwarding_preflight: Path | None = None,
    require_paths_exist: bool = False,
) -> dict[str, Any]:
    expected = {
        _string(site_slug): _string(capture_root)
        for site_slug, capture_root in expected_site_roots.items()
        if _string(site_slug)
    }
    pipeline = {
        _string(site_slug): _string(capture_root)
        for site_slug, capture_root in pipeline_site_roots.items()
        if _string(site_slug)
    }
    blockers: list[str] = []
    if not expected:
        blockers.append("expected_beta_site_roots_empty")
    if not pipeline:
        blockers.append("pipeline_capture_root_by_site_empty")

    webapp_site_slugs, webapp_payload, webapp_blockers = _webapp_preflight_site_slugs(
        webapp_forwarding_preflight
    )
    blockers.extend(webapp_blockers)

    site_results: list[dict[str, Any]] = []
    for site_slug in sorted(expected):
        expected_root = expected[site_slug]
        pipeline_root = pipeline.get(site_slug, "")
        site_blockers: list[str] = []
        if not pipeline_root:
            site_blockers.append(f"missing_pipeline_capture_root_for_site:{site_slug}")
        elif expected_root and Path(pipeline_root).expanduser().resolve() != Path(
            expected_root
        ).expanduser().resolve():
            site_blockers.append(f"pipeline_capture_root_mismatch_for_site:{site_slug}")
        if webapp_forwarding_preflight is not None and site_slug not in webapp_site_slugs:
            site_blockers.append(f"webapp_forwarding_preflight_missing_site:{site_slug}")
        if require_paths_exist and pipeline_root and not Path(pipeline_root).expanduser().is_dir():
            site_blockers.append(f"pipeline_capture_root_path_missing_for_site:{site_slug}")
        blockers.extend(site_blockers)
        site_results.append(
            {
                "site_slug": site_slug,
                "expected_capture_root": expected_root or None,
                "pipeline_capture_root": pipeline_root or None,
                "webapp_preflight_covered": (
                    site_slug in webapp_site_slugs
                    if webapp_forwarding_preflight is not None
                    else None
                ),
                "status": "passed" if not site_blockers else "blocked",
                "blockers": site_blockers,
            }
        )

    unexpected_pipeline_sites = sorted(set(pipeline) - set(expected))
    if unexpected_pipeline_sites:
        blockers.append("pipeline_capture_root_map_has_unexpected_sites")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not blockers else "blocked",
        "expected_site_count": len(expected),
        "pipeline_site_count": len(pipeline),
        "webapp_site_count": len(webapp_site_slugs) if webapp_forwarding_preflight else None,
        "expected_site_slugs": sorted(expected),
        "pipeline_site_slugs": sorted(pipeline),
        "webapp_site_slugs": sorted(webapp_site_slugs)
        if webapp_forwarding_preflight
        else None,
        "site_results": site_results,
        "unexpected_pipeline_sites": unexpected_pipeline_sites,
        "blockers": sorted(set(blockers)),
        "webapp_forwarding_preflight": {
            "path": str(webapp_forwarding_preflight) if webapp_forwarding_preflight else None,
            "status": webapp_payload.get("status") if webapp_payload else None,
        },
        "claim_boundary": {
            "pipeline_capture_root_by_site_map_checked": True,
            "webapp_forwarding_preflight_checked": webapp_forwarding_preflight is not None,
            "paths_exist_checked": require_paths_exist,
            "live_forwarding_or_pipeline_processing_proven": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate beta site_slug -> capture_root coverage across Pipeline and WebApp preflight."
    )
    parser.add_argument(
        "--expected-site-roots-json",
        required=True,
        help="JSON object mapping every beta site slug to its expected Pipeline capture root.",
    )
    parser.add_argument(
        "--pipeline-site-roots-json",
        default=os.getenv(PIPELINE_ENV, ""),
        help=f"Pipeline JSON map. Defaults to ${PIPELINE_ENV}.",
    )
    parser.add_argument("--webapp-forwarding-preflight", type=Path)
    parser.add_argument("--require-paths-exist", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.pipeline_site_roots_json:
        raise SystemExit(f"missing --pipeline-site-roots-json or ${PIPELINE_ENV}")
    report = validate_coverage(
        expected_site_roots=_parse_mapping_json(
            args.expected_site_roots_json,
            label="expected-site-roots-json",
        ),
        pipeline_site_roots=_parse_mapping_json(
            args.pipeline_site_roots_json,
            label="pipeline-site-roots-json",
        ),
        webapp_forwarding_preflight=args.webapp_forwarding_preflight,
        require_paths_exist=args.require_paths_exist,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
