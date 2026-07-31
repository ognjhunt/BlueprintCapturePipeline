"""Discover staged captures and audit first-GPU E2E readiness candidates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .artifact_storage import default_artifact_cache_root
from .first_gpu_e2e_readiness import PROVISIONERS, build_first_gpu_e2e_readiness
from .local_capture import resolve_local_capture_context
from .simulation_automation import SIMULATOR_FRAMEWORKS


FIRST_GPU_CANDIDATE_AUDIT_SCHEMA_VERSION = "first_gpu_candidate_audit.v1"
RAW_VIDEO_SUFFIXES = {".mov", ".mp4", ".m4v"}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _append_unique(target: List[Path], paths: Iterable[Path]) -> None:
    seen = {str(item) for item in target}
    for path in paths:
        key = str(path)
        if key not in seen:
            target.append(path)
            seen.add(key)


def _capture_root_from_completion_marker(path: Path) -> Path | None:
    if path.name != "capture_upload_complete.json" or path.parent.name != "raw":
        return None
    return path.parent.parent


def _discover_capture_roots(search_roots: Sequence[str | Path]) -> List[Path]:
    discovered: List[Path] = []
    for search_root in search_roots:
        root = Path(search_root).expanduser().resolve()
        if not root.exists():
            continue
        markers = (
            _capture_root_from_completion_marker(path)
            for path in root.rglob("capture_upload_complete.json")
            if path.is_file()
        )
        _append_unique(discovered, (path for path in markers if path is not None))
    return sorted(discovered)


def _explicit_capture_roots(capture_roots: Sequence[str | Path]) -> List[Path]:
    roots: List[Path] = []
    for capture_root in capture_roots:
        context = resolve_local_capture_context(capture_root)
        _append_unique(roots, [context.capture_root])
    return roots


def _raw_videos(capture_root: Path) -> List[str]:
    raw = capture_root / "raw"
    if not raw.is_dir():
        return []
    return [
        str(path)
        for path in sorted(raw.iterdir())
        if path.is_file() and path.suffix.lower() in RAW_VIDEO_SUFFIXES
    ]


def _candidate_summary(
    *,
    capture_root: Path,
    webapp_site_slug: str,
    simulator: str,
    provisioner: str,
    simulator_command: str | None,
    require_webapp_forwarding: bool,
    require_webapp_staged_request: bool,
    require_gpu_gates: bool,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    raw_videos = _raw_videos(context.capture_root)
    try:
        readiness = build_first_gpu_e2e_readiness(
            capture_root=context.capture_root,
            webapp_site_slug=webapp_site_slug or context.scene_id,
            simulator=simulator,
            provisioner=provisioner,
            simulator_command=simulator_command,
            require_webapp_forwarding=require_webapp_forwarding,
            require_webapp_staged_request=require_webapp_staged_request,
            require_gpu_gates=require_gpu_gates,
        )
        status = readiness.get("status")
        blockers = list(readiness.get("blockers") or [])
        warnings = list(readiness.get("warnings") or [])
    except Exception as exc:  # pragma: no cover - defensive, surfaced in manifest
        readiness = {}
        status = "audit_failed"
        blockers = [f"audit_exception:{exc.__class__.__name__}"]
        warnings = []
    if not raw_videos and "capture_preflight:missing_raw_input:raw_video" not in blockers:
        blockers.append("candidate_discovery:raw_video_missing")
    return {
        "capture_root": str(context.capture_root),
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "raw_video_paths": raw_videos,
        "has_raw_video": bool(raw_videos),
        "readiness_status": status,
        "ready_for_first_gpu_attempt": bool(readiness.get("ready_for_first_gpu_attempt")),
        "owner_gpu_proof_ready": bool(readiness.get("owner_gpu_proof_ready")),
        "blockers": blockers,
        "warnings": warnings,
        "readiness_manifest_written_by_audit": (
            str(context.capture_root / "pipeline" / "first_gpu_e2e_readiness_manifest.json")
        ),
        "proof_boundary": (
            "Candidate audit discovers and validates staged capture inputs only; it does not run "
            "providers, WebApp, simulators, GPU provisioning, policy execution, or proof upgrades."
        ),
    }


def build_first_gpu_candidate_audit(
    *,
    search_roots: Sequence[str | Path] = (),
    capture_roots: Sequence[str | Path] = (),
    webapp_site_slug: str = "",
    simulator: str = "isaac_sim",
    provisioner: str = "runpod",
    simulator_command: str | None = None,
    require_webapp_forwarding: bool = True,
    require_webapp_staged_request: bool = True,
    require_gpu_gates: bool = True,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    resolved_output_path = output_path or (
        default_artifact_cache_root() / "first_gpu_candidate_audit_manifest.json"
    )
    explicit_roots = _explicit_capture_roots(capture_roots)
    discovered_roots = _discover_capture_roots(search_roots)
    candidates: List[Path] = []
    _append_unique(candidates, explicit_roots)
    _append_unique(candidates, discovered_roots)

    candidate_summaries = [
        _candidate_summary(
            capture_root=candidate,
            webapp_site_slug=webapp_site_slug,
            simulator=simulator,
            provisioner=provisioner,
            simulator_command=simulator_command,
            require_webapp_forwarding=require_webapp_forwarding,
            require_webapp_staged_request=require_webapp_staged_request,
            require_gpu_gates=require_gpu_gates,
        )
        for candidate in sorted(candidates)
    ]
    ready_candidates = [
        candidate
        for candidate in candidate_summaries
        if candidate["ready_for_first_gpu_attempt"]
    ]
    video_backed_candidates = [
        candidate for candidate in candidate_summaries if candidate["has_raw_video"]
    ]
    blockers = []
    if not candidate_summaries:
        blockers.append("no_capture_roots_found")
    if candidate_summaries and not ready_candidates:
        blockers.append("no_ready_first_gpu_candidates")
    result = {
        "schema_version": FIRST_GPU_CANDIDATE_AUDIT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "ready_candidate_found" if ready_candidates else "blocked",
        "search_roots": [str(Path(path).expanduser()) for path in search_roots],
        "explicit_capture_roots": [str(Path(path).expanduser()) for path in capture_roots],
        "candidate_count": len(candidate_summaries),
        "video_backed_candidate_count": len(video_backed_candidates),
        "ready_candidate_count": len(ready_candidates),
        "candidates": candidate_summaries,
        "blockers": blockers,
        "claim_boundary": {
            "artifact_purpose": "first_gpu_candidate_discovery_audit",
            "live_provider_calls_performed": False,
            "webapp_requests_submitted": False,
            "simulator_execution_performed": False,
            "gpu_provisioning_performed": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
    output = Path(resolved_output_path).expanduser()
    ensure_dir(output.parent)
    write_json(output, result)
    return result | {"output_path": str(output)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Discover staged captures and audit first-GPU E2E readiness candidates"
    )
    parser.add_argument("--search-root", action="append", default=[])
    parser.add_argument("--capture-root", action="append", default=[])
    parser.add_argument("--webapp-site-slug", default="")
    parser.add_argument("--simulator", choices=SIMULATOR_FRAMEWORKS, default="isaac_sim")
    parser.add_argument("--provisioner", choices=PROVISIONERS, default="runpod")
    parser.add_argument("--simulator-command", default=None)
    parser.add_argument("--no-require-webapp-forwarding", action="store_true")
    parser.add_argument("--no-require-webapp-staged-request", action="store_true")
    parser.add_argument("--no-require-gpu-gates", action="store_true")
    parser.add_argument(
        "--output",
        default=str(default_artifact_cache_root() / "first_gpu_candidate_audit_manifest.json"),
    )
    args = parser.parse_args(argv)

    result = build_first_gpu_candidate_audit(
        search_roots=args.search_root,
        capture_roots=args.capture_root,
        webapp_site_slug=args.webapp_site_slug,
        simulator=args.simulator,
        provisioner=args.provisioner,
        simulator_command=args.simulator_command,
        require_webapp_forwarding=not args.no_require_webapp_forwarding,
        require_webapp_staged_request=not args.no_require_webapp_staged_request,
        require_gpu_gates=not args.no_require_gpu_gates,
        output_path=args.output,
    )
    print(f"[first-gpu-candidate-audit] status={result['status']}")
    print(f"[first-gpu-candidate-audit] candidates={result['candidate_count']}")
    print(f"[first-gpu-candidate-audit] ready_candidates={result['ready_candidate_count']}")
    print(f"[first-gpu-candidate-audit] manifest={result['output_path']}")
    if result["blockers"]:
        print("[first-gpu-candidate-audit] blockers=" + ",".join(result["blockers"]))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
