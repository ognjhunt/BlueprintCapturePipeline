#!/usr/bin/env python3
"""Dispatch NeoVerse remote jobs and normalize outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blueprint_pipeline.world_model_service_client import (
    WorldModelServiceClient,
    WorldModelServiceConfig,
)
from neoverse_contract_adapter import normalize_backend_manifest


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run NeoVerse remote service")
    parser.add_argument("--job-spec", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--capture-id", required=True)
    args = parser.parse_args(argv)

    job_spec = _load_json(Path(args.job_spec))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    backend_report_path = output_dir / "neoverse_backend_report.json"

    client = WorldModelServiceClient(WorldModelServiceConfig.from_env("neoverse"))
    result = client.wait_for_completion(
        backend="neoverse",
        scene_id=args.scene_id,
        capture_id=args.capture_id,
        job_spec=job_spec,
    )
    backend_report = {
        "schema_version": "v1",
        "backend": "neoverse",
        "job_id": result.job_id,
        "status": result.status,
        "model_version": result.payload.get("model_version"),
        "conditioning_used": result.payload.get("conditioning_used") or ["rgb_video"],
        "service_latency_seconds": round(float(result.latency_seconds), 6),
        "native_artifact_manifest_location": (
            result.payload.get("result_manifest_url")
            or result.payload.get("result_manifest_uri")
            or result.payload.get("result_manifest_path")
        ),
    }
    backend_report_path.write_text(json.dumps(backend_report, indent=2), encoding="utf-8")
    normalize_backend_manifest(
        result_manifest=dict(result.result_manifest),
        output_dir=output_dir,
        backend_report_path=backend_report_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
