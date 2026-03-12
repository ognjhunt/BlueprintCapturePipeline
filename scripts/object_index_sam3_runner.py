#!/usr/bin/env python3
"""SAM3 adapter for object-index stage.

Wraps the existing sam3_detect.py workflow and converts its output into the
object-index stage payload contract.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SAM3_DETECT = REPO_ROOT / "scripts" / "sam3_detect.py"


def _read_payload(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _environment(payload: Mapping[str, Any]) -> str:
    env = str(payload.get("environment") or "auto").strip().lower()
    return env if env in {"auto", "default", "warehouse", "kitchen", "bedroom", "office"} else "auto"


def _run_sam3(payload: Mapping[str, Any]) -> Dict[str, Any]:
    video_path = Path(str(payload.get("video_path") or "").strip())
    if not video_path.is_file():
        return {"objects": [], "backend_status": "skipped", "reason": "video_not_found"}
    if not SAM3_DETECT.is_file():
        return {"objects": [], "backend_status": "failed", "reason": f"missing_script:{SAM3_DETECT}"}

    with tempfile.TemporaryDirectory(prefix="object_index_sam3_") as tmp_dir:
        output_path = Path(tmp_dir) / "object_index.json"
        env = os.environ.copy()
        prompt_bank = payload.get("prompt_bank") if isinstance(payload.get("prompt_bank"), Mapping) else {}
        prompts = prompt_bank.get("all") if isinstance(prompt_bank.get("all"), list) else []
        if prompts:
            env["SAM3_DETECTION_PROMPTS"] = json.dumps(prompts)
        command = [
            sys.executable,
            str(SAM3_DETECT),
            "--video",
            str(video_path),
            "--output",
            str(output_path),
            "--environment",
            _environment(payload),
            "--tracking-backend",
            "image_model",
            "--n-frames",
            str(int(payload.get("sam3_n_frames") or 8)),
            "--min-frame-detections",
            str(int(payload.get("sam3_min_frame_detections") or 1)),
        ]
        proc = subprocess.run(command, check=False, text=True, capture_output=True, env=env)
        report: Dict[str, Any] = {
            "backend_status": "ok" if output_path.is_file() else "failed",
            "return_code": proc.returncode,
            "command": command,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
        if not output_path.is_file():
            report["objects"] = []
            return report
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        objects = payload.get("objects") if isinstance(payload, Mapping) and isinstance(payload.get("objects"), list) else []
        report["objects"] = [dict(item) for item in objects if isinstance(item, Mapping)]
        return report


def main(argv: List[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) != 2:
        print("usage: object_index_sam3_runner.py <input_json> <output_json>", file=sys.stderr)
        return 2
    input_path = Path(args[0])
    output_path = Path(args[1])
    payload = _read_payload(input_path)
    result = _run_sam3(payload)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
