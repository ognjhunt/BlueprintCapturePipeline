#!/usr/bin/env python3
"""Tiny deterministic Postshot stand-in for hermetic worker simulations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=[
        "healthy",
        "gpu-active-quiet",
        "silent-hung",
        "dead",
        "nonzero",
        "secret-echo",
        "p1-success-p2-failed",
        "watcher-restart",
        "heartbeat-loss",
        "results-uploaded-instance-fails",
    ])
    parser.add_argument("--profile", default="Splat3")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--secret-email", default="")
    parser.add_argument("--secret-password", default="")
    parser.add_argument("--filename", default="scene.ply")
    parser.add_argument("--hang-seconds", type=float, default=60.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train-log.txt"
    log_path.write_text(json.dumps({"event": "started", "profile": args.profile}) + "\n", encoding="utf-8")

    if args.scenario == "silent-hung":
        time.sleep(max(0.0, args.hang_seconds))
        return 0
    if args.scenario == "dead":
        return 73
    if args.scenario == "nonzero":
        log_path.write_text(log_path.read_text(encoding="utf-8") + "exit=109\n", encoding="utf-8")
        return 109
    if args.scenario == "secret-echo":
        log_path.write_text(log_path.read_text(encoding="utf-8") + f"email={args.secret_email} password={args.secret_password}\n", encoding="utf-8")
        return 0
    if args.scenario == "gpu-active-quiet":
        # The worker must rely on GPU telemetry, not a fabricated log line.
        (args.output_dir / "gpu-active.marker").write_text("gpu telemetry is supplied by the harness\n", encoding="utf-8")
        return 0
    if args.scenario == "watcher-restart":
        (args.output_dir / "watcher-restart.marker").write_text("restart-safe state is external\n", encoding="utf-8")
        return 0
    if args.scenario == "heartbeat-loss":
        # No pulse is emitted by the fake; the policy layer owns the five-minute kill.
        return 0
    if args.scenario == "results-uploaded-instance-fails":
        (args.output_dir / "results-uploaded.marker").write_text("result uploaded before instance failure\n", encoding="utf-8")
        return 0
    if args.scenario == "p1-success-p2-failed" and args.profile.lower().replace(" ", "") == "splatmcmc":
        log_path.write_text(log_path.read_text(encoding="utf-8") + "profile failed after P1\n", encoding="utf-8")
        return 7

    log_path.write_text(log_path.read_text(encoding="utf-8") + "progress=1\nprogress=2\n", encoding="utf-8")
    (args.output_dir / args.filename).write_bytes(b"fake-postshot-output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
