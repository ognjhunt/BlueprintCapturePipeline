"""Fail-closed environment guard for production runtime units."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from .common import parse_bool
from .launch_proof_policy import PRODUCTION_MODE, launch_proof_mode

SCHEMA_VERSION = "blueprint.production_runtime_env_guard.v1"

REQUIRED_TRUE_FLAGS = (
    "PRIVACY_PIPELINE_ENABLED",
    "PRIVACY_FAIL_CLOSED",
    "PIPELINE_SYNC_REQUIRED",
    "RETRIEVAL_REQUIRE_PRIVACY_SAFE_VIDEO",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_production_runtime_env_guard(
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    source = os.environ if env is None else env
    blockers: list[str] = []
    mode = launch_proof_mode(source)
    if mode != PRODUCTION_MODE:
        blockers.append("missing_BLUEPRINT_LAUNCH_PROOF_MODE_production")

    flag_status: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_TRUE_FLAGS:
        raw = source.get(name)
        enabled = parse_bool(raw, default=False)
        flag_status[name] = {
            "configured": raw is not None and str(raw).strip() != "",
            "enabled": enabled,
        }
        if not enabled:
            blockers.append(f"missing_or_false_{name}")

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "launch_proof_mode": mode or None,
        "required_true_flags": flag_status,
        "claim_boundary": (
            "This guard verifies production fail-closed runtime posture only. "
            "It is not proof of deployed health, Pub/Sub message consumption, "
            "WebApp forwarding, buyer delivery, simulator execution, or live "
            "provider success."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print the full guard report.")
    args = parser.parse_args(argv)
    report = build_production_runtime_env_guard()
    if args.json or report["status"] != "ready":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("[production-runtime-env-guard] status=ready")
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
