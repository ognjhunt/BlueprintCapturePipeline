"""CLI for agent review over an existing pipeline capture."""

from __future__ import annotations

import argparse
from typing import List, Optional

from .agent_runtime.orchestrator import run_agent_review


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run agent review over a local capture pipeline")
    parser.add_argument("--capture-root", required=True, help="Local capture root path")
    parser.add_argument("--provider", required=True, choices=("claude", "openai"))
    parser.add_argument("--mode", default="qualification", choices=("qualification",))
    args = parser.parse_args(argv)

    try:
        payload = run_agent_review(
            capture_root=args.capture_root,
            provider_name=args.provider,
            mode=args.mode,
        )
    except Exception as exc:
        print(f"[agent-review] FAILED: {exc}")
        return 1

    print(f"[agent-review] provider={payload['provider']} readiness={payload['readiness_state']}")
    print(f"[agent-review] final_memo={payload['final_memo_path']}")
    print(f"[agent-review] final_bundle={payload['final_bundle_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
