"""CLI for agent review over an existing pipeline capture."""

from __future__ import annotations

import argparse
from typing import List, Optional

from .agent_runtime.orchestrator import run_agent_review
from .agent_runtime.openai_phase2 import OpenAIPhase2Config


def _openai_phase2_config_from_args(args: argparse.Namespace) -> Optional[OpenAIPhase2Config]:
    mode = str(getattr(args, "openai_phase2_mode", "") or "").strip()
    model = str(getattr(args, "openai_phase2_model", "") or "").strip()
    codex_bin = str(getattr(args, "openai_phase2_codex_bin", "") or "").strip()
    timeout_seconds = getattr(args, "openai_phase2_timeout_seconds", None)
    reasoning_effort = str(getattr(args, "openai_phase2_reasoning_effort", "") or "").strip()
    if not any([mode, model, codex_bin, timeout_seconds, reasoning_effort]):
        return None
    env_default = OpenAIPhase2Config.from_env()
    return OpenAIPhase2Config(
        mode=mode or env_default.mode,
        model=model or env_default.model,
        codex_bin=codex_bin or env_default.codex_bin,
        timeout_seconds=int(timeout_seconds or env_default.timeout_seconds),
        reasoning_effort=reasoning_effort or env_default.reasoning_effort,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run agent review over a local capture pipeline")
    parser.add_argument("--capture-root", required=True, help="Local capture root path")
    parser.add_argument("--provider", required=True, choices=("claude", "openai"))
    parser.add_argument("--mode", default="qualification", choices=("qualification",))
    parser.add_argument("--openai-phase2-mode", choices=("disabled", "codex_cli"))
    parser.add_argument("--openai-phase2-model")
    parser.add_argument("--openai-phase2-codex-bin")
    parser.add_argument("--openai-phase2-timeout-seconds", type=int)
    parser.add_argument("--openai-phase2-reasoning-effort")
    args = parser.parse_args(argv)

    try:
        payload = run_agent_review(
            capture_root=args.capture_root,
            provider_name=args.provider,
            mode=args.mode,
            openai_phase2_config=_openai_phase2_config_from_args(args),
        )
    except Exception as exc:
        print(f"[agent-review] FAILED: {exc}")
        return 1

    print(f"[agent-review] provider={payload['provider']} readiness={payload['readiness_state']}")
    print(f"[agent-review] final_memo={payload['final_memo_path']}")
    print(f"[agent-review] final_bundle={payload['final_bundle_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
