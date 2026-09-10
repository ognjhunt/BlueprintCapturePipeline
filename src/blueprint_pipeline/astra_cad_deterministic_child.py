"""Trusted child entry point. Must only be launched by SandboxedAssetRunner.

The pinned translator may execute generated Python through cadpy, so this entire
module runs inside the OS boundary, with no model credentials or network.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback


def main() -> None:
    request_path, result_path = map(Path, sys.argv[1:3])
    request = json.loads(request_path.read_text())
    sys.dont_write_bytecode = True
    sys.modules.update(aider=None, litellm=None)
    from multi_agent_cad import nodes
    from multi_agent_cad.schemas import ArchitectPlan, CADBrief

    def deny(*args, **kwargs):
        raise RuntimeError("deterministic_child_external_call_forbidden")

    nodes._llm_client = nodes.OpenAI = nodes._fill_unsupported_with_aider = deny
    state = request["state"]
    if state.get("cad_brief"):
        state["cad_brief"] = CADBrief.model_validate(state["cad_brief"])
    plan = ArchitectPlan.model_validate(request["architect_plan"])
    state["architect_plan"] = plan
    try:
        result = nodes._node_python_coder_deterministic(state, plan, request["iteration"])
        envelope = {"status": "unsupported" if result is None else "completed", "result": result}
    except Exception:
        envelope = {"status": "failed", "error": traceback.format_exc()}
    result_path.write_text(json.dumps(envelope, indent=2, default=lambda value:
        value.model_dump(mode="json") if hasattr(value, "model_dump") else str(value)) + "\n")


if __name__ == "__main__":
    main()
