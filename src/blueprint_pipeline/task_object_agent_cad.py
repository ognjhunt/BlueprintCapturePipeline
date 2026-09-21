"""Execute agent-authored CAD through the existing pinned CLI and sandbox.

ADP-009B/day-21: expose the compiler as a tool so one author can repair its
program from actual errors without invoking a second planning/model chain.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
import sys

from .astra_cad_skill_runtime import _read_step, verify_cad_sources
from .task_object_astra_authoring import AssetAuthoringError, file_record, save_json


def execute_cad_program(*, program, output_root, request, cad_root, mac_root,
                        sandbox, verified_sources=None):
    if not isinstance(program, str) or not program.strip() or len(program.encode()) > 60_000:
        raise AssetAuthoringError("agent_cad_program_invalid")
    tree = ast.parse(program)
    if not any(isinstance(node, ast.FunctionDef) and node.name == "gen_step" for node in tree.body):
        raise AssetAuthoringError("agent_cad_gen_step_required")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    sandbox.preflight()
    sources = verify_cad_sources(mac_root, cad_root, verified_sources)
    save_json(root / "source-receipt.json", sources)
    script, step, stl = root / "asset.py", root / "candidate.step", root / "candidate.stl"
    script.write_text(program, encoding="utf-8")
    # This is the same CAD skill CLI used by execute_mac_candidate. Export and
    # readback remain outside model-authored code; there is no extra model call.
    loader = [str(cad_root / "packages/cadpy/src"), *os.environ.get("PYTHONPATH", "").split(os.pathsep)]
    completed = sandbox([sys.executable, str(cad_root / "skills/cad/scripts/step"), str(script),
        "--output", str(step), "--stl", stl.name, "--mesh-tolerance", "0.005",
        "--mesh-angular-tolerance", "0.1"], cwd=str(root), timeout=120,
        env={"PYTHONPATH": os.pathsep.join(p for p in loader if p)},
        capture_output=True, text=True, check=False)
    (root / "stdout.txt").write_text((completed.stdout or "")[-100000:])
    (root / "stderr.txt").write_text((completed.stderr or "")[-100000:])
    if verify_cad_sources(mac_root, cad_root, verified_sources) != sources:
        raise AssetAuthoringError("agent_cad_source_changed")
    if completed.returncode:
        raise AssetAuthoringError("CAD compiler failed:\n" + (completed.stderr or "")[-6000:]
                                 + (completed.stdout or "")[-2000:])
    step_record, stl_record = file_record(step), file_record(stl)
    readback = _read_step(step, tuple(v * 1000 for v in request.dimensions_m),
                          request.maximum_export_error_m * 1000)
    save_json(root / "step-readback.json", readback)
    if not readback["passed"]:
        raise AssetAuthoringError("CAD geometry rejected: " + str(readback))
    receipt = {"claim": "development_only_candidate", "passed": True, "graph_invoked": False,
        "execution": "agent_program_through_pinned_cad_cli", "source_unchanged_after": True,
        "source_receipt": file_record(root / "source-receipt.json"), "program": file_record(script),
        "step": step_record, "stl": stl_record, "readback": readback,
        "step_path": str(step), "stl_path": str(stl),
        "measured_dimensions_m": [v / 1000 for v in readback["measured_dimensions_mm"]]}
    save_json(root / "candidate-receipt.json", receipt)
    return receipt
