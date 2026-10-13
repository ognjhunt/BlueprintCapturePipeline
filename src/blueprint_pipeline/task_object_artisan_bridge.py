"""Run the real upstream CLI, then describe its actual storage/export manifests."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from .task_object_artisan_hooks import CONTEXT_ENV
from .task_object_artisan_metering import Meter, install_metering


def main() -> int:
    context = json.loads(Path(os.environ["BLUEPRINT_ARTISAN_BRIDGE_CONTEXT_FILE"]).read_text())
    os.environ[CONTEXT_ENV] = json.dumps(context["sandbox"])
    # Upstream creates its workspace after the CLI's existing-run check. The
    # tracked hook preflights confinement before its first model request.
    # Canonical scoped credential exists only in this controller. The generated
    # worker's environment and OS read roots exclude its file and value.
    key_path = Path(context["api_key_file"])
    if key_path.is_symlink() or not key_path.is_file() or key_path.stat().st_mode & 0o077:
        raise ValueError("artisan_scoped_credential_file_invalid")
    os.environ["OPENAI_API_KEY"] = key_path.read_text().strip()
    if context.get("project_id"):
        os.environ["OPENAI_PROJECT_ID"] = context["project_id"]
    os.environ["ARTISAN_BLENDER_BIN"] = context["sandbox"]["blender"]
    os.environ["ARTISAN_CACHE_DIR"] = str(Path(context["workspace"]) / "cache")
    os.environ["BLUEPRINT_ARTISAN_OBJECT_KIND"] = context["object_kind"]
    from artisan.config import apply_runtime_config
    from pipeline.llm.model_registry import validate_runtime_configuration
    apply_runtime_config(context["model_configuration"], {})
    models = validate_runtime_configuration(require_credentials=True)
    # Read-only availability checks precede the first billed request. Retain
    # server observations; model support in a local registry is insufficient.
    from openai import OpenAI
    with OpenAI(max_retries=0, timeout=30) as client:
        availability = {model: client.models.retrieve(model).id for model in
                        set(models["text_models"].values()) | {models["compaction_model"],
                            models["image_model"], models["image_tool_model"]}}
    meter = Meter(Path(context["usage_file"]),
        maximum_cost_usd=context["maximum_cost_usd"], maximum_requests=context["maximum_requests"],
        deadline_epoch=context["deadline_epoch"])
    install_metering(meter)
    from artisan.cli import main as upstream_main
    status = upstream_main(sys.argv[1:])
    from pipeline.state import load_checkpoint
    from pipeline.workspace import paths
    state = load_checkpoint(Path(context["workspace"]), context["upstream_run_id"])
    layout = paths(Path(context["workspace"]), state.run_id)
    from artisan.cli import _canonical_record_id
    record_id = _canonical_record_id(Path(context["workspace"]), state.run_id)
    value = {"schema_version": "blueprint_artisan_terminal.v1", "returncode": status,
             "run_id": state.run_id, "phase": state.phase.value, "status": state.status,
             "registered": record_id is not None, "canonical_record_id": record_id,
             "cost_usd": state.cost_usd, "candidate_id": state.candidate_id,
             "reviewed_physics_report_sha256": state.reviewed_physics_report_sha256,
             "termination_reason": state.termination_reason, "limitations": state.terminal_details,
             "paths": {key: str(path) for key, path in layout.items()},
             "usage": meter.read()}
    value["model_configuration"] = models
    value["model_availability"] = availability
    Path(context["terminal_file"]).write_text(json.dumps(value, allow_nan=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
