"""Production-only agent authoring for one supplemental passive destination.

The agent writes a STEP-first build123d generator from owner-authored metric
constraints.  It may not browse, change dimensions, grant rights, or qualify
its own output.  The pinned CAD skill executes the generator and deterministic
inspection reopens the STEP before any static/native SimReady consumer may use
it.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import subprocess  # nosec B404 - argv is fixed to a pinned skill checkout
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .decision_evidence_contracts import canonical_digest
from .production_cad_skill_sources import (
    DEFAULT_ROOT as DEFAULT_CAD_SOURCE_ROOT,
    SOURCE_SPECS,
    validate_production_cad_skill_sources,
)
from .simready_cad_agent_contract import materialize_step_inspection_receipt
from .task_evaluation_supervisor.agents_sdk import (
    AgentsSDKAgentSpec,
    AgentsSDKInvoker,
    OpenAIAgentsSDKConfig,
    OpenAIAgentsSDKInvoker,
)


REQUEST_SCHEMA_VERSION = "task_evaluation_passive_destination_cad_request.v1"
RESULT_SCHEMA_VERSION = "task_evaluation_passive_destination_cad_result.v1"
MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "high"
MAX_OUTPUT_TOKENS = 12_000
DEFAULT_MAX_COST_USD = 0.75
_FORBIDDEN_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "input",
    "locals",
    "open",
    "vars",
}
_FORBIDDEN_MODULES = {
    "http",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "urllib",
}


class PassiveDestinationCadAgentError(RuntimeError):
    """The request, agent proposal, or deterministic CAD output was invalid."""


class PassiveDestinationCadOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cad_brief_markdown: str = Field(min_length=1, max_length=12_000)
    generator_source: str = Field(min_length=1, max_length=40_000)
    outer_x_mm: float = Field(gt=0)
    outer_y_mm: float = Field(gt=0)
    base_thickness_mm: float = Field(gt=0)
    wall_thickness_mm: float = Field(gt=0)
    wall_height_above_base_mm: float = Field(gt=0)
    assumptions: list[str] = Field(default_factory=list, max_length=30)
    cited_web_sources: list[str] = Field(default_factory=list, max_length=20)
    uncertainty: str = Field(min_length=1, max_length=2_000)

    @field_validator(
        "outer_x_mm",
        "outer_y_mm",
        "base_thickness_mm",
        "wall_thickness_mm",
        "wall_height_above_base_mm",
    )
    @classmethod
    def _finite(cls, value: float) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("passive_destination_cad_value_non_finite")
        return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _file(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise PassiveDestinationCadAgentError("passive_destination_cad_file_invalid")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def validate_passive_destination_cad_request(value: Mapping[str, Any]) -> dict[str, Any]:
    request = json.loads(json.dumps(dict(value), allow_nan=False))
    identity = request.get("destination_identity")
    dimensions = request.get("dimensions_m")
    rights = request.get("rights")
    backend = request.get("cad_backend")
    expected_backend = SOURCE_SPECS[0]
    required_dimensions = {
        "outer_x",
        "outer_y",
        "base_thickness",
        "wall_thickness",
        "wall_height_above_base",
        "minimum_interior_x",
        "minimum_interior_y",
    }
    if (
        request.get("schema_version") != REQUEST_SCHEMA_VERSION
        or not str(request.get("run_id") or "")
        or len(str(request.get("expected_production_commit") or "")) != 40
        or not isinstance(identity, Mapping)
        or set(identity) != {"id", "version"}
        or not all(str(identity.get(name) or "") for name in identity)
        or request.get("relation") != "inside"
        or not str(request.get("visible_label") or "")
        or not isinstance(dimensions, Mapping)
        or set(dimensions) != required_dimensions
        or any(
            isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in dimensions.values()
        )
        or float(dimensions["outer_x"])
        - 2.0 * float(dimensions["wall_thickness"])
        < float(dimensions["minimum_interior_x"])
        or float(dimensions["outer_y"])
        - 2.0 * float(dimensions["wall_thickness"])
        < float(dimensions["minimum_interior_y"])
        or not isinstance(backend, Mapping)
        or any(
            backend.get(field) != expected_backend[field]
            for field in ("repository", "commit", "tree", "license", "license_sha256")
        )
        or backend.get("skill") != "cad"
        or backend.get("agent_model") != MODEL
        or backend.get("web_research_allowed") is not False
        or not isinstance(rights, Mapping)
        or rights.get("generated_for_private_development") is not True
        or rights.get("private_provider_processing_allowed") is not True
        or rights.get("provider_training_allowed") is not False
        or rights.get("public_redistribution_allowed") is not False
        or request.get("maximum_agent_cost_usd") != DEFAULT_MAX_COST_USD
        or request.get("automatic_retries") != 0
        or request.get("request_digest")
        != canonical_digest(request, digest_field="request_digest")
    ):
        raise PassiveDestinationCadAgentError(
            "passive_destination_cad_request_invalid"
        )
    return request


def materialize_passive_destination_cad_request(
    *,
    run_id: str,
    expected_production_commit: str,
    destination_identity: Mapping[str, str],
    visible_label: str,
    dimensions_m: Mapping[str, float],
    output_path: str | Path,
) -> dict[str, Any]:
    spec = SOURCE_SPECS[0]
    value: dict[str, Any] = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "run_id": run_id,
        "expected_production_commit": expected_production_commit,
        "destination_identity": dict(destination_identity),
        "relation": "inside",
        "visible_label": visible_label,
        "dimensions_m": {name: float(item) for name, item in dimensions_m.items()},
        "cad_backend": {
            "repository": spec["repository"],
            "commit": spec["commit"],
            "tree": spec["tree"],
            "license": spec["license"],
            "license_sha256": spec["license_sha256"],
            "skill": "cad",
            "agent_model": MODEL,
            "web_research_allowed": False,
        },
        "rights": {
            "generated_for_private_development": True,
            "private_provider_processing_allowed": True,
            "provider_training_allowed": False,
            "public_redistribution_allowed": False,
        },
        "maximum_agent_cost_usd": DEFAULT_MAX_COST_USD,
        "automatic_retries": 0,
        "request_digest": "",
    }
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    validated = validate_passive_destination_cad_request(value)
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    payload = json.dumps(validated, sort_keys=True, separators=(",", ":")) + "\n"
    try:
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        destination.chmod(0o440)
    except FileExistsError:
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != payload:
            raise PassiveDestinationCadAgentError(
                "passive_destination_cad_request_conflict"
            ) from None
    return validated


def _stable_instructions(skill_text: str) -> str:
    return (
        "You are Blueprint's production passive-destination CAD author. "
        "Write one STEP-first build123d generator from immutable metric constraints. "
        "The source must define gen_step(), return one labeled positive-volume solid, "
        "use millimeters, center XY at the origin, place its support plane at Z=0, "
        "and perform no file, process, environment, network, or dynamic-import access. "
        "Do not change dimensions, physics, rights, task semantics, or identifiers. "
        "Do not browse the web; cited_web_sources must be empty. Geometry is a proposal "
        "that independent deterministic and native validators may reject. Return raw "
        "Python in generator_source without markdown fences.\n\nPinned CAD skill:\n"
        + skill_text
        + "\n\nStructured output schema:\n"
        + json.dumps(
            PassiveDestinationCadOutput.model_json_schema(),
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _validate_generator_source(source: str) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise PassiveDestinationCadAgentError(
            "passive_destination_cad_generator_syntax_invalid"
        ) from exc
    functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    if "gen_step" not in functions:
        raise PassiveDestinationCadAgentError(
            "passive_destination_cad_gen_step_missing"
        )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name != "build123d" for alias in node.names):
                raise PassiveDestinationCadAgentError(
                    "passive_destination_cad_generator_import_forbidden"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module != "build123d" or node.level:
                raise PassiveDestinationCadAgentError(
                    "passive_destination_cad_generator_import_forbidden"
                )
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise PassiveDestinationCadAgentError(
                "passive_destination_cad_generator_operation_forbidden"
            )
        elif isinstance(node, ast.Attribute) and (
            node.attr.startswith("__")
            or isinstance(node.value, ast.Name)
            and node.value.id in _FORBIDDEN_MODULES
        ):
            raise PassiveDestinationCadAgentError(
                "passive_destination_cad_generator_operation_forbidden"
            )
        elif isinstance(
            node,
            (
                ast.AsyncFunctionDef,
                ast.Await,
                ast.ClassDef,
                ast.Global,
                ast.Lambda,
                ast.Nonlocal,
                ast.Try,
                ast.With,
            ),
        ):
            raise PassiveDestinationCadAgentError(
                "passive_destination_cad_generator_operation_forbidden"
            )


def _expected_mm(request: Mapping[str, Any]) -> dict[str, float]:
    dimensions = request["dimensions_m"]
    return {
        "outer_x_mm": float(dimensions["outer_x"]) * 1000.0,
        "outer_y_mm": float(dimensions["outer_y"]) * 1000.0,
        "base_thickness_mm": float(dimensions["base_thickness"]) * 1000.0,
        "wall_thickness_mm": float(dimensions["wall_thickness"]) * 1000.0,
        "wall_height_above_base_mm": float(dimensions["wall_height_above_base"])
        * 1000.0,
    }


def execute_passive_destination_cad_agent(
    *,
    request_path: str | Path,
    output_root: str | Path,
    cad_source_root: str | Path = DEFAULT_CAD_SOURCE_ROOT,
    invoker: AgentsSDKInvoker,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    request_file = Path(request_path).expanduser().resolve()
    request = validate_passive_destination_cad_request(
        json.loads(request_file.read_text(encoding="utf-8"))
    )
    repository_root = Path(__file__).resolve().parents[2]
    head = subprocess.run(  # nosec B603 - fixed git argv
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != request["expected_production_commit"]:
        raise PassiveDestinationCadAgentError(
            "passive_destination_cad_execution_commit_mismatch"
        )
    sources = validate_production_cad_skill_sources(cad_source_root)
    source_by_id = {row["id"]: row for row in sources["sources"]}
    text_root = Path(source_by_id["text-to-cad"]["path"])
    skill_path = text_root / "skills/cad/SKILL.md"
    skill_text = skill_path.read_text(encoding="utf-8")
    output = Path(output_root).expanduser()
    output.mkdir(parents=True, exist_ok=False, mode=0o750)
    dimensions = _expected_mm(request)
    spec = AgentsSDKAgentSpec(
        run_id=request["run_id"],
        capability="passive_destination_cad_authoring",
        name="Blueprint passive destination CAD author",
        instructions=_stable_instructions(skill_text),
        model=MODEL,
        max_turns=1,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        output_type=PassiveDestinationCadOutput,
        stable_developer_prefix=_stable_instructions(skill_text),
        dynamic_suffix_fields=("request",),
    )
    invocation = invoker.invoke(
        spec,
        json.dumps(
            {"request": request, "required_dimensions_mm": dimensions},
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    proposal = PassiveDestinationCadOutput.model_validate(invocation.output)
    if proposal.cited_web_sources or any(
        not math.isclose(
            float(getattr(proposal, name)), expected, rel_tol=0.0, abs_tol=1.0e-9
        )
        for name, expected in dimensions.items()
    ):
        raise PassiveDestinationCadAgentError(
            "passive_destination_cad_agent_changed_constraints"
        )
    _validate_generator_source(proposal.generator_source)
    generator = output / "passive_destination.py"
    generator.write_text(proposal.generator_source, encoding="utf-8")
    brief = output / "CAD_BRIEF.md"
    brief.write_text(proposal.cad_brief_markdown + "\n", encoding="utf-8")
    step = output / "passive_destination.step"
    stl = output / "passive_destination.stl"
    glb = output / "passive_destination.glb"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(text_root / "packages/cadpy/src"),
            str(text_root / "packages/cadpy_metadata/src"),
        ]
    )
    completed = runner(
        [
            python_executable,
            str(text_root / "skills/cad/scripts/step"),
            str(generator),
            "--output",
            str(step),
            "--stl",
            str(stl),
            "--glb",
            str(glb),
            "--force",
        ],
        cwd=output,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode != 0:
        raise PassiveDestinationCadAgentError(
            "passive_destination_cad_skill_execution_failed"
        )
    inspection_path = output / "step_inspection.v1.json"
    inspection = materialize_step_inspection_receipt(
        step_path=step, output_path=inspection_path
    )
    expected_envelope = [
        dimensions["outer_x_mm"],
        dimensions["outer_y_mm"],
        dimensions["base_thickness_mm"]
        + dimensions["wall_height_above_base_mm"],
    ]
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=0.25)
        for actual, expected in zip(
            inspection["measured_envelope_mm"], expected_envelope, strict=True
        )
    ):
        raise PassiveDestinationCadAgentError(
            "passive_destination_cad_metric_envelope_mismatch"
        )
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "candidate_authored_pending_visual_static_native_qualification",
        "run_id": request["run_id"],
        "request": _file(request_file),
        "request_digest": request["request_digest"],
        "destination_identity": request["destination_identity"],
        "agent": {
            "provider": invocation.provider,
            "model": invocation.model,
            "sdk_version": invocation.sdk_version,
            "usage": dict(invocation.usage),
            "actual_cost_usd": invocation.cost_usd,
            "cost_status": invocation.cost_status,
            "web_research_allowed": False,
            "web_research_performed": False,
            "candidate_only": True,
        },
        "cad_skill_source": {
            "receipt_digest": sources["receipt_digest"],
            "skill_file": _file(skill_path),
            "commit": source_by_id["text-to-cad"]["commit"],
            "tree": source_by_id["text-to-cad"]["tree"],
        },
        "artifacts": {
            "cad_brief": _file(brief),
            "generator_source": _file(generator),
            "step": _file(step),
            "stl": _file(stl),
            "glb": _file(glb),
            "inspection": _file(inspection_path),
        },
        "measured_envelope_mm": inspection["measured_envelope_mm"],
        "review_render_required": True,
        "static_qualification_required": True,
        "native_import_qualification_required": True,
        "scene_placement_qualification_required": True,
        "simready_qualified": False,
        "agent_self_grading_forbidden": True,
        "provider_mutations_performed": 0,
        "automatic_retries": 0,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    result_path = output / "passive_destination_cad_result.v1.json"
    result_path.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    request_parser = subparsers.add_parser("request")
    request_parser.add_argument("--input", required=True)
    request_parser.add_argument("--output", required=True)
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--request", required=True)
    execute_parser.add_argument("--output-root", required=True)
    execute_parser.add_argument("--cad-source-root", default=DEFAULT_CAD_SOURCE_ROOT)
    args = parser.parse_args(argv)
    if args.command == "request":
        raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
        result = materialize_passive_destination_cad_request(
            run_id=raw["run_id"],
            expected_production_commit=raw["expected_production_commit"],
            destination_identity=raw["destination_identity"],
            visible_label=raw["visible_label"],
            dimensions_m=raw["dimensions_m"],
            output_path=args.output,
        )
    else:
        config = OpenAIAgentsSDKConfig(
            model=MODEL,
            max_turns=1,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            allow_live_invocation=True,
            max_inference_cost_usd=DEFAULT_MAX_COST_USD,
            input_cost_per_million_tokens_usd=4.0,
            output_cost_per_million_tokens_usd=20.0,
        )
        result = execute_passive_destination_cad_agent(
            request_path=args.request,
            output_root=args.output_root,
            cad_source_root=args.cad_source_root,
            invoker=OpenAIAgentsSDKInvoker(config),
        )
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = [
    "DEFAULT_MAX_COST_USD",
    "MODEL",
    "PassiveDestinationCadAgentError",
    "PassiveDestinationCadOutput",
    "execute_passive_destination_cad_agent",
    "materialize_passive_destination_cad_request",
    "validate_passive_destination_cad_request",
]


if __name__ == "__main__":
    raise SystemExit(main())
