"""Bounded adapter for the immutable MAC graph; geometry remains development-only."""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from .task_evaluation_supervisor.agents_sdk import AgentsSDKAgentSpec, AgentsSDKInvoker

MAC_COMMIT = "42737c408534e7c00c63081d73ce7565a9464e56"
CAD_COMMIT = "4fd71ea75fbb8a80b0d7c76862e0fd73c52a8989"
_LOCK = threading.Lock()


class AstraCADRuntimeBlocked(RuntimeError):
    """Admission, budget, source identity, or geometry verification failed."""


class _TextOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str


def _save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=lambda v: v.model_dump(mode="json")
                               if isinstance(v, BaseModel) else str(v)) + "\n")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_source(root: Path, expected: str) -> dict[str, Any]:
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip()
    if commit != expected or dirty:
        raise AstraCADRuntimeBlocked("cad_source_not_exact_clean_pin")
    tracked = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"]).decode().split("\0")
    hashes = {name: _digest(root / name) for name in tracked if name.endswith((".py", ".md", ".toml"))}
    return {"root": str(root), "commit": commit, "tracked_file_sha256": hashes, "source_diff": ""}


def _deny(*_args: Any, **_kwargs: Any) -> Any:
    raise AstraCADRuntimeBlocked("direct_model_or_execution_bypass_denied")


class _SDKChatBridge:
    """The upstream response shape is adapted; upstream provider options confer no authority."""

    def __init__(self, invoker: AgentsSDKInvoker, root: Path, brief: str, run_id: str,
                 max_input_tokens: int, max_output_tokens: int, max_calls: int,
                 object_label: str = "candidate"):
        self.invoker, self.root, self.brief, self.run_id = invoker, root, brief, run_id
        self.max_input_tokens, self.max_output_tokens, self.max_calls = max_input_tokens, max_output_tokens, max_calls
        self.object_label = object_label
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, *, messages: list[dict[str, Any]], **_kwargs: Any) -> Any:
        if len(self.calls) >= self.max_calls:
            raise AstraCADRuntimeBlocked("cad_invocation_budget_exhausted")
        payload = json.dumps({"object_label": self.object_label,
                              "immutable_original_brief": self.brief, "upstream_messages": messages})
        # UTF-8 byte count is a conservative token upper bound, checked before invoking.
        if len(payload.encode()) > self.max_input_tokens:
            raise AstraCADRuntimeBlocked("cad_input_token_ceiling_exceeded")
        index = len(self.calls)
        record: dict[str, Any] = {"index": index, "model": "gpt-6-astra", "max_turns": 1,
                                  "input_sha256": hashlib.sha256(payload.encode()).hexdigest()}
        self.calls.append(record)  # Failed calls consume the local allowance, too.
        _save(self.root / f"invocation-{index:02d}-input.json", json.loads(payload))
        spec = AgentsSDKAgentSpec(
            run_id=self.run_id, capability=f"astra_cad_candidate:{self.object_label}", name="Astra CAD candidate",
            instructions=("Use the supplied immutable brief as hard constraints. Preserve every exact "
                          "dimension in millimeters without silent rounding. Return the requested JSON "
                          "or complete Python code in content. Generated CAD is development_only."),
            model="gpt-6-astra", reasoning_effort="high", max_turns=1,
            max_input_tokens=self.max_input_tokens, max_output_tokens=self.max_output_tokens,
            output_type=_TextOutput, tool_bindings=(),
        )
        try:
            result = self.invoker.invoke(spec, payload)
            content = result.output.content
            if not isinstance(content, str) or not content.strip():
                raise AstraCADRuntimeBlocked("cad_empty_model_output")
            record.update({"usage": dict(result.usage), "cost_usd": result.cost_usd,
                           "cost_status": result.cost_status, "trace_id": result.trace_id,
                           "provider": result.provider, "sdk_version": result.sdk_version,
                           "output_sha256": hashlib.sha256(content.encode()).hexdigest()})
            (self.root / f"invocation-{index:02d}-output.txt").write_text(content)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
        except Exception as exc:
            record["failure"] = type(exc).__name__
            raise
        finally:
            _save(self.root / "invocations.json", self.calls)


@contextlib.contextmanager
def _source_modules(mac: Path, cad: Path):
    """Import exact source in an isolated worker; restore interpreter state on exit."""
    names = ("multi_agent_cad", "cadpy")
    previous = {k: v for k, v in sys.modules.items() if k.startswith(names)}
    old_path, old_cwd, old_env = sys.path[:], Path.cwd(), dict(os.environ)
    old_bytecode = sys.dont_write_bytecode
    blocked_packages = {name: sys.modules.get(name) for name in ("aider", "litellm")}
    import openai.resources.chat.completions as completions
    original_create = completions.Completions.create
    try:
        for name in previous:
            sys.modules.pop(name, None)
        sys.path[:0] = [str(mac), str(cad / "packages/cadpy/src")]
        sys.dont_write_bytecode = True
        sys.modules.update(aider=None, litellm=None)
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"
        completions.Completions.create = _deny
        graph = importlib.import_module("multi_agent_cad.graph")
        nodes = importlib.import_module("multi_agent_cad.nodes")
        yield graph, nodes
    finally:
        completions.Completions.create = original_create
        for name in list(sys.modules):
            if name.startswith(names):
                sys.modules.pop(name, None)
        sys.modules.update(previous)
        for name, module in blocked_packages.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        sys.dont_write_bytecode = old_bytecode
        sys.path[:] = old_path
        os.chdir(old_cwd)
        os.environ.clear()
        os.environ.update(old_env)


def _read_step(step: Path, expected: tuple[float, float, float], tolerance: float) -> dict[str, Any]:
    import build123d
    shape = build123d.import_step(str(step))
    size = shape.bounding_box().size
    measured = [float(size.X), float(size.Y), float(size.Z)]
    valid = shape.is_valid() if callable(shape.is_valid) else shape.is_valid
    result = {"measured_dimensions_mm": measured, "expected_dimensions_mm": list(expected),
              "absolute_tolerance_mm": tolerance, "valid": bool(valid), "volume_mm3": float(shape.volume),
              "build123d": importlib.metadata.version("build123d")}
    result["kernel_versions"] = {package: importlib.metadata.version(package) for package in
                                 importlib.metadata.packages_distributions().get("OCP", [])}
    result["passed"] = bool(valid and shape.volume > 0 and all(
        math.isfinite(a) and abs(a - b) <= tolerance for a, b in zip(measured, expected)
    ))
    return result


def execute_mac_candidate(
    brief: str, output_root: Path, mac_source_root: Path, cad_source_root: Path,
    invoker: AgentsSDKInvoker, *, expected_dimensions_mm: tuple[float, float, float],
    subprocess_runner: Callable[..., Any] | None = None, run_id: str = "astra-cad-candidate",
    object_label: str = "candidate",
    max_input_tokens: int = 80_000, max_output_tokens: int = 12_000,
    max_calls: int = 7, repair_budget: int = 2, dimension_tolerance_mm: float = 0.01,
) -> dict[str, Any]:
    """Execute the real pinned graph through a supplied budgeted SDK invoker and sandbox.

    Run in a dedicated worker process. The caller supplies an executor with preflight(),
    filesystem confinement and no network; no ambient subprocess fallback is provided.
    """
    if subprocess_runner is None or not callable(getattr(subprocess_runner, "preflight", None)):
        raise AstraCADRuntimeBlocked("sandboxed_subprocess_runner_required")
    if not brief.strip() or not (0 <= repair_budget <= 2 and 1 <= max_calls <= 15):
        raise AstraCADRuntimeBlocked("cad_bounds_invalid")
    if not object_label.strip() or len(object_label) > 200:
        raise AstraCADRuntimeBlocked("cad_object_label_invalid")
    if not (256 <= max_output_tokens <= 32_000 and 1 <= max_input_tokens <= 120_000):
        raise AstraCADRuntimeBlocked("cad_token_bounds_invalid")
    if len(expected_dimensions_mm) != 3 or any(not math.isfinite(v) or v <= 0 for v in expected_dimensions_mm):
        raise AstraCADRuntimeBlocked("cad_exact_dimensions_required")
    if not math.isfinite(dimension_tolerance_mm) or not 0 <= dimension_tolerance_mm <= 0.1:
        raise AstraCADRuntimeBlocked("cad_dimension_tolerance_invalid")
    root, mac, cad = (Path(p).resolve() for p in (output_root, mac_source_root, cad_source_root))
    if root.exists() and any(root.iterdir()):
        raise AstraCADRuntimeBlocked("cad_output_root_not_empty")
    if root.is_relative_to(mac) or root.is_relative_to(cad):
        raise AstraCADRuntimeBlocked("cad_output_inside_immutable_source")
    subprocess_runner.preflight()
    sources = {"mac": _verify_source(mac, MAC_COMMIT), "cad": _verify_source(cad, CAD_COMMIT)}
    if importlib.util.find_spec("langgraph") is None or importlib.util.find_spec("build123d") is None:
        raise AstraCADRuntimeBlocked("cad_runtime_dependency_missing")
    root.mkdir(parents=True, exist_ok=True)
    _save(root / "source-receipt.json", sources)
    parameters = {"brief": brief, "object_label": object_label, "run_id": run_id,
                  "expected_dimensions_mm": list(expected_dimensions_mm),
                  "repair_budget": repair_budget, "max_calls": max_calls,
                  "max_input_tokens": max_input_tokens, "max_output_tokens": max_output_tokens,
                  "dimension_tolerance_mm": dimension_tolerance_mm}
    _save(root / "parameters.json", parameters)
    bridge = _SDKChatBridge(invoker, root, brief, run_id, max_input_tokens, max_output_tokens, max_calls, object_label)
    receipt: dict[str, Any] = {"claim": "development_only_candidate", "passed": False,
                               "source_receipt": str(root / "source-receipt.json")}
    with _LOCK, _source_modules(mac, cad) as (graph, nodes):
        os.chdir(root)
        nodes._CACHE_DIR = root / "pipeline_cache"
        nodes._llm_client = lambda: bridge
        nodes.OpenAI = _deny
        nodes._node_python_coder_deterministic = lambda *_args, **_kwargs: None
        nodes._fill_unsupported_with_aider = _deny
        nodes.generate_initial_solution = _deny
        # Printing orientation is unrelated to task coordinates and would desynchronize STEP/STL.
        nodes._optimize_print_orientation = lambda path: (path, "disabled_preserve_task_coordinates")
        nodes._CFG_MAX_RETRIES = repair_budget + 1  # one initial QA + at most two repairs
        nodes._CFG_MAX_EXEC_RETRIES = 0
        graph._MAX_SELF_RETRIES = 1
        nodes._SP_MODEL = nodes._ARCH_MODEL = nodes._CODER_MODEL = nodes._REPAIR_MODEL = "gpt-6-astra"
        nodes._SPEC_PLANNER_KWARGS = nodes._ARCHITECT_KWARGS = nodes._CODER_KWARGS = nodes._REPAIR_KWARGS = {}
        original_json = nodes._call_llm_json_with_retry
        nodes._call_llm_json_with_retry = lambda *a, **kw: original_json(*a, **{**kw, "max_retries": 1})
        repair_calls = 0

        def repair(*, script_path, error_details, user_request, **_kw):
            nonlocal repair_calls
            if repair_calls >= repair_budget:
                return False
            repair_calls += 1
            path = Path(script_path).resolve()
            if not path.is_relative_to(root):
                raise AstraCADRuntimeBlocked("cad_script_outside_scratch")
            before = path.read_text()
            response = bridge.create(messages=[{"role": "user", "content": json.dumps({
                "brief": user_request, "errors": error_details, "script": before,
                "task": "Return complete corrected Python code. Preserve exact millimeter dimensions."})}])
            code = nodes._extract_code_from_llm_response(response.choices[0].message.content)
            if not code.strip():
                return False
            (root / f"repair-{repair_calls}-before.py").write_text(before)
            (root / f"repair-{repair_calls}-after.py").write_text(code)
            path.write_text(code)
            return True

        def run(argv, **kwargs):
            # Every upstream executable subprocess goes through caller confinement.
            kwargs.pop("env", None)
            kwargs.pop("encoding", None)
            kwargs.pop("errors", None)
            kwargs["env"] = {"PYTHONPATH": os.pathsep.join((str(mac / "packages/cadpy/src"),
                                                            str(cad / "packages/cadpy/src")))}
            kwargs["cwd"] = str(root)
            kwargs["timeout"] = min(float(kwargs.get("timeout", 120)), 120)
            return subprocess_runner(argv, **kwargs)

        nodes._run_repair_on_script = repair
        nodes._run_direct_repair_fallback = _deny
        nodes.subprocess = SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired)
        nodes._prompt_iteration_choice = lambda **_kw: 1
        for name in ("node_spec_planner", "node_geometric_architect", "node_python_coder", "node_autonomous_skill_loop"):
            original = getattr(nodes, name)

            def guarded(state, fn=original):
                result = fn(state)
                brief_value = result.get("cad_brief")
                part_name = getattr(brief_value, "part_name", "")
                if any(c in part_name for c in ("/", "\\", "..")):
                    raise AstraCADRuntimeBlocked("cad_part_name_unsafe")
                for field in ("current_step_path", "current_stl_path", "current_python_code_path"):
                    if result.get(field) and not Path(result[field]).resolve().is_relative_to(root):
                        raise AstraCADRuntimeBlocked("cad_artifact_outside_scratch")
                _save(root / f"node-{fn.__name__}.json", result)
                return result

            setattr(graph, name, guarded)
        state = graph.get_default_initial_state()
        state.update(user_request=brief, force_refresh=True, max_iterations=repair_budget + 1)
        try:
            result = graph.build_graph().invoke(state, config={"recursion_limit": 8})
            receipt["graph_invoked"] = True
            _save(root / "graph-result.json", result)
            step, stl = Path(result.get("current_step_path") or ""), Path(result.get("current_stl_path") or "")
            if not step.is_file() or not stl.is_file():
                raise AstraCADRuntimeBlocked("cad_graph_missing_exports")
            readback = _read_step(step, expected_dimensions_mm, dimension_tolerance_mm)
            _save(root / "step-readback.json", readback)
            if not readback["passed"] or str(getattr(result.get("error_type"), "value", result.get("error_type"))) != "none":
                raise AstraCADRuntimeBlocked("cad_geometry_or_upstream_qa_failed")
            receipt.update(passed=True, step_path=str(step), stl_path=str(stl), readback=readback)
            receipt["source_unchanged_after"] = all(
                _verify_source(path, commit) == sources[label]
                for label, path, commit in (("mac", mac, MAC_COMMIT), ("cad", cad, CAD_COMMIT))
            )
        except Exception as exc:
            receipt["failure"] = str(exc)
            raise
        finally:
            receipt.update(invocation_count=len(bridge.calls), repair_calls=repair_calls)
            receipt["artifacts"] = {str(p.relative_to(root)): _digest(p) for p in root.rglob("*")
                                    if p.is_file() and p.suffix in (".py", ".step", ".stl", ".json", ".txt")}
            _save(root / "candidate-receipt.json", receipt)
    return receipt
