"""Bounded adapter for the immutable MAC graph; geometry remains development-only."""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import threading
from types import SimpleNamespace
from typing import Any, Callable
import zipfile

from pydantic import BaseModel, ConfigDict

from .decision_evidence_contracts import canonical_digest
from .production_cad_skill_sources import SOURCE_SPECS
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


def _archive_file_inventory(path: Path) -> dict[str, str]:
    """Hash exact safe ZIP files without extracting or following member links."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 512 * 1024**2:
        raise AstraCADRuntimeBlocked("cad_source_archive_invalid")
    hashes: dict[str, str] = {}
    names: set[str] = set()
    total = 0
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > 100_000:
                raise AstraCADRuntimeBlocked("cad_source_archive_size_limit")
            for member in members:
                name = member.filename.rstrip("/") if member.is_dir() else member.filename
                parts = PurePosixPath(name)
                if (not name or "\\" in name or "\x00" in name or parts.is_absolute()
                        or ":" in parts.parts[0] or any(part in ("", ".", "..") for part in name.split("/"))
                        or parts.as_posix() != name or name in names):
                    raise AstraCADRuntimeBlocked("cad_source_archive_path_invalid")
                names.add(name)
                mode = (member.external_attr >> 16) & 0xFFFF
                kind = stat.S_IFMT(mode)
                if stat.S_ISLNK(mode) or kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise AstraCADRuntimeBlocked("cad_source_archive_link_or_special_file")
                if member.is_dir():
                    if kind not in (0, stat.S_IFDIR) or member.file_size:
                        raise AstraCADRuntimeBlocked("cad_source_archive_directory_invalid")
                    continue
                if kind == stat.S_IFDIR or member.flag_bits & 1:
                    raise AstraCADRuntimeBlocked("cad_source_archive_file_invalid")
                total += member.file_size
                if total > 1024**3:
                    raise AstraCADRuntimeBlocked("cad_source_archive_size_limit")
                digest = hashlib.sha256()
                with archive.open(member) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                hashes[name] = digest.hexdigest()
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        if isinstance(exc, AstraCADRuntimeBlocked):
            raise
        raise AstraCADRuntimeBlocked("cad_source_archive_invalid") from exc
    if not hashes or any(parent.as_posix() in hashes for name in names for parent in PurePosixPath(name).parents):
        raise AstraCADRuntimeBlocked("cad_source_archive_file_directory_collision")
    return hashes


def _verify_packaged_source(root: Path, expected: str, supplied: dict[str, Any]) -> dict[str, Any]:
    """The original sealed component receipt is authoritative, not caller inventory."""
    if not isinstance(supplied, dict):
        raise AstraCADRuntimeBlocked("cad_packaged_source_receipt_invalid")
    supplied_root = Path(str(supplied.get("root") or ""))
    if (root.is_symlink() or not root.is_dir() or not supplied_root.is_absolute()
            or supplied_root.is_symlink() or supplied_root.resolve() != root.resolve()
            or supplied.get("commit") != expected or supplied.get("source_diff") != ""):
        raise AstraCADRuntimeBlocked("cad_packaged_source_root_or_pin_invalid")
    receipt_path = Path(str(supplied.get("source_receipt_path") or ""))
    archive_path = Path(str(supplied.get("archive_path") or ""))
    if (not receipt_path.is_absolute() or receipt_path.is_symlink() or not receipt_path.is_file()
            or not archive_path.is_absolute() or archive_path.is_symlink() or not archive_path.is_file()):
        raise AstraCADRuntimeBlocked("cad_packaged_source_paths_invalid")
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, ValueError) as exc:
        raise AstraCADRuntimeBlocked("cad_packaged_source_receipt_invalid") from exc
    if (not isinstance(receipt, dict)
            or receipt.get("schema_version") != "task_evaluation_cad_skill_component_source.v1"
            or receipt.get("status") != "pinned_sources_packaged"
            or receipt.get("scene_specific_source") is not False or receipt.get("skill_count") != 10
            or not isinstance(receipt.get("sources"), list) or len(receipt["sources"]) != 2
            or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
            or receipt["receipt_digest"] != supplied.get("source_receipt_digest")):
        raise AstraCADRuntimeBlocked("cad_packaged_source_receipt_invalid")
    if any(not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in receipt["sources"]):
        raise AstraCADRuntimeBlocked("cad_packaged_source_receipt_pins_invalid")
    rows = {row["id"]: row for row in receipt["sources"]}
    if set(rows) != {spec["id"] for spec in SOURCE_SPECS}:
        raise AstraCADRuntimeBlocked("cad_packaged_source_receipt_pins_invalid")
    selected = None
    for spec in SOURCE_SPECS:
        row = rows[spec["id"]]
        if (any(row.get(key) != spec[key] for key in ("repository", "commit", "tree", "license", "license_sha256"))
                or row.get("skills") != list(spec["skills"])):
            raise AstraCADRuntimeBlocked("cad_packaged_source_receipt_pins_invalid")
        if spec["commit"] == expected:
            selected = spec
    if selected is None:
        raise AstraCADRuntimeBlocked("cad_packaged_source_pin_unknown")
    archive_hash = "sha256:" + _digest(archive_path)
    if archive_hash != supplied.get("archive_sha256") or archive_hash != rows[selected["id"]].get("archive_sha256"):
        raise AstraCADRuntimeBlocked("cad_packaged_source_archive_digest_mismatch")
    archive_inventory = _archive_file_inventory(archive_path)
    if "sha256:" + archive_inventory.get("LICENSE", "") != selected["license_sha256"]:
        raise AstraCADRuntimeBlocked("cad_packaged_source_license_bytes_mismatch")
    inventory = {}
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()) or not path.resolve().is_relative_to(root.resolve()):
            raise AstraCADRuntimeBlocked("cad_packaged_source_extracted_path_invalid")
        if path.is_file():
            inventory[path.relative_to(root).as_posix()] = _digest(path)
    if inventory != archive_inventory or inventory != supplied.get("tracked_file_sha256"):
        raise AstraCADRuntimeBlocked("cad_packaged_source_file_inventory_mismatch")
    return {"root": str(root.resolve()), "commit": expected, "tree": selected["tree"],
            "tracked_file_sha256": inventory, "source_diff": "", "admission": "sealed_component_archive",
            "source_receipt_digest": receipt["receipt_digest"], "source_receipt_path": str(receipt_path),
            "archive_path": str(archive_path), "archive_sha256": archive_hash}


def verify_cad_sources(mac_source_root: Path, cad_source_root: Path,
                       verified_sources: dict[str, Any] | None = None) -> dict[str, Any]:
    """Pre-paid and post-run exact source verification for clones or sealed archives."""
    if verified_sources is not None and (not isinstance(verified_sources, dict) or set(verified_sources) != {"mac", "cad"}):
        raise AstraCADRuntimeBlocked("cad_verified_sources_invalid")
    result = {}
    for name, root, commit in (("mac", Path(mac_source_root), MAC_COMMIT), ("cad", Path(cad_source_root), CAD_COMMIT)):
        result[name] = (_verify_source(root, commit) if verified_sources is None
                        else _verify_packaged_source(root, commit, verified_sources[name]))
    if verified_sources is not None and result["mac"]["source_receipt_digest"] != result["cad"]["source_receipt_digest"]:
        raise AstraCADRuntimeBlocked("cad_source_receipts_differ")
    return result


def _deny(*_args: Any, **_kwargs: Any) -> Any:
    raise AstraCADRuntimeBlocked("direct_model_or_execution_bypass_denied")


class _SDKChatBridge:
    """The upstream response shape is adapted; upstream provider options confer no authority."""

    def __init__(self, invoker: AgentsSDKInvoker, root: Path, brief: str, run_id: str,
                 max_input_tokens: int, max_output_tokens: int, max_calls: int,
                 object_label: str = "candidate", stable_prefix: str = ""):
        self.invoker, self.root, self.brief, self.run_id = invoker, root, brief, run_id
        self.max_input_tokens, self.max_output_tokens, self.max_calls = max_input_tokens, max_output_tokens, max_calls
        self.object_label = object_label
        self.stable_prefix = stable_prefix
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
        instructions = (
            "Use the supplied immutable brief as hard constraints. Preserve every exact "
            "dimension in millimeters without silent rounding. Return the requested JSON "
            "or complete Python code in content. Generated CAD is development_only. "
            "Never repeat the raw prompt/evidence packet. For user_request_raw return only "
            "a concise one-line object name; the harness restores the canonical request. "
            "For digital candidates manufacturing_method must be the schema enum 'unspecified'. "
            "Use only requested schema fields and keep narrative concise.")
        stable_prefix = instructions + '\n' + self.stable_prefix if self.stable_prefix else None
        if stable_prefix:
            from .asset_authoring_prompt_cache import asset_cache_policy
            policy = asset_cache_policy(family='cad', output_type=_TextOutput, stable_prefix=stable_prefix)
        else:
            policy = None
        spec = AgentsSDKAgentSpec(
            run_id=self.run_id, capability=f"astra_cad_candidate:{self.object_label}", name="Astra CAD candidate",
            instructions=instructions,
            model="gpt-6-astra", reasoning_effort="high", max_turns=1,
            max_input_tokens=self.max_input_tokens, max_output_tokens=self.max_output_tokens,
            output_type=_TextOutput, tool_bindings=(),
            stable_developer_prefix=stable_prefix, cache_policy=policy,
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
              "solid_count": len(shape.solids()),
              "build123d": importlib.metadata.version("build123d")}
    result["kernel_versions"] = {package: importlib.metadata.version(package) for package in
                                 importlib.metadata.packages_distributions().get("OCP", [])}
    result["passed"] = bool(valid and shape.volume > 0 and result["solid_count"] == 1 and all(
        math.isfinite(a) and abs(a - b) <= tolerance for a, b in zip(measured, expected)
    ))
    return result


def _adopt_completed_phases(source: Path, budget_root: Path, parameters: dict[str, Any], nodes: Any):
    """Authenticate retained typed outputs against completed SDK receipts."""
    source = source.resolve(strict=True)
    previous = json.loads((source / "parameters.json").read_text())
    for key in ("brief", "expected_dimensions_mm", "run_id", "object_label"):
        if previous.get(key) != parameters[key]:
            raise AstraCADRuntimeBlocked("cad_adoption_parameters_mismatch:" + key)
    completions = []
    for path in (budget_root / "inference_reservations/completed").glob("*.json"):
        row = json.loads(path.read_text())
        if (row.get("run_id") == parameters["run_id"]
                and row.get("capability") == "astra_cad_candidate:" + parameters["object_label"]
                and row.get("model") == "gpt-6-astra" and row.get("provider") == "openai"
                and row.get("inference_completion_digest") == canonical_digest(row, digest_field="inference_completion_digest")):
            completions.append((path, row))
    adopted, evidence = {}, []
    for index, (name, field, model) in enumerate((
        ("node_spec_planner", "cad_brief", nodes.CADBrief),
        ("node_geometric_architect", "architect_plan", nodes.ArchitectPlan),
    )):
        raw_path = source / f"invocation-{index:02d}-output.txt"
        raw = raw_path.read_text()
        digest = canonical_digest({"content": raw})
        matching = [(path, row) for path, row in completions if row.get("structured_output_digest") == digest]
        if len(matching) != 1:
            raise AstraCADRuntimeBlocked("cad_adoption_missing_sdk_completion:" + name)
        data = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
        typed = model.model_validate(data)
        if field == "cad_brief":
            typed.user_request_raw = parameters["brief"]
        retained_path = source / f"node-{name}.json"
        retained = json.loads(retained_path.read_text())
        if model.model_validate(retained[field]).model_dump(mode="json") != typed.model_dump(mode="json"):
            raise AstraCADRuntimeBlocked("cad_adoption_node_output_mismatch:" + name)
        adopted[name] = {field: typed, "node_history": ["planner"] if index == 0 else ["planner", "architect"],
                         "execution_log": [name + ": adopted verified completed Astra output"]}
        evidence.append({"phase": name, "source_output": str(raw_path), "output_sha256": _digest(raw_path),
                         "node_path": str(retained_path), "node_sha256": _digest(retained_path),
                         "completion_path": str(matching[0][0]), "completion_sha256": _digest(matching[0][0]),
                         "structured_output_digest": digest})
    return adopted, {"schema_version": "astra_cad_phase_adoption.v1", "source": str(source),
                     "source_parameters_sha256": _digest(source / "parameters.json"), "phases": evidence}


def _compact_coder_prompt(**kwargs: Any) -> str:
    def compact(value):
        if isinstance(value, dict):
            return {key: compact(item) for key, item in value.items() if item is not None and item != [] and item != {}}
        if isinstance(value, list):
            return [compact(item) for item in value]
        return value
    return json.dumps({"architect_plan": compact(json.loads(kwargs["plan_json"])),
                       "previous_feedback": kwargs.get("previous_feedback"),
                       "step_path": kwargs["step_path"], "stl_path": kwargs["stl_path"],
                       "task": "Implement the complete exact plan, including custom curved profiles in notes. "
                       "Return self-contained build123d Python defining gen_step(), and a main block that calls it "
                       "and exports STEP/STL to the supplied paths. Use from build123d import *. "
                       "Do not copy the brief or source evidence into code. Preserve every exact dimension; no sizing objects. "
                       "No network, external files, subprocesses, or viewers. Geometry remains development_only."},
                      separators=(",", ":"))


def execute_mac_candidate(
    brief: str, output_root: Path, mac_source_root: Path, cad_source_root: Path,
    invoker: AgentsSDKInvoker, *, expected_dimensions_mm: tuple[float, float, float],
    subprocess_runner: Callable[..., Any] | None = None, run_id: str = "astra-cad-candidate",
    object_label: str = "candidate",
    max_input_tokens: int = 80_000, max_output_tokens: int = 12_000,
    max_calls: int = 7, repair_budget: int = 2, dimension_tolerance_mm: float = 0.01,
    verified_sources: dict[str, Any] | None = None,
    adopt_state_from: Path | None = None, adoption_budget_root: Path | None = None,
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
    runtime_loader = [str(Path(p).resolve()) for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if root.exists() and any(root.iterdir()):
        raise AstraCADRuntimeBlocked("cad_output_root_not_empty")
    if root.is_relative_to(mac) or root.is_relative_to(cad):
        raise AstraCADRuntimeBlocked("cad_output_inside_immutable_source")
    subprocess_runner.preflight()
    sources = verify_cad_sources(mac, cad, verified_sources)
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
    bridge = _SDKChatBridge(invoker, root, brief, run_id, max_input_tokens, max_output_tokens, max_calls, object_label,
                           stable_prefix=(cad / "skills/cad/SKILL.md").read_text())
    receipt: dict[str, Any] = {"claim": "development_only_candidate", "passed": False,
                               "source_receipt": str(root / "source-receipt.json")}
    with _LOCK, _source_modules(mac, cad) as (graph, nodes):
        os.chdir(root)
        nodes._CACHE_DIR = root / "pipeline_cache"
        nodes._llm_client = lambda: bridge
        nodes.OpenAI = _deny
        def deterministic_child(state, architect_plan, iteration):
            request_path = root / f"deterministic-{iteration}-request.json"
            result_path = root / f"deterministic-{iteration}-result.json"
            _save(request_path, {"state": dict(state), "architect_plan": architect_plan, "iteration": iteration})
            # Preserve every upstream loader root, while adding exact pinned source roots.
            loader = [str(mac), str(cad / "packages/cadpy/src"), *runtime_loader]
            result = subprocess_runner([sys.executable, str(Path(__file__).with_name("astra_cad_deterministic_child.py")),
                                        str(request_path), str(result_path)], cwd=str(root), timeout=120,
                                       env={"PYTHONPATH": os.pathsep.join(dict.fromkeys(loader))},
                                       capture_output=True, text=True, check=False)
            (root / f"deterministic-{iteration}-stdout.txt").write_text(result.stdout or "")
            (root / f"deterministic-{iteration}-stderr.txt").write_text(result.stderr or "")
            if result.returncode or not result_path.is_file():
                raise AstraCADRuntimeBlocked("cad_deterministic_child_failed")
            envelope = json.loads(result_path.read_text())
            if envelope.get("status") == "unsupported":
                return None
            if envelope.get("status") != "completed" or not isinstance(envelope.get("result"), dict):
                raise AstraCADRuntimeBlocked("cad_deterministic_child_failed")
            update = envelope["result"]
            if update.get("qa_report"):
                update["qa_report"] = nodes.QAReport.model_validate(update["qa_report"])
            return update

        nodes._node_python_coder_deterministic = deterministic_child
        nodes._build_coder_user_prompt = _compact_coder_prompt
        nodes.SYSTEM_PROMPT_PYTHON_CODER = "Implement the supplied ArchitectPlan as complete executable build123d Python."
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
        adopted = {}
        if adopt_state_from is not None:
            if adoption_budget_root is None:
                raise AstraCADRuntimeBlocked("cad_adoption_budget_root_required")
            adopted, adoption_receipt = _adopt_completed_phases(Path(adopt_state_from), Path(adoption_budget_root), parameters, nodes)
            _save(root / "adoption-receipt.json", adoption_receipt)
        for name in ("node_spec_planner", "node_geometric_architect", "node_python_coder", "node_autonomous_skill_loop"):
            original = getattr(nodes, name)

            def guarded(state, fn=original, node_name=name):
                result = dict(adopted[node_name]) if node_name in adopted else fn(state)
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
        except Exception as exc:
            receipt["failure"] = str(exc)
            raise
        finally:
            try:
                receipt["source_unchanged_after"] = verify_cad_sources(mac, cad, verified_sources) == sources
                if not receipt["source_unchanged_after"]:
                    raise AstraCADRuntimeBlocked("cad_source_changed_during_execution")
            except Exception as exc:
                receipt.update(passed=False, source_verification_failure=str(exc))
                raise
            finally:
                receipt.update(invocation_count=len(bridge.calls), repair_calls=repair_calls)
                receipt["artifacts"] = {str(p.relative_to(root)): _digest(p) for p in root.rglob("*")
                                        if p.is_file() and p.suffix in (".py", ".step", ".stl", ".json", ".txt")}
                _save(root / "candidate-receipt.json", receipt)
    return receipt
