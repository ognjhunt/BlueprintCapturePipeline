"""A sandbox model load under upstream's accepted inspection changes no dependency file.

Website job, 2026-11-12 (17:58Z and 23:53Z): stage 3 failed in ``P4_build`` with
"accepted inspection dependencies do not match candidate". Upstream ARTISAN
holds the accepted candidate in a process-local ``ContextVar`` and, while it is
set, reuses the accepted mesh bytes. Our sandbox loaded the model in a new
process without it, recompiled the CadQuery meshes (whose bytes differ from
process to process) and overwrote ``assets/meshes/*.obj``.

The upstream modules below are the smallest faithful stand-ins for the pinned
ARTISAN functions on this path (``sdk/runtime/accepted_inspection.py``,
``sdk/runtime/release_candidate.py::_dependency_manifest``,
``sdk/_core/v0/assets.py`` preserved-inventory rules and ``pipeline/compiler.py``);
the test environment has no ARTISAN checkout. Our hook and worker run unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from blueprint_pipeline import task_object_artisan_hooks as hooks

SRC = Path(__file__).resolve().parents[1] / "src"

UPSTREAM = {
    "sdk/__init__.py": "",
    "sdk/runtime/__init__.py": "",
    "sdk/_core/__init__.py": "",
    "sdk/_core/v0/__init__.py": "",
    "sdk/_core/v0/motion_engine/__init__.py": "",
    "pipeline/__init__.py": "",
    "sdk/_core/v0/_testing/__init__.py": "",
    "sdk/_core/v0/_testing/common.py": """
        class TestReport: pass
        class TestFailure: pass
        class AllowedOverlap: pass
    """,
    "sdk/_core/v0/motion_engine/frozen_build.py": """
        def _model_types():
            return {}
        def _encode(value):
            return value
        def _decode(value, root):
            return value
    """,
    "sdk/runtime/release_candidate.py": """
        import hashlib, json
        from pathlib import Path
        class CandidateError(ValueError): pass
        def _dependency_manifest(model_path):
            root = Path(model_path).resolve().parent
            files = [p for r in ("inputs", "references/inputs", "assets/meshes") if (root / r).is_dir()
                     for p in (root / r).rglob("*") if p.is_file()]
            if (root / "tasks.json").is_file():
                files.append(root / "tasks.json")
            return [{"path": p.relative_to(root).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                     "size_bytes": p.stat().st_size} for p in sorted(set(files))]
        def dependency_sha256_for_model(model_path):
            return hashlib.sha256(json.dumps(_dependency_manifest(model_path), sort_keys=True).encode()).hexdigest()
    """,
    # Upstream sdk/runtime/accepted_inspection.py, with the manifest read reduced to its digest.
    "sdk/runtime/accepted_inspection.py": """
        import json
        from contextlib import contextmanager
        from contextvars import ContextVar
        from pathlib import Path
        _REFERENCE = ContextVar("accepted_inspection", default=None)
        def current_reference():
            reference = _REFERENCE.get()
            return dict(reference) if reference is not None else None
        def candidate_reference(manifest_path, candidate_id):
            return {"manifest_path": str(manifest_path), "candidate_id": candidate_id}
        def _verify(model_path, reference):
            from sdk.runtime.release_candidate import CandidateError, dependency_sha256_for_model
            if not isinstance(reference, dict) or set(reference) != {"manifest_path", "candidate_id"}:
                raise CandidateError("invalid accepted inspection reference")
            manifest = json.loads(Path(reference["manifest_path"]).read_text())
            if dependency_sha256_for_model(model_path) != manifest["dependency_sha256"]:
                raise CandidateError("accepted inspection dependencies do not match candidate")
        @contextmanager
        def accepted_inspection(model_path, reference):
            if reference is None:
                yield
                return
            from sdk._core.v0.assets import preserve_mesh_inventory
            from sdk.runtime.release_candidate import _dependency_manifest
            _verify(model_path, reference)
            inventory = {(model_path.parent / e["path"]).resolve(): (model_path.parent / e["path"]).read_bytes()
                         for e in _dependency_manifest(model_path) if e["path"].endswith(".obj")}
            token = _REFERENCE.set(dict(reference))
            try:
                with preserve_mesh_inventory(inventory):
                    try:
                        yield
                    finally:
                        _verify(model_path, reference)
            finally:
                _REFERENCE.reset(token)
    """,
    # Upstream sdk/_core/v0/assets.py: managed meshes overwrite the base file unless an
    # accepted inventory is preserved; then they reuse it and never create a mesh or sidecar.
    "sdk/_core/v0/assets.py": """
        import contextlib, contextvars
        from pathlib import Path
        _SESSION = contextvars.ContextVar("artisan_asset_session", default=None)
        _PRESERVED_MESHES = contextvars.ContextVar("artisan_preserved_mesh_inventory", default=None)
        @contextlib.contextmanager
        def preserve_mesh_inventory(files):
            token = _PRESERVED_MESHES.set(dict(files))
            try:
                yield
            finally:
                _PRESERVED_MESHES.reset(token)
        class AssetSession:
            def __init__(self, root):
                self.root = Path(root).resolve()
            def register_mesh_text(self, name, payload):
                path = (self.root / "assets/meshes" / f"{name}.obj").resolve()
                inventory = _PRESERVED_MESHES.get()
                if inventory is not None:
                    if path not in inventory:
                        raise ValueError(f"Accepted inspection cannot create mesh: {path.name}")
                    return path
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists() or path.read_bytes() != payload.encode():
                    path.write_bytes(payload.encode())
                return path
            def register_input_mesh(self, name):
                source = self.root / "inputs" / f"{name}.obj"
                path = self.register_mesh_text(name, source.read_text())
                sidecar = source.with_suffix(".obj.anchors.json")
                if sidecar.is_file() and _PRESERVED_MESHES.get() is None:
                    path.with_suffix(".obj.anchors.json").write_bytes(sidecar.read_bytes())
                return path
        def current_asset_session():
            return _SESSION.get()
        def asset_session_for_script(script_path, *, preserve_existing_base=False):
            return AssetSession(Path(script_path).resolve().parent)
        @contextlib.contextmanager
        def activate_asset_session(session):
            token = _SESSION.set(session)
            try:
                yield session
            finally:
                _SESSION.reset(token)
    """,
    "pipeline/compiler.py": """
        import runpy
        def load_model_globals(script_path, *, sdk_package="sdk"):
            return runpy.run_path(str(script_path))
    """,
}

# A base part and a handle attached to it; CadQuery's recompiled bytes differ per process.
MODEL = """
import os, uuid
from sdk._core.v0.assets import current_asset_session
session = current_asset_session()
nonce = f"# compile {os.getpid()} {uuid.uuid4().hex}"
shell = session.register_mesh_text("tray_shell", f"{nonce}\\nv 0 0 0\\nv 1 0 0\\nv 0 1 0\\nf 1 2 3\\n")
handle = session.register_mesh_text("tray_handle", f"{nonce}\\nv 0 0 1\\nv 1 0 1\\nv 0 1 1\\nf 1 2 3\\n")
EXTRA
object_model = {"parts": ["base", "handle"], "attach": {"subject": "handle", "target": "base"},
                "meshes": [shell.name, handle.name]}
"""


def _write(root: Path, files: dict[str, str]) -> None:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text))


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    root = tmp_path / "upstream"
    _write(root, UPSTREAM)
    monkeypatch.syspath_prepend(str(root))
    for name in [m for m in sys.modules if m.split(".")[0] in {"sdk", "pipeline"}]:
        monkeypatch.delitem(sys.modules, name)
    yield root
    for name in [m for m in sys.modules if m.split(".")[0] in {"sdk", "pipeline"}]:
        sys.modules.pop(name, None)


class _Plain:
    """The kernel sandbox, minus the kernel: argv, cwd and env exactly as the hook passes them."""

    def preflight(self):
        return None

    def __call__(self, argv, *, cwd, env, timeout, check):
        return subprocess.run(argv, cwd=cwd, env=env, timeout=timeout, check=check)


def _workspace(tmp_path: Path, upstream: Path, monkeypatch, extra: str = "") -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "model.py").write_text(MODEL.replace("EXTRA", extra))
    (workspace / "tasks.json").write_text('{"schema_version": "2.0", "tasks": []}')
    monkeypatch.setattr(hooks, "sandbox", lambda *_: _Plain())
    monkeypatch.setenv(hooks.CONTEXT_ENV, json.dumps({
        "write_root": str(workspace), "read_roots": [], "write_paths": [],
        "import_roots": [str(SRC), str(upstream)], "blender": "/nonexistent"}))
    return workspace


def _stage(workspace: Path) -> dict:
    """Compile once in the calling process and record the candidate's dependency digest."""
    from sdk._core.v0.assets import activate_asset_session, asset_session_for_script
    from sdk.runtime.accepted_inspection import candidate_reference
    from sdk.runtime.release_candidate import dependency_sha256_for_model
    from pipeline.compiler import load_model_globals
    model = workspace / "model.py"
    with activate_asset_session(asset_session_for_script(model)):
        load_model_globals(model)
    manifest = workspace / "candidate/manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"dependency_sha256": dependency_sha256_for_model(model)}))
    return candidate_reference(manifest, "candidate-1")


def _mesh_state(workspace: Path) -> dict:
    meshes = workspace / "assets/meshes"
    return {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(meshes.iterdir())}


def test_sandbox_reload_under_an_accepted_inspection_changes_no_dependency(tmp_path, upstream, monkeypatch):
    workspace = _workspace(tmp_path, upstream, monkeypatch)
    model = workspace / "model.py"
    reference = _stage(workspace)
    from sdk.runtime.accepted_inspection import accepted_inspection
    from sdk.runtime.release_candidate import dependency_sha256_for_model
    staged, meshes = dependency_sha256_for_model(model), _mesh_state(workspace)
    assert sorted(meshes) == ["tray_handle.obj", "tray_shell.obj"]
    # Upstream's P4 acceptance: inspect the staged candidate; the model load goes to the sandbox.
    with accepted_inspection(model, reference):
        loaded = hooks.load_model_globals(model)
    assert loaded["object_model"]["attach"] == {"subject": "handle", "target": "base"}
    assert dependency_sha256_for_model(model) == staged
    assert _mesh_state(workspace) == meshes


def test_without_protection_the_sandbox_recompile_rewrites_the_meshes(tmp_path, upstream, monkeypatch):
    """Why the protection is needed: the same reload outside an inspection changes the digest."""
    workspace = _workspace(tmp_path, upstream, monkeypatch)
    model = workspace / "model.py"
    _stage(workspace)
    from sdk.runtime.release_candidate import dependency_sha256_for_model
    staged = dependency_sha256_for_model(model)
    hooks.load_model_globals(model)
    assert dependency_sha256_for_model(model) != staged


def test_the_sandbox_request_carries_the_accepted_reference_only_while_one_is_active(tmp_path, upstream, monkeypatch):
    workspace = _workspace(tmp_path, upstream, monkeypatch)
    model = workspace / "model.py"
    reference = _stage(workspace)
    calls = []

    def record(operation, **payload):
        calls.append((operation, payload))
        return {"object_model": {}, "has_tests": True} if operation == "load_model" else {}

    monkeypatch.setattr(hooks, "_rpc", record)
    monkeypatch.setattr(hooks, "_decode", lambda value, _root: value)
    from sdk.runtime.accepted_inspection import accepted_inspection
    outside = hooks.load_model_globals(model)
    outside["run_tests"]()
    hooks.run_qa_tests(model, "def qa_tests(ctx): pass")
    with accepted_inspection(model, reference):
        inside = hooks.load_model_globals(model)
        inside["run_tests"]()
        hooks.run_qa_tests(model, "def qa_tests(ctx): pass")
    # A run_tests made inside but called after the inspection ended carries nothing.
    inside["run_tests"]()
    carried = [(operation, payload.get("accepted_candidate")) for operation, payload in calls]
    assert carried == [("load_model", None), ("authored_tests", None), ("qa", None),
                       ("load_model", reference), ("authored_tests", reference), ("qa", reference),
                       ("authored_tests", None)]
    assert all("accepted_candidate" not in payload for _operation, payload in calls[:3] + calls[6:])


def _run_worker(workspace: Path, upstream: Path, operation: str, payload: dict) -> dict:
    ipc = workspace / "blueprint_sandbox_ipc"
    ipc.mkdir(exist_ok=True)
    request, result = ipc / "case.request.json", ipc / "case.result.json"
    request.write_text(json.dumps({"operation": operation, "payload": payload, "result": str(result),
                                   "context": {"write_root": str(workspace), "blender": "/nonexistent"}}))
    env = {"PYTHONPATH": os.pathsep.join([str(SRC), str(upstream)])}
    subprocess.run([sys.executable, "-S", "-m", "blueprint_pipeline.task_object_artisan_sandbox", str(request)],
                   cwd=workspace, env=env, check=True, timeout=120)
    return json.loads(result.read_text())


def test_the_worker_refuses_a_new_mesh_or_sidecar_while_the_protection_is_on(tmp_path, upstream, monkeypatch):
    workspace = _workspace(tmp_path, upstream, monkeypatch)
    model = workspace / "model.py"
    reference = _stage(workspace)
    meshes = _mesh_state(workspace)
    # The accepted model gains a third mesh after staging: under protection it is refused.
    model.write_text(MODEL.replace("EXTRA", 'session.register_mesh_text("tray_lid", "v 0 0 0\\n")'))
    refused = _run_worker(workspace, upstream, "load_model",
                          {"model_path": str(model), "sdk_package": "sdk", "accepted_candidate": reference})
    assert refused["ok"] is False and "cannot create mesh: tray_lid.obj" in refused["error_message"]
    assert _mesh_state(workspace) == meshes

    # An input mesh with an anchor sidecar: the accepted bytes are reused and no sidecar is written.
    (workspace / "inputs").mkdir()
    (workspace / "inputs/tray_shell.obj").write_text("v 9 9 9\n")
    (workspace / "inputs/tray_shell.obj.anchors.json").write_text("{}")
    model.write_text(MODEL.replace("EXTRA", 'session.register_input_mesh("tray_shell")'))
    reference = _stage_again(workspace, reference)
    meshes = _mesh_state(workspace)
    reused = _run_worker(workspace, upstream, "load_model",
                         {"model_path": str(model), "sdk_package": "sdk", "accepted_candidate": reference})
    assert reused["ok"] is True, reused
    assert _mesh_state(workspace) == meshes
    assert not (workspace / "assets/meshes/tray_shell.obj.anchors.json").exists()


def _stage_again(workspace: Path, reference: dict) -> dict:
    from sdk.runtime.release_candidate import dependency_sha256_for_model
    Path(reference["manifest_path"]).write_text(json.dumps(
        {"dependency_sha256": dependency_sha256_for_model(workspace / "model.py")}))
    return reference


def test_the_worker_refuses_an_accepted_reference_outside_its_workspace(tmp_path, upstream, monkeypatch):
    workspace = _workspace(tmp_path, upstream, monkeypatch)
    model = workspace / "model.py"
    reference = _stage(workspace)
    outside = tmp_path / "manifest.json"
    outside.write_text(Path(reference["manifest_path"]).read_text())
    meshes = _mesh_state(workspace)
    response = _run_worker(workspace, upstream, "load_model", {
        "model_path": str(model), "sdk_package": "sdk",
        "accepted_candidate": {"manifest_path": str(outside), "candidate_id": "candidate-1"}})
    assert response == {"ok": False, "error_type": "ValueError",
                        "error_message": "artisan_accepted_candidate_outside_workspace"}
    assert _mesh_state(workspace) == meshes
