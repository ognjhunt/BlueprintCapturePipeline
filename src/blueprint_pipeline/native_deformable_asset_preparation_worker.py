"""Pinned native worker for one deformable-asset preparation execution.

The module intentionally stays thin.  It verifies the exact Isaac Lab source
files that own the frozen APIs, resolves only those APIs, starts one caller-
owned Isaac application, and invokes the task-neutral preparation seam once.
Its output is worker-authored payload; only the enclosing trusted execution
envelope may establish execution authority or native qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import stat
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

from .native_deformable_asset_preparation import (
    DEFORMABLE_AUTHORING_API,
    DEFORMABLE_BODY_CFG,
    DEFORMABLE_MATERIAL_API,
    DEFORMABLE_MATERIAL_CFG,
    DEFORMABLE_PHYSICS_BINDING_API,
    NATIVE_REQUIRED_API_SYMBOLS,
    PINNED_NATIVE_CALL_CONTRACT,
    execute_native_deformable_asset_preparation,
)
from .native_deformable_asset_stage_adapter import OpenUsdNativeDeformableStageAdapter


MAX_PLAN_BYTES = 16 * 1024 * 1024
WORKER_SCHEMA_VERSION = "native_deformable_asset_preparation_worker_terminal.v1"
_DIGEST_PREFIX = "sha256:"
_ISAAC_SYMBOLS = (
    DEFORMABLE_MATERIAL_CFG,
    DEFORMABLE_MATERIAL_API,
    DEFORMABLE_BODY_CFG,
    DEFORMABLE_AUTHORING_API,
    DEFORMABLE_PHYSICS_BINDING_API,
)
_ALL_SYMBOLS = NATIVE_REQUIRED_API_SYMBOLS
REQUIRED_RUNTIME_IMPORTS = ("pytetwild",)
_MAX_EXCEPTION_MESSAGE_CHARS = 2048
_MAX_EXCEPTION_TRACEBACK_CHARS = 8192


class NativeDeformableAssetPreparationWorkerError(ValueError):
    """Stable worker-boundary failure."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _valid_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith(_DIGEST_PREFIX)
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _git_blob_sha1(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content, usedforsecurity=False).hexdigest()


def _snapshot_regular_file(path: Path, *, maximum_size: int, error: str) -> bytes:
    """Read one absolute regular file through descriptor-relative no-follow opens."""

    absolute = path.expanduser().absolute()
    parts = PurePosixPath(absolute.as_posix()).parts
    if not parts or parts[0] != "/":
        raise NativeDeformableAssetPreparationWorkerError([error])
    directory_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in parts[1:-1]:
            if component in {"", ".", ".."}:
                raise NativeDeformableAssetPreparationWorkerError([error])
            next_fd = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        leaf = parts[-1]
        if leaf in {"", ".", ".."}:
            raise NativeDeformableAssetPreparationWorkerError([error])
        file_fd = os.open(
            leaf,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
        try:
            before = os.fstat(file_fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_size:
                raise NativeDeformableAssetPreparationWorkerError([error])
            chunks: list[bytes] = []
            remaining = maximum_size + 1
            while remaining:
                chunk = os.read(file_fd, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            content = b"".join(chunks)
            after = os.fstat(file_fd)

            def identity(row: os.stat_result) -> tuple[int, int, int, int]:
                return row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns

            if (
                len(content) != before.st_size
                or len(content) > maximum_size
                or identity(before) != identity(after)
            ):
                raise NativeDeformableAssetPreparationWorkerError([error])
            return content
        finally:
            os.close(file_fd)
    except (OSError, ValueError) as exc:
        if isinstance(exc, NativeDeformableAssetPreparationWorkerError):
            raise
        raise NativeDeformableAssetPreparationWorkerError([error]) from exc
    finally:
        os.close(directory_fd)


def _source_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for key in ("material_spawn", "deformable_authoring", "physics_material_binding"):
        row = PINNED_NATIVE_CALL_CONTRACT[key]
        rows[str(row["symbol"])] = dict(row)
    for symbol, row in PINNED_NATIVE_CALL_CONTRACT["configuration_sources"].items():
        rows[str(symbol)] = dict(row)
    return rows


def _resolve_symbol(symbol: str, importer: Callable[[str], ModuleType]) -> tuple[ModuleType, Any]:
    module_name, separator, attribute_name = symbol.partition(":")
    if not separator or not module_name or not attribute_name:
        raise NativeDeformableAssetPreparationWorkerError(
            ["native_deformable_worker_symbol_invalid"]
        )
    try:
        module = importer(module_name)
        value = getattr(module, attribute_name)
    except (ImportError, AttributeError) as exc:
        raise NativeDeformableAssetPreparationWorkerError(
            [f"native_deformable_worker_symbol_unavailable:{symbol}"]
        ) from exc
    if not callable(value):
        raise NativeDeformableAssetPreparationWorkerError(
            [f"native_deformable_worker_symbol_not_callable:{symbol}"]
        )
    return module, value


def _build_registry(
    *,
    isaaclab_source_root: Path,
    importer: Callable[[str], ModuleType],
) -> dict[str, Callable[..., Any]]:
    source_root = isaaclab_source_root.expanduser().absolute()
    if not source_root.is_dir() or source_root.is_symlink():
        raise NativeDeformableAssetPreparationWorkerError(
            ["native_deformable_worker_isaaclab_source_root_invalid"]
        )
    source_rows = _source_rows()
    registry: dict[str, Callable[..., Any]] = {}
    for symbol in _ALL_SYMBOLS:
        module, value = _resolve_symbol(symbol, importer)
        registry[symbol] = value
        row = source_rows[symbol]
        relative = PurePosixPath(str(row["source_relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise NativeDeformableAssetPreparationWorkerError(
                ["native_deformable_worker_source_contract_invalid"]
            )
        expected_path = source_root.joinpath(*relative.parts)
        content = _snapshot_regular_file(
            expected_path,
            maximum_size=8 * 1024 * 1024,
            error=f"native_deformable_worker_source_file_invalid:{symbol}",
        )
        if _git_blob_sha1(content) != row["source_git_blob_sha1"]:
            raise NativeDeformableAssetPreparationWorkerError(
                [f"native_deformable_worker_source_blob_mismatch:{symbol}"]
            )
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str) or Path(module_file).absolute() != expected_path:
            raise NativeDeformableAssetPreparationWorkerError(
                [f"native_deformable_worker_symbol_origin_invalid:{symbol}"]
            )
        parameters = row.get("parameters")
        if parameters is not None:
            try:
                observed = list(inspect.signature(value).parameters)
            except (TypeError, ValueError) as exc:
                raise NativeDeformableAssetPreparationWorkerError(
                    [f"native_deformable_worker_symbol_signature_invalid:{symbol}"]
                ) from exc
            if observed != list(parameters):
                raise NativeDeformableAssetPreparationWorkerError(
                    [f"native_deformable_worker_symbol_signature_invalid:{symbol}"]
                )
    return registry


def build_pinned_native_api_registry(
    *, isaaclab_source_root: str | Path
) -> dict[str, Callable[..., Any]]:
    """Resolve the exact frozen native APIs from the pinned Isaac Lab source root."""

    return _build_registry(
        isaaclab_source_root=Path(isaaclab_source_root), importer=importlib.import_module
    )


def _verify_required_runtime_imports(
    checker: Callable[[str], object | None] = importlib.util.find_spec,
) -> None:
    """Fail before native cook if pinned optional runtime imports are absent."""

    missing = [name for name in REQUIRED_RUNTIME_IMPORTS if checker(name) is None]
    if missing:
        raise NativeDeformableAssetPreparationWorkerError(
            [
                "native_deformable_worker_runtime_dependency_unavailable:"
                + ",".join(missing)
            ]
        )


def run_native_deformable_asset_preparation_worker(
    *,
    plan_path: str | Path,
    expected_plan_digest: str,
    package_root: str | Path,
    output_root: str | Path,
    isaaclab_source_root: str | Path,
    stage_api: object | None = None,
    registry_builder: Callable[..., Mapping[str, Callable[..., Any]]] = (
        build_pinned_native_api_registry
    ),
    runtime_import_checker: Callable[[str], object | None] = importlib.util.find_spec,
) -> dict[str, Any]:
    """Read one frozen plan and execute it once inside an already started runtime."""

    if not _valid_digest(expected_plan_digest):
        raise NativeDeformableAssetPreparationWorkerError(
            ["native_deformable_worker_expected_plan_digest_invalid"]
        )
    content = _snapshot_regular_file(
        Path(plan_path), maximum_size=MAX_PLAN_BYTES, error="native_deformable_worker_plan_invalid"
    )
    try:
        plan = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeDeformableAssetPreparationWorkerError(
            ["native_deformable_worker_plan_json_invalid"]
        ) from exc
    if not isinstance(plan, dict):
        raise NativeDeformableAssetPreparationWorkerError(
            ["native_deformable_worker_plan_json_invalid"]
        )
    _verify_required_runtime_imports(runtime_import_checker)
    registry = dict(registry_builder(isaaclab_source_root=isaaclab_source_root))
    adapter = stage_api if stage_api is not None else OpenUsdNativeDeformableStageAdapter()
    return execute_native_deformable_asset_preparation(
        plan=plan,
        expected_plan_digest=expected_plan_digest,
        package_root=package_root,
        output_root=output_root,
        stage_api=adapter,
        native_api_registry=registry,
    )


def _write_terminal(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise NativeDeformableAssetPreparationWorkerError(
            ["native_deformable_worker_terminal_output_exists"]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    try:
        file_fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(file_fd, payload[offset:])
            os.fsync(file_fd)
        finally:
            os.close(file_fd)
    except OSError as exc:
        raise NativeDeformableAssetPreparationWorkerError(
            ["native_deformable_worker_terminal_write_failed"]
        ) from exc


def _exception_diagnostic(exc: BaseException) -> dict[str, Any]:
    """Return bounded exception diagnostics for paid native failures."""

    message = str(exc)
    formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return {
        "schema_version": "native_deformable_worker_exception_diagnostic.v1",
        "type": type(exc).__name__,
        "message": message[:_MAX_EXCEPTION_MESSAGE_CHARS],
        "message_truncated": len(message) > _MAX_EXCEPTION_MESSAGE_CHARS,
        "traceback_tail": formatted[-_MAX_EXCEPTION_TRACEBACK_CHARS:],
        "traceback_truncated": len(formatted) > _MAX_EXCEPTION_TRACEBACK_CHARS,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--expected-plan-digest", required=True)
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--isaaclab-source-root", required=True)
    parser.add_argument("--terminal-output", required=True)
    args = parser.parse_args(argv)

    application: object | None = None
    terminal: dict[str, Any] = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "status": "blocked",
        "expected_plan_digest": args.expected_plan_digest,
        "worker_result_digest": None,
        "errors": [],
        "exception_diagnostic": None,
        "claim_boundary": {
            "worker_payload_only": True,
            "trusted_execution_authority": False,
            "native_cook_qualified": False,
            "simulator_qualified": False,
            "physical_material_equivalence": False,
        },
    }
    terminal_path = Path(args.terminal_output)
    terminal_written = False
    try:
        from isaacsim.simulation_app import SimulationApp

        application = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
        result = run_native_deformable_asset_preparation_worker(
            plan_path=args.plan,
            expected_plan_digest=args.expected_plan_digest,
            package_root=args.package_root,
            output_root=args.output_root,
            isaaclab_source_root=args.isaaclab_source_root,
        )
        terminal["status"] = "worker_payload_materialized_pending_trusted_execution_join"
        terminal["worker_result_digest"] = result.get("worker_result_digest")
    except Exception as exc:  # noqa: BLE001 - terminal boundary retains a typed null
        errors = getattr(exc, "errors", None)
        terminal["errors"] = list(errors) if errors else [type(exc).__name__]
        terminal["exception_diagnostic"] = _exception_diagnostic(exc)
    finally:
        if application is not None:
            _write_terminal(terminal_path, terminal)
            terminal_written = True
            close = getattr(application, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # noqa: BLE001 - retain cleanup failure
                    if not terminal_written:
                        terminal["status"] = "blocked"
                        terminal["errors"] = sorted(
                            {
                                *terminal["errors"],
                                f"application_close_failed:{type(exc).__name__}",
                            }
                        )
    if not terminal_written:
        _write_terminal(terminal_path, terminal)
    return 0 if terminal["status"].startswith("worker_payload_materialized") else 1


if __name__ == "__main__":  # pragma: no cover - exercised by native bundle
    raise SystemExit(main())


__all__ = [
    "NativeDeformableAssetPreparationWorkerError",
    "WORKER_SCHEMA_VERSION",
    "build_pinned_native_api_registry",
    "main",
    "run_native_deformable_asset_preparation_worker",
]
