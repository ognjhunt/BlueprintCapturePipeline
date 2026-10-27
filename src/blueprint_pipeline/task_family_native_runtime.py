"""Construct a family runtime from an operator-qualified installed SDK.

Beta: job-to-evaluation. The job selects a runtime id, never executable Python.
The operator's external catalog pins the registered Gymnasium factory, its
installed source bytes and bounded qualification. This adapter applies the
sealed scene and forwards native state; it never invents task measurements.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import stat
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_family_native_contracts import validate_task_family_native_runtime_contract

CATALOG_SCHEMA = "task_family_native_runtime_catalog.v1"
QUALIFICATION_SCHEMA = "task_family_native_runtime_qualification.v1"
GYMNASIUM_FACTORY = "gymnasium_native_task_family.v1"
_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MODULE = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\Z")
MAX_OPERATOR_RECORD_BYTES = 8 * 1024 * 1024
_CHECKS = {"scene_application", "reset", "native_state", "policy_observation", "action_step", "close",
           "robot_free_probe", "deterministic_scripted_control", "native_camera_readiness"}


def _require(condition: Any, code: str) -> None:
    if not condition:
        raise ValueError("task_family_native_runtime_" + code)


def _read(path: Path, *, code: str, expected_sha256: str | None = None) -> dict[str, Any]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as handle:
            status = os.fstat(handle.fileno())
            _require(stat.S_ISREG(status.st_mode) and status.st_uid in {0, os.getuid()}
                     and not status.st_mode & 0o022 and status.st_size <= MAX_OPERATOR_RECORD_BYTES,
                     "operator_record_untrusted")
            raw = handle.read(MAX_OPERATOR_RECORD_BYTES + 1)
    except OSError as exc:
        raise ValueError("task_family_native_runtime_" + code) from exc
    _require(len(raw) <= MAX_OPERATOR_RECORD_BYTES, "operator_record_too_large")
    if expected_sha256 is not None:
        _require("sha256:" + hashlib.sha256(raw).hexdigest() == expected_sha256, "qualification_bytes_changed")
    try:
        value = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError("task_family_native_runtime_" + code) from exc
    _require(isinstance(value, dict), code)
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _qualified_registration(native: Mapping[str, Any], selection: Mapping[str, Any],
                            catalog_path: Path, *, now: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate operator records before loading any third-party SDK code."""
    _require(isinstance(selection, Mapping) and set(selection) == {"runtime_id", "runtime_catalog_digest"}
             and isinstance(selection["runtime_id"], str) and selection["runtime_id"], "selection_invalid")
    catalog = _read(catalog_path, code="catalog_required")
    _require(catalog.get("schema_version") == CATALOG_SCHEMA
             and catalog.get("catalog_digest") == canonical_digest(catalog, digest_field="catalog_digest")
             and catalog.get("catalog_digest") == selection["runtime_catalog_digest"], "catalog_binding_mismatch")
    rows = catalog.get("runtimes")
    _require(isinstance(rows, list) and all(isinstance(row, dict) for row in rows), "catalog_invalid")
    matches = [row for row in rows if row.get("runtime_id") == selection["runtime_id"]]
    _require(len(matches) == 1, "not_registered")
    row = matches[0]
    _require(row.get("factory_id") == GYMNASIUM_FACTORY, "factory_unregistered")
    _require(row.get("runtime_identity_digest") == native["runtime_identity_digest"]
             and row.get("engine_id") == native["task_state_binding"]["engine_id"], "identity_mismatch")
    families = {native["task_family"], *(stage["task_family"] for stage in native["task_spec"]["task_success_contract"]["stages"])}
    _require(isinstance(row.get("task_families"), list)
             and all(isinstance(family, str) for family in row["task_families"])
             and families <= set(row["task_families"]),
             "family_not_qualified")
    _require(isinstance(row.get("robot_interfaces"), list)
             and native["robot_configuration"] in row["robot_interfaces"], "robot_interface_not_qualified")
    _require(isinstance(row.get("policy_interfaces"), list) and row["policy_interfaces"]
             and all(isinstance(interface, str) and interface for interface in row["policy_interfaces"]),
             "policy_interfaces_required")
    _require(isinstance(row.get("gymnasium_environment_id"), str) and row["gymnasium_environment_id"]
             and isinstance(row.get("registration_module"), str)
             and _MODULE.fullmatch(row["registration_module"])
             and isinstance(row.get("entry_point"), str) and ":" in row["entry_point"], "sdk_binding_invalid")
    entry_module, entry_name = row["entry_point"].split(":", 1)
    _require(_MODULE.fullmatch(entry_module) and entry_name.isidentifier(), "sdk_binding_invalid")
    sources = row.get("runtime_sources")
    _require(isinstance(sources, list) and sources and all(isinstance(source, dict)
             and set(source) == {"module", "sha256"} and isinstance(source["module"], str)
             and _MODULE.fullmatch(source["module"]) and _SHA.fullmatch(str(source["sha256"]))
             for source in sources), "source_identity_required")
    modules = [source["module"] for source in sources]
    _require(len(set(modules)) == len(modules) and {row["registration_module"], entry_module} <= set(modules),
             "source_identity_required")
    qualification_ref = row.get("qualification")
    _require(isinstance(qualification_ref, dict) and set(qualification_ref) == {"path", "sha256"}
             and _SHA.fullmatch(str(qualification_ref["sha256"])), "qualification_required")
    relative = Path(str(qualification_ref["path"]))
    _require(not relative.is_absolute() and ".." not in relative.parts, "qualification_path_invalid")
    qualification_path = catalog_path.parent / relative
    _require(qualification_path.resolve().is_relative_to(catalog_path.parent.resolve()), "qualification_path_invalid")
    qualification = _read(qualification_path, code="qualification_required", expected_sha256=qualification_ref["sha256"])
    expiry = qualification.get("expires_at_epoch")
    _require(qualification.get("schema_version") == QUALIFICATION_SCHEMA
             and qualification.get("runtime_registration_digest") == canonical_digest(
                 {key: value for key, value in row.items() if key != "qualification"})
             and qualification.get("decision") == "qualified_for_bounded_simulation"
             and isinstance(qualification.get("reviewed_by"), str) and qualification["reviewed_by"].strip()
             and type(expiry) in (int, float) and math.isfinite(expiry) and now < expiry
             and qualification.get("checks") == {key: True for key in _CHECKS}
             and qualification.get("physical_validation") is False
             and qualification.get("qualification_digest") == canonical_digest(
                 qualification, digest_field="qualification_digest"), "qualification_invalid")
    return copy.deepcopy(row), qualification


def _verify_installed_sources(registration: Mapping[str, Any]) -> None:
    # Resolving an SDK submodule may import its parent. Verify every declared
    # parent first; the operator must pin that executable package code too.
    sources = {row["module"]: row["sha256"] for row in registration["runtime_sources"]}
    for module in sources:
        parts = module.split(".")
        _require(all(".".join(parts[:index]) in sources for index in range(1, len(parts))),
                 "sdk_parent_source_required")
    for module in sorted(sources, key=lambda name: (name.count("."), name)):
        try:
            spec = importlib.util.find_spec(module)
            path = Path(spec.origin) if spec is not None and spec.origin else None
            _require(path is not None and path.is_file() and not path.is_symlink()
                     and _sha(path) == sources[module], "sdk_source_changed:" + module)
        except (ImportError, OSError, AttributeError, ValueError) as exc:
            raise ValueError("task_family_native_runtime_sdk_source_changed:" + module) from exc


def _asset_paths(native: Mapping[str, Any], asset_root: Path) -> dict[str, str]:
    root = asset_root.resolve()
    paths: dict[str, str] = {}
    for asset in native["assets"]:
        value = asset.get("bundle_relative_path", asset.get("path"))
        _require(isinstance(value, str) and value, "asset_path_required:" + asset["asset_id"])
        relative = Path(value)
        path = root / relative
        _require(not relative.is_absolute() and ".." not in relative.parts and path.is_file() and path.stat().st_size > 0
                 and not path.is_symlink() and path.resolve().is_relative_to(root),
                 "asset_path_invalid:" + asset["asset_id"])
        _require(_sha(path) == asset["content_digest"], "asset_bytes_changed:" + asset["asset_id"])
        paths[asset["asset_id"]] = str(path)
    return paths


class TaskFamilyNativeRuntime:
    """Forward the qualified SDK's applied scene, clock, sensors and actions."""

    def __init__(self, environment: Any, native: Mapping[str, Any], registration: Mapping[str, Any],
                 qualification: Mapping[str, Any], application: Mapping[str, Any]) -> None:
        self._environment = environment
        self._native = environment.unwrapped
        self._closed = False
        self._expected_robot_present = True
        self.native_runtime_contract = copy.deepcopy(native)
        self.runtime_registration = copy.deepcopy(registration)
        self.qualification = copy.deepcopy(qualification)
        self.construction_receipt = copy.deepcopy(application)

    @property
    def runtime_identity_digest(self) -> str:
        return self._native.runtime_identity_digest

    @property
    def engine_id(self) -> str:
        return self._native.engine_id

    @property
    def native_runtime_contract_digest(self) -> str:
        return self._native.native_runtime_contract_digest

    def reset(self, *, seed: int) -> Any:
        return self._native.reset(seed=seed)

    def _assert_identity(self) -> None:
        _require(self.runtime_identity_digest == self.native_runtime_contract["runtime_identity_digest"]
                 and self.engine_id == self.native_runtime_contract["task_state_binding"]["engine_id"]
                 and self.native_runtime_contract_digest == self.native_runtime_contract["contract_digest"],
                 "native_identity_readback_mismatch")

    def native_state(self) -> Any:
        self._assert_identity()
        state = self._native.native_state()
        self._assert_identity()
        _require(isinstance(state, Mapping) and state.get("robot_present") is self._expected_robot_present,
                 "native_robot_presence_readback_mismatch")
        return state

    def policy_observation(self) -> Any:
        return self._native.policy_observation()

    def step(self, action: list[float], *, duration_seconds: float) -> Any:
        return self._native.step_native_action(action, duration_seconds=duration_seconds)

    def set_robot_enabled(self, *, enabled: bool) -> Any:
        method = getattr(self._native, "set_robot_enabled", None)
        _require(callable(method), "native_robot_presence_control_required")
        result = method(enabled=enabled)
        self._expected_robot_present = enabled
        return result

    def step_without_robot_action(self, *, duration_seconds: float) -> Any:
        method = getattr(self._native, "step_without_robot_action", None)
        _require(callable(method), "native_unactuated_step_required")
        return method(duration_seconds=duration_seconds)

    def scripted_control_action(self, observation: Mapping[str, Any], *, step_index: int) -> Any:
        method = getattr(self._native, "scripted_control_action", None)
        _require(callable(method), "native_scripted_control_required")
        return method(observation, step_index=step_index)

    def native_scene_readiness(self) -> Any:
        method = getattr(self._native, "native_scene_readiness", None)
        _require(callable(method), "native_scene_readiness_required")
        return method()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._environment.close()


def build_task_family_native_runtime(native_contract: Mapping[str, Any], *,
                                    qualified_runtime_selection: Mapping[str, Any],
                                    runtime_catalog_path: str | Path, asset_root: str | Path) -> TaskFamilyNativeRuntime:
    """Build inside the worker using its trusted catalog, never a job's module.

    ``runtime_catalog_path`` must come from operator configuration. Qualification
    and asset/source bytes are checked before native construction. Allocation,
    rights and policy admission remain with the existing worker.
    """
    native = validate_task_family_native_runtime_contract(native_contract)
    catalog_path = Path(runtime_catalog_path)
    _require(catalog_path.is_absolute(), "catalog_path_invalid")
    registration, qualification = _qualified_registration(native, qualified_runtime_selection,
        catalog_path, now=time.time())
    paths = _asset_paths(native, Path(asset_root))
    _verify_installed_sources(registration)
    try:
        import gymnasium
    except ImportError as exc:
        raise ValueError("task_family_native_runtime_gymnasium_sdk_unavailable") from exc
    importlib.import_module(registration["registration_module"])
    try:
        spec = gymnasium.spec(registration["gymnasium_environment_id"])
    except Exception as exc:
        raise ValueError("task_family_native_runtime_sdk_environment_unregistered") from exc
    _require(spec.entry_point == registration["entry_point"]
             and not getattr(spec, "additional_wrappers", ()), "sdk_factory_identity_mismatch")
    # Environment kwargs in a shared Gym registry are mutable process state.
    # Keep constructor arguments in the qualified record and refuse divergence.
    kwargs = registration.get("environment_kwargs", {})
    _require(isinstance(kwargs, dict) and "render_mode" not in kwargs
             and dict(getattr(spec, "kwargs", {})) == kwargs, "sdk_factory_arguments_mismatch")
    environment = gymnasium.make(registration["gymnasium_environment_id"], render_mode="rgb_array")
    try:
        target = environment.unwrapped
        _require(all(callable(getattr(target, method, None)) for method in (
            "apply_native_task_contract", "reset", "native_state", "policy_observation", "step_native_action"))
            and callable(getattr(environment, "close", None)), "sdk_native_interface_missing")
        application = target.apply_native_task_contract(contract=copy.deepcopy(native), asset_paths=paths)
        try:
            application = json.loads(json.dumps(application, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("task_family_native_runtime_native_application_not_json") from exc
        from .task_evaluation_task_family_native_adapter import native_runtime_provenance
        expected = {**native_runtime_provenance(native), "runtime_identity_digest": native["runtime_identity_digest"],
            "engine_id": native["task_state_binding"]["engine_id"], "native_contract_applied": True,
            "asset_digests": {asset["asset_id"]: asset["content_digest"] for asset in native["assets"]},
            "task_state_binding_digest": canonical_digest(native["task_state_binding"]),
            "robot_configuration": native["robot_configuration"]}
        _require(isinstance(application, Mapping) and all(application.get(key) == value for key, value in expected.items()),
                 "native_application_readback_mismatch")
        _require(getattr(target, "runtime_identity_digest", None) == native["runtime_identity_digest"]
                 and getattr(target, "engine_id", None) == native["task_state_binding"]["engine_id"]
                 and getattr(target, "native_runtime_contract_digest", None) == native["contract_digest"],
                 "native_identity_readback_mismatch")
        return TaskFamilyNativeRuntime(environment, native, registration, qualification, application)
    except BaseException:
        close = getattr(environment, "close", None)
        if callable(close):
            close()
        raise


__all__ = ["build_task_family_native_runtime", "TaskFamilyNativeRuntime"]
