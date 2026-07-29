"""Evaluation Run spec: the general composition every eval job is built from.

An evaluation run is scene bundle + robot adapter + task/scenario pack +
policy adapter + runtime/provider profile + proof contract. Sites, robots,
tasks, and policies are configuration ("packs") passed into one engine —
never the shape of the engine itself. The historical G1 kitchen lane is the
first pack; its legacy schema/filename identifiers are pinned here verbatim
so previously emitted evidence stays valid.

This module is deliberately stdlib-only at import time (robot profiles and
scenario families are resolved lazily) so orchestrators can depend on it
without dragging provider/runtime imports.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

# ``evaluation_run_contract.EvaluationRunSpec`` is the canonical runtime leaf
# contract.  This module predates that contract and owns static compatibility
# packs for the historical kitchen/warehouse lanes.  Re-export the canonical
# compiler entrypoints, but keep the pack type explicitly named so new callers
# cannot accidentally build against a second runtime contract.
from . import evaluation_run_contract as _contract

DEFAULT_EVALUATION_RUN_ADAPTERS = _contract.DEFAULT_EVALUATION_RUN_ADAPTERS
EVALUATION_RUN_COMPONENTS = _contract.EVALUATION_RUN_COMPONENTS
EVALUATION_RUN_MODES = _contract.EVALUATION_RUN_MODES
EVALUATION_RUN_PLAN_SCHEMA_VERSION = _contract.EVALUATION_RUN_PLAN_SCHEMA_VERSION
EVALUATION_RUN_SCHEMA_VERSION = _contract.EVALUATION_RUN_SCHEMA_VERSION
EvaluationRunAdapterDescriptor = _contract.EvaluationRunAdapterDescriptor
EvaluationRunAdapterRegistry = _contract.EvaluationRunAdapterRegistry
compile_evaluation_run = _contract.compile_evaluation_run
default_evaluation_run_adapter_registry = (
    _contract.default_evaluation_run_adapter_registry
)
validate_evaluation_run_spec = _contract.validate_evaluation_run_spec

SPEC_SCHEMA_VERSION = "evaluation_run_spec.v1"

_VERSIONED_SCHEMA_RE = re.compile(r"\.v\d+$")


class EvaluationRunSpecError(RuntimeError):
    """Fail-closed spec error carrying the full blocker list."""

    def __init__(self, blockers: Sequence[str]):
        self.blockers = list(blockers)
        super().__init__("; ".join(self.blockers))


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


# ----------------------------- components -----------------------------


@dataclass(frozen=True)
class SceneBundle:
    """Which scene assets a run stands up, and how the worker addresses them."""

    scene_id: str
    main_usd_relative: str
    bundle_mount_name: str
    worker_bundle_dir: str = "/workspace/bundle"
    request_usd_key: str = "scene_usd"
    evidence_key_prefix: str = "scene"
    layout_validation_schema: str = "scene_asset_layout_validation.v1"
    inventory_schema: str = "scene_asset_inventory_checksums.v1"
    max_asset_archive_bytes: int = 2 * 1024 * 1024 * 1024

    @property
    def main_usd_basename(self) -> str:
        return Path(self.main_usd_relative).name

    @property
    def worker_main_usd_path(self) -> str:
        return f"{self.worker_bundle_dir}/{self.bundle_mount_name}/{self.main_usd_relative}"

    def worker_usd_path(self, selected_relative: str) -> str:
        return f"{self.worker_bundle_dir}/{self.bundle_mount_name}/{selected_relative}"

    def layout_label(self, selected_relative: str) -> str:
        snake = _snake(Path(self.main_usd_basename).stem)
        if not selected_relative:
            return "unknown"
        if selected_relative == self.main_usd_relative:
            return (f"collected_{snake}" if "/" in self.main_usd_relative else f"root_{snake}")
        if selected_relative == self.main_usd_basename:
            return f"root_{snake}"
        return f"nested_{snake}"

    def validation_blockers(self) -> list[str]:
        blockers: list[str] = []
        if not _string(self.scene_id):
            blockers.append("scene_id_missing")
        if not _string(self.main_usd_relative):
            blockers.append("main_usd_relative_missing")
        if not _string(self.bundle_mount_name):
            blockers.append("bundle_mount_name_missing")
        if not _string(self.worker_bundle_dir).startswith("/"):
            blockers.append("worker_bundle_dir_not_absolute")
        if not _string(self.request_usd_key):
            blockers.append("request_usd_key_missing")
        if not _string(self.evidence_key_prefix):
            blockers.append("evidence_key_prefix_missing")
        for field_name in ("layout_validation_schema", "inventory_schema"):
            if not _VERSIONED_SCHEMA_RE.search(_string(getattr(self, field_name))):
                blockers.append(f"{field_name}_not_versioned")
        if int(self.max_asset_archive_bytes) <= 0:
            blockers.append("max_asset_archive_bytes_not_positive")
        return blockers


@dataclass(frozen=True)
class RobotAdapter:
    """Which embodiment executes the run; resolves to a registered RobotProfile."""

    robot_profile_id: str
    robot_usd_relative: str
    request_usd_key: str = "robot_usd"

    def resolve_profile(self):
        from .scene_placement.robot_profile import get_robot_profile

        return get_robot_profile(self.robot_profile_id)

    def validation_blockers(self) -> list[str]:
        blockers: list[str] = []
        if not _string(self.robot_profile_id):
            blockers.append("robot_profile_id_missing")
        else:
            from .scene_placement.robot_profile import known_robot_ids

            if self.robot_profile_id not in known_robot_ids():
                blockers.append(f"robot_profile_unknown:{self.robot_profile_id}")
        if not _string(self.robot_usd_relative):
            blockers.append("robot_usd_relative_missing")
        if not _string(self.request_usd_key):
            blockers.append("request_usd_key_missing")
        return blockers


@dataclass(frozen=True)
class TaskScenarioPack:
    """The scenarios a run executes; fail-closed when neither inline rows nor a task file exist."""

    pack_id: str
    scenarios: tuple[Mapping[str, Any], ...] = ()
    task_file: str | None = None
    description: str = ""

    def validation_blockers(self) -> list[str]:
        blockers: list[str] = []
        if not _string(self.pack_id):
            blockers.append("pack_id_missing")
        if not self.scenarios and not _string(self.task_file):
            blockers.append("no_scenarios_or_task_file")
        for index, row in enumerate(self.scenarios):
            if not _string(dict(row).get("scenario_id")):
                blockers.append(f"scenario_id_missing_at_index_{index}")
        return blockers


@dataclass(frozen=True)
class PolicyAdapter:
    """Which policy drives the robot, and how remote policy runtimes are configured."""

    policy_id: str
    remote_runtime_policy_ids: tuple[str, ...] = ()
    policy_command_env: str = ""
    policy_server_url_env: str = ""
    policy_runtime_mode_env: str = ""

    def validation_blockers(self) -> list[str]:
        blockers: list[str] = []
        if not _string(self.policy_id):
            blockers.append("policy_id_missing")
        if self.remote_runtime_policy_ids:
            for field_name in ("policy_command_env", "policy_server_url_env", "policy_runtime_mode_env"):
                if not _string(getattr(self, field_name)):
                    blockers.append(f"{field_name}_missing_for_remote_runtime_policies")
        return blockers


@dataclass(frozen=True)
class RuntimeProviderProfile:
    """Where and under what caps the run is allowed to spend."""

    lane_id: str
    launch_name: str
    default_providers: tuple[str, ...]
    image_ref_env: str
    image_ref_file_env: str = ""
    fallback_image_ref_env: str = ""
    default_image_ref_file: str = ""
    default_image_ref: str = ""
    min_gpu_ram_mb: int = 24000
    requires_rtx: bool = True
    max_spend_env: str = ""
    default_startup_no_runtime_timeout_seconds: int = 900

    def validation_blockers(self) -> list[str]:
        blockers: list[str] = []
        if not _string(self.lane_id):
            blockers.append("lane_id_missing")
        if not _string(self.launch_name):
            blockers.append("launch_name_missing")
        if not self.default_providers or not all(_string(p) for p in self.default_providers):
            blockers.append("default_providers_missing")
        if not _string(self.image_ref_env):
            blockers.append("image_ref_env_missing")
        if int(self.min_gpu_ram_mb) <= 0:
            blockers.append("min_gpu_ram_mb_not_positive")
        if int(self.default_startup_no_runtime_timeout_seconds) <= 0:
            blockers.append("startup_no_runtime_timeout_not_positive")
        return blockers


@dataclass(frozen=True)
class ProofContractBinding:
    """The evidence schemas and closure contract a run must emit to claim anything."""

    job_schema: str
    job_manifest_filename: str
    launch_attempt_trace_filename: str
    request_schema: str
    bundle_schema: str
    harness_schema: str
    launch_attempts_schema: str
    spend_guard_schema: str
    closure_schema: str
    closure_required_blocker: str
    result_filename: str

    _VERSIONED_FIELDS = (
        "job_schema",
        "request_schema",
        "bundle_schema",
        "harness_schema",
        "launch_attempts_schema",
        "spend_guard_schema",
        "closure_schema",
    )
    _JSON_FILENAME_FIELDS = (
        "job_manifest_filename",
        "launch_attempt_trace_filename",
        "result_filename",
    )

    def validation_blockers(self) -> list[str]:
        blockers: list[str] = []
        for field_name in self._VERSIONED_FIELDS:
            if not _VERSIONED_SCHEMA_RE.search(_string(getattr(self, field_name))):
                blockers.append(f"{field_name}_not_versioned")
        for field_name in self._JSON_FILENAME_FIELDS:
            if not _string(getattr(self, field_name)).endswith(".json"):
                blockers.append(f"{field_name}_not_json")
        if not _string(self.closure_required_blocker):
            blockers.append("closure_required_blocker_missing")
        return blockers


# ----------------------------- composed spec -----------------------------

_COMPONENT_MANIFEST_KEYS: tuple[tuple[str, str, type], ...] = (
    ("scene_bundle", "scene", SceneBundle),
    ("robot_adapter", "robot", RobotAdapter),
    ("task_scenario_pack", "tasks", TaskScenarioPack),
    ("policy_adapter", "policy", PolicyAdapter),
    ("runtime_provider_profile", "runtime", RuntimeProviderProfile),
    ("proof_contract", "proof", ProofContractBinding),
)

_TUPLE_FIELDS = {
    "scenarios",
    "remote_runtime_policy_ids",
    "default_providers",
}


@dataclass(frozen=True)
class LegacyEvaluationPackSpec:
    """Deprecated static pack definition translated into a canonical leaf spec.

    The wire schema remains ``evaluation_run_spec.v1`` for historical artifact
    readability. New execution callers must use
    :class:`evaluation_run_contract.EvaluationRunSpec` (``evaluation_run.v1``).
    """

    spec_id: str
    scene: SceneBundle
    robot: RobotAdapter
    tasks: TaskScenarioPack
    policy: PolicyAdapter
    runtime: RuntimeProviderProfile
    proof: ProofContractBinding
    claim_boundary: str = ""

    def validation_blockers(self) -> list[str]:
        blockers: list[str] = []
        if not _string(self.spec_id):
            blockers.append("spec:spec_id_missing")
        for _, attr, _ in _COMPONENT_MANIFEST_KEYS:
            component = getattr(self, attr)
            blockers.extend(f"{attr}:{blocker}" for blocker in component.validation_blockers())
        return blockers

    def assert_valid(self) -> "LegacyEvaluationPackSpec":
        blockers = self.validation_blockers()
        if blockers:
            raise EvaluationRunSpecError(blockers)
        return self

    def to_manifest(self) -> dict:
        blockers = self.validation_blockers()
        manifest: dict[str, Any] = {
            "schema_version": SPEC_SCHEMA_VERSION,
            "spec_id": self.spec_id,
            "claim_boundary": self.claim_boundary,
        }
        for manifest_key, attr, _ in _COMPONENT_MANIFEST_KEYS:
            manifest[manifest_key] = dataclasses.asdict(getattr(self, attr))
        manifest["validation"] = {
            "status": "PASS" if not blockers else "FAIL",
            "blockers": blockers,
        }
        return manifest


# Compatibility alias for existing imports and persisted pack builders.  The
# distinct canonical name above is the migration signal; removing this alias
# would break historical callers without improving the runtime boundary.
EvaluationRunSpec = LegacyEvaluationPackSpec


def _component_from_dict(component_type: type, payload: Mapping[str, Any], *, manifest_key: str):
    if not isinstance(payload, Mapping):
        raise EvaluationRunSpecError([f"{manifest_key}_not_a_mapping"])
    field_names = {f.name for f in dataclasses.fields(component_type)}
    unknown = sorted(set(payload) - field_names)
    if unknown:
        raise EvaluationRunSpecError(
            [f"{manifest_key}_unknown_key:{key}" for key in unknown]
        )
    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _TUPLE_FIELDS and isinstance(value, (list, tuple)):
            if key == "scenarios":
                value = tuple(dict(row) for row in value)
            else:
                value = tuple(value)
        kwargs[key] = value
    try:
        return component_type(**kwargs)
    except TypeError as exc:
        raise EvaluationRunSpecError([f"{manifest_key}_invalid:{exc}"]) from exc


def evaluation_run_spec_from_dict(payload: Mapping[str, Any]) -> EvaluationRunSpec:
    """Strict parse: unknown keys, missing components, or schema drift fail closed."""
    if not isinstance(payload, Mapping):
        raise EvaluationRunSpecError(["spec_payload_not_a_mapping"])
    allowed = {"schema_version", "spec_id", "claim_boundary", "validation"}
    allowed.update(key for key, _, _ in _COMPONENT_MANIFEST_KEYS)
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise EvaluationRunSpecError([f"spec_unknown_key:{key}" for key in unknown])
    declared_schema = _string(payload.get("schema_version"))
    if declared_schema and declared_schema != SPEC_SCHEMA_VERSION:
        raise EvaluationRunSpecError([f"spec_schema_mismatch:{declared_schema}"])
    spec_id = _string(payload.get("spec_id"))
    if not spec_id:
        raise EvaluationRunSpecError(["spec_id_missing"])
    components: dict[str, Any] = {}
    missing = [key for key, _, _ in _COMPONENT_MANIFEST_KEYS if key not in payload]
    if missing:
        raise EvaluationRunSpecError([f"spec_component_missing:{key}" for key in missing])
    for manifest_key, attr, component_type in _COMPONENT_MANIFEST_KEYS:
        components[attr] = _component_from_dict(
            component_type, payload[manifest_key], manifest_key=manifest_key
        )
    return EvaluationRunSpec(
        spec_id=spec_id,
        claim_boundary=_string(payload.get("claim_boundary")),
        **components,
    )


def evaluation_run_spec_from_json_file(path: str | Path) -> EvaluationRunSpec:
    return evaluation_run_spec_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def legacy_evaluation_pack_to_leaf_spec(
    pack: LegacyEvaluationPackSpec,
    *,
    run_id: str,
    scene_uri: str,
    scene_content_digest: str,
    robot_asset_ref: str,
    policy_id: str | None = None,
    providers: Sequence[str] | None = None,
    simulator: str = "isaac_sim",
    max_spend_usd: float = 0.0,
    required_evidence: Sequence[str] = ("adapter_execution_receipt",),
) -> dict[str, Any]:
    """Translate a legacy pack into the canonical six-part runtime leaf.

    Runtime identity cannot be inferred from a static pack, so scene URI,
    content digest, and robot asset reference are mandatory.  This prevents a
    legacy default from silently becoming qualification or execution evidence.
    """

    pack.assert_valid()
    if not _string(run_id):
        raise EvaluationRunSpecError(["leaf_run_id_missing"])
    if not _string(scene_uri):
        raise EvaluationRunSpecError(["leaf_scene_uri_missing"])
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", _string(scene_content_digest)):
        raise EvaluationRunSpecError(["leaf_scene_content_digest_invalid"])
    if not _string(robot_asset_ref):
        raise EvaluationRunSpecError(["leaf_robot_asset_ref_missing"])

    scenario_rows: list[dict[str, Any]] = []
    task_ids: list[str] = []
    for index, raw in enumerate(pack.tasks.scenarios):
        row = dict(raw)
        task_id = _string(row.get("task_id")) or pack.tasks.pack_id
        row["task_id"] = task_id
        row.setdefault("scenario_id", f"{pack.tasks.pack_id}-{index + 1}")
        scenario_rows.append(row)
        if task_id not in task_ids:
            task_ids.append(task_id)

    selected_providers = [
        _string(value)
        for value in (providers or pack.runtime.default_providers)
        if _string(value)
    ]
    leaf = {
        "schema_version": EVALUATION_RUN_SCHEMA_VERSION,
        "run_id": _string(run_id),
        "mode": "evaluate",
        "scene_bundle": {
            "adapter_id": "openusd_scene_bundle",
            "adapter_version": "1",
            "bundle_id": pack.scene.scene_id,
            "uri": _string(scene_uri),
            "entrypoint": pack.scene.main_usd_relative,
            "content_digest": _string(scene_content_digest),
        },
        "robot_adapter": {
            "adapter_id": "isaac_robot_asset",
            "adapter_version": "1",
            "robot_profile_id": pack.robot.robot_profile_id,
            "asset_ref": _string(robot_asset_ref),
        },
        "task_scenario_pack": {
            "adapter_id": "manifest_task_scenario_pack",
            "adapter_version": "1",
            "pack_id": pack.tasks.pack_id,
            "tasks": [{"task_id": task_id} for task_id in task_ids],
            "scenarios": scenario_rows,
        },
        "policy_adapter": {
            "adapter_id": "isaac_g1_deterministic_controller",
            "adapter_version": "1",
            "policy_id": _string(policy_id) or pack.policy.policy_id,
            "observation_schema_ref": "legacy_pack_observation.v1",
            "action_schema_ref": "legacy_pack_action.v1",
        },
        "runtime_provider_profile": {
            "adapter_id": "isaac_provider_runtime",
            "adapter_version": "1",
            "profile_id": pack.runtime.lane_id,
            "providers": selected_providers,
            "simulator": _string(simulator),
            "max_spend_usd": float(max_spend_usd),
        },
        "proof_contract": {
            "adapter_id": "declared_evidence_proof_contract",
            "adapter_version": "1",
            "contract_id": pack.proof.job_schema,
            "required_evidence": [
                _string(value) for value in required_evidence if _string(value)
            ],
            "claim_ceiling": {
                "level": "legacy_pack_execution_only",
                "physical_success": False,
                "deployment_readiness": False,
                "safety_certification": False,
            },
            "prohibited_claims": [
                "physical_success",
                "deployment_readiness",
                "safety_certification",
            ],
        },
        "metadata": {
            "translated_from_schema": SPEC_SCHEMA_VERSION,
            "legacy_pack_id": pack.spec_id,
            "legacy_defaults_are_qualification_evidence": False,
        },
    }
    validation = validate_evaluation_run_spec(leaf)
    if validation["status"] != "passed":
        raise EvaluationRunSpecError(validation["errors"])
    return leaf


# ----------------------------- generic scene asset inspection -----------------------------


def inspect_scene_asset_namelist(
    names: Sequence[str],
    *,
    scene: SceneBundle,
    source: str,
    byte_size: int | None = None,
) -> dict:
    """Validate a staged scene asset zip/tree before a GPU worker tries to open it."""
    prefix = scene.evidence_key_prefix
    basename = scene.main_usd_basename
    files = sorted(
        {
            str(name).lstrip("/")
            for name in names
            if str(name).strip() and not str(name).endswith("/")
        }
    )
    candidates = list(dict.fromkeys([scene.main_usd_relative, basename]))
    selected = next((candidate for candidate in candidates if candidate in files), "")
    if not selected:
        nested = sorted(
            (name for name in files if name.endswith(f"/{basename}") or name == basename),
            key=lambda name: (len(Path(name).parts), name),
        )
        selected = nested[0] if nested else ""
    blockers: list[str] = []
    if not files:
        blockers.append(f"{prefix}_asset_empty")
    if not selected:
        blockers.append(f"{prefix}_main_usd_missing")
    return {
        "schema_version": scene.layout_validation_schema,
        "status": "PASS" if not blockers else "FAIL",
        "source": source,
        "blockers": blockers,
        "file_count": len(files),
        "zip_bytes": byte_size,
        f"selected_{prefix}_main_usd_relative": selected or None,
        f"expected_worker_{prefix}_usd": (
            scene.worker_usd_path(selected) if selected else None
        ),
        "layout": scene.layout_label(selected),
        "sample_files": files[:40],
        "raw_url_values_recorded": False,
        "claim_boundary": (
            f"{prefix.capitalize()} asset layout validation proves only that the staged asset "
            f"bundle contains a usable {basename} path for the worker request. It does not prove "
            "Isaac can render the scene, task success, WAM quality, physical reach, safety, or "
            "deployment readiness."
        ),
    }


def inspect_scene_asset_dir_layout(path: str | Path, *, scene: SceneBundle) -> dict:
    root = Path(path)
    prefix = scene.evidence_key_prefix
    if not root.is_dir():
        return {
            "schema_version": scene.layout_validation_schema,
            "status": "FAIL",
            "source": "local_asset_dir",
            "blockers": [f"{prefix}_asset_dir_missing"],
            "path": str(root),
            "raw_url_values_recorded": False,
        }
    names = [
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file()
    ]
    detail = inspect_scene_asset_namelist(names, scene=scene, source="local_asset_dir")
    detail["path"] = str(root)
    return detail


def inspect_scene_asset_zip(data: bytes, *, scene: SceneBundle, source: str) -> dict:
    """Inspect zip bytes and, on PASS, attach a per-file sha256 content inventory."""
    prefix = scene.evidence_key_prefix
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        detail = inspect_scene_asset_namelist(
            zf.namelist(), scene=scene, source=source, byte_size=len(data)
        )
        if detail.get("status") == "PASS":
            main_usd = str(detail[f"selected_{prefix}_main_usd_relative"])
            files: list[dict[str, Any]] = []
            for info in sorted(zf.infolist(), key=lambda item: item.filename):
                if info.is_dir():
                    continue
                digest = hashlib.sha256()
                with zf.open(info, "r") as member:
                    while chunk := member.read(1024 * 1024):
                        digest.update(chunk)
                files.append(
                    {
                        "path": info.filename,
                        "sha256": digest.hexdigest(),
                        "bytes": int(info.file_size),
                    }
                )
            detail["content_inventory"] = {
                "schema_version": scene.inventory_schema,
                "main_usd": main_usd,
                "file_count": len(files),
                "total_bytes": sum(int(item["bytes"]) for item in files),
                "archive_sha256": hashlib.sha256(data).hexdigest(),
                "files": files,
            }
    return detail


# ----------------------------- generic runner request -----------------------------


def build_runner_request(
    spec: EvaluationRunSpec,
    *,
    scenarios: Sequence[Mapping[str, Any]],
    steps: int,
    policy_id: str | None = None,
    scene_main_usd_relative: str | None = None,
    robot_usd_relative: str | None = None,
    render_noise_audit_plan: Mapping[str, Any] | None = None,
) -> dict:
    """The runner's request.json for any pack; keys come from the spec, not the engine."""
    scene_relative = _string(scene_main_usd_relative) or spec.scene.main_usd_relative
    request: dict[str, Any] = {
        "schema_version": spec.proof.request_schema,
        spec.scene.request_usd_key: spec.scene.worker_usd_path(scene_relative),
        spec.robot.request_usd_key: _string(robot_usd_relative) or spec.robot.robot_usd_relative,
        "policy_id": _string(policy_id) or spec.policy.policy_id,
        "steps": steps,
        "scenarios": list(scenarios),
    }
    if render_noise_audit_plan is not None:
        request["render_noise_audit"] = dict(render_noise_audit_plan)
    return request


# ----------------------------- pack registry -----------------------------

_PACK_REGISTRY: dict[str, EvaluationRunSpec] = {}


def register_evaluation_pack(spec: EvaluationRunSpec) -> EvaluationRunSpec:
    spec.assert_valid()
    if spec.spec_id in _PACK_REGISTRY:
        raise EvaluationRunSpecError([f"evaluation_pack_already_registered:{spec.spec_id}"])
    _PACK_REGISTRY[spec.spec_id] = spec
    return spec


def get_evaluation_pack(pack_id: str) -> EvaluationRunSpec:
    try:
        return _PACK_REGISTRY[pack_id]
    except KeyError:
        raise EvaluationRunSpecError(
            [f"evaluation_pack_unknown:{pack_id}", f"known_packs:{','.join(sorted(_PACK_REGISTRY))}"]
        ) from None


def known_evaluation_pack_ids() -> tuple[str, ...]:
    return tuple(sorted(_PACK_REGISTRY))


# ----------------------------- built-in packs -----------------------------

_GROOT_REMOTE_POLICY_IDS = (
    "groot_sonic",
    "groot",
    "groot_n17_sonic",
    "unitree_groot_n17_sonic_policy",
)

_G1_RUNTIME_COMMON = dict(
    default_providers=("digitalocean",),
    image_ref_env="BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF",
    image_ref_file_env="BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF_FILE",
    fallback_image_ref_env="BLUEPRINT_ROBOT_EVAL_WORKER_IMAGE_REF",
    default_image_ref_file="~/.blueprint-secrets/isaac_eval_worker_image_ref",
    default_image_ref="docker.io/nijelhunt/blueprint-isaac-eval-worker:20260626-faststart-amd64",
    min_gpu_ram_mb=48000,
    requires_rtx=True,
    max_spend_env="BLUEPRINT_ISAAC_G1_MAX_SPEND_USD",
    default_startup_no_runtime_timeout_seconds=900,
)

_G1_POLICY_ADAPTER = PolicyAdapter(
    policy_id="blueprint_default_walk_to_target_smoke_policy",
    remote_runtime_policy_ids=_GROOT_REMOTE_POLICY_IDS,
    policy_command_env="BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND",
    policy_server_url_env="BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL",
    policy_runtime_mode_env="BLUEPRINT_ISAAC_G1_GROOT_POLICY_RUNTIME_MODE",
)

_G1_ROBOT_USD_RELATIVE = "Isaac/Robots/Unitree/G1/g1.usd"


def _build_g1_kitchen_pack() -> EvaluationRunSpec:
    """The historical kitchen lane, expressed as pure configuration. Every identifier
    below predates this module and must stay byte-identical to keep old evidence valid."""
    return EvaluationRunSpec(
        spec_id="g1_kitchen",
        scene=SceneBundle(
            scene_id="lightwheel_kitchen",
            main_usd_relative="Collected_KitchenRoom/KitchenRoom.usd",
            bundle_mount_name="kitchen",
            request_usd_key="kitchen_usd",
            evidence_key_prefix="kitchen",
            layout_validation_schema="kitchen_asset_layout_validation.v1",
            inventory_schema="kitchen_asset_inventory_checksums.v1",
        ),
        robot=RobotAdapter(
            robot_profile_id="unitree_g1",
            robot_usd_relative=_G1_ROBOT_USD_RELATIVE,
            request_usd_key="g1_usd",
        ),
        tasks=TaskScenarioPack(
            pack_id="g1_kitchen_walk_to_target",
            scenarios=(
                {
                    "scenario_id": "kitchen_walk_to_target_smoke",
                    "spawn_position_xyz": [0.0, 0.0, 0.8],
                    "target_position_xyz": [2.49, 1.15, 1.02],
                },
            ),
            description="MuJoCo-parity walk-to-target smoke scenarios in the kitchen scene.",
        ),
        policy=_G1_POLICY_ADAPTER,
        runtime=RuntimeProviderProfile(
            lane_id="isaac_g1_kitchen_parity",
            launch_name="blueprint-isaac-g1-kitchen-parity",
            **_G1_RUNTIME_COMMON,
        ),
        proof=ProofContractBinding(
            job_schema="isaac_g1_kitchen_parity_job.v1",
            job_manifest_filename="isaac_g1_kitchen_parity_job_manifest.json",
            launch_attempt_trace_filename="isaac_g1_kitchen_parity_launch_attempts.json",
            request_schema="isaac_g1_kitchen_parity_request.v1",
            bundle_schema="isaac_g1_kitchen_parity_bundle.v2",
            harness_schema="isaac_g1_kitchen_parity_harness.v1",
            launch_attempts_schema="isaac_g1_kitchen_parity_launch_attempts.v1",
            spend_guard_schema="isaac_g1_kitchen_parity_prelaunch_spend_guard.v1",
            closure_schema="g1_kitchen_attempt_closure.v1",
            closure_required_blocker="g1_kitchen_attempt_closure_missing",
            result_filename="isaac_g1_kitchen_parity_result.json",
        ),
        claim_boundary=(
            "An evaluation-run spec proves only composition and identifier compatibility. "
            "It does not prove Isaac renders the scene, task success, WAM quality, physical "
            "reach, safety, or deployment readiness."
        ),
    )


def _warehouse_scenarios() -> tuple[dict, ...]:
    from .warehouse_isaac_scenarios import WAREHOUSE_SCENARIO_DEFINITIONS

    rows: list[dict] = []
    for definition in WAREHOUSE_SCENARIO_DEFINITIONS:
        waypoints = [list(point) for point in definition["route_waypoints"]]
        rows.append(
            {
                "scenario_id": str(definition["scenario_id"]),
                "spawn_position_xyz": waypoints[0],
                "target_position_xyz": waypoints[-1],
                "task_id": str(definition["task_id"]),
                "task_text": str(definition["task_text"]),
                "target_object_id": str(definition["target_object_id"]),
            }
        )
    return tuple(rows)


def _build_g1_warehouse_pack() -> EvaluationRunSpec:
    return EvaluationRunSpec(
        spec_id="g1_warehouse",
        scene=SceneBundle(
            scene_id="warehouse_task_min",
            main_usd_relative="Collected_WarehouseRoom/WarehouseRoom.usd",
            bundle_mount_name="scene",
        ),
        robot=RobotAdapter(
            robot_profile_id="unitree_g1",
            robot_usd_relative=_G1_ROBOT_USD_RELATIVE,
        ),
        tasks=TaskScenarioPack(
            pack_id="g1_warehouse_material_handling",
            scenarios=_warehouse_scenarios(),
            description="Warehouse scenario family rows projected onto the generic runner request.",
        ),
        policy=_G1_POLICY_ADAPTER,
        runtime=RuntimeProviderProfile(
            lane_id="isaac_g1_warehouse_scenarios",
            launch_name="blueprint-isaac-g1-warehouse",
            **_G1_RUNTIME_COMMON,
        ),
        proof=ProofContractBinding(
            job_schema="isaac_g1_warehouse_job.v1",
            job_manifest_filename="isaac_g1_warehouse_job_manifest.json",
            launch_attempt_trace_filename="isaac_g1_warehouse_launch_attempts.json",
            request_schema="isaac_g1_warehouse_request.v1",
            bundle_schema="isaac_g1_warehouse_bundle.v1",
            harness_schema="isaac_g1_warehouse_harness.v1",
            launch_attempts_schema="isaac_g1_warehouse_launch_attempts.v1",
            spend_guard_schema="isaac_g1_warehouse_prelaunch_spend_guard.v1",
            closure_schema="g1_warehouse_attempt_closure.v1",
            closure_required_blocker="g1_warehouse_attempt_closure_missing",
            result_filename="isaac_g1_warehouse_result.json",
        ),
        claim_boundary=(
            "pack_definition_only: this pack proves the engine accepts a second site as pure "
            "configuration. No GPU run has executed it; nothing beyond spec composition may be "
            "claimed until a live run emits its proof contract."
        ),
    )


register_evaluation_pack(_build_g1_kitchen_pack())
register_evaluation_pack(_build_g1_warehouse_pack())


# ----------------------------- CLI -----------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect, emit, or validate evaluation-run pack specs."
    )
    parser.add_argument("--pack", help="Built-in pack id to emit as a spec manifest.")
    parser.add_argument("--out", help="Where to write the spec manifest JSON.")
    parser.add_argument("--list-packs", action="store_true", help="List registered pack ids.")
    parser.add_argument("--spec-json", help="Validate an external spec manifest file.")
    args = parser.parse_args(argv)

    if args.list_packs:
        for pack_id in known_evaluation_pack_ids():
            print(pack_id)
        return 0

    if args.spec_json:
        try:
            spec = evaluation_run_spec_from_json_file(args.spec_json)
        except EvaluationRunSpecError as exc:
            print(json.dumps({"status": "FAIL", "blockers": exc.blockers}, indent=2))
            return 1
        blockers = spec.validation_blockers()
        print(json.dumps({"status": "PASS" if not blockers else "FAIL", "blockers": blockers}, indent=2))
        return 0 if not blockers else 1

    if args.pack:
        spec = get_evaluation_pack(args.pack)
        manifest = spec.to_manifest()
        rendered = json.dumps(manifest, indent=2)
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
