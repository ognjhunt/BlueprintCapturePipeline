"""Execute the CPU stages of a scene-configuration run before any GPU is rented.

Only native Isaac import qualification and scene assembly (stages 5-6) need
the Isaac host. The prepared background, Astra CAD/Blender authoring and
static SimReady qualification (stages 1-4) are CPU work. The control plane
runs that prefix from the same sealed bundle, through the bundle's own
entrypoint, under the same paid attempt authority and the same OpenAI stage
gates, at the *same per-run logical paths* the paid run uses
(``<work_dir>/task_evaluation_scene_configuration_provider_bundle``): the
resume binding and every artifact record are path-bound, so the paid run
restores the sealed prefix at those exact paths and adopts it without
rewriting anything it signed.

Transport mirrors the ArtiFixer semantic pretraining capsule: exact bytes,
a digest- and size-bound reference the runtime verifies before use, and no
GPU call. A prefix that does not complete never rents a GPU.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable, Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json

STAGE_LIMIT_ENV = "BLUEPRINT_SCENE_CONFIGURATION_CPU_PRESTAGE_STAGE_LIMIT"
WORK_DIR_ENV = "BLUEPRINT_SCENE_CONFIGURATION_CPU_PRESTAGE_WORK_DIR"
PREFIX_URL_ENV = "BLUEPRINT_SCENE_CONFIGURATION_STAGE_PREFIX_CAPSULE_URL"
PREFIX_SHA_ENV = "BLUEPRINT_SCENE_CONFIGURATION_STAGE_PREFIX_CAPSULE_SHA256"
PREFIX_BYTES_ENV = "BLUEPRINT_SCENE_CONFIGURATION_STAGE_PREFIX_CAPSULE_BYTES"
RECEIPT_SCHEMA = "task_evaluation_scene_configuration_cpu_prestage_receipt.v1"
TRANSPORT_SCHEMA = "task_evaluation_scene_configuration_cpu_prestage_transport.v1"
RESTORE_SCHEMA = "task_evaluation_scene_configuration_cpu_prestage_restore.v1"
MARKER_SCHEMA = "scene_configuration_completed_stage_checkpoint.v1"
# The paid run's work dir (``WORK_DIR=/workspace`` in the Vast onstart).
DEFAULT_WORK_DIR = Path("/workspace")
BUNDLE_DIRNAME = "task_evaluation_scene_configuration_provider_bundle"
CHECKPOINT_NAME = "task_evaluation_scene_configuration_stage_checkpoint.zip"
ENTRYPOINT = "provider_runtime/run_task_evaluation_scene_configuration_provider.sh"
MANIFEST_NAME = "provider_runtime/task_evaluation_scene_configuration_provider_bundle.v1.json"
RESULT_NAME = "task_evaluation_scene_configuration_provider_result.v1.json"
MAX_CAPSULE_BYTES = 8 * 1024**3
DEFAULT_TTL_SECONDS = 4 * 3600
DEFAULT_CLOSURE_RESERVE_SECONDS = 600
PRESTAGE_START_MARGIN_SECONDS = 900
# Adapters whose stages run on a CPU host. A limit that would include any
# other adapter is refused: the paid run keeps those stages.
CPU_ADAPTERS = frozenset({
    "website_prepared_appearance", "website_prepared_collision",
    "provided_mesh_appearance_excision", "provided_mesh_rigid_authoring",
    "content_agents_rigid_replacement", "simready_static_rigid_qualification",
})
_MARKER_NAME = "completed_stage_checkpoint.json"
_TRANSPORT_NAME = "cpu_prestage_transport.json"
_BINDING_NAME = "astra_same_run_resume_binding.json"


class CpuPrestageError(RuntimeError):
    """The CPU prefix could not be executed, archived or restored safely."""


def _require(ok: bool, code: str) -> None:
    if not ok:
        raise CpuPrestageError("cpu_prestage_" + code)


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), "record_path_invalid")
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), "record_invalid")
    return value


def _safe_infos(zipped: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    infos = zipped.infolist()
    _require(len(infos) <= 200_000 and sum(i.file_size for i in infos) <= MAX_CAPSULE_BYTES,
             "archive_budget_exceeded")
    _require(len({i.filename for i in infos}) == len(infos), "archive_duplicate_member")
    for info in infos:
        parts = Path(info.filename).parts
        _require(bool(parts) and not info.filename.startswith("/") and ".." not in parts
                 and not Path(info.filename).is_absolute(), "archive_path_invalid")
    return infos


def _extract(archive: Path, root: Path, *, allowed: Callable[[str], bool] | None = None) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with zipfile.ZipFile(archive) as zipped:
        for info in _safe_infos(zipped):
            if allowed is not None and not allowed(info.filename):
                continue
            zipped.extract(info, root)
            extracted.append(info.filename)
    return extracted


def _bundle_manifest(bundle: Path) -> dict[str, Any]:
    with zipfile.ZipFile(bundle) as zipped:
        _require(MANIFEST_NAME in zipped.namelist() and ENTRYPOINT in zipped.namelist(), "bundle_layout_invalid")
        return json.loads(zipped.read(MANIFEST_NAME))


def prestage_stage_limit(bundle_receipt: Mapping[str, Any], environment: Mapping[str, str]) -> str | None:
    """The configured limit, or None; a limit the paid run could not adopt is refused."""
    limit = str(environment.get(STAGE_LIMIT_ENV) or "").strip()
    if not limit:
        return None
    from .task_evaluation_scene_configuration_bundle import portable_construction_envelope

    _require(int(bundle_receipt.get("carried_completed_stage_count") or 0) == 0, "carried_prefix_conflict")
    stages = portable_construction_envelope(bundle_receipt)["recipe"]["stage_sequence"]
    ids = [str(stage["stage_id"]) for stage in stages]
    _require(limit in ids, "stage_limit_invalid")
    for stage in stages[: ids.index(limit) + 1]:
        _require(str(stage["adapter"]["id"]) in CPU_ADAPTERS, "stage_limit_includes_gpu_stage")
    # Only the Astra backend seals same-root checkpoints the paid run adopts.
    manifest = _bundle_manifest(Path(str(bundle_receipt["bundle_path"])))
    _require(manifest.get("replacement_authoring_backend") == "astra_cad_blender_v1", "backend_not_resumable")
    return limit


def prestage_ttl_seconds(bundle_receipt: Mapping[str, Any], stage_limit: str, *,
                         closure_reserve_seconds: int = DEFAULT_CLOSURE_RESERVE_SECONDS) -> int:
    """The prefix's own bounded deadline: the scheduled stages' allowances, no GPU stage."""
    from .task_evaluation_scene_configuration_bundle import portable_construction_envelope
    from .task_evaluation_scene_configuration_runtime_budget import required_remaining_stage_seconds

    stages = portable_construction_envelope(bundle_receipt)["recipe"]["stage_sequence"]
    ids = [str(stage["stage_id"]) for stage in stages]
    _require(stage_limit in ids, "stage_limit_invalid")
    scheduled = stages[: ids.index(stage_limit) + 1]
    return (required_remaining_stage_seconds(scheduled, start_index=0) + int(closure_reserve_seconds)
            + PRESTAGE_START_MARGIN_SECONDS)


def validate_stage_prefix_capsule(archive: Path, *, expected_stage_ids: list[str] | None = None) -> dict[str, Any]:
    """The runner's completed-prefix checkpoint with its transport record."""
    _require(archive.is_file() and not archive.is_symlink()
             and 0 < archive.stat().st_size <= MAX_CAPSULE_BYTES, "capsule_path_invalid")
    with zipfile.ZipFile(archive) as zipped:
        names = [info.filename for info in _safe_infos(zipped)]
        _require(_MARKER_NAME in names and _TRANSPORT_NAME in names, "capsule_marker_missing")
        marker = json.loads(zipped.read(_MARKER_NAME))
        transport = json.loads(zipped.read(_TRANSPORT_NAME))
    ids = marker.get("completed_stage_ids")
    _require(isinstance(marker, dict) and marker.get("schema_version") == MARKER_SCHEMA
             and marker.get("status") == "completed_prefix_only"
             and marker.get("whole_run_completed") is False
             and marker.get("qualification_authority_granted") is False
             and isinstance(ids, list) and bool(ids) and all(isinstance(i, str) and i.startswith("stage-") for i in ids)
             and len(set(ids)) == len(ids), "capsule_marker_invalid")
    _require(isinstance(transport, dict) and transport.get("schema_version") == TRANSPORT_SCHEMA
             and transport.get("completed_stage_ids") == ids
             and transport.get("transport_digest") == canonical_digest(transport, digest_field="transport_digest")
             and transport.get("gpu_execution_performed") is False
             and transport.get("new_paid_allocation_authorized") is False, "capsule_transport_invalid")
    if expected_stage_ids is not None:
        _require(list(ids) == list(expected_stage_ids), "capsule_stage_ids_mismatch")
    completed = set(ids)
    for name in names:
        parts = name.split("/")
        _require(parts[0] != "stages" or len(parts) <= 2 or parts[1] in completed, "capsule_member_outside_prefix")
    _require(f"stages/{_BINDING_NAME}" in names
             and all(f"stages/{stage_id}/{_MARKER_NAME}" in names for stage_id in ids), "capsule_prefix_incomplete")
    return {**marker, "transport": transport}


@contextmanager
def _exclusive_work_dir(work_dir: Path):
    """One prestage at a time: the bundle path under ``work_dir`` is fixed."""
    fd = os.open(work_dir / ".cpu-prestage.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CpuPrestageError("cpu_prestage_workspace_busy") from exc
        yield
    finally:
        os.close(fd)


def _clear_work_products(work_dir: Path) -> None:
    for name in (BUNDLE_DIRNAME, BUNDLE_DIRNAME + ".zip", CHECKPOINT_NAME,
                 "task_evaluation_scene_configuration_provider_output.zip"):
        target = work_dir / name
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            # The entrypoint seals its toolchain read-only. Make only our
            # disposable directories writable; never follow runtime symlinks.
            for directory, _, _ in os.walk(target, followlinks=False):
                Path(directory).chmod(0o700)
            shutil.rmtree(target)


def prepare_stage_prefix_before_gpu(
    *,
    bundle_receipt: Mapping[str, Any],
    authority: Mapping[str, Any],
    job_dir: str | Path,
    environment: Mapping[str, str],
    stage_limit: str,
    runner: Callable[..., Any] = subprocess.run,
    work_dir: str | Path | None = None,
    python_bin_dir: str | Path | None = None,
    now: Callable[[], float] = time.time,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    closure_reserve_seconds: int = DEFAULT_CLOSURE_RESERVE_SECONDS,
    reservation_root: str | Path | None = None,
    disk_usage: Callable[..., Any] = shutil.disk_usage,
    which: Callable[[str], str | None] = shutil.which,
) -> dict[str, Any]:
    """Run stages up to ``stage_limit`` on this host and archive the completed prefix.

    The bundle's own entrypoint executes at the paid run's logical paths under
    ``work_dir`` with the paid run's composed environment; only the execution
    site differs. A retained receipt for the same bundle, authority and limit
    is returned unchanged. Nothing here allocates a provider.
    """
    from .control_plane_disk_budget import DEFAULT_RESERVATION_ROOT, reserve_control_plane_disk

    job = Path(job_dir)
    receipt_path = job / "cpu_prestage_receipt.json"
    if receipt_path.exists():
        existing = _json(receipt_path)
        _require(existing.get("receipt_digest") == canonical_digest(existing, digest_field="receipt_digest")
                 and existing.get("bundle_sha256") == bundle_receipt["bundle_sha256"]
                 and existing.get("authority_digest") == authority["authority_digest"]
                 and existing.get("stage_limit") == stage_limit
                 and _sha(Path(existing["capsule_path"])) == existing["capsule_sha256"], "retained_receipt_invalid")
        return existing
    _require(not (job / "cpu_prestage_output.zip").exists(), "prior_attempt_requires_reconciliation")
    bundle = Path(str(bundle_receipt["bundle_path"]))
    _require(_sha(bundle) == bundle_receipt["bundle_sha256"], "source_bundle_changed")
    _bundle_manifest(bundle)
    work = Path(work_dir or environment.get(WORK_DIR_ENV) or DEFAULT_WORK_DIR)
    _require(work.is_absolute() and work.is_dir() and not work.is_symlink() and work.resolve() == work,
             "work_dir_unavailable")
    python_bin = Path(python_bin_dir) if python_bin_dir else Path(sys.executable).parent
    _require((python_bin / "python3").exists(), "python_runtime_missing")
    for name in ("bash", "timeout"):
        _require(which(name) is not None, f"host_command_missing:{name}")
    # Secrets are read from outside the work dir; the capsule never carries them.
    for name, value in environment.items():
        if name.endswith("API_KEY_FILE") and value:
            _require(not Path(str(value)).resolve().is_relative_to(work), "secret_inside_work_dir")
    with zipfile.ZipFile(bundle) as zipped:
        source_unpacked_bytes = sum(row.file_size for row in _safe_infos(zipped))
    peak_bytes = 3 * source_unpacked_bytes + 512 * 1024**2
    root = work / BUNDLE_DIRNAME
    with _exclusive_work_dir(work):
        with reserve_control_plane_disk("cpu_prestage", target_root=work, expected_bytes=peak_bytes,
                                        reservation_root=reservation_root or DEFAULT_RESERVATION_ROOT,
                                        disk_usage=disk_usage):
            # The exclusive lock proves no cooperating producer owns leftovers.
            _clear_work_products(work)
            _extract(bundle, root)
            runtime = root / "provider_runtime"
            output = root / "runtime_output"
            _require((root / ENTRYPOINT).is_file(), "entrypoint_missing")
            output.mkdir(mode=0o750)
            checkpoint_path = work / CHECKPOINT_NAME
            deadline = float(now()) + int(ttl_seconds)
            parent_deadline = environment.get("BLUEPRINT_SCENE_CONFIGURATION_PARENT_DEADLINE_EPOCH")
            if parent_deadline:
                parent_deadline = float(parent_deadline)
                _require(math.isfinite(parent_deadline), "parent_deadline_invalid")
                deadline = min(deadline, parent_deadline)
            _require(deadline > float(now()) + int(closure_reserve_seconds), "parent_deadline_exhausted")
            values = {
                **{str(k): str(v) for k, v in environment.items()},
                "BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS": "1",
                "BLUEPRINT_VAST_WORK_DIR": str(work),
                "BLUEPRINT_SCENE_CONFIGURATION_RUNTIME_ROOT": str(runtime),
                "BLUEPRINT_SCENE_CONFIGURATION_OUTPUT_ROOT": str(output),
                "BLUEPRINT_SCENE_CONFIGURATION_STAGE_CHECKPOINT_PATH": str(checkpoint_path),
                "BLUEPRINT_SCENE_CONFIGURATION_STAGE_LIMIT": stage_limit,
                "BLUEPRINT_SCENE_CONFIGURATION_PARENT_DEADLINE_EPOCH": repr(deadline),
                "BLUEPRINT_SCENE_CONFIGURATION_OUTPUT_CLOSURE_RESERVE_SECONDS": str(int(closure_reserve_seconds)),
                "PATH": str(python_bin) + ":" + str(environment.get("PATH") or "/usr/local/bin:/usr/bin:/bin"),
            }
            values.pop(PREFIX_URL_ENV, None)
            log_path = job / "cpu_prestage_entrypoint.log"
            try:
                with log_path.open("wb") as stream:
                    completed = runner(["bash", str(runtime / "run_task_evaluation_scene_configuration_provider.sh")],
                                       cwd=str(runtime), env=values, stdout=stream, stderr=subprocess.STDOUT,
                                       check=False)
                returncode = int(getattr(completed, "returncode", 1))
                result_path = output / RESULT_NAME
                _require(result_path.is_file(), "runner_result_missing")
                result = _json(result_path)
                shutil.copyfile(result_path, job / "cpu_prestage_provider_result.json")
                _require(returncode == 0 and result.get("status") == "completed_prefix"
                         and result.get("run_id") == bundle_receipt["run_id"]
                         and result.get("source_commit") == bundle_receipt["source_commit"],
                         "prefix_not_completed:" + str(result.get("status")))
                chain = result.get("stage_chain") or {}
                _require(chain.get("stage_limit") == stage_limit and chain.get("whole_run_completed") is False
                         and chain.get("status") == "completed_prefix", "prefix_binding_invalid")
                expected_ids = [str(row["stage_id"]) for row in chain.get("stage_results") or []]
                _require(bool(expected_ids) and expected_ids[-1] == stage_limit
                         and len(expected_ids) == int(chain.get("stage_count") or 0), "prefix_results_invalid")
                binding = _json(output / "stages" / _BINDING_NAME)
                _require(binding.get("schema_version") == "astra_split_stage_resume_binding.v1"
                         and binding.get("output_root") == str(output / "stages")
                         and binding.get("prefix_stage_limit") == stage_limit
                         and binding.get("parent_deadline_epoch") == deadline, "prefix_binding_invalid")
                capsule = job / "cpu_prestage_capsule.zip"
                _require(not capsule.exists() and checkpoint_path.is_file(), "checkpoint_missing")
                shutil.copyfile(checkpoint_path, capsule)
                transport = {
                    "schema_version": TRANSPORT_SCHEMA, "work_dir": str(work), "runtime_root": str(runtime),
                    "output_root": str(output), "run_id": result["run_id"], "source_commit": result["source_commit"],
                    "bundle_sha256": bundle_receipt["bundle_sha256"], "authority_digest": authority["authority_digest"],
                    "stage_limit": stage_limit, "completed_stage_ids": expected_ids,
                    "prefix_deadline_epoch": deadline, "prefix_binding_digest": binding.get("binding_digest"),
                    "execution_site": "control_plane", "gpu_execution_performed": False,
                    "new_paid_allocation_authorized": False,
                }
                transport["transport_digest"] = canonical_digest(transport, digest_field="transport_digest")
                with zipfile.ZipFile(capsule, "a", compression=zipfile.ZIP_DEFLATED) as zipped:
                    zipped.writestr(_TRANSPORT_NAME, canonical_json(transport) + "\n")
                validate_stage_prefix_capsule(capsule, expected_stage_ids=expected_ids)
            finally:
                # Keep failed-stage authoring and cost reservations as well as
                # successful checkpoints before removing the scratch runtime.
                # A crash after spending must not look like an untouched job.
                from .task_evaluation_scene_configuration_output_archive import write_output_archive
                write_output_archive(output, job / "cpu_prestage_output.zip")
                _clear_work_products(work)
    receipt = {
        "schema_version": RECEIPT_SCHEMA, "status": "completed_prefix_before_gpu_allocation",
        "bundle_sha256": bundle_receipt["bundle_sha256"], "authority_digest": authority["authority_digest"],
        "run_id": result["run_id"], "source_commit": result["source_commit"], "stage_limit": stage_limit,
        "completed_stage_ids": expected_ids, "work_dir": str(work),
        "capsule_path": str(capsule), "capsule_sha256": _sha(capsule), "capsule_bytes": capsule.stat().st_size,
        "transport_digest": transport["transport_digest"], "provider_result_digest": result.get("result_digest"),
        "entrypoint_returncode": returncode, "entrypoint_log_path": str(log_path),
        "prefix_deadline_epoch": deadline, "source_unpacked_bytes": source_unpacked_bytes,
        "reserved_peak_bytes": peak_bytes, "gpu_execution_performed": False,
        "provider_mutations_performed": 0, "execution_site": "control_plane",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path.write_text(canonical_json(receipt) + "\n")
    return receipt


def consume_stage_prefix_capsule(*, environment: Mapping[str, str], output_root: str | Path,
                                 expected_run_id: str, opener: Callable[..., Any] | None = None) -> dict[str, Any] | None:
    """Restore a control-plane prefix at its exact paths before the paid run's chain."""
    url = str(environment.get(PREFIX_URL_ENV) or "")
    if not url:
        return None
    _require(not str(environment.get("BLUEPRINT_SCENE_CONFIGURATION_STAGE_LIMIT") or ""), "capsule_recursion")
    expected = str(environment.get(PREFIX_SHA_ENV) or "")
    expected_bytes = int(environment.get(PREFIX_BYTES_ENV) or 0)
    _require(expected.startswith("sha256:") and len(expected) == 71 and 0 < expected_bytes <= MAX_CAPSULE_BYTES
             and (url.startswith("https://") or opener is not None), "transport_invalid")
    output = Path(output_root)
    _require(output.is_dir() and not output.is_symlink() and output.resolve() == output, "output_root_invalid")
    stages = output / "stages"
    _require(not stages.exists() or (stages.is_dir() and not any(stages.iterdir())), "output_root_not_empty")
    if opener is None:
        import urllib.request
        opener = urllib.request.urlopen
    with tempfile.TemporaryDirectory(prefix="cpu-prestage-") as temporary:
        archive = Path(temporary) / "capsule.zip"
        with opener(url, timeout=120) as response, archive.open("wb") as target:
            remaining = expected_bytes
            while remaining:
                block = response.read(min(1024**2, remaining))
                _require(bool(block), "download_truncated")
                target.write(block)
                remaining -= len(block)
            _require(not response.read(1), "download_exceeds_binding")
        _require(_sha(archive) == expected, "archive_digest_invalid")
        marker = validate_stage_prefix_capsule(archive)
        transport = marker["transport"]
        _require(transport["output_root"] == str(output) and transport["run_id"] == expected_run_id,
                 "capsule_bound_to_other_run_or_path")
        restored = _extract(archive, output, allowed=lambda name: name.startswith("stages/"))
    value = {"schema_version": RESTORE_SCHEMA, "status": "restored",
             "completed_stage_ids": list(marker["completed_stage_ids"]), "capsule_sha256": expected,
             "capsule_bytes": expected_bytes, "restored_member_count": len(restored),
             "transport_digest": transport["transport_digest"], "execution_site": "control_plane"}
    value["restore_digest"] = canonical_digest(value, digest_field="restore_digest")
    (output / "cpu_prestage_restore.json").write_text(canonical_json(value) + "\n")
    print("BLUEPRINT_SCENE_CONFIGURATION_STAGE_PREFIX_RESTORED:" + json.dumps(
        {"completed_stage_ids": value["completed_stage_ids"], "capsule_sha256": expected}, sort_keys=True), flush=True)
    return value


__all__ = [
    "BUNDLE_DIRNAME", "CPU_ADAPTERS", "CpuPrestageError", "DEFAULT_WORK_DIR", "PREFIX_BYTES_ENV",
    "PREFIX_SHA_ENV", "PREFIX_URL_ENV", "RECEIPT_SCHEMA", "STAGE_LIMIT_ENV", "WORK_DIR_ENV",
    "consume_stage_prefix_capsule", "prepare_stage_prefix_before_gpu", "prestage_stage_limit",
    "prestage_ttl_seconds", "validate_stage_prefix_capsule",
]
