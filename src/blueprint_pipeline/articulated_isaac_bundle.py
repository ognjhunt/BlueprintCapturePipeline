"""Package the frozen articulated probe for a provider Isaac run.

The existing SimReady Isaac lane was built for a rigid can: its required files
are drop, slide, tip and gripper stimulus stages, none of which mean anything
for a hinged door. Rather than fake those filenames, this builds the
articulated equivalent - the blank physics stage, the articulation stage, the
exact candidate bytes and the frozen probe spec - alongside a worker that
performs the eleven preregistered readbacks.

The bundle refuses a probe that is not still frozen, or whose stage bytes no
longer match the digests the spec recorded. That matters because the whole
value of the run is that its expectations were fixed before execution; a probe
edited after freezing would prove nothing.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Sequence

from .common import ensure_dir, write_json
from .decision_evidence_contracts import canonical_digest
# Reuse the digest the SimReady Isaac lane already pins, so the allocator's
# image check and this bundle cannot drift apart.
from .public_scene_simready_isaac_bundle import DEFAULT_IMAGE


ARTICULATED_ISAAC_BUNDLE_SCHEMA_VERSION = "adp009d_articulated_isaac_bundle.v1"
PROBE_SPEC_FILENAME = "articulated_native_probe_spec.json"
PROBE_SPEC_SCHEMA_VERSION = "articulated_native_probe_spec.v1"
PRIMARY_STAGE_NAME = "articulation_stage"
ENTRYPOINT = r'''#!/usr/bin/env bash
set +e
BUNDLE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${BLUEPRINT_ISAAC_OUTPUT_DIR:-$BUNDLE_DIR/runtime_output}"
RESULT="$OUTPUT_DIR/isaac_runtime_result.json"
mkdir -p "$OUTPUT_DIR"
/isaac-sim/python.sh "$BUNDLE_DIR/provider_runtime/isaac_realistic_runtime_runner.py" \
  --spec "$BUNDLE_DIR/provider_runtime/native/__PROBE_SPEC_FILENAME__" \
  --output "$RESULT"
runner_rc=$?
write_missing_result() {
  if [ -s "$RESULT" ]; then return 0; fi
  /isaac-sim/python.sh - "$RESULT" "$runner_rc" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": "adp009d_articulated_isaac_result.v1",
    "status": "blocked_isaac_process_exited_without_result",
    "blockers": [f"isaac_runner_process_exited_without_runtime_result:{sys.argv[2]}"],
    "native_isaac_executed": False,
    "articulation_qualified": False,
    "physical_success_established": False,
    "provider_zero_required_after_return": True
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}
write_missing_result
exit 0
'''


class ArticulatedIsaacBundleError(ValueError):
    """Stable, sorted articulated Isaac bundle failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def build_articulated_isaac_bundle(
    *,
    probe_root: str | Path,
    job_dir: str | Path,
    worker_source: str | Path,
    source_commit_sha: str,
    container_image: str = DEFAULT_IMAGE,
    generated_at: str | None = None,
    probe_spec_filename: str = PROBE_SPEC_FILENAME,
    probe_spec_schema_version: str = PROBE_SPEC_SCHEMA_VERSION,
    primary_stage_name: str = PRIMARY_STAGE_NAME,
    extra_native_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Zip the frozen probe plus the articulated worker into a provider bundle."""

    root = Path(probe_root).expanduser().resolve()
    job = Path(job_dir).expanduser().resolve()
    worker = Path(worker_source).expanduser().resolve()
    errors: list[str] = []
    if not root.is_dir():
        errors.append("articulated_isaac_bundle_probe_root_missing")
    if not worker.is_file():
        errors.append("articulated_isaac_bundle_worker_missing")
    if len(str(source_commit_sha)) != 40:
        errors.append("articulated_isaac_bundle_source_commit_invalid")
    if errors:
        raise ArticulatedIsaacBundleError(errors)

    # Probe kinds multiply; the transport does not. Forking this builder per
    # kind would fork the slot layout, the image pin and the digest checks with
    # it, and those are the parts whose drift costs a launch to discover.
    # The base Isaac image carries isaacsim but not isaaclab or Arena. A worker
    # that needs them boots, spends four minutes bringing Isaac up, and dies on
    # its first import - and nothing before that can tell, because the bundle is
    # well-formed and the dry run is clean. Refuse the pairing at build time.
    worker_source_text = worker.read_text(encoding="utf-8", errors="ignore")
    requires_arena = any(
        token in worker_source_text
        for token in ("isaaclab_arena", "import isaaclab", "from isaaclab")
    )
    if requires_arena and container_image == DEFAULT_IMAGE:
        raise ArticulatedIsaacBundleError(
            [
                "articulated_isaac_bundle_worker_needs_arena_image:"
                "worker imports isaaclab but the bare isaac-sim image does not "
                "provide it; use the Arena-provisioned lane image"
            ]
        )

    spec_path = root / str(probe_spec_filename)
    if not spec_path.is_file():
        raise ArticulatedIsaacBundleError(["articulated_isaac_bundle_probe_spec_missing"])
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ArticulatedIsaacBundleError(
            ["articulated_isaac_bundle_probe_spec_invalid"]
        ) from exc
    if spec.get("status") != "frozen_not_executed":
        errors.append("articulated_isaac_bundle_probe_not_frozen")
    if spec.get("schema_version") != str(probe_spec_schema_version):
        errors.append("articulated_isaac_bundle_probe_schema_invalid")

    stages = spec.get("stages") or {}
    staged: dict[str, Path] = {}
    for name, row in stages.items():
        candidate = root / Path(str(row.get("path") or "")).name
        if not candidate.is_file():
            errors.append(f"articulated_isaac_bundle_stage_missing:{name}")
            continue
        if _sha256(candidate) != row.get("sha256"):
            errors.append(f"articulated_isaac_bundle_stage_digest_mismatch:{name}")
            continue
        staged[name] = candidate
    if errors:
        raise ArticulatedIsaacBundleError(errors)

    if job.exists():
        shutil.rmtree(job)
    runtime = job / "provider_runtime"
    native = runtime / "native"
    ensure_dir(native)
    for path in staged.values():
        shutil.copy2(path, native / path.name)
    shutil.copy2(spec_path, native / spec_path.name)
    # The stage list is digest-pinned USD; a worker that imports repo modules
    # needs a different kind of payload alongside it, and leaving it out means
    # the runtime boots and then dies on an import.
    extra_missing = [
        str(item)
        for item in extra_native_paths
        if not Path(str(item)).expanduser().resolve().is_file()
    ]
    if extra_missing:
        raise ArticulatedIsaacBundleError(
            [f"articulated_isaac_bundle_extra_native_path_missing:{p}" for p in extra_missing]
        )
    for item in extra_native_paths:
        source_file = Path(str(item)).expanduser().resolve()
        shutil.copy2(source_file, native / source_file.name)
    shutil.copy2(worker, runtime / "isaac_realistic_runtime_runner.py")
    entrypoint = runtime / "run_isaac_realistic_runtime.sh"
    # The runner is told which spec to open. Leaving one kind's filename baked
    # in here boots Isaac against a path that does not exist, and the launch is
    # spent on a result that says nothing about the probe - while the bundle
    # stays well-formed and the dry run stays clean.
    entrypoint.write_text(
        ENTRYPOINT.replace("__PROBE_SPEC_FILENAME__", spec_path.name),
        encoding="utf-8",
    )
    entrypoint.chmod(0o755)

    # The Isaac lane's transport declares a fixed set of slots. Fill them with
    # the articulated equivalents rather than inventing a parallel layout: the
    # articulation stage is the scene the runtime opens, and the three
    # manifests carry the same placeholder contract the rigid probe uses.
    articulation = native / Path(
        str((stages.get(str(primary_stage_name)) or {}).get("path") or "")
    ).name
    for name in ("generated_site_scene.usda", "generated_site_scene.usd"):
        shutil.copy2(articulation, runtime / name)
    common = {
        "schema_version": "adp009b_simready_isaac_placeholder_contract.v1",
        "source_commit_sha": str(source_commit_sha),
        "probe_spec_sha256": _sha256(spec_path),
        "status": "bounded_articulated_readback_probe",
    }
    if generated_at is not None:
        common["generated_at"] = generated_at
    for name in (
        "scenario_eval_matrix.json",
        "camera_manifest.json",
        "episode_spec_manifest.json",
    ):
        write_json(runtime / name, {**common, "artifact": name.removesuffix(".json")})
    write_json(
        runtime / "isaac_provider_eval_manifest.json",
        {
            "schema_version": "isaac_provider_eval_manifest.v1",
            "job_id": "adp009d-articulated-native-readback",
            "relative_paths": {
                "generated_site_scene_usda": "generated_site_scene.usda",
                "generated_site_scene_usd": "generated_site_scene.usd",
                "scenario_eval_matrix": "scenario_eval_matrix.json",
                "camera_manifest": "camera_manifest.json",
                "episode_spec_manifest": "episode_spec_manifest.json",
                "runtime_runner": "isaac_realistic_runtime_runner.py",
                "entrypoint": "run_isaac_realistic_runtime.sh",
                "probe_spec": f"native/{spec_path.name}",
            },
            "proof_boundaries": {
                "native_articulation_readback_only": True,
                "articulation_qualification_is_not_task_success": True,
                "physical_success_established": False,
            },
        },
    )

    bundle_path = job / "adp009d_articulated_isaac_provider_bundle.zip"
    members = sorted(
        (path for path in runtime.rglob("*") if path.is_file()),
        key=lambda path: str(path.relative_to(job)),
    )
    with zipfile.ZipFile(
        bundle_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in members:
            info = zipfile.ZipInfo(str(path.relative_to(job)))
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.external_attr = (0o755 if path.suffix == ".sh" else 0o644) << 16
            archive.writestr(info, path.read_bytes())

    expected = spec.get("expected") or {}
    receipt: dict[str, Any] = {
        "schema_version": ARTICULATED_ISAAC_BUNDLE_SCHEMA_VERSION,
        "status": "ready",
        "blockers": [],
        "retry_cap": 0,
        "source_commit_sha": str(source_commit_sha),
        "container_image": container_image,
        "bundle_path": str(bundle_path),
        "bundle_sha256": _sha256(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
        "probe_spec_sha256": _sha256(spec_path),
        "probe_receipt_digest": spec.get("receipt_digest"),
        "candidate_usd_sha256": spec.get("candidate_usd_sha256"),
        "expected": expected,
        "required_readbacks": list(spec.get("required_readbacks") or []),
        # The lane validates the returned probe set against what the bundle
        # declares, so an articulated run is checked as strictly as a rigid one
        # without inheriting drop/slide/tip/gripper semantics.
        "probe_names": sorted(str(name) for name in (spec.get("required_readbacks") or [])),
        "extra_native_file_count": len(list(extra_native_paths)),
        "worker_requires_arena": requires_arena,
        "result_relative_path": "runtime_output/isaac_runtime_result.json",
        "relative_paths": {
            "entrypoint": "provider_runtime/run_isaac_realistic_runtime.sh",
            "worker": "provider_runtime/isaac_realistic_runtime_runner.py",
            "probe_spec": f"provider_runtime/native/{spec_path.name}",
        },
        "proof_boundaries": {
            "native_readback_only": True,
            "articulation_qualification_is_not_task_success": True,
            "physical_success_established": False,
        },
        "receipt_digest": "",
    }
    if generated_at is not None:
        receipt["generated_at"] = generated_at
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    write_json(job / "adp009d_articulated_isaac_bundle_receipt.json", receipt)
    return json.loads(json.dumps(receipt))


__all__ = [
    "ARTICULATED_ISAAC_BUNDLE_SCHEMA_VERSION",
    "ArticulatedIsaacBundleError",
    "build_articulated_isaac_bundle",
]
