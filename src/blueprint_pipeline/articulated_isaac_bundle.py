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
ENTRYPOINT = r'''#!/usr/bin/env bash
set +e
BUNDLE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${BLUEPRINT_ISAAC_OUTPUT_DIR:-$BUNDLE_DIR/runtime_output}"
RESULT="$OUTPUT_DIR/articulated_isaac_result.json"
mkdir -p "$OUTPUT_DIR"
/isaac-sim/python.sh "$BUNDLE_DIR/provider_runtime/articulated_isaac_worker.py" \
  --spec "$BUNDLE_DIR/provider_runtime/native/articulated_native_probe_spec.json" \
  --output "$RESULT"
runner_rc=$?
if [ ! -s "$RESULT" ]; then
  /isaac-sim/python.sh - "$RESULT" "$runner_rc" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": "adp009d_articulated_isaac_result.v1",
    "status": "blocked",
    "blockers": [f"isaac_runner_process_exited_without_runtime_result:{sys.argv[2]}"],
    "native_isaac_executed": False,
    "articulation_qualified": False,
    "physical_success_established": False,
    "provider_zero_required_after_return": True
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
fi
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

    spec_path = root / PROBE_SPEC_FILENAME
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
    if spec.get("schema_version") != "articulated_native_probe_spec.v1":
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
    shutil.copy2(spec_path, native / PROBE_SPEC_FILENAME)
    shutil.copy2(worker, runtime / "articulated_isaac_worker.py")
    entrypoint = runtime / "run_articulated_isaac_runtime.sh"
    entrypoint.write_text(ENTRYPOINT, encoding="utf-8")
    entrypoint.chmod(0o755)

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
        "relative_paths": {
            "entrypoint": "provider_runtime/run_articulated_isaac_runtime.sh",
            "worker": "provider_runtime/articulated_isaac_worker.py",
            "probe_spec": f"provider_runtime/native/{PROBE_SPEC_FILENAME}",
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
