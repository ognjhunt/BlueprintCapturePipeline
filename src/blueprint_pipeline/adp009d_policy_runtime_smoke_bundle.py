"""Build a provider bundle for an outcome-blind policy readiness handshake."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .adp009d_native_microcheck_bundle import DEFAULT_IMAGE
from .adp009d_policy_candidate_admission import EXPECTED_CANDIDATES
from .adp009d_policy_provisioning import build_provisioning_script
from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .openpi_droid_policy_runtime import load_policy_spec


PROBE_KIND = "adp009d-policy-runtime-smoke"
SCHEMA_VERSION = "adp009d_policy_runtime_smoke_bundle.v2"
RESULT_FILENAME = "adp009d_native_microcheck.json"
ALLOWED_CANDIDATES = frozenset({"pi05_droid", "groot_n17_droid"})
PROVISION_TIMEOUT_SECONDS = 2_700

ENTRYPOINT = r'''#!/usr/bin/env bash
set +e
RUNTIME_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${BLUEPRINT_ADP_ARENA_OUTPUT_DIR:-$RUNTIME_DIR/../runtime_output}"
candidate="@@CANDIDATE@@"
mkdir -p "$OUT_DIR"

echo "BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:runtime_smoke_${candidate}:started"
RUNTIME_DIR="$RUNTIME_DIR" OUT_DIR="$OUT_DIR" \
  setsid bash "$RUNTIME_DIR/adp009d_policy_provisioning.$candidate.sh" \
  >"$OUT_DIR/adp009d_policy_provisioning.$candidate.log" 2>&1 &
provisioning_pid=$!
waited=0
while kill -0 "$provisioning_pid" 2>/dev/null; do
  sleep 60
  waited=$((waited + 60))
  echo "BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:runtime_smoke_${candidate}_working:${waited}s"
  if [ "$waited" -ge @@PROVISION_TIMEOUT@@ ]; then
    echo "BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:runtime_smoke_${candidate}:timed_out"
    kill -TERM -- "-$provisioning_pid" 2>/dev/null || true
    sleep 10
    kill -KILL -- "-$provisioning_pid" 2>/dev/null || true
    break
  fi
done
wait "$provisioning_pid" 2>/dev/null
provisioning_rc=$?

/isaac-sim/python.sh "$RUNTIME_DIR/adp009d_policy_runtime_smoke_worker.py" \
  --candidate-id "$candidate" \
  --server-receipt "$OUT_DIR/adp009d_policy_server_receipt.$candidate.json" \
  --provisioning-exit-code "$provisioning_rc" \
  --output "$OUT_DIR/adp009d_native_microcheck.json"
worker_rc=$?
echo "BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:runtime_smoke_${candidate}:completed:rc=$worker_rc"
exit "$worker_rc"
'''


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _stage_openpi_identity(runtime: Path, repo_root: Path) -> dict[str, Any]:
    experiment = repo_root / "docs/experiments/policy_ranking_thesis_20260726"
    cohort = experiment / "warehouse_policy_cohort_v2_joint_position.json"
    inventory = experiment / "openpi_polaris_checkpoint_inventory.json"
    spec = load_policy_spec(cohort, policy_id="pi05_droid_jointpos_polaris")
    execution_spec = {
        "schema_version": "native_task_arena_policy_execution_spec.v1",
        "candidate_id": "pi05_droid",
        "purpose": "outcome_blind_policy_runtime_smoke",
        "policy_spec": asdict(spec),
    }
    execution_spec["execution_spec_digest"] = canonical_digest(
        execution_spec, digest_field="execution_spec_digest"
    )
    write_json(runtime / "adp009d_policy_execution_spec.json", execution_spec)
    shutil.copy2(inventory, runtime / "adp009d_openpi_checkpoint_inventory.json")
    return {
        "execution_spec_digest": execution_spec["execution_spec_digest"],
        "checkpoint_inventory_sha256": _file_sha256(inventory),
    }


def build_policy_runtime_smoke_bundle(
    *,
    job_dir: str | Path,
    candidate_id: str,
    implementation_commit: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build immutable bytes for one zero-inference server handshake."""

    if candidate_id not in ALLOWED_CANDIDATES or candidate_id not in EXPECTED_CANDIDATES:
        raise ValueError("policy_runtime_smoke_candidate_invalid")
    if len(implementation_commit) != 40 or any(
        character not in "0123456789abcdef" for character in implementation_commit
    ):
        raise ValueError("policy_runtime_smoke_implementation_commit_invalid")
    job = Path(job_dir).expanduser().resolve()
    if job.exists():
        shutil.rmtree(job)
    runtime = job / "provider_runtime"
    ensure_dir(runtime)
    source = Path(__file__).resolve().parent
    for name in (
        "adp009d_checkpoint_fetch_worker.py",
        "adp009d_provisioning_preflight.py",
        "adp009d_policy_server_worker.py",
        "adp009d_policy_runtime_smoke_worker.py",
        "adp009d_groot_worker_identity.py",
        "adp009d_gated_backbone.py",
        "groot_n17_droid_policy_runtime.py",
        "openpi_droid_policy_runtime.py",
        "droid_policy_bridge.py",
        "decision_evidence_contracts.py",
    ):
        shutil.copy2(source / name, runtime / name)
    identity_inputs: dict[str, Any] = {}
    if candidate_id == "pi05_droid":
        identity_inputs = _stage_openpi_identity(runtime, source.parents[1])
    _write_executable(
        runtime / f"adp009d_policy_provisioning.{candidate_id}.sh",
        build_provisioning_script(candidate_id, stop_after_handshake=True),
    )
    _write_executable(
        runtime / "run_adp_arena_provider_runtime.sh",
        ENTRYPOINT.replace("@@CANDIDATE@@", candidate_id).replace(
            "@@PROVISION_TIMEOUT@@", str(PROVISION_TIMEOUT_SECONDS)
        ),
    )
    expected = EXPECTED_CANDIDATES[candidate_id]
    generated = generated_at or utc_now_iso()
    binding = {
        "candidate_id": candidate_id,
        "source_repository": expected["source_repository"],
        "source_revision": expected["source_revision"],
        "checkpoint_repository": expected["checkpoint_repository"],
        "checkpoint_revision": expected["checkpoint_revision"],
        "checkpoint_inventory_digest": expected["checkpoint_inventory_digest"],
        "synthetic_query_count": 0,
        "readiness_method": "identity_bound_transport_handshake_without_inference",
        "actions_executed": False,
        "task_scene_loaded": False,
        **identity_inputs,
    }
    input_digest = canonical_digest(binding)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated,
        "status": "ready",
        "probe_kind": PROBE_KIND,
        "execution_mode": "outcome_blind_policy_runtime_smoke",
        "implementation_commit": implementation_commit,
        "policy_candidate_id": candidate_id,
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": False,
        "controls_requested": False,
        "input_digest": input_digest,
        "protocol_digest": input_digest,
        "identity_binding": binding,
        "container_image": DEFAULT_IMAGE,
        "runtime_entrypoint": "provider_runtime/run_adp_arena_provider_runtime.sh",
        "expected_output_filename": RESULT_FILENAME,
        "provider_zero_required_after_return": True,
        "retry_cap": 0,
        "claim_ceiling": "development_only_runtime_smoke",
        "blockers": [],
        "raw_secret_values_recorded": False,
    }
    write_json(runtime / "adp009d_policy_runtime_smoke_manifest.json", manifest)
    bundle_path = job / "adp009d_policy_runtime_smoke_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w") as archive:
        for path in sorted(runtime.rglob("*")):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(
                path.relative_to(job).as_posix(), date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.create_system = 3
            info.external_attr = (path.stat().st_mode & 0xFFFF) << 16
            archive.writestr(
                info,
                path.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    receipt = {
        **manifest,
        "bundle_path": str(bundle_path),
        "bundle_sha256": _file_sha256(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
    }
    write_json(job / "adp009d_policy_runtime_smoke_bundle_receipt.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Build one commit-bound runtime-smoke bundle without provider mutation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--candidate-id", required=True, choices=sorted(ALLOWED_CANDIDATES))
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)
    try:
        receipt = build_policy_runtime_smoke_bundle(
            job_dir=args.job_dir,
            candidate_id=args.candidate_id,
            implementation_commit=args.implementation_commit,
            generated_at=args.generated_at,
        )
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [f"{type(exc).__name__}:{exc}"],
                    "provider_mutation_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt.get("status") == "ready" else 2


__all__ = ["PROBE_KIND", "build_policy_runtime_smoke_bundle"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
