"""Seal one G1 development campaign for the paid controller.

The two scene packets and public publisher checkout travel in this bundle.
The much larger, verified Isaac runtime source packet remains a content-addressed
external dependency fetched by the Vast adapter after allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_g1_development_campaign import plan_g1_development_campaign
from .native_g1_development_pair import PAIR_ORDER
from .native_g1_provider_runtime import RESULT_FILENAME
from .native_g1_publisher_source_stage import verify_g1_publisher_source
from .native_task_arena_bundle import _write_zip_file, verify_native_task_arena_packet
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE
from .native_task_runtime_source_packet import verify_native_task_runtime_source_packet


SCHEMA = "native_g1_provider_bundle.v1"
PROVIDER_BUNDLE_KIND = "native_g1_development_campaign"
MANIFEST = "provider_runtime/native_g1_provider_manifest.json"
ENTRYPOINT = "provider_runtime/run_adp_arena_provider_runtime.sh"
BUNDLE_NAME = "native_g1_provider_bundle.zip"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _files(root: Path) -> list[Path]:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("g1_provider_bundle_source_tree_invalid")
    rows: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("g1_provider_bundle_source_symlink:" + str(path))
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("g1_provider_bundle_source_file_invalid:" + str(path))
        rows.append(path)
    return rows


def _entrypoint() -> str:
    return '''#!/usr/bin/env bash
set -u
RUNTIME_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${BLUEPRINT_ADP_ARENA_OUTPUT_DIR:-$RUNTIME_DIR/../runtime_output}"
mkdir -p "$OUT_DIR"
export BLUEPRINT_G1_PINNED_ISAAC_IMAGE="nvcr.io/nvidia/isaac-sim:6.0.1@sha256:b1c542b2ecc549b3d1ebb78c25664aa3bacba1709e6ad8e0a68e09426d57dedb"
cd "$RUNTIME_DIR"
echo BLUEPRINT_G1_STAGE_STARTED:runtime-source-provisioning
/isaac-sim/python.sh -m blueprint_pipeline.native_task_runtime_source_provision \\
  --source-receipt "$RUNTIME_DIR/native_task_runtime_sources/native_task_runtime_source_packet.v1.json" \\
  --source-packet "$RUNTIME_DIR/native_task_runtime_sources/native_task_runtime_sources.zip" \\
  --extraction-dir "$RUNTIME_DIR/provisioned_runtime_sources" \\
  --output "$OUT_DIR/native_task_runtime_source_provisioning.v1.json" \\
  --simulator-root /isaac-sim
provision_rc=$?
if [ "$provision_rc" -eq 0 ]; then
  echo BLUEPRINT_G1_STAGE_FINISHED:runtime-source-provisioning
  /isaac-sim/python.sh -m blueprint_pipeline.native_g1_provider_runtime \\
    --runtime-root "$RUNTIME_DIR" --output-dir "$OUT_DIR"
  runner_rc=$?
else
  runner_rc="$provision_rc"
fi
if [ ! -f "$OUT_DIR/native_g1_provider_campaign_result.v1.json" ]; then
  /isaac-sim/python.sh - "$OUT_DIR" "$runner_rc" <<'PY'
import json
import sys
from pathlib import Path
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
out = Path(sys.argv[1])
value = {
    "schema_version": "native_g1_provider_campaign_result.v1",
    "status": "blocked",
    "claim_ceiling": "development_only",
    "pairs": [],
    "policy_query_counts": {},
    "candidate_policy_queried": False,
    "ranking_eligible": False,
    "physical_outcome_claimed": False,
    "stage_reached": "runtime-source-provisioning" if int(sys.argv[2]) else "runner-start",
    "blockers": ["g1_provider_runner_exited_without_terminal_result"],
}
value["result_digest"] = canonical_digest(value, digest_field="result_digest")
(out / "native_g1_provider_campaign_result.v1.json").write_text(
    json.dumps(value, sort_keys=True, indent=2) + "\\n", encoding="utf-8"
)
PY
  runner_rc=2
fi
exit "$runner_rc"
'''


def build_g1_provider_bundle(
    *,
    job_dir: Path,
    manipulation_packet: Path,
    movement_packet: Path,
    book_handoff: Path,
    rights_review_paths: Mapping[str, Path],
    navigation_authority: Path,
    publisher_source: Path,
    runtime_source_receipt: Path,
    implementation_commit: str,
) -> dict[str, Any]:
    """Verify all inputs, then write one immutable controller bundle and receipt."""

    if re.fullmatch(r"[0-9a-f]{40}", implementation_commit) is None:
        raise ValueError("g1_provider_bundle_implementation_commit_invalid")
    package = Path(__file__).resolve().parent
    repository = package.parents[1]
    inventory = repository / "configs/g1_humanoidarena_checkpoint_inventory.v1.json"
    sonic_inventory = repository / "configs/g1_sonic_default_asset_inventory.v1.json"
    lock = repository / "configs/g1_humanoidarena_lerobot_pi_py312_linux_x86_64.requirements.txt"
    source = Path(publisher_source)
    if set(rights_review_paths) != set(PAIR_ORDER):
        raise ValueError("g1_provider_bundle_rights_set_invalid")
    campaign = plan_g1_development_campaign(
        book_handoff_path=book_handoff,
        manipulation_packet=manipulation_packet,
        movement_packet=movement_packet,
        inventory_path=inventory,
        rights_review_paths=rights_review_paths,
        navigation_authority_path=navigation_authority,
    )
    publisher = verify_g1_publisher_source(source)
    staged_publisher = json.loads(
        (source / "native_g1_publisher_source_stage.v1.json").read_text(encoding="utf-8")
    )
    if (
        staged_publisher.get("receipt_digest")
        != canonical_digest(staged_publisher, digest_field="receipt_digest")
        or any(
            staged_publisher.get(field) != publisher.get(field)
            for field in (
                "source_repository", "source_revision", "inventory_file_sha256",
                "policy_server_sha256", "sonic_provider_sha256",
                "lerobot_pyproject_sha256", "model_weights_included", "gpu_allocated",
            )
        )
    ):
        raise ValueError("g1_provider_bundle_publisher_receipt_invalid")
    runtime_source = verify_native_task_runtime_source_packet(runtime_source_receipt)
    packets = {
        "manipulation": verify_native_task_arena_packet(manipulation_packet),
        "movement": verify_native_task_arena_packet(movement_packet),
    }
    if not runtime_source.get("redistribution_permitted"):
        raise ValueError("g1_provider_bundle_runtime_source_rights_invalid")
    job = Path(job_dir)
    if not job.is_absolute() or job.is_symlink() or job.exists() or not job.parent.is_dir():
        raise ValueError("g1_provider_bundle_job_path_invalid")
    source_files = _files(source / "source")
    module_files = sorted(path for path in package.glob("*.py") if path.is_file())
    fetchers = [
        repository / "scripts/fetch_g1_humanoidarena_checkpoint.py",
        repository / "scripts/fetch_g1_sonic_assets.py",
    ]
    for path in [inventory, sonic_inventory, lock, *fetchers, *rights_review_paths.values()]:
        if path.is_symlink() or not path.is_file():
            raise ValueError("g1_provider_bundle_input_file_invalid:" + str(path))
    manifest = {
        "schema_version": SCHEMA,
        "status": "ready",
        "provider_bundle_kind": PROVIDER_BUNDLE_KIND,
        "implementation_commit": implementation_commit,
        "container_image": NATIVE_TASK_ARENA_IMAGE,
        "scene_id": campaign["scene_id"],
        "task_id": campaign["task_id"],
        "campaign_plan_digest": campaign["plan_digest"],
        "publisher_source_receipt_digest": staged_publisher["receipt_digest"],
        "runtime_source_packet": {
            "receipt_digest": runtime_source["receipt_digest"],
            "packet_sha256": runtime_source["packet_sha256"],
            "packet_size_bytes": runtime_source["packet_size_bytes"],
            "packet_path": runtime_source["verified_packet_path"],
            "transport": "content_addressed_external_layer.v1",
            "embedded_in_provider_bundle": False,
        },
        "packet_receipt_digests": {
            name: packet[1]["receipt_digest"] for name, packet in packets.items()
        },
        "candidate_ids": list(PAIR_ORDER),
        "expected_output_filename": RESULT_FILENAME,
        "runtime_entrypoint": ENTRYPOINT,
        "claim_ceiling": "development_only",
        "provider_zero_required_after_return": True,
    }
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    job.mkdir()
    bundle = job / BUNDLE_NAME
    with zipfile.ZipFile(bundle, "w", allowZip64=True) as archive:
        for name, (root, _receipt, rows) in packets.items():
            for row in rows:
                relative = row["relative_path"]
                _write_zip_file(
                    archive, source=root / relative,
                    archive_path=f"provider_runtime/inputs/scene_packets/{name}/{relative}",
                )
        # A shallow Git checkout has empty refs/ and objects/info/ directories.
        # Git rejects an extracted checkout if those directories disappear.
        for path in sorted((source / "source").rglob("*")):
            if path.is_dir():
                archive.writestr(
                    "provider_runtime/publisher-source/source/"
                    + path.relative_to(source / "source").as_posix()
                    + "/",
                    b"",
                )
        for path in source_files:
            relative = path.relative_to(source / "source").as_posix()
            if relative == ".git/config":
                # ZIP extraction may not preserve executable bits. Git must
                # compare content, not host-specific extraction file modes.
                config = path.read_text(encoding="utf-8")
                config += "\n[core]\n\tfilemode = false\n"
                archive.writestr("provider_runtime/publisher-source/source/.git/config", config)
                continue
            _write_zip_file(
                archive, source=path,
                archive_path="provider_runtime/publisher-source/source/" + relative,
            )
        for path in module_files:
            _write_zip_file(
                archive, source=path,
                archive_path="provider_runtime/blueprint_pipeline/" + path.name,
            )
        for path in fetchers:
            _write_zip_file(archive, source=path, archive_path="provider_runtime/scripts/" + path.name)
        for path in (inventory, sonic_inventory, lock):
            _write_zip_file(archive, source=path, archive_path="configs/" + path.name)
        for candidate in PAIR_ORDER:
            _write_zip_file(
                archive, source=rights_review_paths[candidate],
                archive_path="provider_runtime/inputs/rights/" + candidate + ".json",
            )
        for path, relative in (
            (book_handoff, "provider_runtime/inputs/book_handoff.json"),
            (navigation_authority, "provider_runtime/inputs/navigation_authority.json"),
            (source / "native_g1_publisher_source_stage.v1.json", "provider_runtime/publisher-source/native_g1_publisher_source_stage.v1.json"),
            (runtime_source_receipt, "provider_runtime/native_task_runtime_sources/native_task_runtime_source_packet.v1.json"),
        ):
            _write_zip_file(archive, source=path, archive_path=relative)
        archive.writestr(
            "provider_runtime/inputs/native_g1_development_campaign.v1.json",
            json.dumps(campaign, indent=2, sort_keys=True) + "\n",
        )
        archive.writestr(MANIFEST, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        script = _entrypoint()
        info = zipfile.ZipInfo(ENTRYPOINT, date_time=(1980, 1, 1, 0, 0, 0))
        info.create_system = 3
        info.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(info, script)
    receipt = {
        **manifest,
        "bundle_path": str(bundle),
        "bundle_size_bytes": bundle.stat().st_size,
        "bundle_sha256": _sha256(bundle),
    }
    (job / (SCHEMA + ".json")).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def load_verified_g1_provider_bundle(
    receipt_path: Path, *, expected_implementation_commit: str
) -> dict[str, Any]:
    """Recheck the exact dry-run bytes without rebuilding before paid launch."""

    path = Path(receipt_path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("g1_provider_bundle_receipt_missing")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("g1_provider_bundle_receipt_invalid")
    bundle = Path(str(receipt.get("bundle_path") or ""))
    if not bundle.is_absolute() or bundle.is_symlink() or not bundle.is_file():
        raise ValueError("g1_provider_bundle_bytes_missing")
    manifest = {
        key: value
        for key, value in receipt.items()
        if key not in {"bundle_path", "bundle_size_bytes", "bundle_sha256"}
    }
    if (
        receipt.get("schema_version") != SCHEMA
        or receipt.get("status") != "ready"
        or receipt.get("provider_bundle_kind") != PROVIDER_BUNDLE_KIND
        or receipt.get("implementation_commit") != expected_implementation_commit
        or receipt.get("manifest_digest")
        != canonical_digest(manifest, digest_field="manifest_digest")
        or receipt.get("bundle_size_bytes") != bundle.stat().st_size
        or receipt.get("bundle_sha256") != _sha256(bundle)
    ):
        raise ValueError("g1_provider_bundle_receipt_binding_invalid")
    with zipfile.ZipFile(bundle) as archive:
        embedded = json.loads(archive.read(MANIFEST))
        if embedded != manifest or archive.testzip() is not None:
            raise ValueError("g1_provider_bundle_manifest_binding_invalid")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--manipulation-packet", type=Path, required=True)
    parser.add_argument("--movement-packet", type=Path, required=True)
    parser.add_argument("--book-handoff", type=Path, required=True)
    parser.add_argument("--rights-review", action="append", required=True, metavar="CANDIDATE=PATH")
    parser.add_argument("--navigation-authority", type=Path, required=True)
    parser.add_argument("--publisher-source", type=Path, required=True)
    parser.add_argument("--runtime-source-receipt", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    args = parser.parse_args(argv)
    rights = {}
    for row in args.rights_review:
        candidate, separator, raw_path = row.partition("=")
        if not separator or candidate in rights:
            parser.error("--rights-review requires distinct CANDIDATE=PATH values")
        rights[candidate] = Path(raw_path)
    receipt = build_g1_provider_bundle(
        job_dir=args.job_dir,
        manipulation_packet=args.manipulation_packet,
        movement_packet=args.movement_packet,
        book_handoff=args.book_handoff,
        rights_review_paths=rights,
        navigation_authority=args.navigation_authority,
        publisher_source=args.publisher_source,
        runtime_source_receipt=args.runtime_source_receipt,
        implementation_commit=args.implementation_commit,
    )
    print(json.dumps({"status": receipt["status"], "bundle_sha256": receipt["bundle_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
