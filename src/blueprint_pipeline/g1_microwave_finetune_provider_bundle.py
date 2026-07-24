"""Build the small, hash-bound provider bundle for microwave fine-tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import tarfile
from typing import Any, Sequence
import zipfile

from .g1_microwave_finetune_preflight import (
    BOUNDED_MAX_STEPS,
    EMBODIMENT_TAG,
    PINNED_GROOT_N17_REVISION,
    SEALED_SONIC_WARM_START_PATH,
    SEALED_SONIC_WARM_START_REPO,
    SEALED_SONIC_WARM_START_REVISION,
)


SCHEMA_VERSION = "g1_microwave_finetune_provider_bundle.v1"
IMAGE_REF = (
    "docker.io/nijelhunt/blueprint-groot-oscar-eval@sha256:"
    "ab8fbccb714242b55811aa5142933001dfba76d56b5cc29dead4d0bdf1346e88"
)
DATASET_ARCHIVE_NAME = "microwave_owned_lerobot_v21_20260717.tar.gz"
WORKER_NAME = "g1_microwave_finetune_worker.py"
MANIFEST_NAME = "provider_bundle_manifest.json"
BUNDLE_URL_ENV = "BLUEPRINT_G1_MICROWAVE_FINETUNE_BUNDLE_URL"
OUTPUT_PUT_URL_ENV = "BLUEPRINT_G1_MICROWAVE_FINETUNE_OUTPUT_PUT_URL"
CHECKPOINT_PUT_URL_ENV = "BLUEPRINT_G1_MICROWAVE_FINETUNE_CHECKPOINT_PUT_URL"
CHECKPOINT_PART_PUT_URLS_ENV = (
    "BLUEPRINT_G1_MICROWAVE_FINETUNE_CHECKPOINT_PART_PUT_URLS"
)
MAX_PROVIDER_BUNDLE_BYTES = 8 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _zip_info(name: str, *, executable: bool = False) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    mode = 0o755 if executable else 0o644
    info.external_attr = (stat.S_IFREG | mode) << 16
    return info


def _dataset_preflight(dataset_archive: Path) -> dict[str, Any]:
    member_name = (
        "microwave_owned_lerobot_v21_20260717/"
        "groot_n17_finetune_preflight.json"
    )
    with tarfile.open(dataset_archive, "r:gz") as archive:
        apple_double_members = [
            member.name
            for member in archive.getmembers()
            if "__MACOSX" in Path(member.name).parts
            or Path(member.name).name.startswith("._")
        ]
        if apple_double_members:
            raise ValueError(
                "g1_microwave_provider_bundle_appledouble_members_forbidden"
            )
        try:
            member = archive.getmember(member_name)
        except KeyError as exc:
            raise ValueError("g1_microwave_provider_bundle_preflight_missing") from exc
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError("g1_microwave_provider_bundle_preflight_unreadable")
        payload = json.loads(handle.read().decode("utf-8"))
    if payload.get("status") != "qualified_exact_groot_n1_7_training_data_preflight":
        raise ValueError("g1_microwave_provider_bundle_preflight_not_qualified")
    plan = payload.get("bounded_finetune_plan") or {}
    if (
        plan.get("warm_starts_from_sealed_sonic_checkpoint") is not True
        or int(plan.get("max_steps") or 0) != BOUNDED_MAX_STEPS
    ):
        raise ValueError("g1_microwave_provider_bundle_plan_mismatch")
    return payload


def build_provider_bundle(
    *, dataset_archive: str | Path, output_path: str | Path
) -> dict[str, Any]:
    dataset = Path(dataset_archive).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not dataset.is_file() or dataset.is_symlink():
        raise FileNotFoundError("g1_microwave_provider_bundle_dataset_missing_or_unsafe")
    preflight = _dataset_preflight(dataset)
    worker_path = Path(__file__).with_name(WORKER_NAME)
    worker_bytes = worker_path.read_bytes()
    dataset_bytes = dataset.read_bytes()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "qualified_provider_bundle",
        "image_ref": IMAGE_REF,
        "dataset": {
            "name": DATASET_ARCHIVE_NAME,
            "sha256": _sha256_bytes(dataset_bytes),
            "size_bytes": len(dataset_bytes),
            "preflight_status": preflight["status"],
        },
        "worker": {
            "name": WORKER_NAME,
            "sha256": _sha256_bytes(worker_bytes),
            "size_bytes": len(worker_bytes),
        },
        "training": {
            "groot_revision": PINNED_GROOT_N17_REVISION,
            "embodiment_tag": EMBODIMENT_TAG,
            "warm_start_path": SEALED_SONIC_WARM_START_PATH,
            "warm_start_repo": SEALED_SONIC_WARM_START_REPO,
            "warm_start_revision": SEALED_SONIC_WARM_START_REVISION,
            "max_steps": BOUNDED_MAX_STEPS,
            "gpu_count": 1,
        },
        "claim_boundary": {
            "bundle_creation_allocates_gpu": False,
            "fine_tune_not_run": True,
            "worker_requires_open_loop_improvement_over_sealed_warm_start": True,
            "open_loop_exact_owned_training_trajectory_only": True,
            "checkpoint_not_qualified": True,
            "isaac_semantic_success_not_proven": True,
        },
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(_zip_info(DATASET_ARCHIVE_NAME), dataset_bytes)
        archive.writestr(_zip_info(WORKER_NAME, executable=True), worker_bytes)
        archive.writestr(_zip_info(MANIFEST_NAME), manifest_bytes)
    bundle_sha = _sha256(output)
    result = {
        **manifest,
        "bundle": {
            "path": str(output),
            "sha256": bundle_sha,
            "size_bytes": output.stat().st_size,
            "members": [DATASET_ARCHIVE_NAME, WORKER_NAME, MANIFEST_NAME],
        },
    }
    result_path = output.with_suffix(output.suffix + ".json")
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def render_provider_bootstrap(*, expected_bundle_sha256: str) -> str:
    """Render a URL-redacted downloader followed by the bundled worker."""

    digest = str(expected_bundle_sha256).strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("g1_microwave_provider_bundle_sha256_invalid")
    return f"""set -Eeuo pipefail
umask 077
ROOT=/workspace/g1_microwave_finetune_provider
BUNDLE="$ROOT/provider_bundle.zip"
mkdir -p "$ROOT"
BOOTSTRAP_LOG="$ROOT/provider_bootstrap.log"
BOOTSTRAP_STAGE="$ROOT/provider_bootstrap_stage.txt"
exec > >(tee -a "$BOOTSTRAP_LOG") 2>&1
write_bootstrap_stage() {{
  printf '%s\n' "$1" > "$BOOTSTRAP_STAGE"
}}
upload_bootstrap_failure() {{
  rc=$?
  trap - EXIT
  if [ "$rc" -eq 0 ]; then return 0; fi
  set +e
  BOOTSTRAP_RC="$rc" /opt/gr00t-venv/bin/python - "$ROOT" <<'PY'
import json, os, pathlib, sys, urllib.parse, urllib.request, zipfile
root = pathlib.Path(sys.argv[1])
failure = root / "failure_output"
failure.mkdir(parents=True, exist_ok=True)
source_report = pathlib.Path(
    "/workspace/g1_microwave_finetune/g1_microwave_finetune_worker_report.json"
)
report_path = failure / "g1_microwave_finetune_worker_report.json"
stage_path = root / "provider_bootstrap_stage.txt"
try:
    bootstrap_phase = stage_path.read_text(encoding="utf-8").strip()
except OSError:
    bootstrap_phase = "bootstrap_stage_unavailable"
try:
    payload = json.loads(source_report.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("worker report not object")
except Exception:
    payload = {{
        "schema_version": "g1_microwave_finetune_worker.v1",
        "status": "blocked",
        "blockers": ["g1_microwave_finetune_provider_bootstrap_failed"],
        "claim_boundary": {{
            "fine_tune_completed": False,
            "checkpoint_open_loop_qualified": False,
            "isaac_semantic_episode_success_not_proven": True,
        }},
    }}
payload["status"] = "blocked"
payload["provider_bootstrap_exit_code"] = int(os.environ["BOOTSTRAP_RC"])
payload["provider_bootstrap_phase"] = bootstrap_phase
payload.setdefault("blockers", []).append(
    f"g1_microwave_finetune_provider_bootstrap_exit_{{os.environ['BOOTSTRAP_RC']}}"
)
payload["blockers"] = sorted(set(payload["blockers"]))
report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
archive_path = root / "g1_microwave_finetune_bootstrap_failure.zip"
with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write(report_path, report_path.name)
    for candidate in (
        root / "provider_bootstrap.log",
        stage_path,
        pathlib.Path("/workspace/g1_microwave_finetune/training.log"),
        pathlib.Path("/workspace/g1_microwave_finetune/microwave_open_loop_warm_start.log"),
        pathlib.Path("/workspace/g1_microwave_finetune/microwave_open_loop_finetuned.log"),
        pathlib.Path("/workspace/g1_microwave_finetune/microwave_open_loop_warm_start.json"),
        pathlib.Path("/workspace/g1_microwave_finetune/microwave_open_loop_finetuned.json"),
    ):
        if candidate.is_file() and not candidate.is_symlink():
            archive.write(candidate, candidate.name)
put_url = os.environ.get({OUTPUT_PUT_URL_ENV!r}, "")
parsed = urllib.parse.urlsplit(put_url)
if parsed.scheme == "https" and parsed.hostname and not parsed.username and not parsed.password:
    request = urllib.request.Request(
        put_url,
        data=archive_path.read_bytes(),
        headers={{"Content-Type": "application/zip"}},
        method="PUT",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        if response.status not in {{200, 201, 204}}:
            raise SystemExit("g1_microwave_bootstrap_failure_upload_failed")
PY
  exit "$rc"
}}
trap upload_bootstrap_failure EXIT
write_bootstrap_stage bundle_download_started
/opt/gr00t-venv/bin/python - "$BUNDLE" <<'PY'
import hashlib
import os
from pathlib import Path
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile

url = os.environ.get({BUNDLE_URL_ENV!r}, "")
parsed = urllib.parse.urlsplit(url)
if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
    raise SystemExit("g1_microwave_finetune_bundle_url_invalid")

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirect forbidden", headers, fp)

request = urllib.request.Request(url, method="GET")
with urllib.request.build_opener(NoRedirect()).open(request, timeout=180) as response:
    if response.status != 200:
        raise SystemExit("g1_microwave_finetune_bundle_download_failed")
    payload = response.read({MAX_PROVIDER_BUNDLE_BYTES + 1})
if len(payload) > {MAX_PROVIDER_BUNDLE_BYTES}:
    raise SystemExit("g1_microwave_finetune_bundle_too_large")
if hashlib.sha256(payload).hexdigest() != {digest!r}:
    raise SystemExit("g1_microwave_finetune_bundle_sha256_mismatch")
destination = Path(sys.argv[1])
destination.write_bytes(payload)
root = destination.parent.resolve()
with zipfile.ZipFile(destination) as archive:
    names = set(archive.namelist())
    required = {{{DATASET_ARCHIVE_NAME!r}, {WORKER_NAME!r}, {MANIFEST_NAME!r}}}
    if names != required:
        raise SystemExit("g1_microwave_finetune_bundle_members_invalid")
    for info in archive.infolist():
        target = (root / info.filename).resolve()
        if root != target and root not in target.parents:
            raise SystemExit("g1_microwave_finetune_bundle_path_traversal")
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise SystemExit("g1_microwave_finetune_bundle_link_forbidden")
    archive.extractall(root)
print("g1_microwave_finetune_bundle_ready")
PY
write_bootstrap_stage bundle_verified
DATASET_SHA=$(/opt/gr00t-venv/bin/python - "$ROOT/{MANIFEST_NAME}" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["dataset"]["sha256"])
PY
)
write_bootstrap_stage dataset_manifest_verified
chmod 700 "$ROOT/{WORKER_NAME}"
write_bootstrap_stage worker_invocation_started
/opt/gr00t-venv/bin/python "$ROOT/{WORKER_NAME}" \
  --dataset-archive "$ROOT/{DATASET_ARCHIVE_NAME}" \
  --expected-dataset-sha256 "$DATASET_SHA" \
  --workspace /workspace/g1_microwave_finetune
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-archive", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = build_provider_bundle(
            dataset_archive=args.dataset_archive,
            output_path=args.output_path,
        )
        script = render_provider_bootstrap(
            expected_bundle_sha256=result["bundle"]["sha256"]
        )
        syntax = subprocess.run(
            ["bash", "-n"], input=script, check=False, capture_output=True, text=True
        )
        if syntax.returncode != 0:
            return 1
    except (OSError, ValueError, json.JSONDecodeError, tarfile.TarError):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
