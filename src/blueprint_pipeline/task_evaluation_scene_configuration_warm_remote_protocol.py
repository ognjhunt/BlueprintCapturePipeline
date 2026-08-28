"""Fixed remote protocol for one warm scene diagnostic iteration."""

from __future__ import annotations

import shlex
import json
from collections.abc import Mapping
from typing import Any

from .vast_provider_transfer_upload import provider_output_upload_shell_fragment


REMOTE_ROOT = "/workspace/task_evaluation_scene_configuration_warm"
BASE_RUNTIME_ROOT = (
    "/workspace/task_evaluation_scene_configuration_provider_bundle/provider_runtime"
)
_FIXED_FORBIDDEN_ENV_NAMES = (
    "OPENAI_API_KEY",
    "OPENAI_API_KEY_FILE",
    "OPENAI_ADMIN_API_KEY_FILE",
    "BLUEPRINT_OPENAI_ADMIN_KEY",
    "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_FILE",
    "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE",
    "OPENAI_CONTENT_AGENTS_API_KEY_FILE",
    "BLUEPRINT_OPENAI_ARTIFIXER_SEMANTIC_TEACHER_COST_SCOPE_ATTESTATION_FILE",
    "BLUEPRINT_OPENAI_ARTIFIXER_VISUAL_REVIEW_COST_SCOPE_ATTESTATION_FILE",
    "BLUEPRINT_OPENAI_CONTENT_AGENTS_COST_SCOPE_ATTESTATION_FILE",
    "BLUEPRINT_OPENAI_COST_SCOPE_ATTESTATION_FILE",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)
_ARTIFIXER_ALLOWED_SECRET_FILE_NAMES = (
    "OPENAI_ADMIN_API_KEY_FILE",
    "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE",
    "OPENAI_CONTENT_AGENTS_API_KEY_FILE",
    "BLUEPRINT_OPENAI_ARTIFIXER_VISUAL_REVIEW_COST_SCOPE_ATTESTATION_FILE",
    "BLUEPRINT_OPENAI_CONTENT_AGENTS_COST_SCOPE_ATTESTATION_FILE",
)
_ARTIFIXER_CONTINUATION_KIND = (
    "artifixer_post_training_visual_review_and_remaining_chain"
)


def artifixer_warm_secret_install_remote_argv(*, iteration_id: str) -> list[str]:
    """Return a fixed installer; secret bytes arrive only on encrypted stdin."""

    program = r'''import json, os, pathlib, re, shutil, sys
iteration = sys.argv[1]
allowed = {
    "OPENAI_ADMIN_API_KEY_FILE",
    "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE",
    "OPENAI_CONTENT_AGENTS_API_KEY_FILE",
    "BLUEPRINT_OPENAI_ARTIFIXER_VISUAL_REVIEW_COST_SCOPE_ATTESTATION_FILE",
    "BLUEPRINT_OPENAI_CONTENT_AGENTS_COST_SCOPE_ATTESTATION_FILE",
}
if re.fullmatch(r"i[0-9]{3}-[0-9a-f]{12}", iteration) is None:
    raise SystemExit(71)
base = pathlib.Path("/workspace/task_evaluation_scene_configuration_warm/pending-secrets")
root = base / iteration
try:
    value = json.load(sys.stdin)
    files = value.get("files") if isinstance(value, dict) else None
    if value.get("iteration_id") != iteration or not isinstance(files, dict) or set(files) != allowed:
        raise ValueError("scope")
    if root.exists() or root.is_symlink():
        raise ValueError("replay")
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(base, 0o700)
    root.mkdir(mode=0o700)
    manifest = {}
    for name in sorted(allowed):
        payload = files[name]
        if not isinstance(payload, str) or not payload or len(payload.encode()) > 65536 or "\x00" in payload:
            raise ValueError("payload")
        filename = name.lower()
        path = root / filename
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        manifest[name] = filename
    manifest_path = root / "manifest.json"
    descriptor = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({"names": manifest}, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print("BLUEPRINT_SCENE_ARTIFIXER_WARM_SECRET_FILES_READY")
except Exception:
    shutil.rmtree(root, ignore_errors=True)
    raise SystemExit(72)
'''
    return ["python3", "-c", program, iteration_id]


def artifixer_warm_secret_envelope(
    *, iteration_id: str, secret_values: Mapping[str, str]
) -> bytes:
    if set(secret_values) != set(_ARTIFIXER_ALLOWED_SECRET_FILE_NAMES):
        raise ValueError("scene_configuration_artifixer_warm_secret_scope_invalid")
    payload = json.dumps(
        {"iteration_id": iteration_id, "files": dict(secret_values)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > 330_000 or b"\x00" in payload:
        raise ValueError("scene_configuration_artifixer_warm_secret_scope_invalid")
    return payload


def _warm_no_secret_shell_command(command: str) -> str:
    fixed = " \\\n".join(f"  -u {name}" for name in _FIXED_FORBIDDEN_ENV_NAMES)
    return f"""SECRET_ENV_UNSETS=()
while IFS= read -r secret_env_name; do
  case "$secret_env_name" in
    BLUEPRINT_VAST_RUNTIME_SECRET_B64_[A-Z0-9_]*)
      SECRET_ENV_UNSETS+=( -u "$secret_env_name" ) ;;
  esac
done < <(compgen -e)
env "${{SECRET_ENV_UNSETS[@]}}" \\
{fixed} \\
  {command}"""


def _warm_artifixer_scoped_secret_shell_command(command: str) -> str:
    forbidden = tuple(
        name
        for name in _FIXED_FORBIDDEN_ENV_NAMES
        if name not in _ARTIFIXER_ALLOWED_SECRET_FILE_NAMES
    )
    fixed = " \\\n".join(f"  -u {name}" for name in forbidden)
    return f"""SECRET_ENV_UNSETS=()
while IFS= read -r secret_env_name; do
  case "$secret_env_name" in
    BLUEPRINT_VAST_RUNTIME_SECRET_B64_[A-Z0-9_]*)
      SECRET_ENV_UNSETS+=( -u "$secret_env_name" ) ;;
  esac
done < <(compgen -e)
env "${{SECRET_ENV_UNSETS[@]}}" \\
{fixed} \\
  {command}"""


def _remote_iteration_script(
    *,
    authority: Mapping[str, Any],
    session: Mapping[str, Any],
    overlay_url: str,
    output_put_url: str,
) -> str:
    """Return the only admitted remote operation for one warm iteration."""

    artifixer_continuation = (
        authority.get("continuation_kind") == _ARTIFIXER_CONTINUATION_KIND
    )
    values = {
        "ITERATION_ID": str(authority["iteration_id"]),
        "OVERLAY_URL": overlay_url,
        "OUTPUT_PUT_URL": output_put_url,
        "OVERLAY_SHA": str(authority["source_overlay_archive_sha256"]),
        "OVERLAY_MANIFEST_DIGEST": str(
            authority["source_overlay_manifest_digest"]
        ),
        "SOURCE_COMMIT": str(authority["source_commit"]),
        "SOURCE_CHECKPOINT_DIGEST": str(authority["source_checkpoint_digest"]),
        "SCIENTIFIC_BINDING_DIGEST": str(authority["scientific_binding_digest"]),
        "SOURCE_CHECKPOINT_ROOT": str(authority["remote_checkpoint_root"]),
        "WATCHDOG_DEADLINE": str(authority["watchdog_deadline_epoch"]),
        "OUTPUT_MAX_BYTES": str(authority["maximum_output_archive_bytes"]),
        "CONTINUATION_KIND": str(authority.get("continuation_kind") or ""),
        "ARTIFIXER_CHECKPOINT_ROOT": str(
            authority.get("remote_artifixer_post_training_checkpoint_root") or ""
        ),
    }
    if artifixer_continuation:
        public = authority.get("openai_public_environment") or {}
        caps = authority.get("openai_stage_max_cost_usd") or {}
        values.update(
            {
                "OPENAI_PROJECT_ID_VALUE": str(public["OPENAI_PROJECT_ID"]),
                "OPENAI_REVIEW_KEY_ID_VALUE": str(
                    public["OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_ID"]
                ),
                "OPENAI_CONTENT_KEY_ID_VALUE": str(
                    public["OPENAI_CONTENT_AGENTS_API_KEY_ID"]
                ),
                "SCENE_AUTHORITY_DIGEST_VALUE": str(
                    public["BLUEPRINT_SCENE_CONFIGURATION_AUTHORITY_DIGEST"]
                ),
                "OPENAI_REVIEW_CAP_VALUE": str(
                    caps["artifixer_visual_review"]
                ),
                "OPENAI_CONTENT_CAP_VALUE": str(caps["content_agents"]),
                "OPENAI_TOTAL_CAP_VALUE": str(
                    float(caps["artifixer_visual_review"])
                    + float(caps["content_agents"])
                ),
                "OPENAI_MAX_REQUESTS_VALUE": str(
                    authority["openai_maximum_requests"]
                ),
            }
        )
    assignments = "\n".join(
        f"{name}={shlex.quote(value)}" for name, value in values.items()
    )
    entrypoint_command = (
        _warm_artifixer_scoped_secret_shell_command(
            'bash "$ITERATION_ROOT/runtime/run_task_evaluation_scene_configuration_provider.sh"'
        )
        if artifixer_continuation
        else _warm_no_secret_shell_command(
            'bash "$ITERATION_ROOT/runtime/run_task_evaluation_scene_configuration_provider.sh"'
        )
    )
    secret_setup = (
        f'''PENDING_SECRET_ROOT={REMOTE_ROOT}/pending-secrets/$ITERATION_ID
ARTIFIXER_SECRET_ROOT="$ITERATION_ROOT/.runtime-secrets"
if [ ! -d "$PENDING_SECRET_ROOT" ] || [ -L "$PENDING_SECRET_ROOT" ]; then
  echo BLUEPRINT_SCENE_WARM_BLOCKED:artifixer_secret_install_missing
  exit 76
fi
export PENDING_SECRET_ROOT
python3 - <<'PY'
import json, os, pathlib, stat
root = pathlib.Path(os.environ["PENDING_SECRET_ROOT"])
manifest = json.loads((root / "manifest.json").read_text())
expected = {sorted(_ARTIFIXER_ALLOWED_SECRET_FILE_NAMES)!r}
names = manifest.get("names") if isinstance(manifest, dict) else None
paths = list(root.iterdir())
if (
    root.stat().st_uid != 0
    or stat.S_IMODE(root.stat().st_mode) != 0o700
    or not isinstance(names, dict)
    or sorted(names) != expected
    or len(paths) != len(expected) + 1
):
    raise SystemExit(76)
for name in expected:
    path = root / names[name]
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o600 or path.stat().st_uid != 0:
        raise SystemExit(76)
PY
mv "$PENDING_SECRET_ROOT" "$ARTIFIXER_SECRET_ROOT"
trap 'rm -rf -- "$ARTIFIXER_SECRET_ROOT" "$PENDING_SECRET_ROOT"' EXIT
export OPENAI_ADMIN_API_KEY_FILE="$ARTIFIXER_SECRET_ROOT/openai_admin_api_key_file"
export OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE="$ARTIFIXER_SECRET_ROOT/openai_artifixer_visual_review_api_key_file"
export OPENAI_CONTENT_AGENTS_API_KEY_FILE="$ARTIFIXER_SECRET_ROOT/openai_content_agents_api_key_file"
export BLUEPRINT_OPENAI_ARTIFIXER_VISUAL_REVIEW_COST_SCOPE_ATTESTATION_FILE="$ARTIFIXER_SECRET_ROOT/blueprint_openai_artifixer_visual_review_cost_scope_attestation_file"
export BLUEPRINT_OPENAI_CONTENT_AGENTS_COST_SCOPE_ATTESTATION_FILE="$ARTIFIXER_SECRET_ROOT/blueprint_openai_content_agents_cost_scope_attestation_file"
export OPENAI_PROJECT_ID="$OPENAI_PROJECT_ID_VALUE"
export OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_ID="$OPENAI_REVIEW_KEY_ID_VALUE"
export OPENAI_CONTENT_AGENTS_API_KEY_ID="$OPENAI_CONTENT_KEY_ID_VALUE"
export BLUEPRINT_SCENE_CONFIGURATION_AUTHORITY_DIGEST="$SCENE_AUTHORITY_DIGEST_VALUE"
export BLUEPRINT_SCENE_CONFIGURATION_OPENAI_ARTIFIXER_VISUAL_REVIEW_MAX_COST_USD="$OPENAI_REVIEW_CAP_VALUE"
export BLUEPRINT_SCENE_CONFIGURATION_OPENAI_CONTENT_AGENTS_MAX_COST_USD="$OPENAI_CONTENT_CAP_VALUE"
export BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_COST_USD="$OPENAI_TOTAL_CAP_VALUE"
export BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_REQUESTS="$OPENAI_MAX_REQUESTS_VALUE"
echo BLUEPRINT_SCENE_ARTIFIXER_WARM_SECRET_FILES_INSTALLED'''
        if artifixer_continuation
        else ""
    )
    post_checkpoint_copy = (
        '''artifixer_checkpoint_target = runtime / "input/artifixer_post_training_checkpoint"
if artifixer_checkpoint_target.exists():
    shutil.rmtree(artifixer_checkpoint_target)
shutil.copytree(Path(os.environ["BLUEPRINT_SCENE_WARM_ARTIFIXER_CHECKPOINT_ROOT"]), artifixer_checkpoint_target, copy_function=shutil.copy2, symlinks=False)'''
        if artifixer_continuation
        else ""
    )
    post_checkpoint_export = (
        'export BLUEPRINT_SCENE_CONFIGURATION_ARTIFIXER_POST_TRAINING_CHECKPOINT_ROOT="$ITERATION_ROOT/runtime/input/artifixer_post_training_checkpoint"'
        if artifixer_continuation
        else ""
    )
    secret_scrub = (
        '''rm -rf -- "$ARTIFIXER_SECRET_ROOT" "$PENDING_SECRET_ROOT"
if [ -e "$ARTIFIXER_SECRET_ROOT" ] || [ -e "$PENDING_SECRET_ROOT" ]; then
  echo BLUEPRINT_SCENE_WARM_BLOCKED:artifixer_secret_scrub_unproven
  PROVIDER_RC=86
else
  echo BLUEPRINT_SCENE_ARTIFIXER_WARM_SECRET_FILES_SCRUBBED
fi
trap - EXIT'''
        if artifixer_continuation
        else ""
    )
    upload_fragment = provider_output_upload_shell_fragment()
    return f"""#!/usr/bin/env bash
set -euo pipefail
{assignments}
mkdir -p {REMOTE_ROOT}/iterations
chmod 700 {REMOTE_ROOT} {REMOTE_ROOT}/iterations
exec 9>{REMOTE_ROOT}/owner.lock
if ! flock -n 9; then
  echo BLUEPRINT_SCENE_WARM_BLOCKED:another_iteration_running
  exit 73
fi
ITERATION_ROOT={REMOTE_ROOT}/iterations/$ITERATION_ID
if [ -e "$ITERATION_ROOT" ]; then
  echo BLUEPRINT_SCENE_WARM_BLOCKED:iteration_replayed
  exit 74
fi
export SOURCE_CHECKPOINT_ROOT OUTPUT_MAX_BYTES
export BLUEPRINT_SCENE_WARM_ARTIFIXER_CHECKPOINT_ROOT="$ARTIFIXER_CHECKPOINT_ROOT"
if ! python3 - <<'PY'
import re
import shutil
from pathlib import Path

iterations = Path("/workspace/task_evaluation_scene_configuration_warm/iterations")
checkpoint = Path(__import__("os").environ["SOURCE_CHECKPOINT_ROOT"])
preserve = None
try:
    relative = checkpoint.relative_to(iterations)
except ValueError:
    relative = None
if relative is not None and relative.parts:
    preserve = relative.parts[0]
for child in iterations.iterdir():
    if child.name == preserve:
        continue
    if child.is_symlink() or not child.is_dir() or re.fullmatch(r"i[0-9]{{3}}-[0-9a-f]{{12}}", child.name) is None:
        raise SystemExit("unsafe retained iteration entry")
    shutil.rmtree(child)
PY
then
  echo BLUEPRINT_SCENE_WARM_BLOCKED:iteration_gc_unproven
  exit 74
fi
mkdir -p "$ITERATION_ROOT"
chmod 700 "$ITERATION_ROOT"
export BLUEPRINT_SCENE_WARM_ITERATION_ROOT="$ITERATION_ROOT"
echo BLUEPRINT_SCENE_WARM_REMOTE_SETUP_STARTED_EPOCH_NS:$(date +%s%N)
TRANSFER_REMAINING="$(( ${{WATCHDOG_DEADLINE%.*}} - $(date +%s) - 120 ))"
if [ "$TRANSFER_REMAINING" -le 0 ]; then
  echo BLUEPRINT_SCENE_WARM_BLOCKED:overlay_download_deadline_exhausted
  exit 75
fi
OVERLAY_TIMEOUT="$TRANSFER_REMAINING"
if [ "$OVERLAY_TIMEOUT" -gt 300 ]; then OVERLAY_TIMEOUT=300; fi
if ! curl -fsS --http1.1 --retry 3 --retry-connrefused --retry-delay 2 \
  --connect-timeout 15 --max-time "$OVERLAY_TIMEOUT" \
  "$OVERLAY_URL" -o "$ITERATION_ROOT/overlay.zip"; then
  echo BLUEPRINT_SCENE_WARM_BLOCKED:overlay_download_failed
  exit 75
fi
ACTUAL_SHA="sha256:$(sha256sum "$ITERATION_ROOT/overlay.zip" | cut -d' ' -f1)"
if [ "$ACTUAL_SHA" != "$OVERLAY_SHA" ]; then
  echo BLUEPRINT_SCENE_WARM_BLOCKED:overlay_digest_mismatch
  exit 75
fi
if ! python3 - <<'PY'
import os
from pathlib import Path

base = Path("/workspace/task_evaluation_scene_configuration_provider_bundle/provider_runtime")
checkpoint = Path(os.environ["SOURCE_CHECKPOINT_ROOT"])
iteration = Path(os.environ["BLUEPRINT_SCENE_WARM_ITERATION_ROOT"])
limit = int(os.environ["OUTPUT_MAX_BYTES"])
def tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file() and not path.is_symlink())
if base.is_symlink() or not base.is_dir() or checkpoint.is_symlink() or not checkpoint.is_dir():
    raise SystemExit("runtime or checkpoint unavailable")
stats = os.statvfs(iteration)
free = int(stats.f_bavail) * int(stats.f_frsize)
required = tree_bytes(base) + tree_bytes(checkpoint) + (5 * limit) + (1024 ** 3)
if free < required:
    raise SystemExit("insufficient retained iteration capacity")
PY
then
  echo BLUEPRINT_SCENE_WARM_BLOCKED:iteration_disk_capacity_insufficient
  exit 75
fi
export BLUEPRINT_SCENE_WARM_BASE_RUNTIME={BASE_RUNTIME_ROOT}
export BLUEPRINT_SCENE_WARM_SOURCE_CHECKPOINT_ROOT="$SOURCE_CHECKPOINT_ROOT"
export BLUEPRINT_SCENE_WARM_EXPECTED_MANIFEST_DIGEST="$OVERLAY_MANIFEST_DIGEST"
export BLUEPRINT_SCENE_WARM_EXPECTED_SOURCE_COMMIT="$SOURCE_COMMIT"
export BLUEPRINT_SCENE_WARM_EXPECTED_CHECKPOINT_DIGEST="$SOURCE_CHECKPOINT_DIGEST"
export BLUEPRINT_SCENE_WARM_EXPECTED_SCIENTIFIC_BINDING_DIGEST="$SCIENTIFIC_BINDING_DIGEST"
mkdir -p "$ITERATION_ROOT/runtime"
if ! cp -a --reflink=auto {BASE_RUNTIME_ROOT}/. "$ITERATION_ROOT/runtime/"; then
  rm -rf "$ITERATION_ROOT/runtime"
  mkdir -p "$ITERATION_ROOT/runtime"
  cp -a {BASE_RUNTIME_ROOT}/. "$ITERATION_ROOT/runtime/"
fi
python3 - <<'PY'
import hashlib
import json
import os
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

root = Path(os.environ["BLUEPRINT_SCENE_WARM_ITERATION_ROOT"])
base = Path(os.environ["BLUEPRINT_SCENE_WARM_BASE_RUNTIME"])
checkpoint = Path(os.environ["BLUEPRINT_SCENE_WARM_SOURCE_CHECKPOINT_ROOT"])
runtime = root / "runtime"
archive_path = root / "overlay.zip"
manifest_name = "overlay/task_evaluation_scene_configuration_warm_source_overlay.v1.json"
if not base.is_dir() or base.is_symlink() or not checkpoint.is_dir() or checkpoint.is_symlink():
    raise SystemExit("BLUEPRINT_SCENE_WARM_BLOCKED:base_runtime_or_checkpoint_invalid")
with zipfile.ZipFile(archive_path) as archive:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)) or manifest_name not in names:
        raise SystemExit("BLUEPRINT_SCENE_WARM_BLOCKED:overlay_inventory_invalid")
    manifest = json.loads(archive.read(manifest_name))
    if (
        manifest.get("source_commit") != os.environ["BLUEPRINT_SCENE_WARM_EXPECTED_SOURCE_COMMIT"]
        or manifest.get("source_checkpoint_digest") != os.environ["BLUEPRINT_SCENE_WARM_EXPECTED_CHECKPOINT_DIGEST"]
        or manifest.get("scientific_binding_digest") != os.environ["BLUEPRINT_SCENE_WARM_EXPECTED_SCIENTIFIC_BINDING_DIGEST"]
        or manifest.get("manifest_digest") != os.environ["BLUEPRINT_SCENE_WARM_EXPECTED_MANIFEST_DIGEST"]
        or manifest.get("diagnostic_only") is not True
        or manifest.get("qualification_eligible") is not False
        or manifest.get("configured_revision_publication_permitted") is not False
        or manifest.get("offering_publication_permitted") is not False
        or manifest.get("terminal_e2e_completion_permitted") is not False
        or manifest.get("arbitrary_command_permitted") is not False
        or manifest.get("replacement_roots") != ["provider_runtime/blueprint_pipeline"]
        or manifest.get("exact_replacement_files") != [
            "provider_runtime/task_evaluation_scene_configuration_provider_runner.py",
            "provider_runtime/run_task_evaluation_scene_configuration_provider.sh",
        ]
    ):
        raise SystemExit("BLUEPRINT_SCENE_WARM_BLOCKED:overlay_manifest_invalid")
    expected = {{manifest_name}}
    by_name = {{info.filename: info for info in infos}}
    package_root = runtime / "blueprint_pipeline"
    if package_root.is_symlink():
        raise SystemExit("BLUEPRINT_SCENE_WARM_BLOCKED:overlay_base_package_symlink")
    if package_root.exists():
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True)
    for exact_relative in manifest["exact_replacement_files"]:
        exact_parts = PurePosixPath(exact_relative).parts
        exact_path = runtime.joinpath(*exact_parts[1:])
        if exact_path.exists() or exact_path.is_symlink():
            exact_path.unlink()
    for row in manifest.get("inventory") or []:
        relative = PurePosixPath(str(row.get("provider_relative_path") or ""))
        name = "overlay/" + relative.as_posix()
        if relative.is_absolute() or ".." in relative.parts or name in expected:
            raise SystemExit("BLUEPRINT_SCENE_WARM_BLOCKED:overlay_inventory_invalid")
        info = by_name.get(name)
        mode = int(row.get("mode") or -1)
        payload = archive.read(name) if info is not None else b""
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        if (
            info is None
            or stat.S_IFMT(info.external_attr >> 16) != stat.S_IFREG
            or stat.S_IMODE(info.external_attr >> 16) != mode
            or len(payload) != row.get("size_bytes")
            or digest != row.get("sha256")
        ):
            raise SystemExit("BLUEPRINT_SCENE_WARM_BLOCKED:overlay_inventory_invalid")
        destination = runtime.joinpath(*relative.parts[1:])
        if relative.parts[0] != "provider_runtime" or not destination.is_relative_to(runtime):
            raise SystemExit("BLUEPRINT_SCENE_WARM_BLOCKED:overlay_path_invalid")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # The base was reflinked/copied. Unlink before writing so even an
        # unexpectedly shared inode can never be modified in place.
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(destination, mode)
        expected.add(name)
    if set(names) != expected:
        raise SystemExit("BLUEPRINT_SCENE_WARM_BLOCKED:overlay_extra_member")
    expected_package_files = {{
        PurePosixPath(str(row["provider_relative_path"])).relative_to(
            "provider_runtime"
        ).as_posix()
        for row in manifest["inventory"]
        if str(row["provider_relative_path"]).startswith(
            "provider_runtime/blueprint_pipeline/"
        )
    }}
    actual_package_files = {{
        path.relative_to(runtime).as_posix()
        for path in package_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }}
    if actual_package_files != expected_package_files:
        raise SystemExit("BLUEPRINT_SCENE_WARM_BLOCKED:overlay_final_inventory_mismatch")
manifest_path = runtime / "task_evaluation_scene_configuration_warm_source_overlay.v1.json"
manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
checkpoint_target = runtime / "input/diagnostic_checkpoint"
if checkpoint_target.exists():
    shutil.rmtree(checkpoint_target)
shutil.copytree(checkpoint, checkpoint_target, copy_function=shutil.copy2, symlinks=False)
{post_checkpoint_copy}
PY
echo BLUEPRINT_SCENE_WARM_OVERLAY_APPLIED
echo BLUEPRINT_SCENE_WARM_OVERLAY_APPLIED_EPOCH_NS:$(date +%s%N)
mkdir -p "$ITERATION_ROOT/output"
export BLUEPRINT_SCENE_CONFIGURATION_RUNTIME_ROOT="$ITERATION_ROOT/runtime"
export BLUEPRINT_SCENE_CONFIGURATION_OUTPUT_ROOT="$ITERATION_ROOT/output"
export BLUEPRINT_SCENE_CONFIGURATION_PROVIDER_RESULT="$ITERATION_ROOT/output/task_evaluation_scene_configuration_provider_result.v1.json"
export BLUEPRINT_SCENE_CONFIGURATION_DIAGNOSTIC_SOURCE_COMMIT="$SOURCE_COMMIT"
export BLUEPRINT_SCENE_CONFIGURATION_DIAGNOSTIC_SOURCE_OVERLAY_MANIFEST="$ITERATION_ROOT/runtime/task_evaluation_scene_configuration_warm_source_overlay.v1.json"
export BLUEPRINT_SCENE_CONFIGURATION_DIAGNOSTIC_SOURCE_OVERLAY_MANIFEST_DIGEST="$OVERLAY_MANIFEST_DIGEST"
export BLUEPRINT_SCENE_CONFIGURATION_WARM_SESSION_DIGEST={shlex.quote(str(session["session_digest"]))}
export BLUEPRINT_SCENE_CONFIGURATION_WARM_PROVIDER_INSTANCE_ID={shlex.quote(str(session["provider_instance_id"]))}
export BLUEPRINT_SCENE_CONFIGURATION_WARM_BOOTSTRAP_ALLOCATION_BINDING_DIGEST={shlex.quote(str(session["bootstrap_allocation_binding_digest"]))}
export BLUEPRINT_SCENE_CONFIGURATION_PARENT_DEADLINE_EPOCH="$WATCHDOG_DEADLINE"
{post_checkpoint_export}
{secret_setup}
echo BLUEPRINT_SCENE_WARM_ENTRYPOINT_STARTED
echo BLUEPRINT_SCENE_WARM_ENTRYPOINT_STARTED_EPOCH_NS:$(date +%s%N)
set +e
{entrypoint_command}
PROVIDER_RC=$?
set -e
echo BLUEPRINT_SCENE_WARM_ENTRYPOINT_EXIT_CODE:$PROVIDER_RC
{secret_scrub}
export OUTPUT_MAX_BYTES
python3 - <<'PY'
import os
import zipfile
from pathlib import Path
root = Path(os.environ["BLUEPRINT_SCENE_WARM_ITERATION_ROOT"])
output = root / "provider_runtime_output.zip"
maximum_archive_bytes = int(os.environ["OUTPUT_MAX_BYTES"])
maximum_expanded_bytes = maximum_archive_bytes * 4
with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
    files = [path for path in sorted((root / "output").rglob("*")) if path.is_file() and not path.is_symlink()]
    if len(files) > 10000:
        raise SystemExit("BLUEPRINT_SCENE_WARM_BLOCKED:output_member_count_invalid")
    expanded_bytes = sum(path.stat().st_size for path in files)
    if expanded_bytes > maximum_expanded_bytes:
        raise SystemExit("BLUEPRINT_SCENE_WARM_BLOCKED:output_expansion_invalid")
    for path in files:
        if path.is_file() and not path.is_symlink():
            archive.write(path, path.relative_to(root / "output").as_posix())
with zipfile.ZipFile(output) as archive:
    members = archive.infolist()
    names = [member.filename for member in members]
    if len(members) > 10000 or len(names) != len(set(names)):
        raise SystemExit("BLUEPRINT_SCENE_WARM_BLOCKED:output_member_count_invalid")
    if sum(member.file_size for member in members) > maximum_expanded_bytes:
        raise SystemExit("BLUEPRINT_SCENE_WARM_BLOCKED:output_expansion_invalid")
PY
OUTPUT_SIZE="$(wc -c < "$ITERATION_ROOT/provider_runtime_output.zip" | tr -d ' ')"
if [ "$OUTPUT_SIZE" -le 0 ] || [ "$OUTPUT_SIZE" -gt "$OUTPUT_MAX_BYTES" ]; then
  echo BLUEPRINT_SCENE_WARM_BLOCKED:output_archive_size_invalid
  exit 76
fi
export BLUEPRINT_VAST_EXPECTED_PROVIDER_UPLOAD_BYTES="$OUTPUT_MAX_BYTES"
{upload_fragment}
if ! blueprint_upload_put "$OUTPUT_PUT_URL" "$ITERATION_ROOT/provider_runtime_output.zip"; then
  echo BLUEPRINT_SCENE_WARM_BLOCKED:output_upload_failed
  exit 76
fi
chmod -R a-w "$ITERATION_ROOT/runtime" "$ITERATION_ROOT/output"
echo BLUEPRINT_SCENE_WARM_PROVIDER_OUTPUT_UPLOAD_OK
"""
