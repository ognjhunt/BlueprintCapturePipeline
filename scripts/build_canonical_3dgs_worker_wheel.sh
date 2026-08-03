#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <40-hex-source-commit> <output-directory>" >&2
  exit 64
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
requested_commit=$1
output_directory=$2

if [[ ! $requested_commit =~ ^[0-9a-f]{40}$ ]]; then
  echo "canonical_worker_source_commit_invalid" >&2
  exit 65
fi
resolved_commit=$(git -C "$repo_root" rev-parse --verify "${requested_commit}^{commit}")
if [[ $resolved_commit != "$requested_commit" ]]; then
  echo "canonical_worker_source_commit_not_exact" >&2
  exit 65
fi

runtime_python="$repo_root/.venv/bin/python"
if [[ ! -x $runtime_python ]]; then
  echo "canonical_worker_builder_venv_missing" >&2
  exit 66
fi

scratch_root=$(mktemp -d "${TMPDIR:-/private/tmp}/blueprint-canonical-worker-build.XXXXXX")
cleanup() {
  rm -rf "$scratch_root"
}
trap cleanup EXIT
source_root="$scratch_root/source"
wheel_root="$scratch_root/wheel"
mkdir -p "$source_root" "$wheel_root"
git -C "$repo_root" archive "$resolved_commit" | tar -x -C "$source_root"
(cd "$source_root" && uv build --wheel --out-dir "$wheel_root" >/dev/null)

shopt -s nullglob
wheels=("$wheel_root"/*.whl)
shopt -u nullglob
if [[ ${#wheels[@]} -ne 1 ]]; then
  echo "canonical_worker_wheel_count_invalid" >&2
  exit 67
fi
wheel=${wheels[0]}

source_digest=$(PYTHONPATH="$source_root/src" "$runtime_python" -c \
  'from blueprint_pipeline.canonical_3dgs_pipeline import canonical_3dgs_worker_package_digest; print(canonical_3dgs_worker_package_digest())')
wheel_digest=$(PYTHONPATH="$source_root/src" "$runtime_python" -c \
  'import sys; from blueprint_pipeline.canonical_3dgs_pipeline import canonical_3dgs_worker_wheel_package_digest; print(canonical_3dgs_worker_wheel_package_digest(sys.argv[1]))' \
  "$wheel")
if [[ $source_digest != "$wheel_digest" ]]; then
  echo "canonical_worker_wheel_source_digest_mismatch" >&2
  exit 68
fi

mkdir -p "$output_directory"
destination="$output_directory/$(basename "$wheel")"
if [[ -e $destination ]]; then
  if ! cmp -s "$wheel" "$destination"; then
    echo "canonical_worker_wheel_immutable_conflict" >&2
    exit 69
  fi
else
  cp "$wheel" "$destination"
fi

wheel_sha256=$(shasum -a 256 "$destination" | awk '{print $1}')
printf '{"source_commit_sha":"%s","worker_python_package_digest":"%s","wheel_digest":"sha256:%s","wheel_path":"%s"}\n' \
  "$resolved_commit" "$source_digest" "$wheel_sha256" "$destination"
