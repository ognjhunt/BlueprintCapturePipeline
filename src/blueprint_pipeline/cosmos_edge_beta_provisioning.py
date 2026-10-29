"""Emit Cosmos' isolated, pinned server bootstrap for an admitted beta worker.

This module creates no resource and performs no network calls. The existing
paid-resource allocator owns the worker, TTL, charge ceiling and teardown.
"""
from .adp009d_policy_provisioning import UV_ARCHIVE_SHA256, UV_ARCHIVE_URL, UV_VERSION_OUTPUT
from .cosmos_edge_beta_runtime import CANDIDATE_ID, SNAPSHOT_STAGED_NAME
from .cosmos_edge_droid_policy_runtime import COSMOS_FRAMEWORK_REVISION, MODEL_ID, MODEL_REVISION

DEPENDENCY_LOCK_SHA256 = "f5ec25154f3dc63f49680c91c0dee807b02ea6b846833d6f204bc5659f67e555"
# Runtime-only client wheels from that same uv.lock. Isaac supplies NumPy;
# installing a second numerical stack into its interpreter is not permitted.
CLIENT_REQUIREMENTS = """openpi-client==0.1.2 --hash=sha256:b8aa6d2ea7172d3b7c1fcf867d43930052b5ff19ea7b777966781105aa989897
websockets==16.0 --hash=sha256:1637db62fad1dc833276dded54215f2c7fa46912301a24bd94d45d46a011ceec --hash=sha256:9b5aca38b67492ef518a8ab76851862488a478602229112c4b0d58d63a7a4d5c
msgpack==1.1.2 --hash=sha256:372839311ccf6bdaf39b00b61288e0557916c3729529b301c52c2d88842add42
typing-extensions==4.15.0 --hash=sha256:f0fa19c6845758ab08074a0cfa8b7aecb71c999ca73d62883bc25cc018c4e548
"""


def build_provisioning_script() -> str:
    return f'''#!/usr/bin/env bash
set -euo pipefail
candidate={CANDIDATE_ID}
source_root=/opt/blueprint-cosmos-edge/source
venv_root=/opt/blueprint-cosmos-edge/venv
checkpoint_root=/opt/blueprint-cosmos-edge/checkpoint
uv_root=/opt/blueprint-cosmos-edge/uv
export PYTHONPATH="$RUNTIME_DIR${{PYTHONPATH:+:$PYTHONPATH}}"
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN || true
mkdir -p "$uv_root" "$checkpoint_root"
uv_archive="$(mktemp /tmp/blueprint-cosmos-uv.XXXXXX.tar.gz)"
trap 'rm -f "$uv_archive"' EXIT
curl -LsSf {UV_ARCHIVE_URL} -o "$uv_archive"
printf '%s  %s\\n' {UV_ARCHIVE_SHA256} "$uv_archive" | sha256sum --check --strict -
tar -xzf "$uv_archive" -C "$uv_root" --strip-components=1
uv_bin="$uv_root/uv"
test "$("$uv_bin" --version)" = "{UV_VERSION_OUTPUT}"
git clone --filter=blob:none https://github.com/NVIDIA/cosmos-framework.git "$source_root"
git -C "$source_root" fetch --depth 1 origin {COSMOS_FRAMEWORK_REVISION}
git -C "$source_root" checkout --detach FETCH_HEAD
test "$(git -C "$source_root" rev-parse HEAD)" = "{COSMOS_FRAMEWORK_REVISION}"
printf '%s  %s\\n' {DEPENDENCY_LOCK_SHA256} "$source_root/uv.lock" | sha256sum --check --strict -
"$uv_bin" venv --python /usr/bin/python3 "$venv_root"
"$venv_root/bin/python" -c 'import sys; assert "isaac-sim" not in sys.prefix'
VIRTUAL_ENV="$venv_root" "$uv_bin" sync --project "$source_root" --active --frozen --no-dev --all-extras --group cu128-train --group policy-server
cat > "$RUNTIME_DIR/cosmos-client-requirements.txt" <<'CLIENT'
{CLIENT_REQUIREMENTS}CLIENT
"$uv_bin" pip install --python /isaac-sim/kit/python/bin/python3 --target "$RUNTIME_DIR/cosmos_policy_client" --no-deps --only-binary :all: --require-hashes -r "$RUNTIME_DIR/cosmos-client-requirements.txt"
"$venv_root/bin/python" - "$checkpoint_root" <<'DOWNLOAD'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id={MODEL_ID!r}, revision={MODEL_REVISION!r}, local_dir=sys.argv[1], token=False)
DOWNLOAD
# The server verifies every retained runtime file before model load; the
# handshake compares its exact source/model/snapshot identity without inference.
"$venv_root/bin/python" "$RUNTIME_DIR/adp009d_policy_server_worker.py" \\
  --candidate-id "$candidate" --source-root "$source_root" --checkpoint-root "$checkpoint_root" \\
  --python "$venv_root/bin/python" --host 127.0.0.1 --port 8000 \\
  --policy-spec "$RUNTIME_DIR/adp009d_policy_execution_spec.json" \\
  --checkpoint-inventory "$RUNTIME_DIR/{SNAPSHOT_STAGED_NAME}" \\
  --log "$OUT_DIR/adp009d_policy_server.$candidate.log" \\
  --receipt "$OUT_DIR/adp009d_policy_server_receipt.$candidate.json"
'''
