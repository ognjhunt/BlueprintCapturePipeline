"""Bind offline host prerequisites to a selected, not-admitted G1 bundle."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

from . import native_g1_team_vm_bootstrap as bootstrap

HOST_PACKAGE_ROOT = "provider_runtime/host-python-package"
VM_ENTRYPOINT = "provider_runtime/run_g1_team_vm_host.sh"
POLICY_ARTIFACT_PATH = "provider_runtime/inputs/team-policy/policy.tar"
BOOTSTRAP_RESULT_FILENAME = "native_g1_team_vm_bootstrap_entrypoint.v1.json"


def host_package_sources(root: Path | None, commit: str):
    if root is None:
        raise ValueError("g1_team_bundle_host_python_package_required")
    manifest = bootstrap.verify_g1_vm_host_python(root, expected_implementation_commit=commit)
    sources = [(HOST_PACKAGE_ROOT + "/" + name, root / name) for name in
               (bootstrap.MANIFEST_NAME, *(row["filename"] for row in bootstrap.ASSETS.values()))]
    return {"relative_root": HOST_PACKAGE_ROOT, "manifest_digest": manifest["manifest_digest"]}, sources


def verify_host_package_in_bundle(manifest: Mapping[str, Any], packet: Mapping[str, Any], read) -> None:
    """Reopen embedded package identity; every asset is also a bundle artifact."""
    paired = packet["delivery_mode"] != "authenticated_endpoint"
    if not paired:
        if manifest.get("host_python_package") is not None or manifest.get("policy_artifact") is not None:
            raise ValueError("g1_team_bundle_host_package_mode_invalid")
        return
    declared = manifest.get("host_python_package")
    runtime_entries = {VM_ENTRYPOINT,
                       "provider_runtime/blueprint_pipeline/native_g1_team_vm_host.py",
                       "provider_runtime/blueprint_pipeline/native_g1_team_vm_bootstrap.py"}
    if not runtime_entries <= {row["relative_path"] for row in manifest["artifacts"]}:
        raise ValueError("g1_team_bundle_host_package_runtime_missing")
    expected = {
        "schema_version": bootstrap.SCHEMA, "status": "sealed_runtime_prerequisites",
        "implementation_commit": packet["implementation_commit"], "platform": "linux-x86_64",
        "python_abi": "cp312", "assets": bootstrap.ASSETS,
        "provider_network_install_required": False, "provider_mutation_performed": False,
        "rights_authorized": False, "gpu_runtime_qualified": False, "claim_ceiling": "development_only",
    }
    expected["manifest_digest"] = bootstrap._digest(expected)
    if declared != {"relative_root": HOST_PACKAGE_ROOT, "manifest_digest": expected["manifest_digest"]}:
        raise ValueError("g1_team_bundle_host_package_binding_invalid")
    expected_rows = {HOST_PACKAGE_ROOT + "/" + row["filename"]: row for row in bootstrap.ASSETS.values()}
    observed_rows = {row["relative_path"]: row for row in manifest["artifacts"]
                     if row["relative_path"].startswith(HOST_PACKAGE_ROOT + "/")}
    names = {*expected_rows, HOST_PACKAGE_ROOT + "/" + bootstrap.MANIFEST_NAME}
    if set(observed_rows) != names:
        raise ValueError("g1_team_bundle_host_package_inventory_invalid")
    for name, row in expected_rows.items():
        if any(observed_rows[name].get(key) != row[key] for key in ("sha256", "size_bytes")):
            raise ValueError("g1_team_bundle_host_package_asset_invalid")
    with read(HOST_PACKAGE_ROOT + "/" + bootstrap.MANIFEST_NAME) as stream:
        value = json.loads(stream.read(16385), object_pairs_hook=bootstrap._object)
    if bootstrap._canonical(value) != bootstrap._canonical(expected):
        raise ValueError("g1_team_bundle_host_package_manifest_invalid")


def verify_policy_artifact_in_bundle(manifest: Mapping[str, Any], packet: Mapping[str, Any]) -> None:
    rows = [row for row in manifest["artifacts"]
            if row["relative_path"].startswith("provider_runtime/inputs/team-policy/")]
    declared = manifest.get("policy_artifact")
    if packet["delivery_mode"] != "noncontainer_artifact":
        if declared is not None or rows:
            raise ValueError("g1_team_bundle_policy_artifact_mode_invalid")
        return
    expected_sha = packet["request"]["policy_profile"]["delivery"]["artifact_sha256"]
    if (len(rows) != 1 or rows[0]["relative_path"] != POLICY_ARTIFACT_PATH
            or rows[0]["sha256"] != expected_sha or type(rows[0]["size_bytes"]) is not int
            or not 0 < rows[0]["size_bytes"] <= 16 * 1024**3
            or declared != rows[0]):
        raise ValueError("g1_team_bundle_policy_artifact_binding_invalid")


def vm_host_entrypoint(commit: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("g1_team_bundle_vm_entrypoint_commit_invalid")
    # The simulator still uses the existing guest entrypoint. No network install,
    # provider mutation, image pull, credential or authority renewal occurs here.
    template = r'''#!/usr/bin/env bash
set -u
umask 077
RUNTIME_DIR="$(cd "$(dirname "$0")" && pwd)"
BUNDLE_DIR="$(cd "$RUNTIME_DIR/.." && pwd)"
OUT_DIR="$BUNDLE_DIR/runtime_output"
HOST_PYTHON_ROOT="$BUNDLE_DIR/host-python-runtime"
mkdir -p "$OUT_DIR"
phase=host-python-provisioning
digest="$(python3 -I -B -S - "$RUNTIME_DIR/native_g1_team_provider_manifest.json" <<'PY'
import json,re,sys
value=json.load(open(sys.argv[1]))['manifest_digest']
assert isinstance(value,str) and re.fullmatch(r'sha256:[0-9a-f]{64}',value)
print(value)
PY
)"
rc=$?
if [ "$rc" -eq 0 ]; then
  python3 -I -B -S "$RUNTIME_DIR/blueprint_pipeline/native_g1_team_vm_bootstrap.py" \
    --package-root "$RUNTIME_DIR/host-python-package" --implementation-commit __COMMIT__ \
    --destination-root "$HOST_PYTHON_ROOT" --provider-runtime-root "$RUNTIME_DIR" \
    --provider-manifest-digest "$digest" >"$OUT_DIR/host_python_provisioning.private.log" 2>&1
  rc=$?
fi
if [ "$rc" -eq 0 ]; then
  phase=vm-host
  "$HOST_PYTHON_ROOT/python/bin/python3.12" -I -B - "$RUNTIME_DIR" "$OUT_DIR" <<'PY' >"$OUT_DIR/vm_host.private.log" 2>&1
import sys
runtime,output=sys.argv[1:]
sys.path.insert(0,runtime)
from blueprint_pipeline.native_g1_team_vm_host import main
raise SystemExit(main(['--runtime-root',runtime,'--output-dir',output]))
PY
  rc=$?
fi
python3 -I -B -S - "$OUT_DIR" "$phase" "$rc" <<'PY'
import hashlib,json,sys
from pathlib import Path
value={'schema_version':'native_g1_team_vm_bootstrap_entrypoint.v1',
 'status':'host_exited' if int(sys.argv[3])==0 else 'blocked',
 'stage_reached':sys.argv[2],'runner_exit_code':int(sys.argv[3]),
 'implementation_commit':'__COMMIT__','provider_mutation_performed':False,
 'gpu_runtime_qualified':False,'claim_ceiling':'development_only'}
encoded=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
value['receipt_digest']='sha256:'+hashlib.sha256(encoded.encode()).hexdigest()
with (Path(sys.argv[1])/'native_g1_team_vm_bootstrap_entrypoint.v1.json').open('x') as stream:
 json.dump(value,stream,sort_keys=True,allow_nan=False);stream.write('\n')
PY
receipt_rc=$?
if [ "$receipt_rc" -ne 0 ]; then rc=2; fi
exit "$rc"
'''
    return template.replace("__COMMIT__", commit)
