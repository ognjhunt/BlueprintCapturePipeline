"""Run a real, no-scene synthetic pre-observation sandbox on a dedicated VM."""
import json
import os
import secrets
import socket
import sys
from hashlib import sha256
from pathlib import Path

from blueprint_pipeline.company_policy_sandbox_executor import (
    HttpCredentialBroker, SubprocessCommandRunner, _signed_receipt,
    execute_company_policy_sandbox_preobservation,
)
from blueprint_pipeline.company_policy_sandbox_v2 import build_company_policy_sandbox_plan

if len(sys.argv) != 3 or sys.argv[2] != "synthetic-dedicated-vm-only":
    raise SystemExit("A protected input file and synthetic-dedicated-vm-only acknowledgement are required")
if sys.platform != "linux" or os.geteuid() != 0:
    raise SystemExit("Dedicated Linux worker with root runtime control required")
input_path = Path(sys.argv[1])
if input_path.stat().st_mode & 0o077:
    raise SystemExit("Input file must be owner-only")
inputs = json.loads(input_path.read_text())
contract = inputs["contract"]
if contract["container"]["gpu_required"] or contract["container"]["serve_command"] != ["python", "/opt/policy.py"]:
    raise SystemExit("Only the Blueprint synthetic CPU policy is admitted by this rehearsal")
security_root = Path("/etc/blueprint/company-policy")
seccomp = security_root / "seccomp-v2.json"
apparmor = security_root / "apparmor-v2.profile"
addresses = sorted({row[4][0] for row in socket.getaddrinfo(inputs["registry_host"], 443, type=socket.SOCK_STREAM)})
plan = build_company_policy_sandbox_plan(
    admission_receipt=inputs["admission"], contract=contract,
    sandbox_attempt_id="blueprint-synthetic-attempt-20260928",
    pipeline_release_sha=inputs["pipeline_release_sha"], worker_identity="blueprint-policy-proof-dedicated-lima",
    runtime_class="runsc", blueprint_proxy_image=inputs["image"],
    blueprint_proxy_contract_digest="sha256:" + sha256(Path("/tmp/src/blueprint_pipeline/company_policy_proxy.py").read_bytes()).hexdigest(),
    seccomp_profile_id="blueprint-policy-seccomp-v2", seccomp_profile_path=str(seccomp),
    seccomp_profile_digest="sha256:" + sha256(seccomp.read_bytes()).hexdigest(),
    apparmor_profile_id="blueprint-policy-apparmor-v2", apparmor_profile_source_path=str(apparmor),
    apparmor_profile_digest="sha256:" + sha256(apparmor.read_bytes()).hexdigest(),
    registry_addresses=addresses, allowed_registry_hosts=[inputs["registry_host"]],
)
key = secrets.token_bytes(32)
mounts = Path("/proc/mounts").read_text()
if any(marker in mounts for marker in ("/Users/", "virtiofs", "9p")):
    raise SystemExit("Dedicated no-workspace-mount worker required")
boot = _signed_receipt({
    "schema_version": "company_policy_worker_boot_receipt.v1", "source": "trusted_company_policy_worker_bootstrap",
    "status": "dedicated_ephemeral_worker_ready", "worker_identity": plan["worker_identity"],
    "pipeline_release_sha": plan["pipeline_release_sha"], "sandbox_attempt_id": plan["sandbox_attempt_id"],
    "dedicated_ephemeral_worker": True, "scene_bytes_present": False, "observation_bytes_present": False,
    "mounted_customer_input_paths": [], "mount_inventory_sha256": sha256(mounts.encode()).hexdigest(),
}, key=key, key_id="ephemeral-synthetic-proof")
Path("/tmp/blueprint-proof-plan.json").write_text(json.dumps(plan, indent=2))
Path("/tmp/blueprint-proof-boot.json").write_text(json.dumps(boot, indent=2))
broker = HttpCredentialBroker(base_url="http://127.0.0.1:8803/api/internal/pipeline", token_file=Path("/tmp/blueprint-proof-broker-token"),
    client_id="blueprint-policy-sandbox-worker")
result = execute_company_policy_sandbox_preobservation(
    plan=plan, contract=contract, broker=broker, runner=SubprocessCommandRunner(),
    attestation_key=key, attestation_key_id="ephemeral-synthetic-proof", worker_boot_receipt=boot,
    output_path=Path("/tmp/blueprint-proof-result.json"),
)
print(json.dumps({"status": result["status"], "blockers": result.get("blockers", []),
    "cleanup_complete": result["terminal_receipt"]["cleanup_complete"]}))
raise SystemExit(0 if result["status"] == "qualified_dry_run_no_real_observation" else 1)
