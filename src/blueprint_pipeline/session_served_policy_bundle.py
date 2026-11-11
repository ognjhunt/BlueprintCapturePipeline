"""What a policy canary bundle carries for a session-served policy, control plane only.

The provisioning script (the native candidate's, ending with the contract
server), the entrypoint lines that make the session's private token, and the
manifest record. The modules that run on the GPU are ``SESSION_RUNTIME_MODULES``.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Shipped in every policy canary bundle beside the team clients (``native_task_arena_policy_canary_bundle``).
SESSION_RUNTIME_MODULES = ("session_policy_client.py", "session_served_policy.py", "openpi_droid_contract_server.py",
                           "session_policy_server_start.py")


def provisioning_script(spec: Mapping[str, Any]) -> str:
    """The native candidate's provisioning, ending with this contract server in place of its own."""

    from .adp009d_policy_provisioning import (
        CHECKPOINT_INVENTORY_STAGED_NAME, CHECKPOINT_ROOT, POLICY_EXECUTION_SPEC_STAGED_NAME, build_provisioning_script,
        policy_venv_root,
    )
    from .session_served_policy import validate_session_served_spec

    spec = validate_session_served_spec(spec)
    native = spec["session_policy_server"]["native_candidate_id"]
    candidate = spec["candidate_id"]
    python = f"{policy_venv_root(native)}/bin/python"
    block = f'''echo "BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:provision_{candidate}_contract_server:started"
PYTHONPATH="$RUNTIME_DIR" PYTHONDONTWRITEBYTECODE=1 "{python}" -m blueprint_pipeline.session_policy_server_start \\
  --python "{python}" --runtime "$RUNTIME_DIR" --output "$OUT_DIR" \\
  --execution-spec "$RUNTIME_DIR/{POLICY_EXECUTION_SPEC_STAGED_NAME}" \\
  --checkpoint-dir "{CHECKPOINT_ROOT}/{native}" \\
  --checkpoint-inventory "$RUNTIME_DIR/{CHECKPOINT_INVENTORY_STAGED_NAME}"
echo "BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:provision_{candidate}_contract_server:completed"'''
    return build_provisioning_script(native, server_block=block)


def entrypoint_setup() -> str:
    """Entrypoint lines: a private per-session token file outside every uploaded directory."""

    from .session_served_policy import TOKEN_ENV

    return f'''SESSION_POLICY_DIR="$(mktemp -d /tmp/blueprint-session-policy.XXXXXX)"
chmod 700 "$SESSION_POLICY_DIR"
(umask 077 && /isaac-sim/python.sh -c 'import secrets; print(secrets.token_hex(32))' >"$SESSION_POLICY_DIR/token")
token_rc=$?
if [ $token_rc -ne 0 ] || [ ! -s "$SESSION_POLICY_DIR/token" ]; then
  write_fallback_result session_policy_token_unavailable 86
  exit 86
fi
export {TOKEN_ENV}="$SESSION_POLICY_DIR/token"'''


def bundle_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    """What the sealed provider manifest records about the session-served candidate."""

    from .session_served_policy import validate_session_served_spec

    block = validate_session_served_spec(spec)["session_policy_server"]
    return {"server": block["server"], "module": block["module"], "native_candidate_id": block["native_candidate_id"],
            "port": block["port"], "startup_timeout_seconds": block["startup_timeout_seconds"],
            "token_env": block["token_env"], "token_origin": "generated_on_session", "outbound_calls": "none"}


__all__ = ["SESSION_RUNTIME_MODULES", "bundle_contract", "entrypoint_setup", "provisioning_script"]
