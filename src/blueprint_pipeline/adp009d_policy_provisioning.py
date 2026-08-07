"""Generate the worker-side script that fetches a checkpoint and serves a policy.

Items 2 and 3 of the eval critical path are one piece of work: the checkpoint is
useless without a server and the server cannot start without the checkpoint, and
both happen on the same ephemeral GPU worker in the same container.

The topology is settled by evidence rather than preference.  The Vast transport
launches exactly one entrypoint in one image, so a sidecar container is not
available; and the shipped groot_oscar image already runs a GR00T policy server,
a client and Isaac as three concurrent processes in one Isaac Sim container with
four separate Python environments.  So: one container, two interpreters, the
policy on loopback.

Two things are enforced rather than trusted:

* **The policy environment is separate from Isaac's.**  Isaac ships its own
  CPython and pip-mutating it to satisfy a policy's dependency tree is how you
  get an Isaac that no longer starts.  A build gate in the shipped image already
  fails on exactly this.
* **JAX preallocation is disabled for pi05.**  Blueprint's standalone OpenPI
  image sets a 0.80 device-memory fraction because it owns the GPU; co-resident
  with Isaac that claims 80 percent of the card and kills Isaac as SIGABRT or
  SIGKILL -- signals no Python except clause can catch.  The fraction is claimed
  at first use, after Isaac has built its scene, so ordering does not save it.

This module emits a script and validates a receipt.  It never runs the script,
never fetches bytes, and never contacts a provider.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .adp009d_checkpoint_materialization import (
    CANDIDATE_SOURCES,
    SOURCE_PUBLIC_GCS,
    plan_checkpoint_materialization,
)
from .adp009d_policy_candidate_admission import EXPECTED_CANDIDATES, PROGRAM_ID
from .decision_evidence_contracts import canonical_digest

PROVISIONING_SCHEMA_VERSION = "adp009d_policy_provisioning.v1"

# Isaac's own interpreter, which must not be mutated to host a policy.
ISAAC_INTERPRETER = "/isaac-sim/python.sh"
# The policy environment lives beside it, never inside it.
POLICY_VENV_ROOT = "/opt/adp009d-policy-venv"
CHECKPOINT_ROOT = "/opt/adp009d-checkpoints"
POLICY_SOURCE_ROOT = "/opt/adp009d-policy-source"
# Measured on the worker: Isaac runs /isaac-sim/kit/python/bin/python3 at
# 3.12.12 under its own prefix, and /usr/bin/python3 at 3.12.3 exists
# separately.  The venv is built from the system interpreter so nothing
# resolves out of Isaac's prefix.
SYSTEM_INTERPRETER = "/usr/bin/python3"
POLICY_HOST = "127.0.0.1"
POLICY_PORT = 8000

# Measured: sigmoid-free, this is the setting that keeps JAX from taking the
# card out from under Isaac.  A smaller fraction narrows the race rather than
# removing it, so preallocation is disabled outright.
JAX_ENVIRONMENT = {
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
}

BLOCKER_UNKNOWN_CANDIDATE = "policy_provisioning_unknown_candidate"
BLOCKER_SHARED_INTERPRETER = "policy_provisioning_shares_isaac_interpreter"
BLOCKER_JAX_PREALLOCATION = "policy_provisioning_jax_preallocation_enabled"
BLOCKER_NOT_LOOPBACK = "policy_provisioning_endpoint_not_loopback"
BLOCKER_CREDENTIALS = "policy_provisioning_credentials_forwarded"


class PolicyProvisioningError(ValueError):
    """Fail-closed provisioning contract errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(e) for e in errors if str(e)}))
        super().__init__(";".join(self.errors))


def _fetch_commands(candidate_id: str) -> list[str]:
    """Credential-free fetch, chosen by where the frozen artifact actually lives."""

    plan = plan_checkpoint_materialization(candidate_id)
    expected = EXPECTED_CANDIDATES[candidate_id]
    target = f"{CHECKPOINT_ROOT}/{candidate_id}"
    if CANDIDATE_SOURCES[candidate_id] == SOURCE_PUBLIC_GCS:
        # The bucket lists and reads anonymously; -u disables credential lookup
        # so a stray token on the host cannot silently change what is fetched.
        return [
            f'mkdir -p "{target}"',
            f'gcloud storage cp -r -u "{plan["checkpoint_repository"]}/*" "{target}/"',
        ]
    repository = plan["checkpoint_repository"].removeprefix("https://huggingface.co/")
    return [
        f'mkdir -p "{target}"',
        f'"{POLICY_VENV_ROOT}/bin/python" -m huggingface_hub.commands.huggingface_cli '
        f'download "{repository}" --revision "{expected["checkpoint_revision"]}" '
        f'--local-dir "{target}"',
    ]


def _install_commands(candidate_id: str) -> list[str]:
    """Install the candidate's pinned policy source into the policy venv.

    Ordered before the checkpoint fetch on purpose: a dependency resolution
    failure should surface in seconds rather than after a 12.4 GB download.

    The source revision is the one the candidate contract froze, checked out
    detached and verified, so a moved branch cannot silently change what runs.
    """

    expected = EXPECTED_CANDIDATES[candidate_id]
    repository = str(expected["source_repository"])
    revision = str(expected["source_revision"])
    source = f"{POLICY_SOURCE_ROOT}/{candidate_id}"
    return [
        f'git clone --filter=blob:none "{repository}" "{source}"',
        f'git -C "{source}" fetch --depth 1 origin "{revision}"',
        f'git -C "{source}" checkout --detach FETCH_HEAD',
        f'test "$(git -C "{source}" rev-parse HEAD)" = "{revision}"',
        f'"{POLICY_VENV_ROOT}/bin/python" -m pip install --no-build-isolation -e "{source}"',
    ]

def build_provisioning_script(candidate_id: str) -> str:
    """Emit the worker-side provisioning script for one candidate."""

    if candidate_id not in EXPECTED_CANDIDATES:
        raise PolicyProvisioningError([f"{BLOCKER_UNKNOWN_CANDIDATE}:{candidate_id}"])

    jax_exports = "\n".join(
        f'export {name}="{value}"' for name, value in sorted(JAX_ENVIRONMENT.items())
    )
    fetch = "\n".join(_fetch_commands(candidate_id))
    install = "\n".join(_install_commands(candidate_id))
    return f"""#!/usr/bin/env bash
# ADP-009D policy provisioning for {candidate_id}.  Generated, not hand-written.
set -euo pipefail

# The policy environment is built BESIDE Isaac's interpreter, never by mutating
# it: a pip resolve against Isaac's own CPython is how you get an Isaac that no
# longer starts, and the shipped image's build gate already forbids merging them.
#
# A live run measured that this image's /usr/bin/python3 has no ensurepip, so a
# plain venv fails outright.  Install the system venv package first, and fall
# back to uv -- which needs no ensurepip and is what the shipped groot_oscar
# image already uses to build venvs beside Isaac in this container family.
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq python3-venv python3.12-venv >/dev/null 2>&1 || true

if ! "{SYSTEM_INTERPRETER}" -m venv "{POLICY_VENV_ROOT}"; then
  echo "venv unavailable; falling back to uv"
  rm -rf "{POLICY_VENV_ROOT}"
  export UV_INSTALL_DIR=/opt/adp009d-uv
  curl -LsSf https://astral.sh/uv/install.sh | sh
  "$UV_INSTALL_DIR/uv" venv --python "{SYSTEM_INTERPRETER}" "{POLICY_VENV_ROOT}"
fi

# Prove the venv is real and is not Isaac's interpreter before anything installs.
test -x "{POLICY_VENV_ROOT}/bin/python"
"{POLICY_VENV_ROOT}/bin/python" -c "import sys; assert 'isaac-sim' not in sys.prefix, sys.prefix"
"{POLICY_VENV_ROOT}/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
"{POLICY_VENV_ROOT}/bin/python" -m pip install --upgrade pip

# JAX would otherwise preallocate most of the device and take the card out from
# under Isaac as an uncatchable native abort, after the scene is already built.
{jax_exports}

# Every frozen candidate is public, so no credential is forwarded to this host.
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN || true

{install}

{fetch}

# Readiness is a completed inference round trip, not a listening socket: one
# shipped server writes "model_loaded_ready_to_serve" before it serves at all.
echo "BLUEPRINT_ADP009D_POLICY_PROVISIONED:{candidate_id}"
"""


def describe_provisioning(candidate_id: str) -> dict[str, Any]:
    """Report the provisioning facts for the run receipt."""

    if candidate_id not in EXPECTED_CANDIDATES:
        raise PolicyProvisioningError([f"{BLOCKER_UNKNOWN_CANDIDATE}:{candidate_id}"])
    plan = plan_checkpoint_materialization(candidate_id)
    receipt: dict[str, Any] = {
        "schema_version": PROVISIONING_SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "candidate_id": candidate_id,
        "topology": "shared_worker_separate_interpreter",
        "isaac_interpreter": ISAAC_INTERPRETER,
        "policy_interpreter": f"{POLICY_VENV_ROOT}/bin/python",
        "checkpoint_root": f"{CHECKPOINT_ROOT}/{candidate_id}",
        "checkpoint_source": CANDIDATE_SOURCES[candidate_id],
        "checkpoint_repository": plan["checkpoint_repository"],
        "checkpoint_revision": plan["checkpoint_revision"],
        "expected_total_bytes": plan["expected_total_bytes"],
        "endpoint_host": POLICY_HOST,
        "endpoint_port": POLICY_PORT,
        "credentials_forwarded": False,
        "jax_environment": dict(JAX_ENVIRONMENT),
        "materialize_on": "gpu_worker",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def validate_provisioning(receipt: Mapping[str, Any]) -> list[str]:
    """Refuse a provisioning that would take the GPU out from under Isaac."""

    errors: list[str] = []
    candidate_id = str(receipt.get("candidate_id", ""))
    if candidate_id not in EXPECTED_CANDIDATES:
        return [f"{BLOCKER_UNKNOWN_CANDIDATE}:{candidate_id}"]

    interpreter = str(receipt.get("policy_interpreter", ""))
    if not interpreter or interpreter == ISAAC_INTERPRETER or "isaac-sim" in interpreter:
        errors.append(BLOCKER_SHARED_INTERPRETER)

    if receipt.get("credentials_forwarded") is not False:
        errors.append(BLOCKER_CREDENTIALS)

    if str(receipt.get("endpoint_host")) not in {"127.0.0.1", "localhost", "::1"}:
        errors.append(BLOCKER_NOT_LOOPBACK)

    # Only pi05 brings JAX, but an enabled preallocation is fatal wherever it
    # appears, so the check is not conditioned on the candidate.
    environment = receipt.get("jax_environment") or {}
    preallocate = str(environment.get("XLA_PYTHON_CLIENT_PREALLOCATE", "")).lower()
    if preallocate != "false":
        errors.append(BLOCKER_JAX_PREALLOCATION)

    return sorted(set(errors))


__all__ = [
    "CHECKPOINT_ROOT",
    "ISAAC_INTERPRETER",
    "JAX_ENVIRONMENT",
    "POLICY_HOST",
    "POLICY_PORT",
    "POLICY_SOURCE_ROOT",
    "POLICY_VENV_ROOT",
    "SYSTEM_INTERPRETER",
    "PROVISIONING_SCHEMA_VERSION",
    "PolicyProvisioningError",
    "build_provisioning_script",
    "describe_provisioning",
    "validate_provisioning",
]
