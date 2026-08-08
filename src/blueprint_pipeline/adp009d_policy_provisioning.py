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
from .adp009d_gated_backbone import MODEL_ID as GATED_BACKBONE_MODEL_ID
from .adp009d_gated_backbone import REVISION as GATED_BACKBONE_REVISION
from .adp009d_policy_candidate_admission import EXPECTED_CANDIDATES, PROGRAM_ID
from .decision_evidence_contracts import canonical_digest

PROVISIONING_SCHEMA_VERSION = "adp009d_policy_provisioning.v1"

# Isaac's own interpreter, which must not be mutated to host a policy.
ISAAC_INTERPRETER = "/isaac-sim/python.sh"
# The policy environment lives beside it, never inside it.
POLICY_VENV_PARENT = "/opt/adp009d-policy-venv"


def policy_venv_root(candidate_id: str) -> str:
    """The venv for one candidate.

    Per candidate, not shared.  A live two-policy run had the second candidate
    fail outright because the first had already created the shared path -- and
    even had creation succeeded, openpi and GR00T cannot share an environment:
    one pins JAX and its own torch, the other a different torch, so whichever
    installed second would have silently broken the first.
    """

    return f"{POLICY_VENV_PARENT}/{candidate_id}"
CHECKPOINT_ROOT = "/opt/adp009d-checkpoints"
UV_ROOT = "/opt/adp009d-uv"
POLICY_SOURCE_ROOT = "/opt/adp009d-policy-source"
# Measured on the worker: Isaac runs /isaac-sim/kit/python/bin/python3 at
# 3.12.12 under its own prefix, and /usr/bin/python3 at 3.12.3 exists
# separately.  The venv is built from the system interpreter so nothing
# resolves out of Isaac's prefix.
SYSTEM_INTERPRETER = "/usr/bin/python3"
POLICY_HOST = "127.0.0.1"
POLICY_PORT = 8000
GATED_BACKBONE_AUTH_ENV = "BLUEPRINT_ADP009D_GATED_BACKBONE_AUTHORIZED"
GATED_BACKBONE_HF_HOME = "/opt/adp009d-hf-cache"
GATED_BACKBONE_HUB_CACHE = f"{GATED_BACKBONE_HF_HOME}/hub"

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
        # Plain HTTPS via the GCS JSON API, not gcloud: the Isaac container is
        # not a Google Cloud image and cannot be assumed to carry the SDK.  The
        # bucket was verified to list and read with no Authorization header, so
        # this needs no credential and no third-party package.
        return [
            f'mkdir -p "{target}"',
            f'/isaac-sim/python.sh "$RUNTIME_DIR/adp009d_checkpoint_fetch_worker.py" '
            f'"{plan["checkpoint_repository"]}" "{target}" '
            f'"$OUT_DIR/adp009d_checkpoint_fetch_receipt.json"',
        ]
    repository = plan["checkpoint_repository"].removeprefix("https://huggingface.co/")
    return [
        f'mkdir -p "{target}"',
        f'"{policy_venv_root(candidate_id)}/bin/python" -m huggingface_hub.commands.huggingface_cli '
        f'download "{repository}" --revision "{expected["checkpoint_revision"]}" '
        f'--local-dir "{target}"',
    ]


# The client library has to be importable from *Isaac*, not from the policy
# venv.  The separate-interpreter design keeps the server's dependency tree
# away from Isaac -- which is the whole point, since JAX or a mismatched torch
# will take the card out from under it -- but the episode runs inside Isaac and
# has to speak to that server.  A live run reached the episode, with the server
# ready and the gripper measured, and died on ModuleNotFoundError:
# openpi_client.
#
# Only the thin client goes in.  openpi ships it as a separate subpackage for
# exactly this purpose, at the same pinned revision as the server, so client
# and server cannot drift apart.
CANDIDATE_ISAAC_CLIENT_SUBPACKAGE = {
    "pi05_droid": "packages/openpi-client",
    "cosmos3_edge_policy_droid": "packages/openpi-client",
}


def _isaac_client_commands(candidate_id: str) -> list[str]:
    """Install this candidate's client into Isaac's interpreter."""

    source = f"{POLICY_SOURCE_ROOT}/{candidate_id}"
    subpackage = CANDIDATE_ISAAC_CLIENT_SUBPACKAGE.get(candidate_id)
    if subpackage is None:
        # GR00T's client lives inside its main package rather than a thin one,
        # so it is installed without dependencies: the episode needs the ZMQ
        # client class, and pulling GR00T's full tree into Isaac would risk
        # the torch conflict this design exists to avoid.
        return [
            f'"{ISAAC_INTERPRETER}" -m pip install --no-deps -e "{source}"',
            # Exact thin-client dependencies from the frozen GR00T pyproject.
            # ``server_client.py`` imports msgpack_numpy at module import, so
            # omitting it fails only after the server and Isaac have both paid
            # their startup cost.
            f'"{ISAAC_INTERPRETER}" -m pip install '
            '"pyzmq==27.0.1" "msgpack==1.1.0" "msgpack-numpy==0.4.8"',
        ]
    return [
        f'"{ISAAC_INTERPRETER}" -m pip install -e "{source}/{subpackage}"',
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
        # Installed with uv, into the venv named explicitly rather than via an
        # activated shell, so the target cannot drift.  Build isolation stays
        # on: disabling it once left pip unable to import hatchling.build, the
        # backend openpi's pyproject declares.
        f'VIRTUAL_ENV="{policy_venv_root(candidate_id)}" "$UV" pip install -e "{source}"',
        *_isaac_client_commands(candidate_id),
    ]

def build_provisioning_script(candidate_id: str) -> str:
    """Emit the worker-side provisioning script for one candidate."""

    if candidate_id not in EXPECTED_CANDIDATES:
        raise PolicyProvisioningError([f"{BLOCKER_UNKNOWN_CANDIDATE}:{candidate_id}"])

    uv_root = UV_ROOT
    venv_root = policy_venv_root(candidate_id)
    jax_exports = "\n".join(
        f'export {name}="{value}"' for name, value in sorted(JAX_ENVIRONMENT.items())
    )
    fetch = "\n".join(_fetch_commands(candidate_id))
    install = "\n".join(_install_commands(candidate_id))
    identity = ""
    server_identity_arg = ""
    credential_contract = (
        "unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN || true"
    )
    gated_backbone = ""
    if candidate_id == "groot_n17_droid":
        credential_contract = f'''case "${{{GATED_BACKBONE_AUTH_ENV}:-false}}" in
  1|true|TRUE|yes|YES)
    if [ -z "${{HF_TOKEN:-${{HUGGING_FACE_HUB_TOKEN:-${{HUGGINGFACE_HUB_TOKEN:-}}}}}}" ]; then
      echo "BLUEPRINT_ADP009D_BLOCKER:adp009d_gated_backbone_token_missing"
      exit 86
    fi
    export HF_TOKEN="${{HF_TOKEN:-${{HUGGING_FACE_HUB_TOKEN:-${{HUGGINGFACE_HUB_TOKEN}}}}}}"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    ;;
  *)
    unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN || true
    echo "BLUEPRINT_ADP009D_BLOCKER:adp009d_gated_backbone_authority_missing"
    exit 86
    ;;
esac'''
        identity = f'''# Bind the materialized source, checkpoint bytes, and policy environment before
# the endpoint is allowed to count as this frozen candidate.  Continue only far
# enough to let the server worker translate a blocked identity receipt into its
# own typed readiness receipt; it will never launch a server for invalid bytes.
"{venv_root}/bin/python" "$RUNTIME_DIR/adp009d_groot_worker_identity.py" \\
  --source-root "{POLICY_SOURCE_ROOT}/{candidate_id}" \\
  --checkpoint-root "{CHECKPOINT_ROOT}/{candidate_id}" \\
  --python "{venv_root}/bin/python" \\
  --output "$OUT_DIR/adp009d_groot_worker_identity.{candidate_id}.json" || true
'''
        server_identity_arg = (
            " \\\n  --worker-identity-receipt "
            f'"$OUT_DIR/adp009d_groot_worker_identity.{candidate_id}.json"'
        )
        gated_backbone = f'''# NVIDIA's exact GR00T checkpoint names a separately gated Cosmos backbone.
# Download only the frozen revision under explicit authority, verify every Git/LFS
# object, bind unversioned lookup to that revision, then remove the credential and
# force offline loading before the server starts.
echo "BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:provision_{candidate_id}_gated_backbone:started"
export HF_HOME="{GATED_BACKBONE_HF_HOME}"
"{venv_root}/bin/python" -m huggingface_hub.commands.huggingface_cli download \
  "{GATED_BACKBONE_MODEL_ID}" \
  --revision "{GATED_BACKBONE_REVISION}" \
  --cache-dir "{GATED_BACKBONE_HUB_CACHE}"
"{venv_root}/bin/python" "$RUNTIME_DIR/adp009d_gated_backbone.py" \
  --cache-dir "{GATED_BACKBONE_HUB_CACHE}" \
  --output "$OUT_DIR/adp009d_gated_backbone_identity.{candidate_id}.json"
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN || true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:provision_{candidate_id}_gated_backbone:completed"
'''
    return f"""#!/usr/bin/env bash
# ADP-009D policy provisioning for {candidate_id}.  Generated, not hand-written.
set -euo pipefail

# The policy environment is built BESIDE Isaac's interpreter, never by mutating
# it: a pip resolve against Isaac's own CPython is how you get an Isaac that no
# longer starts, and the shipped image's build gate already forbids merging them.
# uv is the installer, not a fallback.  Two measured reasons: this image's
# /usr/bin/python3 has no ensurepip, so a plain venv cannot be created; and pip
# cannot resolve openpi at all, failing with "resolution-too-deep" after
# backtracking through tensorstore releases.  openpi is itself a uv project, so
# using uv means installing it the way it is packaged rather than fighting it.
# Native build dependencies.  A live run resolved openpi cleanly with uv and
# then failed compiling evdev, which needs linux/input.h: the chain is
# openpi -> lerobot -> pynput -> evdev, pulled in for input-device handling that
# inference never uses but that the dependency graph still requires to build.
# Installed as one set rather than one package per run: each round trip costs a
# paid GPU run, and these are the standard requirements for building any C
# extension.  linux-libc-dev supplies linux/input.h, python3-dev supplies
# Python.h, build-essential the compiler, pkg-config the usual discovery.
# Report every missing prerequisite at once.  Three consecutive paid runs each
# discovered exactly one -- ensurepip, then linux/input.h, then Python.h --
# because the script stopped at the first.  Run before the install, so the
# complete set is known, and again after, so the apt step is proven to have
# fixed what it claimed.
/isaac-sim/python.sh "$RUNTIME_DIR/adp009d_provisioning_preflight.py" \
  "$OUT_DIR/adp009d_provisioning_preflight_before.json" || true

apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq \
  linux-libc-dev build-essential pkg-config \
  python3-dev python3.12-dev >/dev/null 2>&1 || true

/isaac-sim/python.sh "$RUNTIME_DIR/adp009d_provisioning_preflight.py" \
  "$OUT_DIR/adp009d_provisioning_preflight_after.json" || true

export UV_INSTALL_DIR={uv_root}
curl -LsSf https://astral.sh/uv/install.sh | sh
UV="$UV_INSTALL_DIR/uv"
test -x "$UV"

echo "BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:provision_{candidate_id}_venv:started"
"$UV" venv --python "{SYSTEM_INTERPRETER}" "{venv_root}"
echo "BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:provision_{candidate_id}_venv:completed"

# Prove the venv is real and is not Isaac's interpreter before anything installs.
test -x "{venv_root}/bin/python"
"{venv_root}/bin/python" -c "import sys; assert 'isaac-sim' not in sys.prefix, sys.prefix"

# JAX would otherwise preallocate most of the device and take the card out from
# under Isaac as an uncatchable native abort, after the scene is already built.
{jax_exports}

# Public candidates strip credentials.  GR00T N1.7 may retain a Hugging Face
# credential only behind the explicit gated-backbone authority contract below;
# the credential is removed before policy-server start.
{credential_contract}

echo "BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:provision_{candidate_id}_install:started"
{install}
echo "BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:provision_{candidate_id}_install:completed"

# The long pole: pi05's checkpoint alone is 12.4 GB.  Marked at both ends so
# the watchdog sees progress and a stall is attributable to the fetch rather
# than to provisioning generally.
echo "BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:provision_{candidate_id}_checkpoint:started"
{fetch}
echo "BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:provision_{candidate_id}_checkpoint:completed"

{identity}
{gated_backbone}
# Readiness is a completed inference round trip, not a listening socket: one
# shipped server writes "model_loaded_ready_to_serve" before it serves at all,
# and loading 12.4 GB of weights takes far longer than binding a port.  The
# worker starts the server, waits for a real inference returning a well-formed
# chunk, and leaves it running for the episode.
echo "BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:provision_{candidate_id}_server:started"
"{venv_root}/bin/python" "$RUNTIME_DIR/adp009d_policy_server_worker.py" \
  --candidate-id "{candidate_id}" \
  --source-root "{POLICY_SOURCE_ROOT}/{candidate_id}" \
  --checkpoint-root "{CHECKPOINT_ROOT}/{candidate_id}" \
  --python "{venv_root}/bin/python" \
  --host {POLICY_HOST} \
  --log "$OUT_DIR/adp009d_policy_server.{candidate_id}.log" \
  --receipt "$OUT_DIR/adp009d_policy_server_receipt.{candidate_id}.json"{server_identity_arg}

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
        "policy_interpreter": f"{policy_venv_root(candidate_id)}/bin/python",
        "checkpoint_root": f"{CHECKPOINT_ROOT}/{candidate_id}",
        "checkpoint_source": CANDIDATE_SOURCES[candidate_id],
        "checkpoint_repository": plan["checkpoint_repository"],
        "checkpoint_revision": plan["checkpoint_revision"],
        "expected_total_bytes": plan["expected_total_bytes"],
        "endpoint_host": POLICY_HOST,
        "endpoint_port": POLICY_PORT,
        "credentials_forwarded": False,
        "credentials_retained_for_server": False,
        "gated_backbone": (
            {
                "model_id": GATED_BACKBONE_MODEL_ID,
                "revision": GATED_BACKBONE_REVISION,
                "authorization_mode": "explicit_runtime_opt_in",
                "authorization_env": GATED_BACKBONE_AUTH_ENV,
                "offline_after_materialization": True,
            }
            if candidate_id == "groot_n17_droid"
            else None
        ),
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
    "GATED_BACKBONE_AUTH_ENV",
    "GATED_BACKBONE_HF_HOME",
    "GATED_BACKBONE_HUB_CACHE",
    "POLICY_HOST",
    "POLICY_PORT",
    "POLICY_SOURCE_ROOT",
    "UV_ROOT",
    "POLICY_VENV_PARENT",
    "policy_venv_root",
    "SYSTEM_INTERPRETER",
    "PROVISIONING_SCHEMA_VERSION",
    "PolicyProvisioningError",
    "build_provisioning_script",
    "describe_provisioning",
    "validate_provisioning",
]
