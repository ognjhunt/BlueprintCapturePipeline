"""Fail-closed admission contract for company-supplied policy containers.

The ADP-009D policy-diagnostic lane runs exactly the frozen candidates
enumerated in ``adp009d_policy_candidate_admission.EXPECTED_CANDIDATES``, each
with a hand-built provisioning branch.  External robot companies cannot be a
provisioning branch each: they supply their *own* policy as a container image
that serves on loopback on the rented GPU worker, and the only way that scales
without weakening the lane is to make everything the frozen candidates get from
code into declared, validated data:

* **Rights are declared or the contract does not exist.**  The frozen
  candidates carry a sealed ``adp009d_candidate_policy_rights.v1`` receipt;
  a company policy carries the same class of facts (license, provenance,
  provider-use and redistribution status) inline, and any missing field
  refuses.  Unrecorded rights fail closed -- the program rule, not a style
  preference.
* **The image is digest-pinned.**  A tag is a moving branch with a registry
  attached: the same contract admitted twice could run two different policies
  and the receipts would not show it.  Only ``<repo>@sha256:<64 hex>`` is
  admissible, mirroring how candidate source revisions are fetched detached
  and re-verified.
* **The endpoint is not declarable.**  Every policy server in this lane serves
  on loopback (``groot_n17_wire_client.LOOPBACK_HOSTS``,
  ``adp009d_policy_provisioning.POLICY_HOST``); a company contract that could
  name a host would turn the paid worker into a client of arbitrary external
  infrastructure and move the policy query off the machine the receipts
  describe.  The validator *injects* the loopback endpoint and refuses any
  input that tries to declare one.
* **Per-channel action envelopes are declared.**  The DROID gripper taught the
  distinction the hard way (see ``DROID_GRIPPER_RAW_ACCEPTED_BOUNDS`` in
  ``adp009d_droid_action_execution``): a channel has a command interval the
  runtime executes, a wider raw envelope the server may legitimately return,
  and executed semantics explaining the gap.  Company policies declare all
  three per channel so the generalized bounds validator can police the raw
  envelope and report -- not refuse -- command-interval overshoot.

This module is pure validation: it never pulls an image, opens a socket, or
reads a secret.  The provisioning seam that turns a validated contract into
worker-side shell commands lives in ``adp009d_policy_provisioning``.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

try:  # flat provider-bundle layout
    from decision_evidence_contracts import canonical_digest
except ModuleNotFoundError:  # repository package
    from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "company_policy_container_contract.v1"

# The only admissible claim ceiling.  A company-supplied policy has, by
# construction, no sealed prospective adjudication behind it, so nothing it
# produces may climb the claim ladder past development evidence.
CLAIM_CEILING_DEVELOPMENT_ONLY = "development_only"

# The handshake kinds the episode side already knows how to speak: openpi's
# websocket policy protocol, GR00T-style ZeroMQ/MessagePack, and a plain HTTP
# JSON contract for companies that serve neither.  Anything else would need a
# new episode client, which is code, not data -- so it refuses here.
HANDSHAKE_KIND_HTTP_OPENPI = "http_openpi"
HANDSHAKE_KIND_ZMQ_MSGPACK = "zmq_msgpack"
HANDSHAKE_KIND_HTTP_JSON_V1 = "http_json_v1"
HANDSHAKE_KINDS = frozenset(
    {
        HANDSHAKE_KIND_HTTP_OPENPI,
        HANDSHAKE_KIND_ZMQ_MSGPACK,
        HANDSHAKE_KIND_HTTP_JSON_V1,
    }
)

CHANNEL_KIND_BOUNDED_CONTINUOUS = "bounded_continuous"
CHANNEL_KIND_THRESHOLD_SCALAR = "threshold_scalar"
CHANNEL_KINDS = frozenset(
    {CHANNEL_KIND_BOUNDED_CONTINUOUS, CHANNEL_KIND_THRESHOLD_SCALAR}
)

# Injected, never declared.  Matches the loopback security doctrine enforced by
# the frozen-candidate provisioning validator and the GR00T wire client.
LOOPBACK_HOST = "127.0.0.1"

BLOCKER_CONTRACT_INVALID = "company_policy_contract_invalid"
BLOCKER_IMAGE_NOT_DIGEST_PINNED = "company_policy_container_image_not_digest_pinned"
BLOCKER_REMOTE_ENDPOINT_FORBIDDEN = "company_policy_remote_endpoint_forbidden"
BLOCKER_RAW_BOUNDS_NARROWER = (
    "company_policy_channel_raw_bounds_narrower_than_command"
)

# Identifiers that end up in shell commands, docker names, and receipt keys.
# Lowercase snake only: ``company-policy-<policy_id>`` must be a valid docker
# container name, and the id appears verbatim inside double-quoted shell
# strings, so the character class deliberately excludes anything the shell or
# docker could interpret.
_IDENTIFIER = re.compile(r"^[a-z0-9_]{1,128}$")
_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
# ``<repo>@sha256:<64 hex>``.  The repository part follows docker's lowercase
# naming (registry host, optional port, path segments, optional tag before the
# digest).  The class excludes whitespace, quotes, ``$`` and backticks, so an
# admitted image reference is safe to interpolate into a double-quoted shell
# word without further escaping.
_IMAGE_DIGEST_PINNED = re.compile(r"^[a-z0-9][a-z0-9._/:-]*@sha256:[0-9a-f]{64}$")
# Bare filenames only.  These resolve later against the canonical worker
# secrets directory; a separator or dot-segment would let a contract read
# arbitrary host paths through the bind mount, and a colon would corrupt the
# ``-v host:container:ro`` docker syntax.  The allowlist forbids all of that
# structurally instead of blocklisting.
_CREDENTIAL_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# Any of these keys anywhere near the endpoint is an attempt to declare a
# host.  The loopback endpoint is injected by the validator; the only endpoint
# value an *input* may carry is the exact injected form (so a normalized
# contract re-validates as a fixed point), and everything else refuses.
_HOST_DECLARING_KEYS = frozenset({"host", "hostname", "endpoint", "endpoint_host"})

# Unknown fields refuse rather than pass through: the normalized output (and
# therefore the contract digest) contains exactly the known fields, so a field
# this validator does not understand would otherwise be silently dropped from
# the sealed identity while the company believes it was admitted.
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "policy_id",
        "company_id",
        "display_name",
        "checkpoint_identity",
        "claim_ceiling",
        "rights",
        "container",
        "observation_schema",
        "action_schema",
        "endpoint",
        "contract_digest",
    }
)
_CHECKPOINT_FIELDS = frozenset({"repository", "revision", "inventory_digest"})
_RIGHTS_FIELDS = frozenset(
    {
        "license",
        "rights_provenance",
        "provider_use_status",
        "redistribution_status",
        "rights_ready",
    }
)
_CONTAINER_FIELDS = frozenset(
    {
        "image",
        "serve_command",
        "port",
        "handshake_kind",
        "credential_files",
        "gpu_required",
    }
)
_OBSERVATION_FIELDS = frozenset({"cameras", "state_keys"})
_CAMERA_FIELDS = frozenset({"name", "width", "height"})
_ACTION_FIELDS = frozenset({"action_space_id", "chunk_rows", "channels"})
_CHANNEL_FIELDS = frozenset(
    {"name", "kind", "command_interval", "raw_accepted_bounds", "executed_semantics"}
)


class CompanyPolicyContractError(ValueError):
    """Fail-closed contract errors with stable, sorted blocker identifiers."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(e) for e in errors if str(e)}))
        super().__init__(";".join(self.errors))


def _string(value: Any) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def _positive_int(value: Any) -> int | None:
    # bool is an int subclass; ``True`` must not pass as ``1``.
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return int(value)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _interval(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    lower = _finite_number(value[0])
    upper = _finite_number(value[1])
    if lower is None or upper is None:
        return None
    return lower, upper


def _unknown_fields(payload: Mapping[str, Any], known: frozenset[str], path: str) -> list[str]:
    return [
        f"{BLOCKER_CONTRACT_INVALID}:unknown_field:{path}{key}"
        for key in sorted(str(key) for key in payload)
        if key not in known
    ]


def _validate_channels(channels: Any, errors: list[str]) -> list[dict[str, Any]]:
    """Validate the per-channel action contracts (the generalized envelope).

    Each channel mirrors what the DROID gripper hardcodes today: a command
    interval the runtime executes, a raw accepted envelope the server may
    return, and the executed semantics explaining why the two differ.  A raw
    envelope narrower than the command interval is self-contradictory -- the
    runtime would execute values the validator refused -- so it is a named
    blocker rather than a generic invalid-field error.
    """

    normalized: list[dict[str, Any]] = []
    if not isinstance(channels, list) or not channels:
        errors.append(f"{BLOCKER_CONTRACT_INVALID}:action_channels_empty")
        return normalized
    seen_names: set[str] = set()
    for index, channel in enumerate(channels):
        label = f"channel[{index}]"
        if not isinstance(channel, Mapping):
            errors.append(f"{BLOCKER_CONTRACT_INVALID}:action_channel_not_mapping:{label}")
            continue
        errors.extend(_unknown_fields(channel, _CHANNEL_FIELDS, f"action_schema.{label}."))
        name = _string(channel.get("name"))
        if not _IDENTIFIER.match(name):
            errors.append(f"{BLOCKER_CONTRACT_INVALID}:action_channel_name:{label}")
            continue
        if name in seen_names:
            errors.append(f"{BLOCKER_CONTRACT_INVALID}:action_channel_duplicate:{name}")
            continue
        seen_names.add(name)
        kind = _string(channel.get("kind"))
        if kind not in CHANNEL_KINDS:
            errors.append(f"{BLOCKER_CONTRACT_INVALID}:action_channel_kind:{name}")
        command = _interval(channel.get("command_interval"))
        raw = _interval(channel.get("raw_accepted_bounds"))
        executed = _string(channel.get("executed_semantics"))
        if command is None or command[0] >= command[1]:
            errors.append(
                f"{BLOCKER_CONTRACT_INVALID}:action_channel_command_interval:{name}"
            )
        if raw is None:
            errors.append(
                f"{BLOCKER_CONTRACT_INVALID}:action_channel_raw_accepted_bounds:{name}"
            )
        if not executed:
            errors.append(
                f"{BLOCKER_CONTRACT_INVALID}:action_channel_executed_semantics:{name}"
            )
        if command is not None and raw is not None and (
            raw[0] > command[0] or raw[1] < command[1]
        ):
            errors.append(
                f"{BLOCKER_RAW_BOUNDS_NARROWER}:{name}:"
                f"raw=[{raw[0]},{raw[1]}]:command=[{command[0]},{command[1]}]"
            )
        if command is None or raw is None or not executed or kind not in CHANNEL_KINDS:
            continue
        normalized.append(
            {
                "name": name,
                "kind": kind,
                "command_interval": [command[0], command[1]],
                "raw_accepted_bounds": [raw[0], raw[1]],
                "executed_semantics": executed,
            }
        )
    return normalized


def validate_company_policy_container_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one company-supplied policy container contract.

    Returns the normalized contract with the loopback endpoint injected and a
    canonical ``contract_digest`` bound over it.  Refuses -- with every blocker
    reported at once, never just the first -- on any missing, malformed, or
    forbidden field.  The normalized output re-validates to itself, so callers
    downstream of admission can (and do) re-run this validator instead of
    trusting that a mapping was validated earlier.
    """

    if not isinstance(value, Mapping):
        raise CompanyPolicyContractError([f"{BLOCKER_CONTRACT_INVALID}:not_a_mapping"])
    try:
        # Round-tripping through JSON refuses NaN/Infinity and non-JSON types
        # up front, and detaches the payload from caller-held mutable state.
        payload = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise CompanyPolicyContractError(
            [f"{BLOCKER_CONTRACT_INVALID}:not_json_serializable"]
        ) from exc

    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"{BLOCKER_CONTRACT_INVALID}:schema_version")
    errors.extend(_unknown_fields(payload, _TOP_LEVEL_FIELDS, ""))

    # --- endpoint doctrine: loopback is injected, never declared -------------
    for key in sorted(_HOST_DECLARING_KEYS & set(payload)):
        if key != "endpoint":
            errors.append(f"{BLOCKER_REMOTE_ENDPOINT_FORBIDDEN}:{key}")

    policy_id = _string(payload.get("policy_id"))
    if not _IDENTIFIER.match(policy_id):
        errors.append(f"{BLOCKER_CONTRACT_INVALID}:policy_id")
    company_id = _string(payload.get("company_id"))
    if not _IDENTIFIER.match(company_id):
        errors.append(f"{BLOCKER_CONTRACT_INVALID}:company_id")
    display_name = _string(payload.get("display_name"))
    if not display_name:
        errors.append(f"{BLOCKER_CONTRACT_INVALID}:display_name")

    # --- checkpoint identity -------------------------------------------------
    checkpoint = payload.get("checkpoint_identity")
    checkpoint_normalized: dict[str, Any] = {}
    if not isinstance(checkpoint, Mapping):
        errors.append(f"{BLOCKER_CONTRACT_INVALID}:checkpoint_identity")
    else:
        errors.extend(
            _unknown_fields(checkpoint, _CHECKPOINT_FIELDS, "checkpoint_identity.")
        )
        repository = _string(checkpoint.get("repository"))
        revision = _string(checkpoint.get("revision"))
        if not repository:
            errors.append(f"{BLOCKER_CONTRACT_INVALID}:checkpoint_repository")
        if not revision:
            errors.append(f"{BLOCKER_CONTRACT_INVALID}:checkpoint_revision")
        checkpoint_normalized = {"repository": repository, "revision": revision}
        if "inventory_digest" in checkpoint:
            inventory_digest = _string(checkpoint.get("inventory_digest"))
            if not _SHA256_DIGEST.match(inventory_digest):
                errors.append(
                    f"{BLOCKER_CONTRACT_INVALID}:checkpoint_inventory_digest"
                )
            checkpoint_normalized["inventory_digest"] = inventory_digest

    # --- claim ceiling -------------------------------------------------------
    if payload.get("claim_ceiling") != CLAIM_CEILING_DEVELOPMENT_ONLY:
        errors.append(
            f"{BLOCKER_CONTRACT_INVALID}:claim_ceiling_must_be_development_only"
        )

    # --- rights: any missing field refuses (fail closed on rights) -----------
    rights = payload.get("rights")
    rights_normalized: dict[str, Any] = {}
    if not isinstance(rights, Mapping):
        errors.append(f"{BLOCKER_CONTRACT_INVALID}:rights")
    else:
        errors.extend(_unknown_fields(rights, _RIGHTS_FIELDS, "rights."))
        for field in (
            "license",
            "rights_provenance",
            "provider_use_status",
            "redistribution_status",
        ):
            text = _string(rights.get(field))
            if not text:
                errors.append(f"{BLOCKER_CONTRACT_INVALID}:rights_{field}")
            rights_normalized[field] = text
        if rights.get("rights_ready") is not True:
            errors.append(f"{BLOCKER_CONTRACT_INVALID}:rights_rights_ready")
        rights_normalized["rights_ready"] = True

    # --- container -----------------------------------------------------------
    container = payload.get("container")
    container_normalized: dict[str, Any] = {}
    port: int | None = None
    if not isinstance(container, Mapping):
        errors.append(f"{BLOCKER_CONTRACT_INVALID}:container")
    else:
        for key in sorted(_HOST_DECLARING_KEYS & set(container)):
            errors.append(f"{BLOCKER_REMOTE_ENDPOINT_FORBIDDEN}:container.{key}")
        errors.extend(
            _unknown_fields(
                container, _CONTAINER_FIELDS | _HOST_DECLARING_KEYS, "container."
            )
        )
        image = _string(container.get("image"))
        if not _IMAGE_DIGEST_PINNED.match(image):
            # Tag-only ("repo:latest") and bare ("repo") references both land
            # here: without a digest the same admitted contract can run two
            # different policies on two days and no receipt would show it.
            errors.append(f"{BLOCKER_IMAGE_NOT_DIGEST_PINNED}:{image or 'missing'}")
        serve_command = container.get("serve_command")
        if (
            not isinstance(serve_command, list)
            or not serve_command
            or any(not isinstance(arg, str) or not arg for arg in serve_command)
        ):
            errors.append(f"{BLOCKER_CONTRACT_INVALID}:container_serve_command")
            serve_command = []
        port_value = container.get("port")
        if (
            isinstance(port_value, bool)
            or not isinstance(port_value, int)
            or not 1024 <= port_value <= 65535
        ):
            # Below 1024 is the privileged range and never a policy server;
            # refusing it also keeps a contract from squatting on ssh/http.
            errors.append(f"{BLOCKER_CONTRACT_INVALID}:container_port")
        else:
            port = int(port_value)
        handshake_kind = _string(container.get("handshake_kind"))
        if handshake_kind not in HANDSHAKE_KINDS:
            errors.append(f"{BLOCKER_CONTRACT_INVALID}:container_handshake_kind")
        credential_files_value = container.get("credential_files")
        credential_files: list[str] = []
        if not isinstance(credential_files_value, list):
            errors.append(f"{BLOCKER_CONTRACT_INVALID}:container_credential_files")
        else:
            for entry in credential_files_value:
                filename = entry if isinstance(entry, str) else ""
                if (
                    not filename
                    or "/" in filename
                    or "\\" in filename
                    or not _CREDENTIAL_FILENAME.match(filename)
                ):
                    # Absolute paths and separators would resolve outside the
                    # canonical secrets directory; the allowlist regex also
                    # keeps ``:`` out of the docker ``-v`` syntax.
                    errors.append(
                        f"{BLOCKER_CONTRACT_INVALID}:container_credential_file:"
                        f"{filename or 'empty'}"
                    )
                    continue
                if filename in credential_files:
                    errors.append(
                        f"{BLOCKER_CONTRACT_INVALID}:container_credential_file_duplicate:"
                        f"{filename}"
                    )
                    continue
                credential_files.append(filename)
        gpu_required = container.get("gpu_required")
        if not isinstance(gpu_required, bool):
            errors.append(f"{BLOCKER_CONTRACT_INVALID}:container_gpu_required")
            gpu_required = False
        container_normalized = {
            "image": image,
            "serve_command": [str(arg) for arg in serve_command],
            "port": port,
            "handshake_kind": handshake_kind,
            "credential_files": credential_files,
            "gpu_required": bool(gpu_required),
        }

    # --- endpoint fixed point ------------------------------------------------
    # The validator's own output carries the injected loopback endpoint, and
    # re-validating that output must succeed (downstream seams re-validate
    # rather than trust).  So exactly one endpoint value is admissible on
    # input: the injected form itself.  Anything else -- another host, another
    # port, "localhost" spelled differently -- is a declaration attempt.
    endpoint_normalized = {"host": LOOPBACK_HOST, "port": port}
    if "endpoint" in payload and payload.get("endpoint") != endpoint_normalized:
        errors.append(
            f"{BLOCKER_REMOTE_ENDPOINT_FORBIDDEN}:endpoint_must_be_injected_loopback"
        )

    # --- observation schema --------------------------------------------------
    observation = payload.get("observation_schema")
    cameras_normalized: list[dict[str, Any]] = []
    state_keys_normalized: list[str] = []
    if not isinstance(observation, Mapping):
        errors.append(f"{BLOCKER_CONTRACT_INVALID}:observation_schema")
    else:
        errors.extend(
            _unknown_fields(observation, _OBSERVATION_FIELDS, "observation_schema.")
        )
        cameras = observation.get("cameras")
        if not isinstance(cameras, list) or not cameras:
            errors.append(f"{BLOCKER_CONTRACT_INVALID}:observation_cameras_empty")
        else:
            seen_cameras: set[str] = set()
            for index, camera in enumerate(cameras):
                label = f"camera[{index}]"
                if not isinstance(camera, Mapping):
                    errors.append(
                        f"{BLOCKER_CONTRACT_INVALID}:observation_camera_not_mapping:{label}"
                    )
                    continue
                errors.extend(
                    _unknown_fields(
                        camera, _CAMERA_FIELDS, f"observation_schema.{label}."
                    )
                )
                name = _string(camera.get("name"))
                width = _positive_int(camera.get("width"))
                height = _positive_int(camera.get("height"))
                if not name:
                    errors.append(
                        f"{BLOCKER_CONTRACT_INVALID}:observation_camera_name:{label}"
                    )
                    continue
                if name in seen_cameras:
                    errors.append(
                        f"{BLOCKER_CONTRACT_INVALID}:observation_camera_duplicate:{name}"
                    )
                    continue
                seen_cameras.add(name)
                if width is None or height is None:
                    errors.append(
                        f"{BLOCKER_CONTRACT_INVALID}:observation_camera_dimensions:{name}"
                    )
                    continue
                cameras_normalized.append(
                    {"name": name, "width": width, "height": height}
                )
        state_keys = observation.get("state_keys")
        if not isinstance(state_keys, list) or any(
            not isinstance(key, str) or not key.strip() for key in state_keys
        ):
            errors.append(f"{BLOCKER_CONTRACT_INVALID}:observation_state_keys")
        else:
            state_keys_normalized = [key.strip() for key in state_keys]

    # --- action schema -------------------------------------------------------
    action = payload.get("action_schema")
    action_space_id = ""
    chunk_rows: int | None = None
    channels_normalized: list[dict[str, Any]] = []
    if not isinstance(action, Mapping):
        errors.append(f"{BLOCKER_CONTRACT_INVALID}:action_schema")
    else:
        errors.extend(_unknown_fields(action, _ACTION_FIELDS, "action_schema."))
        action_space_id = _string(action.get("action_space_id"))
        if not action_space_id:
            errors.append(f"{BLOCKER_CONTRACT_INVALID}:action_space_id")
        chunk_rows = _positive_int(action.get("chunk_rows"))
        if chunk_rows is None:
            errors.append(f"{BLOCKER_CONTRACT_INVALID}:action_chunk_rows")
        channels_normalized = _validate_channels(action.get("channels"), errors)

    if errors:
        raise CompanyPolicyContractError(errors)

    normalized: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy_id": policy_id,
        "company_id": company_id,
        "display_name": display_name,
        "checkpoint_identity": checkpoint_normalized,
        "claim_ceiling": CLAIM_CEILING_DEVELOPMENT_ONLY,
        "rights": rights_normalized,
        "container": container_normalized,
        "endpoint": endpoint_normalized,
        "observation_schema": {
            "cameras": cameras_normalized,
            "state_keys": state_keys_normalized,
        },
        "action_schema": {
            "action_space_id": action_space_id,
            "chunk_rows": chunk_rows,
            "channels": channels_normalized,
        },
    }
    digest = canonical_digest(normalized, digest_field="contract_digest")
    # An input that carried a digest must carry the *right* one: a stale or
    # forged digest silently rebound would let two different byte contents
    # claim the same sealed identity.
    if "contract_digest" in payload and payload.get("contract_digest") != digest:
        raise CompanyPolicyContractError(
            [f"{BLOCKER_CONTRACT_INVALID}:contract_digest_mismatch"]
        )
    normalized["contract_digest"] = digest
    return normalized


def company_policy_channel_contracts(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the validated per-channel contracts for the action validator.

    Always re-validates: the channels feed
    ``adp009d_droid_action_execution.validate_candidate_action_bounds`` as its
    declared envelope, and handing that validator unvalidated bounds would put
    an unadmitted contract on the execution path.
    """

    normalized = validate_company_policy_container_contract(contract)
    return [dict(channel) for channel in normalized["action_schema"]["channels"]]


__all__ = [
    "BLOCKER_CONTRACT_INVALID",
    "BLOCKER_IMAGE_NOT_DIGEST_PINNED",
    "BLOCKER_RAW_BOUNDS_NARROWER",
    "BLOCKER_REMOTE_ENDPOINT_FORBIDDEN",
    "CHANNEL_KINDS",
    "CLAIM_CEILING_DEVELOPMENT_ONLY",
    "CompanyPolicyContractError",
    "HANDSHAKE_KINDS",
    "LOOPBACK_HOST",
    "SCHEMA_VERSION",
    "company_policy_channel_contracts",
    "validate_company_policy_container_contract",
]
