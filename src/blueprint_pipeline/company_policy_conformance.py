"""Pure conformance checks for company-supplied policy containers.

Before a company policy is allowed to drive a paid episode, its server must
prove -- on the declared contract, with a synthetic observation -- that it
speaks the schema it declared.  The pattern is
``groot_n17_wire_client.run_wire_codec_self_check``: exercise the exact
contract surface without a network request, so a mismatch surfaces as a typed
refusal on the worker rather than as a mid-episode shape error after the GPU
is already paid for.

This module owns both halves of that check as pure functions:

* :func:`build_conformance_probe` constructs the synthetic observation --
  zero-filled images sized exactly per declared camera, zero state -- and the
  expectations the response must meet.  No sockets, no policy import.
* :func:`evaluate_conformance_response` judges a chunk the company server
  returned (the live wire call happens elsewhere, in the launch profile's
  runtime) against the declared ``action_schema`` using the *generalized*
  bounds validator, so the same per-channel raw-envelope doctrine that
  governs the frozen DROID candidates governs company channels.

Zeros are deliberate for the probe observation: the probe proves schema
conformance, not behavior, and a contentful observation would tempt reading
the returned actions as evidence.  Whatever the policy returns to zeros is
validated for shape and envelope and then discarded.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

try:  # flat provider-bundle layout
    from adp009d_droid_action_execution import (
        DroidActionExecutionError,
        validate_candidate_action_bounds,
    )
    from company_policy_container_contract import (
        validate_company_policy_container_contract,
    )
except ModuleNotFoundError:  # repository package
    from .adp009d_droid_action_execution import (
        DroidActionExecutionError,
        validate_candidate_action_bounds,
    )
    from .company_policy_container_contract import (
        validate_company_policy_container_contract,
    )


CONFORMANCE_SCHEMA_VERSION = "company_policy_conformance.v1"
BLOCKER_CONFORMANCE_FAILED = "company_policy_conformance_failed"

# The probe image contract: standard interleaved RGB rows, matching what every
# camera adapter in the lane emits.  Declared here rather than assumed so the
# probe and the live observation encoder cannot drift apart silently.
PROBE_IMAGE_CHANNELS = 3
PROBE_IMAGE_DTYPE = "uint8"


class CompanyPolicyConformanceError(ValueError):
    """Fail-closed conformance errors with stable, sorted blocker identifiers."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(e) for e in errors if str(e)}))
        super().__init__(";".join(self.errors))


def build_conformance_probe(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Construct the synthetic observation and expectations for one contract.

    Re-validates the contract first: a probe built from an unadmitted mapping
    would launder that mapping's schema into the conformance receipt.  Pure --
    no network, no policy import; the caller sends ``observation`` over the
    declared handshake elsewhere and brings the response back to
    :func:`evaluate_conformance_response`.
    """

    import numpy as np

    normalized = validate_company_policy_container_contract(contract)
    observation_schema = normalized["observation_schema"]
    action_schema = normalized["action_schema"]
    # (height, width, channels): the row-major image layout every consumer in
    # this lane uses.  Sized exactly from the declaration so a server that
    # only works at some other resolution fails here, not mid-episode.
    images = {
        camera["name"]: np.zeros(
            (camera["height"], camera["width"], PROBE_IMAGE_CHANNELS),
            dtype=np.uint8,
        )
        for camera in observation_schema["cameras"]
    }
    state = {key: 0.0 for key in observation_schema["state_keys"]}
    return {
        "schema_version": CONFORMANCE_SCHEMA_VERSION,
        "policy_id": normalized["policy_id"],
        "contract_digest": normalized["contract_digest"],
        "handshake_kind": normalized["container"]["handshake_kind"],
        "endpoint": dict(normalized["endpoint"]),
        "observation": {"images": images, "state": state},
        "image_dtype": PROBE_IMAGE_DTYPE,
        "image_channels": PROBE_IMAGE_CHANNELS,
        "expected_response": {
            "chunk_rows": action_schema["chunk_rows"],
            "chunk_width": len(action_schema["channels"]),
        },
        # Recorded so a conformance receipt can never be mistaken for evidence
        # that the policy was queried live: this half is arithmetic only.
        "network_performed": False,
    }


def evaluate_conformance_response(
    contract: Mapping[str, Any], response_chunk: Any
) -> dict[str, Any]:
    """Judge a company server's action chunk against its declared schema.

    Fail-closed: any shape, chunk-row, or channel-envelope violation raises
    with the ``company_policy_conformance_failed:`` prefix (wrapping the
    generalized validator's own blockers, so the refusal names the channel).
    A passing chunk yields a ``status: "conformant"`` receipt carrying the
    per-channel envelope report -- including command-interval overshoot,
    which is reported, never refused, exactly as for the frozen candidates.
    """

    import numpy as np

    normalized = validate_company_policy_container_contract(contract)
    action_schema = normalized["action_schema"]
    channels = action_schema["channels"]
    expected_rows = int(action_schema["chunk_rows"])
    expected_width = len(channels)
    try:
        values = np.asarray(response_chunk, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CompanyPolicyConformanceError(
            [f"{BLOCKER_CONFORMANCE_FAILED}:chunk_not_numeric"]
        ) from exc
    if values.ndim != 2:
        raise CompanyPolicyConformanceError(
            [f"{BLOCKER_CONFORMANCE_FAILED}:chunk_not_2d:shape={tuple(values.shape)}"]
        )
    if int(values.shape[1]) != expected_width:
        raise CompanyPolicyConformanceError(
            [
                f"{BLOCKER_CONFORMANCE_FAILED}:chunk_width:"
                f"expected={expected_width}:observed={int(values.shape[1])}"
            ]
        )
    if int(values.shape[0]) != expected_rows:
        # Exact equality, not at-least: the declared chunk length is part of
        # the interface identity.  A server that returns a different horizon
        # than it declared would execute a different open-loop plan than the
        # receipts describe.
        raise CompanyPolicyConformanceError(
            [
                f"{BLOCKER_CONFORMANCE_FAILED}:chunk_rows:"
                f"expected={expected_rows}:observed={int(values.shape[0])}"
            ]
        )
    try:
        bounds_receipt = validate_candidate_action_bounds(
            values,
            action_space=action_schema["action_space_id"],
            channel_contracts=channels,
        )
    except DroidActionExecutionError as exc:
        raise CompanyPolicyConformanceError(
            [f"{BLOCKER_CONFORMANCE_FAILED}:{error}" for error in exc.errors]
        ) from exc
    return {
        "schema_version": CONFORMANCE_SCHEMA_VERSION,
        "status": "conformant",
        "policy_id": normalized["policy_id"],
        "contract_digest": normalized["contract_digest"],
        "chunk_rows": int(values.shape[0]),
        "chunk_width": int(values.shape[1]),
        "bounds_receipt": bounds_receipt,
        "network_performed": False,
    }


__all__ = [
    "BLOCKER_CONFORMANCE_FAILED",
    "CONFORMANCE_SCHEMA_VERSION",
    "CompanyPolicyConformanceError",
    "PROBE_IMAGE_CHANNELS",
    "PROBE_IMAGE_DTYPE",
    "build_conformance_probe",
    "evaluate_conformance_response",
]
