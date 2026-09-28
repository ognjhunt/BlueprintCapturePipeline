"""Non-secret identity shared by a signed PUT/GET object capability pair."""
from __future__ import annotations

import hashlib
import json
from urllib.parse import urlparse


def signed_output_object_binding_sha256(put_url: str, get_url: str) -> str:
    """Hash the non-secret origin/path identity shared by one PUT/GET pair."""

    identities: list[str] = []
    for label, value in (("put", put_url), ("get", get_url)):
        parsed = urlparse(str(value or ""))
        if (
            parsed.scheme.lower() != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise ValueError(f"signed_output_{label}_url_invalid")
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError(f"signed_output_{label}_url_invalid") from exc
        identity = (
            parsed.scheme.lower(),
            parsed.hostname.lower(),
            port or 443,
            parsed.path,
        )
        identities.append(json.dumps(identity, separators=(",", ":")))
    if identities[0] != identities[1]:
        raise ValueError("signed_output_put_get_object_identity_mismatch")
    return hashlib.sha256(identities[0].encode("utf-8")).hexdigest()

