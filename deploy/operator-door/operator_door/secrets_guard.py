"""Keep credentials out of everything the door serves.

Secrets on the host are not confined to ``/etc/blueprint/provider-secrets``:
env files, their backups, a service-account key and forwarded tokens also sit
in state directories the service account can read. systemd hides the known
locations; this module is the second and third layer. Names that look like
credentials are refused outright, and every byte range, archive member and
journal line is scanned for credential-shaped content before it leaves.

Over-refusal is the accepted failure mode: a refused file is reported by name
and rule, never by content.
"""

from __future__ import annotations

import re
from pathlib import PurePath

REDACTED = "<redacted: credential-shaped content>"

_NAME_RULES = re.compile(
    r"(?ix)"
    r"(^\.env$|\.env$|\.env\.)"  # env files and their backups
    r"|secret|token|credential|password|passwd"
    r"|service[-_]?account"
    r"|api[-_]?key|private[-_]?key"
    r"|(^|[_.-])id_(rsa|ed25519|ecdsa|dsa)"
    r"|\.(pem|key|p12|pfx|jks|keystore|kdbx|gpg|asc)$"
    r"|^\.(netrc|pgpass|pypirc|npmrc|dockercfg)$"
)

_CONTENT_RULES: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("private_key_block", re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY( BLOCK)?-----")),
    # Fields that are secret at any length (a service-account key file, OAuth clients).
    ("json_secret_field", re.compile(rb'(?i)"(?:private_key|private_key_id|client_secret)"\s*:\s*"[^"]+"')),
    (
        "json_secret_field",
        re.compile(
            rb'(?i)"[a-z0-9_]*?(?:token|secret|password|passwd|private_key|api_key|apikey)"'
            rb'\s*:\s*"(?!/)[^"]{8,}"'
        ),
    ),
    (
        "env_secret_assignment",
        re.compile(
            rb"(?m)^\s*(?:export\s+)?[A-Z][A-Z0-9_]*"
            rb"(?:KEY|TOKEN|SECRET|SECRETS|PASSWORD|PASSWD|CREDENTIAL|CREDENTIALS)(?:_JSON)?"
            rb"\s*=\s*(?!['\"]?/)['\"]?\S"
        ),
    ),
    ("openai_key", re.compile(rb"\bsk-(?:proj-|live-|svcacct-)?[A-Za-z0-9_-]{20,}")),
    ("stripe_key", re.compile(rb"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}")),
    ("aws_access_key", re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("google_api_key", re.compile(rb"AIza[0-9A-Za-z_-]{35}")),
    ("google_oauth_token", re.compile(rb"\b(?:ya29\.[A-Za-z0-9_-]{20,}|1//0[A-Za-z0-9_-]{20,})")),
    ("github_token", re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})")),
    ("slack_token", re.compile(rb"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("blueprint_agent_key", re.compile(rb"\bbpk_[A-Za-z0-9_-]{32,}")),
    ("jwt", re.compile(rb"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    (
        "signed_url",
        re.compile(rb"(?i)[?&](?:x-goog-signature|x-amz-signature|signature|sig)=[0-9a-z%+/=_-]{32,}"),
    ),
    ("bearer_header", re.compile(rb"(?i)authorization\s*[:=]\s*bearer\s+[A-Za-z0-9._~+/=-]{20,}")),
)


def refused_name(path: PurePath | str) -> bool:
    """True when any component of ``path`` looks like a credential file name."""

    parts = PurePath(path).parts
    return any(_NAME_RULES.search(part) for part in parts if part not in ("/", ""))


def scan_bytes(data: bytes) -> str | None:
    """Return the first matching rule name, or ``None`` for ordinary content."""

    for reason, pattern in _CONTENT_RULES:
        if pattern.search(data):
            return reason
    return None


def redact_lines(text: str) -> str:
    """Replace every credential-shaped line; keep line structure intact."""

    lines = text.splitlines(keepends=True)
    out: list[str] = []
    for line in lines:
        if scan_bytes(line.encode("utf-8", "surrogateescape")) is None:
            out.append(line)
        else:
            ending = line[len(line.rstrip("\r\n")):]
            out.append(REDACTED + ending)
    return "".join(out)
