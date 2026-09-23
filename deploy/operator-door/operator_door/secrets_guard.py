"""Keep credentials out of everything the door serves.

Secrets on the host are not confined to ``/etc/blueprint/provider-secrets``:
env files, their backups, a service-account key and forwarded tokens also sit
in state directories the service account can read. systemd hides the known
locations; this module is the second and third layer. Names that look like
credentials are refused outright, and every byte range, archive member and
journal line is scanned for credential-shaped content before it leaves.

Over-refusal is the accepted failure mode: a refused file is reported by name
and rule, never by content.

Every pattern is bounded (no unbounded quantifier next to an alternation or a
line anchor), because the scanner runs over megabytes of capture data and a
pathological input must not pin the door's CPU.
"""

from __future__ import annotations

import re
from pathlib import PurePath

REDACTED = "<redacted: credential-shaped content>"

_NAME_RULES = re.compile(
    r"(?ix)"
    r"(^|[._-])env($|[._~-])|^\.envrc$"  # env files, their backups and variants
    r"|secret|token|credential|password|passwd"
    r"|service[-_]?account"
    r"|api[-_]?key|private[-_]?key"
    r"|(^|[_.-])id_(rsa|ed25519|ecdsa|dsa)"
    r"|\.(pem|key|p12|pfx|jks|keystore|kdbx|gpg|asc)$"
    r"|^\.(netrc|pgpass|pypirc|npmrc|dockercfg|git)$"  # .git: packed objects hide content
)

_ENV_SUFFIX = rb"(?:KEY|TOKEN|SECRET|SECRETS|PASSWORD|PASSWD|PASS|PWD|CREDENTIAL|CREDENTIALS|B64|BASE64)"

_CONTENT_RULES: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("private_key_block", re.compile(rb"-----BEGIN [A-Z0-9 ]{0,40}PRIVATE KEY( BLOCK)?-----")),
    # Fields that are secret at any length (a service-account key file, OAuth clients).
    ("json_secret_field", re.compile(rb'(?i)"(?:private_key|private_key_id|client_secret)"[ \t]{0,8}:[ \t]{0,8}"[^"]{1,4096}"')),
    (
        "json_secret_field",
        re.compile(
            rb'(?i)"[a-z0-9_]{0,64}(?:token|secret|password|passwd|private_key|api_key|apikey)"'
            rb'[ \t]{0,8}:[ \t]{0,8}"(?!/)[^"]{8,4096}"'
        ),
    ),
    (
        "env_secret_assignment",
        re.compile(
            rb"(?m)^[ \t]{0,32}(?:export[ \t]{1,8})?[A-Z][A-Z0-9_]{0,96}" + _ENV_SUFFIX +
            rb"(?:_JSON)?[ \t]{0,8}=[ \t]{0,8}(?!['\"]?/)['\"]?[^\s'\"]"
        ),
    ),
    (
        "config_secret_line",
        re.compile(
            rb"(?im)^[ \t]{0,32}[a-z_]{0,32}(?:password|passwd|secret|token)[a-z_]{0,16}[ \t]{0,8}"
            rb"[:=][ \t]{0,8}['\"]?(?!/)[^\s'\"#]{6,4096}"
        ),
    ),
    ("kubeconfig_key_data", re.compile(rb"(?i)client-key-data[ \t]{0,8}:[ \t]{0,8}[A-Za-z0-9+/=]{20,4096}")),
    ("docker_auth", re.compile(rb'"auth"[ \t]{0,8}:[ \t]{0,8}"[A-Za-z0-9+/=]{16,4096}"')),
    ("openai_key", re.compile(rb"\bsk-(?:proj-|live-|svcacct-)?[A-Za-z0-9_-]{20,4096}")),
    ("stripe_key", re.compile(rb"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,4096}")),
    ("aws_access_key", re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("google_api_key", re.compile(rb"AIza[0-9A-Za-z_-]{35}")),
    ("google_oauth_token", re.compile(rb"\b(?:ya29\.[A-Za-z0-9_-]{20,4096}|1//0[A-Za-z0-9_-]{20,4096})")),
    ("github_token", re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{36,4096}|github_pat_[A-Za-z0-9_]{22,4096})")),
    ("slack_token", re.compile(rb"\bxox[abprs]-[A-Za-z0-9-]{10,4096}")),
    ("notion_token", re.compile(rb"\b(?:ntn|secret)_[A-Za-z0-9]{40,4096}")),
    ("huggingface_token", re.compile(rb"\bhf_[A-Za-z0-9]{30,4096}")),
    ("blueprint_agent_key", re.compile(rb"\bbpk_[A-Za-z0-9_-]{32,4096}")),
    ("jwt", re.compile(rb"\beyJ[A-Za-z0-9_-]{10,4096}\.eyJ[A-Za-z0-9_-]{10,4096}\.[A-Za-z0-9_-]{10,4096}")),
    (
        "webhook_url",
        re.compile(
            rb"https://hooks\.slack\.com/services/[A-Za-z0-9/_-]{10,4096}"
            rb"|https://(?:discord|discordapp)\.com/api/webhooks/[0-9]{1,32}/[A-Za-z0-9_-]{10,4096}"
        ),
    ),
    ("url_credentials", re.compile(rb"\b[a-z][a-z0-9+.-]{0,15}://[^\s/:@'\"]{0,64}:[^\s/@'\"]{3,256}@")),
    (
        "url_query_secret",
        re.compile(
            rb"(?i)[?&](?:access_token|refresh_token|id_token|token|api_key|apikey|key|secret"
            rb"|client_secret|password|auth)=[A-Za-z0-9._~%+/=-]{16,4096}"
        ),
    ),
    (
        "signed_url",
        re.compile(rb"(?i)[?&](?:x-goog-signature|x-amz-signature|signature|sig)=[0-9a-z%+/=_-]{32,4096}"),
    ),
    ("bearer_header", re.compile(rb"(?i)authorization[ \t]{0,8}[:=][ \t]{0,8}bearer[ \t]{1,8}[A-Za-z0-9._~+/=-]{20,4096}")),
)

_PEM_BEGIN = re.compile(r"-----BEGIN [A-Z0-9 ]{0,40}PRIVATE KEY( BLOCK)?-----")
_PEM_END = re.compile(r"-----END [A-Z0-9 ]{0,40}PRIVATE KEY( BLOCK)?-----")


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
    """Replace every credential-shaped line, and whole private-key blocks."""

    out: list[str] = []
    in_key = False
    for line in text.splitlines(keepends=True):
        ending = line[len(line.rstrip("\r\n")):]
        if in_key or _PEM_BEGIN.search(line):
            in_key = not _PEM_END.search(line)
            out.append(REDACTED + ending)
        elif scan_bytes(line.encode("utf-8", "surrogateescape")) is None:
            out.append(line)
        else:
            out.append(REDACTED + ending)
    return "".join(out)
