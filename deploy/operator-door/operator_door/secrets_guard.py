"""Keep credentials out of everything the door serves.

Secrets on the host are not confined to ``/etc/blueprint/provider-secrets``:
env files, their backups, a service-account key and forwarded tokens also sit
in state directories the service account can read. systemd hides the known
locations; this module is the second and third layer. Names that look like
credentials are refused outright, and every byte range, archive member and
journal line is scanned for credential-shaped content before it leaves.

Over-refusal is the accepted failure mode: a refused file is reported by name
and rule, never by content.

Bytes that a format defines as numbers, not text, are read by
``scan_numeric_bytes`` instead (``fsview`` decides which bytes those are).

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
            # `reservation_token` is the disk budget's lease id (a uuid4) in every deploy
            # receipt; exempting that exact key keeps receipts readable.
            rb'(?i)"(?!reservation_token")[a-z0-9_]{0,64}(?:token|secret|password|passwd|private_key|api_key|apikey)"'
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

# Rules that match a credential's own shape, with no key, field name or other text around it.
_BARE_TOKEN_RULES = frozenset({
    "openai_key", "stripe_key", "aws_access_key", "google_api_key", "google_oauth_token", "github_token",
    "slack_token", "notion_token", "huggingface_token", "blueprint_agent_key", "jwt",
})
# The bytes a string ends at: NUL (a C string), ASCII whitespace, a quote, and the "=", ":" and ","
# between a key and its value. None of them can be part of a token.
_STRING_EDGE = rb"\x00\t\n\x0b\x0c\r \"'=:,"
# In numbers a bare token shape counts only where it stands as a string: a string edge, or the end of
# the data, on both sides. Every other rule needs text around its match and applies unchanged.
_NUMERIC_RULES: tuple[tuple[str, re.Pattern[bytes]], ...] = tuple(
    (reason, re.compile(rb"(?<![^" + _STRING_EDGE + rb"])(?:" + pattern.pattern + rb")(?![^" + _STRING_EDGE + rb"])")
     if reason in _BARE_TOKEN_RULES else pattern)
    for reason, pattern in _CONTENT_RULES
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


def scan_numeric_bytes(data: bytes, start: int = 0) -> str | None:
    """``scan_bytes`` for bytes a format defines as numbers, such as an SPZ splat's body.

    Quantized numbers often fall in the ASCII letter range: a dark colour run in a
    splat reads ``hf_`` and thirty letters. There a bare token shape counts only
    where it stands as a string, with NUL, whitespace, a quote, ``=``, ``:`` or
    ``,`` (or the end of ``data``) on both sides; a token written into such bytes
    as a string is still found. Every rule that needs text around its match, such
    as a key name or an assignment, applies exactly as it does to text.

    Bytes before ``start`` are context only: no match begins in them, but they are
    what a match at ``start`` stands after.
    """

    for reason, pattern in _NUMERIC_RULES:
        if pattern.search(data, start):
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
