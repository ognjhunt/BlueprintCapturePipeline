"""The secret guard: refuse secret-looking names, detect credential-shaped bytes."""

# Covers (for impacted-test selection):
#   deploy/operator-door/operator_door/secrets_guard.py

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "operator-door"))

from operator_door.secrets_guard import (  # noqa: E402
    _BARE_TOKEN_RULES,
    _CONTENT_RULES,
    REDACTED,
    redact_lines,
    refused_name,
    scan_bytes,
    scan_numeric_bytes,
)


@pytest.mark.parametrize(
    "name",
    [
        "pipeline-control-plane.env",
        "pipeline-control-plane.env.bak-c52-20260823T134014Z",
        "pipeline-control-plane.env.WjvaU2",
        "release.env",
        ".env",
        "vast_ssh_id_ed25519",
        "id_rsa",
        "id_ed25519.pub",
        "episode-interpreter-service-account.json",
        "srv-d9t8gg1t0dsc73am9q70-CAPTURE_UPLOAD_INTAKE_FORWARD_TOKEN.json",
        "client_secret.json",
        "credentials.json",
        "cert.pem",
        "tls.key",
        "creds.p12",
        ".netrc",
        ".git-credentials",
        "token_budget.json",
        "openai_api_key_semantic",
        "api-key.txt",
        ".env-local",
        ".env_prod",
        "settings.env~",
        "env.production",
        ".envrc",
        ".git",
    ],
)
def test_secret_looking_names_are_refused(name: str) -> None:
    assert refused_name(Path("/var/lib/blueprint/x") / name)


@pytest.mark.parametrize(
    "name",
    [
        "progression.json",
        "validation-progress.json",
        "known_hosts",
        "provider-known-hosts",
        "iteration_ac39dabb9af5.json",
        "cpu_prestage_entrypoint.log",
        "website_scene_sponsorship.json",
        "keyframes.json",
        "environment.json",
    ],
)
def test_ordinary_run_state_names_are_allowed(name: str) -> None:
    assert not refused_name(Path("/var/lib/blueprint/x") / name)


_CREDENTIALS = [
    (b"-----BEGIN PRIVATE KEY-----\nMIIE\n", "private_key_block"),
    (b"-----BEGIN OPENSSH PRIVATE KEY-----\nb3Bl\n", "private_key_block"),
    (b'{"type": "service_account", "private_key": "x"}', "json_secret_field"),
    (b'{"client_secret": "abcdefghijklmnop"}', "json_secret_field"),
    (b'{"refresh_token": "1//0abcdefghijk"}', "json_secret_field"),
    (b'{"api_key": "abcdefghijklmnop12"}', "json_secret_field"),
    (b"key=sk-" + b"A" * 40, "openai_key"),
    (b"AKIA" + b"B" * 16, "aws_access_key"),
    (b"AIza" + b"C" * 35, "google_api_key"),
    (b"ghp_" + b"d" * 36, "github_token"),
    (b"github_pat_" + b"e" * 40, "github_token"),
    (b"xoxb-1234-5678-abcdefghij", "slack_token"),
    (b"bpk_" + b"f" * 64, "blueprint_agent_key"),
    (b"Authorization: Bearer abcdefghijklmnopqrstuvwx", "bearer_header"),
    (b"OPENAI_API_KEY=sk-live-value\n", "env_secret_assignment"),
    (b"export PIPELINE_SYNC_TOKEN='abc123456'\n", "env_secret_assignment"),
    (b"DB_PASSWORD=hunter2hunter2\n", "env_secret_assignment"),
    (b"BLUEPRINT_LIVE_PIPELINE_CLIENT_SECRETS_JSON={\"a\":\"b\"}\n", "env_secret_assignment"),
    (b'{"alert": "https://hooks.slack.com/services/T000/B000/abcdefghijklmnop"}', "webhook_url"),
    (b"https://discord.com/api/webhooks/123/abcdefghijklmnopqrst", "webhook_url"),
    (b"postgres://svc:hunter2hunter2@db.internal:5432/app", "url_credentials"),
    (b"GET /files?id=4&access_token=abcdefghijklmnopqrstuvwx HTTP/1.1", "url_query_secret"),
    (b"users:\n- user:\n    client-key-data: " + b"Q" * 64, "kubeconfig_key_data"),
    (b'{"auths": {"ghcr.io": {"auth": "dXNlcjpwYXNzd29yZDEyMw=="}}}', "docker_auth"),
    (b"SERVICE_ACCOUNT_B64=eyJ0eXBlIjoi\n", "env_secret_assignment"),
    (b"DB_PASS=hunter2\n", "env_secret_assignment"),
    (b"database:\n  password: s3cretvalue\n", "config_secret_line"),
    (b"[auth]\ntoken = abcdefghijkl\n", "config_secret_line"),
    (b"ntn_" + b"a" * 46, "notion_token"),
    (b"hf_" + b"b" * 34, "huggingface_token"),
    (b"redis://:s3cretpw@cache.internal:6379/0", "url_credentials"),
    # Only the deploy tool's exact lease key is exempt, not keys that merely contain it.
    (b'{"disk_reservation_token": "0123456789abcdef0123"}', "json_secret_field"),
    (b'{"lease_token": "0123456789abcdef0123"}', "json_secret_field"),
]


@pytest.mark.parametrize(("payload", "reason"), _CREDENTIALS)
def test_credential_shaped_content_is_detected(payload: bytes, reason: str) -> None:
    assert scan_bytes(payload) == reason


@pytest.mark.parametrize(("payload", "reason"), _CREDENTIALS)
def test_numbers_hold_every_credential_written_as_text(payload: bytes, reason: str) -> None:
    """Text written into bytes that are numbers by their format is judged as text is."""

    assert scan_numeric_bytes(payload) == reason
    for framed in (b"\x00" + payload + b"\x00", b"\x9c\x00" + payload + b"\x00\x9c", b"\x9c\n" + payload + b"\n"):
        assert scan_numeric_bytes(framed) == scan_bytes(framed)


# A dark brown voxel's colour, as an SPZ splat stores it, reads "hf_"; ten dark grey ones read "adb" each.
_CHANCE_COLOURS = b"\x8b\x8b\x8bhf_" + b"adb" * 10 + b"\x8b\x8b\x8b"
_HF_TOKEN = b"hf_" + b"QzWxEcRvTbYnUmIkOlPjHgFdSaQwErTyUi"


@pytest.mark.parametrize(
    "payload",
    [
        _CHANCE_COLOURS,
        b"\x9c" + _HF_TOKEN + b"\x9c",
        b"\x9c[" + _HF_TOKEN + b"]\x9c",  # punctuation that colours and scales are full of
        b"\x9csk-" + b"a" * 30 + b"\x9c",
        b"\x9cAKIA" + b"B" * 16 + b"\x9c",
        b"\xa0ghp_" + b"d" * 36 + b"\xa0",
        b"\xa0AIza" + b"C" * 35 + b"\xa0",
        b"\xa0xoxb-1234-5678-abcdefghij\xa0",
        b"\xa0eyJ" + b"a" * 12 + b".eyJ" + b"b" * 12 + b"." + b"c" * 12 + b"\xa0",
        b"\x80" + _HF_TOKEN,  # a token shape up to the end still needs an edge before it
    ],
)
def test_token_shapes_that_numbers_make_by_chance_pass_as_numbers(payload: bytes) -> None:
    assert scan_bytes(payload) is not None
    assert scan_numeric_bytes(payload) is None


@pytest.mark.parametrize(
    "payload",
    [
        b"\x00" + _HF_TOKEN + b"\x00",
        b"\x9c\x00" + _HF_TOKEN + b"\x00\x9c",
        b'\x9c"' + _HF_TOKEN + b'"\x9c',
        b"\x9c'" + _HF_TOKEN + b"'\x9c",
        b"\x9c " + _HF_TOKEN + b"\t\x9c",
        b"\x9c\n" + _HF_TOKEN + b"\r\n\x9c",
        b"\x9c=" + _HF_TOKEN + b",\x9c",
        b"\x9c:" + _HF_TOKEN + b"\x00",
        _HF_TOKEN,
        b"\x9c\x00" + _HF_TOKEN,
    ],
)
def test_a_token_written_into_numbers_as_a_string_is_found(payload: bytes) -> None:
    assert scan_numeric_bytes(payload) == "huggingface_token"


def test_bytes_before_start_are_context_only() -> None:
    assert scan_numeric_bytes(b"\x9c" + _HF_TOKEN, 1) is None
    assert scan_numeric_bytes(b" " + _HF_TOKEN, 1) == "huggingface_token"
    assert scan_numeric_bytes(_HF_TOKEN + b" " + b"\x9c" * 8, 1) is None


def test_only_known_bare_token_rules_are_anchored() -> None:
    assert _BARE_TOKEN_RULES <= {reason for reason, _ in _CONTENT_RULES}


@pytest.mark.parametrize(
    "payload",
    [
        b'{"token_count": 12, "tokens_used": 40}',
        b'{"digest": "sha256:' + b"a" * 64 + b'"}',
        b"commit 83c3b09fbc59264d5361fd5bf53ca80c32674ee2",
        b'{"api_key_file": "/etc/blueprint/provider-secrets/gemini_api_key"}',
        b"VAST_API_KEY_FILE=/etc/blueprint/provider-secrets/vast_api_key\n",
        b'{"secret_values_exposed": false, "api_key": ""}',
        b"progress: 42% of attempts; bearer tokens are refused unless legacy mode",
        b"https://paperclip.tryblueprint.io/api/live-pipeline/version?token_hint=no",
        b"git@github.com:ognjhunt/BlueprintCapturePipeline.git",
        # Every deploy receipt carries the disk budget's lease id (a uuid4 hex).
        b'{"disk_reservation": {"reservation_token": "0123456789abcdef0123456789abcdef"}}',
        b"",
    ],
)
def test_ordinary_content_is_not_flagged(payload: bytes) -> None:
    assert scan_bytes(payload) is None


def test_redact_lines_replaces_only_matching_lines() -> None:
    text = "stage ok\nGEMINI_API_KEY=abcdefgh123\nnext stage\n"
    assert redact_lines(text) == f"stage ok\n{REDACTED}\nnext stage\n"


def test_multi_line_private_keys_are_redacted_whole() -> None:
    text = "before\n-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkq\nhkiG9w0BAQEFAASC\n-----END PRIVATE KEY-----\nafter\n"
    assert redact_lines(text) == f"before\n{REDACTED}\n{REDACTED}\n{REDACTED}\n{REDACTED}\nafter\n"


_PATHOLOGICAL = [
    b"\n" * (1 << 20), b"\r\n" * (1 << 19), b"a." * (1 << 19), b"A" * (1 << 20), b'"' * (1 << 20),
    b"a:" * (1 << 19), b"eyJ" * 349_525, b"-" * (1 << 20), b" \t" * (1 << 19), bytes(range(256)) * 4096,
]


@pytest.mark.parametrize("payload", _PATHOLOGICAL)
def test_pathological_megabytes_scan_in_linear_time(payload: bytes) -> None:
    import time

    started = time.perf_counter()
    scan_bytes(payload)
    redact_lines(payload.decode("latin-1")[: 1 << 18])
    assert time.perf_counter() - started < 3.0


@pytest.mark.parametrize("payload", _PATHOLOGICAL + [b"\x00hf_" * (1 << 18), (b" sk-" + b"a" * 4100) * 250])
def test_pathological_megabytes_scan_as_numbers_in_linear_time(payload: bytes) -> None:
    import time

    started = time.perf_counter()
    scan_numeric_bytes(payload)
    scan_numeric_bytes(payload, 1)
    assert time.perf_counter() - started < 3.0

