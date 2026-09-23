"""The secret guard: refuse secret-looking names, detect credential-shaped bytes."""

# Covers (for impacted-test selection):
#   deploy/operator-door/operator_door/secrets_guard.py

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "operator-door"))

from operator_door.secrets_guard import REDACTED, redact_lines, refused_name, scan_bytes  # noqa: E402


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


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
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
    ],
)
def test_credential_shaped_content_is_detected(payload: bytes, reason: str) -> None:
    assert scan_bytes(payload) == reason


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


@pytest.mark.parametrize(
    "payload",
    [b"\n" * (1 << 20), b"\r\n" * (1 << 19), b"a." * (1 << 19), b"A" * (1 << 20), b'"' * (1 << 20),
     b"a:" * (1 << 19), b"eyJ" * 349_525, b"-" * (1 << 20), b" \t" * (1 << 19), bytes(range(256)) * 4096],
)
def test_pathological_megabytes_scan_in_linear_time(payload: bytes) -> None:
    import time

    started = time.perf_counter()
    scan_bytes(payload)
    redact_lines(payload.decode("latin-1")[: 1 << 18])
    assert time.perf_counter() - started < 3.0
