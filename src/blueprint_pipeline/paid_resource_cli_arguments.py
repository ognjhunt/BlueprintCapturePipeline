"""Reusable argument groups for the canonical paid-resource allocator."""

from __future__ import annotations

import argparse


def add_adp_resource_arguments(parser: argparse.ArgumentParser) -> None:
    """Preserve the native task-arena resource limits and machine selectors."""
    parser.add_argument("--adp-max-hourly-rate-usd", type=float, default=0.80)
    parser.add_argument("--adp-max-spend-usd", type=float, default=2.00)
    parser.add_argument("--adp-hard-ttl-seconds", type=int, default=7200)
    parser.add_argument("--adp-machine-avoidlist")
    parser.add_argument(
        "--adp-allowed-vast-machine-id", action="append", type=int, default=[],
        help="Operator-selected machine IDs for a controlled native policy run.",
    )
    parser.add_argument(
        "--adp-excluded-vast-machine-id",
        action="append",
        type=int,
        default=[],
        help="Immutable repeatable Vast machine exclusion bound before allocation.",
    )
    parser.add_argument(
        "--adp-allowed-active-vast-instance-id",
        action="append",
        type=int,
        default=[],
    )


def add_cpu_arguments(
    parser: argparse.ArgumentParser, *, require_provider: bool = True
) -> None:
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--packet-manifest", required=True)
    parser.add_argument("--builder-evidence", required=True)
    parser.add_argument("--spend", required=True)
    parser.add_argument("--token-file", default="~/.blueprint-secrets/digitalocean_api_token")
    parser.add_argument("--docker-username-file", default="~/.blueprint-secrets/docker_username")
    parser.add_argument("--docker-password-file", default="~/.blueprint-secrets/docker_pat")
    parser.add_argument("--hf-token-file", default="~/.blueprint-secrets/hf_token")
    parser.add_argument(
        "--runpod-s3-access-key-file",
        default="~/.blueprint-secrets/runpod_s3_access_key",
    )
    parser.add_argument(
        "--runpod-s3-secret-key-file",
        default="~/.blueprint-secrets/runpod_s3_secret_key",
    )
    parser.add_argument("--login-private-key", required=require_provider)
    parser.add_argument("--host-private-key", required=require_provider)
    parser.add_argument("--ssh-key-id", required=require_provider, type=int)
    parser.add_argument("--region", default="sfo3")
    parser.add_argument("--allow-paid", action="store_true")
