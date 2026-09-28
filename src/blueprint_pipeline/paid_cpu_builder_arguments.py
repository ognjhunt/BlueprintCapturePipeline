"""Serialize the approved CPU builder arguments for detached execution."""

from __future__ import annotations

import argparse


def cpu_vector(args: argparse.Namespace) -> list[str]:
    values = [
        "--output-dir",
        args.output_dir,
        "--packet-manifest",
        args.packet_manifest,
        "--builder-evidence",
        args.builder_evidence,
        "--spend",
        args.spend,
        "--token-file",
        args.token_file,
        "--docker-username-file",
        args.docker_username_file,
        "--docker-password-file",
        args.docker_password_file,
        "--hf-token-file",
        args.hf_token_file,
        "--runpod-s3-access-key-file",
        args.runpod_s3_access_key_file,
        "--runpod-s3-secret-key-file",
        args.runpod_s3_secret_key_file,
        "--login-private-key",
        args.login_private_key,
        "--host-private-key",
        args.host_private_key,
        "--ssh-key-id",
        str(args.ssh_key_id),
        "--region",
        args.region,
    ]
    if args.allow_paid:
        values.append("--allow-paid")
    return values
