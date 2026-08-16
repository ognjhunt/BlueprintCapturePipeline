#!/usr/bin/env python3
"""Publish one profile manifest to an exclusive content-addressed object.

The command performs no compute-provider mutation.  It uploads only the named
JSON file, downloads the complete object again, verifies SHA-256 and size, and
writes a secret-free self-digesting receipt.  A pre-existing destination or a
provider without full-byte readback fails closed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from blueprint_pipeline.robot_eval_provider_input_setup import (
    LIVE_PROFILE_MANIFEST_PUBLICATION_SEAMS,
    publish_content_addressed_manifest,
)


def _destination_prefix(args: argparse.Namespace) -> str:
    if args.destination_prefix:
        return str(args.destination_prefix)
    bucket_file_text = args.bucket_file or os.getenv(
        "BLUEPRINT_WAM_OBJECT_STORE_BUCKET_FILE"
    )
    if not bucket_file_text:
        raise ValueError("immutable_manifest_bucket_file_required")
    bucket_file = Path(bucket_file_text).expanduser().resolve()
    if bucket_file.is_symlink() or not bucket_file.is_file():
        raise ValueError("immutable_manifest_bucket_file_invalid")
    bucket = bucket_file.read_text(encoding="utf-8").strip()
    if not bucket or "/" in bucket or bucket.startswith("gs:"):
        raise ValueError("immutable_manifest_bucket_name_invalid")
    return f"gs://{bucket}/{str(args.key_prefix).strip('/')}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--profile-builder",
        required=True,
        choices=sorted(LIVE_PROFILE_MANIFEST_PUBLICATION_SEAMS),
        help="Website-reachable live-profile builder that will consume this URI.",
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--destination-prefix",
        help="Explicit gs://, s3://, or r2:// content-addressed destination prefix.",
    )
    destination.add_argument("--bucket-file")
    parser.add_argument(
        "--key-prefix",
        default="task-evaluation/immutable-manifests",
        help="Object prefix used with --bucket-file.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        receipt = publish_content_addressed_manifest(
            source=args.manifest,
            destination_prefix=_destination_prefix(args),
            receipt_path=args.output,
            profile_builder=args.profile_builder,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "task_evaluation_immutable_manifest_publication.v1",
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_compute_mutation_performed": False,
                    "paid_resource_allocated": False,
                    "raw_secret_values_recorded": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": "published",
                "receipt_path": str(Path(args.output).expanduser().resolve()),
                "receipt_digest": receipt["receipt_digest"],
                "source_sha256": receipt["source"]["sha256"],
                "profile_builder": receipt["profile_builder"],
                "storage_scheme": receipt["storage_scheme"],
                "provider_compute_mutation_performed": False,
                "paid_resource_allocated": False,
                "raw_secret_values_recorded": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
