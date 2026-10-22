"""Resume one admitted Atlas operation through the canonical allocator.

ADP-040 / Day 28. ``atlas_video_inputs`` prepares the reviewed frames first.
The key is read only into this process; it is never printed or persisted here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-context", type=Path, required=True)
    parser.add_argument("--price-review", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--api-base-url", default="https://api.atlas-beta.worldlabs.ai/api/v2")
    parser.add_argument("--provider-retention", choices=["account_does_not_retain", "project_library_retention_approved"], required=True)
    parser.add_argument("--experimental", action="store_true", help="Require a clean, remotely retained codex branch")
    parser.add_argument("--max-polls", type=int, default=2)
    args = parser.parse_args()
    from .paid_resource_allocator import run_sponsored_website_atlas_pose
    from .website_atlas_pose import ATLAS_API_BASE_URL_ENV, ATLAS_API_KEY_ENV, ATLAS_PRICE_REVIEW_ENV, ATLAS_PROVIDER_RETENTION_ENV
    os.environ[ATLAS_API_BASE_URL_ENV] = args.api_base_url
    if args.api_key_file is not None:
        if args.api_key_file.is_symlink() or args.api_key_file.stat().st_mode & 0o077:
            raise ValueError("atlas_operator_key_file_unsafe")
        os.environ[ATLAS_API_KEY_ENV] = args.api_key_file.read_text().strip()
    os.environ[ATLAS_PRICE_REVIEW_ENV] = args.price_review.read_text()
    os.environ[ATLAS_PROVIDER_RETENTION_ENV] = args.provider_retention
    context = json.loads(args.task_context.read_text())
    result = run_sponsored_website_atlas_pose(output_root=args.output_root, capture_id=context["capture_id"],
        task_context=context, experimental=args.experimental, max_polls=args.max_polls)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
