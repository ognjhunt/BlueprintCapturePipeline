"""Materialize a non-authorizing provider NuRec Isaac request from clean Git HEAD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import write_json
from .external_provider_nurec import (
    ExternalProviderNuRecError,
    build_provider_nurec_isaac_request_from_checkout,
)


def materialize_provider_nurec_isaac_request(
    *,
    request_template: Mapping[str, Any],
    source_checkout: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Write an exact request only after binding it to the checkout's clean HEAD."""

    template = json.loads(json.dumps(dict(request_template), allow_nan=False))
    template.pop("schema_version", None)
    template.pop("isaac_verification_request_digest", None)
    request = build_provider_nurec_isaac_request_from_checkout(
        template,
        source_checkout=source_checkout,
    )
    destination = Path(output_path)
    if destination.is_symlink():
        raise ExternalProviderNuRecError(
            ["provider_isaac_request_output_symlink_forbidden"]
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, request)
    return request


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-template", required=True)
    parser.add_argument("--source-checkout", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        template = json.loads(
            Path(args.request_template).read_text(encoding="utf-8")
        )
        if not isinstance(template, Mapping):
            raise ExternalProviderNuRecError(
                ["provider_isaac_request_template_not_object"]
            )
        request = materialize_provider_nurec_isaac_request(
            request_template=template,
            source_checkout=args.source_checkout,
            output_path=args.output,
        )
    except (OSError, json.JSONDecodeError, ExternalProviderNuRecError) as exc:
        blockers = (
            list(exc.codes)
            if isinstance(exc, ExternalProviderNuRecError)
            else [f"provider_isaac_request_preparation_error:{type(exc).__name__}"]
        )
        print(json.dumps({"status": "blocked", "blockers": sorted(blockers)}))
        return 2
    print(
        json.dumps(
            {
                "status": "prepared_not_authorized",
                "source_commit_sha": request["source_commit_sha"],
                "isaac_verification_request_digest": request[
                    "isaac_verification_request_digest"
                ],
                "provider_allocation_performed": False,
                "proof_effect": "none",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
