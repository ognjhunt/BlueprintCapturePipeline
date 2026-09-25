"""Send a verified, privately registered G1 review to the owner-only WebApp.

The caller supplies the real owner and organization identifiers. A signed
request is attempted only after all four episodes and all 12 registered media
artifacts have been reverified against the sealed paid result.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .core.security_controls import strict_identifier
from .native_g1_paid_campaign import verify_g1_paid_output
from .native_g1_private_review import project_g1_private_review
from .native_g1_private_review_delivery import _media, _read
from .native_g1_provider_bundle import load_verified_g1_provider_bundle
from .task_evaluation_result_delivery import (
    _artifact_id,
    resolve_task_evaluation_result_artifact,
)
from .webapp_sync import _pipeline_sync_headers


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def _ingest_url(base_url: str) -> str:
    parts = urllib.parse.urlsplit(base_url)
    if (
        parts.scheme != "https" or not parts.hostname or parts.username or parts.password
        or parts.fragment or parts.query
    ):
        raise ValueError("g1_review_ingest_webapp_url_invalid")
    return urllib.parse.urlunsplit((
        "https", parts.netloc, "/api/internal/pipeline/native-g1-reviews", "", "",
    ))


def _verified_review(
    *, adapter_result_path: Path, bundle_receipt_path: Path,
    retained_review_path: Path, delivery_receipt_path: Path,
    result_root: Path, run_id: str,
) -> dict[str, Any]:
    run = strict_identifier(run_id, field="run_id", max_length=192)
    adapter = _read(adapter_result_path)
    bundle_receipt = _read(bundle_receipt_path)
    bundle = load_verified_g1_provider_bundle(
        bundle_receipt_path,
        expected_implementation_commit=bundle_receipt["implementation_commit"],
    )
    review = project_g1_private_review(
        verification=verify_g1_paid_output(adapter, bundle), bundle=bundle,
    )
    if _read(retained_review_path) != review:
        raise ValueError("g1_review_ingest_retained_review_changed")
    delivery = _read(delivery_receipt_path)
    expected_root = Path(result_root) / f"{run}-activation"
    if (
        not Path(result_root).is_absolute() or Path(result_root).is_symlink()
        or expected_root.is_symlink() or not expected_root.is_dir()
        or delivery.get("status") != "registered_private_development_review"
        or delivery.get("run_id") != run
        or delivery.get("review_digest") != review["review_digest"]
        or delivery.get("artifact_count") != 12
        or delivery.get("run_root") != str(expected_root)
        or delivery.get("public_redistribution_authorized") is not False
    ):
        raise ValueError("g1_review_ingest_delivery_identity_invalid")
    registry = _read(expected_root / "artifacts/result_delivery/artifact_registry.json")
    if (
        registry.get("run_id") != run
        or registry.get("delivery_digest") != review["review_digest"]
        or registry.get("registry_digest") != delivery.get("registry_digest")
    ):
        raise ValueError("g1_review_ingest_registry_identity_invalid")
    for role, row in _media(review):
        artifact_id = _artifact_id(role, row["relative_path"], row["sha256"])
        _, registered = resolve_task_evaluation_result_artifact(
            run_root=expected_root, run_id=run, artifact_id=artifact_id,
        )
        if any(registered.get(key) != row[key] for key in (
            "relative_path", "sha256", "size_bytes",
        )) or registered.get("role") != role:
            raise ValueError("g1_review_ingest_registered_artifact_changed")
    return review


def _post_review(
    *, url: str, token: str, payload: Mapping[str, Any], timeout_seconds: int = 30,
) -> dict[str, Any]:
    if not token:
        raise ValueError("g1_review_ingest_sync_token_missing")
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    request = urllib.request.Request(
        url, data=body, headers=_pipeline_sync_headers(token, body), method="POST",
    )
    try:
        with urllib.request.build_opener(_NoRedirect()).open(
            request, timeout=timeout_seconds,
        ) as response:
            if response.status not in (200, 201) or response.geturl() != url:
                raise ValueError("g1_review_ingest_response_invalid")
            result = json.loads(response.read(64 * 1024))
    except urllib.error.HTTPError as exc:
        raise ValueError(f"g1_review_ingest_http_status_{exc.code}") from None
    except urllib.error.URLError as exc:
        raise ValueError("g1_review_ingest_transport_failed") from exc
    if (
        not isinstance(result, dict)
        or result.get("status") not in {"ingested", "already_ingested"}
        or result.get("run_id") != payload["run_id"]
        or result.get("review_digest") != payload["review"]["review_digest"]
    ):
        raise ValueError("g1_review_ingest_response_invalid")
    return result


def ingest_g1_private_review(
    *, adapter_result_path: Path, bundle_receipt_path: Path,
    retained_review_path: Path, delivery_receipt_path: Path,
    result_root: Path, run_id: str, owner_user_id: str,
    organization_id: str, webapp_url: str, sync_token: str,
) -> dict[str, Any]:
    run = strict_identifier(run_id, field="run_id", max_length=192)
    owner = strict_identifier(owner_user_id, field="owner_user_id", max_length=192)
    organization = strict_identifier(organization_id, field="organization_id", max_length=192)
    url = _ingest_url(webapp_url)
    review = _verified_review(
        adapter_result_path=adapter_result_path,
        bundle_receipt_path=bundle_receipt_path,
        retained_review_path=retained_review_path,
        delivery_receipt_path=delivery_receipt_path,
        result_root=result_root, run_id=run,
    )
    response = _post_review(url=url, token=sync_token, payload={
        "schema_version": "native_g1_private_review_ingest.v1",
        "run_id": run,
        "owner_user_id": owner,
        "organization_id": organization,
        "review": review,
    })
    return {
        "schema_version": "native_g1_private_review_ingest_receipt.v1",
        "status": response["status"],
        "run_id": run,
        "review_digest": review["review_digest"],
        "owner_user_id": owner,
        "organization_id": organization,
        "review_url": urllib.parse.urlunsplit((
            "https", urllib.parse.urlsplit(url).netloc,
            "/app/g1-reviews/" + urllib.parse.quote(run, safe=""), "", "",
        )),
        "access_visibility": "owner_only",
        "claim_ceiling": "development_only",
        "public_redistribution_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-result", type=Path, required=True)
    parser.add_argument("--bundle-receipt", type=Path, required=True)
    parser.add_argument("--retained-review", type=Path, required=True)
    parser.add_argument("--delivery-receipt", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--owner-user-id", required=True)
    parser.add_argument("--organization-id", required=True)
    parser.add_argument("--webapp-url", default=os.environ.get("PIPELINE_SYNC_WEBAPP_URL", ""))
    args = parser.parse_args(argv)
    result = ingest_g1_private_review(
        adapter_result_path=args.adapter_result,
        bundle_receipt_path=args.bundle_receipt,
        retained_review_path=args.retained_review,
        delivery_receipt_path=args.delivery_receipt,
        result_root=args.result_root,
        run_id=args.run_id,
        owner_user_id=args.owner_user_id,
        organization_id=args.organization_id,
        webapp_url=args.webapp_url,
        sync_token=os.environ.get("PIPELINE_SYNC_TOKEN", ""),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
