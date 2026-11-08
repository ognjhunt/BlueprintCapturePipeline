"""Exact object versions in the artifact bucket, and the typed refusal withdrawal deletion uses.

Kept apart from ``website_capture_withdrawal_deletion`` (which re-exports both, the same objects) so a
reader that only opens or lists an archived object, such as the website scene preview served by the
intake service, never imports the withdrawal path's screening and paid-run modules.
"""
from __future__ import annotations

import re
from typing import Any, BinaryIO


class WithdrawalDeletionError(ValueError):
    """A typed refusal; never carries file contents."""


def _require(value: Any, code: str) -> None:
    if not value:
        raise WithdrawalDeletionError("website_withdrawal_deletion_" + code)


class B2ObjectVersions:
    """Exact object versions through the existing artifact store client (boto3 S3 API on B2)."""

    def __init__(self, client: Any, bucket: str):
        self.client, self.bucket = client, bucket

    @classmethod
    def configured(cls) -> B2ObjectVersions:
        from .task_evaluation_configured_scene_object_store import _artifact_object_store_client
        return cls(*_artifact_object_store_client())

    def key(self, uri: str) -> str:
        prefix = f"s3://{self.bucket}/"
        _require(uri.startswith(prefix) and len(uri) > len(prefix), "object_outside_artifact_bucket")
        key = uri[len(prefix):]
        _require(all(part not in {"", ".", ".."} for part in key.split("/")), "object_key_invalid")
        return key

    def versions(self, uri: str) -> list[list[str]]:
        # B2 only hides an object deleted without a version id: absence is an empty version listing.
        key, rows, markers = self.key(uri), [], {}
        for _page in range(10_000):
            page = self.client.list_object_versions(Bucket=self.bucket, Prefix=key, **markers)
            for row in [*(page.get("Versions") or []), *(page.get("DeleteMarkers") or [])]:
                if row.get("Key") == key:
                    _require(row.get("VersionId"), "object_listing_invalid")
                    rows.append([key, str(row["VersionId"])])
            if not page.get("IsTruncated"):
                return sorted(rows)
            markers = {"KeyMarker": page["NextKeyMarker"], "VersionIdMarker": page["NextVersionIdMarker"]}
        raise WithdrawalDeletionError("website_withdrawal_deletion_object_listing_unbounded")

    def named(self, prefix: str, filename: str, *, continuation_request_id: str | None = None,
              suffix: str = ".zip") -> list[str]:
        """Exact original and job-scoped continuation archive names, in any version."""
        keys, markers = set(), {}
        continuation = (re.compile(re.escape(f"{continuation_request_id}-supplement-") + r"[a-f0-9]{40}"
                                   + re.escape(suffix)) if continuation_request_id is not None else None)
        for _page in range(10_000):
            page = self.client.list_object_versions(Bucket=self.bucket, Prefix=prefix, **markers)
            for row in [*(page.get("Versions") or []), *(page.get("DeleteMarkers") or [])]:
                key = str(row.get("Key") or "")
                name = key.rpartition("/")[2]
                if key.startswith(prefix) and (name == filename or continuation and continuation.fullmatch(name)):
                    keys.add(key)
            if not page.get("IsTruncated"):
                return sorted(f"s3://{self.bucket}/{key}" for key in keys)
            markers = {"KeyMarker": page["NextKeyMarker"], "VersionIdMarker": page["NextVersionIdMarker"]}
        raise WithdrawalDeletionError("website_withdrawal_deletion_object_listing_unbounded")

    def delete(self, key: str, version: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key, VersionId=version)

    def open(self, uri: str) -> BinaryIO:
        return self.client.get_object(Bucket=self.bucket, Key=self.key(uri))["Body"]


__all__ = ["B2ObjectVersions", "WithdrawalDeletionError"]
