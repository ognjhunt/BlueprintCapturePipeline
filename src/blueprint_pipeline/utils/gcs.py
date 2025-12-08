"""Google Cloud Storage utilities for artifact management."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class GCSPath:
    """Parsed GCS URI components."""
    bucket: str
    blob: str

    @classmethod
    def from_uri(cls, uri: str) -> "GCSPath":
        """Parse gs://bucket/path/to/blob into components."""
        match = re.match(r"gs://([^/]+)/(.+)", uri)
        if not match:
            raise ValueError(f"Invalid GCS URI: {uri}")
        return cls(bucket=match.group(1), blob=match.group(2))

    @property
    def uri(self) -> str:
        return f"gs://{self.bucket}/{self.blob}"

    def join(self, *parts: str) -> "GCSPath":
        """Join additional path components."""
        new_blob = "/".join([self.blob.rstrip("/")] + list(parts))
        return GCSPath(bucket=self.bucket, blob=new_blob)


class GCSClient:
    """Wrapper around Google Cloud Storage client with caching and batch operations."""

    def __init__(self, project: Optional[str] = None):
        """Initialize GCS client.

        Args:
            project: GCP project ID. If None, uses default from environment.
        """
        self._project = project
        self._client = None
        self._bucket_cache: dict = {}

    @property
    def client(self):
        """Lazy-load the storage client."""
        if self._client is None:
            try:
                from google.cloud import storage
                self._client = storage.Client(project=self._project)
            except ImportError:
                raise ImportError(
                    "google-cloud-storage is required. Install with: "
                    "pip install google-cloud-storage"
                )
        return self._client

    def _get_bucket(self, bucket_name: str):
        """Get bucket object with caching."""
        if bucket_name not in self._bucket_cache:
            self._bucket_cache[bucket_name] = self.client.bucket(bucket_name)
        return self._bucket_cache[bucket_name]

    def download(
        self,
        gcs_uri: str,
        local_path: Path,
        create_dirs: bool = True,
    ) -> Path:
        """Download a blob from GCS to local path.

        Args:
            gcs_uri: Full GCS URI (gs://bucket/path/to/blob)
            local_path: Local destination path
            create_dirs: Create parent directories if needed

        Returns:
            Path to downloaded file
        """
        parsed = GCSPath.from_uri(gcs_uri)
        bucket = self._get_bucket(parsed.bucket)
        blob = bucket.blob(parsed.blob)

        if create_dirs:
            local_path.parent.mkdir(parents=True, exist_ok=True)

        blob.download_to_filename(str(local_path))
        return local_path

    def upload(
        self,
        local_path: Path,
        gcs_uri: str,
        content_type: Optional[str] = None,
    ) -> str:
        """Upload a local file to GCS.

        Args:
            local_path: Local file path
            gcs_uri: Destination GCS URI
            content_type: Optional MIME type

        Returns:
            GCS URI of uploaded blob
        """
        parsed = GCSPath.from_uri(gcs_uri)
        bucket = self._get_bucket(parsed.bucket)
        blob = bucket.blob(parsed.blob)

        if content_type:
            blob.content_type = content_type

        blob.upload_from_filename(str(local_path))
        return gcs_uri

    def upload_bytes(
        self,
        data: bytes,
        gcs_uri: str,
        content_type: Optional[str] = None,
    ) -> str:
        """Upload bytes directly to GCS.

        Args:
            data: Bytes to upload
            gcs_uri: Destination GCS URI
            content_type: Optional MIME type

        Returns:
            GCS URI of uploaded blob
        """
        parsed = GCSPath.from_uri(gcs_uri)
        bucket = self._get_bucket(parsed.bucket)
        blob = bucket.blob(parsed.blob)

        if content_type:
            blob.content_type = content_type

        blob.upload_from_string(data)
        return gcs_uri

    def download_directory(
        self,
        gcs_prefix: str,
        local_dir: Path,
        max_workers: int = 8,
    ) -> List[Path]:
        """Download all blobs under a GCS prefix.

        Args:
            gcs_prefix: GCS URI prefix (gs://bucket/prefix/)
            local_dir: Local directory to download to
            max_workers: Number of parallel downloads

        Returns:
            List of downloaded local paths
        """
        parsed = GCSPath.from_uri(gcs_prefix)
        bucket = self._get_bucket(parsed.bucket)
        blobs = list(bucket.list_blobs(prefix=parsed.blob))

        downloaded = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for blob in blobs:
                if blob.name.endswith("/"):
                    continue  # Skip directories
                relative = blob.name[len(parsed.blob):].lstrip("/")
                local_path = local_dir / relative
                local_path.parent.mkdir(parents=True, exist_ok=True)

                future = executor.submit(blob.download_to_filename, str(local_path))
                futures[future] = local_path

            for future in as_completed(futures):
                future.result()  # Raise any exceptions
                downloaded.append(futures[future])

        return downloaded

    def upload_directory(
        self,
        local_dir: Path,
        gcs_prefix: str,
        max_workers: int = 8,
    ) -> List[str]:
        """Upload all files in a local directory to GCS.

        Args:
            local_dir: Local directory to upload
            gcs_prefix: GCS URI prefix destination
            max_workers: Number of parallel uploads

        Returns:
            List of uploaded GCS URIs
        """
        parsed = GCSPath.from_uri(gcs_prefix)
        bucket = self._get_bucket(parsed.bucket)

        uploaded = []
        files = [f for f in local_dir.rglob("*") if f.is_file()]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for local_path in files:
                relative = local_path.relative_to(local_dir)
                blob_name = f"{parsed.blob.rstrip('/')}/{relative}"
                blob = bucket.blob(blob_name)

                future = executor.submit(blob.upload_from_filename, str(local_path))
                futures[future] = f"gs://{parsed.bucket}/{blob_name}"

            for future in as_completed(futures):
                future.result()  # Raise any exceptions
                uploaded.append(futures[future])

        return uploaded

    def list_blobs(
        self,
        gcs_prefix: str,
        pattern: Optional[str] = None,
    ) -> Iterator[str]:
        """List blobs under a GCS prefix.

        Args:
            gcs_prefix: GCS URI prefix
            pattern: Optional regex pattern to filter blob names

        Yields:
            GCS URIs of matching blobs
        """
        parsed = GCSPath.from_uri(gcs_prefix)
        bucket = self._get_bucket(parsed.bucket)

        compiled_pattern = re.compile(pattern) if pattern else None

        for blob in bucket.list_blobs(prefix=parsed.blob):
            if blob.name.endswith("/"):
                continue
            if compiled_pattern and not compiled_pattern.search(blob.name):
                continue
            yield f"gs://{parsed.bucket}/{blob.name}"

    def exists(self, gcs_uri: str) -> bool:
        """Check if a blob exists in GCS."""
        parsed = GCSPath.from_uri(gcs_uri)
        bucket = self._get_bucket(parsed.bucket)
        blob = bucket.blob(parsed.blob)
        return blob.exists()

    def delete(self, gcs_uri: str) -> None:
        """Delete a blob from GCS."""
        parsed = GCSPath.from_uri(gcs_uri)
        bucket = self._get_bucket(parsed.bucket)
        blob = bucket.blob(parsed.blob)
        blob.delete()


# Module-level convenience functions
_default_client: Optional[GCSClient] = None


def _get_default_client() -> GCSClient:
    global _default_client
    if _default_client is None:
        _default_client = GCSClient()
    return _default_client


def download_blob(gcs_uri: str, local_path: Path) -> Path:
    """Download a blob using the default client."""
    return _get_default_client().download(gcs_uri, local_path)


def upload_blob(local_path: Path, gcs_uri: str) -> str:
    """Upload a blob using the default client."""
    return _get_default_client().upload(local_path, gcs_uri)


def list_blobs(gcs_prefix: str, pattern: Optional[str] = None) -> Iterator[str]:
    """List blobs using the default client."""
    return _get_default_client().list_blobs(gcs_prefix, pattern)
