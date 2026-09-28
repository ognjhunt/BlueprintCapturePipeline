"""Bounded signed-object transport for warm scene iterations."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .wam_provider_object_store import signed_output_object_binding_sha256

from .task_evaluation_scene_configuration_warm_overlay import (
    SceneConfigurationWarmDiagnosticError,
)

SIGNED_URL_RETRIEVAL_RESERVE_SECONDS = 120
SIGNED_URL_MAXIMUM_POST_WATCHDOG_SECONDS = 900
SIGNED_URL_SIGNING_DELAY_TOLERANCE_SECONDS = 300


def _signed_https_url(path: Path) -> str:
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError("unsafe signed URL path")
        value = path.read_text(encoding="utf-8").strip()
        parsed = urllib.parse.urlsplit(value)
        hostname = parsed.hostname
        parsed.port
    except (OSError, UnicodeError, ValueError) as exc:
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_iteration_signed_url_invalid"
        ) from exc
    if (
        parsed.scheme != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_iteration_signed_url_invalid"
        )
    return value


def validated_warm_staging_urls(
    *,
    staging_dir: Path,
    staging: Mapping[str, Any],
    overlay_archive: Path,
    watchdog_deadline_epoch: float,
) -> dict[str, str]:
    """Reopen the immutable staging binding and bind all three signed URLs."""

    manifest_path = staging_dir / "wam_provider_object_store_staging_manifest.json"
    binding_path = staging_dir / "wam_provider_object_store_staging_binding.json"
    try:
        if any(path.is_symlink() for path in (manifest_path, binding_path)):
            raise OSError("unsafe staging receipt")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        archive_digest = hashlib.sha256(overlay_archive.read_bytes()).hexdigest()
        archive_size = overlay_archive.stat().st_size
        binding_payload = json.dumps(
            {
                "bundle_key": binding["bundle_key"],
                "bundle_sha256": binding["bundle_sha256"],
                "output_key": binding["output_key"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        binding_digest = hashlib.sha256(binding_payload).hexdigest()
        expires_at = datetime.fromisoformat(
            str(manifest["presigned_url_expiry"]["expires_at"]).replace("Z", "+00:00")
        ).astimezone(timezone.utc).timestamp()
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_staging_binding_invalid"
        ) from exc
    if (
        manifest != dict(staging)
        or manifest.get("status") != "completed"
        or binding.get("schema_version") != "wam_provider_object_store_binding.v1"
        or binding.get("bundle_sha256") != archive_digest
        or manifest.get("bundle_sha256") != archive_digest
        or manifest.get("bundle_size_bytes") != archive_size
        or binding.get("staging_binding_sha256") != binding_digest
        or manifest.get("staging_binding_sha256") != binding_digest
        or manifest.get("output_key_run_unique") is not True
        or expires_at
        < watchdog_deadline_epoch + SIGNED_URL_RETRIEVAL_RESERVE_SECONDS
        or expires_at
        > watchdog_deadline_epoch + SIGNED_URL_MAXIMUM_POST_WATCHDOG_SECONDS
    ):
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_staging_binding_invalid"
        )
    urls: dict[str, str] = {}
    for key, filename in (
        ("overlay_url", "provider_bundle_url.txt"),
        ("output_put_url", "provider_output_put_url.txt"),
        ("output_get_url", "provider_output_get_url.txt"),
    ):
        path = staging_dir / filename
        status_key = {
            "overlay_url": "provider_bundle_url_file",
            "output_put_url": "provider_output_put_url_file",
            "output_get_url": "provider_output_get_url_file",
        }[key]
        status = manifest.get(status_key)
        try:
            file_stat = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise SceneConfigurationWarmDiagnosticError(
                "scene_configuration_warm_staging_url_record_invalid"
            ) from exc
        if not isinstance(status, Mapping) or (
            status.get("path") != str(path)
            or status.get("present") is not True
            or status.get("mode_is_0600") is not True
            or stat.S_IMODE(file_stat.st_mode) != 0o600
            or file_stat.st_size != status.get("size_bytes")
            or file_stat.st_mtime_ns != status.get("mtime_ns")
        ):
            raise SceneConfigurationWarmDiagnosticError(
                "scene_configuration_warm_staging_url_record_invalid"
            )
        urls[key] = _signed_https_url(path)
    put = urllib.parse.urlsplit(urls["output_put_url"])
    get = urllib.parse.urlsplit(urls["output_get_url"])
    bundle = urllib.parse.urlsplit(urls["overlay_url"])
    try:
        actual_binding = signed_output_object_binding_sha256(
            urls["output_put_url"], urls["output_get_url"]
        )
        actual_expiries: list[float] = []
        for parsed in (bundle, put, get):
            query = urllib.parse.parse_qs(parsed.query)
            if len(query.get("X-Amz-Date") or ()) != 1 or len(
                query.get("X-Amz-Expires") or ()
            ) != 1:
                raise ValueError("signed URL expiry fields must be singular")
            signed_at = datetime.strptime(
                query["X-Amz-Date"][0], "%Y%m%dT%H%M%SZ"
            ).replace(tzinfo=timezone.utc)
            actual_expiries.append(
                signed_at.timestamp() + int(query["X-Amz-Expires"][0])
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_signed_url_expiry_invalid"
        ) from exc
    bundle_key = str(manifest.get("bundle_key") or "")
    if (
        actual_binding != manifest.get("output_url_object_binding_sha256")
        or not bundle_key
        or not urllib.parse.unquote(bundle.path).endswith("/" + bundle_key)
        or any(
            value
            < watchdog_deadline_epoch + SIGNED_URL_RETRIEVAL_RESERVE_SECONDS
            or value
            > watchdog_deadline_epoch + SIGNED_URL_MAXIMUM_POST_WATCHDOG_SECONDS
            or value < expires_at - 1
            or value
            > expires_at + SIGNED_URL_SIGNING_DELAY_TOLERANCE_SECONDS
            for value in actual_expiries
        )
    ):
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_output_url_binding_mismatch"
        )
    return urls


def _download_bounded_when_ready(
    *, url: str, destination: Path, maximum_bytes: int, deadline_monotonic: float
) -> bool:
    """Poll one signed GET while bounding memory, disk, and final-path visibility."""

    if isinstance(maximum_bytes, bool) or maximum_bytes <= 0:
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_output_limit_invalid"
        )
    def available_bytes() -> int:
        try:
            stats = os.statvfs(destination.parent)
        except OSError as exc:
            raise SceneConfigurationWarmDiagnosticError(
                "scene_configuration_warm_output_disk_capacity_unavailable"
            ) from exc
        return int(stats.f_bavail) * int(stats.f_frsize)

    disk_reserve = 64 * 1024**2
    if available_bytes() < maximum_bytes + disk_reserve:
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_output_disk_capacity_insufficient"
        )
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    while time.monotonic() < deadline_monotonic:
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "BlueprintSceneWarm/1.0"}
            )
            remaining = deadline_monotonic - time.monotonic()
            if remaining <= 0:
                break
            with urllib.request.urlopen(  # nosec B310
                request, timeout=max(1.0, min(10.0, remaining))
            ) as response:
                if response.geturl() != url:
                    raise SceneConfigurationWarmDiagnosticError(
                        "scene_configuration_warm_output_redirect_refused"
                    )
                length = response.headers.get("Content-Length")
                if length is not None:
                    try:
                        declared_length = int(length)
                    except (TypeError, ValueError) as exc:
                        raise SceneConfigurationWarmDiagnosticError(
                            "scene_configuration_warm_output_content_length_invalid"
                        ) from exc
                    if declared_length < 0 or declared_length > maximum_bytes:
                        raise SceneConfigurationWarmDiagnosticError(
                            "scene_configuration_warm_output_exceeds_limit"
                        )
                descriptor = os.open(
                    temporary,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0),
                    0o440,
                )
                observed = 0
                try:
                    while True:
                        if time.monotonic() >= deadline_monotonic:
                            raise SceneConfigurationWarmDiagnosticError(
                                "scene_configuration_warm_output_download_deadline_exceeded"
                            )
                        chunk = response.read(min(1024 * 1024, maximum_bytes + 1 - observed))
                        if not chunk:
                            break
                        observed += len(chunk)
                        if observed > maximum_bytes:
                            raise SceneConfigurationWarmDiagnosticError(
                                "scene_configuration_warm_output_exceeds_limit"
                            )
                        if available_bytes() < (
                            maximum_bytes - observed + disk_reserve
                        ):
                            raise SceneConfigurationWarmDiagnosticError(
                                "scene_configuration_warm_output_disk_capacity_lost"
                            )
                        view = memoryview(chunk)
                        while view:
                            written = os.write(descriptor, view)
                            if written <= 0:
                                raise OSError("short warm output write")
                            view = view[written:]
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            if observed <= 0:
                temporary.unlink(missing_ok=True)
                return False
            os.replace(temporary, destination)
            return True
        except urllib.error.HTTPError as exc:
            temporary.unlink(missing_ok=True)
            if exc.code in {401, 403}:
                raise SceneConfigurationWarmDiagnosticError(
                    "scene_configuration_warm_output_signed_url_rejected"
                ) from exc
            if exc.code not in {404, 408, 425, 429, 500, 502, 503, 504}:
                raise
        except urllib.error.URLError as exc:
            temporary.unlink(missing_ok=True)
            if not isinstance(
                exc.reason, (TimeoutError, ConnectionError, socket.timeout)
            ):
                raise SceneConfigurationWarmDiagnosticError(
                    "scene_configuration_warm_output_transport_nontransient"
                ) from exc
        except (OSError, SceneConfigurationWarmDiagnosticError):
            temporary.unlink(missing_ok=True)
            raise
        time.sleep(min(2.0, max(0.0, deadline_monotonic - time.monotonic())))
    temporary.unlink(missing_ok=True)
    return False


def _output_object_ready(url: str) -> bool:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "BlueprintSceneWarm/1.0", "Range": "bytes=0-0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:  # nosec B310
            return response.geturl() == url and 200 <= int(response.status) < 300
    except urllib.error.HTTPError as exc:
        if exc.code in {404, 416, 500, 502, 503, 504}:
            return False
        raise
    except urllib.error.URLError:
        return False
