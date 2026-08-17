"""On-instance entry point for the Windows Postshot trainer.

UserData cannot carry the trainer licence — EC2 exposes it over IMDS and via
``DescribeInstanceAttribute`` — so the instance fetches the licence itself from
a single signed URL, acknowledges it by deleting the remote object, and holds
the credential only in the environment of the training subprocess.  Nothing
writes it to disk, and the existing worker already redacts it from the training
log and the recorded command line.

This module is the bridge between the boot script and
``canonical_3dgs_cli run-arm``.  It performs no provider allocation, grants no
authority, and cannot decide reconstruction quality.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

ENTRYPOINT_RECEIPT_SCHEMA_VERSION = "windows_worker_entrypoint_receipt.v1"

LICENCE_GET_URL_ENV = "BLUEPRINT_POSTSHOT_LICENCE_GET_URL"
LICENCE_DELETE_URL_ENV = "BLUEPRINT_POSTSHOT_LICENCE_DELETE_URL"
WORKER_ENV_FILE_ENV = "BLUEPRINT_WORKER_ENV_FILE"

#: Keys the licence blob is allowed to define.  Anything else is refused so a
#: compromised or mistaken licence endpoint cannot inject arbitrary runtime
#: configuration into the trainer process.
_ALLOWED_LICENCE_KEYS = frozenset({"POSTSHOT_LOGIN_EMAIL", "POSTSHOT_LOGIN_PASSWORD"})


class WindowsWorkerEntrypointError(RuntimeError):
    """Stable fail-closed worker error.  Never carries credential material."""


def load_worker_environment(path: str | Path) -> dict[str, str]:
    """Read the non-secret environment the boot script staged."""

    text = Path(path).expanduser().read_text(encoding="utf-8")
    values: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        values[key] = value.strip()
    return values


def fetch_licence(
    *,
    get_url: str,
    delete_url: str | None,
    opener: Any = None,
) -> dict[str, str]:
    """Fetch the trainer licence, then acknowledge it by deleting the object.

    The credential is returned in memory only.  Deletion is attempted even when
    the blob is malformed, because a licence object left behind after a fetch is
    a standing exposure regardless of whether this run could use it.
    """

    request_opener = opener or urllib.request.urlopen
    try:
        with request_opener(get_url) as response:  # nosec B310 - operator-supplied signed URL
            body = response.read().decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        raise WindowsWorkerEntrypointError("postshot_licence_fetch_failed") from exc

    licence: dict[str, str] = {}
    unexpected: list[str] = []
    for line in body.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key in _ALLOWED_LICENCE_KEYS:
            licence[key] = value.strip()
        elif key:
            unexpected.append(key)

    deleted = _delete_licence_object(delete_url, opener=request_opener)

    if unexpected:
        raise WindowsWorkerEntrypointError(
            "postshot_licence_blob_contains_unexpected_keys:" + ",".join(sorted(unexpected))
        )
    if sorted(licence) != sorted(_ALLOWED_LICENCE_KEYS) or not all(licence.values()):
        raise WindowsWorkerEntrypointError("postshot_licence_incomplete")
    if delete_url and not deleted:
        raise WindowsWorkerEntrypointError("postshot_licence_delete_not_acknowledged")
    return licence


def _delete_licence_object(delete_url: str | None, *, opener: Any) -> bool:
    if not delete_url:
        return False
    try:
        request = urllib.request.Request(delete_url, method="DELETE")
        with opener(request) as response:  # nosec B310 - operator-supplied signed URL
            status = int(getattr(response, "status", 0) or 0)
        return 200 <= status < 300
    except Exception:  # noqa: BLE001
        return False


def build_trainer_environment(
    *, worker_environment: Mapping[str, str], licence: Mapping[str, str]
) -> dict[str, str]:
    """Compose the trainer's environment without persisting the credential."""

    merged = {k: v for k, v in worker_environment.items() if k not in _ALLOWED_LICENCE_KEYS}
    merged.update({key: str(licence[key]) for key in sorted(_ALLOWED_LICENCE_KEYS)})
    return merged


def redact(value: str, secrets: Sequence[str]) -> str:
    """Remove every known secret from operator-visible text."""

    cleaned = value
    for secret in secrets:
        if secret:
            cleaned = cleaned.replace(secret, "***")
    return cleaned


def run(
    *,
    worker_env_file: str | Path,
    opener: Any = None,
    runner: Any = None,
) -> dict[str, Any]:
    """Fetch the licence and hand the trainer environment to the arm runner."""

    worker_environment = load_worker_environment(worker_env_file)
    get_url = worker_environment.get(LICENCE_GET_URL_ENV) or os.environ.get(
        LICENCE_GET_URL_ENV, ""
    )
    if not get_url:
        raise WindowsWorkerEntrypointError("postshot_licence_get_url_missing")
    delete_url = worker_environment.get(LICENCE_DELETE_URL_ENV) or os.environ.get(
        LICENCE_DELETE_URL_ENV
    )

    licence = fetch_licence(get_url=get_url, delete_url=delete_url, opener=opener)
    secrets = tuple(licence[key] for key in sorted(_ALLOWED_LICENCE_KEYS))
    trainer_environment = build_trainer_environment(
        worker_environment=worker_environment, licence=licence
    )

    if runner is None:  # pragma: no cover - exercised only on a real worker
        from .canonical_3dgs_worker import main as worker_main

        exit_code = worker_main(json.loads(Path(
            worker_environment["BLUEPRINT_WORKER_ARGV_FILE"]
        ).read_text(encoding="utf-8")))
        result: dict[str, Any] = {"exit_code": exit_code}
    else:
        result = dict(runner(environment=trainer_environment))

    receipt = {
        "schema_version": ENTRYPOINT_RECEIPT_SCHEMA_VERSION,
        "licence_fetched": True,
        "licence_object_deleted": bool(delete_url),
        "credential_written_to_disk": False,
        "result": json.loads(redact(json.dumps(result), secrets)),
    }
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker-env-file",
        default=os.environ.get(WORKER_ENV_FILE_ENV, r"C:\work\blueprint_worker.env"),
    )
    arguments = parser.parse_args(argv)
    try:
        receipt = run(worker_env_file=arguments.worker_env_file)
    except WindowsWorkerEntrypointError as exc:
        json.dump({"status": "failed", "error": str(exc)}, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
        return 2
    json.dump(receipt, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


__all__ = [
    "ENTRYPOINT_RECEIPT_SCHEMA_VERSION",
    "LICENCE_DELETE_URL_ENV",
    "LICENCE_GET_URL_ENV",
    "WORKER_ENV_FILE_ENV",
    "WindowsWorkerEntrypointError",
    "build_trainer_environment",
    "fetch_licence",
    "load_worker_environment",
    "redact",
    "run",
]


if __name__ == "__main__":
    raise SystemExit(main())
