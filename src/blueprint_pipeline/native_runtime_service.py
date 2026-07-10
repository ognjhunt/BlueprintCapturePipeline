"""Entrypoint for the native site-world runtime service."""

from __future__ import annotations

import os

import uvicorn

from .native_runtime_backend import NativeWorldModelRuntimeStore, native_runtime_config_from_env
from .runtime_service_app import create_runtime_app, validate_runtime_service_exposure


STORE = NativeWorldModelRuntimeStore(native_runtime_config_from_env())
app = create_runtime_app(
    backend=STORE,
    title="Blueprint Native Site-World Runtime",
    require_auth=True,
)


def main() -> int:
    host = os.getenv("SITE_WORLD_RUNTIME_SERVICE_HOST", "127.0.0.1")
    validate_runtime_service_exposure(host=host)
    port = int(os.getenv("SITE_WORLD_RUNTIME_SERVICE_PORT", "8791"))
    uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
