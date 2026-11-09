"""Signed denial handoff. No remotely callable deletion operation."""
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request

from .common import read_json_any
from .live_pipeline_intake_runtime_controls import _string
from .website_capture_withdrawal import acknowledge_withdrawal


def register_server_website_withdrawal_routes(app: FastAPI, *, require_admission: Callable,
                                             manifest_path_provider: Callable[[], Path],
                                             resolve_client_root: Callable[..., Path | None]) -> None:
    def resolve_root(payload: Mapping[str, Any], client_id: str) -> Path | None:
        # Read only the server's manifest and authenticated client scope.
        path = manifest_path_provider()
        manifest = read_json_any(path) if path.is_file() else {}
        manifest_root = _string(manifest.get("capture_root")) if isinstance(manifest, Mapping) else ""
        return resolve_client_root(
            payload={**payload, "site_submission_id": payload.get("request_id")}, client_id=client_id,
            manifest_capture_root=Path(manifest_root).expanduser().absolute() if manifest_root else None)
    register_website_withdrawal_routes(app, require_admission=require_admission, resolve_root=resolve_root)


def register_website_withdrawal_routes(app: FastAPI, *, require_admission: Callable,
                                      resolve_root: Callable[[Mapping[str, Any], str], Path | None]) -> None:
    # The existing Web transport derives /api/live-pipeline from the configured
    # capture-upload-intakes URL. Both existing base URL shapes use one guard.
    @app.post("/api/live-pipeline/website-capture-withdrawals", dependencies=[Depends(require_admission)])
    @app.post("/website-capture-withdrawals", dependencies=[Depends(require_admission)])
    async def withdraw(request: Request) -> dict[str, Any]:
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("website_withdrawal_command_invalid")  # noqa: TRY004 - typed refusal contract
            # Authenticated server mapping is authoritative; never take a root
            # or a physical deletion authorization from the request body.
            root = resolve_root(payload, request.state.intake_client_id)
            if root is None:
                raise HTTPException(status_code=503, detail="website_withdrawal_server_root_unavailable")
            return acknowledge_withdrawal(capture_root=root, command=payload)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
