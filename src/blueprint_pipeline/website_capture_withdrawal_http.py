"""Signed denial handoff. No remotely callable deletion operation."""
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request

from .common import read_json_any
from .live_pipeline_intake_runtime_controls import _string
from .website_capture_withdrawal import acknowledge_withdrawal
from .website_preparation_status_http import website_capture_root

#: The WebApp's signed client identity: the one the preparation status route admits,
#: and the only one whose withdrawal may be looked up in the listener's tree.
WEBSITE_CLIENT_ID = "blueprint-webapp"
_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}")


def listener_selectors(payload: Mapping[str, Any]) -> dict[str, str] | None:
    """The website capture a withdrawal names, as listener-tree selectors, or None.

    The names the preparation status route admits (``validate_selectors``) for the one
    capture a withdrawal can name, ``site-<request>/captures/walkthrough-<request>``, with
    a request id that admits no path syntax. Anything else gets no fallback.
    """
    request = payload.get("request_id")
    if type(request) is not str or _REQUEST_ID.fullmatch(request) is None:
        return None
    scene, capture = payload.get("scene_id"), payload.get("capture_id")
    if scene != f"site-{request}" or capture != f"walkthrough-{request}":
        return None
    return {"scene_id": scene, "capture_id": capture}


def register_server_website_withdrawal_routes(app: FastAPI, *, require_admission: Callable,
                                             manifest_path_provider: Callable[[], Path],
                                             resolve_client_root: Callable[..., Path | None]) -> None:
    def resolve_root(payload: Mapping[str, Any], client_id: str) -> Path | None:
        # Read only the server's manifest and authenticated client scope.
        path = manifest_path_provider()
        manifest = read_json_any(path) if path.is_file() else {}
        manifest_root = _string(manifest.get("capture_root")) if isinstance(manifest, Mapping) else ""
        root = resolve_client_root(
            payload={**payload, "site_submission_id": payload.get("request_id")}, client_id=client_id,
            manifest_capture_root=Path(manifest_root).expanduser().absolute() if manifest_root else None)
        if root is not None and Path(root).name == payload.get("capture_id"):
            return root
        # A configured map or the control-plane manifest names at most one capture;
        # every other website capture lives in the listener's tree. The preparation
        # status route falls back the same way, for the same signed client. With no
        # match the answer is unchanged: 503 with no root, the binding refusal with one.
        selected = listener_selectors(payload) if client_id == WEBSITE_CLIENT_ID else None
        return (website_capture_root(selected) if selected else None) or root
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
